#!/usr/bin/env python3
"""Freeze, measure and validate isolated cache-aware selector refinements."""
from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import resource
import sys
from time import perf_counter_ns
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from scripts import benchmark_qwen3_selection as legacy
from specfit import selection_refinement_evaluation as experiment
from specfit import selection_matrix as matrix
from specfit import selection_evaluation as base
from specfit.experiment import canonical_json_sha256
from specfit.m1_capture_replay import read_routes, same_json_value
from specfit.r6a_tb2_contract import R6A_TB2_IMAGE_ID
from specfit.selection_methods import SelectionCostModel
from specfit.selection_refinement import CacheCostModel, resident_bitmap, residents_from_logical_state


def arguments():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "validate", "combine"))
    parser.add_argument("--stage", choices=tuple(experiment.SPLITS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--smoke", action="store_true", help="freeze a separate 72-cycle calibration-entry check")
    parser.add_argument("--batch-results", type=Path, nargs="+")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--calibration-manifest", type=Path)
    lock = legacy.load_gsm8k_specbench_lock()
    parser.add_argument("--dataset", type=Path, default=Path("/data/workloads/gsm8k") / lock.revision / lock.filename)
    parser.add_argument("--image-id", default=R6A_TB2_IMAGE_ID)
    parser.add_argument("--device", choices=tuple(k for k in base.GPU_ENVIRONMENTS if k),
                        help="run on a second device at its frozen expert slots")
    args = parser.parse_args()
    if args.action == "freeze" and args.stage is None:
        parser.error("freeze requires --stage")
    if args.action != "freeze" and args.manifest is None:
        parser.error("run, validate and combine require --manifest")
    if args.action == "run" and args.batch is None:
        parser.error("run requires --batch")
    if args.action == "combine" and not args.batch_results:
        parser.error("combine requires all --batch-results")
    return args


def validate_combined(value, manifest, directory):
    descriptors = value.get("batch_results", [])
    if (len({d["basename"] for d in descriptors}) != len(descriptors)
            or any(Path(d["basename"]).name != d["basename"] for d in descriptors)):
        raise ValueError("combined result repeats a batch source")
    indices = [base.read_artifact(directory / d["basename"], d) for d in descriptors]
    expected = experiment.combine(indices, manifest, directory)
    expected["batch_results"] = descriptors
    if not same_json_value(value, expected):
        raise ValueError("combined result cannot be independently reproduced")
    return expected


def checked_calibration(args, identity):
    if args.calibration is None or args.calibration_manifest is None:
        raise ValueError("evaluation needs complete calibration and its manifest")
    frozen = base.read_artifact(args.calibration_manifest)
    expected = experiment.build_manifest(stage="calibration", git_commit=frozen["git_commit"],
        measurement_identity=identity, dataset_path=args.dataset)
    if not same_json_value(frozen, expected):
        raise ValueError("calibration inputs or implementation identity changed")
    result = validate_combined(base.read_artifact(args.calibration), frozen, args.calibration.parent)
    return result["models"]


def run_gpu(args, frozen):
    import torch
    from transformers import AutoTokenizer
    import benchmark_qwen3_m1_full_atlas as m1
    from specfit.checkpoint_audit import checkpoint_digest
    from specfit.r6a_runner import serialize_candidate_pool
    from specfit.runtime import (Eagle3ChainDrafter, Eagle3TreeDrafter, ExpertExecutionPolicy,
        ReductionMode, apply_numeric_mode, load_offloaded_qwen3, warmup_eagle3_tree_drafter)
    from specfit.selection_evaluation_runtime import EvaluationSelector, run_target_only_cycle, run_tree_cycle
    from specfit.selection_refinement_runtime import RefinementSelector, capture_boundary_checkpoint

    config = frozen["config"]
    if (type(args.batch) is not int or not 0 <= args.batch < len(frozen["batches"])
            or args.git_commit != frozen["git_commit"] or args.image_id != R6A_TB2_IMAGE_ID):
        raise ValueError("run must use the frozen batch, clean commit and image")
    device = base.GPU_ENVIRONMENTS[args.device]
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != device["gpu_name"]:
        raise RuntimeError("refinement requires one A100 PCIe 40GB or the selected second device")
    model_identities = m1._source_preflight(args)
    torch.cuda.set_device(0)
    tokenizer = AutoTokenizer.from_pretrained(m1._TARGET, local_files_only=True)
    # The trace interface requires immutable token tuples; normalize only artifact
    # copies, preserving that interface when constructing the workload manifests.
    prepared = legacy.prepare_calibration(tokenizer, frozen)
    trace_manifests = {qid: replace(m1._manifest(args, run_id=f"{frozen['stage']}-q{qid}", prepared=row),
        drafter_mode=experiment.PROTOCOL) for qid, row in prepared.items()}
    index = dict(artifact_type="selection_refinement_batch_result", protocol=experiment.PROTOCOL,
        stage=frozen["stage"], status="running", git_commit=args.git_commit,
        measurement_identity=frozen["measurement_identity"], manifest_sha256=canonical_json_sha256(frozen),
        batch=args.batch, shards=[], gc_policy=config["gc_policy"],
        environment=dict(image_id=args.image_id, models=model_identities, **device))
    loaded = numeric = active_shard = checkpoint = None
    gc_was_enabled = gc.isenabled()

    def check_budget():
        if perf_counter_ns() - args.entry_started_ns > config["max_wall_seconds"] * 10**9:
            raise RuntimeError("refinement exceeded its frozen wall budget")
        if torch.cuda.max_memory_reserved() > config["max_gpu_reserved_bytes"]:
            raise RuntimeError("refinement exceeded its GPU memory budget")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > config["max_process_rss_bytes"]:
            raise RuntimeError("refinement exceeded its process RSS budget")

    def checkpoint_metadata(saved):
        expert_bytes = sum(c.weights.numel() * c.weights.element_size() for c in saved.expert_caches)
        allocator = saved.host_allocator
        pinned_bytes = allocator["after"]["active_bytes.current"] - allocator["before"]["active_bytes.current"]
        if not 0 < expert_bytes <= pinned_bytes <= config["max_checkpoint_host_bytes"]:
            raise RuntimeError("refinement checkpoint exceeded its pinned memory budget")
        return dict(expert_weight_host_bytes=expert_bytes, pinned_allocation_bytes=pinned_bytes,
            host_allocator=allocator, capture_wall_ns=saved.capture_wall_ns,
            reference_prepare_wall_ns=saved.reference_prepare_wall_ns, sha256_before=checkpoint_digest(saved))

    try:
        began = perf_counter_ns()
        loaded = load_offloaded_qwen3(m1._TARGET, device=torch.device("cuda:0"), slots_per_layer=device.get("slots_per_layer", 48),
            packed_checkpoint_root=m1._PACKED, packed_expert_backing="registered_mmap", verify_packed_hashes=True,
            reduction_mode=ReductionMode.CANONICAL, expert_execution_policy=ExpertExecutionPolicy.EXPERT_ID_ASCENDING,
            h2d_prefetch=False)
        index["model_load_timings"] = asdict(loaded.load_timings)
        m1._require_m1_eos_ids(loaded.model)
        numeric = apply_numeric_mode("math_bf16_full_reduction")
        index["environment"].update(numeric_settings=dict(numeric.actual_settings),
            packed_descriptor=base.file_identity(Path(m1._PACKED) / "manifest.json"))
        base.validate_environment(index["environment"])
        drafter = Eagle3ChainDrafter.from_pretrained(m1._DRAFTER, target_root=m1._TARGET,
            device=torch.device("cuda:0"), use_torch_compile=False,
            expected_target_repo_id=model_identities["target"]["repo_id"])
        warmup_eagle3_tree_drafter(drafter, Eagle3TreeDrafter(drafter), max_sequence_length=506,
            max_depth=5, top_k=2, draft_node_budget=18, sequence_anchors=(64, 192, 320, 506))
        index["model_and_drafter_prepare_ns"] = perf_counter_ns() - began
        for ordinal, sid in enumerate(frozen["batches"][args.batch]):
            gc.enable()
            collected = perf_counter_ns()
            gc.collect()
            collected_ns = perf_counter_ns() - collected
            gc.disable()
            check_budget()
            planned = next(s for s in frozen["states"] if s["state_id"] == sid)
            qid, cycle = planned["question_id"], planned["cycle_index"]
            work = prepared[qid]
            trace_manifest = replace(trace_manifests[qid], run_id=sid)
            active_shard = dict(state_id=sid, question_id=qid, cycle_index=cycle, status="running",
                manifest_sha256=index["manifest_sha256"], prepared_prompt=json.loads(json.dumps(work)),
                gc_collect_wall_ns=collected_ns, records=[])
            checkpoint, history = capture_boundary_checkpoint(loaded=loaded, drafter=drafter,
                input_ids=work["input_ids"], manifest=trace_manifest, state_id=sid, cycle_index=cycle)
            metadata = checkpoint_metadata(checkpoint)
            active_shard.update(checkpoint=metadata, reference_history=history)
            source = dict(state_id=sid, base_kv_length=checkpoint.generation.state.cache.get_seq_length(),
                          pre_logical_state_sha256=checkpoint.reference_cache_sha256)
            state = SimpleNamespace(state_id=sid, question_id=qid, cycle_index=cycle,
                                    slots_per_layer=device.get("slots_per_layer", 48), pool=None)
            print(json.dumps(dict(phase="checkpoint_ready", state_id=sid, ordinal=ordinal,
                reference_prepare_seconds=checkpoint.reference_prepare_wall_ns / 1e9)), flush=True)
            probe_selector = EvaluationSelector("route_probe")
            probe, pool = run_tree_cycle(loaded=loaded, drafter=drafter, checkpoint=checkpoint,
                state=state, source_state=source, selector=probe_selector, mode="natural",
                manifest=trace_manifest, job_id=f"{sid}:route_probe", execution_index=-1,
                audit=True, execution="demand")
            state.pool = pool
            state.atlas = read_routes(probe["routes"], "refinement full atlas", pool, tuple(range(19)))
            checkpoint = replace(checkpoint, pool_fingerprint=pool.fingerprint)
            active_shard.update(source_state=source, pool=serialize_candidate_pool(probe_selector.pool), route_probe=probe)
            choices = experiment.choices_for(state, stage=frozen["stage"], logical_cache=probe["pre_logical_state"],
                models=frozen["models"], guard_margin=frozen["guard_margin"])
            active_shard["choices"] = choices
            choice_path = args.output.with_name(f"{args.output.stem}-{sid}-choices.json")
            active_shard["choice_manifest"] = base.write_artifact(choice_path,
                dict(state_id=sid, manifest_sha256=index["manifest_sha256"], choices=choices,
                     models_sha256=canonical_json_sha256(frozen["models"]),
                     route_probe_sha256=canonical_json_sha256(probe), frozen_at_ns=perf_counter_ns()))
            references = {}
            for job in planned["schedule"]:
                ids = tuple(choices[job["mode"]][experiment.choice_key(job["method"],
                    job.get("draft_node_count"))]["selected_raw_ids"])
                key = matrix.query_key(job["method"], ids)
                if key in references:
                    continue
                common = dict(loaded=loaded, checkpoint=checkpoint, state=state, source_state=source,
                    mode="natural", manifest=trace_manifest, job_id=f"{sid}:query:{key}",
                    execution_index=-1, audit=True, execution="demand")
                if job["method"] == "target_only":
                    reference = run_target_only_cycle(**common)
                else:
                    selector = EvaluationSelector("frozen_selection", expected_pool=pool.fingerprint, expected_ids=ids)
                    reference, _ = run_tree_cycle(**common, drafter=drafter, selector=selector)
                references[key] = reference
                check_budget()
            collection_finished = perf_counter_ns()
            reference_path = args.output.with_name(f"{args.output.stem}-{sid}-queries.json")
            active_shard["query_references"] = base.write_artifact(reference_path,
                dict(state_id=sid, manifest_sha256=index["manifest_sha256"],
                     collection_finished_ns=collection_finished, references=references))
            natural_mask = resident_bitmap(residents_from_logical_state(probe["pre_logical_state"]))
            reference_hashes = {key: canonical_json_sha256(value) for key, value in references.items()}
            for job in planned["schedule"]:
                check_budget()
                mode, method = job["mode"], job["method"]
                ids = tuple(choices[mode][experiment.choice_key(method, job.get("draft_node_count"))]["selected_raw_ids"])
                key = matrix.query_key(method, ids)
                common = dict(loaded=loaded, checkpoint=checkpoint, state=state, source_state=source,
                    mode=mode, manifest=trace_manifest, job_id=f"{sid}:{mode}:{experiment.choice_key(method, job.get('draft_node_count'))}",
                    execution_index=job["execution_index"], audit=job["phase"] == "audit", execution=experiment.EXECUTION)
                selector = None
                if method == "target_only":
                    record = run_target_only_cycle(**common)
                else:
                    models = frozen["models"]
                    selector = RefinementSelector(method, loaded=loaded, atlas=state.atlas,
                        model=None if models is None else SelectionCostModel.from_dict(models["by_mode"][mode]["legacy"]),
                        cache_model=None if models is None else CacheCostModel.from_dict(models["by_mode"][mode]["cache"]),
                        acceptance_exponent=1.0 if models is None else models["acceptance_exponent"],
                        expected_resident_mask=0 if mode == "cold" else natural_mask,
                        guard_margin=frozen["guard_margin"], expected_pool=pool.fingerprint,
                        expected_ids=ids, calibration_count=job.get("draft_node_count"))
                    record, _ = run_tree_cycle(**common, drafter=drafter, selector=selector)
                observed_mask = None if selector is None else selector.observed_resident_mask
                record.update(job, query_reference_sha256=reference_hashes[key],
                    choice_manifest_sha256=active_shard["choice_manifest"]["sha256"],
                    query_collection_finished_ns=collection_finished,
                    selector_resident_sha256=None if observed_mask is None else canonical_json_sha256(hex(observed_mask)))
                if matrix.reference_value(record) != matrix.reference_value(references[key]):
                    raise RuntimeError("refinement changed exact-query logits, routes or acceptance")
                active_shard["records"].append(record)
                print(json.dumps(dict(state_id=sid, **job, selected=len(ids),
                    committed=len(record["acceptance"]["committed_token_ids"]),
                    cycle_ms=record["timings"]["cycle_wall_ns"] / 1e6,
                    selector_ms=record["timings"]["explicit_selector_ns"] / 1e6)), flush=True)
                check_budget()
            metadata["sha256_after"] = checkpoint_digest(checkpoint)
            metadata["unchanged"] = metadata["sha256_before"] == metadata["sha256_after"]
            active_shard["status"] = "complete"
            # Validate after JSON normalization, including files bound before timing.
            experiment.validate_shard(json.loads(json.dumps(active_shard)), frozen, args.output.parent,
                                      slots_per_layer=state.slots_per_layer)
            shard_path = args.output.with_name(f"{args.output.stem}-{sid}.json")
            index["shards"].append(base.write_artifact(shard_path, active_shard))
            checkpoint = active_shard = common = references = reference = record = probe = selector = None
            check_budget()
        gc.enable()
        gc.collect()
        index.update(status="complete", gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            entry_to_validation_complete_wall_ns=perf_counter_ns() - args.entry_started_ns)
        experiment.validate_result(index, frozen, args.output.parent)
        index["entry_to_validation_complete_wall_ns"] = perf_counter_ns() - args.entry_started_ns
        check_budget()
        descriptor = base.write_artifact(args.output, index)
        print(json.dumps(dict(status="complete", output=str(args.output), descriptor=descriptor)), flush=True)
    except Exception as error:
        index.update(status="invalid", error=f"{type(error).__name__}: {error}",
                     traceback=traceback.format_exc(), failed_state=active_shard)
        if not args.output.exists():
            base.write_artifact(args.output, index)
        raise
    finally:
        if gc_was_enabled:
            gc.enable()
        else:
            gc.disable()
        if numeric is not None:
            numeric.restore()
        if loaded is not None:
            loaded.close()


def main():
    started = perf_counter_ns()
    args = arguments()
    args.entry_started_ns, args.git_commit = started, legacy.head()
    identity = experiment.identity(ROOT)
    if args.action != "validate" and args.output.exists():
        raise FileExistsError(args.output)
    frozen = None if args.action == "freeze" else base.read_artifact(args.manifest)
    stage = args.stage if frozen is None else frozen["stage"]
    if args.stage is not None and stage != args.stage:
        raise ValueError("stage differs from its manifest")
    models = None if stage == "calibration" else checked_calibration(args, identity)
    expected = experiment.build_manifest(stage=stage,
        git_commit=args.git_commit if frozen is None else frozen["git_commit"],
        measurement_identity=identity, dataset_path=args.dataset,
        calibration_descriptor=None if models is None else base.file_identity(args.calibration), models=models,
        smoke=args.smoke if frozen is None else frozen["scope"] == "smoke")
    if frozen is not None and not same_json_value(frozen, expected):
        raise ValueError("frozen refinement protocol, source or models changed")
    if args.action == "freeze":
        descriptor = base.write_artifact(args.output, expected)
        print(json.dumps(dict(status="frozen", descriptor=descriptor)), flush=True)
    elif args.action == "run":
        run_gpu(args, frozen)
    elif args.action == "combine":
        if any(path.parent.resolve() != args.output.parent.resolve() for path in args.batch_results):
            raise ValueError("batch files and combined result must share their artifact directory")
        indices = [base.read_artifact(path) for path in args.batch_results]
        value = experiment.combine(indices, frozen, args.output.parent)
        value["batch_results"] = [base.file_identity(path) for path in args.batch_results]
        descriptor = base.write_artifact(args.output, value)
        print(json.dumps(dict(status="combined", descriptor=descriptor)), flush=True)
    else:
        value = base.read_artifact(args.output)
        if value.get("artifact_type") == "selection_refinement_combined_result":
            validate_combined(value, frozen, args.output.parent)
        else:
            experiment.validate_result(value, frozen, args.output.parent)
        print(json.dumps(dict(status="validated", output=str(args.output))), flush=True)


if __name__ == "__main__":
    main()
