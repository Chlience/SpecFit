#!/usr/bin/env python3
"""Run the frozen five-selector by four-executor matrix in independent batches."""
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
from specfit import selection_matrix as experiment
from specfit.experiment import canonical_json_sha256
from specfit.f0_direct import load_sources
from specfit.m1_capture_replay import read_routes, same_json_value
from specfit.r6a_tb2_contract import R6A_TB2_IMAGE_ID
from specfit.selection_evaluation import (CALIBRATION_QUESTIONS, calibration_choices, file_identity, read_artifact,
    read_shards, validate_checkpoint, validate_environment, write_artifact)
from specfit.selection_methods import METHODS, SelectionCostModel

CALIBRATION_CONFIG = experiment.CALIBRATION_CONFIG
prepare_calibration = legacy.prepare_calibration


def arguments():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("calibration", "evaluation"))
    parser.add_argument("action", choices=("freeze", "run", "validate", "combine"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--calibration-manifest", type=Path)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--batch", type=int, choices=range(4))
    scope.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch-results", type=Path, nargs=4)
    parser.add_argument("--scored-controls", action="store_true", help="include two separate binary-score controls")
    parser.add_argument("--capture", type=Path, default=Path("/data/inputs/route-capture/m1-full-atlas-v2-e3bf7a2.json"))
    parser.add_argument("--screen", type=Path, default=Path("/data/inputs/route-capture/m1-multistate-screen-v2-e3bf7a2.json"))
    lock = legacy.load_gsm8k_specbench_lock()
    parser.add_argument("--dataset", type=Path, default=Path("/data/workloads/gsm8k") / lock.revision / lock.filename)
    parser.add_argument("--token-lock", type=Path, default=ROOT / "workloads/m1_gsm8k_prompt_tokens.lock.json")
    parser.add_argument("--image-id", default=R6A_TB2_IMAGE_ID)
    args = parser.parse_args()
    if args.stage == "calibration" and (args.smoke or args.batch is not None or args.action == "combine"):
        parser.error("calibration has no evaluation batches")
    if args.stage == "evaluation" and args.action == "run" and not args.smoke and args.batch is None:
        parser.error("evaluation run requires --smoke or --batch")
    return args


def checked_calibration(args, identity):
    if args.calibration is None or args.calibration_manifest is None:
        raise ValueError("evaluation requires calibration and its manifest")
    frozen = read_artifact(args.calibration_manifest)
    expected = experiment.calibration_manifest(git_commit=frozen["git_commit"],
        measurement_identity=identity, dataset_path=args.dataset)
    if frozen != expected:
        raise ValueError("calibration source, inputs or schedule changed")
    return experiment.validate_result(read_artifact(args.calibration), frozen, None, args.calibration.parent)


def run_gpu(args, frozen, sources):
    evaluation_batch = args.batch
    import torch
    from transformers import AutoTokenizer
    import benchmark_qwen3_m1_full_atlas as m1
    from specfit.checkpoint_audit import checkpoint_digest
    from specfit.f0_direct_runtime import capture_trial_checkpoint
    from specfit.r6a_runner import serialize_candidate_pool
    from specfit.runtime import (Eagle3ChainDrafter, Eagle3TreeDrafter, ExpertExecutionPolicy,
        ReductionMode, apply_numeric_mode, load_offloaded_qwen3, warmup_eagle3_tree_drafter)
    from specfit.selection_evaluation_runtime import (EvaluationSelector,
        capture_initial_checkpoint, run_target_only_cycle, run_tree_cycle)

    calibration = args.stage == "calibration"
    if evaluation_batch is not None and (calibration or type(evaluation_batch) is not int
            or not 0 <= evaluation_batch < len(frozen.get("execution_batches", ()))):
        raise ValueError("invalid frozen evaluation batch")
    config = frozen["config"]
    if args.git_commit != frozen["git_commit"] or args.image_id != R6A_TB2_IMAGE_ID:
        raise ValueError("the run must use the frozen clean commit and container image")
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != "NVIDIA A100-PCIE-40GB":
        raise RuntimeError("measurement requires one A100 PCIe 40GB")
    models = m1._source_preflight(args)
    torch.cuda.set_device(0)
    tokenizer = AutoTokenizer.from_pretrained(m1._TARGET, local_files_only=True)
    prepared = (prepare_calibration(tokenizer, frozen) if calibration else
        {p["question_id"]: p for p in m1._prepare(tokenizer, args.dataset, args.token_lock)})
    trace_manifests = {qid: replace(m1._manifest(args, run_id=f"{args.stage}-q{qid}", prepared=work),
        drafter_mode=frozen["protocol"]) for qid, work in prepared.items()}
    state_map = {} if calibration else {s.state_id: s for s in sources.projection.states}
    source_map = {} if calibration else {s["state_id"]: s for s in sources.capture["states"]}
    reference_map = {} if calibration else {r["question_id"]: r for r in sources.capture["reference_rollouts"]}
    decisions = {} if calibration else {s["state_id"]: s for s in frozen["states"]}
    cost_models = {} if calibration else {e: SelectionCostModel.from_dict(value)
        for e, value in frozen["cost_models"].items()}
    reference_prefetch = frozen.get("reference_h2d_prefetch", True)
    index = dict(artifact_type=f"selection_matrix_{args.stage}_result", protocol=frozen["protocol"],
        git_commit=args.git_commit, measurement_identity=frozen["measurement_identity"],
        manifest_sha256=canonical_json_sha256(frozen), status="running", shards=[],
        environment=dict(image_id=args.image_id, models=models, gpu_name=torch.cuda.get_device_name(0),
            server="a100-host", gpu_index=0), gc_policy=config["gc_policy"],
        entry_preparation_timings_ns=args.entry_preparation_timings_ns)
    if evaluation_batch is not None:
        index["batch"] = evaluation_batch
    index["run_scope"] = "calibration" if calibration else "smoke" if args.smoke else "batch"
    loaded = numeric = active_shard = checkpoint = None
    gc_was_enabled = gc.isenabled()

    def check_budget():
        if perf_counter_ns() - args.entry_started_ns > config["max_wall_seconds"] * 10**9:
            raise RuntimeError("measurement exceeded its frozen wall budget")
        if torch.cuda.max_memory_reserved() > config["max_gpu_reserved_bytes"]:
            raise RuntimeError("measurement exceeded its GPU memory budget")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > config["max_process_rss_bytes"]:
            raise RuntimeError("measurement exceeded its process RSS budget")

    def checkpoint_metadata(saved):
        expert_bytes = sum(c.weights.numel() * c.weights.element_size() for c in saved.expert_caches)
        allocator = saved.host_allocator
        pinned_bytes = allocator["after"]["active_bytes.current"] - allocator["before"]["active_bytes.current"]
        if not 0 < expert_bytes <= pinned_bytes <= config["max_checkpoint_host_bytes"]:
            raise RuntimeError("checkpoint expert payload or pinned allocation exceeded its budget")
        return dict(expert_weight_host_bytes=expert_bytes, pinned_allocation_bytes=pinned_bytes,
            host_allocator=allocator, capture_wall_ns=saved.capture_wall_ns,
            reference_prepare_wall_ns=saved.reference_prepare_wall_ns,
            sha256_before=checkpoint_digest(saved))

    try:
        began = perf_counter_ns()
        loaded = load_offloaded_qwen3(m1._TARGET, device=torch.device("cuda:0"), slots_per_layer=48,
            packed_checkpoint_root=m1._PACKED, packed_expert_backing="registered_mmap",
            verify_packed_hashes=True, reduction_mode=ReductionMode.CANONICAL,
            expert_execution_policy=ExpertExecutionPolicy.EXPERT_ID_ASCENDING,
            h2d_prefetch=reference_prefetch)
        index["model_load_timings"] = asdict(loaded.load_timings)
        m1._require_m1_eos_ids(loaded.model)
        numeric = apply_numeric_mode("math_bf16_full_reduction")
        index["environment"].update(numeric_settings=dict(numeric.actual_settings),
            packed_descriptor=file_identity(Path(m1._PACKED) / "manifest.json"))
        validate_environment(index["environment"])
        drafter = Eagle3ChainDrafter.from_pretrained(m1._DRAFTER, target_root=m1._TARGET,
            device=torch.device("cuda:0"), use_torch_compile=False,
            expected_target_repo_id=models["target"]["repo_id"])
        warmup_eagle3_tree_drafter(drafter, Eagle3TreeDrafter(drafter), max_sequence_length=506,
            max_depth=5, top_k=2, draft_node_budget=18, sequence_anchors=(64, 192, 320, 506))
        index["model_and_drafter_prepare_ns"] = perf_counter_ns() - began
        order = (CALIBRATION_QUESTIONS if calibration else frozen["smoke_states"]
                 if args.smoke else frozen["execution_batches"][evaluation_batch])
        for ordinal, key in enumerate(order):
            gc.enable()
            collect_started = perf_counter_ns()
            gc.collect()
            collected_ns = perf_counter_ns() - collect_started
            gc.disable()
            check_budget()
            qid = key if calibration else state_map[key].question_id
            sid = f"calibration-q{qid}-c0" if calibration else key
            work = prepared[qid]
            trace_manifest = replace(trace_manifests[qid], run_id=sid)
            active_shard = dict(state_id=sid, question_id=qid, status="running",
                manifest_sha256=index["manifest_sha256"], gc_collect_wall_ns=collected_ns, records=[])
            if calibration:
                loaded.set_expert_h2d_prefetch(False)
                checkpoint = capture_initial_checkpoint(loaded=loaded, drafter=drafter,
                    input_ids=work["input_ids"], manifest=trace_manifest, state_id=sid)
                metadata = checkpoint_metadata(checkpoint)
                source = dict(state_id=sid, base_kv_length=checkpoint.generation.state.cache.get_seq_length(),
                    pre_logical_state_sha256=checkpoint.reference_cache_sha256)
                state = SimpleNamespace(state_id=sid, question_id=qid, cycle_index=0, pool=None)
                probe_selector = EvaluationSelector("route_probe")
                probe, pool = run_tree_cycle(loaded=loaded, drafter=drafter, checkpoint=checkpoint,
                    state=state, source_state=source, selector=probe_selector, mode="natural",
                    manifest=trace_manifest, job_id=f"{sid}:route_probe", execution_index=-1, audit=True, execution="demand")
                state.pool = pool
                state.atlas = read_routes(probe["routes"], "calibration full atlas", pool, tuple(range(19)))
                checkpoint = replace(checkpoint, pool_fingerprint=pool.fingerprint)
                choices = calibration_choices(pool, state.atlas)
                active_shard.update(prepared_prompt=work, source_state=source, route_probe=probe,
                    pool=serialize_candidate_pool(probe_selector.pool), choices=choices)
                schedule = next(p["schedule"] for p in frozen["prompts"] if p["question_id"] == qid)
                expected = {(e, method, k): tuple(choices[method][str(k)]["selected_raw_ids"])
                    for e in experiment.EXECUTORS for method in CALIBRATION_CONFIG["selection_probes"] for k in range(8)}
            else:
                state, source = state_map[sid], source_map[sid]
                loaded.set_expert_h2d_prefetch(reference_prefetch)
                checkpoint = capture_trial_checkpoint(loaded=loaded, drafter=drafter,
                    source_state=source, state=state, reference=reference_map[qid],
                    prompt_ids=work["input_ids"], manifest=trace_manifest,
                    job=dict(job_id=f"{sid}:reference", selected_raw_ids=
                             decisions[sid]["choices"]["demand"]["probability_fixed"]["selected_raw_ids"]),
                    execution_index=0, audit=True)
                metadata = checkpoint_metadata(checkpoint)
                schedule = decisions[sid]["schedule"]
                expected = {(e, method, None): tuple(choice["selected_raw_ids"])
                    for e, methods in decisions[sid]["choices"].items() for method, choice in methods.items()}
            active_shard["checkpoint"] = metadata
            print(json.dumps(dict(phase="checkpoint_ready", state_id=sid, ordinal=ordinal,
                reference_prepare_seconds=checkpoint.reference_prepare_wall_ns / 1e9,
                checkpoint_capture_seconds=checkpoint.capture_wall_ns / 1e9)), flush=True)
            # Collect exact selected-query routes independently, before this state's
            # audited warmups or formal samples. Selection choices are already fixed.
            references = {}
            for (execution, method, count), ids in expected.items():
                key = experiment.query_key(method, ids)
                if key in references:
                    continue
                common = dict(loaded=loaded, checkpoint=checkpoint, state=state, source_state=source,
                    mode="natural", manifest=trace_manifest, job_id=f"{sid}:query:{key}",
                    execution_index=-1, audit=True, execution="demand")
                if method == "target_only":
                    reference = run_target_only_cycle(**common)
                else:
                    selector = EvaluationSelector("frozen_selection", expected_pool=state.pool.fingerprint,
                        expected_ids=ids)
                    reference, _ = run_tree_cycle(**common, drafter=drafter, selector=selector)
                references[key] = reference
                check_budget()
            collection_finished = perf_counter_ns()
            reference_path = args.output.with_name(f"{args.output.stem}-{sid}-queries.json")
            active_shard["query_references"] = write_artifact(reference_path, dict(state_id=sid,
                manifest_sha256=index["manifest_sha256"], collection_finished_ns=collection_finished,
                references=references))
            print(json.dumps(dict(phase="query_references_ready", state_id=sid,
                queries=len(references), descriptor=active_shard["query_references"])), flush=True)
            reference_hashes = {key: canonical_json_sha256(value) for key, value in references.items()}
            for scheduled in schedule:
                check_budget()
                method, count = scheduled["method"], scheduled.get("draft_node_count")
                execution = scheduled["execution"]
                ids = expected[execution, method, count]
                reference_key = experiment.query_key(method, ids)
                reference = references[reference_key]
                predictions = (experiment.predictions_for(reference, execution)
                    if execution in ("predicted_same_layer", *experiment.SCORED_CONTROLS) else None)
                common = dict(loaded=loaded, checkpoint=checkpoint, state=state, source_state=source,
                    mode=scheduled["mode"], manifest=trace_manifest,
                    job_id=f"{sid}:{execution}:{method}:{count}",
                    execution_index=scheduled["execution_index"], audit=scheduled["phase"] == "audit",
                    execution=execution, predictions=predictions)
                if method == "target_only":
                    record = run_target_only_cycle(**common)
                else:
                    selector = EvaluationSelector(method, atlas=state.atlas,
                        model=cost_models.get(execution, cost_models.get("predicted_same_layer")),
                        expected_pool=state.pool.fingerprint, expected_ids=ids, calibration_count=count)
                    record, _ = run_tree_cycle(**common, drafter=drafter, selector=selector)
                record.update(scheduled, query_reference_sha256=reference_hashes[reference_key],
                    query_collection_finished_ns=collection_finished)
                if experiment.reference_value(record) != experiment.reference_value(reference):
                    raise RuntimeError("matrix execution changed exact query logits, routes or acceptance")
                active_shard["records"].append(record)
                print(json.dumps(dict(state_id=sid, **scheduled, committed=len(
                    record["acceptance"]["committed_token_ids"]), selected=len(record["selected_raw_ids"]),
                    cycle_ms=record["timings"]["cycle_wall_ns"] / 1e6,
                    prepare_seconds=record["prepare_wall_ns"] / 1e9)), flush=True)
                check_budget()
            experiment.validate_records(active_shard["records"], schedule, state=state, source_state=source,
                choices=expected, references=references)
            metadata["sha256_after"] = checkpoint_digest(checkpoint)
            metadata["unchanged"] = metadata["sha256_after"] == metadata["sha256_before"]
            validate_checkpoint(metadata, config=config)
            active_shard["status"] = "complete"
            shard_path = args.output.with_name(f"{args.output.stem}-{sid}.json")
            index["shards"].append(write_artifact(shard_path, active_shard))
            checkpoint = active_shard = common = references = reference = record = probe = None
            check_budget()
        gc.enable()
        gc.collect()
        shards = list(read_shards(index, args.output.parent))
        if calibration:
            models, diagnostics = experiment.fit_models(shards)
            index.update(cost_models={e: m.to_dict() for e, m in models.items()}, fit_diagnostics=diagnostics)
        else:
            index["summary"] = experiment.summarize([r for s in shards for r in s["records"]])
        index.update(status="complete", gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            entry_to_validation_complete_wall_ns=perf_counter_ns() - args.entry_started_ns)
        experiment.validate_result(index, frozen, sources, args.output.parent)
        index["entry_to_validation_complete_wall_ns"] = perf_counter_ns() - args.entry_started_ns
        check_budget()
        written = perf_counter_ns()
        descriptor = write_artifact(args.output, index)
        print(json.dumps(dict(status="complete", output=str(args.output), descriptor=descriptor,
            result_write_and_hash_wall_ns=perf_counter_ns() - written,
            entry_to_write_and_hash_wall_ns=perf_counter_ns() - args.entry_started_ns)), flush=True)
    except Exception as error:
        index.update(status="invalid", error=f"{type(error).__name__}: {error}",
                     traceback=traceback.format_exc(), failed_state=active_shard)
        if not args.output.exists():
            write_artifact(args.output, index)
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
    sources = None
    if args.stage == "calibration":
        expected = experiment.calibration_manifest(git_commit=args.git_commit,
            measurement_identity=identity, dataset_path=args.dataset)
    else:
        sources = load_sources(capture_path=args.capture, screen_path=args.screen,
            workload_path=args.dataset, token_lock_path=args.token_lock)
        models = checked_calibration(args, identity)
        expected = experiment.evaluation_manifest(sources, git_commit=args.git_commit,
            measurement_identity=identity, calibration_descriptor=file_identity(args.calibration), models=models,
            scored_controls=args.scored_controls)
    if args.action == "freeze":
        print(json.dumps(write_artifact(args.output, expected)))
        return
    if args.manifest is None:
        raise ValueError("run/validate/combine requires --manifest")
    frozen = read_artifact(args.manifest)
    expected["git_commit"] = frozen["git_commit"]
    if not same_json_value(expected, frozen):
        raise ValueError("matrix source, inputs, cost models, choices or schedule changed")
    args.entry_preparation_timings_ns = dict(source_and_manifest_validation=perf_counter_ns()-started)
    if args.action == "run":
        run_gpu(args, frozen, sources)
    elif args.action == "combine":
        if not args.batch_results or any(p.parent != args.output.parent for p in args.batch_results):
            raise ValueError("combine requires four batch results in the output directory")
        result = experiment.combine(args.batch_results, frozen, sources)
        print(json.dumps(dict(status="complete", output=write_artifact(args.output, result))))
    else:
        result = read_artifact(args.output)
        if result.get("artifact_type") == "selection_matrix_combined_result":
            paths = [args.output.parent / row["basename"] for row in result["batch_results"]]
            for path, descriptor in zip(paths, result["batch_results"], strict=True):
                read_artifact(path, descriptor)
            if experiment.combine(paths, frozen, sources) != result:
                raise ValueError("combined matrix cannot be reproduced from its batches")
        else:
            experiment.validate_result(result, frozen, sources, args.output.parent)
        print(json.dumps(dict(status="valid", source=file_identity(args.output))))


if __name__ == "__main__":
    main()
