#!/usr/bin/env python3
"""Freeze, measure and independently validate five named selection methods."""
from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import resource
import subprocess
from time import perf_counter_ns
import traceback
from types import SimpleNamespace

from specfit.experiment import canonical_json_sha256
from specfit.f0_direct import PACKED_SOURCE, load_sources
from specfit.gsm8k import load_gsm8k_specbench_lock
from specfit.m1_capture_replay import read_routes, same_json_value
from specfit.r6a_tb2_contract import R6A_TB2_IMAGE_ID
from specfit.selection_evaluation import (CALIBRATION_CONFIG, CALIBRATION_PROTOCOL,
    CALIBRATION_QUESTIONS, CONFIG, PROTOCOL, build_calibration_manifest, build_manifest,
    calibration_choices, calibration_model_from_states, file_identity, measurement_identity,
    read_artifact, read_shards, summarize_measurements, validate_calibration,
    validate_checkpoint, validate_environment, validate_evaluation, validate_manifest,
    validate_records, write_artifact)
from specfit.selection_methods import METHODS, SelectionCostModel

ROOT = Path(__file__).resolve().parents[1]


def head():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    if git("status", "--porcelain"):
        raise RuntimeError("measurement commands require a clean Git checkout")
    return git("rev-parse", "HEAD")


def arguments():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("calibration", "evaluation"))
    parser.add_argument("action", choices=("freeze", "run", "validate"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--capture", type=Path,
        default=Path("/data/inputs/route-capture/m1-full-atlas-v2-e3bf7a2.json"))
    parser.add_argument("--screen", type=Path,
        default=Path("/data/inputs/route-capture/m1-multistate-screen-v2-e3bf7a2.json"))
    lock = load_gsm8k_specbench_lock()
    parser.add_argument("--dataset", type=Path,
        default=Path("/data/workloads/gsm8k") / lock.revision / lock.filename)
    parser.add_argument("--token-lock", type=Path,
        default=ROOT / "workloads/m1_gsm8k_prompt_tokens.lock.json")
    parser.add_argument("--image-id", default=R6A_TB2_IMAGE_ID)
    return parser.parse_args()


def checked_calibration(args, identity):
    if args.calibration is None or args.calibration_manifest is None:
        raise ValueError("evaluation requires --calibration and --calibration-manifest")
    manifest = read_artifact(args.calibration_manifest)
    expected = build_calibration_manifest(git_commit=manifest["git_commit"], identity=identity,
                                           dataset_path=args.dataset)
    if not same_json_value(manifest, expected):
        raise ValueError("calibration manifest changed its inputs, schedule or implementation")
    index = read_artifact(args.calibration)
    return validate_calibration(index, manifest, args.calibration.parent)


def prepare_calibration(tokenizer, manifest):
    from specfit.gsm8k_depth5_experiment import max_new_tokens_for_input
    prepared = {}
    for prompt in manifest["prompts"]:
        rendered = tokenizer.apply_chat_template([dict(role="user", content=prompt["raw_prompt"])],
            tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(rendered)
        prepared[prompt["question_id"]] = dict(question_id=prompt["question_id"],
            raw_prompt=prompt["raw_prompt"], rendered=rendered, input_ids=tuple(encoded["input_ids"]),
            attention_mask=tuple(encoded["attention_mask"]),
            max_new_tokens=max_new_tokens_for_input(len(encoded["input_ids"])))
    return prepared


def run_gpu(args, frozen, sources, *, evaluation_validator=validate_evaluation,
            summarizer=summarize_measurements, evaluation_batch=None):
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
    cost_model = None if calibration else SelectionCostModel.from_dict(frozen["cost_model"])
    method_definitions = {} if calibration else {m["id"]: m for m in frozen["methods"]}
    prefetch_model = (SelectionCostModel.from_dict(frozen["prefetch_cost_model"])
                     if "prefetch_cost_model" in frozen else None)
    reference_prefetch = frozen.get("reference_h2d_prefetch", True)
    index = dict(artifact_type=f"selection_{args.stage}_result", protocol=frozen["protocol"],
        git_commit=args.git_commit, measurement_identity=frozen["measurement_identity"],
        manifest_sha256=canonical_json_sha256(frozen), status="running", shards=[],
        environment=dict(image_id=args.image_id, models=models, gpu_name=torch.cuda.get_device_name(0),
            server="a100-host", gpu_index=0), gc_policy=config["gc_policy"],
        entry_preparation_timings_ns=args.entry_preparation_timings_ns)
    if evaluation_batch is not None:
        index["evaluation_batch"] = evaluation_batch
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
        order = (CALIBRATION_QUESTIONS if calibration else frozen["state_order"]
                 if evaluation_batch is None else frozen["execution_batches"][evaluation_batch])
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
                checkpoint = capture_initial_checkpoint(loaded=loaded, drafter=drafter,
                    input_ids=work["input_ids"], manifest=trace_manifest, state_id=sid)
                metadata = checkpoint_metadata(checkpoint)
                source = dict(state_id=sid, base_kv_length=checkpoint.generation.state.cache.get_seq_length(),
                    pre_logical_state_sha256=checkpoint.reference_cache_sha256)
                state = SimpleNamespace(state_id=sid, question_id=qid, cycle_index=0, pool=None)
                probe_selector = EvaluationSelector("route_probe")
                probe, pool = run_tree_cycle(loaded=loaded, drafter=drafter, checkpoint=checkpoint,
                    state=state, source_state=source, selector=probe_selector, mode="natural",
                    manifest=trace_manifest, job_id=f"{sid}:route_probe", execution_index=-1, audit=True)
                state.pool = pool
                state.atlas = read_routes(probe["routes"], "calibration full atlas", pool, tuple(range(19)))
                checkpoint = replace(checkpoint, pool_fingerprint=pool.fingerprint)
                choices = calibration_choices(pool, state.atlas)
                active_shard.update(prepared_prompt=work, source_state=source, route_probe=probe,
                    pool=serialize_candidate_pool(probe_selector.pool), choices=choices)
                schedule = next(p["schedule"] for p in frozen["prompts"] if p["question_id"] == qid)
                expected = {(method, k): tuple(choices[method][str(k)]["selected_raw_ids"])
                    for method in CALIBRATION_CONFIG["selection_probes"] for k in range(8)}
            else:
                state, source = state_map[sid], source_map[sid]
                loaded.set_expert_h2d_prefetch(reference_prefetch)
                checkpoint = capture_trial_checkpoint(loaded=loaded, drafter=drafter,
                    source_state=source, state=state, reference=reference_map[qid],
                    prompt_ids=work["input_ids"], manifest=trace_manifest,
                    job=dict(job_id=f"{sid}:reference", selected_raw_ids=
                             decisions[sid]["choices"]["probability_fixed"]["selected_raw_ids"]),
                    execution_index=0, audit=True)
                metadata = checkpoint_metadata(checkpoint)
                schedule = decisions[sid]["schedule"]
                expected = {(method, None): tuple(decisions[sid]["choices"][method]["selected_raw_ids"])
                            for method in method_definitions}
            active_shard["checkpoint"] = metadata
            print(json.dumps(dict(phase="checkpoint_ready", state_id=sid, ordinal=ordinal,
                reference_prepare_seconds=checkpoint.reference_prepare_wall_ns / 1e9,
                checkpoint_capture_seconds=checkpoint.capture_wall_ns / 1e9)), flush=True)
            for scheduled in schedule:
                check_budget()
                method, count = scheduled["method"], scheduled.get("draft_node_count")
                definition = method_definitions.get(method, {})
                enabled = definition.get("h2d_prefetch", reference_prefetch)
                loaded.set_expert_h2d_prefetch(enabled)
                common = dict(loaded=loaded, checkpoint=checkpoint, state=state, source_state=source,
                    mode=scheduled["mode"], manifest=trace_manifest, job_id=f"{sid}:{method}:{count}",
                    execution_index=scheduled["execution_index"], audit=scheduled["phase"] == "audit")
                if method == "target_only":
                    record = run_target_only_cycle(**common)
                else:
                    selector = EvaluationSelector(definition.get("selection_method", method),
                        atlas=state.atlas, model=prefetch_model if definition.get("cost_model") == "prefetch" else cost_model,
                        expected_pool=state.pool.fingerprint, expected_ids=expected[method, count],
                        calibration_count=count)
                    record, _ = run_tree_cycle(**common, drafter=drafter, selector=selector)
                record.update(scheduled)
                if "h2d_prefetch" in scheduled:
                    record["h2d_prefetch"] = enabled
                active_shard["records"].append(record)
                print(json.dumps(dict(state_id=sid, **scheduled, committed=len(
                    record["acceptance"]["committed_token_ids"]), selected=len(record["selected_raw_ids"]),
                    cycle_ms=record["timings"]["cycle_wall_ns"] / 1e6,
                    prepare_seconds=record["prepare_wall_ns"] / 1e9)), flush=True)
                check_budget()
            validate_records(active_shard["records"], schedule, state=state, source_state=source,
                             expected_choices=expected)
            metadata["sha256_after"] = checkpoint_digest(checkpoint)
            metadata["unchanged"] = metadata["sha256_after"] == metadata["sha256_before"]
            validate_checkpoint(metadata, config=config)
            active_shard["status"] = "complete"
            shard_path = args.output.with_name(f"{args.output.stem}-{sid}.json")
            index["shards"].append(write_artifact(shard_path, active_shard))
            checkpoint = active_shard = common = None
            check_budget()
        gc.enable()
        gc.collect()
        shards = list(read_shards(index, args.output.parent))
        if calibration:
            model, diagnostics = calibration_model_from_states(shards)
            index.update(cost_model=model.to_dict(), fit_diagnostics=diagnostics)
        else:
            index["summary"] = summarizer([r for s in shards for r in s["records"]])
        index.update(status="complete", gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            entry_to_validation_complete_wall_ns=perf_counter_ns() - args.entry_started_ns)
        if calibration:
            validate_calibration(index, frozen, args.output.parent)
        else:
            evaluation_validator(index, frozen, sources, args.output.parent)
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
    entry = perf_counter_ns()
    args = arguments()
    args.entry_started_ns = entry
    args.git_commit = head()
    identity = measurement_identity(ROOT)
    if args.action != "validate" and args.output.exists():
        raise FileExistsError(args.output)
    sources = None
    if args.stage == "calibration":
        expected = build_calibration_manifest(git_commit=args.git_commit, identity=identity,
                                               dataset_path=args.dataset)
    else:
        sources = load_sources(capture_path=args.capture, screen_path=args.screen,
            workload_path=args.dataset, token_lock_path=args.token_lock)
        model = checked_calibration(args, identity)
        expected = build_manifest(sources, git_commit=args.git_commit, identity=identity,
            calibration_descriptor=file_identity(args.calibration), calibration_model=model)
    if args.action == "freeze":
        print(json.dumps(write_artifact(args.output, expected)))
        return
    if args.manifest is None:
        raise ValueError("run/validate requires --manifest")
    manifest = read_artifact(args.manifest)
    expected["git_commit"] = manifest["git_commit"]
    if not same_json_value(manifest, expected):
        raise ValueError("manifest differs from the current measurement code, inputs or schedule")
    args.entry_preparation_timings_ns = dict(source_and_manifest_validation=perf_counter_ns() - entry)
    if args.action == "run":
        run_gpu(args, manifest, sources)
    elif args.stage == "calibration":
        model = validate_calibration(read_artifact(args.output), manifest, args.output.parent)
        print(json.dumps(dict(status="valid", source=file_identity(args.output), cost_model=model.to_dict())))
    else:
        summary = validate_evaluation(read_artifact(args.output), manifest, sources, args.output.parent)
        print(json.dumps(dict(status="valid", source=file_identity(args.output), summary=summary)))


if __name__ == "__main__":
    main()
