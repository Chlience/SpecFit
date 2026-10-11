"""Frozen selection-by-executor experiment with audited warmup and two samples."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
import random
from statistics import mean
from types import SimpleNamespace

from . import selection_evaluation as base
from .cross_layer_prefetch import predictions_for, replay_events
from .experiment import canonical_json_sha256
from .f0_direct import expert_union
from .f0_direct_analysis import COUNTERS, EXPERT_BYTES, cache_totals, require_ns, validate_outcome, validate_trial
from .m1_capture_replay import logical_state_sha256, read_logical_state, read_routes, same_json_value
from .r6a_experiment import PoolRecord
from .selection_methods import METHODS, METHOD_NAMES, SelectionCostModel, fit_cost_model, select_nodes, selection_features

PROTOCOL = "selection-executor-matrix-v1"
CALIBRATION_PROTOCOL = "selection-executor-matrix-calibration-v1"
EXECUTORS = ("demand", "one_ahead", "ready_demand", "predicted_same_layer")
SCORED_CONTROLS = ("scored_lru", "scored_probability")
EXECUTOR_NAMES = dict(demand="on-demand execution", one_ahead="one expert ahead in the same layer",
    ready_demand="dynamic-ready empty prediction", predicted_same_layer="dynamic-ready plus same-layer exact expert prediction")
CONFIG = dict(base.CONFIG, repetitions=2, warmups=0, audits=1, warmup_in_audit=True,
    automatic_extra_repetitions=False)
CALIBRATION_CONFIG = dict(base.CALIBRATION_CONFIG, repetitions=2, warmups=0, audits=1,
    warmup_in_audit=True)
SMOKE_STATES = ("q416-c37", "q466-c77", "q426-c16")
REFERENCE_FIELDS = ("selected_raw_ids", "pool_fingerprint", "routes", "acceptance",
                    "logits_shape", "logits_dtype", "logits_sha256")


def identity(root: Path):
    files = {row["path"]: row for row in base.measurement_identity(root)["files"]}
    for name in ("src/specfit/selection_evaluation.py", "scripts/benchmark_qwen3_selection_matrix.py"):
        files[name] = dict(path=name, sha256=base.file_identity(root / name)["sha256"])
    rows = [files[key] for key in sorted(files)]
    return dict(files=rows, sha256=canonical_json_sha256(rows))


def schedule(state_id, *, calibration=False, scored_controls=False):
    rng = random.Random("matrix-480:" + str(state_id))
    if calibration:
        jobs = [dict(method=m, draft_node_count=k, execution=e, mode="natural")
                for e in EXECUTORS for m in CALIBRATION_CONFIG["selection_probes"] for k in range(8)]
    else:
        jobs = [dict(method=m, execution=e, mode=mode)
                for e in EXECUTORS for m in METHODS for mode in CONFIG["cache_modes"]]
        if scored_controls:
            jobs += [dict(method="expert_cost_dynamic", execution=e, mode=mode)
                     for e in SCORED_CONTROLS for mode in CONFIG["cache_modes"]]
    rows = []
    for phase, repetitions in (("audit", 1), ("measurement", 2)):
        for repetition in range(repetitions):
            order = list(jobs)
            rng.shuffle(order)
            for job in order:
                rows.append(dict(job, phase=phase, repetition=repetition, execution_index=len(rows)))
    return rows


def calibration_manifest(*, git_commit, measurement_identity, dataset_path):
    manifest = deepcopy(base.build_calibration_manifest(git_commit=git_commit,
        identity=measurement_identity, dataset_path=dataset_path))
    manifest.update(protocol=CALIBRATION_PROTOCOL, config=CALIBRATION_CONFIG,
        executors=list(EXECUTORS), reference_h2d_prefetch=False,
        prediction_source="separate exact selected-query demand runs before all scheduled cycles of each state")
    for prompt in manifest["prompts"]:
        prompt["schedule"] = schedule(prompt["question_id"], calibration=True)
    return manifest


def evaluation_manifest(sources, *, git_commit, measurement_identity, calibration_descriptor, models,
                        scored_controls=False):
    if set(models) != set(EXECUTORS):
        raise ValueError("every executor needs its own frozen calibration")
    manifest = base.build_manifest(sources, git_commit=git_commit, identity=measurement_identity,
        calibration_descriptor=calibration_descriptor, calibration_model=models["demand"])
    states = {s.state_id: s for s in sources.projection.states}
    for row in manifest["states"]:
        state = states[row["state_id"]]
        row["choices"] = {e: {m: dict(selected_raw_ids=list(ids),
            **selection_features(state.pool, state.atlas, ids)) for m in METHODS
            for ids in [select_nodes(state.pool, state.atlas, m, models[e])]} for e in EXECUTORS}
        if scored_controls:
            for e in SCORED_CONTROLS:
                row["choices"][e] = {"expert_cost_dynamic": deepcopy(row["choices"]["predicted_same_layer"]["expert_cost_dynamic"])}
        row["schedule"] = schedule(state.state_id, scored_controls=scored_controls)
    manifest.update(protocol=PROTOCOL, config=CONFIG, executors=list(EXECUTORS),
        cost_models={e: m.to_dict() for e, m in models.items()},
        execution_batches=[manifest["state_order"][i:i + 8] for i in range(0, 32, 8)],
        smoke_states=list(SMOKE_STATES), reference_h2d_prefetch=False,
        counts=dict(states=32, prompts=8, combinations=20, warmup_audits=1280, measurements=2560, cycles=3840),
        prediction_source="exact selected-query demand replay from the same checkpoint; bound and written before state measurements",
        scored_controls=list(SCORED_CONTROLS) if scored_controls else [])
    manifest["counts"].update(main_cycles=3840, supplementary_cycles=384 if scored_controls else 0,
        cycles=4224 if scored_controls else 3840, supplementary_combinations=2 if scored_controls else 0)
    manifest.pop("cost_model")
    manifest["limitations"].update(cross_executor_comparison_includes_recalibration=True,
        separate_warmup_removed=True, no_automatic_extra_repetitions=True,
        prediction_collection_cost_excluded=True, prediction_ranking="expert ID ascending",
        lookahead_layers=0, max_inflight=2)
    return manifest


def query_key(method, ids):
    return ("target" if method == "target_only" else "tree") + ":" + ",".join(map(str, ids))


def reference_value(record):
    return {field: record[field] for field in REFERENCE_FIELDS}


def validate_cycle(record, *, state, source_state):
    """Validate real cache decisions for all four executors, including no draft."""
    execution, selected = record["execution"], tuple(record["selected_raw_ids"])
    if execution not in (*EXECUTORS, *SCORED_CONTROLS):
        raise ValueError("unknown matrix executor")
    by_id = {node.raw_id: node for node in state.pool.nodes}
    if (len(set(selected)) != len(selected) or len(selected) > 18
            or any(i not in by_id for i in selected)
            or any(by_id[i].parent_raw_id not in (0, *selected) for i in selected)
            or record["state_id"] != state.state_id or record["question_id"] != state.question_id
            or record["pool_fingerprint"] != state.pool.fingerprint
            or record["preparation_method"] != "checkpoint_restore"
            or record["reference_cache_sha256"] != source_state["pre_logical_state_sha256"]):
        raise ValueError("matrix query or checkpoint identity changed")
    capacity = getattr(state, 'slots_per_layer', 48)
    pre, _ = read_logical_state(record["pre_logical_state"], "matrix restored cache",
                                expected_capacity=capacity)
    if record["mode"] == "natural":
        if logical_state_sha256(pre) != record["reference_cache_sha256"]:
            raise ValueError("natural cache restoration changed")
    elif record["mode"] == "cold":
        if any(s["expert_id"] is not None for layer in pre["layers"] for s in layer["slots"]) or any(
                value for layer in pre["layers"] for value in layer["counters"].values()):
            raise ValueError("cold cache was not cleared")
    else:
        raise ValueError("unknown cache mode")
    routes = read_routes(record["routes"], "matrix own routes", state.pool, (0, *selected))
    unions = expert_union(routes, selected)
    actual = {(lid, e) for lid, ids in enumerate(unions) for e in ids}
    post = replay_events(pre, record["cache_events"], actual,
                         "ready_demand" if execution in SCORED_CONTROLS else execution)
    counters = {k: cache_totals(post)[k] - cache_totals(pre)[k] for k in COUNTERS}
    if (logical_state_sha256(post) != record["post_logical_state_sha256"]
            or counters != record["cache_counter_delta"] or record["F_own"] != len(actual)
            or record["max_own_layer_union"] != max(map(len, unions))
            or record["logical_h2d_bytes"] != counters["h2d_submissions"] * EXPERT_BYTES):
        raise ValueError("matrix cache event replay or H2D accounting changed")
    validate_outcome(record, state.pool, base_length=source_state["base_kv_length"], selected=selected)
    if execution in ("demand", "one_ahead"):
        validate_trial(record, job=dict(job_id=record["job_id"], selected_raw_ids=list(selected)),
            state=state, source_state=source_state, variable_budget=True, max_draft_nodes=18)
        if record["prefetch_stats"]:
            raise ValueError("static executor unexpectedly used the prediction scheduler")
    else:
        stats = record["prefetch_stats"]
        if (stats.get("complete") is not True or stats["max_inflight"] != 2
                or stats["peak_inflight"] > 2 or stats["lookahead_layers"] != 0
                or stats["prefetch_slot_limits"] != [capacity - 1] * 48
                or len(stats["layers"]) != 48
                or sum(layer["computed_experts"] for layer in stats["layers"]) != len(actual)):
            raise ValueError("matrix scheduler budget or completion failed")
        copies = [0] * 48
        for event in record["cache_events"]:
            if event[0] == "h2d_submit":
                copies[event[1]] += 1
        for lid, layer in enumerate(stats["layers"]):
            if (layer["actual_experts"] != len(unions[lid]) or layer["computed_experts"] != len(unions[lid])
                    or layer["ready_at_router"] + layer["inflight_at_router"] + layer["missing_at_router"] != len(unions[lid])
                    or layer["prediction_copies"] + layer["demand_copies"] != copies[lid]
                    or layer["unused_prediction_copies"] != 0
                    or layer["predicted_experts"] != (0 if execution == "ready_demand" else
                        min(capacity - 1, len(unions[lid])) if execution in SCORED_CONTROLS else len(unions[lid]))):
                raise ValueError("matrix predicted-expert or copy metrics changed")
    target = record["method"] == "target_only"
    timing = record["timings"]
    for name, value in timing.items():
        require_ns(value, name, allow_zero=name == "explicit_selector_ns" or (target and name == "proposal_wall_ns"))
    if (type(record["draft_forward_calls"]) is not int
            or (target and (selected or record["draft_forward_calls"] or timing["proposal_wall_ns"] or timing["explicit_selector_ns"]))
            or (not target and record["draft_forward_calls"] <= 0)
            or record["gc_enabled_during_cycle"]
            or len(record["cycle_wall_interval_ns"]) != 2
            or record["cycle_wall_interval_ns"][1] - record["cycle_wall_interval_ns"][0] != timing["cycle_wall_ns"]
            or timing["cycle_wall_ns"] <= timing["explicit_selector_ns"]):
        raise ValueError("matrix draft execution or timing semantics changed")
    digest = record["logits_sha256"]
    if (record["logits_dtype"] != "torch.bfloat16" or len(record["logits_shape"]) != 3
            or record["logits_shape"][:2] != [1, 1 + len(selected)]
            or not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("matrix full BF16 logits were not recorded")


def validate_records(records, jobs, *, state, source_state, choices, references):
    if len(records) != len(jobs):
        raise ValueError("matrix schedule coverage is incomplete")
    expected_keys = {query_key(job["method"], choices[job["execution"], job["method"], job.get("draft_node_count")]) for job in jobs}
    if set(references) != expected_keys:
        raise ValueError("query-reference coverage differs from frozen choices")
    for key, reference in references.items():
        if key != query_key(reference["method"], reference["selected_raw_ids"]) or reference["execution"] != "demand":
            raise ValueError("query reference did not use the original demand executor")
        validate_cycle(reference, state=state, source_state=source_state)
        base.validate_audit(reference, state, source_state)
    first = {}
    reference_hashes = {key: canonical_json_sha256(value) for key, value in references.items()}
    for record, job in zip(records, jobs, strict=True):
        if any(record.get(k) != v for k, v in job.items()):
            raise ValueError("matrix execution differs from frozen schedule")
        ids = choices[job["execution"], job["method"], job.get("draft_node_count")]
        if record["selected_raw_ids"] != list(ids):
            raise ValueError("matrix selected tree differs from frozen choice")
        reference_key = query_key(job["method"], ids)
        reference = references[reference_key]
        if (record["query_reference_sha256"] != reference_hashes[reference_key]
                or reference_value(record) != reference_value(reference)
                or record["query_collection_finished_ns"] >= record["cycle_wall_interval_ns"][0]):
            raise ValueError("exact query routes, logits, outcomes or prediction availability changed")
        if job["execution"] in SCORED_CONTROLS:
            predicted = predictions_for(reference, "predicted_same_layer")
            matrix = [[int(e in predicted[lid]) for e in range(128)] for lid in range(48)]
            if record.get("score_input_sha256") != canonical_json_sha256(matrix):
                raise ValueError("scored control changed exact-query binary probabilities")
        validate_cycle(record, state=state, source_state=source_state)
        key = job["execution"], job["method"], job.get("draft_node_count"), job["mode"]
        signature = reference_value(record)
        if job["phase"] == "audit":
            if key in first:
                raise ValueError("duplicate combined warmup and audit")
            base.validate_audit(record, state, source_state)
            first[key] = signature
        elif key not in first or signature != first[key]:
            raise ValueError("formal cycle lacks a matching audited warmup")


def fit_models(shards):
    models, diagnostics = {}, {}
    for execution in EXECUTORS:
        points = []
        for shard in shards:
            for method in CALIBRATION_CONFIG["selection_probes"]:
                for count in range(8):
                    rows = [r for r in shard["records"] if r["phase"] == "measurement"
                            and r["execution"] == execution and r["method"] == method and r["draft_node_count"] == count]
                    if len(rows) != 2:
                        raise ValueError("calibration needs exactly two formal samples per executor and probe")
                    features = shard["choices"][method][str(count)]
                    points.append(dict(question_id=shard["question_id"], method=method,
                        draft_node_count=count, expert_union=features["expert_union_proxy"],
                        cycle_without_selector_ns=mean(r["timings"]["cycle_wall_ns"] - r["timings"]["explicit_selector_ns"] for r in rows)))
        model, fit = fit_cost_model(points)
        models[execution], diagnostics[execution] = model, dict(fit, points=points)
    return models, diagnostics


def summarize(records):
    result = {e: base.summarize_measurements([r for r in records if r["execution"] == e]) for e in EXECUTORS}
    comparison = {mode: {method: {e: result[e]["by_mode"][mode]["methods"][method]["tokens_per_second"] /
        result["demand"]["by_mode"][mode]["methods"][method]["tokens_per_second"] - 1 for e in EXECUTORS}
        for method in METHODS} for mode in CONFIG["cache_modes"]}
    controls = {}
    for execution in SCORED_CONTROLS:
        selected = [r for r in records if r["execution"] == execution and r["phase"] == "measurement"]
        if not selected:
            continue
        controls[execution] = {}
        for mode in CONFIG["cache_modes"]:
            rows = [r for r in selected if r["mode"] == mode]
            baseline = [r for r in records if r["execution"] == "predicted_same_layer" and r["method"] == "expert_cost_dynamic"
                        and r["phase"] == "measurement" and r["mode"] == mode]
            rate = lambda rs: sum(len(r["acceptance"]["committed_token_ids"]) for r in rs)*1e9 / sum(r["timings"]["cycle_wall_ns"] for r in rs)
            if Counter(r["state_id"] for r in rows) != Counter(r["state_id"] for r in baseline):
                raise ValueError("scored controls have a different state population")
            controls[execution][mode] = dict(tokens_per_second=rate(rows), gain_vs_accurate_id_lru=rate(rows)/rate(baseline)-1,
                states={sid: dict(cycle_samples_ns=[r["timings"]["cycle_wall_ns"] for r in rows if r["state_id"]==sid],
                    selected_raw_ids=next(r["selected_raw_ids"] for r in rows if r["state_id"]==sid)) for sid in sorted({r["state_id"] for r in rows})})
    return dict(by_executor=result, combined_gain_vs_demand=comparison, scored_controls=controls,
        cross_executor_comparisons_include_recalibration=True, historical_timing_combined=False)


def validate_result(index, manifest, sources, directory):
    base._validate_index(index, manifest)
    calibration = manifest["protocol"] == CALIBRATION_PROTOCOL
    shards = list(base.read_shards(index, directory))
    if calibration:
        if [s["question_id"] for s in shards] != list(base.CALIBRATION_QUESTIONS):
            raise ValueError("calibration prompt coverage changed")
    else:
        expected_order = manifest["smoke_states"] if index["run_scope"] == "smoke" else manifest["execution_batches"][index["batch"]]
        if [s["state_id"] for s in shards] != expected_order:
            raise ValueError("evaluation state coverage or order changed")
        states = {s.state_id: s for s in sources.projection.states}
        raw = {s["state_id"]: s for s in sources.capture["states"]}
        decisions = {s["state_id"]: s for s in manifest["states"]}
    for shard in shards:
        if shard.get("status") != "complete" or shard["manifest_sha256"] != index["manifest_sha256"]:
            raise ValueError("incomplete or unbound state shard")
        sid = shard["state_id"]
        if calibration:
            prompt = next(p for p in manifest["prompts"] if p["question_id"] == shard["question_id"])
            source = shard["source_state"]
            pool = PoolRecord.read(shard["pool"])
            state = SimpleNamespace(state_id=sid, question_id=shard["question_id"], cycle_index=0,
                pool=pool, atlas=read_routes(shard["route_probe"]["routes"], "calibration full atlas", pool, tuple(range(19))))
            prepared = shard["prepared_prompt"]
            if (prepared["raw_prompt"] != prompt["raw_prompt"] or prepared["question_id"] != state.question_id
                    or len(prepared["input_ids"]) != source["base_kv_length"]
                    or prepared["attention_mask"] != [1] * source["base_kv_length"]):
                raise ValueError("calibration prompt binding changed")
            validate_cycle(shard["route_probe"], state=state, source_state=source)
            base.validate_audit(shard["route_probe"], state, source)
            expected = base.calibration_choices(pool, state.atlas)
            if shard["choices"] != expected:
                raise ValueError("calibration choices changed")
            choices = {(e, m, k): tuple(expected[m][str(k)]["selected_raw_ids"])
                for e in EXECUTORS for m in CALIBRATION_CONFIG["selection_probes"] for k in range(8)}
            jobs = prompt["schedule"]
        else:
            state, source = states[sid], raw[sid]
            choices = {(e, m, None): tuple(choice["selected_raw_ids"])
                for e, methods in decisions[sid]["choices"].items() for m, choice in methods.items()}
            jobs = decisions[sid]["schedule"]
        reference_artifact = base.read_artifact(directory / shard["query_references"]["basename"], shard["query_references"])
        if reference_artifact["manifest_sha256"] != index["manifest_sha256"] or reference_artifact["state_id"] != sid:
            raise ValueError("query reference artifact belongs to a different experiment")
        finished = reference_artifact["collection_finished_ns"]
        if (any(r["query_collection_finished_ns"] != finished for r in shard["records"])
                or any(r["cycle_wall_interval_ns"][1] >= finished for r in reference_artifact["references"].values())):
            raise ValueError("query reference collection timing changed")
        validate_records(shard["records"], jobs, state=state, source_state=source,
            choices=choices, references=reference_artifact["references"])
        base.validate_checkpoint(shard["checkpoint"], config=manifest["config"])
    if calibration:
        models, diagnostics = fit_models(shards)
        if index["cost_models"] != {e: m.to_dict() for e, m in models.items()} or index["fit_diagnostics"] != diagnostics:
            raise ValueError("executor cost models cannot be reproduced from calibration")
        return models
    summary = summarize([r for shard in shards for r in shard["records"]])
    if not same_json_value(index["summary"], summary):
        raise ValueError("matrix summary differs from original samples")
    return summary


def combine(paths, manifest, sources):
    if len(paths) != 4 or len({p.parent for p in paths}) != 1:
        raise ValueError("the matrix requires four separate validated batches")
    indexes = [base.read_artifact(p) for p in paths]
    if [i.get("batch") for i in indexes] != list(range(4)) or any(i.get("run_scope") != "batch" for i in indexes):
        raise ValueError("matrix batches must be complete, unique and ordered")
    records, states = [], []
    for path, index in zip(paths, indexes, strict=True):
        validate_result(index, manifest, sources, path.parent)
        if index["environment"] != indexes[0]["environment"]:
            raise ValueError("matrix batch environments differ")
        for shard in base.read_shards(index, path.parent):
            states.append(shard["state_id"])
            records.extend(shard["records"])
    if states != manifest["state_order"] or len(records) != manifest["counts"]["cycles"]:
        raise ValueError("combined matrix state or cycle coverage changed")
    return dict(artifact_type="selection_matrix_combined_result", protocol=PROTOCOL,
        status="complete", git_commit=manifest["git_commit"], measurement_identity=manifest["measurement_identity"],
        manifest_sha256=canonical_json_sha256(manifest), counts=manifest["counts"],
        batch_results=[base.file_identity(p) for p in paths], summary=summarize(records),
        environment=indexes[0]["environment"], resources=dict(
            max_gpu_reserved_bytes=max(i["gpu_peak_reserved_bytes"] for i in indexes),
            max_process_rss_bytes=max(i["process_peak_rss_bytes"] for i in indexes),
            batch_wall_ns=[i["entry_to_validation_complete_wall_ns"] for i in indexes]))
