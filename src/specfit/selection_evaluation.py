"""Frozen calibration and full-state comparisons with stable, named baselines."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import random
from statistics import mean
from types import SimpleNamespace

from .experiment import canonical_json_sha256
from .f0_direct import CAPTURE_SOURCE, PACKED_SOURCE
from .f0_direct_analysis import require_ns, trial_signature, validate_trial
from .f0_proxy_screen import F0_M1_SCREEN_SOURCE
from .m1_capture_replay import same_json_value, read_routes, logical_state_sha256
from .r6a_experiment import PoolRecord
from .r6a_identity import R6A_MODEL_IDENTITIES
from .r6a_tb2_contract import R6A_TB2_IMAGE_ID, R6A_TB2_NUMERIC_SETTINGS
from .selection_methods import (METHODS, METHOD_NAMES, SelectionCostModel, fit_cost_model,
    select_nodes, selection_features, probability_ids, minimum_expert_ids)

PROTOCOL = "selection-comparison-v1"
CALIBRATION_PROTOCOL = "selection-cost-calibration-v1"
CALIBRATION_QUESTIONS = (401, 402, 403)
CONFIG = dict(max_draft_nodes=7, repetitions=3, warmups=1, audits=1,
    cache_modes=["cold", "natural"], schedule_seed=480, sampled_states=32,
    gc_policy="collect_between_states_disabled_within_state",
    route_information="offline_full_pool_atlas", draft_pool_nodes=18,
    preparation_method="pinned_checkpoint_restore", max_wall_seconds=10800,
    max_gpu_reserved_bytes=38 * 1024**3, max_process_rss_bytes=128 * 1024**3,
    max_checkpoint_host_bytes=24 * 1024**3)
CALIBRATION_CONFIG = dict(question_ids=list(CALIBRATION_QUESTIONS), state="first_postprefill_boundary",
    cache_mode="natural", node_counts=list(range(8)), repetitions=3, warmups=1, audits=1,
    selection_probes=["calibration_probability", "calibration_minimum_experts"],
    schedule_seed=480, gc_policy=CONFIG["gc_policy"], max_wall_seconds=5400,
    max_gpu_reserved_bytes=CONFIG["max_gpu_reserved_bytes"],
    max_process_rss_bytes=CONFIG["max_process_rss_bytes"],
    max_checkpoint_host_bytes=CONFIG["max_checkpoint_host_bytes"])


def file_identity(path: Path) -> dict:
    raw = path.read_bytes()
    return dict(basename=path.name, size_bytes=len(raw), sha256=sha256(raw).hexdigest())


def write_artifact(path: Path, value: dict) -> dict:
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
    descriptor = file_identity(path)
    if descriptor["sha256"] != sha256(raw).hexdigest():
        raise RuntimeError("artifact readback differs from the serialized bytes")
    return descriptor


def read_artifact(path: Path, descriptor: dict | None = None) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("artifact must be a regular file")
    raw = path.read_bytes()
    if descriptor is not None and descriptor != dict(
            basename=path.name, size_bytes=len(raw), sha256=sha256(raw).hexdigest()):
        raise ValueError("artifact byte identity differs from its index")
    def unique_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("artifact contains duplicate object keys")
            value[key] = item
        return value
    def invalid_constant(value):
        raise ValueError("artifact contains a nonfinite number")
    def finite_float(value):
        result = float(value)
        if not isfinite(result):
            raise ValueError("artifact contains a nonfinite number")
        return result
    value = json.loads(raw, object_pairs_hook=unique_pairs,
                       parse_constant=invalid_constant, parse_float=finite_float)
    if not isinstance(value, dict):
        raise ValueError("artifact root must be an object")
    return value


def measurement_identity(root: Path) -> dict:
    # Analysis/report changes can be independently validated without rerunning an
    # unchanged measurement implementation. Every actual cycle dependency is bound.
    paths = sorted([*(p for p in root.joinpath("src/specfit").rglob("*.py")
                      if p.name != "selection_evaluation.py"),
        *(root / name for name in ("scripts/benchmark_qwen3_selection.py",
            "scripts/benchmark_qwen3_m1_full_atlas.py", "docker/models.lock.json",
            "docker/requirements.overlay.lock", "docker/base-packages.lock.json"))])
    files = [dict(path=str(path.relative_to(root)), sha256=sha256(path.read_bytes()).hexdigest())
             for path in paths]
    return dict(files=files, sha256=canonical_json_sha256(files))


def calibration_schedule(question_id: int) -> list[dict]:
    rng = random.Random(CALIBRATION_CONFIG["schedule_seed"] + question_id)
    schedule = []
    for phase, repetitions in (("warmup", 1), ("audit", 1), ("measurement", 3)):
        for repetition in range(repetitions):
            jobs = [(method, k) for method in CALIBRATION_CONFIG["selection_probes"] for k in range(8)]
            rng.shuffle(jobs)
            for method, count in jobs:
                schedule.append(dict(execution_index=len(schedule), phase=phase, repetition=repetition,
                    method=method, draft_node_count=count, mode="natural"))
    return schedule


def build_calibration_manifest(*, git_commit, identity, dataset_path):
    from .gsm8k import load_gsm8k_specbench
    examples = {e.question_id: e for e in load_gsm8k_specbench(dataset_path)}
    return dict(artifact_type="selection_calibration_manifest", protocol=CALIBRATION_PROTOCOL,
        git_commit=git_commit, measurement_identity=identity, config=CALIBRATION_CONFIG,
        dataset=file_identity(dataset_path), prompts=[dict(question_id=q,
            raw_prompt=examples[q].prompt, schedule=calibration_schedule(q))
            for q in CALIBRATION_QUESTIONS])


def calibration_choices(pool, atlas):
    return {method: {str(k): dict(selected_raw_ids=list(ids),
                **selection_features(pool, atlas, ids))
            for k in range(8)
            for ids in [(probability_ids(pool, k) if method == "calibration_probability"
                         else minimum_expert_ids(pool, atlas, k))]}
        for method in CALIBRATION_CONFIG["selection_probes"]}


def evaluation_schedule(state_id: str, *, methods=METHODS) -> list[dict]:
    rng = random.Random(f"{CONFIG['schedule_seed']}:{state_id}")
    schedule = []
    for phase, repetitions in (("warmup", 1), ("audit", 1), ("measurement", CONFIG["repetitions"])):
        for repetition in range(repetitions):
            shuffled = list(methods)
            rng.shuffle(shuffled)
            for method_index, method in enumerate(shuffled):
                modes = CONFIG["cache_modes"] if (method_index + repetition) % 2 == 0 else CONFIG["cache_modes"][::-1]
                for mode in modes:
                    schedule.append(dict(execution_index=len(schedule), phase=phase,
                        repetition=repetition, method=method, mode=mode))
    return schedule


def build_manifest(sources, *, git_commit: str, identity: dict,
                   calibration_descriptor: dict, calibration_model: SelectionCostModel) -> dict:
    states = []
    for state in sources.projection.states:
        choices = {}
        for method in METHODS:
            selected = select_nodes(state.pool, state.atlas, method, calibration_model)
            choices[method] = dict(selected_raw_ids=list(selected),
                **selection_features(state.pool, state.atlas, selected))
        states.append(dict(state_id=state.state_id, question_id=state.question_id,
            cycle_index=state.cycle_index, choices=choices, schedule=evaluation_schedule(state.state_id)))
    if (len(states) != 32 or len({s["question_id"] for s in states}) != 8
            or set(CALIBRATION_QUESTIONS) & {s["question_id"] for s in states}):
        raise ValueError("evaluation must cover 32 states from eight separate evaluation prompts")
    order = [s["state_id"] for s in states]
    random.Random(CONFIG["schedule_seed"]).shuffle(order)
    return dict(artifact_type="selection_evaluation_manifest", protocol=PROTOCOL,
        git_commit=git_commit, measurement_identity=identity, config=dict(CONFIG),
        methods=[dict(id=method, name=METHOD_NAMES[method]) for method in METHODS],
        primary_baselines=["target_only", "probability_fixed"],
        dynamic_baseline="probability_dynamic", state_order=order, states=states,
        input_sources=dict(route_capture=CAPTURE_SOURCE, route_screen=F0_M1_SCREEN_SOURCE,
                           packed_weights=PACKED_SOURCE),
        calibration_source=calibration_descriptor, cost_model=calibration_model.to_dict(),
        counts=dict(states=32, prompts=8, warmups=320, audits=320, measurements=960),
        aggregation="per_state_means_then_ratio_of_sums",
        limitations=dict(local_cycles_only=True, offline_route_collection_cost_excluded=True,
            complete_selector_measured_in_cycle=True, full_draft_pool_built_before_selection=True,
            historical_timing_combined=False, independent_held_out_prompt_claim=False))


def validate_manifest(manifest, sources, *, identity: dict) -> None:
    expected = build_manifest(sources, git_commit=manifest["git_commit"], identity=identity,
        calibration_descriptor=manifest["calibration_source"],
        calibration_model=SelectionCostModel.from_dict(manifest["cost_model"]))
    if not same_json_value(expected, manifest):
        raise ValueError("evaluation manifest differs from its frozen methods or schedule")


GPU_ENVIRONMENTS = {None: dict(gpu_name="NVIDIA A100-PCIE-40GB", server="a100-host", gpu_index=0),
    # Second device: host rtx4090-host's RTX 4090, at the expert slots of its main comparison (48 do not fit).
    "rtx4090": dict(gpu_name="NVIDIA GeForce RTX 4090", server="rtx4090-host", gpu_index=1, slots_per_layer=32)}


def validate_environment(environment: dict) -> None:
    device = {k: environment[k] for k in ("gpu_name", "server", "gpu_index", "slots_per_layer") if k in environment}
    if (environment.get("image_id") != R6A_TB2_IMAGE_ID
            or device not in GPU_ENVIRONMENTS.values()
            or environment.get("packed_descriptor") != PACKED_SOURCE
            or not same_json_value(environment.get("numeric_settings"), R6A_TB2_NUMERIC_SETTINGS)
            or any(environment.get("models", {}).get(k) != v for k, v in R6A_MODEL_IDENTITIES.items())):
        raise ValueError("measurement environment differs from the frozen model/runtime configuration")


def validate_audit(record: dict, state, source_state: dict) -> None:
    if record.get("boundary_checks") != dict(generation_state_exact=True, expert_weights_exact=True):
        raise ValueError("checkpoint boundary audit is incomplete")
    audit = record.get("commit_audit") or {}
    if record["method"] == "target_only":
        if (audit.get("kind") != "target_only" or audit.get("kv_layer_count") != 48
                or audit.get("verification_node_count") != 1
                or any(audit.get(k) is not True for k in
                       ("kv_prefix_exact", "kv_length_exact", "pending_seed_exact"))):
            raise ValueError("target-only KV/commit audit failed")
        return
    selected = record["selected_raw_ids"]
    positions = [0, *(selected.index(raw) + 1 for raw in record["acceptance"]["accepted_raw_ids"])]
    flags = ("kv_prefix_exact", "kv_selected_suffix_exact", "kv_length_exact", "aux_length_exact",
             "aux_prefix_exact", "aux_selected_suffix_exact", "pending_seed_exact")
    if (any(audit.get(k) is not True for k in flags) or audit.get("kv_layer_count") != 48
            or audit.get("iteration") != state.cycle_index
            or audit.get("base_kv_length") != source_state["base_kv_length"]
            or audit.get("verification_node_count") != 1 + len(selected)
            or list(audit.get("verification_positions", ())) != positions
            or list(audit.get("rejected_node_ids", ())) != [i for i in range(1 + len(selected)) if i not in positions]
            or audit.get("aux_prefix_length") != source_state["base_kv_length"]
            or audit.get("aux_length_after") != record["acceptance"]["kv_length_after"]):
        raise ValueError("tree KV/aux commit audit failed")


def validate_records(records, schedule, *, state, source_state, expected_choices, max_nodes=7):
    if len(records) != len(schedule):
        raise ValueError("state measurement coverage is incomplete")
    signatures = {}
    for record, scheduled in zip(records, schedule, strict=True):
        if not same_json_value({k: record.get(k) for k in scheduled}, scheduled):
            raise ValueError("record identity or execution order differs from the schedule")
        key = record["method"], record.get("draft_node_count"), record["mode"]
        if record.get("question_id") != state.question_id:
            raise ValueError("record prompt differs from its frozen state")
        selected = expected_choices[record["method"], record.get("draft_node_count")]
        validate_trial(record, job=dict(job_id=record["job_id"], selected_raw_ids=list(selected)),
                       state=state, source_state=source_state, variable_budget=True, max_draft_nodes=max_nodes)
        if (record.get("preparation_method") != "checkpoint_restore"
                or record.get("gc_enabled_during_cycle") is not False
                or type(record.get("draft_forward_calls")) is not int
                or record["draft_forward_calls"] < 0
                or (record["method"] != "target_only" and record["draft_forward_calls"] == 0)):
            raise ValueError("record changed preparation, GC or draft execution semantics")
        interval = record.get("cycle_wall_interval_ns", ())
        if (len(interval) != 2 or any(type(x) is not int for x in interval)
                or interval[1] - interval[0] != record["timings"]["cycle_wall_ns"]):
            raise ValueError("cycle timestamps do not reproduce the measured duration")
        signature = trial_signature(record)
        if record["phase"] == "warmup":
            if key in signatures:
                raise ValueError("duplicate state/method warmup")
            signatures[key] = signature
        elif signatures.get(key) != signature:
            raise ValueError("routes, cache or committed tokens changed between repetitions")
        if record["phase"] == "audit":
            validate_audit(record, state, source_state)


def validate_checkpoint(record: dict, *, config: dict) -> None:
    if (record.get("sha256_before") != record.get("sha256_after")
            or not isinstance(record.get("sha256_before"), str)
            or len(record["sha256_before"]) != 64
            or record.get("unchanged") is not True
            or not 0 < record.get("expert_weight_host_bytes", 0) <= config["max_checkpoint_host_bytes"]
            or not record["expert_weight_host_bytes"] <= record.get("pinned_allocation_bytes", 0)
                       <= config["max_checkpoint_host_bytes"]):
        raise ValueError("checkpoint identity, immutability or memory budget failed")


def validate_run_budget(index, config):
    for field, maximum in (("gpu_peak_reserved_bytes", config["max_gpu_reserved_bytes"]),
            ("process_peak_rss_bytes", config["max_process_rss_bytes"]),
            ("entry_to_validation_complete_wall_ns", config["max_wall_seconds"] * 10**9)):
        if not 0 < require_ns(index.get(field), field) <= maximum:
            raise ValueError(f"run exceeded its frozen {field} budget")


def _validate_index(index, manifest):
    if (index.get("status") != "complete" or index.get("protocol") != manifest["protocol"]
            or index.get("manifest_sha256") != canonical_json_sha256(manifest)
            or index.get("measurement_identity") != manifest["measurement_identity"]
            or index.get("git_commit") != manifest["git_commit"]):
        raise ValueError("result is incomplete or belongs to a different frozen measurement")
    validate_environment(index["environment"])
    validate_run_budget(index, manifest["config"])


def read_shards(index, directory):
    seen = set()
    for descriptor in index["shards"]:
        name = descriptor["basename"]
        if Path(name).name != name or name in seen:
            raise ValueError("shard name must be unique and local to the result directory")
        seen.add(name)
        yield read_artifact(directory / name, descriptor)


def validate_calibration(index, manifest, directory):
    _validate_index(index, manifest)
    shards = list(read_shards(index, directory))
    if [s["question_id"] for s in shards] != list(CALIBRATION_QUESTIONS):
        raise ValueError("calibration prompt coverage changed")
    for shard, prompt in zip(shards, manifest["prompts"], strict=True):
        if shard.get("manifest_sha256") != index["manifest_sha256"] or shard.get("status") != "complete":
            raise ValueError("calibration shard is incomplete or unbound")
        prepared = shard["prepared_prompt"]
        source = shard["source_state"]
        pool = PoolRecord.read(shard["pool"])
        state = SimpleNamespace(state_id=shard["state_id"], question_id=shard["question_id"],
            cycle_index=0, pool=pool, atlas=read_routes(shard["route_probe"]["routes"],
                "calibration route atlas", pool, tuple(range(19))))
        if (prepared["raw_prompt"] != prompt["raw_prompt"]
                or prepared["question_id"] != prompt["question_id"]
                or len(prepared["input_ids"]) != source["base_kv_length"]
                or prepared["attention_mask"] != [1] * source["base_kv_length"]
                or source["pre_logical_state_sha256"] != logical_state_sha256(
                    shard["route_probe"]["pre_logical_state"])):
            raise ValueError("calibration prompt or initial boundary changed")
        probe = shard["route_probe"]
        validate_trial(probe, job=dict(job_id=probe["job_id"], selected_raw_ids=list(range(1, 19))),
            state=state, source_state=source, variable_budget=True, max_draft_nodes=18)
        validate_audit(probe, state, source)
        expected = calibration_choices(pool, state.atlas)
        if not same_json_value(shard["choices"], expected):
            raise ValueError("calibration choice used information outside its frozen probes")
        choices = {(method, k): tuple(expected[method][str(k)]["selected_raw_ids"])
            for method in CALIBRATION_CONFIG["selection_probes"] for k in range(8)}
        validate_records(shard["records"], prompt["schedule"], state=state, source_state=source,
                         expected_choices=choices)
        validate_checkpoint(shard["checkpoint"], config=CALIBRATION_CONFIG)
    model, diagnostics = calibration_model_from_states(shards)
    if not same_json_value(index["cost_model"], model.to_dict()) or not same_json_value(
            index["fit_diagnostics"], diagnostics):
        raise ValueError("calibration model cannot be reproduced from its measurements")
    return model


def validate_evaluation(index, manifest, sources, directory):
    _validate_index(index, manifest)
    states = {s.state_id: s for s in sources.projection.states}
    raw = {s["state_id"]: s for s in sources.capture["states"]}
    decisions = {s["state_id"]: s for s in manifest["states"]}
    shards = list(read_shards(index, directory))
    if [s["state_id"] for s in shards] != manifest["state_order"]:
        raise ValueError("evaluation state coverage or order changed")
    records = []
    for shard in shards:
        sid = shard["state_id"]
        if shard.get("manifest_sha256") != index["manifest_sha256"] or shard.get("status") != "complete":
            raise ValueError("evaluation shard is incomplete or unbound")
        choices = {(method, None): tuple(decisions[sid]["choices"][method]["selected_raw_ids"])
                   for method in METHODS}
        validate_records(shard["records"], decisions[sid]["schedule"], state=states[sid],
                         source_state=raw[sid], expected_choices=choices)
        validate_checkpoint(shard["checkpoint"], config=CONFIG)
        records.extend(shard["records"])
    summary = summarize_measurements(records)
    if not same_json_value(index["summary"], summary):
        raise ValueError("summary cannot be reproduced from all scheduled measurements")
    return summary


def calibration_model_from_states(states: list[dict]) -> tuple[SelectionCostModel, dict]:
    points = []
    for state in states:
        for method in CALIBRATION_CONFIG["selection_probes"]:
            for count in range(8):
                rows = [r for r in state["records"] if r["phase"] == "measurement"
                        and r["method"] == method and r["draft_node_count"] == count]
                if len(rows) != CALIBRATION_CONFIG["repetitions"]:
                    raise ValueError("calibration repetition coverage differs from its contract")
                features = state["choices"][method][str(count)]
                points.append(dict(question_id=state["question_id"], method=method,
                    draft_node_count=count, expert_union=features["expert_union_proxy"],
                    cycle_without_selector_ns=mean(r["timings"]["cycle_wall_ns"]
                        - r["timings"]["explicit_selector_ns"] for r in rows)))
    model, diagnostics = fit_cost_model(points)
    return model, {**diagnostics, "points": points}


COMPARISONS = (("expert_cost_fixed", "probability_fixed"),
    ("probability_dynamic", "probability_fixed"),
    ("expert_cost_dynamic", "expert_cost_fixed"),
    ("expert_cost_dynamic", "probability_dynamic"))


def summarize_measurements(records: list[dict], *, methods=METHODS,
                           method_names=METHOD_NAMES, comparisons=COMPARISONS) -> dict:
    by_mode = {}
    for mode in CONFIG["cache_modes"]:
        method_states = {}
        for method in methods:
            relevant = [r for r in records if r["phase"] == "measurement"
                        and r["mode"] == mode and r["method"] == method]
            states = {}
            for sid in sorted({r["state_id"] for r in relevant}):
                rows = [r for r in relevant if r["state_id"] == sid]
                states[sid] = dict(question_id=rows[0]["question_id"], repetitions=len(rows),
                    committed_tokens=mean(len(r["acceptance"]["committed_token_ids"]) for r in rows),
                    cycle_ns=mean(r["timings"]["cycle_wall_ns"] for r in rows),
                    selector_ns=mean(r["timings"]["explicit_selector_ns"] for r in rows),
                    proposal_ns=mean(r["timings"]["proposal_wall_ns"] for r in rows),
                    verification_cuda_ns=mean(r["timings"]["target_verification_cuda_ns"] for r in rows),
                    draft_nodes=len(rows[0]["selected_raw_ids"]),
                    expert_union_actual=mean(r["F_own"] for r in rows),
                    h2d_submissions=mean(r["cache_counter_delta"]["h2d_submissions"] for r in rows),
                    cycle_samples_ns=[r["timings"]["cycle_wall_ns"] for r in rows])
            if not states:
                raise ValueError("a named baseline is missing from the comparison")
            method_states[method] = states
        populations = [set(rows) for rows in method_states.values()]
        if any(population != populations[0] for population in populations):
            raise ValueError("methods were evaluated on different state populations")

        def rate(method, questions=None):
            rows = [r for r in method_states[method].values()
                    if questions is None or r["question_id"] in questions]
            return sum(r["committed_tokens"] for r in rows) * 1e9 / sum(r["cycle_ns"] for r in rows)

        method_summary = {}
        for method, states in method_states.items():
            method_summary[method] = dict(name=method_names[method], tokens_per_second=rate(method),
                gain_vs_target_only=rate(method) / rate("target_only") - 1,
                gain_vs_probability_fixed=rate(method) / rate("probability_fixed") - 1,
                mean_draft_nodes=mean(r["draft_nodes"] for r in states.values()),
                node_count_distribution={str(count): frequency for count, frequency in sorted(
                    Counter(r["draft_nodes"] for r in states.values()).items())},
                mean_selector_ns=mean(r["selector_ns"] for r in states.values()),
                mean_proposal_ns=mean(r["proposal_ns"] for r in states.values()),
                mean_verification_cuda_ns=mean(r["verification_cuda_ns"] for r in states.values()),
                total_mean_h2d_submissions=sum(r["h2d_submissions"] for r in states.values()),
                states=states)
        questions = sorted({r["question_id"] for r in method_states["target_only"].values()})
        comparison_summary = {}
        for method, baseline in comparisons:
            gains = {str(q): rate(method, {q}) / rate(baseline, {q}) - 1 for q in questions}
            comparison_summary[f"{method}_vs_{baseline}"] = dict(gain=rate(method) / rate(baseline) - 1,
                prompt_gains=gains, positive_prompts=sum(g > 0 for g in gains.values()),
                negative_prompts=sum(g < 0 for g in gains.values()),
                paired_prompt_bootstrap=prompt_bootstrap(method_states[method], method_states[baseline]))
        by_mode[mode] = dict(methods=method_summary, comparisons=comparison_summary,
                             state_count=len(populations[0]), prompt_count=len(questions))
    return dict(by_mode=by_mode, historical_timing_combined=False,
                scope="local_cycles_with_offline_route_information")


def prompt_bootstrap(method, baseline):
    """Resample whole prompts, retaining correlated states and paired methods."""
    questions = sorted({row["question_id"] for row in method.values()})
    totals = {}
    for q in questions:
        totals[q] = tuple((sum(r["committed_tokens"] for r in rows.values() if r["question_id"] == q),
                           sum(r["cycle_ns"] for r in rows.values() if r["question_id"] == q))
                          for rows in (method, baseline))
    rng = random.Random(480)
    gains = []
    for _ in range(10000):
        sampled = [totals[rng.choice(questions)] for _ in questions]
        rates = [sum(row[i][0] for row in sampled) / sum(row[i][1] for row in sampled) for i in (0, 1)]
        gains.append(rates[0] / rates[1] - 1)
    gains.sort()
    def percentile(p):
        x = (len(gains) - 1) * p
        low = int(x)
        return gains[low] + (gains[min(low + 1, len(gains) - 1)] - gains[low]) * (x - low)
    return dict(resamples=10000, seed=480, unit="prompt", prompt_count=len(questions),
        percentile_95_interval=[percentile(0.025), percentile(0.975)],
        scope="descriptive_uncertainty_for_the_eight_fixed_evaluation_prompts")
