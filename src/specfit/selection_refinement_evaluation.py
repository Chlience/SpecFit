"""Frozen prompt splits and independent checks for selector refinement runs."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
from statistics import mean
from types import SimpleNamespace

from . import selection_evaluation as base
from . import selection_matrix as matrix
from .experiment import canonical_json_sha256
from .gsm8k import load_gsm8k_specbench
from .m1_capture_replay import logical_state_sha256, read_routes, same_json_value
from .r6a_experiment import PoolRecord
from .selection_methods import SelectionCostModel, fit_cost_model, minimum_expert_ids, probability_ids
from .selection_refinement import (CacheCostModel, REFINEMENT_METHODS, REFINEMENT_NAMES,
    VARIANTS, acceptance_observations, acceptance_probabilities, fit_acceptance,
    fit_cache_cost_model, query_features, resident_bitmap, residents_from_logical_state, select_refined)

PROTOCOL = "selection-refinement-v1"
EXECUTION = "ready_demand"
SPLITS = {"calibration": (401, 402, 403), "development": (404, 414, 424, 434),
          "holdout": (409, 419, 429, 439, 449, 459, 469, 479)}
BOUNDARIES = {"calibration": (0, 4, 8), "development": (0, 4, 8), "holdout": (0, 4, 8, 12)}
REPETITIONS = {"calibration": 2, "development": 3, "holdout": 5}
PROBES = ("calibration_probability", "calibration_minimum_experts")
GUARD_MARGIN = 0.02
CONFIG = dict(base.CONFIG, protocol=PROTOCOL, execution=EXECUTION,
    warmups=0, audits=1, warmup_in_audit=True, reference_probability_nodes=7,
    max_wall_seconds=10800, automatic_extra_repetitions=False)
COMPARISONS = (*base.COMPARISONS,
    *((method, "probability_dynamic") for method in VARIANTS),
    ("expert_fast_dynamic", "expert_cost_dynamic"),
    ("expert_cache_dynamic", "expert_fast_dynamic"),
    ("expert_calibrated_dynamic", "expert_cache_dynamic"),
    ("expert_guarded_dynamic", "expert_calibrated_dynamic"))


def identity(root):
    rows = {row["path"]: row for row in matrix.identity(root)["files"]}
    name = "scripts/benchmark_qwen3_selection_refinement.py"
    rows[name] = dict(path=name, sha256=base.file_identity(root / name)["sha256"])
    files = [rows[name] for name in sorted(rows)]
    return dict(files=files, sha256=canonical_json_sha256(files))


def choice_key(method, count=None):
    return method if count is None else f"{method}:{count}"


def schedule(stage, state_id, *, smoke=False):
    if stage not in SPLITS:
        raise ValueError("unknown refinement stage")
    if smoke and stage != "calibration":
        raise ValueError("the entry smoke check uses only calibration prompts")
    node_counts = (0, 4, 7) if smoke else range(8)
    combinations = ([dict(method=m, draft_node_count=k, mode=mode, execution=EXECUTION)
                     for m in PROBES for k in node_counts for mode in CONFIG["cache_modes"]]
                    if stage == "calibration" else
                    [dict(method=m, mode=mode, execution=EXECUTION)
                     for m in REFINEMENT_METHODS for mode in CONFIG["cache_modes"]])
    rng = random.Random(f"{PROTOCOL}:480:{stage}:{state_id}")
    result = []
    for phase, repetitions in (("audit", 1), ("measurement", REPETITIONS[stage])):
        for rep in range(repetitions):
            jobs = list(combinations)
            rng.shuffle(jobs)
            for job in jobs:
                result.append(dict(job, phase=phase, repetition=rep, execution_index=len(result)))
    return result


def build_manifest(*, stage, git_commit, measurement_identity, dataset_path,
                   calibration_descriptor=None, models=None, smoke=False):
    if stage not in SPLITS or (stage == "calibration") != (models is None):
        raise ValueError("calibration and evaluation model requirements differ")
    if (stage == "calibration") != (calibration_descriptor is None):
        raise ValueError("evaluation must bind its complete calibration source")
    if smoke and stage != "calibration":
        raise ValueError("smoke checks cannot inspect development or holdout prompts")
    if len(set(q for qs in SPLITS.values() for q in qs)) != sum(map(len, SPLITS.values())):
        raise ValueError("training, development and holdout prompts overlap")
    lock = json.loads((Path(__file__).resolve().parents[2] / "workloads/gsm8k_specbench.lock.json").read_text())["dataset"]
    descriptor = base.file_identity(dataset_path)
    if descriptor != dict(basename=lock["filename"], size_bytes=lock["size_bytes"], sha256=lock["sha256"]):
        raise ValueError("workload bytes differ from the fixed dataset lock")
    examples = {q.question_id: q for q in load_gsm8k_specbench(dataset_path)}
    questions = SPLITS[stage][:1] if smoke else SPLITS[stage]
    boundaries = (0, 4) if smoke else BOUNDARIES[stage]
    states = [dict(question_id=qid, cycle_index=cycle, state_id=f"q{qid}-c{cycle}",
                   schedule=schedule(stage, f"q{qid}-c{cycle}", smoke=smoke))
              for qid in questions for cycle in boundaries]
    config = dict(CONFIG, repetitions=REPETITIONS[stage],
                  max_wall_seconds=5400 if stage == "calibration" else 10800)
    return dict(artifact_type="selection_refinement_manifest", protocol=PROTOCOL, stage=stage,
        scope="smoke" if smoke else "complete",
        git_commit=git_commit, measurement_identity=measurement_identity, config=config,
        dataset=descriptor,
        prompts=[dict(question_id=q, raw_prompt=examples[q].prompt) for q in questions],
        states=states, batches=[[s["state_id"] for s in states if s["question_id"] == q] for q in questions],
        methods=list(PROBES if stage == "calibration" else REFINEMENT_METHODS),
        calibration_source=calibration_descriptor, models=deepcopy(models),
        guard_margin=GUARD_MARGIN, reference_h2d_prefetch=False,
        split={name: list(qs) for name, qs in SPLITS.items()},
        limitations=dict(local_cycles_only=True, route_collection_cost_excluded=True,
            route_information="offline_full_pool_atlas", resident_read_is_timed=True,
            no_evaluation_acceptance_in_selection=True, no_post_holdout_tuning=True,
            previously_seen_historical_states_not_a_new_holdout=True))


def choices_for(state, *, stage, logical_cache, models=None, guard_margin=GUARD_MARGIN):
    natural_mask = resident_bitmap(residents_from_logical_state(logical_cache))
    choices = {}
    for mode in CONFIG["cache_modes"]:
        mask = 0 if mode == "cold" else natural_mask
        rows = {}
        if stage == "calibration":
            jobs = [(m, k) for m in PROBES for k in range(8)]
            legacy = cache = None
        else:
            if models is None:
                raise ValueError("evaluation has no frozen models")
            jobs = [(method, None) for method in REFINEMENT_METHODS]
            legacy = SelectionCostModel.from_dict(models["by_mode"][mode]["legacy"])
            cache = CacheCostModel.from_dict(models["by_mode"][mode]["cache"])
        for method, count in jobs:
            if method == "calibration_probability":
                ids = probability_ids(state.pool, count)
            elif method == "calibration_minimum_experts":
                ids = minimum_expert_ids(state.pool, state.atlas, count)
            else:
                ids = select_refined(state.pool, state.atlas, method, legacy_model=legacy,
                    cache_model=cache, resident_mask=mask,
                    acceptance_exponent=models["acceptance_exponent"], guard_margin=guard_margin)
            features = query_features(state.pool, state.atlas, ids, mask)
            rows[choice_key(method, count)] = dict(selected_raw_ids=list(ids), **features)
        if stage != "calibration" and rows["expert_fast_dynamic"] != rows["expert_cost_dynamic"]:
            raise ValueError("incremental search changed the original expert decision")
        choices[mode] = rows
    return choices


def training_points(shards, mode):
    points = []
    for shard in shards:
        for method in PROBES:
            for count in range(8):
                rows = [r for r in shard["records"] if r["phase"] == "measurement"
                        and r["mode"] == mode and r["method"] == method and r["draft_node_count"] == count]
                if len(rows) != REPETITIONS["calibration"]:
                    raise ValueError("incomplete calibration samples")
                features = shard["choices"][mode][choice_key(method, count)]
                points.append(dict(question_id=shard["question_id"], state_id=shard["state_id"],
                    method=method, draft_node_count=count, expert_union=features["expert_union"],
                    initial_missing=features["initial_missing"],
                    cycle_without_selector_ns=mean(r["timings"]["cycle_wall_ns"]
                        - r["timings"]["explicit_selector_ns"] for r in rows)))
    return points


def fit_models(shards):
    expected = [(qid, cycle) for qid in SPLITS["calibration"] for cycle in BOUNDARIES["calibration"]]
    if [(s["question_id"], s["cycle_index"]) for s in shards] != expected:
        raise ValueError("only the complete predetermined training cohort may fit models")
    models, diagnostics = {}, {}
    for mode in CONFIG["cache_modes"]:
        points = training_points(shards, mode)
        legacy, legacy_fit = fit_cost_model(points)
        cache, cache_fit = fit_cache_cost_model(points, mode=mode)
        folds = []
        for qid in SPLITS["calibration"]:
            training = [p for p in points if p["question_id"] != qid]
            held_out = [p for p in points if p["question_id"] == qid]
            old, _ = fit_cost_model(training)
            new, _ = fit_cache_cost_model(training, mode=mode)
            def errors(model, cached):
                return [((model.predict_ns(p["draft_node_count"], p["expert_union"], p["initial_missing"])
                          if cached else model.predict_ns(p["draft_node_count"], p["expert_union"]))
                         / p["cycle_without_selector_ns"] - 1) for p in held_out]
            old_errors, new_errors = errors(old, False), errors(new, True)
            folds.append(dict(question_id=qid,
                legacy_mean_absolute_relative_error=mean(abs(e) for e in old_errors),
                cache_mean_absolute_relative_error=mean(abs(e) for e in new_errors)))
        models[mode] = dict(legacy=legacy.to_dict(), cache=cache.to_dict())
        diagnostics[mode] = dict(legacy=legacy_fit, cache=cache_fit, held_out_prompt_folds=folds, points=points)
    observations = []
    for shard in shards:
        observations.extend(acceptance_observations(PoolRecord.read(shard["pool"]),
            shard["route_probe"]["acceptance"]["verifier_token_ids"], shard["question_id"]))
    exponent, acceptance_fit = fit_acceptance(observations)
    bundle = dict(by_mode=models, acceptance_exponent=exponent,
                  training_questions=list(SPLITS["calibration"]), guard_margin=GUARD_MARGIN)
    return bundle, dict(cost=diagnostics, acceptance=acceptance_fit)


def summarize(records):
    result = base.summarize_measurements(records, methods=REFINEMENT_METHODS,
        method_names=REFINEMENT_NAMES, comparisons=COMPARISONS)
    result["execution"] = EXECUTION
    for mode in result["by_mode"].values():
        for comparison in mode["comparisons"].values():
            comparison["paired_prompt_bootstrap"]["scope"] = "descriptive_paired_uncertainty_for_this_frozen_prompt_cohort"
    return result


def validate_history(shard):
    history, prepared = shard["reference_history"], shard["prepared_prompt"]
    source = shard["source_state"]
    count = shard["cycle_index"]
    audits = history["commit_audits"]
    if (len(audits) != count or history["reference_probability_nodes"] != 7
            or history["base_kv_length"] != source["base_kv_length"]
            or history["base_kv_length"] != len(prepared["input_ids"]) + len(history["committed_token_ids"])
            or len(history["selected_raw_ids"]) != count):
        raise ValueError("reference generation prefix or boundary changed")
    length = len(prepared["input_ids"])
    committed = 0
    for cycle, (audit, ids) in enumerate(zip(audits, history["selected_raw_ids"], strict=True)):
        flags = ("kv_prefix_exact", "kv_selected_suffix_exact", "kv_length_exact", "aux_length_exact",
                 "aux_prefix_exact", "aux_selected_suffix_exact", "pending_seed_exact")
        if (any(audit.get(flag) is not True for flag in flags) or audit["iteration"] != cycle
                or audit["base_kv_length"] != length or audit["verification_node_count"] != 8
                or audit["kv_layer_count"] != 48 or len(ids) != 7
                or audit["aux_prefix_length"] != length):
            raise ValueError("reference prefix commit audit failed")
        step = len(audit["verification_positions"])
        length += step
        committed += step
        if audit["aux_length_after"] != length:
            raise ValueError("reference prefix accepted length changed")
    if committed != len(history["committed_token_ids"]) or length != source["base_kv_length"]:
        raise ValueError("reference history does not reach the declared checkpoint")


def validate_shard(shard, manifest, directory, *, slots_per_layer=48):
    if shard.get("status") != "complete" or shard.get("manifest_sha256") != canonical_json_sha256(manifest):
        raise ValueError("incomplete or unbound refinement state")
    planned = next((s for s in manifest["states"] if s["state_id"] == shard["state_id"]), None)
    if planned is None or any(shard[key] != planned[key] for key in ("question_id", "cycle_index")):
        raise ValueError("state is outside the predetermined cohort")
    prompt = next(p for p in manifest["prompts"] if p["question_id"] == shard["question_id"])
    prepared = shard["prepared_prompt"]
    if (prepared["raw_prompt"] != prompt["raw_prompt"] or prepared["question_id"] != prompt["question_id"]
            or prepared["attention_mask"] != [1] * len(prepared["input_ids"])):
        raise ValueError("prompt token preparation changed")
    validate_history(shard)
    source = shard["source_state"]
    pool = PoolRecord.read(shard["pool"])
    probe = shard["route_probe"]
    state = SimpleNamespace(state_id=shard["state_id"], question_id=shard["question_id"],
        cycle_index=shard["cycle_index"], slots_per_layer=slots_per_layer, pool=pool,
        atlas=read_routes(probe["routes"], "refinement full atlas", pool, tuple(range(19))))
    if (probe["method"] != "route_probe" or probe["execution"] != "demand"
            or probe["selected_raw_ids"] != list(range(1, 19)) or probe["mode"] != "natural"
            or logical_state_sha256(probe["pre_logical_state"]) != source["pre_logical_state_sha256"]):
        raise ValueError("route source is not an independent full-pool demand query")
    matrix.validate_cycle(probe, state=state, source_state=source)
    base.validate_audit(probe, state, source)
    choices = choices_for(state, stage=manifest["stage"], logical_cache=probe["pre_logical_state"],
                          models=manifest["models"], guard_margin=manifest["guard_margin"])
    if not same_json_value(shard["choices"], choices):
        raise ValueError("selection used information outside the frozen models and observable inputs")
    for field in ("choice_manifest", "query_references"):
        name = shard[field]["basename"]
        if Path(name).name != name:
            raise ValueError("state metadata must be local to its artifact directory")
    frozen = base.read_artifact(directory / shard["choice_manifest"]["basename"], shard["choice_manifest"])
    if (frozen["manifest_sha256"] != shard["manifest_sha256"] or frozen["state_id"] != shard["state_id"]
            or frozen["choices"] != choices or frozen["models_sha256"] != canonical_json_sha256(manifest["models"])
            or frozen["route_probe_sha256"] != canonical_json_sha256(probe)
            or frozen["frozen_at_ns"] <= probe["cycle_wall_interval_ns"][1]):
        raise ValueError("per-state choices are not independently bound before timing")
    reference_artifact = base.read_artifact(directory / shard["query_references"]["basename"], shard["query_references"])
    if (reference_artifact["state_id"] != state.state_id
            or reference_artifact["manifest_sha256"] != shard["manifest_sha256"]
            or reference_artifact["collection_finished_ns"] <= frozen["frozen_at_ns"]):
        raise ValueError("query references have invalid source or collection order")
    references = reference_artifact["references"]
    jobs = planned["schedule"]
    expected_keys = {matrix.query_key(j["method"], choices[j["mode"]][choice_key(j["method"],
        j.get("draft_node_count"))]["selected_raw_ids"]) for j in jobs}
    if set(references) != expected_keys or len(shard["records"]) != len(jobs):
        raise ValueError("reference or scheduled measurement coverage is incomplete")
    for key, reference in references.items():
        if (matrix.query_key(reference["method"], reference["selected_raw_ids"]) != key
                or reference["execution"] != "demand" or reference["mode"] != "natural"
                or reference["cycle_wall_interval_ns"][0] < frozen["frozen_at_ns"]
                or reference["cycle_wall_interval_ns"][1] > reference_artifact["collection_finished_ns"]):
            raise ValueError("query reference did not use independent demand execution")
        matrix.validate_cycle(reference, state=state, source_state=source)
        base.validate_audit(reference, state, source)
    audits = {}
    for record, job in zip(shard["records"], jobs, strict=True):
        if any(record.get(k) != value for k, value in job.items()):
            raise ValueError("measurement order differs from the frozen schedule")
        choice = choices[job["mode"]][choice_key(job["method"], job.get("draft_node_count"))]
        if record["selected_raw_ids"] != choice["selected_raw_ids"]:
            raise ValueError("live selection did not reproduce its frozen decision")
        reference = references[matrix.query_key(job["method"], record["selected_raw_ids"])]
        if (matrix.reference_value(record) != matrix.reference_value(reference)
                or record["query_reference_sha256"] != canonical_json_sha256(reference)
                or record["choice_manifest_sha256"] != shard["choice_manifest"]["sha256"]
                or record["query_collection_finished_ns"] != reference_artifact["collection_finished_ns"]
                or record["query_collection_finished_ns"] >= record["cycle_wall_interval_ns"][0]):
            raise ValueError("exact-query output, reference identity or prediction timing changed")
        matrix.validate_cycle(record, state=state, source_state=source)
        reads_cache = record["method"] in VARIANTS[1:]
        expected_input = (canonical_json_sha256(hex(resident_bitmap(residents_from_logical_state(
            record["pre_logical_state"])))) if reads_cache else None)
        if record.get("selector_resident_sha256") != expected_input:
            raise ValueError("selector did not read the restored cache before target execution")
        key = job["mode"], choice_key(job["method"], job.get("draft_node_count"))
        if job["phase"] == "audit":
            if key in audits:
                raise ValueError("duplicate audited warmup")
            base.validate_audit(record, state, source)
            audits[key] = matrix.reference_value(record)
        elif audits.get(key) != matrix.reference_value(record):
            raise ValueError("formal measurement has no matching audited warmup")
    base.validate_checkpoint(shard["checkpoint"], config=manifest["config"])
    return shard


def validate_result(index, manifest, directory):
    base._validate_index(index, manifest)
    batch = index.get("batch")
    if type(batch) is not int or not 0 <= batch < len(manifest["batches"]):
        raise ValueError("invalid refinement batch")
    shards = list(base.read_shards(index, directory))
    if [s["state_id"] for s in shards] != manifest["batches"][batch]:
        raise ValueError("batch state coverage or order changed")
    # The environment was checked against the known devices; it fixes the expert slots of every cache state.
    for shard in shards:
        validate_shard(shard, manifest, directory, slots_per_layer=index["environment"].get("slots_per_layer", 48))
    return shards


def combine(indices, manifest, directory):
    if len(indices) != len(manifest["batches"]):
        raise ValueError("the stage is missing a planned batch")
    shards = []
    for batch, index in enumerate(indices):
        if index["batch"] != batch:
            raise ValueError("batch order or identity changed")
        shards.extend(validate_result(index, manifest, directory))
    devices = {canonical_json_sha256({k: i["environment"].get(k) for k in ("gpu_name", "server", "gpu_index", "slots_per_layer")})
               for i in indices}
    if len(devices) != 1:
        raise ValueError("a stage must measure every batch on one device")
    result = dict(artifact_type="selection_refinement_combined_result", protocol=PROTOCOL,
        stage=manifest["stage"], scope=manifest["scope"], status="complete", git_commit=manifest["git_commit"],
        manifest_sha256=canonical_json_sha256(manifest), measurement_identity=manifest["measurement_identity"],
        counts=dict(prompts=len(manifest["prompts"]), states=len(shards),
                    cycles=sum(len(s["records"]) for s in shards)),
        resources=dict(max_gpu_reserved_bytes=max(i["gpu_peak_reserved_bytes"] for i in indices),
                       max_process_rss_bytes=max(i["process_peak_rss_bytes"] for i in indices)))
    if manifest["scope"] == "smoke":
        result["smoke_validation_only"] = True
    elif manifest["stage"] == "calibration":
        result["models"], result["fit_diagnostics"] = fit_models(shards)
    else:
        result["summary"] = summarize([record for shard in shards for record in shard["records"]])
    return result
