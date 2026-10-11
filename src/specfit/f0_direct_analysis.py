from __future__ import annotations

from statistics import median

from .experiment import canonical_json_sha256
from .f0_direct import DirectSources, expert_union, PROTOCOL, LEGACY_PROTOCOL, CONFIG, PACKED_SOURCE
from .m1_capture_replay import (advance_logical_state, logical_state_sha256,
                               read_logical_state, read_routes, same_json_value)
from .m1_capture_validation import M1_EOS_TOKEN_IDS
from .r6a_identity import R6A_MODEL_IDENTITIES
from .r6a_tb2_contract import R6A_TB2_IMAGE_ID, R6A_TB2_NUMERIC_SETTINGS

COUNTERS = ("h2d_submissions", "resident_hits", "inflight_joins", "evictions")
EXPERT_BYTES = 9437184


def require_ns(value: object, name: str, *, allow_zero: bool = False) -> int:
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be a valid integer number of nanoseconds")
    return value


def cache_totals(state: dict) -> dict:
    return {name: sum(layer["counters"][name] for layer in state["layers"]) for name in COUNTERS}


def validate_outcome(record: dict, pool, *, base_length: int, selected: tuple[int, ...]) -> None:
    outcome = record["acceptance"]
    verifier = outcome["verifier_token_ids"]
    committed = outcome["committed_token_ids"]
    accepted = outcome["accepted_raw_ids"]
    if any(type(x) is not int or x < 0 for xs in (verifier, committed, accepted) for x in xs):
        raise ValueError("F0 acceptance token/raw ID type invalid")
    if len(verifier) != 1 + len(selected) or not committed:
        raise ValueError("F0 verification/commit length is invalid")
    by_id = {n.raw_id: n for n in pool.nodes}
    positions = (0, *selected)
    path = []
    parent = 0
    while True:
        children = [i for i in selected if by_id[i].parent_raw_id == parent
                    and by_id[i].target_token_id == verifier[positions.index(parent)]]
        if not children:
            break
        if len(children) != 1:
            raise ValueError("F0 target matches multiple siblings")
        parent = children[0]
        path.append(parent)
    tokens = [pool.seed_token_id, *(by_id[i].target_token_id for i in path)]
    for i, token in enumerate(tokens):
        if token in M1_EOS_TOKEN_IDS:
            tokens = tokens[:i + 1]
            break
    tokens = tokens[:512 - base_length]
    path = path[:len(tokens) - 1]
    expected = {"verifier_token_ids": verifier, "committed_token_ids": tokens,
        "accepted_raw_ids": path, "accepted_draft_tokens": len(path),
        "next_pending_seed_token_id": verifier[positions.index(path[-1] if path else 0)],
        "kv_length_after": base_length + len(tokens)}
    if not same_json_value(outcome, expected):
        raise ValueError("F0 acceptance/commit does not reproduce target verification semantics")


def validate_trial(record: dict, *, job: dict, state, source_state: dict,
                   variable_budget: bool = False, max_draft_nodes: int = 7) -> dict:
    selected = tuple(job["selected_raw_ids"])
    by_id = {n.raw_id: n for n in state.pool.nodes}
    if (len(selected) > max_draft_nodes or (not variable_budget and len(selected) != 7)
            or len(set(selected)) != len(selected)
            or any(type(i) is not int or i not in by_id for i in selected)
            or any(by_id[i].parent_raw_id not in (0, *selected) for i in selected)):
        raise ValueError("trial selection violates its node budget or ancestor closure")
    if (record["job_id"] != job["job_id"] or record["state_id"] != state.state_id
            or record["selected_raw_ids"] != list(selected)
            or record["pool_fingerprint"] != state.pool.fingerprint):
        raise ValueError("F0 trial inconsistent with frozen selection")
    mode = record["mode"]
    if mode not in ("cold", "natural"):
        raise ValueError("F0 direct only accepts cold/natural")
    pre, snapshot = read_logical_state(record["pre_logical_state"], "F0 pre-state",
                                       expected_capacity=getattr(state, "slots_per_layer", 48))
    if record["reference_cache_sha256"] != source_state["pre_logical_state_sha256"]:
        raise ValueError("F0 reference cache binding is inconsistent")
    if mode == "natural":
        if logical_state_sha256(pre) != source_state["pre_logical_state_sha256"]:
            raise ValueError("F0 natural did not restore the complete reference cache")
    elif any(slot["expert_id"] is not None for layer in pre["layers"] for slot in layer["slots"]) or any(
            value for layer in pre["layers"] for value in layer["counters"].values()):
        raise ValueError("F0 cold cache start point is not empty")
    routes = read_routes(record["routes"], "F0 own routes", state.pool, (0, *selected))
    own = expert_union(routes, selected)
    actual_f = sum(map(len, own))
    expected_post, _ = advance_logical_state(pre, snapshot, routes)
    expected_counts = {name: cache_totals(expected_post)[name] - cache_totals(pre)[name]
                       for name in COUNTERS}
    if (record["post_logical_state_sha256"] != logical_state_sha256(expected_post)
            or not same_json_value(record["cache_counter_delta"], expected_counts)
            or sum(expected_counts[k] for k in COUNTERS[:3]) != actual_f
            or record["logical_h2d_bytes"] != expected_counts["h2d_submissions"] * EXPERT_BYTES
            or record["F_own"] != actual_f or record["max_own_layer_union"] != max(map(len, own))):
        raise ValueError("F0 own-route/cache transition/H2D conservation failed")
    validate_outcome(record, state.pool, base_length=source_state["base_kv_length"], selected=selected)
    timing = record["timings"]
    wall = require_ns(timing["cycle_wall_ns"], "cycle wall")
    selector = require_ns(timing["explicit_selector_ns"], "explicit selector", allow_zero=True)
    target_only = variable_budget and record.get("method") == "target_only"
    if target_only and (selected or selector or record.get("draft_forward_calls") != 0):
        raise ValueError("target-only trials cannot execute draft or selection work")
    require_ns(timing["proposal_wall_ns"], "proposal", allow_zero=target_only)
    if target_only and timing["proposal_wall_ns"] != 0:
        raise ValueError("target-only proposal time must be zero")
    require_ns(timing["target_verification_cuda_ns"], "verification CUDA")
    prepare_wall = require_ns(record["prepare_wall_ns"], "prepare wall")
    if "prepare_timings_ns" in record:
        parts = record["prepare_timings_ns"]
        if not isinstance(parts, dict) or set(parts) != {
            "initialization_and_prefill", "reference_history", "boundary_validation_and_cache_setup"
        }:
            raise ValueError("F0 prepare sub-fields inconsistent")
        if sum(require_ns(value, f"prepare {name}", allow_zero=True)
               for name, value in parts.items()) != prepare_wall:
            raise ValueError("F0 prepare sub-items do not match total duration")
    if wall <= selector:
        raise ValueError("F0 cycle wall time is insufficient to contain selection")
    return {"F_own": actual_f, "max_own_layer_union": max(map(len, own))}


def trial_signature(record: dict) -> str:
    return canonical_json_sha256({key: record[key] for key in (
        "pool_fingerprint", "selected_raw_ids", "routes", "acceptance",
        "post_logical_state_sha256", "cache_counter_delta")})


def _policy_summary(rows: list[dict]) -> dict:
    tokens = sum(r["committed_tokens"] for r in rows)
    ns = sum(r["reported_cycle_ns"] for r in rows)
    return {"committed_tokens": tokens, "cycle_ns": ns,
            "committed_tokens_per_second": tokens * 1e9 / ns,
            "h2d_submissions": sum(r["h2d_submissions"] for r in rows),
            "logical_h2d_bytes": sum(r["logical_h2d_bytes"] for r in rows)}


def summarize_rows(rows: list[dict], *, pilot_all_same: bool, all_difference_count: int) -> dict:
    by_mode = {}
    for mode in ("cold", "natural"):
        subset = [r for r in rows if r["mode"] == mode]
        if not subset:
            raise ValueError("F0 is missing cache settings")
        totals = {p: _policy_summary([r for r in subset if r["policy"] == p]) for p in ("p7", "f0")}
        gain = totals["f0"]["committed_tokens_per_second"] / totals["p7"]["committed_tokens_per_second"] - 1
        prompts = {}
        for qid in sorted({r["question_id"] for r in subset}):
            t = {p: _policy_summary([r for r in subset if r["question_id"] == qid and r["policy"] == p])
                 for p in ("p7", "f0")}
            prompts[str(qid)] = t["f0"]["committed_tokens_per_second"] / t["p7"]["committed_tokens_per_second"] - 1
        by_mode[mode] = {"policies": totals, "gain_vs_p7": gain, "prompt_gains": prompts,
            "positive_prompts": sum(g > 0 for g in prompts.values()),
            "negative_prompts": sum(g < 0 for g in prompts.values()),
            "predictor_total_budget_ns": totals["f0"]["committed_tokens"] / totals["p7"]["committed_tokens"]
                * totals["p7"]["cycle_ns"] - totals["f0"]["cycle_ns"]}
    natural = by_mode["natural"]
    if pilot_all_same and all_difference_count:
        decision = "diagnose_remaining_states"
    elif natural["gain_vs_p7"] > 0 and natural["positive_prompts"] >= 6:
        decision = "enter_d3"
    elif natural["negative_prompts"] >= 6:
        decision = "stop_current_configuration"
    else:
        decision = "one_bounded_diagnostic"
    return {"by_mode": by_mode, "decision": decision, "formal_5_percent_gate_evaluated": False,
            "h2d_attribution": "not_measured_preloaded_deferred"}


def analyze_result(result: dict, manifest: dict, sources: DirectSources) -> dict:
    if any(record.get("preparation_method", "history_replay") != "history_replay"
           for phase in ("warmups", "audits", "trials") for record in result.get(phase, ())):
        raise ValueError("checkpoint restoration measurements cannot enter the F0 pilot")
    if result["manifest_sha256"] != canonical_json_sha256(manifest):
        raise ValueError("F0 result manifest binding inconsistent")
    environment = result["environment"]
    if (result.get("artifact_type") != "f0_direct_result"
            or result.get("protocol") not in (LEGACY_PROTOCOL, PROTOCOL)
            or result.get("protocol") != manifest.get("protocol")
            or result.get("status") != "complete" or not same_json_value(result.get("config"), CONFIG)
            or environment.get("git_commit") != manifest["git_commit"]
            or environment.get("image_id") != R6A_TB2_IMAGE_ID
            or environment.get("gpu_name") != "NVIDIA A100-PCIE-40GB"
            or environment.get("server") != "a100-host" or environment.get("device") != "cuda:0"
            or not same_json_value(environment.get("numeric_settings"), R6A_TB2_NUMERIC_SETTINGS)
            or environment.get("packed_descriptor") != PACKED_SOURCE
            or any(environment.get("models", {}).get(k) != v for k, v in R6A_MODEL_IDENTITIES.items())):
        raise ValueError("F0 result fixed environment/config inconsistent")
    states = {s.state_id: s for s in sources.projection.states}
    raw_states = {s["state_id"]: s for s in sources.capture["states"]}
    decisions = {s["state_id"]: s for s in manifest["states"]}
    jobs = {j["job_id"]: j for j in manifest["jobs"]}
    expected_pairs = {(j, mode) for j in jobs for mode in ("cold", "natural")}
    references = {}
    for phase in ("warmups", "audits"):
        records = result[phase]
        if len(records) != len(expected_pairs):
            raise ValueError(f"F0 {phase} coverage is insufficient or duplicated")
        seen = set()
        for record in records:
            key = record["job_id"], record["mode"]
            if key in seen or key not in expected_pairs:
                raise ValueError("F0 warmup/audit identity duplicated or unknown")
            seen.add(key)
            job = jobs[key[0]]
            state = states[job["state_id"]]
            validate_trial(record, job=job, state=state, source_state=raw_states[state.state_id])
            if phase == "warmups":
                references[key] = trial_signature(record)
            else:
                if trial_signature(record) != references[key]:
                    raise ValueError("F0 audit does not reproduce warmup routes/cache/commit")
                audit = record["commit_audit"]
                flags = ("kv_prefix_exact", "kv_selected_suffix_exact", "kv_length_exact", "aux_length_exact",
                         "aux_prefix_exact", "aux_selected_suffix_exact", "pending_seed_exact")
                positions = [0, *(job["selected_raw_ids"].index(raw) + 1
                    for raw in record["acceptance"]["accepted_raw_ids"])]
                if (any(audit.get(k) is not True for k in flags) or audit["kv_layer_count"] != 48
                        or audit["iteration"] != state.cycle_index
                        or audit["base_kv_length"] != raw_states[state.state_id]["base_kv_length"]
                        or audit["verification_node_count"] != 8
                        or list(audit["verification_positions"]) != positions
                        or list(audit["rejected_node_ids"]) != [i for i in range(8) if i not in positions]
                        or audit["aux_prefix_length"] != audit["base_kv_length"]
                        or audit["aux_length_after"] != record["acceptance"]["kv_length_after"]):
                    raise ValueError("F0 KV/aux audit failed")
    selector_times = {}
    measurements = result["selector_measurements"]
    expected_selector = {(sid, p) for sid in manifest["pilot_state_ids"] for p in ("p7", "f0")}
    if len(measurements) != len(expected_selector):
        raise ValueError("F0 selector sample coverage insufficient")
    for item in measurements:
        key = item["state_id"], item["policy"]
        if key not in expected_selector or key in selector_times or len(item["samples_ns"]) != 6:
            raise ValueError("F0 selector measurement duplicated or sample count mismatched")
        if item["selected_raw_ids"] != decisions[key[0]]["decisions"][key[1]]["selected_raw_ids"]:
            raise ValueError("F0 selector measurement did not reproduce the frozen selection")
        selector_times[key] = median(require_ns(t, "selector") for t in item["samples_ns"])
    trials = result["trials"]
    schedule = manifest["schedule"]["paired"]
    if len(trials) != len(schedule):
        raise ValueError("F0 formal trial coverage is insufficient")
    rows = []
    for index, (record, scheduled) in enumerate(zip(trials, schedule, strict=True)):
        if any(record.get(k) != v for k, v in scheduled.items()) or record["execution_index"] != index:
            raise ValueError("F0 formal run does not match frozen order")
        key = record["job_id"], record["mode"]
        job = jobs[key[0]]
        state = states[job["state_id"]]
        validate_trial(record, job=job, state=state, source_state=raw_states[state.state_id])
        if trial_signature(record) != references[key]:
            raise ValueError("F0 formal run route/cache/commit must not repeat")
        for policy in job["policies"]:
            selector_ns = selector_times[state.state_id, policy]
            decision = decisions[state.state_id]["decisions"][policy]
            timing = record["timings"]
            reported = timing["cycle_wall_ns"] - timing["explicit_selector_ns"] + selector_ns
            rows.append({"state_id": state.state_id, "question_id": state.question_id,
                "policy": policy, "mode": record["mode"], "repetition": record["repetition"],
                "committed_tokens": len(record["acceptance"]["committed_token_ids"]),
                "reported_cycle_ns": reported, "selector_ns": selector_ns,
                "explicit_cycle_ns": timing["cycle_wall_ns"],
                "explicit_selector_ns": timing["explicit_selector_ns"],
                "F_atlas": decision["F_atlas"], "F_own": record["F_own"],
                "expected_committed_tokens": decision["expected_committed_tokens"],
                "h2d_submissions": record["cache_counter_delta"]["h2d_submissions"],
                "logical_h2d_bytes": record["logical_h2d_bytes"]})
    summary = summarize_rows(rows, pilot_all_same=all(len(j["policies"]) == 2 for j in jobs.values()),
                             all_difference_count=manifest["gate"]["decision_difference_count"])
    state_summaries = []
    for sid in manifest["pilot_state_ids"]:
        for mode in ("cold", "natural"):
            policies = {}
            for policy in ("p7", "f0"):
                items = [r for r in rows if (r["state_id"], r["mode"], r["policy"]) == (sid, mode, policy)]
                times = [r["reported_cycle_ns"] for r in items]
                policies[policy] = {**_policy_summary(items), "cycle_samples_ns": times,
                    "cycle_min_ns": min(times), "cycle_max_ns": max(times),
                    "mean_committed_tokens": sum(r["committed_tokens"] for r in items) / len(items),
                    "F_atlas": items[0]["F_atlas"], "F_own": items[0]["F_own"],
                    "expected_committed_tokens": items[0]["expected_committed_tokens"]}
            state_summaries.append({"state_id": sid, "mode": mode, "policies": policies,
                "actual_delta_committed_tokens": policies["f0"]["mean_committed_tokens"] - policies["p7"]["mean_committed_tokens"],
                "predicted_delta_committed_tokens": policies["f0"]["expected_committed_tokens"] - policies["p7"]["expected_committed_tokens"],
                "mean_delta_cycle_ns": (policies["f0"]["cycle_ns"] - policies["p7"]["cycle_ns"]) / 3,
                "predicted_delta_cycle_ns": (policies["f0"]["F_atlas"] - policies["p7"]["F_atlas"]) *
                    (manifest["coefficients"]["preloaded_ns_per_expert"] + manifest["coefficients"]["h2d_ns_per_expert"])})
    return {"method_rows": rows, "state_summaries": state_summaries, **summary}
