"""Frozen local-cycle comparison and independent predicted-cache event audit."""
from __future__ import annotations

from copy import deepcopy
import random
from statistics import mean

from .f0_direct import expert_union
from .f0_direct_analysis import (COUNTERS, EXPERT_BYTES, cache_totals, require_ns,
                                 validate_outcome, validate_trial)
from .m1_capture_replay import logical_state_sha256, read_logical_state, read_routes
from .selection_evaluation import validate_audit

PROTOCOL = "cross-layer-prefetch-local-cycles-v1"
STATE_IDS = ("q416-c37", "q466-c77", "q426-c16")
VARIANTS = ("demand", "one_ahead", "ready_demand", "predicted_same_layer",
            "predicted_cross_layer", "wrong_cross_layer")


def schedule(state_id):
    rng = random.Random("cross-layer-480:" + state_id)
    rows = []
    for phase, repetitions in (("warmup", 1), ("audit", 1), ("measurement", 3)):
        for repetition in range(repetitions):
            # Wrong predictions are a correctness audit, not a timed contender.
            variants = VARIANTS if phase == "audit" else VARIANTS[:-1]
            pairs = [(v, mode) for v in variants for mode in ("cold", "natural")]
            rng.shuffle(pairs)
            for variant, mode in pairs:
                rows.append(dict(variant=variant, mode=mode, phase=phase,
                    repetition=repetition, execution_index=len(rows)))
    return rows


def predictions_for(reference, variant):
    actual = {layer: set() for layer in range(48)}
    for row in reference["routes"]["rows"]:
        actual[row["layer_id"]].update(row["expert_ids"])
    if variant == "ready_demand":
        return {}
    if variant == "wrong_cross_layer":
        return {layer: [e for e in range(128) if e not in ids][:len(ids)]
                for layer, ids in actual.items()}
    return {layer: sorted(ids) for layer, ids in actual.items()}


class CompactTransitions:
    """Append supplied host decisions only; identical observer on every variant."""

    def __init__(self):
        self.events = []

    def record(self, *, kind, expert_id, slot_id, generation,
               submission_sequence=None, victim_expert_id=None, **unused):
        self.events.append((kind.value, expert_id.layer_id, expert_id.expert_id,
            slot_id, generation, submission_sequence,
            None if victim_expert_id is None else victim_expert_id.expert_id))


def replay_events(pre, events, actual, variant):
    """Rebuild assignments, generations, leases, LRU and counters from host events.

    Completion timing and GPU read safety are tested separately. Events describe
    actual decisions; this checks their consistency, not optimal victim choice.
    The baseline one-ahead path acquires its first expert normally, then defers
    touches for later experts (the experiment fixes 48 slots, no external leases).
    """
    if variant not in VARIANTS:
        raise ValueError("unknown runtime variant")
    post = deepcopy(pre)
    consumed, pending_evictions = set(), {}
    accurate_layers = set()
    if variant in ("predicted_same_layer", "predicted_cross_layer"):
        accurate_layers = {lid for lid, layer in enumerate(post["layers"])
                           if sum(key[0] == lid for key in actual) <= len(layer["slots"]) - 1}
    defer = {}
    first = set()
    counter_names = dict(h2d_submit="h2d_submissions", resident_hit="resident_hits",
                         inflight_join="inflight_joins", evict="evictions")
    def touch(layer, slot):
        layer["lru_sequence"] += 1
        slot["last_used_sequence"] = layer["lru_sequence"]
    for event in events:
        if len(event) != 7:
            raise ValueError("invalid cache event shape")
        kind, lid, expert, sid, generation, submission, victim = event
        if (any(type(v) is not int or v < 0 for v in (lid, expert, sid, generation))
                or not lid < len(post["layers"]) or not expert < 128):
            raise ValueError("invalid event identity")
        layer = post["layers"][lid]
        if not sid < len(layer["slots"]):
            raise ValueError("event slot outside cache")
        slot = layer["slots"][sid]
        key, location = (lid, expert), (lid, sid)
        if kind == "evict":
            if lid in accurate_layers and (lid, victim) in actual and (lid, victim) not in consumed:
                raise ValueError("accurate prefetch evicted an unconsumed actual expert")
            if (slot["active_leases"] or slot["expert_id"] != victim or victim is None
                    or generation != slot["generation"] + 1 or location in pending_evictions):
                raise ValueError("invalid eviction or leased victim")
            pending_evictions[location] = (expert, generation)
        elif kind == "h2d_submit":
            if (slot["active_leases"] or generation != slot["generation"] + 1
                    or submission != layer["submission_sequence"]
                    or any(s["expert_id"] == expert for s in layer["slots"])
                    or (slot["expert_id"] is not None and
                        pending_evictions.pop(location, None) != (expert, generation))):
                raise ValueError("invalid copy, generation, sequence or protected slot")
            slot.update(expert_id=expert, generation=generation, ready=True, pending_use_guards=0)
            layer["submission_sequence"] += 1
        elif kind not in ("resident_hit", "inflight_join", "consume_wait", "release"):
            raise ValueError("unknown cache event")
        if kind != "evict" and (slot["expert_id"] != expert or slot["generation"] != generation):
            raise ValueError("event uses a stale slot generation")
        if kind in ("h2d_submit", "resident_hit", "inflight_join"):
            if slot["active_leases"]:
                raise ValueError("experiment acquired the same expert twice concurrently")
            slot["active_leases"] += 1
            deferred = variant != "demand" and (variant != "one_ahead" or lid in first)
            defer[location] = deferred
            first.add(lid)
            if not deferred:
                touch(layer, slot)
        elif kind == "consume_wait":
            if not slot["active_leases"] or key not in actual or key in consumed:
                raise ValueError("unleased, duplicate or non-router expert consumption")
            consumed.add(key)
            if defer.pop(location, False):
                touch(layer, slot)
        elif kind == "release":
            if slot["active_leases"] != 1:
                raise ValueError("release without a live lease")
            slot["active_leases"] -= 1
            defer.pop(location, None)
            touch(layer, slot)
        if kind in counter_names:
            layer["counters"][counter_names[kind]] += 1
    if (consumed != actual or pending_evictions
            or any(s["active_leases"] for layer in post["layers"] for s in layer["slots"])):
        raise ValueError("incomplete actual computation, copy or lease release")
    return post


def validate_record(record, *, state, source_state, reference):
    selected = tuple(reference["selected_raw_ids"])
    if (record["selected_raw_ids"] != list(selected) or record["state_id"] != state.state_id
            or record["question_id"] != state.question_id
            or record["preparation_method"] != "checkpoint_restore"
            or record["method"] != "expert_cost_dynamic"
            or record["pool_fingerprint"] != state.pool.fingerprint
            or record["routes"] != reference["routes"]
            or record["acceptance"] != reference["acceptance"]):
        raise ValueError("fixed selection, own routes or acceptance changed")
    pre, _ = read_logical_state(record["pre_logical_state"], "prefetch pre")
    if record["reference_cache_sha256"] != source_state["pre_logical_state_sha256"]:
        raise ValueError("reference checkpoint binding changed")
    if record["mode"] == "natural":
        if logical_state_sha256(pre) != record["reference_cache_sha256"]:
            raise ValueError("natural cache restoration changed")
    elif record["mode"] == "cold":
        if any(s["expert_id"] is not None for l in pre["layers"] for s in l["slots"]) or any(
                v for l in pre["layers"] for v in l["counters"].values()):
            raise ValueError("cold cache is not empty")
    else:
        raise ValueError("unknown cache mode")
    routes = read_routes(record["routes"], "own routes", state.pool, (0, *selected))
    own = expert_union(routes, selected)
    actual = {(lid, e) for lid, ids in enumerate(own) for e in ids}
    post = replay_events(pre, record["cache_events"], actual, record["variant"])
    counters = {k: cache_totals(post)[k] - cache_totals(pre)[k] for k in COUNTERS}
    if (record["post_logical_state_sha256"] != logical_state_sha256(post)
            or record["cache_counter_delta"] != counters
            or record["logical_h2d_bytes"] != counters["h2d_submissions"] * EXPERT_BYTES
            or record["F_own"] != len(actual)
            or record["max_own_layer_union"] != max(map(len, own))):
        raise ValueError("independent event replay differs from final cache state or counters")
    validate_outcome(record, state.pool, base_length=source_state["base_kv_length"], selected=selected)
    if record["variant"] in VARIANTS[:2]:
        validate_trial(record, job=dict(job_id=record["job_id"], selected_raw_ids=list(selected)),
            state=state, source_state=source_state, variable_budget=True)
    else:
        stats = record["prefetch_stats"]
        if (stats["complete"] is not True or stats["peak_inflight"] > 2
                or stats["max_inflight"] != 2 or stats["prefetch_slot_limits"] != [47] * 48
                or stats["lookahead_layers"] != int(record["variant"] in VARIANTS[4:])
                or sum(s["computed_experts"] for s in stats["layers"]) != len(actual)):
            raise ValueError("scheduler completion, copy budget or computation count failed")
    timing = record["timings"]
    for name, value in timing.items():
        require_ns(value, name, allow_zero=name == "explicit_selector_ns")
    interval = record["cycle_wall_interval_ns"]
    if (len(interval) != 2 or interval[1] - interval[0] != timing["cycle_wall_ns"]
            or timing["cycle_wall_ns"] <= timing["explicit_selector_ns"]
            or record["draft_forward_calls"] <= 0 or record["gc_enabled_during_cycle"]):
        raise ValueError("cycle timing, draft work or GC scope changed")
    digest = record["logits_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("missing full-logit digest")
    if record["phase"] == "audit":
        validate_audit(record, state, source_state)


def summarize(records):
    result = {}
    for mode in ("cold", "natural"):
        variants = {}
        for variant in VARIANTS[:-1]:
            rows = [r for r in records if r["mode"] == mode and r["variant"] == variant
                    and r["phase"] == "measurement"]
            variants[variant] = dict(repetitions=len(rows),
                tokens_per_second=sum(len(r["acceptance"]["committed_token_ids"]) for r in rows) * 1e9 /
                    sum(r["timings"]["cycle_wall_ns"] for r in rows),
                mean_cycle_ms=mean(r["timings"]["cycle_wall_ns"] / 1e6 for r in rows),
                mean_target_cuda_ms=mean(r["timings"]["target_verification_cuda_ns"] / 1e6 for r in rows),
                mean_h2d_submissions=mean(r["cache_counter_delta"]["h2d_submissions"] for r in rows),
                states={sid: dict(cycle_ms=[r["timings"]["cycle_wall_ns"] / 1e6 for r in rows if r["state_id"] == sid],
                    target_cuda_ms=[r["timings"]["target_verification_cuda_ns"] / 1e6 for r in rows if r["state_id"] == sid],
                    h2d_submissions=[r["cache_counter_delta"]["h2d_submissions"] for r in rows if r["state_id"] == sid])
                    for sid in STATE_IDS})
        for row in variants.values():
            row["gain_vs_one_ahead"] = row["tokens_per_second"] / variants["one_ahead"]["tokens_per_second"] - 1
        result[mode] = variants
    return result
