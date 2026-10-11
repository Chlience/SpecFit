
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from .experiment import canonical_json_sha256
from .replay import LogicalReplayResult, replay_qwen3_chain
from .trace import (
    CacheSnapshotTrace,
    RuntimeEventKind,
    SnapshotPhase,
    TraceBundle,
)

R6A_TB2_ARTIFACT_TYPE = "r6a_tb2_capacity_replay"
R6A_TB2_POLICIES = (
    "probability",
    "unique_efficiency",
    "joint_hindsight",
)
R6A_TB2_SOURCE_SHA256 = {
    "e019": "f2995fb55130c000cd384d865405d353b67198cfd2b2be3d2038a5f325f50167",
    "e020": "0358fe8e0f2090030de69149a2dd3aaba0388315e8ec1dd5bc80f874fb860acd",
}
R6A_TB2_POOL_FINGERPRINT = (
    "f701e7469c957dd35a1e3ff60d2fe38743dd6a6b626679d0e005f6c59de364ff"
)
R6A_TB2_FROZEN_FINGERPRINT = (
    "d1df19a66f0266e253fb08cd73654c200d5f3a583393640d2fe9947f3e68a22e"
)
R6A_TB2_EXPECTED = {
    "probability": {
        "selected": (1, 3, 4, 7, 9, 11, 15),
        "route_digest": "346a08c73a327defcd522523975313d9f5701ef9442b466fb8f1a8c1aeaef191",
        "counters": (1219, 631, 588, 0, 631),
        "accepted": (1, 4),
        "committed": (1654, 525, 2661),
        "pending": 429,
        "post": "5c69b7c4ba24053050908ca95337d9c28e2d6b8f7a00aefd83a3938ee0a50243",
        "initial_resident_reload_h2d": 155,
    },
    "unique_efficiency": {
        "selected": (1, 3, 4, 7, 9, 11, 13),
        "route_digest": "43e019886c2eacf03f0b422b92ae5280c96f43de573e7ebf3d8a92bab0fa80c0",
        "counters": (1024, 514, 510, 0, 514),
        "accepted": (1, 4),
        "committed": (1654, 525, 2661),
        "pending": 429,
        "post": "22efbecceb1633ce17c0e71ce3a95315ac76d5faa46df33366f0c201604e8f62",
        "initial_resident_reload_h2d": 106,
    },
    "joint_hindsight": {
        "selected": (1, 2, 3, 4, 5, 6, 10),
        "route_digest": "720864d13331fb5e51ad0e57968303b778e850b6d32ac5f2793767e5e5e2b106",
        "counters": (869, 427, 442, 0, 427),
        "accepted": (1, 4, 10),
        "committed": (1654, 525, 2661, 429),
        "pending": 1447,
        "post": "f9354df28cd29ab9f5d3e357440dfe284c7247ae5f9ec2a27b99c5c16159dba6",
        "initial_resident_reload_h2d": 80,
    },
}

_REPLAY_CONTRACT = {
    "input": "before_snapshot_and_own_target_routes",
    "execution_order": "layer_ascending_then_unique_expert_id_ascending",
    "lru_touch": "after_acquire_and_after_release",
    "oracle": "cycle_scoped_runtime_events_and_after_snapshot",
    "inflight_join_supported": False,
}


def _snapshot(trace: TraceBundle, phase: SnapshotPhase) -> CacheSnapshotTrace:
    matches = tuple(item for item in trace.snapshots if item.phase is phase)
    if len(matches) != 1:
        raise ValueError(f"TB2 trace must contain exactly one {phase.value} snapshot")
    return matches[0]


def _normalized_state(snapshot: CacheSnapshotTrace) -> tuple[object, ...]:

    return tuple(
        (
            layer.layer_id,
            layer.capacity,
            layer.lru_sequence,
            layer.submission_sequence,
            (
                layer.stats.h2d_submissions,
                layer.stats.resident_hits,
                layer.stats.inflight_joins,
                layer.stats.evictions,
                layer.stats.pending_use_guards,
            ),
            tuple(
                (
                    slot.slot_id,
                    slot.expert_id,
                    slot.generation,
                    slot.active_leases,
                    slot.last_used_sequence,
                    slot.ready,
                    slot.pending_use_guards,
                )
                for slot in layer.slots
            ),
        )
        for layer in snapshot.layers
    )


def trace_cache_state_fingerprint(snapshot: CacheSnapshotTrace) -> str:

    if any(
        layer.cache_id is None
        or layer.host_state_version is None
        or layer.lru_sequence is None
        or layer.stats is None
        for layer in snapshot.layers
    ):
        raise ValueError("state fingerprint requires fully expanded snapshot fields")
    layers = tuple(
        {
            "layer_id": layer.layer_id,
            "cache_id": layer.cache_id,
            "queue_id": layer.queue_id,
            "capacity": layer.capacity,
            "host_state_version": layer.host_state_version,
            "lru_sequence": layer.lru_sequence,
            "submission_sequence": layer.submission_sequence,
            "slots": [
                (
                    slot.slot_id, slot.expert_id, slot.generation,
                    slot.active_leases, slot.last_used_sequence, slot.ready,
                    slot.pending_use_guards,
                )
                for slot in layer.slots
            ],
            "stats": (
                layer.stats.h2d_submissions, layer.stats.resident_hits,
                layer.stats.inflight_joins, layer.stats.evictions,
                layer.stats.pending_use_guards,
            ),
            "pending": [
                (
                    item.expert_id, item.slot_id, item.generation,
                    item.submission_sequence, item.ready_event_completed,
                )
                for item in layer.pending_transfers
            ],
        }
        for layer in snapshot.layers
    )
    encoded = json.dumps(
        (snapshot.runtime_state_version, layers),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _event_projection(events: Sequence[object]) -> tuple[object, ...]:
    return tuple(
        (
            getattr(event, "kind").value,
            getattr(event, "layer_id"),
            getattr(event, "expert_id"),
            getattr(event, "slot_id"),
            getattr(event, "generation"),
            getattr(event, "submission_sequence"),
            getattr(event, "victim_expert_id"),
        )
        for event in events
    )


def _replay_input(trace: TraceBundle, pre: CacheSnapshotTrace) -> object:
    tree = trace.trees[0]
    verified = tuple(node.node_id for node in tree.nodes if node.verified)
    routes = tuple(
        (route.node_id, route.layer_id, route.expert_ids)
        for route in trace.target_routes
        if route.node_id in verified
    )
    return (_REPLAY_CONTRACT["execution_order"], _normalized_state(pre), routes)


def build_r6a_tb2_replay_evidence(trace: TraceBundle) -> dict[str, object]:

    if len(trace.cycles) != 1 or len(trace.trees) != 1:
        raise ValueError("TB2-A only accepts single-cycle, single-tree trace")
    result: LogicalReplayResult = replay_qwen3_chain(trace)
    pre = _snapshot(trace, SnapshotPhase.BEFORE_VERIFY)
    post = _snapshot(trace, SnapshotPhase.AFTER_VERIFY)
    if len(pre.layers) != 48 or len(post.layers) != 48:
        raise ValueError("TB2-A requires a full 48-layer snapshot")
    if any(layer.capacity != 48 or len(layer.slots) != 48 for layer in pre.layers):
        raise ValueError("TB2-A requires 48 full slots per layer")
    scoped = tuple(
        event for event in trace.runtime_events if event.cycle_id is not None
    )
    if result.counters.inflight_joins:
        raise ValueError("TB2-A does not support in-flight join within verification")
    initial_resident = {
        (layer.layer_id, slot.expert_id)
        for layer in pre.layers
        for slot in layer.slots
        if slot.expert_id is not None
    }
    reloaded = sum(
        event.kind is RuntimeEventKind.H2D_SUBMIT
        and (event.layer_id, event.expert_id) in initial_resident
        for event in result.cycles[0].events
    )
    counters = result.counters
    return {
        "replay_input_sha256": canonical_json_sha256(_replay_input(trace, pre)),
        "pre_logical_state_sha256": canonical_json_sha256(_normalized_state(pre)),
        "post_logical_state_sha256": canonical_json_sha256(_normalized_state(post)),
        "logical_event_sha256": canonical_json_sha256(
            _event_projection(result.cycles[0].events)
        ),
        "observed_event_sha256": canonical_json_sha256(_event_projection(scoped)),
        "h2d_submission_order_sha256": canonical_json_sha256(tuple(
            item for item in _event_projection(result.cycles[0].events)
            if item[0] == RuntimeEventKind.H2D_SUBMIT.value
        )),
        "lru_victim_order_sha256": canonical_json_sha256(tuple(
            item for item in _event_projection(result.cycles[0].events)
            if item[0] == RuntimeEventKind.EVICT.value
        )),
        "final_slots_sha256": canonical_json_sha256(tuple(
            (slot.layer_id, slot.slot_id, slot.expert_id, slot.generation,
             slot.last_used_sequence)
            for slot in result.final_slots
        )),
        "counters": {
            "unique_expert_acquisitions": counters.acquisitions,
            "h2d_submissions": counters.h2d_submissions,
            "resident_hits": counters.resident_hits,
            "inflight_joins": counters.inflight_joins,
            "evictions": counters.evictions,
            "consume_waits": counters.consume_waits,
            "releases": counters.releases,
        },
        "initial_resident_reload_h2d": reloaded,
        "event_count": counters.event_count,
        "victim_count": counters.evictions,
        "initial_occupancy": result.initial_occupancy,
        "final_occupancy": result.final_occupancy,
    }


def replay_contract_record() -> dict[str, object]:
    return dict(_REPLAY_CONTRACT)


__all__ = [
    "R6A_TB2_ARTIFACT_TYPE",
    "R6A_TB2_EXPECTED",
    "R6A_TB2_FROZEN_FINGERPRINT",
    "R6A_TB2_POLICIES",
    "R6A_TB2_POOL_FINGERPRINT",
    "R6A_TB2_SOURCE_SHA256",
    "build_r6a_tb2_replay_evidence",
    "replay_contract_record",
    "trace_cache_state_fingerprint",
]
