
from __future__ import annotations

from collections.abc import Mapping, Sequence

from .experiment import canonical_json_sha256
from .m1_capture_pool import ReferencePool, read_full_pool
from .r6a_experiment import PoolRecord
from .r6a_route_observation import RouteObservation
from .replay import replay_qwen3_access_cycles
from .trace import (
    CacheSlotTrace,
    CacheSnapshotTrace,
    CacheStatsTrace,
    LayerCacheSnapshotTrace,
    RuntimeEventKind,
    SnapshotPhase,
)

_LAYER_KEYS = {
    "layer_id", "capacity", "lru_sequence", "submission_sequence",
    "counters", "slots",
}
_COUNTER_KEYS = {
    "h2d_submissions", "resident_hits", "inflight_joins", "evictions",
    "pending_use_guards",
}
_SLOT_KEYS = {
    "slot_id", "expert_id", "generation", "active_leases",
    "last_used_sequence", "ready", "pending_use_guards",
}
def same_json_value(observed: object, expected: object) -> bool:

    if isinstance(expected, Mapping):
        return isinstance(observed, Mapping) and set(observed) == set(expected) and all(
            same_json_value(observed[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(observed, list) and len(observed) == len(expected) and all(
            same_json_value(left, right) for left, right in zip(observed, expected, strict=True)
        )
    return type(observed) is type(expected) and observed == expected


def _object(name: str, value: object, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{name} field set inconsistent")
    return value


def _sequence(name: str, value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an array")
    return value


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be no less than {minimum} as an integer")
    return value


read_pool = read_full_pool


def read_routes(
    value: object, name: str, pool: PoolRecord | ReferencePool,
    positions: tuple[int, ...],
) -> RouteObservation:
    route = _object(name, value,
                    {"pool_fingerprint", "coverage_rows", "digest", "rows"})
    for index, row in enumerate(_sequence(name, route["rows"])):
        _object(f"{name}.rows[{index}]", row,
                {"verification_position", "raw_id", "layer_id", "expert_ids"})
    return RouteObservation.read(
        route, name=name, position_to_raw=positions,
        pool_fingerprint=pool.fingerprint, num_experts=128,
    )


def read_logical_state(
    value: object, name: str, *, expected_capacity: int = 48,
) -> tuple[dict[str, object], CacheSnapshotTrace]:

    if type(expected_capacity) is not int or not 1 <= expected_capacity <= 128:
        raise ValueError('invalid expected cache capacity')

    root = _object(name, value, {"layers"})
    normalized: list[dict[str, object]] = []
    parsed_layers = []
    for index, raw in enumerate(_sequence(name, root["layers"])):
        layer = _object(f"{name}.layers[{index}]", raw, _LAYER_KEYS)
        counters = _object(f"{name}.layers[{index}].counters",
                           layer["counters"], _COUNTER_KEYS)
        parsed_slots = []
        normalized_slots = []
        for slot_index, raw_slot in enumerate(_sequence(name, layer["slots"])):
            slot = _object(
                f"{name}.layers[{index}].slots[{slot_index}]",
                raw_slot, _SLOT_KEYS,
            )
            expert = slot["expert_id"]
            if expert is not None:
                expert = _integer(name, expert)
                if expert >= 128:
                    raise ValueError(f"{name} expert ID out of range")
            slot_id = _integer(name, slot["slot_id"])
            if slot_id != slot_index:
                raise ValueError(f"{name} slots must be in ascending slot_id order")
            parsed = CacheSlotTrace(
                slot_id, expert,
                _integer(name, slot["generation"]),
                _integer(name, slot["active_leases"]),
                _integer(name, slot["last_used_sequence"]), slot["ready"],
                _integer(name, slot["pending_use_guards"]),
            )
            if (
                parsed.active_leases or parsed.pending_use_guards
                or (expert is not None and not parsed.ready)
            ):
                raise ValueError(f"{name} is not a quiescent ready state")
            parsed_slots.append(parsed)
            normalized_slots.append({
                "slot_id": parsed.slot_id, "expert_id": parsed.expert_id,
                "generation": parsed.generation, "active_leases": 0,
                "last_used_sequence": parsed.last_used_sequence,
                "ready": parsed.ready, "pending_use_guards": 0,
            })
        counter_values = {
            key: _integer(name, counters[key]) for key in _COUNTER_KEYS
        }
        stats = CacheStatsTrace(*(counter_values[key] for key in (
            "h2d_submissions", "resident_hits", "inflight_joins",
            "evictions", "pending_use_guards",
        )))
        parsed_layer = LayerCacheSnapshotTrace(
            _integer(name, layer["layer_id"]), f"m1-q-{index}",
            _integer(name, layer["capacity"], 1),
            _integer(name, layer["submission_sequence"]),
            tuple(parsed_slots), cache_id=f"m1-c-{index}",
            host_state_version=0,
            lru_sequence=_integer(name, layer["lru_sequence"]), stats=stats,
        )
        parsed_layers.append(parsed_layer)
        normalized.append({
            "layer_id": parsed_layer.layer_id,
            "capacity": parsed_layer.capacity,
            "lru_sequence": parsed_layer.lru_sequence,
            "submission_sequence": parsed_layer.submission_sequence,
            "counters": counter_values, "slots": normalized_slots,
        })
    if (
        tuple(layer.layer_id for layer in parsed_layers) != tuple(range(48))
        or any(layer.capacity != expected_capacity for layer in parsed_layers)
    ):
        raise ValueError(f"{name} must fully cover 48x{expected_capacity} cache state")
    state = {"layers": normalized}
    snapshot = CacheSnapshotTrace(
        "m1", "m1", "m1", SnapshotPhase.BEFORE_VERIFY, 0, 0,
        tuple(parsed_layers),
    )
    return state, snapshot


def logical_state_sha256(state: Mapping[str, object]) -> str:
    return canonical_json_sha256(state)


def resident_experts_by_layer(
    state: Mapping[str, object],
) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(sorted(
        slot["expert_id"] for slot in layer["slots"]
        if slot["expert_id"] is not None
    )) for layer in state["layers"])


def advance_logical_state(
    state: dict[str, object],
    snapshot: CacheSnapshotTrace,
    routes: RouteObservation,
) -> tuple[dict[str, object], CacheSnapshotTrace]:

    accesses = tuple(tuple(sorted({
        expert
        for _, row_layer, experts in routes.expert_ids_by_raw_layer
        if row_layer == layer_id
        for expert in experts
    })) for layer_id in range(48))
    result = replay_qwen3_access_cycles(
        snapshot, (accesses,), resident_first=True,
    )[0]
    kinds = {layer: [] for layer in range(48)}
    slots = {layer: [] for layer in range(48)}
    for event in result.events:
        kinds[event.layer_id].append(event.kind)
    for slot in result.final_slots:
        slots[slot.layer_id].append({
            "slot_id": slot.slot_id, "expert_id": slot.expert_id,
            "generation": slot.generation, "active_leases": 0,
            "last_used_sequence": slot.last_used_sequence,
            "ready": slot.expert_id is not None, "pending_use_guards": 0,
        })
    layers = []
    for layer_id, old in enumerate(state["layers"]):
        counters = dict(old["counters"])
        for field, kind in (
            ("h2d_submissions", RuntimeEventKind.H2D_SUBMIT),
            ("resident_hits", RuntimeEventKind.RESIDENT_HIT),
            ("evictions", RuntimeEventKind.EVICT),
        ):
            counters[field] += kinds[layer_id].count(kind)
        layers.append({
            "layer_id": layer_id, "capacity": old["capacity"],
            "lru_sequence": result.final_lru_sequences[layer_id],
            "submission_sequence": result.final_submission_sequences[layer_id],
            "counters": counters, "slots": slots[layer_id],
        })
    return read_logical_state({"layers": layers}, "replayed_state",
                              expected_capacity=state["layers"][0]["capacity"])


__all__ = [
    "advance_logical_state", "logical_state_sha256",
    "read_logical_state", "read_pool", "read_routes",
    "resident_experts_by_layer", "same_json_value",
]
