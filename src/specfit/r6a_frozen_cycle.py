
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from math import exp, isclose
from typing import Literal

from .runtime.cache_snapshot import GpuExpertCacheSnapshot
from .runtime.eagle3_selection import candidate_pool_fingerprint
from .runtime.eagle3_types import Eagle3DraftCandidatePool
from .state import ExpertKey
from .trace import CacheSnapshotTrace, LayerCacheSnapshotTrace, TraceBundle


@dataclass(frozen=True, slots=True)
class FrozenNodeRoute:

    raw_id: int
    experts_by_layer: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if self.raw_id < 0 or not self.experts_by_layer:
            raise ValueError("route requires non-negative raw ID and non-empty per-layer routing")
        if any(not experts or len(experts) != len(set(experts))
               for experts in self.experts_by_layer):
            raise ValueError("each layer must contain non-empty, non-duplicated experts")

    @property
    def layout(self) -> tuple[int, int]:
        widths = {len(experts) for experts in self.experts_by_layer}
        if len(widths) != 1:
            raise ValueError("per-layer Top-K width inconsistent")
        return len(self.experts_by_layer), next(iter(widths))

@dataclass(frozen=True, slots=True)
class FrozenRouteTable:

    pool_fingerprint: str
    routes: tuple[FrozenNodeRoute, ...]

    def __post_init__(self) -> None:
        raw_ids = tuple(route.raw_id for route in self.routes)
        if len(self.pool_fingerprint) != 64 or not self.routes:
            raise ValueError("route table requires candidate pool SHA-256 and a non-empty route")
        if raw_ids != tuple(sorted(set(raw_ids))) or raw_ids[0] != 0:
            raise ValueError("route table must start with seed and be ordered by unique ascending raw ID")
        if len({route.layout for route in self.routes}) != 1:
            raise ValueError("per-layer layout in the route table must be consistent")

    def for_raw_id(self, raw_id: int) -> FrozenNodeRoute:
        try:
            return next(route for route in self.routes if route.raw_id == raw_id)
        except StopIteration as error:
            raise KeyError(raw_id) from error

    def validate_pool(self, pool: Eagle3DraftCandidatePool) -> None:
        if candidate_pool_fingerprint(pool) != self.pool_fingerprint:
            raise ValueError("route table fingerprint inconsistent with candidate pool")
        allowed = {0, *(node.raw_id for node in pool.candidates)}
        extra = {route.raw_id for route in self.routes} - allowed
        if extra:
            raise ValueError(f"route table contains raw IDs outside the pool: {sorted(extra)}")

    def validate_selection(
        self,
        pool: Eagle3DraftCandidatePool,
        selected_raw_ids: tuple[int, ...],
    ) -> tuple[int, ...]:
        self.validate_pool(pool)
        ordered = pool.to_tree(selected_raw_ids).source_raw_ids
        present = {route.raw_id for route in self.routes}
        missing = {0, *ordered} - present
        if missing:
            raise ValueError(f"route table does not cover the selected subtree: {sorted(missing)}")
        return ordered

    def validate_complete_pool(self, pool: Eagle3DraftCandidatePool) -> None:
        ordered = tuple(node.raw_id for node in pool.candidates)
        self.validate_selection(pool, ordered)


@dataclass(frozen=True, slots=True)
class CandidateSetScore:

    selected_raw_ids: tuple[int, ...]
    draft_path_probability_mass: float
    draft_utility_proxy: float
    unique_expert_count: int
    initial_nonresident_expert_count: int

    def __post_init__(self) -> None:
        valid_cost = (
            0 <= self.initial_nonresident_expert_count <= self.unique_expert_count
        )
        if not isclose(self.draft_utility_proxy,
                       1.0 + self.draft_path_probability_mass) or not valid_cost:
            raise ValueError("draft utility or expert set count mismatch")

    def efficiency(self, cost: Literal["unique", "initial_nonresident"]) -> float:
        denominator = _cost(self, cost)
        return (
            float("inf") if denominator == 0
            else self.draft_utility_proxy / denominator
        )


def enumerate_ancestor_closed_subsets(
    pool: Eagle3DraftCandidatePool,
    *,
    draft_node_budget: int,
) -> tuple[tuple[int, ...], ...]:
    if isinstance(draft_node_budget, bool) or draft_node_budget < 0:
        raise ValueError("draft_node_budget must be a non-negative integer")
    if draft_node_budget > len(pool.candidates):
        return ()
    parent = {node.raw_id: node.parent_raw_id for node in pool.candidates}
    return tuple(
        pool.to_tree(selected).source_raw_ids
        for selected in combinations(parent, draft_node_budget)
        if all(
            parent[raw_id] == 0 or parent[raw_id] in selected
            for raw_id in selected
        )
    )


def score_candidate_set(
    pool: Eagle3DraftCandidatePool,
    *,
    selected_raw_ids: tuple[int, ...],
    routes: FrozenRouteTable,
    initial_resident_experts: frozenset[ExpertKey],
) -> CandidateSetScore:
    ordered = routes.validate_selection(pool, selected_raw_ids)
    rows = tuple(routes.for_raw_id(raw_id) for raw_id in (0, *ordered))
    keys = frozenset(
        ExpertKey(layer_id, expert_id)
        for row in rows
        for layer_id, experts in enumerate(row.experts_by_layer)
        for expert_id in experts
    )
    by_raw_id = {node.raw_id: node for node in pool.candidates}
    mass = sum(exp(by_raw_id[raw_id].cumulative_log_probability) for raw_id in ordered)
    return CandidateSetScore(
        selected_raw_ids=ordered,
        draft_path_probability_mass=mass,
        draft_utility_proxy=1.0 + mass,
        unique_expert_count=len(keys),
        initial_nonresident_expert_count=len(keys - initial_resident_experts),
    )


def select_max_efficiency(
    scores: tuple[CandidateSetScore, ...],
    *,
    cost: Literal["unique", "initial_nonresident"],
) -> CandidateSetScore:
    if not scores:
        raise ValueError("scores cannot be empty")
    return min(
        scores,
        key=lambda score: (-score.efficiency(cost), -score.draft_utility_proxy,
                           _cost(score, cost), score.selected_raw_ids),
    )


def _cost(
    score: CandidateSetScore,
    cost: Literal["unique", "initial_nonresident"],
) -> int:
    if cost == "unique":
        return score.unique_expert_count
    if cost == "initial_nonresident":
        return score.initial_nonresident_expert_count
    raise ValueError("cost must be unique or initial_nonresident")


def route_table_from_trace(
    trace: TraceBundle,
    *,
    pool: Eagle3DraftCandidatePool,
    verification_position_to_raw_id: tuple[int, ...],
) -> FrozenRouteTable:

    if len(trace.cycles) != 1 or len(trace.trees) != 1:
        raise ValueError("Frozen route extraction requires the trace to contain exactly one cycle")
    cycle = trace.cycles[0]
    selected = verification_position_to_raw_id[1:]
    tree = pool.to_tree(selected)
    if verification_position_to_raw_id != tree.verification_raw_ids:
        raise ValueError("verification position did not use canonical raw-ID order")
    if cycle.verification_token_ids != (pool.seed_token_id, *tree.token_ids):
        raise ValueError("verification token is inconsistent with the raw-ID mapping")
    nodes = tuple(sorted((node for node in trace.trees[0].nodes if node.verified),
                         key=lambda node: node.verification_position))
    if len(nodes) != len(verification_position_to_raw_id):
        raise ValueError("verified node count inconsistent with raw-ID mapping")
    route_map = {(route.node_id, route.layer_id): route
                 for route in trace.target_routes if route.tree_id == cycle.tree_id}
    frozen = []
    for node, raw_id in zip(nodes, verification_position_to_raw_id):
        try:
            layers = tuple(route_map[(node.node_id, layer_id)].expert_ids
                           for layer_id in range(trace.manifest.num_moe_layers))
        except KeyError as error:
            raise ValueError(f"node {node.node_id} missing its own actual route") from error
        frozen.append(FrozenNodeRoute(raw_id, layers))
    table = FrozenRouteTable(candidate_pool_fingerprint(pool), tuple(frozen))
    table.validate_selection(pool, selected)
    if table.routes[0].layout != (
        trace.manifest.num_moe_layers, trace.manifest.experts_per_token
    ):
        raise ValueError("Actual route layout does not match the manifest")
    return table


def resident_experts_from_snapshot(
    snapshot: CacheSnapshotTrace,
) -> frozenset[ExpertKey]:
    return frozenset(
        ExpertKey(layer.layer_id, slot.expert_id)
        for layer in snapshot.layers for slot in layer.slots
        if slot.expert_id is not None and slot.ready
    )


def cache_snapshot_state_fingerprint(snapshot: CacheSnapshotTrace) -> str:
    return _state_fingerprint(
        snapshot.runtime_state_version,
        tuple(_layer_payload(layer.layer_id, layer) for layer in snapshot.layers),
    )


def live_cache_snapshots_state_fingerprint(
    snapshots: tuple[GpuExpertCacheSnapshot, ...],
) -> str:
    return _state_fingerprint(
        sum(layer.host_state_version for layer in snapshots),
        tuple(_layer_payload(layer_id, layer)
              for layer_id, layer in enumerate(snapshots)),
    )


def _layer_payload(
    layer_id: int,
    layer: LayerCacheSnapshotTrace | GpuExpertCacheSnapshot,
) -> dict[str, object]:
    if layer.cache_id is None or layer.host_state_version is None \
            or layer.lru_sequence is None or layer.stats is None:
        raise ValueError("state summary requires complete extended snapshot fields")
    def expert_id(value: object) -> int | None:
        return _expert_id(value, layer_id)

    return {
        "layer_id": layer_id, "cache_id": layer.cache_id, "queue_id": layer.queue_id,
        "capacity": layer.capacity, "host_state_version": layer.host_state_version,
        "lru_sequence": layer.lru_sequence,
        "submission_sequence": layer.submission_sequence,
        "slots": [(slot.slot_id, expert_id(slot.expert_id), slot.generation,
                   slot.active_leases, slot.last_used_sequence, slot.ready,
                   slot.pending_use_guards) for slot in layer.slots],
        "stats": (layer.stats.h2d_submissions, layer.stats.resident_hits,
                  layer.stats.inflight_joins, layer.stats.evictions,
                  layer.stats.pending_use_guards),
        "pending": [(expert_id(item.expert_id), item.slot_id, item.generation,
                     item.submission_sequence, item.ready_event_completed)
                    for item in layer.pending_transfers],
    }


def _expert_id(value: object, layer_id: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, ExpertKey):
        if value.layer_id != layer_id:
            raise ValueError("expert key layer is inconsistent with the cache layer")
        return value.expert_id
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError("cache expert ID must be a non-negative integer or ExpertKey")
    return value


def _state_fingerprint(
    version: int,
    layers: tuple[dict[str, object], ...],
) -> str:
    encoded = json.dumps((version, layers), sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
