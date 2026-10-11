
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .r6a_experiment import (
    FrozenStateRecord,
    PoolRecord,
    R6A_DRAFT_NODE_BUDGET,
    _int,
    _ints,
    _map,
    _seq,
)
from .r6a_route_observation import RouteObservation

R6A_POLICIES = (
    "probability",
    "unique_efficiency",
    "initial_nonresident_efficiency",
)


@dataclass(frozen=True, slots=True)
class SelectionScore:

    selected_raw_ids: tuple[int, ...]
    draft_utility_proxy: float
    unique_expert_count: int
    initial_frozen_nonresident_expert_count: int

    def efficiency(self, cost: Literal["unique", "initial_nonresident"]) -> float:
        denominator = (
            self.unique_expert_count
            if cost == "unique"
            else self.initial_frozen_nonresident_expert_count
        )
        return (
            float("inf")
            if denominator == 0
            else self.draft_utility_proxy / denominator
        )


def enumerate_ancestor_closed_subsets(
    pool: PoolRecord,
    *,
    draft_node_budget: int = R6A_DRAFT_NODE_BUDGET,
) -> tuple[tuple[int, ...], ...]:

    if isinstance(draft_node_budget, bool) or not isinstance(
        draft_node_budget, int
    ) or draft_node_budget < 0:
        raise ValueError("draft_node_budget must be a non-negative integer")
    nodes = tuple((node.raw_id, 1 << node.raw_id, 1 << node.parent_raw_id)
                  for node in pool.nodes)
    count = len(nodes)
    if draft_node_budget > count:
        return ()
    results = []
    selected = []

    def extend(start: int, remaining: int, included: int) -> None:
        if remaining == 0:
            results.append(tuple(selected))
            return
        for index in range(start, count - remaining + 1):
            raw_id, node_bit, parent_bit = nodes[index]
            if included & parent_bit:
                selected.append(raw_id)
                extend(index + 1, remaining - 1, included | node_bit)
                selected.pop()

    extend(0, draft_node_budget, 1)
    return tuple(results)


def _score(
    selected: tuple[int, ...],
    *,
    pool: PoolRecord,
    atlas: RouteObservation,
    frozen: FrozenStateRecord,
) -> SelectionScore:
    by_id = {node.raw_id: node for node in pool.nodes}
    probability_mass = sum(
        math.exp(by_id[raw_id].cumulative_log_probability)
        for raw_id in selected
    )
    keys = atlas.unique_expert_keys(selected)
    resident = frozenset(
        (layer_id, expert_id)
        for layer_id, experts in enumerate(frozen.resident_by_layer)
        for expert_id in experts
    )
    return SelectionScore(
        selected_raw_ids=selected,
        draft_utility_proxy=1.0 + probability_mass,
        unique_expert_count=len(keys),
        initial_frozen_nonresident_expert_count=len(keys - resident),
    )


def _best(
    scores: tuple[SelectionScore, ...],
    cost: Literal["unique", "initial_nonresident"],
) -> SelectionScore:
    def cost_value(score: SelectionScore) -> int:
        return (
            score.unique_expert_count
            if cost == "unique"
            else score.initial_frozen_nonresident_expert_count
        )

    return min(
        scores,
        key=lambda score: (
            -score.efficiency(cost),
            -score.draft_utility_proxy,
            cost_value(score),
            score.selected_raw_ids,
        ),
    )


def expected_policy_decisions(
    pool: PoolRecord,
    atlas: RouteObservation,
    frozen: FrozenStateRecord,
    *,
    subsets: tuple[tuple[int, ...], ...] | None = None,
) -> dict[str, SelectionScore]:

    subsets = (
        enumerate_ancestor_closed_subsets(pool)
        if subsets is None
        else subsets
    )
    if not subsets:
        raise ValueError("The current pool has no fixed-7 ancestor-closed set")
    scores = tuple(
        _score(selected, pool=pool, atlas=atlas, frozen=frozen)
        for selected in subsets
    )
    probability_ids = tuple(
        sorted(
            node.raw_id
            for node in sorted(
                pool.nodes,
                key=lambda node: (
                    -node.cumulative_log_probability,
                    node.depth,
                    node.generation_ordinal,
                ),
            )[:R6A_DRAFT_NODE_BUDGET]
        )
    )
    by_ids = {score.selected_raw_ids: score for score in scores}
    if probability_ids not in by_ids:
        raise ValueError("probability fixed-7 selection does not satisfy ancestor closure")
    return {
        "probability": by_ids[probability_ids],
        "unique_efficiency": _best(scores, "unique"),
        "initial_nonresident_efficiency": _best(
            scores, "initial_nonresident"
        ),
    }


def validate_policy_decisions(
    value: object,
    *,
    pool: PoolRecord,
    atlas: RouteObservation,
    frozen: FrozenStateRecord,
) -> dict[str, tuple[int, ...]]:

    offline = _map("offline_selection", value)
    subsets = enumerate_ancestor_closed_subsets(pool)
    recorded_count = _int(
        "offline_selection.ancestor_closed_fixed_budget_subset_count",
        offline.get("ancestor_closed_fixed_budget_subset_count"),
        1,
    )
    if recorded_count != len(subsets):
        raise ValueError("artifact subset count does not match the current pool topology re-enumeration")
    expected = expected_policy_decisions(
        pool,
        atlas,
        frozen,
        subsets=subsets,
    )
    records = _seq("offline_selection.decisions", offline.get("decisions"))
    if len(records) != len(R6A_POLICIES):
        raise ValueError("offline decisions must contain exactly three formal policies")
    actual: dict[str, tuple[int, ...]] = {}
    for index, value in enumerate(records):
        record = _map(f"decision[{index}]", value)
        policy = record.get("policy")
        if not isinstance(policy, str) or policy not in expected or policy in actual:
            raise ValueError("offline policy name, count, or uniqueness is inconsistent")
        wanted = expected[policy]
        selected = _ints(
            f"decision[{index}].selected_raw_ids",
            record.get("selected_raw_ids"),
            1,
        )
        utility = record.get("draft_utility_proxy")
        if (
            selected != wanted.selected_raw_ids
            or isinstance(utility, bool)
            or not isinstance(utility, (int, float))
            or not math.isclose(
                float(utility), wanted.draft_utility_proxy, rel_tol=1e-9
            )
            or _int(
                "unique_expert_count", record.get("unique_expert_count")
            )
            != wanted.unique_expert_count
            or _int(
                "initial_frozen_nonresident_expert_count",
                record.get("initial_frozen_nonresident_expert_count"),
            )
            != wanted.initial_frozen_nonresident_expert_count
        ):
            raise ValueError(f"{policy} decision inconsistent with atlas recomputation result")
        actual[policy] = selected
    if set(actual) != set(R6A_POLICIES):
        raise ValueError("offline decisions missing formal policy")
    return actual


__all__ = [
    "R6A_POLICIES",
    "SelectionScore",
    "enumerate_ancestor_closed_subsets",
    "expected_policy_decisions",
    "validate_policy_decisions",
]
