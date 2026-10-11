
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .r6a_experiment import (
    FrozenStateRecord,
    PoolRecord,
    R6A_DRAFT_NODE_BUDGET,
    R6A_EXPERTS_PER_TOKEN,
    R6A_MOE_LAYER_COUNT,
)
from .r6a_route_observation import RouteObservation
from .r6a_selection_contract import enumerate_ancestor_closed_subsets

M1_RATIO_GREEDY_EPSILON = 1e-6
M1_UTILITY_RATIO = 0.99
M1_QUESTION_IDS = tuple(range(406, 477, 10))


@dataclass(frozen=True, slots=True)
class CandidateScore:

    selected_raw_ids: tuple[int, ...]
    draft_utility: float
    F_atlas: int
    N_atlas: int


@dataclass(frozen=True, slots=True)
class StateScreen:

    state_id: str
    question_id: int
    cycle_index: int
    rows: tuple[CandidateScore, ...]
    ratio_greedy: CandidateScore
    V_F: CandidateScore
    V_R: CandidateScore
    pareto_frontier_indices: tuple[int, ...]

    @property
    def delta_N_atlas(self) -> int:
        return self.V_F.N_atlas - self.V_R.N_atlas


@dataclass(frozen=True, slots=True)
class GateResult:

    G_atlas: float | None
    delta_N_atlas: int
    baseline_N_atlas: int
    positive_state_count: int
    positive_question_count: int
    validation_passed: bool
    passed: bool


def _atlas_experts(
    pool: PoolRecord,
    atlas: RouteObservation,
) -> dict[int, frozenset[tuple[int, int]]]:
    raw_ids = (0, *(node.raw_id for node in pool.nodes))
    if raw_ids != tuple(range(19)) or atlas.position_to_raw != raw_ids:
        raise ValueError("M1 full atlas must cover raw ID 0..18 by verification position")
    expected = {
        (raw_id, layer_id)
        for raw_id in raw_ids
        for layer_id in range(R6A_MOE_LAYER_COUNT)
    }
    observed: dict[tuple[int, int], tuple[int, ...]] = {}
    for raw_id, layer_id, experts in atlas.expert_ids_by_raw_layer:
        key = (raw_id, layer_id)
        if (
            key in observed
            or len(experts) != R6A_EXPERTS_PER_TOKEN
            or len(set(experts)) != R6A_EXPERTS_PER_TOKEN
        ):
            raise ValueError("M1 full atlas route has duplicate or non-Top-8 rows")
        observed[key] = experts
    if set(observed) != expected:
        raise ValueError("M1 full atlas must fully cover 19x48 route rows")
    return {
        raw_id: frozenset(
            (layer_id, expert_id)
            for layer_id in range(R6A_MOE_LAYER_COUNT)
            for expert_id in observed[raw_id, layer_id]
        )
        for raw_id in raw_ids
    }


def _score(
    selected: tuple[int, ...],
    *,
    probabilities: dict[int, float],
    experts_by_raw: dict[int, frozenset[tuple[int, int]]],
    resident: frozenset[tuple[int, int]],
) -> CandidateScore:
    footprint = frozenset().union(
        experts_by_raw[0], *(experts_by_raw[raw_id] for raw_id in selected)
    )
    return CandidateScore(
        selected_raw_ids=selected,
        draft_utility=sum(
            (probabilities[raw_id] for raw_id in selected), start=0.0,
        ),
        F_atlas=len(footprint),
        N_atlas=len(footprint - resident),
    )


def _ratio_greedy_ids(
    pool: PoolRecord,
    *,
    probabilities: dict[int, float],
    experts_by_raw: dict[int, frozenset[tuple[int, int]]],
) -> tuple[int, ...]:
    by_id = {node.raw_id: node for node in pool.nodes}
    selected: set[int] = set()
    buffer = set(experts_by_raw[0])

    def path_footprint(raw_id: int) -> frozenset[tuple[int, int]]:
        path = []
        while raw_id:
            path.append(raw_id)
            raw_id = by_id[raw_id].parent_raw_id
        return frozenset().union(
            *(experts_by_raw[item] for item in reversed(path))
        )

    for _ in range(R6A_DRAFT_NODE_BUDGET):
        frontier = tuple(
            node for node in pool.nodes
            if node.raw_id not in selected
            and (node.parent_raw_id == 0 or node.parent_raw_id in selected)
        )
        if not frontier:
            raise ValueError("ratio-greedy admissible frontier cannot fill the fixed-7 budget")

        def key(node: object) -> tuple[float, float, int, int]:
            raw_id = node.raw_id
            incremental = len(path_footprint(raw_id) - buffer)
            score = probabilities[raw_id] / (incremental + M1_RATIO_GREEDY_EPSILON)
            return (-score, -probabilities[raw_id], node.depth, raw_id)

        chosen = min(frontier, key=key)
        selected.add(chosen.raw_id)
        buffer.update(path_footprint(chosen.raw_id))
    return tuple(sorted(selected))


def pareto_frontier_indices(rows: Sequence[CandidateScore]) -> tuple[int, ...]:

    return tuple(
        index for index, row in enumerate(rows)
        if not any(
            other.draft_utility >= row.draft_utility
            and other.F_atlas <= row.F_atlas
            and other.N_atlas <= row.N_atlas
            and (
                other.draft_utility > row.draft_utility
                or other.F_atlas < row.F_atlas
                or other.N_atlas < row.N_atlas
            )
            for other in rows
        )
    )


def score_candidate_subsets(
    *,
    pool: PoolRecord,
    atlas: RouteObservation,
    frozen: FrozenStateRecord,
    subsets: Sequence[tuple[int, ...]],
) -> tuple[CandidateScore, ...]:

    if len(frozen.resident_by_layer) != R6A_MOE_LAYER_COUNT:
        raise ValueError("M1 frozen residency must fully cover 48 layers")
    experts_by_raw = _atlas_experts(pool, atlas)
    probabilities = {
        node.raw_id: math.exp(node.cumulative_log_probability)
        for node in pool.nodes
    }
    resident = frozenset(
        (layer_id, expert_id)
        for layer_id, experts in enumerate(frozen.resident_by_layer)
        for expert_id in experts
    )
    return tuple(_score(
        selected,
        probabilities=probabilities,
        experts_by_raw=experts_by_raw,
        resident=resident,
    ) for selected in subsets)


def screen_state(
    *,
    state_id: str,
    question_id: int,
    cycle_index: int,
    pool: PoolRecord,
    atlas: RouteObservation,
    frozen: FrozenStateRecord,
) -> StateScreen:

    subsets = tuple(sorted(enumerate_ancestor_closed_subsets(pool)))
    if not subsets:
        raise ValueError("The current pool has no fixed-7 ancestor-closed set")
    rows = score_candidate_subsets(
        pool=pool, atlas=atlas, frozen=frozen, subsets=subsets,
    )
    experts_by_raw = _atlas_experts(pool, atlas)
    probabilities = {
        node.raw_id: math.exp(node.cumulative_log_probability)
        for node in pool.nodes
    }
    by_ids = {row.selected_raw_ids: row for row in rows}
    ratio_greedy = by_ids[_ratio_greedy_ids(
        pool,
        probabilities=probabilities,
        experts_by_raw=experts_by_raw,
    )]
    V_F = min(
        (row for row in rows
         if row.draft_utility >= M1_UTILITY_RATIO * ratio_greedy.draft_utility),
        key=lambda row: (row.F_atlas, -row.draft_utility,
                         row.selected_raw_ids),
    )
    utility_floor = M1_UTILITY_RATIO * max(
        ratio_greedy.draft_utility, V_F.draft_utility
    )
    V_R = min(
        (row for row in rows
         if row.F_atlas == V_F.F_atlas
         and row.draft_utility >= utility_floor),
        key=lambda row: (row.N_atlas, -row.draft_utility,
                         row.selected_raw_ids),
    )
    return StateScreen(
        state_id=state_id,
        question_id=question_id,
        cycle_index=cycle_index,
        rows=rows,
        ratio_greedy=ratio_greedy,
        V_F=V_F,
        V_R=V_R,
        pareto_frontier_indices=pareto_frontier_indices(rows),
    )


def evaluate_gate(
    states: Sequence[StateScreen],
    *,
    validation_passed: bool,
) -> GateResult:

    if not isinstance(validation_passed, bool):
        raise ValueError("validation_passed must be a boolean")
    grouped: dict[int, list[StateScreen]] = {}
    for state in states:
        grouped.setdefault(state.question_id, []).append(state)
    valid_distribution = (
        len(states) == 32
        and tuple(sorted(grouped)) == M1_QUESTION_IDS
        and len({state.state_id for state in states}) == 32
        and all(
            len(items) == 4
            and len({item.cycle_index for item in items}) == 4
            for items in grouped.values()
        )
    )
    if not valid_distribution:
        raise ValueError("M1 state does not match the pre-registered 8 prompt x 4 offset distribution")
    deltas = tuple(state.delta_N_atlas for state in states)
    if any(delta < 0 for delta in deltas):
        raise ValueError("matched residency oracle produced negative delta_N_atlas")
    baseline = sum(state.V_F.N_atlas for state in states)
    total_delta = sum(deltas)
    G_atlas = total_delta / baseline if baseline else None
    positive_questions = sum(
        sum(item.delta_N_atlas for item in items) > 0
        for items in grouped.values()
    )
    positive_states = sum(delta > 0 for delta in deltas)
    passed = (
        validation_passed
        and G_atlas is not None
        and G_atlas >= 0.05
        and positive_states >= 17
        and positive_questions >= 6
    )
    return GateResult(
        G_atlas=G_atlas,
        delta_N_atlas=total_delta,
        baseline_N_atlas=baseline,
        positive_state_count=positive_states,
        positive_question_count=positive_questions,
        validation_passed=validation_passed,
        passed=passed,
    )
