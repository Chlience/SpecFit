
from __future__ import annotations

from collections.abc import Mapping, Sequence

from .j1_screen import float_record
from .m1_screen import (
    M1_QUESTION_IDS,
    CandidateScore,
    score_candidate_subsets,
    screen_state,
)
from .r6a_experiment import FrozenStateRecord, PoolRecord
from .r6a_route_observation import RouteObservation
from .r6a_selection_contract import enumerate_ancestor_closed_subsets


V1_EPSILONS = (0.0, 0.005, 0.01, 0.02, 0.05)
V1_MAIN_EPSILON = 0.01
V1_FIXED7_CANDIDATE_COUNT = 3_991


def _expected(row: CandidateScore) -> float:
    return 1.0 + row.draft_utility


def _row_record(row: CandidateScore) -> dict[str, object]:
    return {
        "selected_raw_ids": list(row.selected_raw_ids),
        "draft_node_count": len(row.selected_raw_ids),
        "draft_utility": float_record(row.draft_utility),
        "expected_committed_tokens_proxy": float_record(_expected(row)),
        "F_atlas": row.F_atlas,
        "N_atlas": row.N_atlas,
    }


def _pareto_indices(
    rows: Sequence[CandidateScore], *, include_residency: bool,
) -> tuple[int, ...]:

    def dominates(left: CandidateScore, right: CandidateScore) -> bool:
        no_worse = (
            _expected(left) >= _expected(right)
            and len(left.selected_raw_ids) <= len(right.selected_raw_ids)
            and left.F_atlas <= right.F_atlas
            and (not include_residency or left.N_atlas <= right.N_atlas)
        )
        strict = (
            _expected(left) > _expected(right)
            or len(left.selected_raw_ids) < len(right.selected_raw_ids)
            or left.F_atlas < right.F_atlas
            or include_residency and left.N_atlas < right.N_atlas
        )
        return no_worse and strict

    return tuple(
        index for index, row in enumerate(rows)
        if not any(dominates(other, row) for other in rows)
    )


def _unique_choice(
    rows: Sequence[CandidateScore], *, floor: float, fixed: bool,
) -> CandidateScore:
    eligible = tuple(
        row for row in rows
        if _expected(row) >= floor
        and (not fixed or len(row.selected_raw_ids) == 7)
    )
    if not eligible:
        raise ValueError("V1 acceptance benefit lower bound has no feasible candidate")
    return min(eligible, key=lambda row: (
        row.F_atlas,
        len(row.selected_raw_ids),
        -_expected(row),
        row.selected_raw_ids,
    ))


def _residency_choice(
    rows: Sequence[CandidateScore], *, blind: CandidateScore, floor: float,
) -> CandidateScore:
    eligible = tuple(
        row for row in rows
        if len(row.selected_raw_ids) == len(blind.selected_raw_ids)
        and row.F_atlas == blind.F_atlas
        and _expected(row) >= floor
    )
    if not eligible:
        raise ValueError("V1 matched residency has no feasible candidate")
    return min(eligible, key=lambda row: (
        row.N_atlas, -_expected(row), row.selected_raw_ids,
    ))


def _choice_record(
    row: CandidateScore, *, reference: CandidateScore,
) -> dict[str, object]:
    value = _row_record(row)
    ratio = _expected(row) / _expected(reference)
    value.update({
        "reference_selected_raw_ids": list(reference.selected_raw_ids),
        "expected_token_ratio_vs_reference": float_record(ratio),
        "required_cycle_time_reduction_vs_reference": float_record(1.0 - ratio),
    })
    return value


def _validate_structure(rows: Sequence[CandidateScore], pool: PoolRecord) -> None:
    parent = {node.raw_id: node.parent_raw_id for node in pool.nodes}
    by_ids = {row.selected_raw_ids: row for row in rows}
    if len(by_ids) != len(rows):
        raise ValueError("V1 candidate set not unique")
    empty = by_ids.get(())
    if empty is None or empty.draft_utility != 0.0 or empty.F_atlas != 384:
        raise ValueError("V1 empty set does not preserve the mandatory seed cost")
    for selected, row in by_ids.items():
        if len(selected) > 7 or any(
            parent[raw_id] != 0 and parent[raw_id] not in selected
            for raw_id in selected
        ):
            raise ValueError("V1 candidate node count or ancestor closure inconsistent")
        for raw_id in selected:
            smaller = tuple(item for item in selected if item != raw_id)
            previous = by_ids.get(smaller)
            if previous is not None and (
                previous.draft_utility > row.draft_utility
                or previous.F_atlas > row.F_atlas
                or previous.N_atlas > row.N_atlas
            ):
                raise ValueError("V1 C/F/N moved in the wrong direction after adding a node")


def analyze_v1_state(
    *,
    state_id: str,
    question_id: int,
    cycle_index: int,
    pool: PoolRecord,
    atlas: RouteObservation,
    frozen: FrozenStateRecord,
) -> dict[str, object]:

    subsets = tuple(
        selected
        for budget in range(8)
        for selected in enumerate_ancestor_closed_subsets(
            pool, draft_node_budget=budget,
        )
    )
    rows = score_candidate_subsets(
        pool=pool, atlas=atlas, frozen=frozen, subsets=subsets,
    )
    _validate_structure(rows, pool)
    fixed = tuple(row for row in rows if len(row.selected_raw_ids) == 7)
    historical = screen_state(
        state_id=state_id,
        question_id=question_id,
        cycle_index=cycle_index,
        pool=pool,
        atlas=atlas,
        frozen=frozen,
    )
    if fixed != historical.rows:
        raise ValueError("V1 fixed-7 row does not reproduce J1 item by item")

    c7_max = max(_expected(row) for row in fixed)
    tolerances = []
    for epsilon in V1_EPSILONS:
        floor = (1.0 - epsilon) * c7_max
        fixed_unique = _unique_choice(rows, floor=floor, fixed=True)
        variable_unique = _unique_choice(rows, floor=floor, fixed=False)
        fixed_residency = _residency_choice(
            rows, blind=fixed_unique, floor=floor,
        )
        variable_residency = _residency_choice(
            rows, blind=variable_unique, floor=floor,
        )
        tolerances.append({
            "epsilon": float_record(epsilon),
            "expected_token_floor": float_record(floor),
            "fixed_unique": _row_record(fixed_unique),
            "variable_unique": _choice_record(
                variable_unique, reference=fixed_unique,
            ),
            "fixed_residency": _row_record(fixed_residency),
            "variable_residency": _choice_record(
                variable_residency, reference=variable_unique,
            ),
        })

    pareto_unique = _pareto_indices(rows, include_residency=False)
    pareto_residency = _pareto_indices(rows, include_residency=True)
    return {
        "state_id": state_id,
        "question_id": question_id,
        "cycle_index": cycle_index,
        "candidate_count_by_draft_node_count": [
            sum(len(row.selected_raw_ids) == budget for row in rows)
            for budget in range(8)
        ],
        "fixed7_candidate_rows_reproduced": True,
        "fixed7_max_expected_committed_tokens_proxy": float_record(c7_max),
        "pareto_without_residency": [
            _row_record(rows[index]) for index in pareto_unique
        ],
        "pareto_with_residency": [
            _row_record(rows[index]) for index in pareto_residency
        ],
        "tolerances": tolerances,
    }


def summarize_v1_states(states: Sequence[Mapping[str, object]]) -> dict[str, object]:

    grouped: dict[int, list[Mapping[str, object]]] = {}
    for state in states:
        grouped.setdefault(int(state["question_id"]), []).append(state)
    if (
        len(states) != 32
        or tuple(sorted(grouped)) != M1_QUESTION_IDS
        or len({str(state["state_id"]) for state in states}) != 32
        or any(
            len(items) != 4
            or len({int(item["cycle_index"]) for item in items}) != 4
            for items in grouped.values()
        )
    ):
        raise ValueError("V1 state does not match 8 prompt x 4 state")

    summaries = []
    for tolerance_index, epsilon in enumerate(V1_EPSILONS):
        rows = [state["tolerances"][tolerance_index] for state in states]
        fixed_f = sum(row["fixed_unique"]["F_atlas"] for row in rows)
        variable_f = sum(row["variable_unique"]["F_atlas"] for row in rows)
        variable_n = sum(row["variable_unique"]["N_atlas"] for row in rows)
        residency_n = sum(row["variable_residency"]["N_atlas"] for row in rows)
        fixed_residency_n = sum(row["fixed_residency"]["N_atlas"] for row in rows)
        general_deltas = tuple(
            row["fixed_unique"]["F_atlas"] - row["variable_unique"]["F_atlas"]
            if row["variable_unique"]["draft_node_count"] < 7 else 0
            for row in rows
        )
        residency_deltas = tuple(
            row["variable_unique"]["N_atlas"]
            - row["variable_residency"]["N_atlas"]
            for row in rows
        )

        def positive_questions(deltas: Sequence[int]) -> int:
            return sum(
                sum(
                    delta for delta, state in zip(deltas, states, strict=True)
                    if int(state["question_id"]) == question_id
                ) > 0
                for question_id in grouped
            )

        general_gain = (fixed_f - variable_f) / fixed_f
        residency_gain = (
            (variable_n - residency_n) / variable_n if variable_n else 0.0
        )
        general_passed = (
            epsilon == V1_MAIN_EPSILON
            and general_gain >= 0.10
            and sum(delta > 0 for delta in general_deltas) >= 16
            and positive_questions(general_deltas) >= 6
        )
        residency_passed = (
            epsilon == V1_MAIN_EPSILON
            and residency_gain >= 0.05
            and sum(delta > 0 for delta in residency_deltas) >= 8
            and positive_questions(residency_deltas) >= 4
            and residency_n <= fixed_residency_n
        )
        summaries.append({
            "epsilon": float_record(epsilon),
            "fixed_unique_total_F_atlas": fixed_f,
            "variable_unique_total_F_atlas": variable_f,
            "variable_F_atlas_reduction": float_record(general_gain),
            "variable_positive_state_count": sum(
                delta > 0 for delta in general_deltas
            ),
            "variable_positive_question_count": positive_questions(general_deltas),
            "variable_unique_total_N_atlas": variable_n,
            "variable_residency_total_N_atlas": residency_n,
            "fixed_residency_total_N_atlas": fixed_residency_n,
            "matched_residency_N_atlas_reduction": float_record(residency_gain),
            "residency_positive_state_count": sum(
                delta > 0 for delta in residency_deltas
            ),
            "residency_positive_question_count": positive_questions(
                residency_deltas
            ),
            "residency_not_worse_than_fixed_guard": (
                residency_n <= fixed_residency_n
            ),
            "general_gate_passed": general_passed,
            "residency_gate_passed": residency_passed,
        })

    main = summaries[V1_EPSILONS.index(V1_MAIN_EPSILON)]
    enter = main["general_gate_passed"] and main["residency_gate_passed"]
    status = (
        "review_v2_shape_calibration"
        if enter else
        "generic_variable_opportunity_only"
        if main["general_gate_passed"] else
        "stop_variable_budget_extension"
    )
    return {
        "state_count": len(states),
        "question_count": len(grouped),
        "tolerances": summaries,
        "gate": {"status": status, "enter_v2_review": enter},
    }


__all__ = [
    "V1_EPSILONS",
    "V1_FIXED7_CANDIDATE_COUNT",
    "V1_MAIN_EPSILON",
    "analyze_v1_state",
    "summarize_v1_states",
]
