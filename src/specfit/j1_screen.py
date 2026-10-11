
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

from .j1_calibration import CostModel
from .m1_screen import CandidateScore


J1_POLICY_ORDER = (
    "probability",
    "ratio_greedy",
    "unique_joint",
    "residency_joint",
)
J1_PILOT_THRESHOLD = 0.01


def float_record(value: float) -> dict[str, object]:
    if not math.isfinite(value):
        raise ValueError("J1 artifact does not allow non-finite floats")
    return {"value": value, "hex": value.hex()}


def select_joint_candidate(
    rows: Sequence[CandidateScore],
    model: CostModel,
) -> CandidateScore:

    if not rows:
        raise ValueError("J1 selector missing candidate set")

    def key(row: CandidateScore) -> tuple[float, float, float, tuple[int, ...]]:
        expected = 1.0 + row.draft_utility
        predicted = model.predict_ns(row)
        return (-expected / predicted, -expected, predicted, row.selected_raw_ids)

    return min(rows, key=key)


def decision_record(
    row: CandidateScore,
    model: CostModel,
    subset_index: int,
) -> dict[str, object]:

    expected = 1.0 + row.draft_utility
    predicted = model.predict_ns(row)
    return {
        "subset_index": subset_index,
        "selected_raw_ids": list(row.selected_raw_ids),
        "draft_utility": float_record(row.draft_utility),
        "expected_committed_tokens_proxy": float_record(expected),
        "F_atlas": row.F_atlas,
        "N_atlas": row.N_atlas,
        "predicted_cycle_ns": float_record(predicted),
        "expected_committed_tokens_per_second_proxy": float_record(
            expected * 1e9 / predicted
        ),
    }


def _value(decision: Mapping[str, object], field: str) -> float:
    return float(decision[field]["value"])


def summarize_joint_states(
    states: Sequence[Mapping[str, object]],
) -> dict[str, object]:

    totals = {
        policy: {"tokens": 0.0, "time": 0.0} for policy in J1_POLICY_ORDER
    }
    by_question: dict[int, list[Mapping[str, object]]] = {}
    for state in states:
        by_question.setdefault(int(state["question_id"]), []).append(state)
        decisions = state["decisions"]
        for policy in J1_POLICY_ORDER:
            decision = decisions[policy]
            totals[policy]["tokens"] += _value(
                decision, "expected_committed_tokens_proxy"
            )
            totals[policy]["time"] += _value(decision, "predicted_cycle_ns")

    rates = {
        policy: values["tokens"] * 1e9 / values["time"]
        for policy, values in totals.items()
    }
    baseline_policy = max(J1_POLICY_ORDER[:-1], key=rates.__getitem__)
    joint, unique = "residency_joint", "unique_joint"
    gain_best = rates[joint] / rates[baseline_policy] - 1.0
    gain_unique = rates[joint] / rates[unique] - 1.0

    def state_rate(state: Mapping[str, object], policy: str) -> float:
        return _value(
            state["decisions"][policy],
            "expected_committed_tokens_per_second_proxy",
        )

    def prompt_rate(items: Sequence[Mapping[str, object]], policy: str) -> float:
        tokens = sum(
            _value(item["decisions"][policy], "expected_committed_tokens_proxy")
            for item in items
        )
        time = sum(
            _value(item["decisions"][policy], "predicted_cycle_ns")
            for item in items
        )
        return tokens * 1e9 / time

    enter_j2 = gain_best >= J1_PILOT_THRESHOLD or gain_unique >= J1_PILOT_THRESHOLD
    return {
        "state_count": len(states),
        "question_count": len(by_question),
        "policies": {
            policy: {
                "total_expected_committed_tokens_proxy": float_record(
                    values["tokens"]
                ),
                "total_predicted_cycle_ns": float_record(values["time"]),
                "expected_committed_tokens_per_second_proxy": float_record(
                    rates[policy]
                ),
            }
            for policy, values in totals.items()
        },
        "best_non_residency_joint_policy": baseline_policy,
        "residency_joint_gain_vs_best": float_record(gain_best),
        "residency_joint_gain_vs_unique_joint": float_record(gain_unique),
        "residency_joint_vs_unique_decision_difference_count": sum(
            state["decisions"][joint]["selected_raw_ids"]
            != state["decisions"][unique]["selected_raw_ids"]
            for state in states
        ),
        "positive_state_count_vs_best": sum(
            state_rate(state, joint) > state_rate(state, baseline_policy)
            for state in states
        ),
        "positive_state_count_vs_unique_joint": sum(
            state_rate(state, joint) > state_rate(state, unique)
            for state in states
        ),
        "positive_question_count_vs_best": sum(
            prompt_rate(items, joint) > prompt_rate(items, baseline_policy)
            for items in by_question.values()
        ),
        "positive_question_count_vs_unique_joint": sum(
            prompt_rate(items, joint) > prompt_rate(items, unique)
            for items in by_question.values()
        ),
        "gate": {
            "threshold": float_record(J1_PILOT_THRESHOLD),
            "status": "enter_j2" if enter_j2 else "stop_fixed7_gpu_calibration",
            "enter_j2": enter_j2,
        },
    }


__all__ = [
    "J1_PILOT_THRESHOLD",
    "J1_POLICY_ORDER",
    "decision_record",
    "float_record",
    "select_joint_candidate",
    "summarize_joint_states",
]
