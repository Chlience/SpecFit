
from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import subprocess

from .j1_calibration import CostModel
from .j1_screen import float_record, select_joint_candidate
from .m1_screen import CandidateScore, M1_QUESTION_IDS
from .m1_screen_artifact import M1_SCREEN_TYPE
from .v1_screen import V1_FIXED7_CANDIDATE_COUNT


F0_PROXY_ARTIFACT_TYPE = "f0_cold_cache_route_proxy_sensitivity"
F0_POLICY_ORDER = ("probability7", "ratio_greedy7", "hbm7", "f0")
F0_M1_SCREEN_SOURCE = {
    "basename": "m1-multistate-screen-v2-e3bf7a2.json",
    "size_bytes": 1_454_043,
    "sha256": "1227f6670d19132c360f5b7d40a3b0bb9088380e0ae3118586948d36ae763e20",
}


def _rows(state: Mapping[str, object]) -> tuple[CandidateScore, ...]:
    raw_rows = state.get("candidate_sets")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise ValueError("F0 state missing candidate set")
    rows = []
    for value in raw_rows:
        if not isinstance(value, Mapping):
            raise ValueError("F0 candidate must be an object")
        selected = value.get("selected_raw_ids")
        utility = value.get("draft_utility")
        footprint = value.get("F_atlas")
        if (
            not isinstance(selected, Sequence)
            or isinstance(selected, (str, bytes))
            or len(selected) != 7
            or any(isinstance(item, bool) or not isinstance(item, int) for item in selected)
            or tuple(selected) != tuple(sorted(set(selected)))
            or isinstance(utility, bool)
            or not isinstance(utility, (int, float))
            or not math.isfinite(utility)
            or utility < 0.0
            or isinstance(footprint, bool)
            or not isinstance(footprint, int)
            or footprint <= 0
        ):
            raise ValueError("F0 candidate does not satisfy fixed-7 C/F contract")
        rows.append(CandidateScore(tuple(selected), float(utility), footprint, footprint))
    expected_count = state.get("ancestor_closed_fixed_budget_subset_count")
    if expected_count != len(rows) or len({row.selected_raw_ids for row in rows}) != len(rows):
        raise ValueError("F0 candidate count or selected_raw_ids mismatch")
    return tuple(rows)


def _ratio_greedy(state: Mapping[str, object], rows: Sequence[CandidateScore]) -> CandidateScore:
    decisions = state.get("decisions")
    if not isinstance(decisions, Mapping) or not isinstance(decisions.get("ratio_greedy"), Mapping):
        raise ValueError("F0 state is missing ratio-greedy decision")
    selected = decisions["ratio_greedy"].get("selected_raw_ids")
    wanted = tuple(selected) if isinstance(selected, Sequence) else ()
    matches = tuple(row for row in rows if row.selected_raw_ids == wanted)
    if len(matches) != 1:
        raise ValueError("F0 cannot locate ratio-greedy decision in candidate set")
    return matches[0]


def _totals(
    choices: Sequence[CandidateScore], model: CostModel,
) -> tuple[float, float, float]:
    tokens = sum(1.0 + row.draft_utility for row in choices)
    cycle_ns = sum(model.predict_ns(row) for row in choices)
    return tokens, cycle_ns, tokens * 1e9 / cycle_ns


def _gain(left: float, right: float) -> float:
    return left / right - 1.0


def _analyze_slope(
    states: Sequence[Mapping[str, object]],
    *,
    fixed_ns: float,
    preloaded_ns_per_expert: float,
    h2d_ns_per_expert: float,
) -> dict[str, object]:
    preloaded = CostModel(fixed_ns, preloaded_ns_per_expert, 0.0)
    cold = CostModel(
        fixed_ns, preloaded_ns_per_expert + h2d_ns_per_expert, 0.0,
    )
    choices = {policy: [] for policy in F0_POLICY_ORDER}
    identities = []
    for state in states:
        rows = _rows(state)
        probability = min(rows, key=lambda row: (-row.draft_utility, row.selected_raw_ids))
        ratio_greedy = _ratio_greedy(state, rows)
        hbm = select_joint_candidate(rows, preloaded)
        f0 = select_joint_candidate(rows, cold)
        for policy, row in zip(F0_POLICY_ORDER, (probability, ratio_greedy, hbm, f0), strict=True):
            choices[policy].append(row)
        identities.append((str(state.get("state_id")), int(state.get("question_id"))))

    totals = {policy: _totals(rows, cold) for policy, rows in choices.items()}
    rates = {policy: values[2] for policy, values in totals.items()}
    best_baseline = max(F0_POLICY_ORDER[:-1], key=rates.__getitem__)
    differing = []
    for index, (hbm, f0) in enumerate(zip(choices["hbm7"], choices["f0"], strict=True)):
        if hbm.selected_raw_ids == f0.selected_raw_ids:
            continue
        state_id, question_id = identities[index]
        differing.append({
            "state_id": state_id,
            "question_id": question_id,
            "hbm7_selected_raw_ids": list(hbm.selected_raw_ids),
            "f0_selected_raw_ids": list(f0.selected_raw_ids),
            "hbm7_expected_committed_tokens_proxy": float_record(1.0 + hbm.draft_utility),
            "f0_expected_committed_tokens_proxy": float_record(1.0 + f0.draft_utility),
            "hbm7_F_atlas": hbm.F_atlas,
            "f0_F_atlas": f0.F_atlas,
        })

    questions = sorted({question_id for _, question_id in identities})

    def prompt_rate(policy: str, question_id: int) -> float:
        subset = [
            row for row, identity in zip(choices[policy], identities, strict=True)
            if identity[1] == question_id
        ]
        return _totals(subset, cold)[2]

    return {
        "h2d_ns_per_expert": float_record(h2d_ns_per_expert),
        "cold_ns_per_expert": float_record(
            preloaded_ns_per_expert + h2d_ns_per_expert
        ),
        "policies": {
            policy: {
                "total_expected_committed_tokens_proxy": float_record(values[0]),
                "total_predicted_cycle_ns": float_record(values[1]),
                "expected_committed_tokens_per_second_proxy": float_record(values[2]),
                "total_F_atlas": sum(row.F_atlas for row in choices[policy]),
            }
            for policy, values in totals.items()
        },
        "comparison_roles": {
            "primary_baseline": "probability7",
            "related_work_reference": "ratio_greedy7",
            "mechanism_ablation": "hbm7",
        },
        "best_non_f0_policy": best_baseline,
        "f0_gain_vs_best_non_f0": float_record(_gain(rates["f0"], rates[best_baseline])),
        "f0_gain_vs_probability7": float_record(
            _gain(rates["f0"], rates["probability7"])
        ),
        "f0_gain_vs_ratio_greedy7": float_record(_gain(rates["f0"], rates["ratio_greedy7"])),
        "f0_gain_vs_hbm7": float_record(_gain(rates["f0"], rates["hbm7"])),
        "f0_vs_probability7_decision_difference_count": sum(
            f0.selected_raw_ids != probability.selected_raw_ids
            for probability, f0 in zip(
                choices["probability7"], choices["f0"], strict=True,
            )
        ),
        "f0_vs_probability7_total_F_atlas_delta": (
            sum(row.F_atlas for row in choices["f0"])
            - sum(row.F_atlas for row in choices["probability7"])
        ),
        "f0_vs_probability7_expected_committed_tokens_proxy_delta": float_record(
            totals["f0"][0] - totals["probability7"][0]
        ),
        "positive_state_count_vs_probability7": sum(
            (1.0 + f0.draft_utility) / cold.predict_ns(f0)
            > (1.0 + probability.draft_utility) / cold.predict_ns(probability)
            for probability, f0 in zip(
                choices["probability7"], choices["f0"], strict=True,
            )
        ),
        "positive_question_count_vs_probability7": sum(
            prompt_rate("f0", question_id)
            > prompt_rate("probability7", question_id)
            for question_id in questions
        ),
        "f0_vs_hbm7_decision_difference_count": len(differing),
        "f0_vs_hbm7_total_F_atlas_delta": (
            sum(row.F_atlas for row in choices["f0"])
            - sum(row.F_atlas for row in choices["hbm7"])
        ),
        "f0_vs_hbm7_expected_committed_tokens_proxy_delta": float_record(
            totals["f0"][0] - totals["hbm7"][0]
        ),
        "lower_tokens_lower_F_state_count": sum(
            f0.draft_utility < hbm.draft_utility and f0.F_atlas < hbm.F_atlas
            for hbm, f0 in zip(choices["hbm7"], choices["f0"], strict=True)
        ),
        "positive_state_count_vs_hbm7": sum(
            (1.0 + f0.draft_utility) / cold.predict_ns(f0)
            > (1.0 + hbm.draft_utility) / cold.predict_ns(hbm)
            for hbm, f0 in zip(choices["hbm7"], choices["f0"], strict=True)
        ),
        "positive_question_count_vs_hbm7": sum(
            prompt_rate("f0", question_id) > prompt_rate("hbm7", question_id)
            for question_id in questions
        ),
        "differing_states": differing,
    }


def build_f0_proxy_artifact(
    *,
    m1_screen_path: Path,
    git_commit: str,
    fixed_ns: float,
    preloaded_ns_per_expert: float,
    h2d_ns_per_expert_values: Sequence[float],
) -> dict[str, object]:

    if m1_screen_path.is_symlink() or not m1_screen_path.is_file():
        raise ValueError("F0 M1 screen must be a regular file")
    raw = m1_screen_path.read_bytes()
    source_descriptor = {
        "basename": m1_screen_path.name,
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if source_descriptor != F0_M1_SCREEN_SOURCE:
        raise ValueError("F0 M1 screen file identity does not match preregistered contract")
    try:
        source = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("F0 input is not valid JSON") from error
    if not isinstance(source, Mapping):
        raise ValueError("F0 input root node must be an object")
    states = source.get("states")
    limitations = source.get("limitations")
    if (
        source.get("artifact_type") != M1_SCREEN_TYPE
        or source.get("status") != "complete"
        or not isinstance(limitations, Mapping)
        or limitations.get("route_semantics") != "full_atlas_proxy"
        or not isinstance(states, Sequence)
        or len(states) != 32
    ):
        raise ValueError("F0 input does not satisfy M1 full-atlas proxy contract")
    if any(not isinstance(state, Mapping) for state in states):
        raise ValueError("F0 state must be an object")
    grouped: dict[int, list[int]] = {}
    state_ids = set()
    for state in states:
        question_id = state.get("question_id")
        cycle_index = state.get("cycle_index")
        state_id = state.get("state_id")
        if (
            isinstance(question_id, bool)
            or not isinstance(question_id, int)
            or isinstance(cycle_index, bool)
            or not isinstance(cycle_index, int)
            or cycle_index < 0
            or state_id != f"q{question_id}-c{cycle_index}"
            or state_id in state_ids
        ):
            raise ValueError("F0 state identity does not match preregistered contract")
        state_ids.add(state_id)
        grouped.setdefault(question_id, []).append(cycle_index)
    if (
        tuple(sorted(grouped)) != M1_QUESTION_IDS
        or any(len(values) != 4 or len(set(values)) != 4 for values in grouped.values())
    ):
        raise ValueError("F0 state does not match the 8 prompt x 4 offset distribution")
    candidate_count = sum(len(_rows(state)) for state in states)
    if candidate_count != V1_FIXED7_CANDIDATE_COUNT:
        raise ValueError("F0 does not reproduce the pre-registered 3,991 fixed-7 candidates")
    slopes = tuple(sorted(set(float(value) for value in h2d_ns_per_expert_values)))
    if (
        len(git_commit) != 40
        or any(character not in "0123456789abcdef" for character in git_commit)
        or not math.isfinite(fixed_ns)
        or fixed_ns <= 0.0
        or not math.isfinite(preloaded_ns_per_expert)
        or preloaded_ns_per_expert <= 0.0
        or not slopes
        or any(not math.isfinite(value) or value < 0.0 for value in slopes)
    ):
        raise ValueError("F0 Git identity or time coefficient does not satisfy contract")
    return {
        "schema_version": 1,
        "artifact_type": F0_PROXY_ARTIFACT_TYPE,
        "status": "diagnostic_complete",
        "environment": {"git_commit": git_commit, "execution_device": "cpu"},
        "source_m1_screen": dict(source_descriptor),
        "config": {
            "draft_node_budget": 7,
            "state_count": 32,
            "candidate_count": candidate_count,
            "route_semantics": "target_observed_full_atlas_proxy",
            "expected_committed_tokens_formula": "1 + draft_utility",
            "preloaded_time_formula": "fixed_ns + preloaded_ns_per_expert * F_atlas",
            "cold_time_formula": (
                "fixed_ns + (preloaded_ns_per_expert + h2d_ns_per_expert) * F_atlas"
            ),
            "h2d_intercept_ns": float_record(0.0),
            "fixed_ns": float_record(fixed_ns),
            "preloaded_ns_per_expert": float_record(preloaded_ns_per_expert),
            "policy_order": list(F0_POLICY_ORDER),
        },
        "sensitivity": [
            _analyze_slope(
                states,
                fixed_ns=fixed_ns,
                preloaded_ns_per_expert=preloaded_ns_per_expert,
                h2d_ns_per_expert=value,
            )
            for value in slopes
        ],
        "limitations": {
            "provisional_g0_coefficient_decomposition": True,
            "actual_acceptance": False,
            "actual_h2d": False,
            "actual_own_routes": False,
            "selector_runtime_included": False,
            "predictor_runtime_included": False,
            "formal_a1_calibration": False,
            "reads_residency_or_inflight_state": False,
            "per_layer_capacity_eviction_or_reload_modeled": False,
        },
    }


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--m1-screen", type=Path, required=True)
    parser.add_argument("--fixed-ns", type=float, required=True)
    parser.add_argument("--preloaded-ns-per-expert", type=float, required=True)
    parser.add_argument("--h2d-ns-per-expert", type=float, nargs="+", required=True)
    parser.add_argument("--artifact-output", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout
    if dirty:
        raise RuntimeError("F0 diagnostics require a clean working tree")
    output = args.artifact_output.resolve()
    if output.exists():
        raise FileExistsError(f"artifact already exists: {output}")
    artifact = build_f0_proxy_artifact(
        m1_screen_path=args.m1_screen,
        git_commit=head,
        fixed_ns=args.fixed_ns,
        preloaded_ns_per_expert=args.preloaded_ns_per_expert,
        h2d_ns_per_expert_values=args.h2d_ns_per_expert,
    )
    encoded = (json.dumps(
        artifact, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(encoded)
    print(json.dumps({
        "artifact": str(output),
        "size_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "sensitivity": [{
            "h2d_ns_per_expert": item["h2d_ns_per_expert"]["value"],
            "difference_count_vs_probability7": (
                item["f0_vs_probability7_decision_difference_count"]
            ),
            "gain_vs_probability7": item["f0_gain_vs_probability7"]["value"],
            "gain_vs_ratio_greedy7": item["f0_gain_vs_ratio_greedy7"]["value"],
            "gain_vs_hbm7": item["f0_gain_vs_hbm7"]["value"],
        } for item in artifact["sensitivity"]],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "F0_M1_SCREEN_SOURCE", "F0_PROXY_ARTIFACT_TYPE", "build_f0_proxy_artifact",
]
