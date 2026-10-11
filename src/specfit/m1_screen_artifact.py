
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

from .m1_capture_validation import M1CaptureProjection, load_validated_m1_capture
from .m1_screen import (
    M1_RATIO_GREEDY_EPSILON,
    M1_UTILITY_RATIO,
    CandidateScore,
    StateScreen,
    evaluate_gate,
    screen_state,
)

M1_SCREEN_TYPE = "m1_multistate_screen"
M1_SCREEN_CONFIG = {
    "ratio_greedy_epsilon_hex": M1_RATIO_GREEDY_EPSILON.hex(),
    "utility_ratio_hex": M1_UTILITY_RATIO.hex(),
    "gate_ratio_hex": (0.05).hex(),
    "positive_state_minimum": 17,
    "positive_question_minimum": 6,
}
M1_SCREEN_LIMITATIONS = {
    "route_semantics": "full_atlas_proxy",
    "logical_initial_nonresident_only": True,
    "actual_h2d": False,
    "timing": False,
    "accepted_path": False,
    "m2_implemented": False,
}


def _score(score: CandidateScore, *, subset_index: int) -> dict[str, object]:
    return {
        "subset_index": subset_index,
        "selected_raw_ids": list(score.selected_raw_ids),
        "draft_utility": score.draft_utility,
        "draft_utility_hex": score.draft_utility.hex(),
        "F_atlas": score.F_atlas,
        "N_atlas": score.N_atlas,
    }


def _state(value: StateScreen) -> dict[str, object]:
    indices = {row.selected_raw_ids: index for index, row in enumerate(value.rows)}
    decisions = {
        name: {
            "subset_index": indices[score.selected_raw_ids],
            "selected_raw_ids": list(score.selected_raw_ids),
        }
        for name, score in (
            ("ratio_greedy", value.ratio_greedy),
            ("V_F", value.V_F),
            ("V_R", value.V_R),
        )
    }
    return {
        "state_id": value.state_id,
        "question_id": value.question_id,
        "cycle_index": value.cycle_index,
        "ancestor_closed_fixed_budget_subset_count": len(value.rows),
        "candidate_sets": [
            _score(row, subset_index=index)
            for index, row in enumerate(value.rows)
        ],
        "decisions": decisions,
        "pareto_frontier_indices": list(value.pareto_frontier_indices),
        "delta_N_atlas": value.delta_N_atlas,
    }


def _summary(states: tuple[StateScreen, ...]) -> dict[str, object]:
    gate = evaluate_gate(states, validation_passed=True)
    ratio_hex = None if gate.G_atlas is None else gate.G_atlas.hex()
    prompt_delta = {
        str(question_id): sum(
            state.delta_N_atlas for state in states
            if state.question_id == question_id
        )
        for question_id in sorted({state.question_id for state in states})
    }
    return {
        "state_count": len(states),
        "delta_N_atlas": gate.delta_N_atlas,
        "baseline_N_atlas": gate.baseline_N_atlas,
        "G_atlas": gate.G_atlas,
        "G_atlas_hex": ratio_hex,
        "positive_state_count": gate.positive_state_count,
        "positive_question_count": gate.positive_question_count,
        "prompt_delta_N_atlas": prompt_delta,
        "gate": {
            "status": "pass" if gate.passed else "stop",
            "enter_m2": gate.passed,
        },
    }


def build_m1_screen_artifact(
    *, capture_path: Path, workload_path: Path, token_lock_path: Path,
    expected_capture_git_commit: str, git_commit: str,
) -> dict[str, object]:

    if re.fullmatch(r"[0-9a-f]{40}", git_commit or "") is None:
        raise ValueError("screen git_commit must be a full lowercase Git commit")
    capture = load_validated_m1_capture(
        capture_path, workload_path=workload_path,
        token_lock_path=token_lock_path,
        expected_git_commit=expected_capture_git_commit,
    )
    return _build_screen_from_validated_capture(capture, git_commit=git_commit)


def _build_screen_from_validated_capture(
    capture: M1CaptureProjection, *, git_commit: str,
) -> dict[str, object]:
    if re.fullmatch(r"[0-9a-f]{40}", git_commit or "") is None:
        raise ValueError("screen git_commit must be a full lowercase Git commit")
    if capture.source is None:
        raise ValueError("screen build requires file provenance binding")
    states = tuple(screen_state(
        state_id=state.state_id,
        question_id=state.question_id,
        cycle_index=state.cycle_index,
        pool=state.pool,
        atlas=state.atlas,
        frozen=state.frozen,
    ) for state in capture.states)
    basename, size_bytes, sha256 = capture.source
    return {
        "schema_version": 1,
        "artifact_type": M1_SCREEN_TYPE,
        "status": "complete",
        "environment": {"git_commit": git_commit, "execution_device": "cpu"},
        "source_capture": {
            "basename": basename,
            "size_bytes": size_bytes,
            "sha256": sha256,
            "git_commit": capture.git_commit,
        },
        "config": dict(M1_SCREEN_CONFIG),
        "states": [_state(state) for state in states],
        "summary": _summary(states),
        "limitations": dict(M1_SCREEN_LIMITATIONS),
    }


def validate_m1_screen_artifact(
    artifact: Mapping[str, object], *, capture_path: Path,
    workload_path: Path, token_lock_path: Path,
    expected_capture_git_commit: str, expected_git_commit: str,
) -> None:

    if not isinstance(artifact, Mapping):
        raise ValueError("screen artifact must be an object")
    environment = artifact.get("environment")
    if (
        not isinstance(environment, Mapping)
        or environment.get("git_commit") != expected_git_commit
    ):
        raise ValueError("screen git_commit is inconsistent with external expectation")
    expected = build_m1_screen_artifact(
        capture_path=capture_path,
        workload_path=workload_path,
        token_lock_path=token_lock_path,
        expected_capture_git_commit=expected_capture_git_commit,
        git_commit=expected_git_commit,
    )
    _compare_screen_artifact(artifact, expected)


def _validate_screen_from_validated_capture(
    artifact: Mapping[str, object], capture: M1CaptureProjection, *,
    expected_git_commit: str,
) -> None:
    if not isinstance(artifact, Mapping):
        raise ValueError("screen artifact must be an object")
    expected = _build_screen_from_validated_capture(
        capture, git_commit=expected_git_commit,
    )
    _compare_screen_artifact(artifact, expected)


def _compare_screen_artifact(
    artifact: Mapping[str, object], expected: Mapping[str, object],
) -> None:
    try:
        observed_json = json.dumps(
            artifact, ensure_ascii=True, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )
        expected_json = json.dumps(
            expected, ensure_ascii=True, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("screen artifact contains non-serializable or non-finite values") from error
    if observed_json != expected_json:
        raise ValueError("screen artifact cannot be fully rebuilt from the fixed capture")


__all__ = [
    "M1_SCREEN_CONFIG", "M1_SCREEN_LIMITATIONS", "M1_SCREEN_TYPE",
    "build_m1_screen_artifact", "validate_m1_screen_artifact",
]
