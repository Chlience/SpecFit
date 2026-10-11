from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from specfit.j1_calibration import (
    CalibrationPoint,
    CostModel,
    fit_cost_models,
    read_g0_calibration_points,
)
from specfit.j1_screen import (
    select_joint_candidate,
    summarize_joint_states,
)
from specfit.m1_screen import CandidateScore


class J1ScreenTest(unittest.TestCase):
    @staticmethod
    def _state(question_id: int, joint: tuple[float, float],
               baseline: tuple[float, float]) -> dict[str, object]:
        def decision(values: tuple[float, float]) -> dict[str, object]:
            tokens, time = values
            return {
                "selected_raw_ids": [1],
                "expected_committed_tokens_proxy": {"value": tokens},
                "predicted_cycle_ns": {"value": time},
                "expected_committed_tokens_per_second_proxy": {
                    "value": tokens * 1e9 / time
                },
            }

        return {
            "question_id": question_id,
            "decisions": {
                "probability": decision(baseline),
                "ratio_greedy": decision(baseline),
                "unique_joint": decision(baseline),
                "residency_joint": decision(joint),
            },
        }

    def test_summary_uses_ratio_of_sums_instead_of_mean_state_rates(self) -> None:
        summary = summarize_joint_states((
            self._state(406, (100.0, 100.0), (2.0, 1.0)),
            self._state(416, (1.0, 100.0), (1.0, 1_000.0)),
        ))

        self.assertGreater(
            summary["residency_joint_gain_vs_best"]["value"], 0.0
        )

    def test_selector_accepts_lower_expected_tokens_when_time_falls_more(self) -> None:
        high_acceptance = CandidateScore((1,), 2.0, 10, 8)
        low_cost = CandidateScore((2,), 1.5, 8, 0)
        full = CostModel(fixed_ns=100.0, expert_ns=10.0, h2d_ns=50.0)
        unique = CostModel(fixed_ns=100.0, expert_ns=10.0, h2d_ns=0.0)

        self.assertEqual(
            select_joint_candidate((high_acceptance, low_cost), full),
            low_cost,
        )
        self.assertEqual(
            select_joint_candidate((high_acceptance, low_cost), unique),
            high_acceptance,
        )

    def test_three_points_recover_full_model_and_fit_unique_control(self) -> None:
        points = tuple(
            CalibrationPoint(name, footprint, nonresident,
                200 + 3 * footprint + 7 * nonresident)
            for name, footprint, nonresident in (
                ("probability", 10, 7),
                ("unique_efficiency", 8, 4),
                ("joint_hindsight", 5, 3),
            )
        )

        full, unique = fit_cost_models(points)

        self.assertEqual((full.fixed_ns, full.expert_ns, full.h2d_ns),
                         (200.0, 3.0, 7.0))
        self.assertGreater(unique.fixed_ns, 0.0)
        self.assertGreater(unique.expert_ns, 0.0)
        self.assertEqual(unique.h2d_ns, 0.0)

    def test_g0_reader_binds_bytes_and_uses_resident_first_medians(self) -> None:
        policies = ("probability", "unique_efficiency", "joint_hindsight")
        costs = {
            "probability": (100, 60, 1_000),
            "unique_efficiency": (80, 40, 900),
            "joint_hindsight": (60, 25, 800),
        }
        pairs = []
        for policy in policies:
            footprint, nonresident, base = costs[policy]
            for repetition in range(5):
                resident = {
                    "selection_policy": policy,
                    "execution_policy": "resident_first",
                    "phase": "formal",
                    "cache_counter_delta": {
                        "unique_expert_acquisitions": footprint,
                        "h2d_submissions": nonresident,
                        "resident_hits": footprint - nonresident,
                        "inflight_joins": 0,
                    },
                    "timings": {"cycle_wall_ns": base + repetition},
                    "schedule": {
                        "resident_first_exact": True,
                        "initial_resident_reload_h2d": 0,
                    },
                }
                pairs.append({
                    "selection_policy": policy,
                    "repetition_index": repetition,
                    "runs": [{"execution_policy": "expert_id_ascending"}, resident],
                })
        artifact = {
            "artifact_type": "g0_resident_first_q480",
            "status": "complete",
            "environment": {"git_commit": "a" * 40},
            "config": {
                "selection_order": list(policies),
                "formal_repetitions": 5,
                "execution_policy_order": ["expert_id_ascending", "resident_first"],
                "draft_node_budget": 7,
            },
            "formal_pairs": pairs,
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "g0.json"
            raw = json.dumps(artifact, sort_keys=True).encode()
            path.write_bytes(raw)
            descriptor = {
                "basename": path.name,
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "git_commit": "a" * 40,
            }
            with patch("specfit.j1_calibration.J1_G0_SOURCE", descriptor):
                points = read_g0_calibration_points(path)
                self.assertEqual(
                    tuple((point.expert_acquisitions, point.h2d_submissions,
                           point.median_cycle_wall_ns) for point in points),
                    ((100, 60, 1_002), (80, 40, 902), (60, 25, 802)),
                )
                path.write_bytes(raw + b"\n")
                with self.assertRaisesRegex(ValueError, "size/SHA-256"):
                    read_g0_calibration_points(path)


if __name__ == "__main__":
    unittest.main()
