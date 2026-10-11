from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from specfit.f0_proxy_screen import build_f0_proxy_artifact
from specfit.m1_screen_artifact import M1_SCREEN_TYPE
from unittest.mock import patch


class F0ProxyScreenTest(unittest.TestCase):
    @staticmethod
    def _state(index: int) -> dict[str, object]:
        rows = [
            {
                "selected_raw_ids": list(range(1, 8)),
                "draft_utility": 2.0,
                "F_atlas": 10,
                "N_atlas": 1,
            },
            {
                "selected_raw_ids": list(range(2, 9)),
                "draft_utility": 1.5,
                "F_atlas": 8,
                "N_atlas": 8,
            },
        ]
        return {
            "state_id": f"q{406 + 10 * (index // 4)}-c{index % 4}",
            "question_id": 406 + 10 * (index // 4),
            "cycle_index": index % 4,
            "ancestor_closed_fixed_budget_subset_count": len(rows),
            "candidate_sets": rows,
            "decisions": {"ratio_greedy": {"selected_raw_ids": list(range(1, 8))}},
        }

    def test_h2d_weight_can_trade_expected_tokens_for_lower_footprint(self) -> None:
        artifact = {
            "artifact_type": M1_SCREEN_TYPE,
            "status": "complete",
            "limitations": {"route_semantics": "full_atlas_proxy"},
            "states": [self._state(index) for index in range(32)],
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "m1.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            raw = path.read_bytes()
            source = {
                "basename": path.name,
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            with (
                patch("specfit.f0_proxy_screen.F0_M1_SCREEN_SOURCE", source),
                patch("specfit.f0_proxy_screen.V1_FIXED7_CANDIDATE_COUNT", 64),
            ):
                result = build_f0_proxy_artifact(
                    m1_screen_path=path,
                    git_commit="a" * 40,
                    fixed_ns=100.0,
                    preloaded_ns_per_expert=10.0,
                    h2d_ns_per_expert_values=(0.0, 50.0),
                )

        no_h2d, with_h2d = result["sensitivity"]
        self.assertEqual(no_h2d["f0_vs_hbm7_decision_difference_count"], 0)
        self.assertEqual(with_h2d["f0_vs_hbm7_decision_difference_count"], 32)
        self.assertEqual(
            with_h2d["comparison_roles"],
            {
                "primary_baseline": "probability7",
                "related_work_reference": "ratio_greedy7",
                "mechanism_ablation": "hbm7",
            },
        )
        self.assertEqual(
            with_h2d["f0_vs_probability7_decision_difference_count"], 32,
        )
        self.assertEqual(with_h2d["lower_tokens_lower_F_state_count"], 32)
        self.assertGreater(with_h2d["f0_gain_vs_probability7"]["value"], 0.0)
        self.assertGreater(with_h2d["f0_gain_vs_hbm7"]["value"], 0.0)
        self.assertFalse(result["limitations"]["reads_residency_or_inflight_state"])
        self.assertNotIn("N_atlas", json.dumps(result))

    def test_rejects_repeated_state_identity(self) -> None:
        states = [self._state(index) for index in range(32)]
        states[-1] = states[0]
        artifact = {
            "artifact_type": M1_SCREEN_TYPE,
            "status": "complete",
            "limitations": {"route_semantics": "full_atlas_proxy"},
            "states": states,
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "m1.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            raw = path.read_bytes()
            source = {
                "basename": path.name,
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            with patch("specfit.f0_proxy_screen.F0_M1_SCREEN_SOURCE", source):
                with self.assertRaises(ValueError):
                    build_f0_proxy_artifact(
                        m1_screen_path=path,
                        git_commit="a" * 40,
                        fixed_ns=100.0,
                        preloaded_ns_per_expert=10.0,
                        h2d_ns_per_expert_values=(50.0,),
                    )


if __name__ == "__main__":
    unittest.main()
