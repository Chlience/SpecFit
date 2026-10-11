from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from specfit.gsm8k_depth5_experiment import (
    GSM8K_DEPTH5_ARTIFACT_TYPE,
    GSM8K_DEPTH5_MODES,
    GSM8K_DEPTH5_SCHEMA_VERSION,
    acceptance_summary,
    aggregate_depth5_runs,
    assert_depth5_resume_compatible,
    max_new_tokens_for_input,
    speculative_mode_order,
    validate_gsm8k_depth5_artifact,
    write_depth5_checkpoint,
)


class GSM8KDepth5ArtifactTest(unittest.TestCase):
    def artifact(self, *, status: str = "complete") -> dict[str, object]:
        runs = [
            self.make_run(mode, index)
            for index, mode in enumerate(GSM8K_DEPTH5_MODES)
        ]
        if status == "running":
            runs = runs[:1]
        artifact: dict[str, object] = {
            "schema_version": GSM8K_DEPTH5_SCHEMA_VERSION,
            "artifact_type": GSM8K_DEPTH5_ARTIFACT_TYPE,
            "session_id": "depth5-test",
            "status": status,
            "dataset": {
                "repo_id": "RedHatAI/speculator_benchmarks",
                "revision": "dataset-rev",
                "filename": "math_reasoning.jsonl",
                "category": "math_reasoning",
            },
            "selected_question_ids": [401],
            "settings": {
                "modes": list(GSM8K_DEPTH5_MODES),
                "batch_size": 1,
                "decoding": "greedy",
                "max_total_tokens": 512,
                "slots_per_layer": 48,
                "timing_scope": "lightweight_summary",
                "cache_start": "cold_per_sample_mode",
                "tree_top_k": 2,
                "tree_max_depth": 5,
                "expert_bytes": 9 * 2**20,
                "numeric_mode": "default",
                "hindsight_oracle": {
                    "scope": "same_candidate_pool_acceptance_upper_bound",
                    "teacher_information_cost_included": False,
                    "optimizes_cross_cycle_cache_externality": False,
                },
            },
            "models": {
                "target": {"path": "/models/qwen3", "revision": "target-rev"},
                "drafter": {"path": "/models/eagle3", "revision": "draft-rev"},
            },
            "environment": {
                "git_commit": "abc123",
                "image_id": "specfit:ngc25.11-p0",
                "server": "a100-host",
                "device": "cuda:0",
                "numeric_settings": {
                    "allow_bf16_reduced_precision_reduction": True,
                },
            },
            "runs": runs,
        }
        artifact["aggregate"] = aggregate_depth5_runs(runs)
        return artifact

    @staticmethod
    def make_run(mode: str, execution_index: int) -> dict[str, object]:
        speculative = mode != "target_only"
        selected = 0
        if mode.endswith("budget7"):
            selected = 7
        elif mode.endswith("budget11"):
            selected = 11
        elif mode.endswith("oracle"):
            selected = 2
        accepted = 0
        if speculative:
            accepted = 2 if mode.endswith("oracle") else 1
        accepted_per_cycle = [accepted] if speculative else []
        actual_depth = 5
        pool_correct_depth = 2
        expert_bytes = 9 * 2**20
        result = {
            "execution_index": execution_index,
            "run_id": f"depth5-test-q401-{mode}",
            "question_id": 401,
            "mode": mode,
            "input_tokens": 32,
            "max_new_tokens": 480,
            "output_token_ids": [100, 101, 151643],
            "generated_tokens": 3,
            "total_tokens": 35,
            "cycles": 1 if speculative else 0,
            "proposed_draft_tokens": selected,
            "accepted_draft_tokens": accepted,
            "accepted_per_cycle": accepted_per_cycle,
            "committed_output_offsets_per_cycle": [0] if speculative else [],
            "tree_pool_nodes_per_cycle": [18] if speculative else [],
            "selected_tree_nodes_per_cycle": [selected] if speculative else [],
            "path_score_calls_per_cycle": [9] if speculative else [],
            "draft_transformer_steps_per_cycle": [29] if speculative else [],
            "actual_tree_depth_per_cycle": (
                [actual_depth] if speculative else []
            ),
            "pool_correct_depth_per_cycle": (
                [pool_correct_depth] if speculative else []
            ),
            "proposal_wall_ns_per_cycle": [100] if speculative else [],
            "selector_wall_ns_per_cycle": [10] if speculative else [],
            "target_verify_wall_ns_per_cycle": [200] if speculative else [],
            "cycle_wall_ns_per_cycle": [350] if speculative else [],
            "selection_regret_per_cycle": (
                [pool_correct_depth - accepted] if speculative else []
            ),
            "proposal_regret_per_cycle": (
                [actual_depth - pool_correct_depth] if speculative else []
            ),
            "acceptance": acceptance_summary(accepted_per_cycle),
            "target_forward_calls": 2 if speculative else 3,
            "verification_positions": (1 + selected) if speculative else 0,
            "stopped_on_eos": True,
            "stop_token_id": 151643,
            "termination_reason": "eos",
            "timings": {
                "prefill_seconds": 1.0,
                "decode_seconds": 2.0,
                "generation_seconds": 3.0,
                "wall_generation_seconds": 3.1,
                "phase_timing_scope": "host_wall_partial",
                "phase_seconds": {
                    "tree_build_seconds": 90 / 1e9 if speculative else 0.0,
                    "selection_seconds": 10 / 1e9 if speculative else 0.0,
                    "target_verification_seconds": (
                        200 / 1e9 if speculative else 0.0
                    ),
                    "tree_cycle_seconds": 350 / 1e9 if speculative else 0.0,
                },
            },
            "cache_stats": {
                "prefill": {
                    "h2d_submissions": 1,
                    "resident_hits": 2,
                    "inflight_joins": 0,
                    "evictions": 0,
                    "pending_use_guards": 0,
                },
                "decode_counter_delta": {
                    "h2d_submissions": 2,
                    "resident_hits": 3,
                    "inflight_joins": 0,
                    "evictions": 1,
                },
                "end": {
                    "h2d_submissions": 3,
                    "resident_hits": 5,
                    "inflight_joins": 0,
                    "evictions": 1,
                    "pending_use_guards": 0,
                },
                "logical_h2d_bytes": {
                    "prefill": expert_bytes,
                    "decode": 2 * expert_bytes,
                    "total": 3 * expert_bytes,
                },
            },
        }
        if mode.endswith("oracle"):
            result["oracle"] = {
                "scope": "same_candidate_pool_acceptance_upper_bound",
                "teacher_information_cost_included": False,
                "optimizes_cross_cycle_cache_externality": False,
                "selection_records": [{}],
            }
        return result

    def test_complete_four_mode_matrix_is_valid(self) -> None:
        self.assertIsNone(validate_gsm8k_depth5_artifact(self.artifact()))

    def test_running_checkpoint_accepts_completed_prefix_only_when_requested(self) -> None:
        artifact = self.artifact(status="running")
        self.assertIsNone(
            validate_gsm8k_depth5_artifact(artifact, allow_incomplete=True)
        )
        with self.assertRaisesRegex(ValueError, "running"):
            validate_gsm8k_depth5_artifact(artifact)

    def test_total_length_budget_and_balanced_mode_order(self) -> None:
        self.assertEqual(max_new_tokens_for_input(35), 477)
        self.assertEqual(max_new_tokens_for_input(124), 388)
        with self.assertRaises(ValueError):
            max_new_tokens_for_input(512)

        first = speculative_mode_order(401)
        second = speculative_mode_order(402)
        third = speculative_mode_order(403)
        self.assertEqual(first, GSM8K_DEPTH5_MODES[1:])
        self.assertEqual(second, first[1:] + first[:1])
        self.assertEqual(third, first[2:] + first[:2])

    def test_acceptance_summary_records_histogram_and_depth_coverage(self) -> None:
        self.assertEqual(
            acceptance_summary([0, 1, 3, 5]),
            {
                "histogram_0_to_5": [1, 1, 0, 1, 0, 1],
                "reached_depth_counts": [3, 2, 2, 1, 1],
            },
        )

    def test_aggregate_includes_depth_regret_work_and_cycle_timings(self) -> None:
        artifact = self.artifact()
        probability = artifact["aggregate"][GSM8K_DEPTH5_MODES[1]]
        self.assertEqual(
            probability["cycle_work_totals"],
            {
                "actual_tree_depth": 5,
                "pool_correct_depth": 2,
                "path_score_calls": 9,
                "draft_transformer_steps": 29,
                "proposal_wall_ns": 100,
                "selector_wall_ns": 10,
                "target_verify_wall_ns": 200,
                "cycle_wall_ns": 350,
                "selection_regret": 1,
                "proposal_regret": 3,
            },
        )
        self.assertEqual(probability["mean_selection_regret"], 1.0)
        self.assertEqual(probability["mean_proposal_regret"], 3.0)
        target = artifact["aggregate"]["target_only"]
        self.assertTrue(
            all(value == 0 for value in target["cycle_work_totals"].values())
        )

    def test_validates_cycle_depth_work_regret_and_timing_constraints(self) -> None:
        cases: list[tuple[dict[str, object], str]] = []

        artifact = self.artifact()
        artifact["runs"][1]["actual_tree_depth_per_cycle"] = []
        cases.append((artifact, "per-cycle array length"))

        artifact = self.artifact()
        run = artifact["runs"][1]
        run["selected_tree_nodes_per_cycle"] = [19]
        run["proposed_draft_tokens"] = 19
        run["verification_positions"] = 20
        cases.append((artifact, "exceed candidate pool"))

        artifact = self.artifact()
        run = artifact["runs"][3]
        run["selected_tree_nodes_per_cycle"] = [1]
        run["proposed_draft_tokens"] = 1
        run["verification_positions"] = 2
        cases.append((artifact, "cannot exceed the number of selected nodes"))

        artifact = self.artifact()
        artifact["runs"][1]["pool_correct_depth_per_cycle"] = [6]
        cases.append((artifact, "cannot exceed actual_tree_depth"))

        artifact = self.artifact()
        artifact["runs"][1]["selector_wall_ns_per_cycle"] = [101]
        cases.append((artifact, "selector <= proposal <= cycle"))

        artifact = self.artifact()
        artifact["runs"][1]["proposal_wall_ns_per_cycle"] = [351]
        cases.append((artifact, "selector <= proposal <= cycle"))

        artifact = self.artifact()
        artifact["runs"][1]["target_verify_wall_ns_per_cycle"] = [351]
        cases.append((artifact, "target_verify"))

        artifact = self.artifact()
        artifact["runs"][1]["selection_regret_per_cycle"] = [0]
        cases.append((artifact, "pool_correct - accepted"))

        artifact = self.artifact()
        artifact["runs"][1]["proposal_regret_per_cycle"] = [2]
        cases.append((artifact, "actual_depth - pool_correct"))

        artifact = self.artifact()
        run = artifact["runs"][3]
        run["pool_correct_depth_per_cycle"] = [3]
        run["selection_regret_per_cycle"] = [1]
        run["proposal_regret_per_cycle"] = [2]
        cases.append((artifact, "accepted == pool_correct"))

        artifact = self.artifact()
        artifact["runs"][0]["actual_tree_depth_per_cycle"] = [1]
        cases.append((artifact, "target_only"))

        for candidate, message in cases:
            with self.subTest(message=message):
                candidate["aggregate"] = aggregate_depth5_runs(
                    candidate["runs"]
                )
                with self.assertRaisesRegex(ValueError, message):
                    validate_gsm8k_depth5_artifact(candidate)

    def test_rejects_wrong_output_h2d_bytes_and_tree_budget(self) -> None:
        cases = []
        artifact = self.artifact()
        artifact["runs"][1]["output_token_ids"][0] = 999
        cases.append((artifact, "are inconsistent"))

        artifact = self.artifact()
        artifact["runs"][1]["cache_stats"]["logical_h2d_bytes"]["total"] += 1
        cases.append((artifact, "inconsistent with the commit count"))

        artifact = self.artifact()
        artifact["runs"][1]["selected_tree_nodes_per_cycle"] = [8]
        artifact["runs"][1]["proposed_draft_tokens"] = 8
        cases.append((artifact, "exceed the budget 7"))

        for candidate, message in cases:
            with self.subTest(message=message):
                candidate["aggregate"] = aggregate_depth5_runs(candidate["runs"])
                with self.assertRaisesRegex(ValueError, message):
                    validate_gsm8k_depth5_artifact(candidate)

    def test_speculative_run_requires_target_only_first(self) -> None:
        artifact = self.artifact(status="running")
        artifact["runs"] = [self.make_run(GSM8K_DEPTH5_MODES[1], 0)]
        artifact["aggregate"] = aggregate_depth5_runs(artifact["runs"])
        with self.assertRaises(ValueError):
            validate_gsm8k_depth5_artifact(artifact, allow_incomplete=True)

    def test_hindsight_oracle_allows_seed_only_cycle(self) -> None:
        artifact = self.artifact()
        oracle = artifact["runs"][3]
        oracle["selected_tree_nodes_per_cycle"] = [0]
        oracle["proposed_draft_tokens"] = 0
        oracle["accepted_per_cycle"] = [0]
        oracle["accepted_draft_tokens"] = 0
        oracle["pool_correct_depth_per_cycle"] = [0]
        oracle["selection_regret_per_cycle"] = [0]
        oracle["proposal_regret_per_cycle"] = [5]
        oracle["verification_positions"] = 1
        oracle["acceptance"] = acceptance_summary([0])
        artifact["aggregate"] = aggregate_depth5_runs(artifact["runs"])
        self.assertIsNone(validate_gsm8k_depth5_artifact(artifact))

    def test_resume_identity_must_match(self) -> None:
        existing = self.artifact(status="running")
        expected = copy.deepcopy(existing)
        assert_depth5_resume_compatible(existing, expected)
        expected["settings"]["slots_per_layer"] = 32
        with self.assertRaisesRegex(ValueError, "settings"):
            assert_depth5_resume_compatible(existing, expected)

    def test_numeric_mode_is_part_of_the_experiment_identity(self) -> None:
        artifact = self.artifact()
        artifact["settings"]["numeric_mode"] = "math_sdpa"
        artifact["environment"]["numeric_settings"] = {
            "forced_sdpa_backend": "math",
        }
        artifact["aggregate"] = aggregate_depth5_runs(artifact["runs"])
        self.assertIsNone(validate_gsm8k_depth5_artifact(artifact))

        artifact["settings"]["numeric_mode"] = "math_ieee"
        artifact["environment"]["numeric_settings"] = {
            "forced_sdpa_backend": "math",
            "cuda_matmul_fp32_precision": "ieee",
        }
        self.assertIsNone(validate_gsm8k_depth5_artifact(artifact))

        artifact["settings"]["numeric_mode"] = (
            "math_bf16_full_reduction"
        )
        artifact["environment"]["numeric_settings"] = {
            "forced_sdpa_backend": "math",
            "allow_bf16_reduced_precision_reduction": False,
        }
        self.assertIsNone(validate_gsm8k_depth5_artifact(artifact))

        artifact["settings"]["numeric_mode"] = "unknown"
        with self.assertRaisesRegex(ValueError, "numeric_mode"):
            validate_gsm8k_depth5_artifact(artifact)

    def test_checkpoint_write_replaces_complete_json_document(self) -> None:
        artifact = self.artifact(status="running")
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "checkpoint.json"
            write_depth5_checkpoint(path, artifact)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded, artifact)
            self.assertFalse((path.parent / ".checkpoint.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
