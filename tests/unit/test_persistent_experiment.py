from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

from specfit import (
    PERSISTENT_EXPERIMENT_ARTIFACT_TYPE,
    PERSISTENT_EXPERIMENT_SCHEMA_VERSION,
    RuntimeEventKind,
    RuntimeEventTrace,
    canonical_cache_state_sha256,
    canonical_json_sha256,
    canonical_runtime_event_digests,
    stable_trace_id,
    validate_persistent_experiment_artifact,
)


class PersistentExperimentTest(unittest.TestCase):
    def test_runtime_digests_ignore_run_specific_fields(self) -> None:
        def events(run_id: str, host_time_ns: int) -> tuple[RuntimeEventTrace, ...]:
            return (
                RuntimeEventTrace(
                    run_id=run_id,
                    sequence=0,
                    kind=RuntimeEventKind.H2D_SUBMIT,
                    host_time_ns=host_time_ns,
                    layer_id=0,
                    expert_id=7,
                    slot_id=1,
                    generation=1,
                    submission_sequence=0,
                    transfer_id=stable_trace_id("h2d", run_id, 0, 0),
                ),
                RuntimeEventTrace(
                    run_id=run_id,
                    sequence=1,
                    kind=RuntimeEventKind.EVICT,
                    host_time_ns=host_time_ns + 1,
                    layer_id=0,
                    expert_id=8,
                    slot_id=1,
                    generation=2,
                    victim_expert_id=7,
                ),
            )

        self.assertEqual(
            canonical_runtime_event_digests(events("run-a", 10)),
            canonical_runtime_event_digests(events("run-b", 100)),
        )

    def test_cache_state_digest_uses_logical_slot_state(self) -> None:
        slots = (
            SimpleNamespace(
                slot_id=0,
                expert_id=None,
                generation=0,
                last_used_sequence=0,
            ),
            SimpleNamespace(
                slot_id=1,
                expert_id=3,
                generation=2,
                last_used_sequence=4,
            ),
        )
        first = SimpleNamespace(
            lru_sequence=4,
            submission_sequence=2,
            slots=slots,
            host_state_version=10,
        )
        second = SimpleNamespace(
            lru_sequence=4,
            submission_sequence=2,
            slots=slots,
            host_state_version=99,
        )

        self.assertEqual(
            canonical_cache_state_sha256((first,)),
            canonical_cache_state_sha256((second,)),
        )

    def artifact(self) -> dict[str, object]:
        prompts = (
            ("moon", "Tell me about the Moon."),
            ("plants", "Explain photosynthesis."),
        )
        runs: list[dict[str, object]] = []
        for prompt_index, (prompt_id, _) in enumerate(prompts):
            output = [100 + prompt_index, 200 + prompt_index]
            prefill = {
                "h2d_submissions": 1000 + prompt_index,
                "resident_hits": 20,
                "inflight_joins": 0,
                "evictions": 900,
                "pending_use_guards": 0,
            }
            for draft_length in (0, 3):
                acceptance = [] if draft_length == 0 else [1, 0]
                cache_stats = {
                    "h2d_submissions": 1100 + draft_length,
                    "resident_hits": 30 + draft_length,
                    "inflight_joins": 0,
                    "evictions": 950 + draft_length,
                    "pending_use_guards": 0,
                }
                logical_digest = canonical_json_sha256(
                    [prompt_id, draft_length, "logical"]
                )
                final_digest = canonical_json_sha256(
                    [prompt_id, draft_length, "final"]
                )
                for repeat_index in range(2):
                    run_id = f"{prompt_id}-k{draft_length}-r{repeat_index}"
                    runs.append(
                        {
                            "execution_index": len(runs),
                            "prompt_id": prompt_id,
                            "draft_length": draft_length,
                            "repeat_index": repeat_index,
                            "run_id": run_id,
                            "request_id": f"request-{run_id}",
                            "output_token_ids": output.copy(),
                            "accepted_per_cycle": acceptance.copy(),
                            "proposed_draft_tokens": (
                                0 if draft_length == 0 else 6
                            ),
                            "accepted_draft_tokens": sum(acceptance),
                            "target_forward_calls": (
                                2 if draft_length == 3 else 2
                            ),
                            "h2d_submission_order_sha256": logical_digest,
                            "lru_victim_order_sha256": final_digest,
                            "logical_event_sha256": logical_digest,
                            "final_cache_state_sha256": final_digest,
                            "final_cache_state": [
                                prompt_id,
                                draft_length,
                                "final",
                            ],
                            "cache_stats": cache_stats.copy(),
                            "prefill_cache_stats": prefill.copy(),
                            "prefill_event_count": 100,
                            "h2d_submission_count": cache_stats[
                                "h2d_submissions"
                            ],
                            "lru_victim_count": cache_stats["evictions"],
                            "reset": {
                                "layer_count": 2,
                                "empty_slots": 96,
                                "pending_transfers": 0,
                                "active_leases": 0,
                                "pending_use_guards": 0,
                                "host_state_version_sum": 0,
                                "lru_sequence_sum": 0,
                                "submission_sequence_sum": 0,
                                "cache_stats": {
                                    "h2d_submissions": 0,
                                    "resident_hits": 0,
                                    "inflight_joins": 0,
                                    "evictions": 0,
                                    "pending_use_guards": 0,
                                },
                                "observer_event_count": 0,
                                "observer_next_sequence": 0,
                            },
                            "trace": {
                                "path": f"/artifacts/{run_id}.jsonl",
                                "size_bytes": 1024 + repeat_index,
                                "sha256": canonical_json_sha256(run_id),
                                "record_count": 10,
                            },
                            "memory": {
                                "rss_bytes": 10_000 + repeat_index,
                                "cuda_allocated_bytes": 20_000,
                                "cuda_reserved_bytes": 30_000,
                            },
                        }
                    )
        return {
            "schema_version": PERSISTENT_EXPERIMENT_SCHEMA_VERSION,
            "artifact_type": PERSISTENT_EXPERIMENT_ARTIFACT_TYPE,
            "session_id": "r3.5-a100",
            "environment": {
                "git_commit": "abc123",
                "image_id": "specfit:ngc25.11-p0",
                "server": "a100-host",
                "device": "cuda:0",
            },
            "models": {
                "target": {
                    "path": "/models/Qwen3-30B-A3B-Instruct-2507",
                    "revision": "target-sha",
                },
                "drafter": {
                    "path": "/models/Qwen3-30B-A3B-EAGLE3",
                    "revision": "drafter-sha",
                },
            },
            "settings": {
                "draft_lengths": [0, 3],
                "repeats": 2,
                "new_tokens": 2,
                "slots_per_layer": 48,
            },
            "prompts": [
                {
                    "prompt_id": prompt_id,
                    "raw_prompt": raw_prompt,
                    "rendered_prompt": raw_prompt,
                    "input_ids": [1, 2, 3],
                    "attention_mask": [1, 1, 1],
                    "draft_vocabulary": {
                        "supported_output_tokens": 2,
                        "evaluated_output_tokens": 2,
                        "coverage": 1.0,
                    },
                }
                for prompt_id, raw_prompt in prompts
            ],
            "load_counts": {"target": 1, "drafter": 1},
            "memory_after_model_load": {
                "rss_bytes": 10_000,
                "cuda_allocated_bytes": 20_000,
                "cuda_reserved_bytes": 30_000,
            },
            "memory_limits": {"repeat_growth_bytes": 1024},
            "memory_validation": {
                "limit_bytes": 1024,
                "maximum_observed_growth_bytes": 1,
                "comparisons": [
                    {
                        "prompt_id": prompt_id,
                        "draft_length": draft_length,
                        "rss_growth_bytes": 1,
                        "cuda_allocated_growth_bytes": 0,
                    }
                    for prompt_id, _ in prompts
                    for draft_length in (0, 3)
                ],
                "session_growth": {
                    "rss_growth_bytes": 1,
                    "cuda_allocated_growth_bytes": 0,
                },
                "passed": True,
            },
            "runs": runs,
        }

    def test_canonical_json_sha256_ignores_mapping_order(self) -> None:
        first = {"model": "Qwen", "settings": {"k": 3, "slots": 48}}
        second = {"settings": {"slots": 48, "k": 3}, "model": "Qwen"}

        self.assertEqual(
            canonical_json_sha256(first),
            canonical_json_sha256(second),
        )
        self.assertRegex(canonical_json_sha256(first), r"^[0-9a-f]{64}$")

    def test_complete_eight_run_matrix_is_valid(self) -> None:
        self.assertIsNone(
            validate_persistent_experiment_artifact(self.artifact())
        )

    def test_eos_may_end_every_run_before_the_token_budget(self) -> None:
        artifact = self.artifact()
        for run in artifact["runs"]:  # type: ignore[index]
            run["output_token_ids"] = [151645]
            run["stopped_on_eos"] = True
            run["stop_token_id"] = 151645
        for prompt in artifact["prompts"]:  # type: ignore[index]
            prompt["draft_vocabulary"] = {
                "supported_output_tokens": 1,
                "evaluated_output_tokens": 1,
                "coverage": 1.0,
            }

        self.assertIsNone(validate_persistent_experiment_artifact(artifact))

    def test_short_run_requires_a_matching_eos_token(self) -> None:
        artifact = self.artifact()
        run = artifact["runs"][0]  # type: ignore[index]
        run["output_token_ids"] = [100]
        with self.assertRaisesRegex(ValueError, "requires EOS"):
            validate_persistent_experiment_artifact(artifact)

        run["stopped_on_eos"] = True
        run["stop_token_id"] = 999
        with self.assertRaisesRegex(ValueError, "output tail"):
            validate_persistent_experiment_artifact(artifact)

    def test_load_identity_matrix_and_trace_contracts_are_enforced(self) -> None:
        cases: tuple[tuple[str, object, str], ...] = (
            ("environment.image_id", "", "environment.image_id"),
            ("load_counts.target", 2, "load_counts.target"),
            ("settings.draft_lengths", [0, 2], "draft_lengths"),
            ("runs.0.run_id", "moon-k0-r1", "run_id"),
            ("runs.0.trace.path", "/artifacts/moon-k0-r1.jsonl", "trace paths"),
            ("runs.0.trace.size_bytes", 0, "positive"),
            ("runs.0.trace.sha256", "invalid", "SHA256"),
        )
        for dotted_path, value, message in cases:
            with self.subTest(dotted_path=dotted_path):
                artifact = self.artifact()
                target: object = artifact
                parts = dotted_path.split(".")
                for part in parts[:-1]:
                    if isinstance(target, list):
                        target = target[int(part)]
                    else:
                        target = target[part]  # type: ignore[index]
                target[parts[-1]] = value  # type: ignore[index]
                with self.assertRaisesRegex(ValueError, message):
                    validate_persistent_experiment_artifact(artifact)

        artifact = self.artifact()
        artifact["runs"] = artifact["runs"][:-1]  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "eight-run matrix"):
            validate_persistent_experiment_artifact(artifact)

    def test_repeated_runs_must_match_every_deterministic_field(self) -> None:
        replacements: dict[str, object] = {
            "output_token_ids": [999, 999],
            "accepted_per_cycle": [2],
            "proposed_draft_tokens": 999,
            "accepted_draft_tokens": 999,
            "target_forward_calls": 999,
            "h2d_submission_order_sha256": canonical_json_sha256("h2d"),
            "lru_victim_order_sha256": canonical_json_sha256("victim"),
            "final_cache_state_sha256": canonical_json_sha256("changed-final"),
            "cache_stats": {"h2d_submissions": 999},
            "prefill_cache_stats": {"h2d_submissions": 999},
        }
        for field, value in replacements.items():
            with self.subTest(field=field):
                artifact = self.artifact()
                runs = artifact["runs"]  # type: ignore[index]
                runs[1][field] = copy.deepcopy(value)
                with self.assertRaisesRegex(ValueError, field):
                    validate_persistent_experiment_artifact(artifact)

    def test_k0_k3_tokens_and_prompt_prefill_must_match(self) -> None:
        artifact = self.artifact()
        runs = artifact["runs"]  # type: ignore[index]
        runs[2]["output_token_ids"] = [777, 888]
        runs[3]["output_token_ids"] = [777, 888]
        with self.assertRaisesRegex(ValueError, "k=0 and k=3 outputs"):
            validate_persistent_experiment_artifact(artifact)

        artifact = self.artifact()
        runs = artifact["runs"]  # type: ignore[index]
        changed = runs[2]["prefill_cache_stats"].copy()
        changed["h2d_submissions"] = 999
        runs[2]["prefill_cache_stats"] = changed.copy()
        runs[3]["prefill_cache_stats"] = changed.copy()
        with self.assertRaisesRegex(ValueError, "prefill cache stats"):
            validate_persistent_experiment_artifact(artifact)


if __name__ == "__main__":
    unittest.main()
