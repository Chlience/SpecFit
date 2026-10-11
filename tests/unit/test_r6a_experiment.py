from __future__ import annotations

import unittest

from specfit.r6a_experiment import validate_r6a_artifact

from tests.unit.r6a_fixture import R6AFixture


class R6AArtifactTest(unittest.TestCase):
    def test_accepts_complete_frozen_cycle_artifact(self) -> None:
        validate_r6a_artifact(R6AFixture.artifact())

    def test_accepts_non104_topology_and_rejects_count_tampering(self) -> None:
        artifact = R6AFixture.artifact(R6AFixture.alternate_parents)
        subset_count = artifact["offline_selection"][
            "ancestor_closed_fixed_budget_subset_count"
        ]
        self.assertEqual(subset_count, 133)
        validate_r6a_artifact(artifact)
        artifact["offline_selection"][
            "ancestor_closed_fixed_budget_subset_count"
        ] += 1
        with self.assertRaises(ValueError):
            validate_r6a_artifact(artifact)
        artifact = R6AFixture.artifact(R6AFixture.alternate_parents)
        artifact["offline_selection"][
            "ancestor_closed_fixed_budget_subset_count"
        ] = 133.0
        with self.assertRaises(ValueError):
            validate_r6a_artifact(artifact)

    def test_pool_fingerprint_matches_runtime_contract(self) -> None:
        try:
            from specfit.runtime.eagle3_selection import (
                candidate_pool_fingerprint,
            )
            from specfit.runtime.eagle3_types import (
                Eagle3DraftCandidate,
                Eagle3DraftCandidatePool,
            )
        except ModuleNotFoundError as error:
            self.skipTest(f"runtime dependency unavailable: {error}")
        artifact_pool = R6AFixture.pool()
        candidates = tuple(
            Eagle3DraftCandidate(
                **{
                    **candidate,
                    "path": tuple(candidate["path"]),
                }
            )
            for candidate in artifact_pool["candidates"]
        )
        runtime_pool = Eagle3DraftCandidatePool(
            seed_token_id=42,
            top_k=2,
            max_depth=5,
            candidates=candidates,
            path_score_calls=9,
            draft_transformer_steps=29,
        )
        self.assertEqual(
            artifact_pool["fingerprint"],
            candidate_pool_fingerprint(runtime_pool),
        )

    def test_rejects_provenance_or_numeric_identity_drift(self) -> None:
        mutations = (
            ("dataset question", lambda item: item["dataset"].update(question_id=479)),
            ("prompt digest", lambda item: item["prompt"].update(input_ids_sha256="x")),
            (
                "model revision",
                lambda item: item["models"]["target"].update(revision=""),
            ),
            ("dtype", lambda item: item["config"].update(dtype="float16")),
            (
                "numeric mode",
                lambda item: item["config"].update(numeric_mode="default"),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                artifact = R6AFixture.artifact()
                mutate(artifact)
                with self.assertRaises(ValueError):
                    validate_r6a_artifact(artifact)

    def test_rejects_pool_topology_or_fingerprint_drift(self) -> None:
        artifact = R6AFixture.artifact()
        artifact["pool"]["candidates"][6]["parent_raw_id"] = 4
        with self.assertRaisesRegex(ValueError, "topology"):
            validate_r6a_artifact(artifact)
        artifact = R6AFixture.artifact()
        artifact["pool"]["fingerprint"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_r6a_artifact(artifact)

    def test_rejects_policy_set_or_recomputed_score_drift(self) -> None:
        artifact = R6AFixture.artifact()
        artifact["offline_selection"]["decisions"].pop()
        with self.assertRaises(ValueError):
            validate_r6a_artifact(artifact)
        artifact = R6AFixture.artifact()
        decision = artifact["offline_selection"]["decisions"][0]
        decision["unique_expert_count"] += 1
        with self.assertRaises(ValueError):
            validate_r6a_artifact(artifact)
        artifact = R6AFixture.artifact()
        artifact["trials"].pop()
        with self.assertRaises(ValueError):
            validate_r6a_artifact(artifact)

    def test_rejects_mapping_or_route_drift(self) -> None:
        artifact = R6AFixture.artifact()
        mapping = artifact["trials"][0]["selection"]
        last_raw_id = str(mapping["selected_raw_ids"][-1])
        mapping["raw_to_verification_position"][last_raw_id] = 1
        with self.assertRaisesRegex(ValueError, "verification position"):
            validate_r6a_artifact(artifact)
        artifact = R6AFixture.artifact()
        artifact["trials"][0]["routes"]["rows"].pop()
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_r6a_artifact(artifact)
        artifact = R6AFixture.artifact()
        artifact["trials"][0]["routes"]["digest"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "digest"):
            validate_r6a_artifact(artifact)

    def test_rejects_atlas_or_trial_counter_nonclosure(self) -> None:
        for run_key in ("route_atlas_probe", "trials"):
            with self.subTest(run_key=run_key):
                artifact = R6AFixture.artifact()
                run = (
                    artifact["route_atlas_probe"]
                    if run_key == "route_atlas_probe"
                    else artifact["trials"][0]
                )
                run["cache_counter_delta"]["resident_hits"] += 1
                with self.assertRaises(ValueError):
                    validate_r6a_artifact(artifact)
        artifact = R6AFixture.artifact()
        counters = artifact["trials"][0]["cache_counter_delta"]
        counters["evictions"] = counters["h2d_submissions"] + 1
        with self.assertRaises(ValueError):
            validate_r6a_artifact(artifact)

    def test_rejects_reported_cache_volume_tampering(self) -> None:
        fields = (
            "unique_expert_acquisitions",
            "expert_bytes",
            "logical_h2d_bytes",
        )
        for field in fields:
            with self.subTest(field=field):
                artifact = R6AFixture.artifact()
                counters = artifact["trials"][0]["cache_counter_delta"]
                counters[field] += 1
                with self.assertRaises(ValueError):
                    validate_r6a_artifact(artifact)

    def test_rejects_shortened_acceptance_despite_valid_prefix(self) -> None:
        artifact = R6AFixture.artifact()
        acceptance = artifact["trials"][0]["acceptance"]
        self.assertGreater(acceptance["accepted_draft_tokens"], 0)
        acceptance["accepted_draft_tokens"] -= 1
        acceptance["accepted_raw_ids"].pop()
        acceptance["committed_token_ids"].pop()
        committed = len(acceptance["committed_token_ids"])
        acceptance["next_pending_seed_token_id"] = artifact["golden"][
            "output_token_ids"
        ][committed]
        with self.assertRaises(ValueError):
            validate_r6a_artifact(artifact)

    def test_rejects_state_or_negative_timing_drift(self) -> None:
        artifact = R6AFixture.artifact()
        artifact["frozen_state"]["quiescent"] = False
        with self.assertRaisesRegex(ValueError, "quiescent"):
            validate_r6a_artifact(artifact)
        artifact = R6AFixture.artifact()
        artifact["trials"][0]["timings"]["cycle_wall_ns"] = -1
        with self.assertRaises(ValueError):
            validate_r6a_artifact(artifact)


if __name__ == "__main__":
    unittest.main()
