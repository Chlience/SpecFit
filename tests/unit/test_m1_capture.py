from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest

from scripts.benchmark_qwen3_m1_full_atlas import _require_m1_eos_ids
from specfit.experiment import canonical_json_sha256
from specfit.m1_capture_validation import output_token_diagnostic
from specfit.m1_capture import (
    M1Selection,
    ScheduledM1Policy,
    project_frozen_state,
    project_logical_cache_state,
    project_probe_state,
    project_reference_rollout,
    quartile_cycle_indices,
    require_quiescent,
    serialize_m1_pool,
)
from specfit.runtime.eagle3_types import (
    Eagle3DraftCandidate,
    Eagle3DraftCandidatePool,
)
from specfit.trace import (
    CacheSlotTrace,
    CacheSnapshotTrace,
    CacheStatsTrace,
    LayerCacheSnapshotTrace,
    SnapshotPhase,
)
from tests.unit.r6a_fixture import R6AFixture


class M1CaptureTest(unittest.TestCase):
    def test_output_token_diagnostic_covers_value_and_length_splits(self) -> None:
        exact = output_token_diagnostic((1, 2), (1, 2))
        value = output_token_diagnostic((1, 2), (1, 3))
        length = output_token_diagnostic((1, 2), (1, 2, 3))
        self.assertEqual((exact["exact"], exact["first_difference_offset"],
                          exact["different_token_count"]), (True, None, 0))
        self.assertEqual((value["common_prefix_token_count"],
                          value["target_token_id"], value["reference_token_id"],
                          value["different_token_count"]), (1, 2, 3, 1))
        self.assertEqual((length["common_prefix_token_count"],
                          length["target_token_id"], length["reference_token_id"],
                          length["different_token_count"]), (2, None, 3, 1))

    @staticmethod
    def _pool(max_depth: int = 5) -> Eagle3DraftCandidatePool:
        source = R6AFixture.pool()
        nodes = tuple(
            Eagle3DraftCandidate(
                raw_id=item["raw_id"],
                parent_raw_id=item["parent_raw_id"],
                depth=item["depth"],
                target_token_id=item["target_token_id"],
                conditional_log_probability=item["conditional_log_probability"],
                cumulative_log_probability=item["cumulative_log_probability"],
                generation_ordinal=item["generation_ordinal"],
                path=tuple(item["path"]),
            )
            for item in source["candidates"] if item["depth"] <= max_depth
        )
        return Eagle3DraftCandidatePool(
            42, 2, max_depth, nodes, 1 + 2 * (max_depth - 1), 29
        )

    @staticmethod
    def _snapshot(cycle_id: str, phase: SnapshotPhase) -> CacheSnapshotTrace:
        layers = tuple(
            LayerCacheSnapshotTrace(
                layer_id=layer_id,
                queue_id=f"queue-{layer_id}",
                capacity=2,
                submission_sequence=1,
                slots=(
                    CacheSlotTrace(0, layer_id, 1, 0, 1, True, 0),
                    CacheSlotTrace(1, None, 0, 0, 0, False, 0),
                ),
                cache_id=f"cache-{layer_id}",
                host_state_version=1,
                lru_sequence=1,
                stats=CacheStatsTrace(1, 0, 0, 0, 0),
            )
            for layer_id in range(48)
        )
        return CacheSnapshotTrace(
            "run", "tree", cycle_id, phase, 48, 1, layers
        )

    @staticmethod
    def _trace(
        selections: tuple[M1Selection, ...],
        *,
        residual: tuple[int, ...] = (42,),
        stopped_on_eos: bool = True,
        max_new_tokens: int = 10,
    ) -> SimpleNamespace:
        cycles, snapshots, routes = [], [], []
        context_length = 2
        output = []
        for index, selection in enumerate(selections):
            tree = selection.pool.to_tree(selection.selected_raw_ids)
            mapping = (0, *selection.selected_raw_ids)
            cycle_id = f"cycle-{index}"
            tree_id = f"tree-{index}"
            cycles.append(SimpleNamespace(
                trace_id=cycle_id,
                tree_id=tree_id,
                iteration=index,
                base_kv_length=context_length,
                pending_seed_token_id=42,
                verification_token_ids=(42, *tree.token_ids),
                verifier_token_ids=(42,) * len(mapping),
                accepted_draft_node_ids=(),
                committed_token_ids=(42,),
            ))
            snapshots.extend((
                M1CaptureTest._snapshot(cycle_id, SnapshotPhase.BEFORE_VERIFY),
                M1CaptureTest._snapshot(cycle_id, SnapshotPhase.AFTER_VERIFY),
            ))
            routes.extend(
                SimpleNamespace(
                    tree_id=tree_id,
                    node_id=position,
                    layer_id=layer_id,
                    expert_ids=tuple(range(8)),
                )
                for position in range(len(mapping))
                for layer_id in range(48)
            )
            context_length += 1
            output.append(42)
        output.extend(residual)
        generation = SimpleNamespace(
            output_token_ids=tuple(output), residual_token_ids=residual,
            stopped_on_eos=stopped_on_eos,
            stop_token_id=42 if stopped_on_eos else None,
        )
        return SimpleNamespace(
            cycles=tuple(reversed(cycles)),
            snapshots=tuple(snapshots),
            target_routes=tuple(routes),
            generations=(generation,),
            manifest=SimpleNamespace(
                num_moe_layers=48, max_new_tokens=max_new_tokens
            ),
        )

    def test_scheduled_policy_uses_fixed7_then_full18(self) -> None:
        pool = self._pool()
        policy = ScheduledM1Policy(probe_cycle=1)
        first = policy.select_raw_ids(
            pool, draft_node_budget=18, committed_offset=0
        )
        second = policy.select_raw_ids(
            pool, draft_node_budget=18, committed_offset=1
        )
        self.assertEqual((len(first), len(second)), (7, 18))
        self.assertEqual(second, tuple(range(1, 19)))
        with self.assertRaises(RuntimeError):
            policy.select_raw_ids(pool, draft_node_budget=18)
        with self.assertRaises(RuntimeError):
            ScheduledM1Policy().select_raw_ids(pool, draft_node_budget=7)

        short = self._pool(max_depth=2)
        short_ids = ScheduledM1Policy().select_raw_ids(
            short, draft_node_budget=18
        )
        self.assertEqual(short_ids, tuple(range(1, 7)))
        with self.assertRaises(RuntimeError):
            ScheduledM1Policy(probe_cycle=0).select_raw_ids(
                short, draft_node_budget=18
            )

    def test_pool_preserves_authoritative_binary64_values(self) -> None:
        serialized = serialize_m1_pool(self._pool())
        self.assertEqual(serialized["raw_candidate_count"], 18)
        for node in serialized["candidates"]:
            self.assertEqual(
                node["conditional_log_probability_hex"],
                float(node["conditional_log_probability"]).hex(),
            )
            self.assertEqual(
                node["cumulative_log_probability_hex"],
                float(node["cumulative_log_probability"]).hex(),
            )

    def test_logical_state_is_identity_free_and_requires_quiescence(self) -> None:
        snapshot = self._snapshot("cycle", SnapshotPhase.BEFORE_VERIFY)
        logical = project_logical_cache_state(snapshot)
        frozen = project_frozen_state(snapshot)
        self.assertNotIn("cache_id", str(logical))
        self.assertEqual(
            frozen["cache_state_fingerprint"], canonical_json_sha256(logical)
        )
        first = snapshot.layers[0]
        busy = replace(
            snapshot,
            layers=(replace(
                first, slots=(replace(first.slots[0], ready=False), first.slots[1])
            ), *snapshot.layers[1:]),
        )
        with self.assertRaisesRegex(RuntimeError, "quiescent"):
            require_quiescent(busy)

    def test_reference_and_probe_projection_do_not_leak_probe_outcome(self) -> None:
        pool = self._pool()
        behavior = ScheduledM1Policy()
        for offset in range(4):
            behavior.select_raw_ids(
                pool, draft_node_budget=18, committed_offset=offset
            )
        reference = project_reference_rollout(
            question_id=406,
            input_token_ids=(10, 11),
            trace=self._trace(tuple(behavior.records)),
            selections=tuple(behavior.records),
        )
        self.assertEqual(reference["sampled_cycle_indices"], [0, 1, 2, 3])
        self.assertEqual(reference["tail_token_ids"], [42])
        self.assertEqual(reference["max_new_tokens"], 10)
        self.assertTrue(reference["stopped_on_eos"])
        self.assertEqual(reference["stop_token_id"], 42)
        self.assertEqual(reference["cycles"][0]["base_kv_length"], 2)
        self.assertEqual(len(reference["cycles"][0]["verifier_token_ids"]), 8)

        probe_policy = ScheduledM1Policy(probe_cycle=0)
        probe_policy.select_raw_ids(pool, draft_node_budget=18)
        state = project_probe_state(
            question_id=406,
            probe_cycle=0,
            input_token_ids=(10, 11),
            trace=self._trace(tuple(probe_policy.records), residual=()),
            selections=tuple(probe_policy.records),
            reference_cycle=reference["cycles"][0],
        )
        self.assertEqual(state["verification_position_to_raw_id"], list(range(19)))
        self.assertEqual(state["full_atlas_routes"]["coverage_rows"], 912)
        self.assertEqual(state["base_kv_length"], 2)
        forbidden = {
            "accepted_raw_ids", "committed_token_ids", "timings",
            "post_logical_state_sha256", "cache_counter_delta",
        }
        self.assertFalse(forbidden.intersection(state))

    def test_reference_allows_unsampled_short_final_pool(self) -> None:
        policy = ScheduledM1Policy()
        for offset in range(8):
            policy.select_raw_ids(
                self._pool(), draft_node_budget=18, committed_offset=offset
            )
        policy.select_raw_ids(
            self._pool(max_depth=1), draft_node_budget=18, committed_offset=8
        )
        selections = tuple(policy.records)
        trace = self._trace(selections)
        reference = project_reference_rollout(
            question_id=406, input_token_ids=(10, 11), trace=trace,
            selections=selections,
        )
        self.assertEqual(reference["sampled_cycle_indices"], [1, 3, 5, 7])
        self.assertEqual(reference["cycles"][-1]["selected_raw_ids"], [1, 2])
        self.assertEqual(reference["cycles"][-1]["base_kv_length"], 10)

        trace = self._trace(selections)
        next(cycle for cycle in trace.cycles if cycle.iteration == 8
             ).base_kv_length = 9
        with self.assertRaisesRegex(RuntimeError, "context"):
            project_reference_rollout(
                question_id=406, input_token_ids=(10, 11), trace=trace,
                selections=selections,
            )

    def test_quartile_contract_rejects_short_reference(self) -> None:
        self.assertEqual(quartile_cycle_indices(35), (4, 13, 21, 30))
        with self.assertRaises(ValueError):
            quartile_cycle_indices(3)

    def test_runner_requires_fixed_target_eos_set(self) -> None:
        model = SimpleNamespace(
            generation_config=SimpleNamespace(eos_token_id=[151643, 151645]),
            config=SimpleNamespace(eos_token_id=0),
        )
        _require_m1_eos_ids(model)
        model.generation_config.eos_token_id = None
        model.config.eos_token_id = [151645, 151643]
        _require_m1_eos_ids(model)
        model.config.eos_token_id = [151645]
        with self.assertRaisesRegex(RuntimeError, "EOS IDs"):
            _require_m1_eos_ids(model)


if __name__ == "__main__":
    unittest.main()
