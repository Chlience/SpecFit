from __future__ import annotations

import unittest
from dataclasses import replace
from math import log

from specfit.r6a_frozen_cycle import (
    CandidateSetScore,
    FrozenNodeRoute,
    FrozenRouteTable,
    cache_snapshot_state_fingerprint,
    candidate_pool_fingerprint,
    enumerate_ancestor_closed_subsets,
    live_cache_snapshots_state_fingerprint,
    resident_experts_from_snapshot,
    route_table_from_trace,
    score_candidate_set,
    select_max_efficiency,
)
from specfit.runtime.cache_snapshot import (
    CacheSlotSnapshot,
    GpuExpertCacheSnapshot,
    PendingTransferSnapshot,
)
from specfit.runtime.eagle3_selection import (
    candidate_pool_fingerprint as runtime_pool_fingerprint,
)
from specfit.runtime.eagle3_types import (
    Eagle3DraftCandidate,
    Eagle3DraftCandidatePool,
)
from specfit.runtime.lease import ExpertCacheStats
from specfit.state import ExpertKey
from specfit.trace import (
    CacheSlotTrace,
    CacheSnapshotTrace,
    CacheStatsTrace,
    CycleTrace,
    DraftNodeTrace,
    DraftTreeTrace,
    LayerCacheSnapshotTrace,
    LayerRouteTrace,
    PendingTransferTrace,
    SnapshotPhase,
    TraceBundle,
    VerificationNodeRole,
    WorkloadManifestTrace,
)


class R6aFrozenCycleTest(unittest.TestCase):
    def pool(
        self,
        parents: tuple[int, ...] = (0, 0, 1, 1, 2, 3),
        probabilities: tuple[float, ...] = (0.6, 0.4, 0.5, 0.3, 0.5, 0.5),
    ) -> Eagle3DraftCandidatePool:
        candidates = []
        cumulative = {0: 0.0}
        depths = {0: 0}
        paths: dict[int, tuple[int, ...]] = {0: ()}
        for raw_id, (parent, probability) in enumerate(
            zip(parents, probabilities), start=1
        ):
            conditional = log(probability)
            score = cumulative[parent] + conditional
            depth = depths[parent] + 1
            token_id = 100 + raw_id
            path = paths[parent] + (token_id,)
            candidates.append(Eagle3DraftCandidate(
                raw_id, parent, depth, token_id, conditional, score,
                raw_id - 1, path,
            ))
            cumulative[raw_id], depths[raw_id], paths[raw_id] = score, depth, path
        return Eagle3DraftCandidatePool(
            100, 2, max(depths.values()), tuple(candidates),
            len(candidates), len(candidates),
        )

    def balanced_depth_five_pool(self) -> Eagle3DraftCandidatePool:
        parents = (0, 0, 1, 1, 2, 2, 3, 3, 5, 5, 7, 7, 9, 9,
                   11, 11, 13, 13)
        probabilities = (0.6, 0.4) + tuple(
            0.9 if raw_id % 2 else 0.1 for raw_id in range(3, 19)
        )
        return self.pool(parents, probabilities)

    def concentrated_depth_five_pool(self) -> Eagle3DraftCandidatePool:
        parents = (0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 7, 7, 8, 8,
                   11, 11, 12, 12)
        probabilities = (
            0.99, 0.01, 0.6, 0.3, 0.6, 0.3,
            0.6, 0.3, 0.4, 0.3, 0.6, 0.3, 0.4, 0.3,
            0.6, 0.3, 0.4, 0.3,
        )
        return self.pool(parents, probabilities)

    def partial_routes(self) -> FrozenRouteTable:
        return FrozenRouteTable(candidate_pool_fingerprint(self.pool()), (
            FrozenNodeRoute(0, ((0, 1), (0, 1))),
            FrozenNodeRoute(1, ((1, 2), (1, 2))),
            FrozenNodeRoute(3, ((2, 4), (1, 3))),
        ))

    def test_fixed_budget_count_depends_on_parent_topology(self) -> None:
        pool = self.pool()
        self.assertIs(candidate_pool_fingerprint, runtime_pool_fingerprint)
        self.assertEqual(
            enumerate_ancestor_closed_subsets(pool, draft_node_budget=2),
            ((1, 2), (1, 3), (1, 4), (2, 5)),
        )
        balanced = self.balanced_depth_five_pool()
        concentrated = self.concentrated_depth_five_pool()
        for candidate_pool in (balanced, concentrated):
            depth_counts = tuple(
                sum(node.depth == depth for node in candidate_pool.candidates)
                for depth in range(1, 6)
            )
            self.assertEqual(
                (candidate_pool.top_k, candidate_pool.max_depth,
                 candidate_pool.raw_candidate_count, depth_counts),
                (2, 5, 18, (2, 4, 4, 4, 4)),
            )
        self.assertEqual(len(enumerate_ancestor_closed_subsets(
            balanced, draft_node_budget=7,
        )), 104)
        self.assertEqual(len(enumerate_ancestor_closed_subsets(
            concentrated, draft_node_budget=7,
        )), 153)
    def test_partial_full_coverage_and_frozen_footprint(self) -> None:
        pool, partial = self.pool(), self.partial_routes()
        self.assertEqual(partial.validate_selection(pool, (3, 1)), (1, 3))
        with self.assertRaises(ValueError):
            partial.validate_complete_pool(pool)
        with self.assertRaises(ValueError):
            partial.validate_selection(pool, (1, 2))

        score = score_candidate_set(
            pool,
            selected_raw_ids=(3, 1),
            routes=partial,
            initial_resident_experts=frozenset({
                ExpertKey(0, 0), ExpertKey(0, 4), ExpertKey(1, 2),
            }),
        )
        self.assertEqual(score.selected_raw_ids, (1, 3))
        self.assertAlmostEqual(score.draft_path_probability_mass, 0.9)
        self.assertAlmostEqual(score.draft_utility_proxy, 1.9)
        self.assertEqual((score.unique_expert_count,
                          score.initial_nonresident_expert_count), (8, 5))

    def test_efficiency_tie_break_is_deterministic(self) -> None:
        scores = (
            CandidateSetScore((3,), 1.0, 2.0, 4, 2),
            CandidateSetScore((2,), 2.0, 3.0, 6, 3),
            CandidateSetScore((1,), 2.0, 3.0, 6, 3),
        )
        chosen = select_max_efficiency(scores, cost="initial_nonresident")
        self.assertEqual(chosen.selected_raw_ids, (1,))
        with self.assertRaises(ValueError):
            select_max_efficiency(scores, cost="bytes")  # type: ignore[arg-type]

    def manifest(self) -> WorkloadManifestTrace:
        return WorkloadManifestTrace(
            run_id="run", request_id="request", git_commit="abc",
            image_id="image", target_model="target", target_revision="target-rev",
            drafter_model="draft", drafter_revision="draft-rev",
            tokenizer_revision="tok-rev", device="cpu", dtype="float32",
            reduction_mode="test", drafter_mode="test", slots_per_layer=2,
            num_moe_layers=2, num_experts_per_layer=10, experts_per_token=2,
            draft_length=2, max_new_tokens=3, seed=0, raw_prompt="p",
            rendered_prompt="p", input_ids=(10,), attention_mask=(1,),
        )

    def trace_bundle(self) -> TraceBundle:
        tree_id = "tree"
        nodes = (
            DraftNodeTrace(tree_id, 0, None, 0, VerificationNodeRole.SEED,
                           100, None, 0, True, True, True),
            DraftNodeTrace(tree_id, 7, 0, 1, VerificationNodeRole.DRAFT,
                           101, 0.6, 1, True, True, False),
            DraftNodeTrace(tree_id, 9, 7, 2, VerificationNodeRole.DRAFT,
                           103, 0.5, 2, True, True, False),
        )
        tree = DraftTreeTrace(tree_id, "request", 0, nodes)
        cycle = CycleTrace(
            "run", tree_id, "request", 0, 1, 100, (100, 101, 103),
            (999, 999, 999), (), (100,), 2,
        )
        route_rows = self.partial_routes().routes
        node_ids = (0, 7, 9)
        routes = tuple(
            LayerRouteTrace(tree_id, node_id, layer_id, experts, (0.6, 0.4))
            for node_id, row in zip(node_ids, route_rows)
            for layer_id, experts in enumerate(row.experts_by_layer)
        )
        return TraceBundle(self.manifest(), (tree,), (cycle,),
                           target_routes=routes)

    def test_trace_routes_map_positions_to_raw_ids(self) -> None:
        bundle, pool = self.trace_bundle(), self.pool()
        table = route_table_from_trace(
            bundle, pool=pool, verification_position_to_raw_id=(0, 1, 3)
        )
        self.assertEqual(tuple(row.raw_id for row in table.routes), (0, 1, 3))
        self.assertEqual(table.for_raw_id(3).experts_by_layer,
                         ((2, 4), (1, 3)))
        with self.assertRaises(ValueError):
            route_table_from_trace(
                bundle, pool=pool, verification_position_to_raw_id=(0, 1, 2)
            )
        with self.assertRaises(ValueError):
            route_table_from_trace(
                replace(bundle, target_routes=bundle.target_routes[:-1]),
                pool=pool, verification_position_to_raw_id=(0, 1, 3),
            )

    def snapshot_pair(
        self,
    ) -> tuple[GpuExpertCacheSnapshot, CacheSnapshotTrace]:
        live_slots = (
            CacheSlotSnapshot(0, ExpertKey(0, 7), 1, 0, 3, True, 0),
            CacheSlotSnapshot(1, ExpertKey(0, 8), 2, 0, 2, False, 0),
        )
        live_pending = PendingTransferSnapshot(
            ExpertKey(0, 8), 1, 2, 1, object(), False,  # type: ignore[arg-type]
        )
        stats = ExpertCacheStats(2, 1, 0, 0, 0)
        live = GpuExpertCacheSnapshot(
            "qwen3-layer-0", "queue-0", 2, 5, 3, 2,
            live_slots, (live_pending,), stats,
        )
        trace_layer = LayerCacheSnapshotTrace(
            layer_id=0, queue_id="queue-0", capacity=2, submission_sequence=2,
            slots=(CacheSlotTrace(0, 7, 1, 0, 3, True, 0),
                   CacheSlotTrace(1, 8, 2, 0, 2, False, 0)),
            pending_transfers=(PendingTransferTrace(
                "transfer-id", 8, 1, 2, 1, "event-id", False,
            ),),
            cache_id="qwen3-layer-0", host_state_version=5, lru_sequence=3,
            stats=CacheStatsTrace(2, 1, 0, 0, 0),
        )
        trace = CacheSnapshotTrace(
            "run", "tree", "cycle", SnapshotPhase.BEFORE_VERIFY,
            5, 123, (trace_layer,),
        )
        return live, trace

    def test_live_and_trace_snapshots_share_state_fingerprint(self) -> None:
        live, trace = self.snapshot_pair()
        digest = live_cache_snapshots_state_fingerprint((live,))
        self.assertEqual(digest, cache_snapshot_state_fingerprint(trace))
        changed_event = replace(
            live.pending_transfers[0], ready_event=object()  # type: ignore[arg-type]
        )
        self.assertEqual(
            digest,
            live_cache_snapshots_state_fingerprint((
                replace(live, pending_transfers=(changed_event,)),
            )),
        )
        renamed_trace = replace(
            trace, run_id="other", tree_id="other-tree", cycle_id="other-cycle",
            phase=SnapshotPhase.AFTER_VERIFY, captured_at_ns=999,
        )
        self.assertEqual(digest, cache_snapshot_state_fingerprint(renamed_trace))
        self.assertNotEqual(
            digest,
            live_cache_snapshots_state_fingerprint((replace(live, lru_sequence=4),)),
        )
        self.assertEqual(resident_experts_from_snapshot(trace),
                         frozenset({ExpertKey(0, 7)}))


if __name__ == "__main__":
    unittest.main()
