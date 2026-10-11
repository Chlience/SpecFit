from __future__ import annotations

from math import log
from types import SimpleNamespace
import unittest

from specfit.r6a_experiment import compute_r6a_route_digest
from specfit.r6a_frozen_cycle import FrozenNodeRoute, FrozenRouteTable
from specfit.r6a_runner import (
    _CapturingPolicy,
    _counter_delta,
    _require_empty_cache,
    _require_quiescent,
    frozen_state_record,
    selection_mapping,
    serialize_candidate_pool,
    serialize_routes,
    unique_route_acquisition_count,
)
from specfit.runtime import candidate_pool_fingerprint
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


class R6aRunnerHelperTest(unittest.TestCase):
    def pool(self) -> Eagle3DraftCandidatePool:
        candidates = (
            Eagle3DraftCandidate(1, 0, 1, 101, log(0.6), log(0.6), 0, (101,)),
            Eagle3DraftCandidate(2, 0, 1, 102, log(0.4), log(0.4), 1, (102,)),
            Eagle3DraftCandidate(
                3, 1, 2, 103, log(0.5), log(0.3), 2, (101, 103)
            ),
        )
        return Eagle3DraftCandidatePool(100, 2, 2, candidates, 3, 3)

    def routes(self) -> FrozenRouteTable:
        pool = self.pool()
        return FrozenRouteTable(
            candidate_pool_fingerprint(pool),
            (
                FrozenNodeRoute(0, ((0, 1), (3, 4))),
                FrozenNodeRoute(1, ((1, 2), (4, 5))),
                FrozenNodeRoute(3, ((2, 6), (3, 5))),
            ),
        )

    @staticmethod
    def snapshot(
        *,
        h2d: int = 2,
        hits: int = 1,
        joins: int = 0,
        evictions: int = 0,
    ) -> CacheSnapshotTrace:
        layers = tuple(
            LayerCacheSnapshotTrace(
                layer_id=layer_id,
                queue_id=f"queue-{layer_id}",
                capacity=2,
                submission_sequence=h2d,
                slots=(
                    CacheSlotTrace(0, layer_id + 1, 1, 0, 2, True, 0),
                    CacheSlotTrace(1, None, 0, 0, 0, False, 0),
                ),
                cache_id=f"qwen3-layer-{layer_id}",
                host_state_version=h2d + hits + joins + evictions,
                lru_sequence=2,
                stats=CacheStatsTrace(h2d, hits, joins, evictions, 0),
            )
            for layer_id in range(2)
        )
        return CacheSnapshotTrace(
            "run",
            "tree",
            "cycle",
            SnapshotPhase.BEFORE_VERIFY,
            sum(layer.host_state_version or 0 for layer in layers),
            1,
            layers,
        )

    def test_pool_capture_mapping_and_drift_gate(self) -> None:
        pool = self.pool()
        fingerprint = candidate_pool_fingerprint(pool)
        fixed = _CapturingPolicy("fixed", (1, 3), fingerprint)
        self.assertEqual(
            fixed.select_raw_ids(pool, draft_node_budget=2),
            (1, 3),
        )
        self.assertIs(fixed.pool, pool)
        self.assertEqual(
            selection_mapping((1, 3)),
            {
                "selected_raw_ids": [1, 3],
                "verification_position_to_raw_id": [0, 1, 3],
                "raw_to_verification_position": {"0": 0, "1": 1, "3": 2},
            },
        )
        with self.assertRaises(RuntimeError):
            _CapturingPolicy("fixed", (1,), "f" * 64).select_raw_ids(
                pool, draft_node_budget=1
            )

    def test_pool_and_own_routes_are_stably_serialized(self) -> None:
        pool, routes = self.pool(), self.routes()
        serialized_pool = serialize_candidate_pool(pool)
        self.assertEqual(serialized_pool["fingerprint"], routes.pool_fingerprint)
        self.assertEqual(serialized_pool["candidates"][2]["path"], (101, 103))
        serialized = serialize_routes(routes, (0, 1, 3))
        self.assertEqual(serialized["coverage_rows"], 6)
        self.assertEqual(
            serialized["digest"],
            compute_r6a_route_digest(
                pool_fingerprint=routes.pool_fingerprint,
                verification_position_to_raw_id=(0, 1, 3),
                rows=serialized["rows"],
            ),
        )
        self.assertEqual(unique_route_acquisition_count(routes, (0, 1, 3)), 7)

    def test_snapshot_delta_and_frozen_residency(self) -> None:
        pre = self.snapshot(h2d=2, hits=1)
        post = self.snapshot(h2d=4, hits=3, joins=1, evictions=2)
        self.assertEqual(
            _counter_delta(pre, post),
            {
                "h2d_submissions": 4,
                "resident_hits": 4,
                "inflight_joins": 2,
                "evictions": 4,
            },
        )
        frozen = frozen_state_record(pre, "a" * 64)
        self.assertEqual(
            frozen["resident_experts_by_layer"],
            [
                {"layer_id": 0, "expert_ids": [1]},
                {"layer_id": 1, "expert_ids": [2]},
            ],
        )
        with self.assertRaises(RuntimeError):
            _counter_delta(post, pre)

    def test_cache_boundary_guards_reject_nonempty_or_active_state(self) -> None:
        zero = SimpleNamespace(
            h2d_submissions=0,
            resident_hits=0,
            inflight_joins=0,
            evictions=0,
            pending_use_guards=0,
        )
        empty_slot = SimpleNamespace(
            expert_id=None, active_leases=0, pending_use_guards=0
        )
        empty = SimpleNamespace(
            pending_transfers=(), slots=(empty_slot,), stats=zero
        )
        _require_empty_cache((empty,))
        _require_quiescent((empty,))
        occupied = SimpleNamespace(
            expert_id=7, active_leases=0, pending_use_guards=0
        )
        nonempty = SimpleNamespace(
            pending_transfers=(), slots=(occupied,), stats=zero
        )
        with self.assertRaises(RuntimeError):
            _require_empty_cache((nonempty,))
        active = SimpleNamespace(
            expert_id=None, active_leases=1, pending_use_guards=0
        )
        busy = SimpleNamespace(
            pending_transfers=(), slots=(active,), stats=zero
        )
        with self.assertRaises(RuntimeError):
            _require_quiescent((busy,))


if __name__ == "__main__":
    unittest.main()
