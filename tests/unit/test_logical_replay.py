from __future__ import annotations

import unittest
from dataclasses import replace

from specfit import (
    CacheSlotTrace,
    CacheSnapshotTrace,
    CacheStatsTrace,
    CycleReplayResult,
    CycleTrace,
    DraftNodeTrace,
    DraftTreeTrace,
    LayerCacheSnapshotTrace,
    LayerRouteTrace,
    LogicalReplayError,
    LogicalReplayEvent,
    LogicalReplayResult,
    ReplayCounters,
    ReplaySlotState,
    RuntimeEventKind,
    RuntimeEventTrace,
    SnapshotPhase,
    TraceBundle,
    VerificationNodeRole,
    WorkloadManifestTrace,
    replay_qwen3_chain,
    stable_trace_id,
)


class LogicalReplayTest(unittest.TestCase):

    def _manifest(self) -> WorkloadManifestTrace:
        return WorkloadManifestTrace(
            run_id="run-replay",
            request_id="request-replay",
            git_commit="abc123",
            image_id="nvcr-pytorch-25.11",
            target_model="Qwen3-30B-A3B",
            target_revision="target-sha",
            drafter_model="EAGLE-3",
            drafter_revision="draft-sha",
            tokenizer_revision="tokenizer-sha",
            device="cuda:0",
            dtype="bfloat16",
            reduction_mode="canonical",
            drafter_mode="eager",
            slots_per_layer=2,
            num_moe_layers=1,
            num_experts_per_layer=8,
            experts_per_token=2,
            draft_length=1,
            max_new_tokens=2,
            seed=7,
            raw_prompt="Explain expert caching.",
            rendered_prompt="<user>Explain expert caching.",
            input_ids=(10, 11),
            attention_mask=(1, 1),
        )

    def _tree_cycle(
        self,
        iteration: int,
    ) -> tuple[DraftTreeTrace, CycleTrace]:
        tree_id = f"tree-{iteration}"
        seed_token = 100 + 10 * iteration
        draft_token = seed_token + 1
        nodes = (
            DraftNodeTrace(
                tree_id=tree_id,
                node_id=0,
                parent_id=None,
                depth=0,
                role=VerificationNodeRole.SEED,
                token_id=seed_token,
                conditional_probability=None,
                verification_position=0,
                selected=True,
                verified=True,
                accepted=True,
            ),
            DraftNodeTrace(
                tree_id=tree_id,
                node_id=1,
                parent_id=0,
                depth=1,
                role=VerificationNodeRole.DRAFT,
                token_id=draft_token,
                conditional_probability=None,
                verification_position=1,
                selected=True,
                verified=True,
                accepted=True,
            ),
        )
        tree = DraftTreeTrace(
            tree_id=tree_id,
            request_id="request-replay",
            iteration=iteration,
            nodes=nodes,
        )
        base_kv_length = 14 + 2 * iteration
        cycle = CycleTrace(
            run_id="run-replay",
            tree_id=tree_id,
            request_id="request-replay",
            iteration=iteration,
            base_kv_length=base_kv_length,
            pending_seed_token_id=seed_token,
            verification_token_ids=(seed_token, draft_token),
            verifier_token_ids=(draft_token, 999),
            accepted_draft_node_ids=(1,),
            committed_token_ids=(seed_token, draft_token),
            kv_length_after=base_kv_length + 2,
        )
        return tree, cycle

    @staticmethod
    def _stats(
        *,
        h2d: int,
        hits: int,
        evictions: int,
    ) -> CacheStatsTrace:
        return CacheStatsTrace(
            h2d_submissions=h2d,
            resident_hits=hits,
            inflight_joins=0,
            evictions=evictions,
            pending_use_guards=0,
        )

    def _snapshot(
        self,
        *,
        cycle: CycleTrace,
        phase: SnapshotPhase,
        captured_at_ns: int,
        host_state_version: int,
        lru_sequence: int,
        submission_sequence: int,
        slots: tuple[CacheSlotTrace, CacheSlotTrace],
        stats: CacheStatsTrace,
    ) -> CacheSnapshotTrace:
        layer = LayerCacheSnapshotTrace(
            layer_id=0,
            cache_id="layer-0-cache",
            queue_id="layer-0-copy",
            capacity=2,
            host_state_version=host_state_version,
            lru_sequence=lru_sequence,
            submission_sequence=submission_sequence,
            slots=slots,
            stats=stats,
        )
        return CacheSnapshotTrace(
            run_id="run-replay",
            tree_id=cycle.tree_id,
            cycle_id=cycle.trace_id,
            phase=phase,
            runtime_state_version=host_state_version,
            captured_at_ns=captured_at_ns,
            layers=(layer,),
        )

    @staticmethod
    def _slot(
        slot_id: int,
        expert_id: int | None,
        generation: int,
        last_used_sequence: int,
        *,
        active_leases: int = 0,
    ) -> CacheSlotTrace:
        return CacheSlotTrace(
            slot_id=slot_id,
            expert_id=expert_id,
            generation=generation,
            active_leases=active_leases,
            last_used_sequence=last_used_sequence,
            ready=expert_id is not None,
            pending_use_guards=0,
        )

    def _event(
        self,
        *,
        cycle: CycleTrace,
        sequence: int,
        kind: RuntimeEventKind,
        expert_id: int,
        slot_id: int,
        generation: int,
        submission_sequence: int | None = None,
        victim_expert_id: int | None = None,
    ) -> RuntimeEventTrace:
        transfer_id = None
        if kind is RuntimeEventKind.H2D_SUBMIT:
            if submission_sequence is None:
                raise AssertionError("H2D fixture missing submission sequence")
            transfer_id = stable_trace_id(
                "h2d",
                "run-replay",
                0,
                submission_sequence,
            )
        return RuntimeEventTrace(
            run_id="run-replay",
            sequence=sequence,
            kind=kind,
            host_time_ns=1_000 + sequence,
            layer_id=0,
            cache_id="layer-0-cache",
            queue_id="layer-0-copy",
            expert_id=expert_id,
            slot_id=slot_id,
            generation=generation,
            submission_sequence=submission_sequence,
            tree_id=cycle.tree_id,
            cycle_id=cycle.trace_id,
            transfer_id=transfer_id,
            victim_expert_id=victim_expert_id,
            stream_ids=(7,) if kind in (
                RuntimeEventKind.CONSUME_WAIT,
                RuntimeEventKind.RELEASE,
            ) else (),
        )

    def _access_events(
        self,
        *,
        cycle: CycleTrace,
        start_sequence: int,
        kind: RuntimeEventKind,
        expert_id: int,
        slot_id: int,
        generation: int,
        submission_sequence: int | None = None,
        victim_expert_id: int | None = None,
    ) -> tuple[RuntimeEventTrace, ...]:
        events = []
        sequence = start_sequence
        if victim_expert_id is not None:
            events.append(
                self._event(
                    cycle=cycle,
                    sequence=sequence,
                    kind=RuntimeEventKind.EVICT,
                    expert_id=expert_id,
                    slot_id=slot_id,
                    generation=generation,
                    victim_expert_id=victim_expert_id,
                )
            )
            sequence += 1
        events.extend(
            (
                self._event(
                    cycle=cycle,
                    sequence=sequence,
                    kind=kind,
                    expert_id=expert_id,
                    slot_id=slot_id,
                    generation=generation,
                    submission_sequence=submission_sequence,
                ),
                self._event(
                    cycle=cycle,
                    sequence=sequence + 1,
                    kind=RuntimeEventKind.CONSUME_WAIT,
                    expert_id=expert_id,
                    slot_id=slot_id,
                    generation=generation,
                ),
                self._event(
                    cycle=cycle,
                    sequence=sequence + 2,
                    kind=RuntimeEventKind.RELEASE,
                    expert_id=expert_id,
                    slot_id=slot_id,
                    generation=generation,
                ),
            )
        )
        return tuple(events)

    def _first_cycle_fixture(self) -> TraceBundle:
        tree, cycle = self._tree_cycle(0)
        before = self._snapshot(
            cycle=cycle,
            phase=SnapshotPhase.BEFORE_VERIFY,
            captured_at_ns=100,
            host_state_version=4,
            lru_sequence=4,
            submission_sequence=1,
            slots=(
                self._slot(0, 0, 1, 4),
                self._slot(1, None, 0, 0),
            ),
            stats=self._stats(h2d=1, hits=0, evictions=0),
        )
        after = self._snapshot(
            cycle=cycle,
            phase=SnapshotPhase.AFTER_VERIFY,
            captured_at_ns=2_000,
            host_state_version=10,
            lru_sequence=10,
            submission_sequence=3,
            slots=(
                self._slot(0, 2, 2, 10),
                self._slot(1, 1, 1, 8),
            ),
            stats=self._stats(h2d=3, hits=1, evictions=1),
        )
        cycle = replace(
            cycle,
            pre_snapshot_id=before.trace_id,
            post_snapshot_id=after.trace_id,
        )
        routes = (
            LayerRouteTrace(
                tree_id=tree.tree_id,
                node_id=0,
                layer_id=0,
                expert_ids=(0, 1),
                expert_weights=(0.6, 0.4),
            ),
            LayerRouteTrace(
                tree_id=tree.tree_id,
                node_id=1,
                layer_id=0,
                expert_ids=(0, 2),
                expert_weights=(0.7, 0.3),
            ),
        )
        events = (
            *self._access_events(
                cycle=cycle,
                start_sequence=0,
                kind=RuntimeEventKind.RESIDENT_HIT,
                expert_id=0,
                slot_id=0,
                generation=1,
            ),
            *self._access_events(
                cycle=cycle,
                start_sequence=3,
                kind=RuntimeEventKind.H2D_SUBMIT,
                expert_id=1,
                slot_id=1,
                generation=1,
                submission_sequence=1,
            ),
            *self._access_events(
                cycle=cycle,
                start_sequence=6,
                kind=RuntimeEventKind.H2D_SUBMIT,
                expert_id=2,
                slot_id=0,
                generation=2,
                submission_sequence=2,
                victim_expert_id=0,
            ),
        )
        return TraceBundle(
            manifest=self._manifest(),
            trees=(tree,),
            cycles=(cycle,),
            target_routes=routes,
            snapshots=(before, after),
            runtime_events=events,
        )

    def _two_cycle_discontinuous_fixture(self) -> TraceBundle:
        first = self._first_cycle_fixture()
        tree, cycle = self._tree_cycle(1)
        before = self._snapshot(
            cycle=cycle,
            phase=SnapshotPhase.BEFORE_VERIFY,
            captured_at_ns=3_000,
            host_state_version=10,
            lru_sequence=10,
            submission_sequence=3,
            slots=(
                self._slot(0, 3, 2, 10),
                self._slot(1, 1, 1, 8),
            ),
            stats=self._stats(h2d=3, hits=1, evictions=1),
        )
        after = self._snapshot(
            cycle=cycle,
            phase=SnapshotPhase.AFTER_VERIFY,
            captured_at_ns=5_000,
            host_state_version=14,
            lru_sequence=14,
            submission_sequence=3,
            slots=(
                self._slot(0, 3, 2, 14),
                self._slot(1, 1, 1, 12),
            ),
            stats=self._stats(h2d=3, hits=3, evictions=1),
        )
        cycle = replace(
            cycle,
            pre_snapshot_id=before.trace_id,
            post_snapshot_id=after.trace_id,
        )
        routes = tuple(
            LayerRouteTrace(
                tree_id=tree.tree_id,
                node_id=node_id,
                layer_id=0,
                expert_ids=(1, 3),
                expert_weights=(0.6, 0.4),
            )
            for node_id in (0, 1)
        )
        events = (
            *self._access_events(
                cycle=cycle,
                start_sequence=10,
                kind=RuntimeEventKind.RESIDENT_HIT,
                expert_id=1,
                slot_id=1,
                generation=1,
            ),
            *self._access_events(
                cycle=cycle,
                start_sequence=13,
                kind=RuntimeEventKind.RESIDENT_HIT,
                expert_id=3,
                slot_id=0,
                generation=2,
            ),
        )
        return TraceBundle(
            manifest=first.manifest,
            trees=(*first.trees, tree),
            cycles=(*first.cycles, cycle),
            target_routes=(*first.target_routes, *routes),
            snapshots=(*first.snapshots, before, after),
            runtime_events=(*first.runtime_events, *events),
        )

    def test_shared_union_hit_empty_miss_and_lru_eviction_replay(self) -> None:
        bundle = self._first_cycle_fixture()

        result = replay_qwen3_chain(bundle)

        self.assertIsInstance(result, LogicalReplayResult)
        self.assertEqual(len(result.cycles), 1)
        self.assertIsInstance(result.cycles[0], CycleReplayResult)
        self.assertIsInstance(result.counters, ReplayCounters)
        self.assertEqual(result.counters.h2d_submissions, 2)
        self.assertEqual(result.counters.resident_hits, 1)
        self.assertEqual(result.counters.inflight_joins, 0)
        self.assertEqual(result.counters.evictions, 1)
        self.assertEqual(result.counters.consume_waits, 3)
        self.assertEqual(result.counters.releases, 3)
        self.assertEqual(result.counters.acquisitions, 3)
        self.assertEqual(result.counters.event_count, 10)
        self.assertEqual(result.initial_occupancy, 1)
        self.assertEqual(result.final_occupancy, 2)
        self.assertEqual(result.cycles[0].occupancy_before, 1)
        self.assertEqual(result.cycles[0].occupancy_after, 2)

        accesses = tuple(
            event
            for event in result.cycles[0].events
            if event.kind in (
                RuntimeEventKind.H2D_SUBMIT,
                RuntimeEventKind.RESIDENT_HIT,
                RuntimeEventKind.INFLIGHT_JOIN,
            )
        )
        self.assertTrue(
            all(isinstance(event, LogicalReplayEvent) for event in accesses)
        )
        self.assertEqual(
            tuple(event.expert_id for event in accesses),
            (0, 1, 2),
        )
        self.assertEqual(
            result.cycles[0].accessed_experts_by_layer,
            ((0, 1, 2),),
        )
        self.assertTrue(
            all(
                isinstance(slot, ReplaySlotState)
                for slot in result.final_slots
            )
        )
        self.assertEqual(
            tuple(
                (
                    slot.layer_id,
                    slot.slot_id,
                    slot.expert_id,
                    slot.generation,
                    slot.last_used_sequence,
                )
                for slot in result.final_slots
            ),
            (
                (0, 0, 2, 2, 10),
                (0, 1, 1, 1, 8),
            ),
        )

    def test_wrong_lru_victim_is_rejected(self) -> None:
        bundle = self._first_cycle_fixture()
        broken_events = tuple(
            replace(event, victim_expert_id=1)
            if event.kind is RuntimeEventKind.EVICT
            else event
            for event in bundle.runtime_events
        )
        broken = replace(bundle, runtime_events=broken_events)

        with self.assertRaises(LogicalReplayError):
            replay_qwen3_chain(broken)

    def test_target_route_change_cannot_be_hidden_by_recorded_events(self) -> None:
        bundle = self._first_cycle_fixture()
        changed_routes = tuple(
            replace(route, expert_ids=(0, 3))
            if route.node_id == 1
            else route
            for route in bundle.target_routes
        )
        broken = replace(bundle, target_routes=changed_routes)

        with self.assertRaises(LogicalReplayError):
            replay_qwen3_chain(broken)

    def test_non_quiescent_snapshot_boundary_is_rejected(self) -> None:
        bundle = self._first_cycle_fixture()
        cycle = bundle.cycles[0]
        before = next(
            snapshot
            for snapshot in bundle.snapshots
            if snapshot.trace_id == cycle.pre_snapshot_id
        )
        layer = before.layers[0]
        busy_slot = replace(layer.slots[0], active_leases=1)
        busy_layer = replace(
            layer,
            slots=(busy_slot, layer.slots[1]),
        )
        busy_before = replace(before, layers=(busy_layer,))
        snapshots = tuple(
            busy_before if item.trace_id == before.trace_id else item
            for item in bundle.snapshots
        )
        broken = replace(bundle, snapshots=snapshots)

        with self.assertRaises(LogicalReplayError):
            replay_qwen3_chain(broken)

    def test_next_cycle_pre_snapshot_must_continue_previous_post(self) -> None:
        bundle = self._two_cycle_discontinuous_fixture()

        with self.assertRaises(LogicalReplayError):
            replay_qwen3_chain(bundle)


if __name__ == "__main__":
    unittest.main()
