from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError, replace

from specfit import (
    CacheStatsTrace,
    CacheSlotTrace,
    CacheSnapshotTrace,
    CudaEventRole,
    CudaEventTrace,
    CycleTrace,
    DraftNodeTrace,
    DraftTreeTrace,
    ExpertProbabilityTrace,
    GenerationTrace,
    H2DTrace,
    LayerCacheSnapshotTrace,
    LayerRouteTrace,
    ManifestEntry,
    PendingTransferTrace,
    PredictorRouteTrace,
    RuntimeEventKind,
    RuntimeEventTrace,
    SnapshotPhase,
    TraceBundle,
    TargetRouterTopKTrace,
    VerificationNodeRole,
    WorkloadManifestTrace,
    stable_trace_id,
)


class TraceSchemaTest(unittest.TestCase):
    def manifest(self) -> WorkloadManifestTrace:
        return WorkloadManifestTrace(
            run_id="run-0",
            request_id="request-0",
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
            slots_per_layer=1,
            num_moe_layers=48,
            num_experts_per_layer=128,
            experts_per_token=8,
            draft_length=3,
            max_new_tokens=32,
            seed=7,
            raw_prompt="Tell me about the Moon.",
            rendered_prompt="<user>Tell me about the Moon.",
            input_ids=(10, 11),
            attention_mask=(1, 1),
            metadata=(
                ManifestEntry("transformers", "5.6.0"),
                ManifestEntry("torch", "2.10.0a0"),
            ),
        )

    def chain(
        self,
    ) -> tuple[DraftTreeTrace, CycleTrace, DraftNodeTrace, DraftNodeTrace]:
        tree_id = "tree-0"
        seed = DraftNodeTrace(
            tree_id=tree_id,
            node_id=0,
            parent_id=None,
            depth=0,
            role=VerificationNodeRole.SEED,
            token_id=100,
            conditional_probability=None,
            verification_position=0,
            selected=True,
            verified=True,
            accepted=True,
        )
        accepted_draft = DraftNodeTrace(
            tree_id=tree_id,
            node_id=1,
            parent_id=0,
            depth=1,
            role=VerificationNodeRole.DRAFT,
            token_id=101,
            conditional_probability=None,
            verification_position=1,
            selected=True,
            verified=True,
            accepted=True,
        )
        rejected_draft = DraftNodeTrace(
            tree_id=tree_id,
            node_id=2,
            parent_id=1,
            depth=2,
            role=VerificationNodeRole.DRAFT,
            token_id=102,
            conditional_probability=None,
            verification_position=2,
            selected=True,
            verified=True,
            accepted=False,
        )
        tree = DraftTreeTrace(
            tree_id=tree_id,
            request_id="request-0",
            iteration=0,
            nodes=(rejected_draft, seed, accepted_draft),
        )
        cycle = CycleTrace(
            run_id="run-0",
            tree_id=tree_id,
            request_id="request-0",
            iteration=0,
            base_kv_length=14,
            pending_seed_token_id=100,
            verification_token_ids=(100, 101, 102),
            verifier_token_ids=(101, 999, 999),
            accepted_draft_node_ids=(1,),
            committed_token_ids=(100, 101),
            kv_length_after=16,
        )
        return tree, cycle, seed, accepted_draft

    def test_chain_tree_is_canonical_and_validates_node_states(self) -> None:
        tree, _, seed, _ = self.chain()

        self.assertEqual(tuple(node.node_id for node in tree.nodes), (0, 1, 2))
        self.assertEqual(
            seed.trace_id,
            stable_trace_id("node", tree.tree_id, seed.node_id),
        )
        with self.assertRaises(FrozenInstanceError):
            seed.node_id = 9  # type: ignore[misc]
        with self.assertRaisesRegex(ValueError, "missing parent"):
            DraftTreeTrace(
                tree_id=tree.tree_id,
                request_id=tree.request_id,
                iteration=0,
                nodes=(
                    seed,
                    DraftNodeTrace(
                        tree_id=tree.tree_id,
                        node_id=3,
                        parent_id=99,
                        depth=1,
                        role=VerificationNodeRole.DRAFT,
                        token_id=103,
                        conditional_probability=0.5,
                        verification_position=None,
                        selected=False,
                        verified=False,
                        accepted=False,
                    ),
                ),
            )

        unverified_parent = DraftNodeTrace(
            tree_id="tree-verified-closure",
            node_id=1,
            parent_id=0,
            depth=1,
            role=VerificationNodeRole.DRAFT,
            token_id=101,
            conditional_probability=0.5,
            verification_position=None,
            selected=True,
            verified=False,
            accepted=False,
        )
        verified_child = DraftNodeTrace(
            tree_id="tree-verified-closure",
            node_id=2,
            parent_id=1,
            depth=2,
            role=VerificationNodeRole.DRAFT,
            token_id=102,
            conditional_probability=0.5,
            verification_position=1,
            selected=True,
            verified=True,
            accepted=False,
        )
        closure_root = replace(seed, tree_id="tree-verified-closure")
        with self.assertRaisesRegex(ValueError, "verified nodes.*ancestor"):
            DraftTreeTrace(
                tree_id="tree-verified-closure",
                request_id=tree.request_id,
                iteration=0,
                nodes=(closure_root, unverified_parent, verified_child),
            )

    def test_target_routes_and_predictor_routes_remain_separate(self) -> None:
        tree, cycle, seed, _ = self.chain()
        weights = (0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.06, 0.04)
        target = LayerRouteTrace(
            tree_id=tree.tree_id,
            node_id=seed.node_id,
            layer_id=0,
            expert_ids=(0, 1, 2, 3, 4, 5, 6, 7),
            expert_weights=weights,
        )
        predictor = PredictorRouteTrace(
            tree_id=tree.tree_id,
            node_id=2,
            layer_id=0,
            experts=(
                ExpertProbabilityTrace(9, 0.7),
                ExpertProbabilityTrace(3, 0.2),
            ),
        )
        target_topk = TargetRouterTopKTrace(
            tree_id=tree.tree_id,
            node_id=seed.node_id,
            layer_id=0,
            experts=(
                ExpertProbabilityTrace(0, 0.3),
                ExpertProbabilityTrace(1, 0.2),
            ),
        )
        bundle = TraceBundle(
            manifest=self.manifest(),
            trees=(tree,),
            cycles=(cycle,),
            target_routes=(target,),
            target_router_topk=(target_topk,),
            predictor_routes=(predictor,),
        )

        self.assertEqual(bundle.target_routes[0].expert_weights, weights)
        self.assertNotEqual(target.trace_id, predictor.trace_id)
        verified_root = DraftNodeTrace(
            tree_id="tree-unverified",
            node_id=0,
            parent_id=None,
            depth=0,
            role=VerificationNodeRole.SEED,
            token_id=100,
            conditional_probability=None,
            verification_position=0,
            selected=True,
            verified=True,
            accepted=True,
        )
        unverified = DraftNodeTrace(
            tree_id="tree-unverified",
            node_id=1,
            parent_id=0,
            depth=1,
            role=VerificationNodeRole.DRAFT,
            token_id=101,
            conditional_probability=0.4,
            verification_position=None,
            selected=False,
            verified=False,
            accepted=False,
        )
        unverified_tree = DraftTreeTrace(
            tree_id="tree-unverified",
            request_id="request-0",
            iteration=0,
            nodes=(verified_root, unverified),
        )
        unverified_cycle = CycleTrace(
            run_id="run-0",
            tree_id=unverified_tree.tree_id,
            request_id="request-0",
            iteration=0,
            base_kv_length=1,
            pending_seed_token_id=100,
            verification_token_ids=(100,),
            verifier_token_ids=(101,),
            accepted_draft_node_ids=(),
            committed_token_ids=(100,),
            kv_length_after=2,
        )
        with self.assertRaisesRegex(ValueError, "unverified node"):
            TraceBundle(
                manifest=self.manifest(),
                trees=(unverified_tree,),
                cycles=(unverified_cycle,),
                target_routes=(
                    LayerRouteTrace(
                        tree_id=unverified_tree.tree_id,
                        node_id=1,
                        layer_id=0,
                        expert_ids=(0, 1, 2, 3, 4, 5, 6, 7),
                        expert_weights=weights,
                    ),
                ),
            )

    def test_generation_record_reconstructs_chain_tail_and_target_only(
        self,
    ) -> None:
        tree, cycle, _, _ = self.chain()
        chain_generation = GenerationTrace(
            run_id="run-0",
            request_id="request-0",
            output_token_ids=(100, 101, 999),
            cycle_ids=(cycle.trace_id,),
            residual_token_ids=(999,),
            target_forward_calls=2,
        )
        chain_bundle = TraceBundle(
            manifest=replace(
                self.manifest(),
                max_new_tokens=3,
            ),
            trees=(tree,),
            cycles=(cycle,),
            generations=(chain_generation,),
        )

        self.assertEqual(
            TraceBundle.from_json(chain_bundle.to_json()),
            chain_bundle,
        )
        self.assertEqual(
            TraceBundle.from_jsonl(chain_bundle.to_jsonl()),
            chain_bundle,
        )
        legacy_payload = json.loads(chain_bundle.to_json())
        legacy_generation = legacy_payload["generations"][0]
        legacy_generation.pop("stopped_on_eos")
        legacy_generation.pop("stopped_on_cycle_limit")
        legacy_generation.pop("stop_token_id")
        self.assertEqual(
            TraceBundle.from_json(json.dumps(legacy_payload)),
            chain_bundle,
        )

        target_only_generation = GenerationTrace(
            run_id="run-0",
            request_id="request-0",
            output_token_ids=(200, 201, 202),
            cycle_ids=(),
            residual_token_ids=(200, 201, 202),
            target_forward_calls=3,
        )
        target_only_bundle = TraceBundle(
            manifest=replace(
                self.manifest(),
                draft_length=0,
                max_new_tokens=3,
            ),
            trees=(),
            cycles=(),
            generations=(target_only_generation,),
        )
        self.assertEqual(
            TraceBundle.from_jsonl(target_only_bundle.to_jsonl()),
            target_only_bundle,
        )
        self.assertFalse(chain_generation.stopped_on_eos)
        self.assertIsNone(chain_generation.stop_token_id)

        missing_tail = replace(
            chain_generation,
            residual_token_ids=(),
        )
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            TraceBundle(
                manifest=chain_bundle.manifest,
                trees=(tree,),
                cycles=(cycle,),
                generations=(missing_tail,),
            )

        wrong_tail = replace(
            chain_generation,
            output_token_ids=(100, 101, 103),
            residual_token_ids=(103,),
        )
        with self.assertRaisesRegex(ValueError, "pending seed"):
            TraceBundle(
                manifest=chain_bundle.manifest,
                trees=(tree,),
                cycles=(cycle,),
                generations=(wrong_tail,),
            )

        wrong_forward_count = replace(
            chain_generation,
            target_forward_calls=3,
        )
        with self.assertRaisesRegex(ValueError, "target forward"):
            TraceBundle(
                manifest=chain_bundle.manifest,
                trees=(tree,),
                cycles=(cycle,),
                generations=(wrong_forward_count,),
            )

        with self.assertRaisesRegex(ValueError, "target-only"):
            TraceBundle(
                manifest=replace(
                    self.manifest(),
                    draft_length=0,
                ),
                trees=(tree,),
                cycles=(cycle,),
            )

    def test_generation_requires_an_explicit_eos_stop_for_short_output(
        self,
    ) -> None:
        short_generation = GenerationTrace(
            run_id="run-0",
            request_id="request-0",
            output_token_ids=(200, 201),
            cycle_ids=(),
            residual_token_ids=(200, 201),
            target_forward_calls=2,
        )

        with self.assertRaises(ValueError):
            TraceBundle(
                manifest=replace(
                    self.manifest(),
                    draft_length=0,
                    max_new_tokens=3,
                ),
                trees=(),
                cycles=(),
                generations=(short_generation,),
            )

    def test_generation_allows_an_explicit_eos_stop_before_budget(
        self,
    ) -> None:
        eos_generation = GenerationTrace(
            run_id="run-0",
            request_id="request-0",
            output_token_ids=(200, 201),
            cycle_ids=(),
            residual_token_ids=(200, 201),
            target_forward_calls=2,
            stopped_on_eos=True,
            stop_token_id=201,
        )
        bundle = TraceBundle(
            manifest=replace(
                self.manifest(),
                draft_length=0,
                max_new_tokens=3,
            ),
            trees=(),
            cycles=(),
            generations=(eos_generation,),
        )

        self.assertEqual(TraceBundle.from_jsonl(bundle.to_jsonl()), bundle)

    def test_generation_allows_an_explicit_cycle_limit_before_budget(
        self,
    ) -> None:
        tree, cycle, _, _ = self.chain()
        generation = GenerationTrace(
            run_id="run-0",
            request_id="request-0",
            output_token_ids=(100, 101),
            cycle_ids=(cycle.trace_id,),
            residual_token_ids=(),
            target_forward_calls=2,
            stopped_on_cycle_limit=True,
        )
        bundle = TraceBundle(
            manifest=replace(self.manifest(), max_new_tokens=3),
            trees=(tree,),
            cycles=(cycle,),
            generations=(generation,),
        )

        self.assertEqual(TraceBundle.from_jsonl(bundle.to_jsonl()), bundle)
        with self.assertRaisesRegex(ValueError, "residual tokens"):
            TraceBundle(
                manifest=bundle.manifest,
                trees=(tree,),
                cycles=(cycle,),
                generations=(
                    replace(generation, residual_token_ids=(999,)),
                ),
            )
        with self.assertRaisesRegex(ValueError, "below budget"):
            TraceBundle(
                manifest=bundle.manifest,
                trees=(tree,),
                cycles=(cycle,),
                generations=(
                    replace(
                        generation,
                        output_token_ids=(100, 101, 999),
                        residual_token_ids=(),
                    ),
                ),
            )
        with self.assertRaisesRegex(ValueError, "EOS and cycle limit"):
            replace(
                generation,
                stopped_on_eos=True,
                stop_token_id=101,
            )
        with self.assertRaisesRegex(ValueError, "requires a cycle"):
            replace(generation, cycle_ids=())

    def test_generation_rejects_a_stop_token_different_from_the_tail(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            GenerationTrace(
                run_id="run-0",
                request_id="request-0",
                output_token_ids=(200, 201),
                cycle_ids=(),
                residual_token_ids=(200, 201),
                target_forward_calls=2,
                stopped_on_eos=True,
                stop_token_id=202,
            )

    def test_explicit_eos_stop_allows_terminal_tree_truncation(self) -> None:
        tree, cycle, _, _ = self.chain()
        eos_cycle = replace(
            cycle,
            verifier_token_ids=(101, 102, 999),
        )
        eos_generation = GenerationTrace(
            run_id="run-0",
            request_id="request-0",
            output_token_ids=(100, 101),
            cycle_ids=(eos_cycle.trace_id,),
            residual_token_ids=(),
            target_forward_calls=2,
            stopped_on_eos=True,
            stop_token_id=101,
        )
        bundle = TraceBundle(
            manifest=replace(self.manifest(), max_new_tokens=3),
            trees=(tree,),
            cycles=(eos_cycle,),
            generations=(eos_generation,),
        )

        self.assertEqual(TraceBundle.from_jsonl(bundle.to_jsonl()), bundle)

    def test_full_trace_round_trips_with_48_layer_queues(self) -> None:
        tree, cycle, seed, _ = self.chain()
        start = CudaEventTrace(
            run_id="run-0",
            sequence=0,
            stream_id="layer-0-copy",
            role=CudaEventRole.H2D_START,
            elapsed_from_origin_ns=100,
        )
        end = CudaEventTrace(
            run_id="run-0",
            sequence=1,
            stream_id="layer-0-copy",
            role=CudaEventRole.H2D_END,
            elapsed_from_origin_ns=200,
        )
        h2d = H2DTrace(
            run_id="run-0",
            submission_sequence=0,
            layer_id=0,
            expert_id=9,
            slot_id=0,
            generation=1,
            bytes=9 * 2**20,
            host_submit_ns=80,
            start_event_id=start.trace_id,
            end_event_id=end.trace_id,
            tree_id=tree.tree_id,
            cycle_id=cycle.trace_id,
        )
        layers = tuple(
            LayerCacheSnapshotTrace(
                layer_id=layer_id,
                cache_id=f"layer-{layer_id}-cache",
                queue_id=f"layer-{layer_id}-copy",
                capacity=1,
                host_state_version=1 if layer_id == 0 else 0,
                lru_sequence=1 if layer_id == 0 else 0,
                submission_sequence=1 if layer_id == 0 else 0,
                slots=(
                    CacheSlotTrace(
                        slot_id=0,
                        expert_id=9 if layer_id == 0 else None,
                        generation=1 if layer_id == 0 else 0,
                        active_leases=0,
                        last_used_sequence=1 if layer_id == 0 else 0,
                        ready=False,
                        pending_use_guards=0,
                    ),
                ),
                stats=CacheStatsTrace(
                    h2d_submissions=1 if layer_id == 0 else 0,
                    resident_hits=0,
                    inflight_joins=0,
                    evictions=0,
                    pending_use_guards=0,
                ),
                pending_transfers=(
                    (
                        PendingTransferTrace(
                            transfer_id=h2d.trace_id,
                            expert_id=9,
                            slot_id=0,
                            generation=1,
                            submission_sequence=0,
                            ready_event_id=end.trace_id,
                            ready_event_completed=False,
                        ),
                    )
                    if layer_id == 0
                    else ()
                ),
            )
            for layer_id in reversed(range(48))
        )
        before = CacheSnapshotTrace(
            run_id="run-0",
            tree_id=tree.tree_id,
            cycle_id=cycle.trace_id,
            phase=SnapshotPhase.BEFORE_VERIFY,
            runtime_state_version=1,
            captured_at_ns=90,
            layers=layers,
        )
        cycle = CycleTrace(
            run_id=cycle.run_id,
            tree_id=cycle.tree_id,
            request_id=cycle.request_id,
            iteration=cycle.iteration,
            base_kv_length=cycle.base_kv_length,
            pending_seed_token_id=cycle.pending_seed_token_id,
            verification_token_ids=cycle.verification_token_ids,
            verifier_token_ids=cycle.verifier_token_ids,
            accepted_draft_node_ids=cycle.accepted_draft_node_ids,
            committed_token_ids=cycle.committed_token_ids,
            kv_length_after=cycle.kv_length_after,
            pre_snapshot_id=before.trace_id,
        )
        runtime_event = RuntimeEventTrace(
            run_id="run-0",
            sequence=0,
            kind=RuntimeEventKind.H2D_SUBMIT,
            host_time_ns=80,
            layer_id=0,
            cache_id="layer-0-cache",
            queue_id="layer-0-copy",
            expert_id=9,
            slot_id=0,
            generation=1,
            submission_sequence=0,
            tree_id=tree.tree_id,
            cycle_id=cycle.trace_id,
            transfer_id=h2d.trace_id,
            cuda_event_ids=(start.trace_id, end.trace_id),
        )
        target = LayerRouteTrace(
            tree_id=tree.tree_id,
            node_id=seed.node_id,
            layer_id=0,
            expert_ids=(0, 1, 2, 3, 4, 5, 6, 7),
            expert_weights=(0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.06, 0.04),
        )
        bundle = TraceBundle(
            manifest=self.manifest(),
            trees=(tree,),
            cycles=(cycle,),
            target_routes=(target,),
            snapshots=(before,),
            cuda_events=(end, start),
            h2d_transfers=(h2d,),
            runtime_events=(runtime_event,),
        )

        json_round_trip = TraceBundle.from_json(bundle.to_json())
        jsonl_round_trip = TraceBundle.from_jsonl(bundle.to_jsonl())

        self.assertEqual(json_round_trip, bundle)
        self.assertEqual(jsonl_round_trip, bundle)
        self.assertEqual(
            tuple(layer.layer_id for layer in bundle.snapshots[0].layers),
            tuple(range(48)),
        )
        self.assertEqual(bundle.to_json(), json_round_trip.to_json())
        self.assertEqual(bundle.to_jsonl(), jsonl_round_trip.to_jsonl())

        legacy = json.loads(bundle.to_json())
        for legacy_layer in legacy["snapshots"][0]["layers"]:
            for field in (
                "cache_id",
                "host_state_version",
                "lru_sequence",
                "stats",
            ):
                del legacy_layer[field]
        legacy_event = legacy["runtime_events"][0]
        for field in (
            "cache_id",
            "queue_id",
            "submission_sequence",
            "stream_ids",
        ):
            del legacy_event[field]
        legacy_bundle = TraceBundle.from_json(json.dumps(legacy))
        self.assertIsNone(legacy_bundle.snapshots[0].layers[0].cache_id)
        self.assertIsNone(legacy_bundle.snapshots[0].layers[0].stats)
        self.assertIsNone(legacy_bundle.runtime_events[0].cache_id)
        self.assertIsNone(
            legacy_bundle.runtime_events[0].submission_sequence
        )

        layer_zero = before.layers[0]
        pending = layer_zero.pending_transfers[0]
        logical_pending_layer = replace(
            layer_zero,
            pending_transfers=(replace(pending, ready_event_id=None),),
        )
        logical_pending_snapshot = replace(
            before,
            layers=(logical_pending_layer, *before.layers[1:]),
        )
        TraceBundle(
            manifest=self.manifest(),
            trees=(tree,),
            cycles=(cycle,),
            target_routes=(target,),
            snapshots=(logical_pending_snapshot,),
            cuda_events=(start, end),
            h2d_transfers=(h2d,),
            runtime_events=(runtime_event,),
        )

        broken_reference = json.loads(bundle.to_json())
        broken_reference["h2d_transfers"][0]["end_event_id"] = "missing-event"
        with self.assertRaisesRegex(ValueError, "unknown CUDA event"):
            TraceBundle.from_json(json.dumps(broken_reference))

        wrong_phase = replace(before, phase=SnapshotPhase.AFTER_VERIFY)
        wrong_cycle = replace(cycle, pre_snapshot_id=wrong_phase.trace_id)
        with self.assertRaisesRegex(ValueError, "wrong scope or phase"):
            TraceBundle(
                manifest=self.manifest(),
                trees=(tree,),
                cycles=(wrong_cycle,),
                target_routes=(target,),
                snapshots=(wrong_phase,),
                cuda_events=(start, end),
                h2d_transfers=(h2d,),
                runtime_events=(runtime_event,),
            )

        foreign_start = replace(start, run_id="foreign-run")
        foreign_h2d = replace(h2d, start_event_id=foreign_start.trace_id)
        with self.assertRaisesRegex(ValueError, "different run|wrong identity"):
            TraceBundle(
                manifest=self.manifest(),
                trees=(tree,),
                cycles=(cycle,),
                target_routes=(target,),
                snapshots=(before,),
                cuda_events=(foreign_start, end),
                h2d_transfers=(foreign_h2d,),
                runtime_events=(),
            )

        unsnapped_cycle = replace(cycle, pre_snapshot_id=None)
        late_start = replace(start, sequence=2)
        early_end = replace(end, sequence=1)
        reversed_h2d = replace(
            h2d,
            start_event_id=late_start.trace_id,
            end_event_id=early_end.trace_id,
        )
        with self.assertRaisesRegex(ValueError, "wrong identity"):
            TraceBundle(
                manifest=self.manifest(),
                trees=(tree,),
                cycles=(unsnapped_cycle,),
                target_routes=(target,),
                cuda_events=(late_start, early_end),
                h2d_transfers=(reversed_h2d,),
            )

        wrong_role = replace(start, role=CudaEventRole.CYCLE_START)
        with self.assertRaisesRegex(ValueError, "wrong identity"):
            TraceBundle(
                manifest=self.manifest(),
                trees=(tree,),
                cycles=(unsnapped_cycle,),
                target_routes=(target,),
                cuda_events=(wrong_role, end),
                h2d_transfers=(h2d,),
            )

        mismatched_pending = replace(
            pending,
            ready_event_id=start.trace_id,
        )
        mismatched_layer = replace(
            layer_zero,
            pending_transfers=(mismatched_pending,),
        )
        mismatched_snapshot = replace(
            before,
            layers=(mismatched_layer, *before.layers[1:]),
        )
        with self.assertRaisesRegex(
            ValueError, "pending transfer does not match"
        ):
            TraceBundle(
                manifest=self.manifest(),
                trees=(tree,),
                cycles=(cycle,),
                target_routes=(target,),
                snapshots=(mismatched_snapshot,),
                cuda_events=(start, end),
                h2d_transfers=(h2d,),
            )

    def test_logical_cache_trace_round_trips_without_cuda_timing(self) -> None:
        tree, cycle, _, _ = self.chain()
        transfer_id = stable_trace_id("h2d", "run-0", 0, 0)
        layer = LayerCacheSnapshotTrace(
            layer_id=0,
            cache_id="layer-0-cache",
            queue_id="layer-0-copy",
            capacity=1,
            host_state_version=3,
            lru_sequence=2,
            submission_sequence=1,
            slots=(
                CacheSlotTrace(
                    slot_id=0,
                    expert_id=9,
                    generation=1,
                    active_leases=0,
                    last_used_sequence=2,
                    ready=False,
                    pending_use_guards=0,
                ),
            ),
            pending_transfers=(
                PendingTransferTrace(
                    transfer_id=transfer_id,
                    expert_id=9,
                    slot_id=0,
                    generation=1,
                    submission_sequence=0,
                    ready_event_id=None,
                    ready_event_completed=False,
                ),
            ),
            stats=CacheStatsTrace(
                h2d_submissions=1,
                resident_hits=0,
                inflight_joins=0,
                evictions=0,
                pending_use_guards=0,
            ),
        )
        snapshot = CacheSnapshotTrace(
            run_id="run-0",
            tree_id=tree.tree_id,
            cycle_id=cycle.trace_id,
            phase=SnapshotPhase.BEFORE_VERIFY,
            runtime_state_version=3,
            captured_at_ns=90,
            layers=(layer,),
        )
        cycle = replace(cycle, pre_snapshot_id=snapshot.trace_id)
        submit = RuntimeEventTrace(
            run_id="run-0",
            sequence=0,
            kind=RuntimeEventKind.H2D_SUBMIT,
            host_time_ns=80,
            layer_id=0,
            cache_id="layer-0-cache",
            queue_id="layer-0-copy",
            expert_id=9,
            slot_id=0,
            generation=1,
            submission_sequence=0,
            tree_id=tree.tree_id,
            cycle_id=cycle.trace_id,
            transfer_id=transfer_id,
        )

        bundle = TraceBundle(
            manifest=self.manifest(),
            trees=(tree,),
            cycles=(cycle,),
            snapshots=(snapshot,),
            runtime_events=(submit,),
        )

        self.assertEqual(bundle.cuda_events, ())
        self.assertEqual(bundle.h2d_transfers, ())
        self.assertEqual(TraceBundle.from_json(bundle.to_json()), bundle)
        self.assertEqual(TraceBundle.from_jsonl(bundle.to_jsonl()), bundle)

        with self.assertRaisesRegex(ValueError, "pending transfer does not match"):
            TraceBundle(
                manifest=self.manifest(),
                trees=(tree,),
                cycles=(cycle,),
                snapshots=(snapshot,),
                runtime_events=(replace(submit, host_time_ns=100),),
            )

        legacy = json.loads(bundle.to_json())
        legacy_layer = legacy["snapshots"][0]["layers"][0]
        for field in (
            "cache_id",
            "host_state_version",
            "lru_sequence",
            "stats",
        ):
            del legacy_layer[field]
        legacy_event = legacy["runtime_events"][0]
        for field in (
            "cache_id",
            "queue_id",
            "stream_ids",
        ):
            del legacy_event[field]
        legacy_bundle = TraceBundle.from_json(json.dumps(legacy))
        self.assertIsNone(legacy_bundle.snapshots[0].layers[0].cache_id)
        self.assertIsNone(legacy_bundle.snapshots[0].layers[0].stats)
        self.assertIsNone(legacy_bundle.runtime_events[0].cache_id)

    def test_logical_h2d_order_join_and_eviction_are_validated(self) -> None:
        tree, cycle, _, _ = self.chain()

        def submit(
            *, sequence: int, submission: int, generation: int
        ) -> RuntimeEventTrace:
            return RuntimeEventTrace(
                run_id="run-0",
                sequence=sequence,
                kind=RuntimeEventKind.H2D_SUBMIT,
                host_time_ns=80 + sequence,
                layer_id=0,
                cache_id="layer-0-cache",
                queue_id="layer-0-copy",
                expert_id=9 + submission,
                slot_id=0,
                generation=generation,
                submission_sequence=submission,
                transfer_id=stable_trace_id(
                    "h2d", "run-0", 0, submission
                ),
            )

        with self.assertRaisesRegex(ValueError, "layer order"):
            TraceBundle(
                manifest=self.manifest(),
                trees=(tree,),
                cycles=(cycle,),
                runtime_events=(
                    submit(sequence=0, submission=1, generation=2),
                    submit(sequence=1, submission=0, generation=1),
                ),
            )

        later_submit = submit(sequence=1, submission=0, generation=1)
        early_join = RuntimeEventTrace(
            run_id="run-0",
            sequence=0,
            kind=RuntimeEventKind.INFLIGHT_JOIN,
            host_time_ns=79,
            layer_id=0,
            cache_id="layer-0-cache",
            queue_id="layer-0-copy",
            expert_id=9,
            slot_id=0,
            generation=1,
            submission_sequence=0,
            transfer_id=later_submit.transfer_id,
        )
        with self.assertRaisesRegex(ValueError, "join must follow"):
            TraceBundle(
                manifest=self.manifest(),
                trees=(tree,),
                cycles=(cycle,),
                runtime_events=(early_join, later_submit),
            )

        submit_first = submit(sequence=0, submission=0, generation=1)
        mismatched_join = replace(
            early_join,
            sequence=1,
            host_time_ns=81,
            expert_id=10,
            transfer_id=submit_first.transfer_id,
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            TraceBundle(
                manifest=self.manifest(),
                trees=(tree,),
                cycles=(cycle,),
                runtime_events=(submit_first, mismatched_join),
            )

        with self.assertRaisesRegex(ValueError, "eviction requires"):
            RuntimeEventTrace(
                run_id="run-0",
                sequence=0,
                kind=RuntimeEventKind.EVICT,
                host_time_ns=80,
                layer_id=0,
                cache_id="layer-0-cache",
                queue_id="layer-0-copy",
                expert_id=9,
                slot_id=0,
                generation=1,
            )

    def test_snapshot_accepts_an_initial_pending_transfer(self) -> None:
        tree, cycle, _, _ = self.chain()
        pending = PendingTransferTrace(
            transfer_id=stable_trace_id("h2d", "run-0", 0, 0),
            expert_id=9,
            slot_id=0,
            generation=1,
            submission_sequence=0,
            ready_event_id=None,
            ready_event_completed=False,
        )
        layer = LayerCacheSnapshotTrace(
            layer_id=0,
            cache_id="layer-0-cache",
            queue_id="layer-0-copy",
            capacity=1,
            host_state_version=1,
            lru_sequence=1,
            submission_sequence=1,
            slots=(
                CacheSlotTrace(
                    slot_id=0,
                    expert_id=9,
                    generation=1,
                    active_leases=0,
                    last_used_sequence=1,
                    ready=False,
                    pending_use_guards=0,
                ),
            ),
            stats=CacheStatsTrace(1, 0, 0, 0, 0),
            pending_transfers=(pending,),
        )
        snapshot = CacheSnapshotTrace(
            run_id="run-0",
            tree_id=tree.tree_id,
            cycle_id=cycle.trace_id,
            phase=SnapshotPhase.BEFORE_VERIFY,
            runtime_state_version=1,
            captured_at_ns=90,
            layers=(layer,),
        )
        cycle = replace(cycle, pre_snapshot_id=snapshot.trace_id)

        bundle = TraceBundle(
            manifest=self.manifest(),
            trees=(tree,),
            cycles=(cycle,),
            snapshots=(snapshot,),
        )

        self.assertEqual(
            bundle.snapshots[0].layers[0].pending_transfers,
            (pending,),
        )

        initial_join = RuntimeEventTrace(
            run_id="run-0",
            sequence=0,
            kind=RuntimeEventKind.INFLIGHT_JOIN,
            host_time_ns=91,
            layer_id=0,
            cache_id="layer-0-cache",
            queue_id="layer-0-copy",
            expert_id=9,
            slot_id=0,
            generation=1,
            submission_sequence=0,
            tree_id=tree.tree_id,
            cycle_id=cycle.trace_id,
            transfer_id=pending.transfer_id,
        )
        TraceBundle(
            manifest=self.manifest(),
            trees=(tree,),
            cycles=(cycle,),
            snapshots=(snapshot,),
            runtime_events=(initial_join,),
        )

        later_tree_id = "tree-later-pending"
        later_tree = replace(
            tree,
            tree_id=later_tree_id,
            iteration=1,
            nodes=tuple(
                replace(node, tree_id=later_tree_id) for node in tree.nodes
            ),
        )
        later_cycle = replace(
            cycle,
            tree_id=later_tree_id,
            iteration=1,
            pre_snapshot_id=None,
        )
        later_snapshot = replace(
            snapshot,
            tree_id=later_tree_id,
            cycle_id=later_cycle.trace_id,
            captured_at_ns=110,
        )
        later_cycle = replace(
            later_cycle,
            pre_snapshot_id=later_snapshot.trace_id,
        )
        TraceBundle(
            manifest=self.manifest(),
            trees=(tree, later_tree),
            cycles=(cycle, later_cycle),
            snapshots=(snapshot, later_snapshot),
            runtime_events=(initial_join,),
        )
        mutated_pending = replace(
            pending,
            expert_id=10,
            generation=2,
        )
        mutated_layer = replace(
            layer,
            slots=(
                replace(
                    layer.slots[0],
                    expert_id=10,
                    generation=2,
                ),
            ),
            pending_transfers=(mutated_pending,),
        )
        with self.assertRaisesRegex(ValueError, "changes identity"):
            TraceBundle(
                manifest=self.manifest(),
                trees=(tree, later_tree),
                cycles=(cycle, later_cycle),
                snapshots=(
                    snapshot,
                    replace(later_snapshot, layers=(mutated_layer,)),
                ),
                runtime_events=(initial_join,),
            )
        with self.assertRaisesRegex(ValueError, "initial pending"):
            TraceBundle(
                manifest=self.manifest(),
                trees=(tree,),
                cycles=(cycle,),
                snapshots=(snapshot,),
                runtime_events=(replace(initial_join, expert_id=10),),
            )

    def test_timed_transfer_can_be_joined_by_a_later_cycle(self) -> None:
        first_tree, first_cycle, _, _ = self.chain()
        second_tree_id = "tree-1"
        second_tree = replace(
            first_tree,
            tree_id=second_tree_id,
            iteration=1,
            nodes=tuple(
                replace(node, tree_id=second_tree_id)
                for node in first_tree.nodes
            ),
        )
        second_cycle = replace(
            first_cycle,
            tree_id=second_tree_id,
            iteration=1,
        )
        start = CudaEventTrace(
            run_id="run-0",
            sequence=0,
            stream_id="layer-0-copy",
            role=CudaEventRole.H2D_START,
            elapsed_from_origin_ns=10,
        )
        end = CudaEventTrace(
            run_id="run-0",
            sequence=1,
            stream_id="layer-0-copy",
            role=CudaEventRole.H2D_END,
            elapsed_from_origin_ns=20,
        )
        transfer = H2DTrace(
            run_id="run-0",
            submission_sequence=0,
            layer_id=0,
            expert_id=9,
            slot_id=0,
            generation=1,
            bytes=16,
            host_submit_ns=5,
            start_event_id=start.trace_id,
            end_event_id=end.trace_id,
            tree_id=first_tree.tree_id,
            cycle_id=first_cycle.trace_id,
        )
        later_join = RuntimeEventTrace(
            run_id="run-0",
            sequence=0,
            kind=RuntimeEventKind.INFLIGHT_JOIN,
            host_time_ns=30,
            layer_id=0,
            expert_id=9,
            slot_id=0,
            generation=1,
            submission_sequence=0,
            tree_id=second_tree.tree_id,
            cycle_id=second_cycle.trace_id,
            transfer_id=transfer.trace_id,
        )
        TraceBundle(
            manifest=self.manifest(),
            trees=(first_tree, second_tree),
            cycles=(first_cycle, second_cycle),
            cuda_events=(start, end),
            h2d_transfers=(transfer,),
            runtime_events=(later_join,),
        )

        with self.assertRaisesRegex(ValueError, "does not match"):
            TraceBundle(
                manifest=self.manifest(),
                trees=(first_tree, second_tree),
                cycles=(first_cycle, second_cycle),
                cuda_events=(start, end),
                h2d_transfers=(transfer,),
                runtime_events=(
                    replace(
                        later_join,
                        kind=RuntimeEventKind.H2D_SUBMIT,
                    ),
                ),
            )

    def test_unscoped_join_can_reference_an_external_prefill_transfer(
        self,
    ) -> None:
        tree, cycle, _, _ = self.chain()
        transfer_id = stable_trace_id("h2d", "run-0", 0, 7)
        external_join = RuntimeEventTrace(
            run_id="run-0",
            sequence=0,
            kind=RuntimeEventKind.INFLIGHT_JOIN,
            host_time_ns=80,
            layer_id=0,
            cache_id="layer-0-cache",
            queue_id="layer-0-copy",
            expert_id=9,
            slot_id=0,
            generation=3,
            submission_sequence=7,
            transfer_id=transfer_id,
        )

        bundle = TraceBundle(
            manifest=self.manifest(),
            trees=(tree,),
            cycles=(cycle,),
            runtime_events=(external_join,),
        )
        self.assertEqual(bundle.runtime_events, (external_join,))

        with self.assertRaisesRegex(ValueError, "unknown logical H2D"):
            TraceBundle(
                manifest=self.manifest(),
                trees=(tree,),
                cycles=(cycle,),
                runtime_events=(
                    replace(
                        external_join,
                        tree_id=tree.tree_id,
                        cycle_id=cycle.trace_id,
                    ),
                ),
            )

    def test_layout_bounds_cover_snapshots_and_standalone_runtime_events(
        self,
    ) -> None:
        tree, cycle, _, _ = self.chain()
        valid_slot = CacheSlotTrace(
            slot_id=0,
            expert_id=9,
            generation=1,
            active_leases=0,
            last_used_sequence=1,
            ready=True,
            pending_use_guards=0,
        )
        layer = LayerCacheSnapshotTrace(
            layer_id=0,
            cache_id="layer-0-cache",
            queue_id="layer-0-copy",
            capacity=1,
            host_state_version=1,
            lru_sequence=1,
            submission_sequence=1,
            slots=(valid_slot,),
            stats=CacheStatsTrace(
                h2d_submissions=1,
                resident_hits=0,
                inflight_joins=0,
                evictions=0,
                pending_use_guards=0,
            ),
        )
        snapshot = CacheSnapshotTrace(
            run_id="run-0",
            tree_id=tree.tree_id,
            cycle_id=cycle.trace_id,
            phase=SnapshotPhase.BEFORE_VERIFY,
            runtime_state_version=1,
            captured_at_ns=90,
            layers=(layer,),
        )
        cycle_with_snapshot = replace(cycle, pre_snapshot_id=snapshot.trace_id)

        out_of_range_slot = replace(valid_slot, expert_id=999)
        out_of_range_layer = replace(layer, slots=(out_of_range_slot,))
        out_of_range_snapshot = replace(
            snapshot,
            layers=(out_of_range_layer,),
        )
        with self.assertRaisesRegex(ValueError, "snapshot expert exceeds"):
            TraceBundle(
                manifest=self.manifest(),
                trees=(tree,),
                cycles=(cycle_with_snapshot,),
                snapshots=(out_of_range_snapshot,),
            )

        out_of_range_event = RuntimeEventTrace(
            run_id="run-0",
            sequence=0,
            kind=RuntimeEventKind.RESIDENT_HIT,
            host_time_ns=80,
            layer_id=999,
            cache_id="layer-999-cache",
            queue_id="layer-999-copy",
            expert_id=999,
            slot_id=999,
            generation=1,
            tree_id=tree.tree_id,
            cycle_id=cycle.trace_id,
        )
        with self.assertRaisesRegex(ValueError, "runtime event exceeds"):
            TraceBundle(
                manifest=self.manifest(),
                trees=(tree,),
                cycles=(cycle,),
                runtime_events=(out_of_range_event,),
            )

        target_only_manifest = replace(self.manifest(), draft_length=0)
        self.assertEqual(target_only_manifest.draft_length, 0)

    def test_cache_snapshot_allows_multiple_pending_slot_generations(
        self,
    ) -> None:
        first_slot = CacheSlotTrace(
            slot_id=0,
            expert_id=9,
            generation=1,
            active_leases=0,
            last_used_sequence=1,
            ready=False,
            pending_use_guards=0,
        )
        duplicate_resident = replace(first_slot, slot_id=1)
        with self.assertRaisesRegex(ValueError, "resident experts.*unique"):
            LayerCacheSnapshotTrace(
                layer_id=0,
                cache_id="layer-0-cache",
                queue_id="layer-0-copy",
                capacity=2,
                host_state_version=1,
                lru_sequence=1,
                submission_sequence=1,
                slots=(first_slot, duplicate_resident),
                stats=CacheStatsTrace(1, 0, 0, 0, 0),
            )

        first_pending = PendingTransferTrace(
            transfer_id="transfer-0",
            expert_id=9,
            slot_id=0,
            generation=1,
            submission_sequence=0,
            ready_event_id="ready-0",
            ready_event_completed=False,
        )
        current_slot = replace(
            first_slot,
            expert_id=10,
            generation=2,
        )
        second_generation_pending = replace(
            first_pending,
            transfer_id="transfer-1",
            expert_id=10,
            generation=2,
            submission_sequence=1,
            ready_event_id="ready-1",
        )
        queued = LayerCacheSnapshotTrace(
            layer_id=0,
            cache_id="layer-0-cache",
            queue_id="layer-0-copy",
            capacity=1,
            host_state_version=2,
            lru_sequence=1,
            submission_sequence=2,
            slots=(current_slot,),
            stats=CacheStatsTrace(2, 0, 0, 1, 0),
            pending_transfers=(
                first_pending,
                second_generation_pending,
            ),
        )
        self.assertEqual(
            tuple(
                transfer.submission_sequence
                for transfer in queued.pending_transfers
            ),
            (0, 1),
        )

        counter_reaching_pending = replace(
            first_pending,
            submission_sequence=2,
        )
        with self.assertRaisesRegex(ValueError, "snapshot submission counter"):
            LayerCacheSnapshotTrace(
                layer_id=0,
                cache_id="layer-0-cache",
                queue_id="layer-0-copy",
                capacity=1,
                host_state_version=2,
                lru_sequence=1,
                submission_sequence=2,
                slots=(current_slot,),
                stats=CacheStatsTrace(2, 0, 0, 1, 0),
                pending_transfers=(counter_reaching_pending,),
            )

        future_generation = replace(
            second_generation_pending,
            generation=3,
        )
        with self.assertRaisesRegex(ValueError, "generation exceeds"):
            LayerCacheSnapshotTrace(
                layer_id=0,
                cache_id="layer-0-cache",
                queue_id="layer-0-copy",
                capacity=1,
                host_state_version=2,
                lru_sequence=1,
                submission_sequence=2,
                slots=(current_slot,),
                stats=CacheStatsTrace(2, 0, 0, 1, 0),
                pending_transfers=(future_generation,),
            )

    def test_h2d_events_have_unique_ownership_and_follow_submission_order(
        self,
    ) -> None:
        tree, cycle, _, _ = self.chain()
        early_start = CudaEventTrace(
            run_id="run-0",
            sequence=0,
            stream_id="layer-0-copy",
            role=CudaEventRole.H2D_START,
            elapsed_from_origin_ns=100,
        )
        early_end = CudaEventTrace(
            run_id="run-0",
            sequence=1,
            stream_id="layer-0-copy",
            role=CudaEventRole.H2D_END,
            elapsed_from_origin_ns=200,
        )
        late_start = CudaEventTrace(
            run_id="run-0",
            sequence=2,
            stream_id="layer-0-copy",
            role=CudaEventRole.H2D_START,
            elapsed_from_origin_ns=300,
        )
        late_end = CudaEventTrace(
            run_id="run-0",
            sequence=3,
            stream_id="layer-0-copy",
            role=CudaEventRole.H2D_END,
            elapsed_from_origin_ns=400,
        )
        first = H2DTrace(
            run_id="run-0",
            submission_sequence=0,
            layer_id=0,
            expert_id=9,
            slot_id=0,
            generation=1,
            bytes=9 * 2**20,
            host_submit_ns=80,
            start_event_id=late_start.trace_id,
            end_event_id=late_end.trace_id,
            tree_id=tree.tree_id,
            cycle_id=cycle.trace_id,
        )
        second = H2DTrace(
            run_id="run-0",
            submission_sequence=1,
            layer_id=0,
            expert_id=10,
            slot_id=0,
            generation=2,
            bytes=9 * 2**20,
            host_submit_ns=90,
            start_event_id=early_start.trace_id,
            end_event_id=early_end.trace_id,
            tree_id=tree.tree_id,
            cycle_id=cycle.trace_id,
        )
        with self.assertRaisesRegex(ValueError, "submission order"):
            TraceBundle(
                manifest=self.manifest(),
                trees=(tree,),
                cycles=(cycle,),
                cuda_events=(
                    early_start,
                    early_end,
                    late_start,
                    late_end,
                ),
                h2d_transfers=(first, second),
            )

        shared_events = replace(first, layer_id=1)
        with self.assertRaisesRegex(ValueError, "unique H2D transfer"):
            TraceBundle(
                manifest=self.manifest(),
                trees=(tree,),
                cycles=(cycle,),
                cuda_events=(late_start, late_end),
                h2d_transfers=(first, shared_events),
            )

        ordered_first = replace(
            first,
            start_event_id=early_start.trace_id,
            end_event_id=early_end.trace_id,
        )
        repeated_generation = replace(
            second,
            start_event_id=late_start.trace_id,
            end_event_id=late_end.trace_id,
            generation=ordered_first.generation,
        )
        with self.assertRaisesRegex(ValueError, "slot generation"):
            TraceBundle(
                manifest=self.manifest(),
                trees=(tree,),
                cycles=(cycle,),
                cuda_events=(
                    early_start,
                    early_end,
                    late_start,
                    late_end,
                ),
                h2d_transfers=(ordered_first, repeated_generation),
            )
        decreasing_first = replace(ordered_first, generation=2)
        with self.assertRaisesRegex(ValueError, "slot generation"):
            TraceBundle(
                manifest=self.manifest(),
                trees=(tree,),
                cycles=(cycle,),
                cuda_events=(
                    early_start,
                    early_end,
                    late_start,
                    late_end,
                ),
                h2d_transfers=(decreasing_first, repeated_generation),
            )

        alternate_start = replace(
            late_start,
            stream_id="layer-0-copy-alternate",
        )
        alternate_end = replace(
            late_end,
            stream_id="layer-0-copy-alternate",
        )
        alternate_stream_transfer = replace(
            repeated_generation,
            generation=2,
            start_event_id=alternate_start.trace_id,
            end_event_id=alternate_end.trace_id,
        )
        with self.assertRaisesRegex(ValueError, "stream changes"):
            TraceBundle(
                manifest=self.manifest(),
                trees=(tree,),
                cycles=(cycle,),
                cuda_events=(
                    early_start,
                    early_end,
                    alternate_start,
                    alternate_end,
                ),
                h2d_transfers=(ordered_first, alternate_stream_transfer),
            )

        layer_one_start = replace(late_start, sequence=4)
        layer_one_end = replace(late_end, sequence=5)
        layer_one_transfer = replace(
            repeated_generation,
            layer_id=1,
            submission_sequence=0,
            start_event_id=layer_one_start.trace_id,
            end_event_id=layer_one_end.trace_id,
        )
        with self.assertRaisesRegex(ValueError, "stream.*multiple layers"):
            TraceBundle(
                manifest=self.manifest(),
                trees=(tree,),
                cycles=(cycle,),
                cuda_events=(
                    early_start,
                    early_end,
                    layer_one_start,
                    layer_one_end,
                ),
                h2d_transfers=(ordered_first, layer_one_transfer),
            )

    def test_cycle_contract_rejects_wrong_acceptance_and_kv_state(self) -> None:
        tree, cycle, _, _ = self.chain()

        with self.assertRaisesRegex(ValueError, "KV length"):
            replace(cycle, kv_length_after=999)
        with self.assertRaisesRegex(ValueError, "commit at least"):
            replace(cycle, committed_token_ids=())
        wrong_acceptance = replace(
            cycle,
            verifier_token_ids=(101, 102, 999),
        )
        with self.assertRaisesRegex(ValueError, "acceptance"):
            TraceBundle(
                manifest=self.manifest(),
                trees=(tree,),
                cycles=(wrong_acceptance,),
            )

    def test_unicode_jsonl_and_strict_scalar_types_round_trip(self) -> None:
        tree, cycle, _, _ = self.chain()
        manifest = replace(
            self.manifest(),
            raw_prompt="a\u0085b\u2028c\u2029d",
            rendered_prompt="a\u0085b\u2028c\u2029d",
        )
        bundle = TraceBundle(
            manifest=manifest,
            trees=(tree,),
            cycles=(cycle,),
        )

        self.assertEqual(TraceBundle.from_jsonl(bundle.to_jsonl()), bundle)
        with self.assertRaisesRegex(TypeError, "seed must be an integer"):
            replace(manifest, seed=True)

    def test_codec_rejects_unknown_versions_and_broken_stable_ids(self) -> None:
        tree, cycle, _, _ = self.chain()
        bundle = TraceBundle(
            manifest=self.manifest(),
            trees=(tree,),
            cycles=(cycle,),
        )
        encoded = json.loads(bundle.to_json())
        encoded["schema_version"] = 99
        with self.assertRaisesRegex(ValueError, "unsupported trace schema"):
            TraceBundle.from_json(json.dumps(encoded))

        records = [json.loads(line) for line in bundle.to_jsonl().splitlines()]
        tree_record = next(
            record
            for record in records
            if record["record_type"] == "draft_tree"
        )
        tree_record["trace_id"] = "tampered"
        with self.assertRaisesRegex(ValueError, "invalid stable ID"):
            TraceBundle.from_jsonl(
                "\n".join(json.dumps(record) for record in records)
            )

        with self.assertRaisesRegex(ValueError, "unknown trace record type"):
            TraceBundle.from_json(
                json.dumps(
                    {
                        "record_type": "system_snapshot_v1",
                        "schema_version": 1,
                    }
                )
            )
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            TraceBundle.from_json(
                '{"record_type":"trace_bundle","record_type":"trace_bundle"}'
            )

        fake_nested_id = json.loads(bundle.to_json())
        fake_nested_id["manifest"]["metadata"][0]["trace_id"] = "fake"
        with self.assertRaisesRegex(ValueError, "does not define a stable ID"):
            TraceBundle.from_json(json.dumps(fake_nested_id))


if __name__ == "__main__":
    unittest.main()
