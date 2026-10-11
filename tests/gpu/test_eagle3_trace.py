from __future__ import annotations

import gc
import unittest
import weakref

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

import specfit.runtime as runtime_api
from specfit import ExpertKey, TraceBundle, WorkloadManifestTrace


class _ConstantDrafter:
    def __init__(self, token_id: int) -> None:
        self.token_id = token_id

    def propose(
        self,
        *,
        fed_ids: torch.Tensor,
        aux_history: torch.Tensor,
        pending_seed: torch.Tensor,
        draft_length: int,
    ) -> torch.Tensor:
        del aux_history, pending_seed
        return fed_ids.new_full(
            (fed_ids.shape[0], draft_length),
            self.token_id,
        )


class _PatternDrafter:
    def propose(
        self,
        *,
        fed_ids: torch.Tensor,
        aux_history: torch.Tensor,
        pending_seed: torch.Tensor,
        draft_length: int,
    ) -> torch.Tensor:
        del aux_history, pending_seed
        pattern = fed_ids.new_tensor((0, 1, 0))
        return pattern[:draft_length].unsqueeze(0)


def _tiny_target(
    *,
    num_experts: int = 4,
    experts_per_token: int = 2,
) -> Qwen3MoeForCausalLM:
    config = Qwen3MoeConfig(
        dtype=torch.bfloat16,
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=num_experts,
        num_experts_per_tok=experts_per_token,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=63,
        pad_token_id=62,
        use_cache=True,
    )
    model = Qwen3MoeForCausalLM(config).to(torch.bfloat16)
    model.config._experts_implementation = "eager"
    with torch.no_grad():
        model.lm_head.weight.zero_()
    return model.eval()


class TargetOnlyTraceTest(unittest.TestCase):
    def test_target_only_trace_uses_generation_residual(self) -> None:
        manifest = WorkloadManifestTrace(
            run_id="run-target-only",
            request_id="request-0",
            git_commit="test-commit",
            image_id="test-image",
            target_model="tiny-qwen3-moe",
            target_revision="test-target-revision",
            drafter_model="unused-drafter",
            drafter_revision="test-drafter-revision",
            tokenizer_revision="test-tokenizer-revision",
            device="cpu",
            dtype="bfloat16",
            reduction_mode="canonical",
            drafter_mode="target-only",
            slots_per_layer=4,
            num_moe_layers=1,
            num_experts_per_layer=4,
            experts_per_token=2,
            draft_length=0,
            max_new_tokens=3,
            seed=0,
            raw_prompt="target only",
            rendered_prompt="target only",
            input_ids=(1, 7, 11),
            attention_mask=(1, 1, 1),
        )
        recorder = runtime_api.GreedyChainTraceRecorder(manifest)

        with recorder:
            recorder.finish_generation(
                output_tokens=torch.tensor([[4, 5, 6]]),
                target_forward_calls=3,
            )
        trace = recorder.finalize()

        self.assertEqual(trace.trees, ())
        self.assertEqual(trace.cycles, ())
        self.assertEqual(trace.target_routes, ())
        self.assertEqual(trace.snapshots, ())
        self.assertEqual(trace.generations[0].output_token_ids, (4, 5, 6))
        self.assertEqual(trace.generations[0].residual_token_ids, (4, 5, 6))
        self.assertEqual(trace.generations[0].target_forward_calls, 3)
        self.assertEqual(TraceBundle.from_jsonl(trace.to_jsonl()), trace)

        eos_recorder = runtime_api.GreedyChainTraceRecorder(manifest)
        with eos_recorder:
            eos_recorder.finish_generation(
                output_tokens=torch.tensor([[4, 63]]),
                target_forward_calls=2,
                stopped_on_eos=True,
                stop_token_id=63,
            )
        eos_trace = eos_recorder.finalize()
        eos_generation = eos_trace.generations[0]
        self.assertTrue(eos_generation.stopped_on_eos)
        self.assertEqual(eos_generation.stop_token_id, 63)
        self.assertEqual(
            TraceBundle.from_jsonl(eos_trace.to_jsonl()),
            eos_trace,
        )

    def test_seed_only_tree_cycle_round_trips(self) -> None:
        manifest = WorkloadManifestTrace(
            run_id="run-seed-only-tree",
            request_id="request-0",
            git_commit="test-commit",
            image_id="test-image",
            target_model="tiny-qwen3-moe",
            target_revision="test-target-revision",
            drafter_model="hindsight-tree",
            drafter_revision="test-drafter-revision",
            tokenizer_revision="test-tokenizer-revision",
            device="cpu",
            dtype="bfloat16",
            reduction_mode="canonical",
            drafter_mode="pool-hindsight",
            slots_per_layer=4,
            num_moe_layers=1,
            num_experts_per_layer=4,
            experts_per_token=2,
            draft_length=5,
            max_new_tokens=1,
            seed=0,
            raw_prompt="seed only tree",
            rendered_prompt="seed only tree",
            input_ids=(1, 7, 11),
            attention_mask=(1, 1, 1),
        )
        tree = runtime_api.Eagle3DraftTree(
            token_ids=(),
            parent_ids=(),
            depths=(),
            conditional_probabilities=(),
            cumulative_log_probabilities=(),
        )
        recorder = runtime_api.GreedyChainTraceRecorder(manifest)

        with recorder:
            recorder.begin_cycle(
                iteration=0,
                base_kv_length=3,
                verification_token_ids=torch.tensor([[0]]),
                draft_tree=tree,
            )
            recorder.end_cycle(
                verifier_tokens=torch.tensor([[2]]),
                accepted_draft_tokens=0,
                committed_token_ids=torch.tensor([[0]]),
                kv_length_after=4,
                accepted_draft_node_ids=(),
            )
            recorder.finish_generation(
                output_tokens=torch.tensor([[0]]),
                target_forward_calls=2,
            )
        trace = recorder.finalize()

        self.assertEqual(len(trace.trees), 1)
        self.assertEqual(len(trace.trees[0].nodes), 1)
        self.assertEqual(trace.cycles[0].verification_token_ids, (0,))
        self.assertEqual(trace.cycles[0].accepted_draft_node_ids, ())
        self.assertEqual(TraceBundle.from_jsonl(trace.to_jsonl()), trace)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class GreedyChainTraceTest(unittest.TestCase):
    def test_submit_and_inflight_join_follow_public_trace_protocol(
        self,
    ) -> None:
        device = torch.device("cuda:0")
        ready_event = torch.cuda.Event()
        before = runtime_api.GpuExpertCacheSnapshot(
            cache_id="qwen3-layer-0",
            queue_id="qwen3-layer-0-copy",
            capacity=1,
            host_state_version=0,
            lru_sequence=0,
            submission_sequence=0,
            slots=(
                runtime_api.CacheSlotSnapshot(
                    slot_id=0,
                    expert_id=None,
                    generation=0,
                    active_leases=0,
                    last_used_sequence=0,
                    ready=False,
                    pending_use_guards=0,
                ),
            ),
            pending_transfers=(),
            stats=runtime_api.ExpertCacheStats(
                h2d_submissions=0,
                resident_hits=0,
                inflight_joins=0,
                evictions=0,
                pending_use_guards=0,
            ),
        )
        after = runtime_api.GpuExpertCacheSnapshot(
            cache_id="qwen3-layer-0",
            queue_id="qwen3-layer-0-copy",
            capacity=1,
            host_state_version=2,
            lru_sequence=1,
            submission_sequence=1,
            slots=(
                runtime_api.CacheSlotSnapshot(
                    slot_id=0,
                    expert_id=ExpertKey(0, 2),
                    generation=1,
                    active_leases=1,
                    last_used_sequence=1,
                    ready=False,
                    pending_use_guards=1,
                ),
            ),
            pending_transfers=(
                runtime_api.PendingTransferSnapshot(
                    expert_id=ExpertKey(0, 2),
                    slot_id=0,
                    generation=1,
                    submission_sequence=0,
                    ready_event=ready_event,
                    ready_event_completed=False,
                ),
            ),
            stats=runtime_api.ExpertCacheStats(
                h2d_submissions=1,
                resident_hits=0,
                inflight_joins=1,
                evictions=0,
                pending_use_guards=1,
            ),
        )
        snapshot_sequence = iter((before, after))

        def capture_snapshot() -> tuple[
            runtime_api.GpuExpertCacheSnapshot, ...
        ]:
            return (next(snapshot_sequence),)

        manifest = WorkloadManifestTrace(
            run_id="run-submit-join",
            request_id="request-0",
            git_commit="test-commit",
            image_id="test-image",
            target_model="tiny-offloaded-qwen3-moe",
            target_revision="test-target-revision",
            drafter_model="controlled-drafter",
            drafter_revision="test-drafter-revision",
            tokenizer_revision="test-tokenizer-revision",
            device=str(device),
            dtype="bfloat16",
            reduction_mode="canonical",
            drafter_mode="controlled",
            slots_per_layer=1,
            num_moe_layers=1,
            num_experts_per_layer=4,
            experts_per_token=2,
            draft_length=1,
            max_new_tokens=1,
            seed=0,
            raw_prompt="controlled submit and join",
            rendered_prompt="controlled submit and join",
            input_ids=(1, 7, 11),
            attention_mask=(1, 1, 1),
        )
        observer = runtime_api.RecordingCacheTransitionObserver(
            run_id=manifest.run_id
        )
        recorder = runtime_api.GreedyChainTraceRecorder(
            manifest,
            cache_snapshot_provider=capture_snapshot,
            transition_log=observer,
        )
        verification = torch.tensor([[0, 2]], device=device)

        try:
            with recorder:
                recorder.begin_cycle(
                    iteration=0,
                    base_kv_length=3,
                    verification_token_ids=verification,
                )
                observer.record(
                    kind=runtime_api.RuntimeEventKind.H2D_SUBMIT,
                    cache_id="qwen3-layer-0",
                    queue_id="qwen3-layer-0-copy",
                    expert_id=ExpertKey(0, 2),
                    slot_id=0,
                    generation=1,
                    submission_sequence=0,
                    ready_event=ready_event,
                    stream_ids=(7,),
                )
                observer.record(
                    kind=runtime_api.RuntimeEventKind.INFLIGHT_JOIN,
                    cache_id="qwen3-layer-0",
                    queue_id="qwen3-layer-0-copy",
                    expert_id=ExpertKey(0, 2),
                    slot_id=0,
                    generation=1,
                    submission_sequence=0,
                    ready_event=ready_event,
                    stream_ids=(9,),
                )
                recorder.end_cycle(
                    verifier_tokens=torch.tensor([[0, 1]], device=device),
                    accepted_draft_tokens=0,
                    committed_token_ids=torch.tensor([[0]], device=device),
                    kv_length_after=4,
                )
                recorder.finish_generation(
                    output_tokens=torch.tensor([[0]], device=device),
                    target_forward_calls=2,
                )
            trace = recorder.finalize()

            self.assertEqual(
                tuple(event.kind for event in trace.runtime_events),
                (
                    runtime_api.RuntimeEventKind.H2D_SUBMIT,
                    runtime_api.RuntimeEventKind.INFLIGHT_JOIN,
                ),
            )
            submit, join = trace.runtime_events
            self.assertEqual(submit.transfer_id, join.transfer_id)
            self.assertEqual(submit.submission_sequence, 0)
            self.assertEqual(join.submission_sequence, 0)
            self.assertEqual(submit.cycle_id, trace.cycles[0].trace_id)
            self.assertEqual(join.cycle_id, trace.cycles[0].trace_id)
            self.assertEqual(trace.cuda_events, ())
            self.assertEqual(trace.h2d_transfers, ())
            self.assertEqual(TraceBundle.from_jsonl(trace.to_jsonl()), trace)
        finally:
            observer.reset()

    def test_full_accept_trace_records_literal_tree_cycle_and_generation(
        self,
    ) -> None:
        device = torch.device("cuda:0")
        input_ids = torch.tensor([[1, 7, 11]], device=device)
        manifest = WorkloadManifestTrace(
            run_id="run-full-accept",
            request_id="request-0",
            git_commit="test-commit",
            image_id="test-image",
            target_model="tiny-qwen3-moe",
            target_revision="test-target-revision",
            drafter_model="constant-zero-drafter",
            drafter_revision="test-drafter-revision",
            tokenizer_revision="test-tokenizer-revision",
            device=str(device),
            dtype="bfloat16",
            reduction_mode="canonical",
            drafter_mode="constant-zero",
            slots_per_layer=4,
            num_moe_layers=1,
            num_experts_per_layer=4,
            experts_per_token=2,
            draft_length=3,
            max_new_tokens=5,
            seed=0,
            raw_prompt="tiny full-accept trace",
            rendered_prompt="tiny full-accept trace",
            input_ids=(1, 7, 11),
            attention_mask=(1, 1, 1),
        )
        target = _tiny_target().to(device)
        recorder = runtime_api.GreedyChainTraceRecorder(manifest)
        speculator = runtime_api.GreedyChainSpeculator(
            target,
            _ConstantDrafter(0),
            layer_indices=(0,),
        )

        with torch.inference_mode():
            result = speculator.generate(
                input_ids,
                max_new_tokens=5,
                draft_length=3,
                trace=recorder,
            )
        trace = recorder.finalize()

        self.assertIsInstance(trace, TraceBundle)
        self.assertEqual(trace.manifest, manifest)
        self.assertTrue(
            torch.equal(
                result.tokens,
                torch.tensor([[0, 0, 0, 0, 0]], device=device),
            )
        )
        self.assertEqual(result.accepted_per_cycle, (3,))
        self.assertEqual(result.target_forward_calls, 2)
        self.assertEqual(len(trace.trees), 1)
        self.assertEqual(len(trace.cycles), 1)
        self.assertEqual(len(trace.generations), 1)

        tree = trace.trees[0]
        cycle = trace.cycles[0]
        generation = trace.generations[0]
        self.assertEqual(tree.request_id, "request-0")
        self.assertEqual(tree.iteration, 0)
        self.assertEqual(
            tuple(
                (
                    node.node_id,
                    node.parent_id,
                    node.depth,
                    node.role.value,
                    node.token_id,
                    node.conditional_probability,
                    node.verification_position,
                    node.selected,
                    node.verified,
                    node.accepted,
                )
                for node in tree.nodes
            ),
            (
                (0, None, 0, "seed", 0, None, 0, True, True, True),
                (1, 0, 1, "draft", 0, None, 1, True, True, True),
                (2, 1, 2, "draft", 0, None, 2, True, True, True),
                (3, 2, 3, "draft", 0, None, 3, True, True, True),
            ),
        )
        self.assertEqual(cycle.tree_id, tree.tree_id)
        self.assertEqual(cycle.iteration, 0)
        self.assertEqual(cycle.base_kv_length, 3)
        self.assertEqual(cycle.pending_seed_token_id, 0)
        self.assertEqual(cycle.verification_token_ids, (0, 0, 0, 0))
        self.assertEqual(cycle.verifier_token_ids, (0, 0, 0, 0))
        self.assertEqual(cycle.accepted_draft_node_ids, (1, 2, 3))
        self.assertEqual(cycle.committed_token_ids, (0, 0, 0, 0))
        self.assertEqual(cycle.kv_length_after, 7)
        self.assertEqual(generation.output_token_ids, (0, 0, 0, 0, 0))
        self.assertEqual(generation.cycle_ids, (cycle.trace_id,))
        self.assertEqual(generation.residual_token_ids, (0,))
        self.assertEqual(generation.target_forward_calls, 2)

    def test_partial_accept_trace_preserves_each_cycle_boundary(self) -> None:
        device = torch.device("cuda:0")
        input_ids = torch.tensor([[1, 7, 11]], device=device)
        manifest = WorkloadManifestTrace(
            run_id="run-partial-accept",
            request_id="request-0",
            git_commit="test-commit",
            image_id="test-image",
            target_model="tiny-qwen3-moe",
            target_revision="test-target-revision",
            drafter_model="pattern-drafter",
            drafter_revision="test-drafter-revision",
            tokenizer_revision="test-tokenizer-revision",
            device=str(device),
            dtype="bfloat16",
            reduction_mode="canonical",
            drafter_mode="pattern",
            slots_per_layer=4,
            num_moe_layers=1,
            num_experts_per_layer=4,
            experts_per_token=2,
            draft_length=3,
            max_new_tokens=5,
            seed=0,
            raw_prompt="tiny partial-accept trace",
            rendered_prompt="tiny partial-accept trace",
            input_ids=(1, 7, 11),
            attention_mask=(1, 1, 1),
        )
        recorder = runtime_api.GreedyChainTraceRecorder(manifest)
        speculator = runtime_api.GreedyChainSpeculator(
            _tiny_target().to(device),
            _PatternDrafter(),
            layer_indices=(0,),
        )

        with torch.inference_mode():
            result = speculator.generate(
                input_ids,
                max_new_tokens=5,
                draft_length=3,
                trace=recorder,
            )
        trace = recorder.finalize()

        self.assertEqual(result.accepted_per_cycle, (1, 1))
        self.assertEqual(
            tuple(
                tuple(node.accepted for node in tree.nodes)
                for tree in trace.trees
            ),
            (
                (True, True, False, False),
                (True, True, False),
            ),
        )
        generation = trace.generations[0]
        cycles_by_id = {
            cycle.trace_id: cycle for cycle in trace.cycles
        }
        ordered_cycles = tuple(
            cycles_by_id[cycle_id]
            for cycle_id in generation.cycle_ids
        )
        self.assertEqual(
            tuple(
                (
                    cycle.base_kv_length,
                    cycle.accepted_draft_node_ids,
                    cycle.committed_token_ids,
                    cycle.kv_length_after,
                )
                for cycle in ordered_cycles
            ),
            (
                (3, (1,), (0, 0), 5),
                (5, (1,), (0, 0), 7),
            ),
        )
        self.assertEqual(generation.output_token_ids, (0, 0, 0, 0, 0))
        self.assertEqual(
            generation.cycle_ids,
            tuple(cycle.trace_id for cycle in ordered_cycles),
        )
        self.assertEqual(generation.residual_token_ids, (0,))
        self.assertEqual(generation.target_forward_calls, 3)

    def test_verification_routes_match_gate_top8_and_raw_top32(self) -> None:
        device = torch.device("cuda:0")
        target = _tiny_target(
            num_experts=40,
            experts_per_token=8,
        ).to(device)
        observed_gate_outputs: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []

        def capture_gate(
            module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ) -> None:
            del module, inputs
            observed_gate_outputs.append(
                tuple(value.detach().clone() for value in output)
            )

        oracle_handle = target.model.layers[0].mlp.gate.register_forward_hook(
            capture_gate
        )
        manifest = WorkloadManifestTrace(
            run_id="run-route-capture",
            request_id="request-0",
            git_commit="test-commit",
            image_id="test-image",
            target_model="tiny-qwen3-moe",
            target_revision="test-target-revision",
            drafter_model="constant-zero-drafter",
            drafter_revision="test-drafter-revision",
            tokenizer_revision="test-tokenizer-revision",
            device=str(device),
            dtype="bfloat16",
            reduction_mode="canonical",
            drafter_mode="constant-zero",
            slots_per_layer=40,
            num_moe_layers=1,
            num_experts_per_layer=40,
            experts_per_token=8,
            draft_length=3,
            max_new_tokens=5,
            seed=0,
            raw_prompt="tiny route trace",
            rendered_prompt="tiny route trace",
            input_ids=(1, 7, 11),
            attention_mask=(1, 1, 1),
        )
        recorder = runtime_api.GreedyChainTraceRecorder(
            manifest,
            target=target,
        )
        speculator = runtime_api.GreedyChainSpeculator(
            target,
            _ConstantDrafter(0),
            layer_indices=(0,),
        )

        try:
            with recorder, torch.inference_mode():
                speculator.generate(
                    torch.tensor([[1, 7, 11]], device=device),
                    max_new_tokens=5,
                    draft_length=3,
                    trace=recorder,
                )
            trace = recorder.finalize()
        finally:
            oracle_handle.remove()

        self.assertEqual(len(observed_gate_outputs), 2)
        router_logits, routing_weights, selected_experts = (
            observed_gate_outputs[-1]
        )
        self.assertEqual(tuple(router_logits.shape), (4, 40))
        routes_by_node = {
            route.node_id: route for route in trace.target_routes
        }
        self.assertEqual(set(routes_by_node), {0, 1, 2, 3})
        for node_id in range(4):
            route = routes_by_node[node_id]
            self.assertEqual(route.layer_id, 0)
            self.assertEqual(
                route.expert_ids,
                tuple(int(value) for value in selected_experts[node_id].cpu()),
            )
            self.assertEqual(
                route.expert_weights,
                tuple(float(value) for value in routing_weights[node_id].cpu()),
            )

        raw_probabilities = torch.softmax(
            router_logits,
            dtype=torch.float32,
            dim=-1,
        )
        expected_probabilities, expected_experts = torch.topk(
            raw_probabilities,
            k=32,
            dim=-1,
            sorted=True,
        )
        topk_by_node = {
            route.node_id: route for route in trace.target_router_topk
        }
        self.assertEqual(set(topk_by_node), {0, 1, 2, 3})
        for node_id in range(4):
            actual = topk_by_node[node_id]
            self.assertEqual(actual.layer_id, 0)
            self.assertEqual(
                tuple(item.expert_id for item in actual.experts),
                tuple(int(value) for value in expected_experts[node_id].cpu()),
            )
            self.assertEqual(
                tuple(item.probability for item in actual.experts),
                tuple(
                    float(value)
                    for value in expected_probabilities[node_id].cpu()
                ),
            )

        self.assertEqual(TraceBundle.from_json(trace.to_json()), trace)
        self.assertEqual(TraceBundle.from_jsonl(trace.to_jsonl()), trace)

    def test_first_draft_rejection_keeps_cycles_and_releases_hooks(
        self,
    ) -> None:
        device = torch.device("cuda:0")
        target = _tiny_target().to(device)
        manifest = WorkloadManifestTrace(
            run_id="run-first-reject",
            request_id="request-0",
            git_commit="test-commit",
            image_id="test-image",
            target_model="tiny-qwen3-moe",
            target_revision="test-target-revision",
            drafter_model="constant-one-drafter",
            drafter_revision="test-drafter-revision",
            tokenizer_revision="test-tokenizer-revision",
            device=str(device),
            dtype="bfloat16",
            reduction_mode="canonical",
            drafter_mode="constant-one",
            slots_per_layer=4,
            num_moe_layers=1,
            num_experts_per_layer=4,
            experts_per_token=2,
            draft_length=3,
            max_new_tokens=5,
            seed=0,
            raw_prompt="tiny first-reject trace",
            rendered_prompt="tiny first-reject trace",
            input_ids=(1, 7, 11),
            attention_mask=(1, 1, 1),
        )
        recorder = runtime_api.GreedyChainTraceRecorder(
            manifest,
            target=target,
        )
        recorder_reference = weakref.ref(recorder)
        speculator = runtime_api.GreedyChainSpeculator(
            target,
            _ConstantDrafter(1),
            layer_indices=(0,),
        )

        with recorder, torch.inference_mode():
            result = speculator.generate(
                torch.tensor([[1, 7, 11]], device=device),
                max_new_tokens=5,
                draft_length=3,
                trace=recorder,
            )
        trace = recorder.finalize()

        self.assertEqual(result.accepted_per_cycle, (0, 0, 0, 0))
        generation = trace.generations[0]
        trees_by_id = {tree.tree_id: tree for tree in trace.trees}
        cycles_by_id = {cycle.trace_id: cycle for cycle in trace.cycles}
        ordered_cycles = tuple(
            cycles_by_id[cycle_id] for cycle_id in generation.cycle_ids
        )
        self.assertEqual(
            tuple(
                len(trees_by_id[cycle.tree_id].nodes)
                for cycle in ordered_cycles
            ),
            (4, 4, 3, 2),
        )
        self.assertTrue(
            all(
                tuple(
                    node.accepted
                    for node in trees_by_id[cycle.tree_id].nodes
                )
                == (True,)
                + (False,)
                * (len(trees_by_id[cycle.tree_id].nodes) - 1)
                for cycle in ordered_cycles
            )
        )
        self.assertEqual(len(trace.target_routes), 13)
        self.assertEqual(len(trace.target_router_topk), 13)

        del recorder
        gc.collect()
        self.assertIsNone(recorder_reference())


if __name__ == "__main__":
    unittest.main()
