from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import save_file
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

import specfit.runtime.qwen3_offloaded_model as offload_module
from specfit import ExpertKey, RuntimeEventKind, WorkloadManifestTrace
from specfit.adapters.qwen3_packed_checkpoint import pack_qwen3_checkpoint
from specfit.runtime import (
    ExpertExecutionPolicy,
    GreedyChainSpeculator,
    GreedyChainTraceRecorder,
    PackedExpertBackingMode,
    Qwen3OffloadedExperts,
    ReductionMode,
    RecordingCacheTransitionObserver,
    load_offloaded_qwen3,
)
from specfit.runtime.registered_mmap import RegisteredMmapExpertLayer


def _tiny_config() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        dtype=torch.bfloat16,
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=True,
    )


def _two_layer_tiny_config() -> Qwen3MoeConfig:
    config = _tiny_config()
    config.num_hidden_layers = 2
    return config


def _save_qwen3_style_checkpoint(
    model: Qwen3MoeForCausalLM,
    model_root: str,
) -> None:

    root = Path(model_root)
    model.config.save_pretrained(root)
    converted = {}
    intermediate_size = model.config.moe_intermediate_size
    for name, tensor in model.state_dict().items():
        if name.endswith(".experts.gate_up_proj"):
            prefix = name.removesuffix("gate_up_proj")
            for expert_id in range(model.config.num_experts):
                converted[
                    f"{prefix}{expert_id}.gate_proj.weight"
                ] = tensor[expert_id, :intermediate_size].contiguous()
                converted[
                    f"{prefix}{expert_id}.up_proj.weight"
                ] = tensor[expert_id, intermediate_size:].contiguous()
        elif name.endswith(".experts.down_proj"):
            prefix = name.removesuffix("down_proj")
            for expert_id in range(model.config.num_experts):
                converted[
                    f"{prefix}{expert_id}.down_proj.weight"
                ] = tensor[expert_id].contiguous()
        else:
            converted[name] = tensor.contiguous()

    shard = "model-00001-of-00001.safetensors"
    save_file(converted, root / shard)
    total_size = sum(
        tensor.numel() * tensor.element_size()
        for tensor in converted.values()
    )
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": {name: shard for name in converted},
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index),
        encoding="utf-8",
    )


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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class Qwen3OffloadedModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda:0")
        if torch.cuda.get_device_capability(cls.device) < (8, 0):
            raise unittest.SkipTest("BF16 requires compute capability 8.0+")

    def test_prefetch_loader_and_boundary_switch_preserve_logits(self):
        torch.manual_seed(918)
        source = Qwen3MoeForCausalLM(_two_layer_tiny_config()).to(torch.bfloat16)
        with tempfile.TemporaryDirectory() as model_root:
            _save_qwen3_style_checkpoint(source, model_root)
            with load_offloaded_qwen3(model_root, device=self.device, slots_per_layer=2,
                                      h2d_prefetch=False) as loaded:
                modules = [layer.mlp.experts for layer in loaded.model.model.layers]
                self.assertTrue(all(module.h2d_prefetch is False for module in modules))
                ids = torch.tensor([[1, 7, 11, 19]], device=self.device)
                with torch.inference_mode():
                    expected = loaded.model(ids, use_cache=False).logits
                    for enabled in (True, False):
                        loaded.cold_reset_expert_caches()
                        loaded.set_expert_h2d_prefetch(enabled)
                        self.assertTrue(all(module.h2d_prefetch is enabled for module in modules))
                        self.assertTrue(torch.equal(expected, loaded.model(ids, use_cache=False).logits))
                with self.assertRaises(ValueError):
                    loaded.set_expert_h2d_prefetch("false")
                with loaded.expert_caches[-1].acquire(ExpertKey(1, 0)):
                    with self.assertRaisesRegex(RuntimeError, "active leases"):
                        loaded.set_expert_h2d_prefetch(True)
                self.assertTrue(all(module.h2d_prefetch is False for module in modules))
            with self.assertRaisesRegex(RuntimeError, "open caches"):
                loaded.set_expert_h2d_prefetch(True)

    def test_loader_replaces_experts_and_matches_hf_generation(self) -> None:
        torch.manual_seed(71)
        source = Qwen3MoeForCausalLM(_tiny_config()).to(torch.bfloat16)
        input_ids = torch.tensor([[1, 7, 11, 19]], dtype=torch.long)

        with tempfile.TemporaryDirectory() as model_root:
            _save_qwen3_style_checkpoint(source, model_root)
            reference = source.to(self.device)
            reference.config._experts_implementation = "eager"
            loaded = load_offloaded_qwen3(
                model_root,
                device="cuda",
                slots_per_layer=1,
                reduction_mode=ReductionMode.CANONICAL,
            )

        self.assertFalse(
            any(
                isinstance(module, Qwen3MoeExperts)
                for module in loaded.model.modules()
            )
        )
        self.assertEqual(len(loaded.expert_caches), 1)
        self.assertEqual(loaded.expert_caches[0].device, self.device)
        self.assertTrue(
            all(parameter.device == self.device for parameter in loaded.model.parameters())
        )
        self.assertIs(
            loaded.model.model.layers[0].mlp.experts.execution_policy,
            ExpertExecutionPolicy.EXPERT_ID_ASCENDING,
        )

        device_input_ids = input_ids.to(self.device)
        with torch.inference_mode():
            expected_logits = reference(
                input_ids=device_input_ids,
                use_cache=False,
                logits_to_keep=1,
            ).logits
            actual_logits = loaded.model(
                input_ids=device_input_ids,
                use_cache=False,
                logits_to_keep=1,
            ).logits
            expected_tokens = reference.generate(
                device_input_ids,
                max_new_tokens=2,
                do_sample=False,
            )
            actual_tokens = loaded.model.generate(
                device_input_ids,
                max_new_tokens=2,
                do_sample=False,
            )

        self.assertTrue(torch.equal(actual_logits, expected_logits))
        self.assertTrue(torch.equal(actual_tokens, expected_tokens))
        stats = loaded.cache_stats()
        self.assertGreaterEqual(stats.h2d_submissions, 2)
        self.assertGreaterEqual(stats.evictions, 1)
        snapshots = loaded.cache_snapshots()
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(
            tuple(snapshot.cache_id for snapshot in snapshots),
            ("qwen3-layer-0",),
        )
        self.assertEqual(
            tuple(snapshot.queue_id for snapshot in snapshots),
            ("layer-0-copy",),
        )
        self.assertTrue(
            all(
                any(slot.expert_id is not None for slot in snapshot.slots)
                for snapshot in snapshots
            )
        )

        active = loaded.expert_caches[0].acquire(ExpertKey(0, 0))
        before_failed_reset = loaded.cache_snapshots()[0]
        with self.assertRaisesRegex(RuntimeError, "active lease"):
            loaded.cold_reset_expert_caches()
        after_failed_reset = loaded.cache_snapshots()[0]
        self.assertEqual(
            after_failed_reset.host_state_version,
            before_failed_reset.host_state_version,
        )
        self.assertEqual(
            after_failed_reset.submission_sequence,
            before_failed_reset.submission_sequence,
        )
        self.assertEqual(
            after_failed_reset.stats,
            before_failed_reset.stats,
        )
        active.release()

        loaded.cold_reset_expert_caches()
        cold = loaded.cache_snapshots()
        self.assertTrue(
            all(
                snapshot.host_state_version == 0
                and snapshot.lru_sequence == 0
                and snapshot.submission_sequence == 0
                and snapshot.pending_transfers == ()
                and snapshot.stats.h2d_submissions == 0
                and all(
                    slot.expert_id is None for slot in snapshot.slots
                )
                for snapshot in cold
            )
        )

        with torch.inference_mode():
            after_reset_tokens = loaded.model.generate(
                device_input_ids,
                max_new_tokens=2,
                do_sample=False,
            )
        self.assertTrue(torch.equal(after_reset_tokens, expected_tokens))

    def test_packed_checkpoint_is_in_memory_before_forward(self) -> None:
        torch.manual_seed(83)
        source = Qwen3MoeForCausalLM(_tiny_config()).to(torch.bfloat16)
        input_ids = torch.tensor([[1, 5, 13]], dtype=torch.long)

        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root)
            model_root = root / "model"
            packed_root = root / "packed"
            model_root.mkdir()
            _save_qwen3_style_checkpoint(source, str(model_root))
            packed = pack_qwen3_checkpoint(model_root, packed_root)

            reference = source.to(self.device)
            reference.config._experts_implementation = "eager"
            loaded = load_offloaded_qwen3(
                model_root,
                device=self.device,
                slots_per_layer=1,
                packed_checkpoint_root=packed_root,
                packed_expert_backing=PackedExpertBackingMode.PINNED_COPY,
            )
            census = packed.census

        self.assertEqual(
            loaded.load_timings.checkpoint_format,
            "packed-v1",
        )
        self.assertEqual(
            loaded.load_timings.expert_bytes_read,
            census.routed_expert_bytes,
        )
        self.assertEqual(
            loaded.load_timings.nonexpert_bytes_read,
            census.nonexpert_bytes,
        )
        self.assertTrue(
            all(cache.sources_are_pinned for cache in loaded.expert_caches)
        )
        for name, value in asdict(loaded.load_timings).items():
            if name.endswith("_seconds"):
                self.assertGreaterEqual(value, 0.0)

        device_input_ids = input_ids.to(self.device)
        with torch.inference_mode():
            expected = reference(
                input_ids=device_input_ids,
                use_cache=False,
                logits_to_keep=1,
            ).logits
            actual = loaded.model(
                input_ids=device_input_ids,
                use_cache=False,
                logits_to_keep=1,
            ).logits
        self.assertTrue(torch.equal(actual, expected))
        loaded.close()

    def test_trace_mode_preserves_offloaded_chain_and_cache_state(self) -> None:
        torch.manual_seed(87)
        source = Qwen3MoeForCausalLM(
            _two_layer_tiny_config()
        ).to(torch.bfloat16)
        with torch.no_grad():
            source.lm_head.weight.zero_()
        input_ids = torch.tensor([[1, 7, 11]], device=self.device)
        observer = RecordingCacheTransitionObserver(
            run_id="run-offload-equivalence"
        )

        with tempfile.TemporaryDirectory() as model_root:
            _save_qwen3_style_checkpoint(source, model_root)
            loaded = load_offloaded_qwen3(
                model_root,
                device=self.device,
                slots_per_layer=1,
                reduction_mode=ReductionMode.CANONICAL,
                cache_observer=observer,
            )

        try:
            speculator = GreedyChainSpeculator(
                loaded.model,
                _ConstantDrafter(0),
                layer_indices=(0, 1),
            )
            with torch.inference_mode():
                plain = speculator.generate(
                    input_ids,
                    max_new_tokens=5,
                    draft_length=3,
                )
            torch.cuda.synchronize(self.device)
            plain_residency = tuple(
                cache.resident_expert_ids()
                for cache in loaded.expert_caches
            )
            plain_stats = loaded.cache_stats()
            plain_snapshots = loaded.cache_snapshots()

            loaded.cold_reset_expert_caches()
            observer.reset(run_id="run-offload-equivalence")
            manifest = WorkloadManifestTrace(
                run_id="run-offload-equivalence",
                request_id="request-0",
                git_commit="test-commit",
                image_id="test-image",
                target_model="tiny-offloaded-qwen3-moe",
                target_revision="test-target-revision",
                drafter_model="constant-zero-drafter",
                drafter_revision="test-drafter-revision",
                tokenizer_revision="test-tokenizer-revision",
                device=str(self.device),
                dtype="bfloat16",
                reduction_mode="canonical",
                drafter_mode="constant-zero",
                slots_per_layer=1,
                num_moe_layers=2,
                num_experts_per_layer=4,
                experts_per_token=2,
                draft_length=3,
                max_new_tokens=5,
                seed=87,
                raw_prompt="tiny offload equivalence",
                rendered_prompt="tiny offload equivalence",
                input_ids=(1, 7, 11),
                attention_mask=(1, 1, 1),
            )
            live_boundaries = []

            def record_live_snapshots():
                snapshots = loaded.cache_snapshots()
                live_boundaries.append((observer.cursor(), snapshots))
                return snapshots

            recorder = GreedyChainTraceRecorder(
                manifest,
                target=loaded.model,
                cache_snapshot_provider=record_live_snapshots,
                transition_log=observer,
            )
            with recorder, torch.inference_mode():
                traced = speculator.generate(
                    input_ids,
                    max_new_tokens=5,
                    draft_length=3,
                    trace=recorder,
                )
            trace = recorder.finalize()
            traced_residency = tuple(
                cache.resident_expert_ids()
                for cache in loaded.expert_caches
            )
            traced_stats = loaded.cache_stats()
            traced_snapshots = loaded.cache_snapshots()

            self.assertTrue(torch.equal(traced.tokens, plain.tokens))
            self.assertEqual(
                traced.accepted_per_cycle,
                plain.accepted_per_cycle,
            )
            self.assertEqual(
                traced.target_forward_calls,
                plain.target_forward_calls,
            )
            self.assertEqual(traced_residency, plain_residency)
            self.assertEqual(traced_stats, plain_stats)
            self.assertEqual(traced_snapshots, plain_snapshots)
            self.assertEqual(len(trace.target_routes), 8)
            self.assertEqual(len(trace.target_router_topk), 8)
            self.assertEqual(len(trace.snapshots), 2 * len(trace.cycles))
            self.assertEqual(len(live_boundaries), len(trace.snapshots))
            self.assertEqual(trace.cuda_events, ())
            self.assertEqual(trace.h2d_transfers, ())

            snapshots_by_id = {
                snapshot.trace_id: snapshot for snapshot in trace.snapshots
            }
            ordered_cycles = tuple(
                sorted(trace.cycles, key=lambda item: item.iteration)
            )
            for cycle_index, cycle in enumerate(ordered_cycles):
                before = snapshots_by_id[cycle.pre_snapshot_id]
                after = snapshots_by_id[cycle.post_snapshot_id]
                self.assertEqual(before.phase.value, "before_verify")
                self.assertEqual(after.phase.value, "after_verify")
                for serialized, (_, live) in zip(
                    (before, after),
                    live_boundaries[2 * cycle_index : 2 * cycle_index + 2],
                ):
                    self.assertEqual(
                        serialized.runtime_state_version,
                        sum(item.host_state_version for item in live),
                    )
                    for layer_id, (layer, actual) in enumerate(
                        zip(serialized.layers, live)
                    ):
                        self.assertEqual(layer.layer_id, layer_id)
                        self.assertEqual(layer.cache_id, actual.cache_id)
                        self.assertEqual(layer.queue_id, actual.queue_id)
                        self.assertEqual(
                            layer.host_state_version,
                            actual.host_state_version,
                        )
                        self.assertEqual(
                            layer.lru_sequence, actual.lru_sequence
                        )
                        self.assertEqual(
                            layer.submission_sequence,
                            actual.submission_sequence,
                        )
                        self.assertEqual(
                            (
                                layer.stats.h2d_submissions,
                                layer.stats.resident_hits,
                                layer.stats.inflight_joins,
                                layer.stats.evictions,
                                layer.stats.pending_use_guards,
                            ),
                            (
                                actual.stats.h2d_submissions,
                                actual.stats.resident_hits,
                                actual.stats.inflight_joins,
                                actual.stats.evictions,
                                actual.stats.pending_use_guards,
                            ),
                        )
                        self.assertEqual(
                            tuple(
                                (
                                    slot.slot_id,
                                    slot.expert_id,
                                    slot.generation,
                                    slot.active_leases,
                                    slot.last_used_sequence,
                                    slot.ready,
                                    slot.pending_use_guards,
                                )
                                for slot in layer.slots
                            ),
                            tuple(
                                (
                                    slot.slot_id,
                                    slot.expert_id.expert_id
                                    if slot.expert_id is not None
                                    else None,
                                    slot.generation,
                                    slot.active_leases,
                                    slot.last_used_sequence,
                                    slot.ready,
                                    slot.pending_use_guards,
                                )
                                for slot in actual.slots
                            ),
                        )
                        self.assertEqual(
                            tuple(
                                (
                                    transfer.expert_id,
                                    transfer.slot_id,
                                    transfer.generation,
                                    transfer.submission_sequence,
                                    transfer.ready_event_id,
                                    transfer.ready_event_completed,
                                )
                                for transfer in layer.pending_transfers
                            ),
                            tuple(
                                (
                                    transfer.expert_id.expert_id,
                                    transfer.slot_id,
                                    transfer.generation,
                                    transfer.submission_sequence,
                                    None,
                                    transfer.ready_event_completed,
                                )
                                for transfer in actual.pending_transfers
                            ),
                        )

            raw_events = observer.events()
            self.assertEqual(
                tuple(event.sequence for event in trace.runtime_events),
                tuple(event.sequence for event in raw_events),
            )
            first_cycle_cursor = live_boundaries[0][0]
            self.assertGreater(first_cycle_cursor, 0)
            traced_by_sequence = {
                event.sequence: event for event in trace.runtime_events
            }
            for raw in raw_events:
                serialized = traced_by_sequence[raw.sequence]
                self.assertEqual(serialized.host_time_ns, raw.host_time_ns)
                self.assertEqual(serialized.kind, raw.kind)
                self.assertEqual(serialized.cache_id, raw.cache_id)
                self.assertEqual(serialized.queue_id, raw.queue_id)
                self.assertEqual(
                    serialized.expert_id, raw.expert_id.expert_id
                )
                self.assertEqual(serialized.slot_id, raw.slot_id)
                self.assertEqual(serialized.generation, raw.generation)
                self.assertEqual(
                    serialized.submission_sequence,
                    raw.submission_sequence,
                )
                self.assertEqual(serialized.stream_ids, raw.stream_ids)
            self.assertTrue(
                all(
                    traced_by_sequence[event.sequence].cycle_id is None
                    for event in raw_events
                    if event.sequence < first_cycle_cursor
                )
            )
            counter_kinds = {
                "h2d_submissions": RuntimeEventKind.H2D_SUBMIT,
                "resident_hits": RuntimeEventKind.RESIDENT_HIT,
                "inflight_joins": RuntimeEventKind.INFLIGHT_JOIN,
                "evictions": RuntimeEventKind.EVICT,
            }
            for cycle_index, cycle in enumerate(ordered_cycles):
                before = snapshots_by_id[cycle.pre_snapshot_id]
                after = snapshots_by_id[cycle.post_snapshot_id]
                scoped = tuple(
                    event
                    for event in trace.runtime_events
                    if event.cycle_id == cycle.trace_id
                )
                self.assertTrue(scoped)
                for before_layer, after_layer in zip(
                    before.layers, after.layers
                ):
                    for field, kind in counter_kinds.items():
                        self.assertEqual(
                            getattr(after_layer.stats, field)
                            - getattr(before_layer.stats, field),
                            sum(
                                event.layer_id == before_layer.layer_id
                                and event.kind is kind
                                for event in scoped
                            ),
                        )
                start = live_boundaries[2 * cycle_index][0]
                end = live_boundaries[2 * cycle_index + 1][0]
                self.assertEqual(
                    tuple(event.sequence for event in scoped),
                    tuple(range(start, end)),
                )
            self.assertEqual(
                {
                    field: sum(
                        event.kind is kind for event in trace.runtime_events
                    )
                    for field, kind in counter_kinds.items()
                },
                {
                    field: getattr(traced_stats, field)
                    for field in counter_kinds
                },
            )
            self.assertEqual(type(trace).from_jsonl(trace.to_jsonl()), trace)
            observer.reset()
            self.assertEqual(observer.events(), ())
        finally:
            loaded.close()

    def test_loader_propagates_and_switches_execution_policy(self) -> None:
        torch.manual_seed(73)
        source = Qwen3MoeForCausalLM(
            _two_layer_tiny_config()
        ).to(torch.bfloat16)
        input_ids = torch.tensor([[1, 7, 11]], device=self.device)

        with tempfile.TemporaryDirectory() as model_root:
            _save_qwen3_style_checkpoint(source, model_root)
            reference = source.to(self.device)
            reference.config._experts_implementation = "eager"
            loaded = load_offloaded_qwen3(
                model_root,
                device=self.device,
                slots_per_layer=1,
                reduction_mode=ReductionMode.CANONICAL,
                expert_execution_policy=(
                    ExpertExecutionPolicy.RESIDENT_FIRST
                ),
            )

        try:
            expert_modules = tuple(
                layer.mlp.experts for layer in loaded.model.model.layers
            )
            self.assertTrue(
                all(
                    isinstance(module, Qwen3OffloadedExperts)
                    for module in expert_modules
                )
            )
            self.assertTrue(
                all(
                    module.execution_policy
                    is ExpertExecutionPolicy.RESIDENT_FIRST
                    for module in expert_modules
                )
            )
            with torch.inference_mode():
                expected = reference(
                    input_ids=input_ids,
                    use_cache=False,
                    logits_to_keep=1,
                ).logits
                resident_first = loaded.model(
                    input_ids=input_ids,
                    use_cache=False,
                    logits_to_keep=1,
                ).logits

            loaded.cold_reset_expert_caches()
            loaded.set_expert_execution_policy("expert_id_ascending")
            self.assertTrue(
                all(
                    module.execution_policy
                    is ExpertExecutionPolicy.EXPERT_ID_ASCENDING
                    for module in expert_modules
                )
            )
            with self.assertRaises(ValueError):
                loaded.set_expert_execution_policy("unsupported")
            self.assertTrue(
                all(
                    module.execution_policy
                    is ExpertExecutionPolicy.EXPERT_ID_ASCENDING
                    for module in expert_modules
                )
            )
            with torch.inference_mode():
                expert_id_ascending = loaded.model(
                    input_ids=input_ids,
                    use_cache=False,
                    logits_to_keep=1,
                ).logits

            self.assertTrue(
                torch.equal(resident_first, expert_id_ascending)
            )
            self.assertTrue(
                torch.equal(
                    resident_first.argmax(dim=-1),
                    expected.argmax(dim=-1),
                )
            )
            torch.testing.assert_close(
                resident_first,
                expected,
                rtol=1.6e-2,
                atol=1.6e-2,
            )
        finally:
            loaded.close()

    def test_registered_mmap_uses_metadata_only_source_and_matches_hf(
        self,
    ) -> None:
        torch.manual_seed(89)
        source = Qwen3MoeForCausalLM(_tiny_config()).to(torch.bfloat16)
        input_ids = torch.tensor([[1, 17, 23]], dtype=torch.long)

        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root)
            model_root = root / "model"
            packed_root = root / "packed"
            model_root.mkdir()
            _save_qwen3_style_checkpoint(source, str(model_root))
            packed = pack_qwen3_checkpoint(model_root, packed_root)
            for shard in model_root.glob("model-*.safetensors"):
                shard.unlink()

            reference = source.to(self.device)
            reference.config._experts_implementation = "eager"
            control = load_offloaded_qwen3(
                model_root,
                device=self.device,
                slots_per_layer=1,
                packed_checkpoint_root=packed_root,
                packed_expert_backing=PackedExpertBackingMode.PINNED_COPY,
            )
            device_input_ids = input_ids.to(self.device)
            with torch.inference_mode():
                control_expected = reference(
                    input_ids=device_input_ids,
                    use_cache=False,
                    logits_to_keep=1,
                ).logits
                control_actual = control.model(
                    input_ids=device_input_ids,
                    use_cache=False,
                    logits_to_keep=1,
                ).logits
            self.assertTrue(
                torch.equal(
                    control_actual.argmax(dim=-1),
                    control_expected.argmax(dim=-1),
                )
            )
            control.close()

            loaded = load_offloaded_qwen3(
                model_root,
                device=self.device,
                slots_per_layer=1,
                packed_checkpoint_root=packed_root,
            )
            census = packed.census
            timings = loaded.load_timings
            with torch.inference_mode():
                expected = reference(
                    input_ids=device_input_ids,
                    use_cache=False,
                    logits_to_keep=1,
                ).logits
                before_unlink = loaded.model(
                    input_ids=device_input_ids,
                    use_cache=False,
                    logits_to_keep=1,
                ).logits
            self.assertTrue(torch.equal(before_unlink, control_actual))
            self.assertTrue(
                torch.equal(
                    before_unlink.argmax(dim=-1),
                    expected.argmax(dim=-1),
                )
            )
            loaded.cold_reset_expert_caches()

        self.assertEqual(
            timings.expert_backing_mode,
            PackedExpertBackingMode.REGISTERED_MMAP.value,
        )
        self.assertEqual(timings.expert_pinned_allocation_seconds, 0)
        self.assertEqual(timings.expert_weight_read_seconds, 0)
        self.assertEqual(timings.expert_bytes_read, 0)
        self.assertEqual(
            timings.expert_bytes_mapped,
            census.routed_expert_bytes,
        )
        self.assertEqual(
            timings.expert_bytes_registered,
            census.routed_expert_bytes,
        )
        self.assertTrue(
            all(
                cache.sources_are_pinned
                for cache in loaded.expert_caches
            )
        )

        with loaded:
            with torch.inference_mode():
                actual = loaded.model(
                    input_ids=device_input_ids,
                    use_cache=False,
                    logits_to_keep=1,
                ).logits
            self.assertTrue(torch.equal(actual, before_unlink))

        self.assertTrue(loaded.expert_caches[0].closed)
        loaded.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            loaded.expert_caches[0].acquire(ExpertKey(0, 0))

    def test_one_token_trace_keeps_prefill_transitions_unscoped(self) -> None:
        torch.manual_seed(91)
        source = Qwen3MoeForCausalLM(_tiny_config()).to(torch.bfloat16)
        with torch.no_grad():
            source.lm_head.weight.zero_()
        input_ids = torch.tensor([[1, 7, 11]], device=self.device)
        observer = RecordingCacheTransitionObserver(run_id="run-one-token")

        with tempfile.TemporaryDirectory() as model_root:
            _save_qwen3_style_checkpoint(source, model_root)
            loaded = load_offloaded_qwen3(
                model_root,
                device=self.device,
                slots_per_layer=1,
                cache_observer=observer,
            )

        try:
            manifest = WorkloadManifestTrace(
                run_id="run-one-token",
                request_id="request-0",
                git_commit="test-commit",
                image_id="test-image",
                target_model="tiny-offloaded-qwen3-moe",
                target_revision="test-target-revision",
                drafter_model="constant-zero-drafter",
                drafter_revision="test-drafter-revision",
                tokenizer_revision="test-tokenizer-revision",
                device=str(self.device),
                dtype="bfloat16",
                reduction_mode="canonical",
                drafter_mode="constant-zero",
                slots_per_layer=1,
                num_moe_layers=1,
                num_experts_per_layer=4,
                experts_per_token=2,
                draft_length=3,
                max_new_tokens=1,
                seed=91,
                raw_prompt="tiny one-token trace",
                rendered_prompt="tiny one-token trace",
                input_ids=(1, 7, 11),
                attention_mask=(1, 1, 1),
            )
            recorder = GreedyChainTraceRecorder(
                manifest,
                target=loaded.model,
                cache_snapshot_provider=loaded.cache_snapshots,
                transition_log=observer,
            )
            speculator = GreedyChainSpeculator(
                loaded.model,
                _ConstantDrafter(0),
                layer_indices=(0,),
            )
            with recorder, torch.inference_mode():
                result = speculator.generate(
                    input_ids,
                    max_new_tokens=1,
                    draft_length=3,
                    trace=recorder,
                )
            trace = recorder.finalize()

            self.assertEqual(tuple(result.tokens.shape), (1, 1))
            self.assertEqual(trace.cycles, ())
            self.assertEqual(trace.snapshots, ())
            self.assertTrue(trace.runtime_events)
            self.assertTrue(
                all(
                    event.tree_id is None and event.cycle_id is None
                    for event in trace.runtime_events
                )
            )
            self.assertEqual(type(trace).from_jsonl(trace.to_jsonl()), trace)
        finally:
            observer.reset()
            loaded.close()

    def test_registered_mmap_rolls_back_earlier_layer_on_failure(
        self,
    ) -> None:
        class FakeOwner:
            def __init__(self, tensor: torch.Tensor) -> None:
                self.tensor = tensor
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        source = Qwen3MoeForCausalLM(
            _two_layer_tiny_config()
        ).to(torch.bfloat16)
        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root)
            model_root = root / "model"
            packed_root = root / "packed"
            model_root.mkdir()
            _save_qwen3_style_checkpoint(source, str(model_root))
            packed = pack_qwen3_checkpoint(model_root, packed_root)
            first_owner = FakeOwner(
                torch.empty(
                    (
                        packed.census.num_experts,
                        packed.packed_expert_elements,
                    ),
                    dtype=torch.bfloat16,
                    pin_memory=True,
                )
            )

            with mock.patch.object(
                offload_module,
                "_open_registered_packed_layer",
                side_effect=(first_owner, RuntimeError("register layer 1")),
            ):
                with self.assertRaisesRegex(RuntimeError, "register layer 1"):
                    load_offloaded_qwen3(
                        model_root,
                        device=self.device,
                        slots_per_layer=1,
                        packed_checkpoint_root=packed_root,
                        packed_expert_backing=(
                            PackedExpertBackingMode.REGISTERED_MMAP
                        ),
                    )

        self.assertEqual(first_owner.close_calls, 1)

    def test_registered_mmap_reopens_after_exact_close(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            path = Path(temporary_root) / "layer.safetensors"
            experts = torch.arange(
                16,
                dtype=torch.bfloat16,
            ).reshape(2, 8)
            save_file({"experts": experts}, path)

            for _ in range(3):
                owner = RegisteredMmapExpertLayer(
                    path,
                    expected_shape=(2, 8),
                    device=self.device,
                )
                backing = owner.tensor
                self.assertTrue(backing.is_pinned())

                owner.close()
                self.assertTrue(owner.closed)
                self.assertFalse(backing.is_pinned())


if __name__ == "__main__":
    unittest.main()
