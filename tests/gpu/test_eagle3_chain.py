from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from speculators.models.eagle3.attention import create_combined_mask_mod
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.cache_utils import DynamicCache

from specfit.runtime import eagle3_attention
from specfit.runtime import (
    EAGLE3_VALIDATED_TOTAL_TOKENS,
    AuxHiddenStateCollector,
    Eagle3ChainDrafter,
    GreedyChainSpeculator,
    acceptance_prefix_length,
    eagle3_warmup_sequence_lengths,
    target_greedy_generate,
    warmup_eagle3_drafter,
)
from specfit.runtime.eagle3_chain import (
    EAGLE3_DEPTH5_MAX_DRAFTER_KV_TOKENS,
    EAGLE3_DEPTH5_MAX_FED_TOKENS,
    EAGLE3_MAX_COMMITTED_TOKENS,
    EAGLE3_VALIDATED_TREE_DEPTH,
)
from specfit.runtime.eagle3_warmup import (
    eagle3_tree_temporary_kv_tokens,
    eagle3_tree_warmup_anchor_lengths,
    warmup_eagle3_tree_drafter,
)


class _ToyTarget(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [nn.Identity(), nn.Identity(), nn.Identity()]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer_id, layer in enumerate(self.model.layers):
            hidden_states = layer(hidden_states + layer_id + 1)
        return hidden_states


class _FakeEagleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("d2t", torch.tensor([0, 10, 20, 30, 40]))
        supported = torch.zeros(50, dtype=torch.bool)
        supported[[0, 11, 22, 33, 44]] = True
        self.register_buffer("t2d", supported)
        self.embed_tokens = nn.Embedding(50, 4)
        self.lm_head = nn.Linear(4, 5, bias=False)
        self.fc = nn.Linear(6, 4, bias=False)
        self.last_hidden_states = None
        self.last_input_ids = None
        self.last_ttt_steps = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        ttt_steps: int,
    ) -> list[torch.Tensor]:
        self.last_hidden_states = hidden_states
        self.last_input_ids = input_ids
        self.last_ttt_steps = ttt_steps
        token_ids = (2, 4, 1)
        return [
            input_ids.new_full(input_ids.shape, token_ids[step])
            for step in range(ttt_steps)
        ]


class _FakeCompiledEagleModel(_FakeEagleModel):
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        ttt_steps: int,
    ) -> list[torch.Tensor]:
        raise RuntimeError("compiled forward should not run")


_FakeCompiledEagleModel.forward._torchdynamo_orig_callable = (  # type: ignore[attr-defined]
    _FakeEagleModel.forward
)


class _WarmupRecordingEagleModel(_FakeEagleModel):
    def __init__(self) -> None:
        super().__init__()
        self.warmup_calls: list[tuple[tuple[int, ...], bool]] = []

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        ttt_steps: int,
    ) -> list[torch.Tensor]:
        self.warmup_calls.append(
            (tuple(hidden_states.shape), torch.is_grad_enabled())
        )
        return super().forward(
            hidden_states,
            input_ids,
            ttt_steps=ttt_steps,
        )


class _WarmupRecordingTreeDrafter:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def propose_tree(self, **kwargs: object) -> object:
        self.calls.append(
            {
                "fed_shape": tuple(kwargs["fed_ids"].shape),
                "aux_shape": tuple(kwargs["aux_history"].shape),
                "pending_shape": tuple(kwargs["pending_seed"].shape),
                "top_k": kwargs["top_k"],
                "max_depth": kwargs["max_depth"],
                "draft_node_budget": kwargs["draft_node_budget"],
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        return object()


class _StatefulWarmupRecordingTreeDrafter(_WarmupRecordingTreeDrafter):
    def __init__(self) -> None:
        super().__init__()
        self.lifecycle_events: list[str] = []

    def begin_generation(self) -> None:
        self.lifecycle_events.append("begin")

    def end_generation(self) -> None:
        self.lifecycle_events.append("end")


class _EmbeddingRecordingEagleModel(_FakeEagleModel):

    def __init__(self) -> None:
        super().__init__()
        self.embedded_input_ids: list[torch.Tensor] = []
        self.embedded_values: list[torch.Tensor] = []

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        ttt_steps: int,
    ) -> list[torch.Tensor]:
        self.embedded_input_ids.append(input_ids.detach().clone())
        self.embedded_values.append(
            self.embed_tokens(input_ids).detach().clone()
        )
        return super().forward(
            hidden_states,
            input_ids,
            ttt_steps=ttt_steps,
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


class _IncrementingDrafter:
    def __init__(self) -> None:
        self.calls = 0

    def propose(
        self,
        *,
        fed_ids: torch.Tensor,
        aux_history: torch.Tensor,
        pending_seed: torch.Tensor,
        draft_length: int,
    ) -> torch.Tensor:
        del fed_ids, aux_history
        self.calls += 1
        offsets = torch.arange(
            1,
            draft_length + 1,
            dtype=pending_seed.dtype,
            device=pending_seed.device,
        ).unsqueeze(0)
        return (pending_seed + offsets) % 8


class _SequenceCache:
    def __init__(self, length: int) -> None:
        self.length = length

    def get_seq_length(self) -> int:
        return self.length

    def crop(self, max_length: int) -> None:
        self.length = min(self.length, max_length)


class _IncrementingTarget(nn.Module):

    def __init__(self, *, eos_token_id: int = 3) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity()])
        self.config = SimpleNamespace(eos_token_id=eos_token_id)
        self.generation_config = SimpleNamespace(eos_token_id=[eos_token_id])
        self.forward_calls = 0

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: _SequenceCache | None = None,
        use_cache: bool,
        logits_to_keep: int,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        self.forward_calls += 1
        hidden = input_ids.to(torch.float32).unsqueeze(-1)
        hidden = self.model.layers[0](hidden)
        next_ids = (input_ids + 1) % 8
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 8)
        logits.scatter_(2, next_ids.unsqueeze(-1), 1.0)
        if logits_to_keep == 1:
            logits = logits[:, -1:]
        if past_key_values is None:
            cache = _SequenceCache(input_ids.shape[1])
        else:
            cache = past_key_values
            cache.length += input_ids.shape[1]
        return SimpleNamespace(logits=logits, past_key_values=cache)


class _CompressedCorrectionTarget(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity()])
        transitions = torch.arange(50)
        transitions[0] = 11
        transitions[11] = 12
        transitions[12] = 22
        transitions[22] = 44
        self.register_buffer("transitions", transitions)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: _SequenceCache | None = None,
        use_cache: bool,
        logits_to_keep: int,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        hidden = input_ids.to(torch.float32).unsqueeze(-1)
        hidden = self.model.layers[0](hidden)
        del hidden
        next_ids = self.transitions[input_ids]
        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.transitions.shape[0],
            device=input_ids.device,
        )
        logits.scatter_(2, next_ids.unsqueeze(-1), 1.0)
        if logits_to_keep == 1:
            logits = logits[:, -1:]
        if past_key_values is None:
            cache = _SequenceCache(input_ids.shape[1])
        else:
            cache = past_key_values
            cache.length += input_ids.shape[1]
        return SimpleNamespace(logits=logits, past_key_values=cache)


class _CycleOrderTraceSink:

    def __init__(self, events: list[tuple[str, int]]) -> None:
        self.events = events

    def begin_cycle(
        self,
        *,
        iteration: int,
        base_kv_length: int,
        verification_token_ids: torch.Tensor,
    ) -> None:
        del base_kv_length, verification_token_ids
        self.events.append(("begin", iteration))

    def end_cycle(
        self,
        *,
        verifier_tokens: torch.Tensor,
        accepted_draft_tokens: int,
        committed_token_ids: torch.Tensor,
        kv_length_after: int,
    ) -> None:
        del verifier_tokens, accepted_draft_tokens, committed_token_ids
        self.events.append(("end", kv_length_after))

    def abort_cycle(self) -> None:
        self.events.append(("abort", -1))

    def finish_generation(
        self,
        *,
        output_tokens: torch.Tensor,
        target_forward_calls: int,
        stopped_on_eos: bool = False,
        stopped_on_cycle_limit: bool = False,
        stop_token_id: int | None = None,
    ) -> None:
        del output_tokens, stopped_on_eos, stopped_on_cycle_limit, stop_token_id
        self.events.append(("finish", target_forward_calls))


class _GreedyTarget(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: object | None = None,
        use_cache: bool,
        logits_to_keep: int,
    ) -> SimpleNamespace:
        del attention_mask, use_cache, logits_to_keep
        self.forward_calls += 1
        next_ids = (input_ids[:, -1:] + 1) % 8
        logits = torch.zeros(input_ids.shape[0], 1, 8)
        logits.scatter_(2, next_ids.unsqueeze(-1), 1.0)
        cache = past_key_values if past_key_values is not None else object()
        return SimpleNamespace(logits=logits, past_key_values=cache)


def _tiny_target() -> Qwen3MoeForCausalLM:
    config = Qwen3MoeConfig(
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
        eos_token_id=63,
        pad_token_id=62,
        use_cache=True,
    )
    model = Qwen3MoeForCausalLM(config).to(torch.bfloat16)
    model.config._experts_implementation = "eager"
    with torch.no_grad():
        model.lm_head.weight.zero_()
    return model.eval()


class Eagle3ChainComponentsTest(unittest.TestCase):
    def test_target_greedy_reports_prefill_and_unified_result(self) -> None:
        target = _GreedyTarget()
        trace_events: list[tuple[str, int]] = []
        trace = _CycleOrderTraceSink(trace_events)
        prefill_calls: list[int] = []

        result = target_greedy_generate(
            target,
            torch.tensor([[1, 2, 3]]),
            max_new_tokens=3,
            trace=trace,
            on_prefill_complete=lambda: prefill_calls.append(
                target.forward_calls
            ),
        )

        self.assertTrue(torch.equal(result.tokens, torch.tensor([[4, 5, 6]])))
        self.assertEqual(result.cycles, 0)
        self.assertEqual(result.proposed_draft_tokens, 0)
        self.assertEqual(result.accepted_draft_tokens, 0)
        self.assertEqual(result.accepted_per_cycle, ())
        self.assertEqual(result.target_forward_calls, 3)
        self.assertEqual(prefill_calls, [1])
        self.assertEqual(trace_events, [("finish", 3)])

    def test_generation_enforces_committed_length_budget(self) -> None:
        boundary = target_greedy_generate(
            _IncrementingTarget(eos_token_id=7),
            torch.zeros(1, 511, dtype=torch.long),
            max_new_tokens=1,
            stop_on_eos=False,
        )

        self.assertEqual(tuple(boundary.tokens.shape), (1, 1))
        with self.assertRaisesRegex(ValueError, "committed-length"):
            target_greedy_generate(
                _IncrementingTarget(eos_token_id=7),
                torch.zeros(1, 512, dtype=torch.long),
                max_new_tokens=1,
                stop_on_eos=False,
            )
        with self.assertRaisesRegex(ValueError, "committed-length"):
            GreedyChainSpeculator(
                _IncrementingTarget(eos_token_id=7),
                _IncrementingDrafter(),
                layer_indices=(0,),
            ).generate(
                torch.zeros(1, 512, dtype=torch.long),
                max_new_tokens=1,
                stop_on_eos=False,
            )

    def test_query_materialization_stays_outside_compiled_graph(self) -> None:
        compiled_graphs = 0

        def counting_backend(
            graph_module: torch.fx.GraphModule,
            example_inputs: list[torch.Tensor],
        ):
            nonlocal compiled_graphs
            del example_inputs
            compiled_graphs += 1
            return graph_module.forward

        def forward(value: torch.Tensor) -> torch.Tensor:
            value = value + 1
            value = eagle3_attention._materialize_query(
                value.transpose(1, 2)
            )
            return value * 2

        inputs = torch.randn(1, 3, 4, 8)
        expected = forward(inputs)
        actual = torch.compile(forward, backend=counting_backend)(inputs)

        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(actual.is_contiguous())
        self.assertEqual(compiled_graphs, 2)

    def test_flex_compatibility_materializes_query_before_interface(self) -> None:
        hidden = torch.randn(1, 3, 4, 8)
        query = hidden.transpose(1, 2)
        key = torch.randn(1, 2, 3, 8)
        value = torch.randn(1, 2, 3, 8)
        received_query = None
        received_key = None
        received_value = None

        def capture_query(
            module: nn.Module,
            actual_query: torch.Tensor,
            actual_key: torch.Tensor,
            actual_value: torch.Tensor,
            attention_mask: object,
            **kwargs: object,
        ) -> tuple[torch.Tensor, None]:
            nonlocal received_query, received_key, received_value
            del module, attention_mask, kwargs
            received_query = actual_query
            received_key = actual_key
            received_value = actual_value
            return actual_query.transpose(1, 2), None

        with mock.patch.object(
            eagle3_attention,
            "_speculators_flex_attention_forward",
            side_effect=capture_query,
        ):
            output, weights = (
                eagle3_attention._adjusted_flex_attention_forward(
                    nn.Identity(),
                    query,
                    key,
                    value,
                    attention_mask=torch.ones(3, 3, dtype=torch.bool),
                )
            )

        self.assertFalse(query.is_contiguous())
        self.assertIsNotNone(received_query)
        self.assertTrue(received_query.is_contiguous())
        self.assertTrue(torch.equal(received_query, query))
        self.assertIs(received_key, key)
        self.assertIs(received_value, value)
        self.assertEqual(output.shape, hidden.shape)
        self.assertIsNone(weights)

    def test_mask_extension_rebuilds_cross_block_diagonal(self) -> None:
        eagle3_attention.install_eagle3_flex_attention_compatibility()
        sequence_length = 129
        lengths = torch.tensor([sequence_length], dtype=torch.long)
        initial = create_block_mask(
            create_combined_mask_mod(lengths, sequence_length),
            B=None,
            H=None,
            Q_LEN=sequence_length,
            KV_LEN=sequence_length,
            device=torch.device("cpu"),
        )

        extended = eagle3_attention.extend_eagle3_mask_for_draft_tokens(
            initial
        )

        self.assertEqual(
            extended.seq_lengths,
            (sequence_length, sequence_length * 2),
        )
        dense_blocks = extended.to_dense()
        self.assertEqual(tuple(dense_blocks.shape[-2:]), (2, 3))
        self.assertTrue(bool(dense_blocks.all().item()))
        self.assertIs(
            eagle3_attention.eagle3_core.extend_mask_for_draft_tokens,
            eagle3_attention.extend_eagle3_mask_for_draft_tokens,
        )

    def test_acceptance_prefix_stops_at_first_mismatch(self) -> None:
        verifier = torch.tensor([[5, 6, 9]])
        drafts = torch.tensor([[5, 6, 7]])
        self.assertEqual(acceptance_prefix_length(verifier, drafts), 2)

    def test_collector_concatenates_explicit_layer_outputs(self) -> None:
        target = _ToyTarget()
        hidden_states = torch.zeros(1, 2, 3)
        with AuxHiddenStateCollector(target, layer_indices=(0, 2)) as collector:
            target(hidden_states)
            actual = collector.take()

        expected = torch.cat(
            (
                torch.ones_like(hidden_states),
                torch.full_like(hidden_states, 6),
            ),
            dim=-1,
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_drafter_shifts_target_tokens_and_applies_d2t(self) -> None:
        model = _FakeEagleModel()
        drafter = Eagle3ChainDrafter(model)
        fed_ids = torch.tensor([[10, 11, 12]])
        pending_seed = torch.tensor([[13]])
        aux_history = torch.randn(1, 3, 6)

        actual = drafter.propose(
            fed_ids=fed_ids,
            aux_history=aux_history,
            pending_seed=pending_seed,
            draft_length=2,
        )

        self.assertTrue(
            torch.equal(model.last_input_ids, torch.tensor([[11, 12, 13]]))
        )
        self.assertIs(model.last_hidden_states, aux_history)
        self.assertEqual(model.last_ttt_steps, 2)
        self.assertTrue(torch.equal(actual, torch.tensor([[22, 44]])))
        with self.assertRaises(NotImplementedError):
            drafter.propose(
                fed_ids=torch.zeros(
                    1,
                    EAGLE3_VALIDATED_TOTAL_TOKENS - 2,
                    dtype=torch.long,
                ),
                aux_history=torch.zeros(
                    1, EAGLE3_VALIDATED_TOTAL_TOKENS - 2, 6
                ),
                pending_seed=pending_seed,
                draft_length=2,
            )

    def test_committed_and_temporary_length_contracts_are_separate(self) -> None:
        self.assertEqual(EAGLE3_MAX_COMMITTED_TOKENS, 512)
        self.assertEqual(EAGLE3_VALIDATED_TOTAL_TOKENS, 512)
        self.assertEqual(EAGLE3_VALIDATED_TREE_DEPTH, 5)
        self.assertEqual(EAGLE3_DEPTH5_MAX_FED_TOKENS, 506)
        self.assertEqual(EAGLE3_DEPTH5_MAX_DRAFTER_KV_TOKENS, 514)
        self.assertEqual(
            eagle3_tree_temporary_kv_tokens(
                sequence_length=506,
                max_depth=5,
            ),
            514,
        )
        with self.assertRaisesRegex(ValueError, "committed-length"):
            eagle3_tree_temporary_kv_tokens(
                sequence_length=507,
                max_depth=5,
            )

    def test_drafter_warmup_uses_dynamic_shape_anchors(self) -> None:
        self.assertEqual(
            eagle3_warmup_sequence_lengths(
                max_sequence_length=78,
                draft_length=3,
            ),
            (64, 63),
        )
        self.assertEqual(
            eagle3_warmup_sequence_lengths(
                max_sequence_length=138,
                draft_length=3,
            ),
            (64, 63, 129),
        )

        model = _WarmupRecordingEagleModel()
        lengths = warmup_eagle3_drafter(
            Eagle3ChainDrafter(model),
            max_sequence_length=138,
            draft_length=3,
        )

        self.assertEqual(lengths, (64, 63, 129))
        self.assertEqual(
            model.warmup_calls,
            [
                ((1, 64, 6), False),
                ((1, 63, 6), False),
                ((1, 129, 6), False),
            ],
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            eagle3_warmup_sequence_lengths(
                max_sequence_length=0,
                draft_length=3,
            )
        with self.assertRaisesRegex(ValueError, "EAGLE-3 contract"):
            eagle3_warmup_sequence_lengths(
                max_sequence_length=509,
                draft_length=3,
            )

    def test_tree_warmup_uses_depth5_long_shape_without_target(self) -> None:
        model = _WarmupRecordingEagleModel()
        drafter = Eagle3ChainDrafter(model)
        tree_drafter = _WarmupRecordingTreeDrafter()

        lengths = warmup_eagle3_tree_drafter(
            drafter,
            tree_drafter,
            max_sequence_length=506,
            max_depth=5,
            top_k=2,
            draft_node_budget=11,
        )

        self.assertEqual(lengths, (506,))
        self.assertEqual(
            tree_drafter.calls,
            [
                {
                    "fed_shape": (1, 506),
                    "aux_shape": (1, 506, 6),
                    "pending_shape": (1, 1),
                    "top_k": 2,
                    "max_depth": 5,
                    "draft_node_budget": 11,
                    "grad_enabled": False,
                }
            ],
        )

    def test_tree_warmup_defaults_to_top2_depth3_budget7(self) -> None:
        model = _WarmupRecordingEagleModel()
        drafter = Eagle3ChainDrafter(model)
        tree_drafter = _WarmupRecordingTreeDrafter()

        lengths = warmup_eagle3_tree_drafter(
            drafter,
            tree_drafter,
            max_sequence_length=64,
        )

        self.assertEqual(lengths, (64,))
        self.assertEqual(
            tree_drafter.calls[0],
            {
                "fed_shape": (1, 64),
                "aux_shape": (1, 64, 6),
                "pending_shape": (1, 1),
                "top_k": 2,
                "max_depth": 3,
                "draft_node_budget": 7,
                "grad_enabled": False,
            },
        )

    def test_tree_warmup_covers_incremental_suffix_lengths(self) -> None:
        model = _WarmupRecordingEagleModel()
        drafter = Eagle3ChainDrafter(model)
        tree_drafter = _StatefulWarmupRecordingTreeDrafter()

        warmup_eagle3_tree_drafter(
            drafter,
            tree_drafter,
            max_sequence_length=64,
        )

        self.assertEqual(
            [call["fed_shape"] for call in tree_drafter.calls],
            [(1, 64), (1, 54), (1, 55), (1, 57), (1, 60), (1, 64)],
        )
        self.assertEqual(tree_drafter.lifecycle_events, ["begin", "end"])

    def test_tree_warmup_covers_explicit_depth5_shape_anchors(self) -> None:
        model = _WarmupRecordingEagleModel()
        drafter = Eagle3ChainDrafter(model)
        tree_drafter = _StatefulWarmupRecordingTreeDrafter()

        lengths = warmup_eagle3_tree_drafter(
            drafter,
            tree_drafter,
            max_sequence_length=506,
            max_depth=5,
            top_k=2,
            draft_node_budget=7,
            sequence_anchors=(64, 192, 320, 506),
        )

        self.assertEqual(lengths, (64, 192, 320, 506))
        self.assertEqual(
            [call["fed_shape"][1] for call in tree_drafter.calls],
            [
                506,
                43,
                44,
                46,
                49,
                53,
                58,
                64,
                171,
                172,
                174,
                177,
                181,
                186,
                192,
                299,
                300,
                302,
                305,
                309,
                314,
                320,
                485,
                486,
                488,
                491,
                495,
                500,
                506,
            ],
        )
        self.assertEqual(
            tree_drafter.lifecycle_events,
            ["begin", "end"] * 4,
        )
        self.assertTrue(
            all(call["grad_enabled"] is False for call in tree_drafter.calls)
        )

    def test_tree_warmup_anchor_contract_rejects_ambiguous_plans(self) -> None:
        self.assertEqual(
            eagle3_tree_warmup_anchor_lengths(
                max_sequence_length=506,
                max_depth=5,
                sequence_anchors=(64, 192, 320, 506),
            ),
            (64, 192, 320, 506),
        )
        invalid_plans = (
            (),
            (64, 64, 506),
            (192, 64, 506),
            (64, 192, 320),
            (64, True, 506),
        )
        for plan in invalid_plans:
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                eagle3_tree_warmup_anchor_lengths(
                    max_sequence_length=506,
                    max_depth=5,
                    sequence_anchors=plan,
                )

        model = _WarmupRecordingEagleModel()
        with self.assertRaisesRegex(ValueError, "stateful"):
            warmup_eagle3_tree_drafter(
                Eagle3ChainDrafter(model),
                _WarmupRecordingTreeDrafter(),
                max_sequence_length=506,
                max_depth=5,
                sequence_anchors=(64, 192, 320, 506),
            )

    def test_drafter_uses_eager_original_and_reports_vocab_support(self) -> None:
        model = _FakeCompiledEagleModel()
        drafter = Eagle3ChainDrafter(model)
        fed_ids = torch.tensor([[10, 11, 12]])
        pending_seed = torch.tensor([[13]])
        aux_history = torch.randn(1, 3, 6)

        actual = drafter.propose(
            fed_ids=fed_ids,
            aux_history=aux_history,
            pending_seed=pending_seed,
            draft_length=2,
        )

        self.assertTrue(torch.equal(actual, torch.tensor([[22, 44]])))
        support = drafter.target_token_support(torch.tensor([[22, 43, 44]]))
        self.assertTrue(
            torch.equal(support, torch.tensor([[True, False, True]]))
        )
        compiled = Eagle3ChainDrafter(model, use_torch_compile=True)
        self.assertTrue(compiled.use_torch_compile)
        with self.assertRaisesRegex(
            RuntimeError, "compiled forward should not run"
        ):
            compiled.propose(
                fed_ids=fed_ids,
                aux_history=aux_history,
                pending_seed=pending_seed,
                draft_length=1,
            )

    def test_full_vocabulary_mapping_is_identity_and_bounds_are_checked(
        self,
    ) -> None:
        model = _FakeEagleModel()
        model.d2t = None
        model.t2d = None
        model.embed_tokens = nn.Embedding(5, 4)
        drafter = Eagle3ChainDrafter(model)

        values = torch.tensor([[0, 2, 4]])
        self.assertTrue(torch.equal(drafter.map_draft_to_target(values), values))
        self.assertTrue(drafter.target_token_support(values).all())
        self.assertEqual(drafter.vocabulary_mode, "full")
        with self.assertRaisesRegex(ValueError, "target vocabulary"):
            drafter.target_token_support(torch.tensor([[-1]]))
        with self.assertRaisesRegex(ValueError, "draft vocabulary"):
            drafter.map_draft_to_target(torch.tensor([[5]]))

    def test_malformed_compressed_vocabulary_mappings_are_rejected(self) -> None:
        missing_pair = _FakeEagleModel()
        missing_pair.t2d = None
        with self.assertRaisesRegex(ValueError, "appear together"):
            Eagle3ChainDrafter(missing_pair)

        duplicate = _FakeEagleModel()
        duplicate.d2t[1] = -1
        with self.assertRaisesRegex(ValueError, "duplicates"):
            Eagle3ChainDrafter(duplicate)

        disagreeing = _FakeEagleModel()
        disagreeing.t2d[0] = False
        with self.assertRaisesRegex(ValueError, "disagree"):
            Eagle3ChainDrafter(disagreeing)

        out_of_range = _FakeEagleModel()
        out_of_range.d2t[-1] = 50
        with self.assertRaisesRegex(ValueError, "exceeds target vocab"):
            Eagle3ChainDrafter(out_of_range)

    def test_checkpoint_declaring_another_target_is_rejected(self) -> None:
        config = SimpleNamespace(
            speculators_config=SimpleNamespace(
                verifier=SimpleNamespace(
                    name_or_path="Other/Qwen3-30B-A3B-Instruct-2507"
                )
            )
        )
        with mock.patch(
            "specfit.runtime.eagle3_chain."
            "Eagle3DraftModel.config_class.from_pretrained",
            return_value=config,
        ), self.assertRaisesRegex(ValueError, "model lock"):
            Eagle3ChainDrafter.from_pretrained(
                "/models/drafter",
                target_root="/models/Qwen3-30B-A3B-Instruct-2507",
                device="cpu",
                expected_target_repo_id=(
                    "Qwen/Qwen3-30B-A3B-Instruct-2507"
                ),
            )

    def test_oov_correction_uses_full_embedding_in_the_next_cycle(self) -> None:
        input_ids = torch.tensor([[0]])
        with torch.inference_mode():
            expected = target_greedy_generate(
                _CompressedCorrectionTarget(),
                input_ids,
                max_new_tokens=4,
                stop_on_eos=False,
            )

            draft_model = _EmbeddingRecordingEagleModel()
            drafter = Eagle3ChainDrafter(draft_model, layer_indices=(0,))
            actual = GreedyChainSpeculator(
                _CompressedCorrectionTarget(),
                drafter,
            ).generate(
                input_ids,
                max_new_tokens=4,
                draft_length=2,
                stop_on_eos=False,
            )

        correction_token = torch.tensor([[12]])
        self.assertTrue(
            torch.equal(
                expected.tokens,
                torch.tensor([[11, 12, 22, 44]]),
            )
        )
        self.assertTrue(torch.equal(actual.tokens, expected.tokens))
        self.assertEqual(actual.accepted_per_cycle, (0, 2))
        self.assertFalse(drafter.target_token_support(correction_token).item())
        self.assertGreaterEqual(12, drafter.draft_vocab_size)
        self.assertEqual(len(draft_model.embedded_input_ids), 2)
        self.assertEqual(draft_model.embedded_input_ids[1][0, -1].item(), 12)
        expected_embedding = draft_model.embed_tokens(correction_token)
        self.assertTrue(
            torch.equal(
                draft_model.embedded_values[1][:, -1:],
                expected_embedding,
            )
        )

    def test_target_and_chain_stop_at_eos_without_an_extra_forward(self) -> None:
        input_ids = torch.tensor([[0]])
        target = _IncrementingTarget(eos_token_id=3)
        target_only = target_greedy_generate(
            target,
            input_ids,
            max_new_tokens=8,
        )

        self.assertTrue(torch.equal(target_only.tokens, torch.tensor([[1, 2, 3]])))
        self.assertEqual(target_only.target_forward_calls, 3)
        self.assertTrue(target_only.stopped_on_eos)
        self.assertEqual(target_only.stop_token_id, 3)

        target = _IncrementingTarget(eos_token_id=3)
        drafter = _IncrementingDrafter()
        speculator = GreedyChainSpeculator(
            target,
            drafter,
            layer_indices=(0,),
        )
        result = speculator.generate(
            input_ids,
            max_new_tokens=8,
            draft_length=3,
        )

        self.assertTrue(torch.equal(result.tokens, target_only.tokens))
        self.assertEqual(result.accepted_per_cycle, (2,))
        self.assertEqual(result.target_forward_calls, 2)
        self.assertTrue(result.stopped_on_eos)
        self.assertEqual(drafter.calls, 1)

    def test_prefill_eos_skips_drafter(self) -> None:
        target = _IncrementingTarget(eos_token_id=1)
        drafter = _IncrementingDrafter()
        result = GreedyChainSpeculator(
            target,
            drafter,
            layer_indices=(0,),
        ).generate(
            torch.tensor([[0]]),
            max_new_tokens=8,
            draft_length=3,
        )

        self.assertTrue(torch.equal(result.tokens, torch.tensor([[1]])))
        self.assertEqual(result.cycles, 0)
        self.assertEqual(result.target_forward_calls, 1)
        self.assertEqual(drafter.calls, 0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class GreedyChainSpeculatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda:0")

    def test_all_accept_and_first_reject_match_target_greedy(self) -> None:
        target = _tiny_target().to(self.device)
        input_ids = torch.tensor([[1, 7, 11]], device=self.device)
        with torch.inference_mode():
            expected = target.generate(
                input_ids,
                max_new_tokens=5,
                do_sample=False,
            )[:, input_ids.shape[1] :]

            for draft_token, expected_accepted in ((0, True), (1, False)):
                with self.subTest(draft_token=draft_token):
                    speculator = GreedyChainSpeculator(
                        target,
                        _ConstantDrafter(draft_token),
                        layer_indices=(0,),
                    )
                    result = speculator.generate(
                        input_ids,
                        max_new_tokens=5,
                        draft_length=3,
                    )
                    self.assertTrue(torch.equal(result.tokens, expected))
                    self.assertEqual(
                        result.accepted_draft_tokens > 0,
                        expected_accepted,
                    )
                    if expected_accepted:
                        self.assertEqual(result.proposed_draft_tokens, 3)
                        self.assertEqual(result.target_forward_calls, 2)
                    else:
                        self.assertEqual(result.proposed_draft_tokens, 9)
                        self.assertEqual(result.target_forward_calls, 5)

    def test_partial_acceptance_rolls_back_and_matches_target_greedy(self) -> None:
        target = _tiny_target().to(self.device)
        input_ids = torch.tensor([[1, 7, 11]], device=self.device)
        speculator = GreedyChainSpeculator(
            target,
            _PatternDrafter(),
            layer_indices=(0,),
        )
        with torch.inference_mode():
            expected = target.generate(
                input_ids,
                max_new_tokens=5,
                do_sample=False,
            )[:, input_ids.shape[1] :]
            result = speculator.generate(
                input_ids,
                max_new_tokens=5,
                draft_length=3,
            )

        self.assertTrue(torch.equal(result.tokens, expected))
        self.assertEqual(result.accepted_per_cycle, (1, 1))
        self.assertEqual(result.accepted_draft_tokens, 2)
        self.assertEqual(result.proposed_draft_tokens, 5)
        self.assertEqual(result.target_forward_calls, 3)

    def test_trace_cycle_closes_after_kv_crop(self) -> None:
        target = _tiny_target().to(self.device)
        speculator = GreedyChainSpeculator(
            target,
            _PatternDrafter(),
            layer_indices=(0,),
        )
        events: list[tuple[str, int]] = []
        trace = _CycleOrderTraceSink(events)
        original_crop = DynamicCache.crop

        def observed_crop(cache: DynamicCache, max_length: int) -> None:
            events.append(("crop", max_length))
            original_crop(cache, max_length)

        with mock.patch.object(
            DynamicCache,
            "crop",
            new=observed_crop,
        ), torch.inference_mode():
            result = speculator.generate(
                torch.tensor([[1, 7, 11]], device=self.device),
                max_new_tokens=5,
                draft_length=3,
                trace=trace,
                on_prefill_complete=lambda: events.append(("prefill", -1)),
            )

        self.assertEqual(result.accepted_per_cycle, (1, 1))
        self.assertEqual(
            tuple(name for name, _ in events),
            (
                "prefill",
                "begin",
                "crop",
                "end",
                "begin",
                "crop",
                "end",
                "finish",
            ),
        )
        self.assertEqual(events[2][1], events[3][1])
        self.assertEqual(events[5][1], events[6][1])


if __name__ == "__main__":
    unittest.main()
