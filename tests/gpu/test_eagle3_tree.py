from __future__ import annotations

import math
import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import nn
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.cache_utils import DynamicCache

from specfit.runtime import (
    Eagle3DraftCandidate,
    Eagle3DraftCandidatePool,
    Eagle3DraftTree,
    Eagle3TreeDrafter,
    PoolHindsightTreeSelector,
    ShallowTreeSpeculator,
)
from specfit import WorkloadManifestTrace
from specfit.runtime import GreedyChainTraceRecorder
from specfit.runtime.eagle3_tree import (
    CumulativeProbabilityTreeSelector,
    _compact_tree_cache,
    candidate_pool_prefix,
)
from specfit.runtime.eagle3_selection import (
    ExplicitRawIdTreeSelector,
    TreeSelectionDecision,
    candidate_pool_fingerprint,
)


class _IdentityDraftAdapter:
    draft_vocab_size = 16

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids


class _ScriptedTreeBuilder:

    def __init__(self) -> None:
        self.adapter = _IdentityDraftAdapter()
        self.model = None
        self.selector = CumulativeProbabilityTreeSelector()
        self._probabilities = {
            (): {1: 0.6, 2: 0.4},
            (1,): {3: 0.9, 4: 0.1},
            (2,): {5: 0.8, 6: 0.2},
            (1, 3): {7: 0.9, 8: 0.1},
            (2, 5): {9: 0.9, 10: 0.1},
            (1, 3, 7): {11: 0.9, 12: 0.1},
            (2, 5, 9): {13: 0.9, 14: 0.1},
            (1, 3, 7, 11): {15: 0.9, 0: 0.1},
            (2, 5, 9, 13): {15: 0.9, 0: 0.1},
        }

    def _score_next(self, forced_path: tuple[int, ...]) -> torch.Tensor:
        logits = torch.full((16,), -100.0)
        for token_id, probability in self._probabilities[forced_path].items():
            logits[token_id] = math.log(probability)
        return logits

    def build_candidate_pool(
        self,
        *,
        fed_ids: torch.Tensor,
        aux_history: torch.Tensor,
        pending_seed: torch.Tensor,
        top_k: int = 2,
        max_depth: int = 3,
    ) -> Eagle3DraftCandidatePool:
        del fed_ids, aux_history
        candidates: list[Eagle3DraftCandidate] = []
        frontier = ((0, 0.0, -1, ()),)
        path_score_calls = 0
        draft_transformer_steps = 0
        for depth in range(1, max_depth + 1):
            expanded = []
            for parent_raw_id, parent_score, _, parent_path in frontier:
                log_probabilities = self._score_next(parent_path)
                path_score_calls += 1
                draft_transformer_steps += len(parent_path) + 1
                child_ids = torch.argsort(
                    log_probabilities,
                    descending=True,
                    stable=True,
                )[:top_k]
                for child_id in child_ids.tolist():
                    conditional = float(log_probabilities[child_id].item())
                    raw_id = len(candidates) + 1
                    path = parent_path + (int(child_id),)
                    candidate = Eagle3DraftCandidate(
                        raw_id=raw_id,
                        parent_raw_id=parent_raw_id,
                        depth=depth,
                        target_token_id=int(child_id),
                        conditional_log_probability=conditional,
                        cumulative_log_probability=parent_score + conditional,
                        generation_ordinal=len(candidates),
                        path=path,
                    )
                    candidates.append(candidate)
                    expanded.append(
                        (
                            raw_id,
                            candidate.cumulative_log_probability,
                            candidate.generation_ordinal,
                            path,
                        )
                    )
            frontier = tuple(
                sorted(expanded, key=lambda value: (-value[1], value[2]))[
                    :top_k
                ]
            )
        return Eagle3DraftCandidatePool(
            seed_token_id=int(pending_seed[0, 0].item()),
            top_k=top_k,
            max_depth=max_depth,
            candidates=tuple(candidates),
            path_score_calls=path_score_calls,
            draft_transformer_steps=draft_transformer_steps,
        )

    def propose_tree(
        self,
        *,
        draft_node_budget: int = 7,
        committed_offset: int = 0,
        **kwargs: object,
    ) -> Eagle3DraftTree:
        pool = self.build_candidate_pool(**kwargs)
        return self.selector.select(
            pool,
            draft_node_budget=draft_node_budget,
            committed_offset=committed_offset,
        )


class _IncrementingScriptedTreeBuilder(_ScriptedTreeBuilder):

    def __init__(self) -> None:
        super().__init__()
        self._probabilities = {
            (): {2: 0.6, 9: 0.4},
            (2,): {3: 0.9, 7: 0.1},
            (9,): {10: 0.8, 8: 0.2},
            (2, 3): {4: 0.9, 7: 0.1},
            (9, 10): {11: 0.9, 8: 0.1},
            (2, 3, 4): {5: 0.9, 7: 0.1},
            (9, 10, 11): {12: 0.9, 8: 0.1},
            (2, 3, 4, 5): {6: 0.9, 7: 0.1},
            (9, 10, 11, 12): {13: 0.9, 8: 0.1},
        }


class _LengthOnlyDraftCache:
    def __init__(self) -> None:
        self.length = 0

    def get_seq_length(self) -> int:
        return self.length

    def append(self, token_count: int) -> None:
        self.length += token_count

    def crop(self, max_length: int) -> None:
        self.length = min(self.length, max_length)


class _IncrementalScriptedTreeBuilder(Eagle3TreeDrafter):

    def __init__(self) -> None:
        self.adapter = _IdentityDraftAdapter()
        self.model = None
        self.layer_indices = (0,)
        self.selector = CumulativeProbabilityTreeSelector()
        self._generation_active = False
        self.query_lengths: list[int] = []
        self.reset_state()
        self._probabilities = {
            13: {1: 0.6, 2: 0.4},
            1: {3: 0.9, 4: 0.1},
            2: {5: 0.8, 6: 0.2},
            3: {7: 0.9, 8: 0.1},
            5: {9: 0.9, 10: 0.1},
        }

    def reset_state(self) -> None:
        self._stable_cache: _LengthOnlyDraftCache | None = None
        self._stable_length = 0
        self._stable_last_hidden: torch.Tensor | None = None

    def _sync_stable_prefix(
        self,
        *,
        draft_input_ids: torch.Tensor,
        aux_history: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int]:
        del aux_history
        total_length = int(draft_input_ids.shape[1])
        if self._stable_cache is None:
            self._stable_cache = _LengthOnlyDraftCache()
        query_length = total_length - self._stable_length
        if query_length < 0:
            raise RuntimeError("scripted committed prefix became shorter")
        if query_length:
            self.query_lengths.append(query_length)
            self._stable_cache.append(query_length)
            self._stable_length = total_length
            self._stable_last_hidden = (
                draft_input_ids[:, -1:].to(torch.float32).unsqueeze(-1)
            )
        if self._stable_last_hidden is None:
            raise RuntimeError("scripted stable hidden state is absent")
        return self._stable_last_hidden, int(query_length > 0), query_length

    def _next_log_probabilities(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        rows = []
        for token_id in hidden_states[0, :, 0].to(torch.long).tolist():
            logits = torch.full((16,), -100.0)
            for child_id, probability in self._probabilities[token_id].items():
                logits[child_id] = math.log(probability)
            rows.append(logits)
        return torch.stack(rows)

    def _forward_frontier(
        self,
        frontier: object,
        *,
        depth: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del depth
        if self._stable_cache is None:
            raise RuntimeError("scripted frontier has no cache")
        query_length = int(frontier.target_token_ids.shape[0])
        cache_start = self._stable_cache.get_seq_length()
        self.query_lengths.append(query_length)
        self._stable_cache.append(query_length)
        current_slots = torch.arange(
            cache_start,
            cache_start + query_length,
            dtype=torch.long,
        ).unsqueeze(1)
        lineage = torch.cat(
            (frontier.ancestor_cache_slots, current_slots), dim=1
        )
        hidden = frontier.target_token_ids.to(torch.float32).reshape(
            1, query_length, 1
        )
        return hidden, lineage


class _IncrementingIncrementalTreeBuilder(_IncrementalScriptedTreeBuilder):

    def _next_log_probabilities(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        rows = []
        for token_id in hidden_states[0, :, 0].to(torch.long).tolist():
            logits = torch.full((16,), -100.0)
            logits[(token_id + 1) % 16] = math.log(0.9)
            logits[(token_id + 8) % 16] = math.log(0.1)
            rows.append(logits)
        return torch.stack(rows)


class _StaticTreeDrafter:
    def __init__(self, tree: Eagle3DraftTree) -> None:
        self.tree = tree

    def propose_tree(self, **kwargs: object) -> Eagle3DraftTree:
        max_depth = int(kwargs["max_depth"])
        draft_node_budget = int(kwargs["draft_node_budget"])
        selected = tuple(
            index
            for index, depth in enumerate(self.tree.depths)
            if depth <= max_depth
        )[:draft_node_budget]
        old_to_new = {
            old_node_id: new_node_id
            for new_node_id, old_node_id in enumerate(
                (index + 1 for index in selected),
                start=1,
            )
        }
        return Eagle3DraftTree(
            token_ids=tuple(self.tree.token_ids[index] for index in selected),
            parent_ids=tuple(
                0
                if self.tree.parent_ids[index] == 0
                else old_to_new[self.tree.parent_ids[index]]
                for index in selected
            ),
            depths=tuple(self.tree.depths[index] for index in selected),
            conditional_probabilities=tuple(
                self.tree.conditional_probabilities[index]
                for index in selected
            ),
            cumulative_log_probabilities=tuple(
                self.tree.cumulative_log_probabilities[index]
                for index in selected
            ),
        )


class _LifecycleStaticTreeDrafter(_StaticTreeDrafter):
    def __init__(self, tree: Eagle3DraftTree) -> None:
        super().__init__(tree)
        self.events: list[str] = []

    def begin_generation(self) -> None:
        self.events.append("begin")

    def end_generation(self) -> None:
        self.events.append("end")


class _FakeCacheLayer:
    def __init__(self, token_ids: torch.Tensor) -> None:
        values = token_ids.to(torch.float32).reshape(1, 1, -1, 1)
        self.keys = values.clone()
        self.values = values.clone()

    def get_seq_length(self) -> int:
        return self.keys.shape[-2]

    def append(self, token_ids: torch.Tensor) -> None:
        values = token_ids.to(torch.float32).reshape(1, 1, -1, 1)
        self.keys = torch.cat((self.keys, values), dim=-2)
        self.values = torch.cat((self.values, values), dim=-2)

    def crop(self, max_length: int) -> None:
        self.keys = self.keys[..., :max_length, :]
        self.values = self.values[..., :max_length, :]


class _FakeTreeCache:
    def __init__(self, token_ids: torch.Tensor) -> None:
        self.layers = [_FakeCacheLayer(token_ids)]

    def get_seq_length(self) -> int:
        return self.layers[0].get_seq_length()

    def append(self, token_ids: torch.Tensor) -> None:
        for layer in self.layers:
            layer.append(token_ids)

    def crop(self, max_length: int) -> None:
        for layer in self.layers:
            layer.crop(max_length)


class _IncrementingTreeTarget(nn.Module):

    def __init__(self, *, eos_token_id: int | None = None) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity()])
        self.config = SimpleNamespace(
            _attn_implementation="sdpa",
            eos_token_id=eos_token_id,
        )
        self.generation_config = SimpleNamespace(eos_token_id=eos_token_id)
        self.last_cache: _FakeTreeCache | None = None

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        past_key_values: _FakeTreeCache | None = None,
        use_cache: bool,
        logits_to_keep: int,
    ) -> SimpleNamespace:
        del attention_mask, position_ids, use_cache
        hidden = self.model.layers[0](
            input_ids.to(torch.float32).unsqueeze(-1)
        )
        del hidden
        next_ids = (input_ids + 1) % 16
        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            16,
            device=input_ids.device,
        )
        logits.scatter_(2, next_ids.unsqueeze(-1), 1.0)
        if logits_to_keep == 1:
            logits = logits[:, -1:]
        if past_key_values is None:
            cache = _FakeTreeCache(input_ids)
        else:
            cache = past_key_values
            cache.append(input_ids)
        self.last_cache = cache
        return SimpleNamespace(logits=logits, past_key_values=cache)


def _assert_scripted_hindsight_cycle_metrics(
    test_case: unittest.TestCase,
    *,
    device: torch.device,
) -> None:
    result = ShallowTreeSpeculator(
        _IncrementingTreeTarget(),
        _IncrementingScriptedTreeBuilder(),
        layer_indices=(0,),
    ).generate(
        torch.tensor([[0]], device=device),
        max_new_tokens=6,
        top_k=2,
        max_depth=5,
        draft_node_budget=7,
        selector=PoolHindsightTreeSelector((1, 2, 3, 4, 5, 6)),
        collect_cycle_metrics=True,
        stop_on_eos=False,
    )

    test_case.assertTrue(
        torch.equal(
            result.tokens.cpu(),
            torch.tensor([[1, 2, 3, 4, 5, 6]]),
        )
    )
    test_case.assertEqual(result.accepted_per_cycle, (5,))
    test_case.assertEqual(len(result.tree_cycle_metrics), 1)
    metrics = result.tree_cycle_metrics[0]
    test_case.assertEqual(metrics.iteration, 0)
    test_case.assertEqual(metrics.committed_output_offset, 0)
    test_case.assertEqual(metrics.actual_tree_depth, 5)
    test_case.assertEqual(metrics.raw_candidate_nodes, 18)
    test_case.assertEqual(metrics.selected_draft_nodes, 5)
    test_case.assertEqual(metrics.accepted_draft_nodes, 5)
    test_case.assertEqual(metrics.path_score_calls, 9)
    test_case.assertEqual(metrics.draft_transformer_steps, 29)
    test_case.assertGreaterEqual(
        metrics.proposal_wall_ns,
        metrics.selector_wall_ns,
    )
    test_case.assertGreaterEqual(
        metrics.cycle_wall_ns,
        metrics.proposal_wall_ns + metrics.target_verification_wall_ns,
    )
    test_case.assertEqual(
        metrics.verification_raw_ids,
        (0, 1, 3, 7, 11, 15),
    )
    test_case.assertEqual(metrics.accepted_raw_ids, (1, 3, 7, 11, 15))
    test_case.assertEqual(metrics.next_pending_seed_token_id, 7)
    if device.type == "cuda":
        test_case.assertIsNotNone(metrics.target_verification_cuda_ns)
        test_case.assertGreater(metrics.target_verification_cuda_ns, 0)
    else:
        test_case.assertIsNone(metrics.target_verification_cuda_ns)


def _assert_cache_compaction(test_case, devices):
    for positions in ((0, 2, 4), tuple(range(5)), (0,), ()):
        with test_case.subTest(devices=devices, positions=positions):
            cache = DynamicCache()
            expected = []
            for layer_id, device in enumerate(devices):
                keys = torch.arange(48, dtype=torch.bfloat16).reshape(1, 2, 8, 3) + 64 * layer_id
                values = -keys - 16
                expected.append(tuple(torch.cat(
                    [tensor[..., :3, :]] + [tensor[..., 3 + p:4 + p, :] for p in positions],
                    dim=-2) for tensor in (keys, values)))
                cache.update(keys.to(device), values.to(device), layer_id)
            _compact_tree_cache(cache, base_kv_length=3,
                verification_positions=positions, verification_node_count=5)
            for layer, (keys, values) in zip(cache.layers, expected, strict=True):
                test_case.assertEqual(layer.get_seq_length(), 3 + len(positions))
                test_case.assertTrue(torch.equal(layer.keys.cpu(), keys))
                test_case.assertTrue(torch.equal(layer.values.cpu(), values))


class Eagle3DraftTreeTest(unittest.TestCase):
    def test_cache_compaction_preserves_each_layer_prefix_and_selected_rows(self):
        _assert_cache_compaction(self, ("cpu",) * 3)

    @staticmethod
    def _manifest() -> WorkloadManifestTrace:
        return WorkloadManifestTrace(
            run_id="tree-run",
            request_id="request-0",
            git_commit="test-commit",
            image_id="test-image",
            target_model="tiny-target",
            target_revision="target-revision",
            drafter_model="tiny-drafter",
            drafter_revision="drafter-revision",
            tokenizer_revision="tokenizer-revision",
            device="cpu",
            dtype="float32",
            reduction_mode="canonical",
            drafter_mode="eagle3-shallow-tree",
            slots_per_layer=1,
            num_moe_layers=1,
            num_experts_per_layer=1,
            experts_per_token=1,
            draft_length=3,
            max_new_tokens=4,
            seed=0,
            raw_prompt="tree",
            rendered_prompt="tree",
            input_ids=(0,),
            attention_mask=(1,),
        )

    def test_attention_positions_and_non_left_acceptance(self) -> None:
        tree = Eagle3DraftTree(
            token_ids=(7, 2, 6, 3),
            parent_ids=(0, 0, 1, 2),
            depths=(1, 1, 2, 2),
            conditional_probabilities=(0.4, 0.6, 0.5, 0.7),
            cumulative_log_probabilities=(
                math.log(0.4),
                math.log(0.6),
                math.log(0.4 * 0.5),
                math.log(0.6 * 0.7),
            ),
        )

        mask = tree.verification_attention_mask(
            base_kv_length=2,
            device=torch.device("cpu"),
        )
        expected_tree_mask = torch.tensor(
            (
                (1, 0, 0, 0, 0),
                (1, 1, 0, 0, 0),
                (1, 0, 1, 0, 0),
                (1, 1, 0, 1, 0),
                (1, 0, 1, 0, 1),
            ),
            dtype=torch.bool,
        )
        self.assertTrue(mask[..., :2].all())
        self.assertTrue(torch.equal(mask[0, 0, :, 2:], expected_tree_mask))
        self.assertTrue(
            torch.equal(
                tree.verification_position_ids(
                    base_kv_length=2,
                    device=torch.device("cpu"),
                ),
                torch.tensor([[2, 3, 3, 4, 4]]),
            )
        )
        verifier = torch.tensor([[2, 0, 3, 0, 4]])
        self.assertEqual(tree.accepted_path(verifier), (2, 4))
        self.assertEqual(tree.source_raw_ids, (1, 2, 3, 4))
        self.assertEqual(tree.verification_raw_ids, (0, 1, 2, 3, 4))
        self.assertEqual(
            tree.raw_ids_for_verification_positions((0, 2, 4)),
            (0, 2, 4),
        )

    def test_incremental_frontier_mask_isolates_siblings(self) -> None:
        mask = Eagle3TreeDrafter._frontier_attention_mask(
            stable_length=3,
            cache_length_before=5,
            ancestor_cache_slots=torch.tensor([[3], [4]]),
            query_length=2,
            device=torch.device("cpu"),
        )
        query_indices = torch.arange(2).unsqueeze(1)
        key_value_indices = torch.arange(7).unsqueeze(0)
        visible = mask.mask_mod(
            torch.zeros_like(query_indices),
            torch.zeros_like(query_indices),
            query_indices,
            key_value_indices,
        )

        self.assertTrue(
            torch.equal(
                visible,
                torch.tensor(
                    (
                        (True, True, True, True, False, True, False),
                        (True, True, True, False, True, False, True),
                    )
                ),
            )
        )

    def test_default_builder_matches_top2_depth3_global_rerank(self) -> None:
        builder = _ScriptedTreeBuilder()
        pool = builder.build_candidate_pool(
            fed_ids=torch.tensor([[11, 12]]),
            aux_history=torch.zeros(1, 2, 3),
            pending_seed=torch.tensor([[13]]),
        )
        tree = builder.selector.select(pool, draft_node_budget=7)

        self.assertEqual(pool.top_k, 2)
        self.assertEqual(pool.max_depth, 3)
        self.assertEqual(pool.raw_candidate_count, 10)
        self.assertEqual(
            tuple(node.depth for node in pool.candidates),
            (1, 1, 2, 2, 2, 2, 3, 3, 3, 3),
        )
        self.assertEqual(
            tuple(node.parent_raw_id for node in pool.candidates),
            (0, 0, 1, 1, 2, 2, 3, 3, 5, 5),
        )
        self.assertEqual(tree.draft_node_count, 7)
        self.assertEqual(tree.token_ids, (1, 2, 3, 5, 6, 7, 9))
        self.assertEqual(tree.parent_ids, (0, 0, 1, 2, 2, 3, 4))
        self.assertEqual(tree.depths, (1, 1, 2, 2, 2, 3, 3))
        expected_scores = (
            math.log(0.6),
            math.log(0.4),
            math.log(0.6 * 0.9),
            math.log(0.4 * 0.8),
            math.log(0.4 * 0.2),
            math.log(0.6 * 0.9 * 0.9),
            math.log(0.4 * 0.8 * 0.9),
        )
        for actual, expected in zip(
            tree.cumulative_log_probabilities, expected_scores
        ):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_incremental_builder_reuses_prefix_and_batches_frontiers(
        self,
    ) -> None:
        builder = _IncrementalScriptedTreeBuilder()
        builder.begin_generation()
        first = builder.build_candidate_pool(
            fed_ids=torch.tensor([[11, 12]]),
            aux_history=torch.zeros(1, 2, 3),
            pending_seed=torch.tensor([[13]]),
        )

        self.assertEqual(builder.query_lengths, [2, 2, 2])
        self.assertEqual(first.raw_candidate_count, 10)
        self.assertEqual(first.draft_forward_calls, 3)
        self.assertEqual(first.draft_query_tokens, 6)
        self.assertEqual(first.path_score_calls, 5)
        self.assertEqual(first.draft_transformer_steps, 11)
        self.assertIsNotNone(builder._stable_cache)
        self.assertEqual(builder._stable_cache.get_seq_length(), 2)

        repeated = builder.build_candidate_pool(
            fed_ids=torch.tensor([[11, 12]]),
            aux_history=torch.zeros(1, 2, 3),
            pending_seed=torch.tensor([[13]]),
        )
        self.assertEqual(builder.query_lengths[-2:], [2, 2])
        self.assertEqual(repeated.candidates, first.candidates)
        self.assertEqual(repeated.draft_forward_calls, 2)
        self.assertEqual(repeated.draft_query_tokens, 4)
        self.assertEqual(builder._stable_cache.get_seq_length(), 2)

        builder.build_candidate_pool(
            fed_ids=torch.tensor([[11, 12, 1, 3]]),
            aux_history=torch.zeros(1, 4, 3),
            pending_seed=torch.tensor([[13]]),
        )
        self.assertEqual(builder.query_lengths[-3:], [2, 2, 2])
        self.assertEqual(builder._stable_cache.get_seq_length(), 4)
        builder.end_generation()
        self.assertIsNone(builder._stable_cache)

        builder.begin_generation()
        builder.build_candidate_pool(
            fed_ids=torch.tensor([[9, 10, 11]]),
            aux_history=torch.zeros(1, 3, 3),
            pending_seed=torch.tensor([[13]]),
        )
        self.assertEqual(builder.query_lengths[-3:], [3, 2, 2])
        builder.end_generation()

    def test_incremental_builder_stops_after_a_level_like_its_prefix(
        self,
    ) -> None:
        def build(**options: object) -> tuple[
            _IncrementalScriptedTreeBuilder, Eagle3DraftCandidatePool
        ]:
            builder = _IncrementalScriptedTreeBuilder()
            builder.begin_generation()
            pool = builder.build_candidate_pool(
                fed_ids=torch.tensor([[11, 12]]),
                aux_history=torch.zeros(1, 2, 3),
                pending_seed=torch.tensor([[13]]),
                **options,
            )
            self.assertEqual(builder._stable_cache.get_seq_length(), 2)
            return builder, pool

        _, full = build()
        seen: list[Eagle3DraftCandidatePool] = []
        _, same = build(
            stop_after_level=lambda partial: seen.append(partial) or False
        )
        self.assertEqual(same, full)
        self.assertEqual(
            seen,
            [candidate_pool_prefix(full, 1), candidate_pool_prefix(full, 2)],
        )
        self.assertEqual(candidate_pool_prefix(full, 3), full)

        builder, second = build(
            stop_after_level=lambda partial: partial.max_depth == 2
        )
        self.assertEqual(builder.query_lengths, [2, 2])
        self.assertEqual(second, candidate_pool_prefix(full, 2))
        self.assertEqual(
            (
                second.raw_candidate_count,
                second.draft_forward_calls,
                second.draft_query_tokens,
                second.path_score_calls,
                second.draft_transformer_steps,
            ),
            (6, 2, 4, 3, 5),
        )
        builder, first = build(stop_after_level=lambda partial: True)
        self.assertEqual(builder.query_lengths, [2])
        self.assertEqual(first, candidate_pool_prefix(full, 1))
        self.assertEqual(
            (
                first.raw_candidate_count,
                first.draft_forward_calls,
                first.draft_query_tokens,
                first.path_score_calls,
                first.draft_transformer_steps,
            ),
            (2, 1, 2, 1, 1),
        )
        for depth in (0, 4, True):
            with self.assertRaises(ValueError):
                candidate_pool_prefix(full, depth)

    def test_generation_stops_drafting_after_a_level(self) -> None:
        def run(**options: object):
            return ShallowTreeSpeculator(
                _IncrementingTreeTarget(),
                _IncrementingIncrementalTreeBuilder(),
                layer_indices=(0,),
            )._generate(
                torch.tensor([[0]]),
                max_new_tokens=8,
                top_k=2,
                max_depth=5,
                draft_node_budget=7,
                selector=PoolHindsightTreeSelector(tuple(range(1, 9))),
                collect_cycle_metrics=True,
                stop_on_eos=False,
                **options,
            )

        seen: list[tuple[int, int]] = []

        def stop(
            partial: Eagle3DraftCandidatePool, *, committed_offset: int
        ) -> bool:
            seen.append((committed_offset, partial.max_depth))
            return partial.max_depth == 2

        full = run()
        stopped = run(stop_after_level=stop)
        expected = torch.tensor([list(range(1, 9))])
        self.assertTrue(torch.equal(full.tokens, expected))
        self.assertTrue(torch.equal(stopped.tokens, expected))
        self.assertEqual(full.accepted_per_cycle, (5, 1))
        self.assertEqual(stopped.accepted_per_cycle, (2, 2, 1))
        self.assertEqual(seen, [(0, 1), (0, 2), (3, 1), (3, 2)])
        metrics = stopped.tree_cycle_metrics
        self.assertEqual(
            tuple(m.actual_tree_depth for m in metrics), (2, 2, 1)
        )
        self.assertEqual(
            tuple(m.raw_candidate_nodes for m in metrics), (6, 6, 2)
        )
        self.assertEqual(
            tuple(m.draft_forward_calls for m in metrics), (2, 2, 1)
        )
        for metric in metrics:
            self.assertGreaterEqual(
                metric.proposal_wall_ns, metric.selector_wall_ns
            )
        with self.assertRaises(ValueError):
            ShallowTreeSpeculator(
                _IncrementingTreeTarget(),
                _IncrementingIncrementalTreeBuilder(),
                layer_indices=(0,),
            )._generate(
                torch.tensor([[0]]),
                max_new_tokens=8,
                stop_after_level=stop,
            )

    def test_builder_breaks_equal_scores_by_draft_token_id(self) -> None:
        builder = _ScriptedTreeBuilder()
        builder._probabilities[()] = {5: 0.5, 2: 0.5}

        trees = tuple(
            builder.propose_tree(
                fed_ids=torch.tensor([[11, 12]]),
                aux_history=torch.zeros(1, 2, 3),
                pending_seed=torch.tensor([[13]]),
                top_k=2,
                max_depth=1,
                draft_node_budget=2,
            )
            for _ in range(2)
        )

        self.assertEqual(trees[0], trees[1])
        self.assertEqual(set(trees[0].token_ids), {2, 5})

    def test_depth_five_pool_is_reused_for_top_seven_and_top_eleven(
        self,
    ) -> None:
        builder = _ScriptedTreeBuilder()
        pool = builder.build_candidate_pool(
            fed_ids=torch.tensor([[11, 12]]),
            aux_history=torch.zeros(1, 2, 3),
            pending_seed=torch.tensor([[13]]),
            top_k=2,
            max_depth=5,
        )
        selector = CumulativeProbabilityTreeSelector()

        top_seven_raw_ids = selector.select_raw_ids(
            pool,
            draft_node_budget=7,
        )
        top_eleven_raw_ids = selector.select_raw_ids(
            pool,
            draft_node_budget=11,
        )
        top_seven = pool.to_tree(top_seven_raw_ids)
        top_eleven = pool.to_tree(top_eleven_raw_ids)

        self.assertEqual(pool.raw_candidate_count, 18)
        self.assertEqual(pool.path_score_calls, 9)
        self.assertEqual(pool.draft_transformer_steps, 29)
        self.assertEqual(
            top_seven_raw_ids,
            (1, 3, 7, 11, 2, 15, 5),
        )
        self.assertEqual(
            top_eleven_raw_ids,
            (1, 3, 7, 11, 2, 15, 5, 9, 13, 17, 6),
        )
        self.assertEqual(top_seven.draft_node_count, 7)
        self.assertEqual(
            top_seven.source_raw_ids,
            (1, 2, 3, 5, 7, 11, 15),
        )
        self.assertEqual(
            top_seven.raw_ids_for_verification_positions((0, 2, 4, 7)),
            (0, 2, 5, 15),
        )
        self.assertEqual(top_seven.token_ids, (1, 2, 3, 5, 7, 11, 15))
        self.assertEqual(top_seven.parent_ids, (0, 0, 1, 2, 3, 5, 6))
        self.assertEqual(top_eleven.draft_node_count, 11)
        self.assertEqual(
            top_eleven.parent_ids,
            (0, 0, 1, 2, 2, 3, 4, 6, 7, 8, 9),
        )
        with self.assertRaisesRegex(ValueError, "ancestor-closed"):
            pool.to_tree((15,))

    def test_explicit_raw_id_decision_is_stable_and_replayable(self) -> None:
        builder = _ScriptedTreeBuilder()
        pool = builder.build_candidate_pool(
            fed_ids=torch.tensor([[11, 12]]),
            aux_history=torch.zeros(1, 2, 3),
            pending_seed=torch.tensor([[13]]),
            top_k=2,
            max_depth=5,
        )
        repeated_pool = builder.build_candidate_pool(
            fed_ids=torch.tensor([[11, 12]]),
            aux_history=torch.zeros(1, 2, 3),
            pending_seed=torch.tensor([[13]]),
            top_k=2,
            max_depth=5,
        )
        selector = ExplicitRawIdTreeSelector(
            CumulativeProbabilityTreeSelector()
        )

        decision = selector.decide(
            pool,
            draft_node_budget=7,
            committed_offset=4,
        )
        replayed = decision.materialize(repeated_pool)

        self.assertEqual(
            candidate_pool_fingerprint(pool),
            candidate_pool_fingerprint(repeated_pool),
        )
        self.assertEqual(decision.committed_offset, 4)
        self.assertEqual(
            decision.selected_raw_ids,
            (1, 2, 3, 5, 7, 11, 15),
        )
        self.assertEqual(
            decision.verification_raw_ids,
            (0, 1, 2, 3, 5, 7, 11, 15),
        )
        self.assertEqual(replayed.source_raw_ids, decision.selected_raw_ids)
        self.assertEqual(
            selector.select(pool, draft_node_budget=7, committed_offset=4),
            replayed,
        )

        different_pool = builder.build_candidate_pool(
            fed_ids=torch.tensor([[11, 12]]),
            aux_history=torch.zeros(1, 2, 3),
            pending_seed=torch.tensor([[12]]),
            top_k=2,
            max_depth=5,
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            decision.materialize(different_pool)
        with self.assertRaisesRegex(ValueError, "ancestor-closed"):
            TreeSelectionDecision.from_pool(
                pool,
                draft_node_budget=7,
                committed_offset=0,
                selected_raw_ids=(15,),
            )

    def test_pool_fingerprint_accepts_integer_log_probabilities(self) -> None:
        pool = Eagle3DraftCandidatePool(
            seed_token_id=0,
            top_k=1,
            max_depth=1,
            candidates=(
                Eagle3DraftCandidate(
                    raw_id=1,
                    parent_raw_id=0,
                    depth=1,
                    target_token_id=1,
                    conditional_log_probability=0,
                    cumulative_log_probability=0,
                    generation_ordinal=0,
                    path=(1,),
                ),
            ),
            path_score_calls=1,
            draft_transformer_steps=1,
        )

        self.assertEqual(len(candidate_pool_fingerprint(pool)), 64)

    def test_explicit_selector_rejects_unmapped_local_node_ids(self) -> None:
        class LegacyLocalIdSelector:
            def select(
                self,
                pool: Eagle3DraftCandidatePool,
                *,
                draft_node_budget: int,
                committed_offset: int = 0,
            ) -> Eagle3DraftTree:
                del draft_node_budget, committed_offset
                selected = pool.to_tree((1, 3))
                return Eagle3DraftTree(
                    token_ids=selected.token_ids,
                    parent_ids=selected.parent_ids,
                    depths=selected.depths,
                    conditional_probabilities=(
                        selected.conditional_probabilities
                    ),
                    cumulative_log_probabilities=(
                        selected.cumulative_log_probabilities
                    ),
                )

        with self.assertRaisesRegex(RuntimeError, "cannot be reconstructed"):
            ShallowTreeSpeculator(
                _IncrementingTreeTarget(),
                _ScriptedTreeBuilder(),
                layer_indices=(0,),
            ).generate(
                torch.tensor([[0]]),
                max_new_tokens=4,
                top_k=2,
                max_depth=2,
                draft_node_budget=2,
                selector=LegacyLocalIdSelector(),
                stop_on_eos=False,
            )

    def test_pool_materializes_hindsight_path_and_seed_only_tree(self) -> None:
        pool = _ScriptedTreeBuilder().build_candidate_pool(
            fed_ids=torch.tensor([[11, 12]]),
            aux_history=torch.zeros(1, 2, 3),
            pending_seed=torch.tensor([[13]]),
            top_k=2,
            max_depth=5,
        )

        path_tree = pool.to_tree((1, 3, 7, 11, 15))
        seed_only = pool.to_tree(())

        self.assertEqual(path_tree.token_ids, (1, 3, 7, 11, 15))
        self.assertEqual(path_tree.parent_ids, (0, 1, 2, 3, 4))
        self.assertEqual(path_tree.depths, (1, 2, 3, 4, 5))
        self.assertEqual(seed_only.draft_node_count, 0)
        self.assertTrue(
            torch.equal(
                seed_only.verification_token_ids(torch.tensor([[13]])),
                torch.tensor([[13]]),
            )
        )
        self.assertTrue(
            torch.equal(
                seed_only.verification_position_ids(
                    base_kv_length=2,
                    device=torch.device("cpu"),
                ),
                torch.tensor([[2]]),
            )
        )
        self.assertTrue(
            seed_only.verification_attention_mask(
                base_kv_length=2,
                device=torch.device("cpu"),
            ).all()
        )
        self.assertEqual(seed_only.accepted_path(torch.tensor([[14]])), ())

    def test_hindsight_selector_covers_correct_depth_zero_through_five(
        self,
    ) -> None:
        pool = _ScriptedTreeBuilder().build_candidate_pool(
            fed_ids=torch.tensor([[11, 12]]),
            aux_history=torch.zeros(1, 2, 3),
            pending_seed=torch.tensor([[13]]),
            top_k=2,
            max_depth=5,
        )
        correct_path = (1, 3, 7, 11, 15)
        committed_offset = 2

        for expected_depth in range(6):
            with self.subTest(expected_depth=expected_depth):
                suffix = correct_path[:expected_depth]
                if expected_depth < len(correct_path):
                    suffix += (14,)
                golden = (8, 9, pool.seed_token_id) + suffix
                selected = PoolHindsightTreeSelector(golden).select(
                    pool,
                    draft_node_budget=5,
                    committed_offset=committed_offset,
                )

                self.assertEqual(selected.draft_node_count, expected_depth)
                self.assertEqual(selected.token_ids, correct_path[:expected_depth])

    def test_hindsight_selector_rejects_seed_mismatch_at_explicit_offset(
        self,
    ) -> None:
        pool = _ScriptedTreeBuilder().build_candidate_pool(
            fed_ids=torch.tensor([[11, 12]]),
            aux_history=torch.zeros(1, 2, 3),
            pending_seed=torch.tensor([[13]]),
            top_k=2,
            max_depth=5,
        )
        selector = PoolHindsightTreeSelector((8, 9, 12, 1, 3))

        with self.assertRaisesRegex(RuntimeError, "seed differs"):
            selector.select(
                pool,
                draft_node_budget=5,
                committed_offset=2,
            )

    def test_collects_single_scripted_cycle_metrics_on_cpu(self) -> None:
        _assert_scripted_hindsight_cycle_metrics(
            self,
            device=torch.device("cpu"),
        )

    def test_hindsight_seed_only_tail_cycle_preserves_generation_state(
        self,
    ) -> None:
        target = _IncrementingTreeTarget()
        result = ShallowTreeSpeculator(
            target,
            _IncrementingScriptedTreeBuilder(),
            layer_indices=(0,),
        ).generate(
            torch.tensor([[0]]),
            max_new_tokens=8,
            top_k=2,
            max_depth=5,
            draft_node_budget=7,
            selector=PoolHindsightTreeSelector(tuple(range(1, 9))),
            collect_cycle_metrics=True,
            stop_on_eos=False,
        )

        self.assertTrue(
            torch.equal(result.tokens, torch.tensor([list(range(1, 9))]))
        )
        self.assertEqual(result.accepted_per_cycle, (5, 0))
        self.assertEqual(
            tuple(
                metric.committed_output_offset
                for metric in result.tree_cycle_metrics
            ),
            (0, 6),
        )
        self.assertEqual(
            tuple(
                metric.actual_tree_depth
                for metric in result.tree_cycle_metrics
            ),
            (5, 1),
        )
        self.assertEqual(
            tuple(
                metric.raw_candidate_nodes
                for metric in result.tree_cycle_metrics
            ),
            (18, 2),
        )
        self.assertEqual(
            tuple(
                metric.selected_draft_nodes
                for metric in result.tree_cycle_metrics
            ),
            (5, 0),
        )
        self.assertIsNotNone(target.last_cache)
        self.assertEqual(target.last_cache.get_seq_length(), 8)

    def test_tree_rejects_an_inconsistent_cumulative_score(self) -> None:
        with self.assertRaises(ValueError):
            Eagle3DraftTree(
                token_ids=(7,),
                parent_ids=(0,),
                depths=(1,),
                conditional_probabilities=(0.4,),
                cumulative_log_probabilities=(-0.4,),
            )

    def test_speculator_accepts_non_left_branch_and_compacts_suffix(self) -> None:
        tree = Eagle3DraftTree(
            token_ids=(7, 2, 6, 3),
            parent_ids=(0, 0, 1, 2),
            depths=(1, 1, 2, 2),
            conditional_probabilities=(0.4, 0.6, 0.5, 0.7),
            cumulative_log_probabilities=(
                math.log(0.4),
                math.log(0.6),
                math.log(0.4 * 0.5),
                math.log(0.6 * 0.7),
            ),
        )
        target = _IncrementingTreeTarget()
        prefill_callbacks = []
        result = ShallowTreeSpeculator(
            target,
            _StaticTreeDrafter(tree),
            layer_indices=(0,),
        ).generate(
            torch.tensor([[0]]),
            max_new_tokens=4,
            stop_on_eos=False,
            audit_commits=True,
            on_prefill_complete=lambda: prefill_callbacks.append(True),
        )

        self.assertEqual(prefill_callbacks, [True])
        self.assertTrue(torch.equal(result.tokens, torch.tensor([[1, 2, 3, 4]])))
        self.assertEqual(result.accepted_per_cycle, (2,))
        self.assertEqual(result.proposed_draft_tokens, 4)
        self.assertEqual(result.target_forward_calls, 2)
        self.assertIsNotNone(target.last_cache)
        cache_tokens = target.last_cache.layers[0].keys.reshape(-1).to(torch.long)
        self.assertTrue(torch.equal(cache_tokens, torch.tensor([0, 1, 2, 3])))
        self.assertEqual(len(result.commit_audits), 1)
        audit = result.commit_audits[0]
        self.assertTrue(audit.passed)
        self.assertEqual(audit.verification_positions, (0, 2, 4))
        self.assertEqual(audit.rejected_node_ids, (1, 3))
        self.assertEqual(audit.kv_layer_count, 1)
        self.assertTrue(audit.aux_length_exact)
        self.assertEqual(audit.aux_length_after, audit.aux_prefix_length + 3)

    def test_speculator_can_stop_after_one_complete_cycle(self) -> None:
        tree = Eagle3DraftTree(
            token_ids=(2, 7, 3),
            parent_ids=(0, 0, 1),
            depths=(1, 1, 2),
            conditional_probabilities=(0.6, 0.4, 0.5),
            cumulative_log_probabilities=(
                math.log(0.6),
                math.log(0.4),
                math.log(0.6 * 0.5),
            ),
        )
        recorder = GreedyChainTraceRecorder(
            replace(self._manifest(), max_new_tokens=8)
        )
        result = ShallowTreeSpeculator(
            _IncrementingTreeTarget(),
            _StaticTreeDrafter(tree),
            layer_indices=(0,),
        ).generate(
            torch.tensor([[0]]),
            max_new_tokens=8,
            max_cycles=1,
            stop_on_eos=False,
            trace=recorder,
        )
        trace = recorder.finalize()

        self.assertTrue(torch.equal(result.tokens, torch.tensor([[1, 2, 3]])))
        self.assertEqual(result.cycles, 1)
        self.assertEqual(result.accepted_per_cycle, (2,))
        self.assertEqual(result.target_forward_calls, 2)
        self.assertTrue(result.stopped_on_cycle_limit)
        self.assertTrue(trace.generations[0].stopped_on_cycle_limit)

        budget_recorder = GreedyChainTraceRecorder(
            replace(self._manifest(), max_new_tokens=3)
        )
        budget_result = ShallowTreeSpeculator(
            _IncrementingTreeTarget(),
            _StaticTreeDrafter(tree),
            layer_indices=(0,),
        ).generate(
            torch.tensor([[0]]),
            max_new_tokens=3,
            max_cycles=1,
            stop_on_eos=False,
            trace=budget_recorder,
        )
        budget_trace = budget_recorder.finalize()
        self.assertFalse(budget_result.stopped_on_cycle_limit)
        self.assertFalse(
            budget_trace.generations[0].stopped_on_cycle_limit
        )

        with self.assertRaisesRegex(ValueError, "max_cycles"):
            ShallowTreeSpeculator(
                _IncrementingTreeTarget(),
                _StaticTreeDrafter(tree),
                layer_indices=(0,),
            ).generate(
                torch.tensor([[0]]),
                max_new_tokens=8,
                max_cycles=0,
                stop_on_eos=False,
            )

    def test_speculator_isolates_stateful_drafter_per_request(self) -> None:
        tree = Eagle3DraftTree(
            token_ids=(2, 3),
            parent_ids=(0, 1),
            depths=(1, 2),
            conditional_probabilities=(0.8, 0.7),
            cumulative_log_probabilities=(
                math.log(0.8),
                math.log(0.8 * 0.7),
            ),
        )
        drafter = _LifecycleStaticTreeDrafter(tree)
        speculator = ShallowTreeSpeculator(
            _IncrementingTreeTarget(),
            drafter,
            layer_indices=(0,),
        )

        first = speculator.generate(
            torch.tensor([[0]]),
            max_new_tokens=3,
            stop_on_eos=False,
        )
        second = speculator.generate(
            torch.tensor([[0]]),
            max_new_tokens=3,
            stop_on_eos=False,
        )

        self.assertTrue(torch.equal(first.tokens, second.tokens))
        self.assertEqual(drafter.events, ["begin", "end", "begin", "end"])

    def test_speculator_continues_after_root_and_partial_rejections(self) -> None:
        root_reject_tree = Eagle3DraftTree(
            token_ids=(7, 8),
            parent_ids=(0, 0),
            depths=(1, 1),
            conditional_probabilities=(0.6, 0.4),
            cumulative_log_probabilities=(math.log(0.6), math.log(0.4)),
        )
        root_reject = ShallowTreeSpeculator(
            _IncrementingTreeTarget(),
            _StaticTreeDrafter(root_reject_tree),
            layer_indices=(0,),
        ).generate(
            torch.tensor([[0]]),
            max_new_tokens=3,
            stop_on_eos=False,
        )
        self.assertTrue(
            torch.equal(root_reject.tokens, torch.tensor([[1, 2, 3]]))
        )
        self.assertEqual(root_reject.accepted_per_cycle, (0, 0))

        partial_tree = Eagle3DraftTree(
            token_ids=(2, 7, 9),
            parent_ids=(0, 0, 1),
            depths=(1, 1, 2),
            conditional_probabilities=(0.6, 0.4, 0.5),
            cumulative_log_probabilities=(
                math.log(0.6),
                math.log(0.4),
                math.log(0.6 * 0.5),
            ),
        )
        partial = ShallowTreeSpeculator(
            _IncrementingTreeTarget(),
            _StaticTreeDrafter(partial_tree),
            layer_indices=(0,),
        ).generate(
            torch.tensor([[0]]),
            max_new_tokens=4,
            stop_on_eos=False,
        )
        self.assertTrue(torch.equal(partial.tokens, torch.tensor([[1, 2, 3, 4]])))
        self.assertEqual(partial.accepted_per_cycle, (1, 0))

    def test_speculator_stops_at_eos_inside_an_accepted_tree_path(self) -> None:
        tree = Eagle3DraftTree(
            token_ids=(2, 7, 3),
            parent_ids=(0, 0, 1),
            depths=(1, 1, 2),
            conditional_probabilities=(0.6, 0.4, 0.5),
            cumulative_log_probabilities=(
                math.log(0.6),
                math.log(0.4),
                math.log(0.6 * 0.5),
            ),
        )
        recorder = GreedyChainTraceRecorder(self._manifest())
        result = ShallowTreeSpeculator(
            _IncrementingTreeTarget(eos_token_id=2),
            _StaticTreeDrafter(tree),
            layer_indices=(0,),
        ).generate(
            torch.tensor([[0]]),
            max_new_tokens=4,
            max_cycles=1,
            trace=recorder,
        )
        trace = recorder.finalize()

        self.assertTrue(torch.equal(result.tokens, torch.tensor([[1, 2]])))
        self.assertEqual(result.accepted_per_cycle, (1,))
        self.assertTrue(result.stopped_on_eos)
        self.assertFalse(result.stopped_on_cycle_limit)
        self.assertEqual(result.stop_token_id, 2)
        self.assertTrue(trace.generations[0].stopped_on_eos)
        self.assertFalse(trace.generations[0].stopped_on_cycle_limit)
        self.assertEqual(trace.generations[0].stop_token_id, 2)

    def test_trace_preserves_tree_topology_probabilities_and_path(self) -> None:
        tree = Eagle3DraftTree(
            token_ids=(7, 2, 6, 3),
            parent_ids=(0, 0, 1, 2),
            depths=(1, 1, 2, 2),
            conditional_probabilities=(0.4, 0.6, 0.5, 0.7),
            cumulative_log_probabilities=(
                math.log(0.4),
                math.log(0.6),
                math.log(0.4 * 0.5),
                math.log(0.6 * 0.7),
            ),
        )
        recorder = GreedyChainTraceRecorder(self._manifest())
        verify_ids = tree.verification_token_ids(torch.tensor([[1]]))
        recorder.begin_cycle(
            iteration=0,
            base_kv_length=1,
            verification_token_ids=verify_ids,
            draft_tree=tree,
        )
        recorder.end_cycle(
            verifier_tokens=torch.tensor([[2, 0, 3, 0, 4]]),
            accepted_draft_tokens=2,
            accepted_draft_node_ids=(2, 4),
            committed_token_ids=torch.tensor([[1, 2, 3]]),
            kv_length_after=4,
        )
        recorder.finish_generation(
            output_tokens=torch.tensor([[1, 2, 3, 4]]),
            target_forward_calls=2,
        )

        bundle = recorder.finalize()
        traced = bundle.trees[0]
        self.assertEqual(
            tuple(node.parent_id for node in traced.nodes),
            (None, 0, 0, 1, 2),
        )
        self.assertEqual(
            tuple(node.conditional_probability for node in traced.nodes),
            (None, 0.4, 0.6, 0.5, 0.7),
        )
        self.assertEqual(bundle.cycles[0].accepted_draft_node_ids, (2, 4))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class Qwen3TreeAttentionTest(unittest.TestCase):
    def test_cache_compaction_preserves_rows_across_layer_devices(self):
        _assert_cache_compaction(self, ("cuda:0",) * 3)
        _assert_cache_compaction(self, ("cpu", "cuda:0", "cpu"))

    def test_collects_single_scripted_cycle_metrics_on_cuda(self) -> None:
        _assert_scripted_hindsight_cycle_metrics(
            self,
            device=torch.device("cuda"),
        )

    def test_each_tree_node_matches_an_independent_causal_path(self) -> None:
        torch.manual_seed(7)
        config = Qwen3MoeConfig(
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
            attn_implementation="sdpa",
        )
        target = Qwen3MoeForCausalLM(config).cuda().eval()
        target.config._experts_implementation = "eager"
        prefix = torch.tensor([[1, 7, 11]], device="cuda")
        tree = Eagle3DraftTree(
            token_ids=(5, 6, 9, 10),
            parent_ids=(0, 0, 1, 2),
            depths=(1, 1, 2, 2),
            conditional_probabilities=(0.6, 0.4, 0.5, 0.5),
            cumulative_log_probabilities=(
                math.log(0.6),
                math.log(0.4),
                math.log(0.6 * 0.5),
                math.log(0.4 * 0.5),
            ),
        )

        with torch.inference_mode():
            prefill = target(
                input_ids=prefix,
                attention_mask=torch.ones_like(prefix),
                use_cache=True,
                logits_to_keep=1,
            )
            seed = prefill.logits.argmax(dim=-1)
            verify_ids = tree.verification_token_ids(seed)
            tree_outputs = target(
                input_ids=verify_ids,
                attention_mask=tree.verification_attention_mask(
                    base_kv_length=prefix.shape[1],
                    device=prefix.device,
                ),
                position_ids=tree.verification_position_ids(
                    base_kv_length=prefix.shape[1],
                    device=prefix.device,
                ),
                past_key_values=prefill.past_key_values,
                use_cache=True,
                logits_to_keep=0,
            )

            for node_id in range(1 + tree.draft_node_count):
                path = []
                cursor = node_id
                while cursor != 0:
                    path.append(tree.token_ids[cursor - 1])
                    cursor = tree.parent_ids[cursor - 1]
                path.reverse()
                causal_ids = torch.cat(
                    (
                        prefix,
                        seed,
                        prefix.new_tensor(path).unsqueeze(0),
                    ),
                    dim=1,
                )
                expected = target(
                    input_ids=causal_ids,
                    attention_mask=torch.ones_like(causal_ids),
                    use_cache=False,
                    logits_to_keep=1,
                ).logits[0, -1]
                actual = tree_outputs.logits[0, node_id]
                torch.testing.assert_close(
                    actual,
                    expected,
                    atol=2e-4,
                    rtol=1e-2,
                )
                self.assertEqual(
                    int(actual.argmax().item()),
                    int(expected.argmax().item()),
                )


if __name__ == "__main__":
    unittest.main()
