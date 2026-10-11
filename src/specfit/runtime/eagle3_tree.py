
from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from time import perf_counter_ns

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from transformers import DynamicCache

from .eagle3_chain import (
    AuxHiddenStateCollector,
    EAGLE3_DEFAULT_TREE_DEPTH,
    EAGLE3_DEFAULT_TREE_NODE_BUDGET,
    EAGLE3_DEFAULT_TREE_TOP_K,
    EAGLE3_VALIDATED_TREE_DEPTH,
    Eagle3ChainDrafter,
    QWEN3_EAGLE3_LAYER_INDICES,
    _first_eos_position,
    _normalized_eos_token_ids,
    _validate_committed_token_budget,
    _committed_token_limit,
)
from .eagle3_types import (
    Eagle3DraftCandidate,
    Eagle3DraftCandidatePool,
    Eagle3DraftTree,
    Eagle3GenerationResult,
    Eagle3TreeCycleMetrics,
    GreedyChainTraceSink,
    TreeCandidateSelector,
    TreeCommitAudit,
    TreeDrafter,
)


@dataclass(frozen=True, slots=True)
class _TensorFrontier:
    target_token_ids: torch.Tensor
    cumulative_log_probabilities: torch.Tensor
    raw_ids: torch.Tensor
    parent_hidden_states: torch.Tensor
    ancestor_cache_slots: torch.Tensor


class CumulativeProbabilityTreeSelector:

    def select_raw_ids(
        self,
        pool: Eagle3DraftCandidatePool,
        *,
        draft_node_budget: int,
        committed_offset: int = 0,
    ) -> tuple[int, ...]:
        if (
            isinstance(draft_node_budget, bool)
            or not isinstance(draft_node_budget, int)
            or draft_node_budget <= 0
        ):
            raise ValueError("draft node budget must be a positive integer")
        if (
            isinstance(committed_offset, bool)
            or not isinstance(committed_offset, int)
            or committed_offset < 0
        ):
            raise ValueError("committed offset must be a non-negative integer")
        ranked = sorted(
            pool.candidates,
            key=lambda node: (
                -node.cumulative_log_probability,
                node.depth,
                node.generation_ordinal,
            ),
        )
        return tuple(
            node.raw_id
            for node in ranked[: min(draft_node_budget, len(ranked))]
        )

    def select(
        self,
        pool: Eagle3DraftCandidatePool,
        *,
        draft_node_budget: int,
        committed_offset: int = 0,
    ) -> Eagle3DraftTree:

        if (
            isinstance(committed_offset, bool)
            or not isinstance(committed_offset, int)
            or committed_offset < 0
        ):
            raise ValueError("committed offset must be a non-negative integer")
        selected_raw_ids = self.select_raw_ids(
            pool,
            draft_node_budget=draft_node_budget,
        )
        try:
            return pool.to_tree(selected_raw_ids)
        except ValueError as error:
            raise RuntimeError(
                "global EAGLE selection is not ancestor-closed"
            ) from error


class PoolHindsightTreeSelector:

    def __init__(self, golden_token_ids: tuple[int, ...]) -> None:
        if (
            not isinstance(golden_token_ids, tuple)
            or not golden_token_ids
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in golden_token_ids
            )
        ):
            raise ValueError("golden token IDs must be a non-empty token tuple")
        self.golden_token_ids = golden_token_ids

    def select_raw_ids(
        self,
        pool: Eagle3DraftCandidatePool,
        *,
        draft_node_budget: int,
        committed_offset: int,
    ) -> tuple[int, ...]:
        if (
            isinstance(draft_node_budget, bool)
            or not isinstance(draft_node_budget, int)
            or draft_node_budget < 0
        ):
            raise ValueError("draft node budget must be a non-negative integer")
        if (
            isinstance(committed_offset, bool)
            or not isinstance(committed_offset, int)
            or not 0 <= committed_offset < len(self.golden_token_ids)
        ):
            raise ValueError("committed offset is outside the golden output")
        if pool.seed_token_id != self.golden_token_ids[committed_offset]:
            raise RuntimeError(
                "candidate-pool seed differs from the target-only output: "
                f"committed_offset={committed_offset}, "
                f"pool_seed={pool.seed_token_id}, "
                "golden_seed="
                f"{self.golden_token_ids[committed_offset]}"
            )

        remaining_after_seed = len(self.golden_token_ids) - committed_offset - 1
        search_depth = min(
            pool.max_depth,
            draft_node_budget,
            remaining_after_seed,
        )
        expected_suffix = self.golden_token_ids[committed_offset + 1 :]
        matched: Eagle3DraftCandidate | None = None
        for candidate in pool.candidates:
            if candidate.depth > search_depth:
                continue
            if candidate.path == expected_suffix[: candidate.depth] and (
                matched is None or candidate.depth > matched.depth
            ):
                matched = candidate
        if matched is None:
            return ()

        by_raw_id = {node.raw_id: node for node in pool.candidates}
        selected = []
        current = matched
        while True:
            selected.append(current.raw_id)
            if current.parent_raw_id == 0:
                break
            current = by_raw_id[current.parent_raw_id]
        return tuple(reversed(selected))

    def select(
        self,
        pool: Eagle3DraftCandidatePool,
        *,
        draft_node_budget: int,
        committed_offset: int = 0,
    ) -> Eagle3DraftTree:
        return pool.to_tree(
            self.select_raw_ids(
                pool,
                draft_node_budget=draft_node_budget,
                committed_offset=committed_offset,
            )
        )


def _layered_candidate_pool(
    *,
    seed_token_id: int,
    top_k: int,
    depth: int,
    candidates: tuple[Eagle3DraftCandidate, ...],
    draft_forward_calls: int,
    draft_query_tokens: int,
) -> Eagle3DraftCandidatePool:

    return Eagle3DraftCandidatePool(
        seed_token_id=seed_token_id,
        top_k=top_k,
        max_depth=depth,
        candidates=candidates,
        path_score_calls=1 + top_k * (depth - 1),
        draft_transformer_steps=1 + top_k * (depth * (depth + 1) // 2 - 1),
        draft_forward_calls=draft_forward_calls,
        draft_query_tokens=draft_query_tokens,
    )


def candidate_pool_prefix(
    pool: Eagle3DraftCandidatePool, depth: int
) -> Eagle3DraftCandidatePool:

    if (
        isinstance(depth, bool)
        or not isinstance(depth, int)
        or not 1 <= depth <= pool.max_depth
    ):
        raise ValueError("prefix depth is outside the candidate pool")
    count = sum(node.depth <= depth for node in pool.candidates)
    if any(node.depth > depth for node in pool.candidates[:count]):
        raise ValueError("candidate pool is not ordered by level")
    skipped = pool.max_depth - depth
    if pool.draft_forward_calls < skipped:
        raise ValueError("candidate pool lacks layered execution counts")
    return _layered_candidate_pool(
        seed_token_id=pool.seed_token_id,
        top_k=pool.top_k,
        depth=depth,
        candidates=pool.candidates[:count],
        draft_forward_calls=pool.draft_forward_calls - skipped,
        draft_query_tokens=pool.draft_query_tokens - pool.top_k * skipped,
    )


class Eagle3TreeDrafter:

    def __init__(
        self,
        adapter: Eagle3ChainDrafter,
        *,
        selector: TreeCandidateSelector | None = None,
    ) -> None:
        self.adapter = adapter
        self.model = adapter.model
        self.layer_indices = adapter.layer_indices
        self.selector = (
            CumulativeProbabilityTreeSelector()
            if selector is None
            else selector
        )
        self._generation_active = False
        self.reset_state()

    def reset_state(self) -> None:

        self._stable_cache: DynamicCache | None = None
        self._stable_length = 0
        self._stable_last_hidden: torch.Tensor | None = None

    def begin_generation(self) -> None:

        self.reset_state()
        self._generation_active = True

    def end_generation(self) -> None:

        self._generation_active = False
        self.reset_state()

    @staticmethod
    def _causal_incremental_mask(
        *,
        query_start: int,
        query_length: int,
        key_value_length: int,
        device: torch.device,
    ) -> object:

        def mask_mod(
            _batch: torch.Tensor,
            _head: torch.Tensor,
            query_index: torch.Tensor,
            key_value_index: torch.Tensor,
        ) -> torch.Tensor:
            return key_value_index <= query_start + query_index

        return create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=query_length,
            KV_LEN=key_value_length,
            device=device,
        )

    @staticmethod
    def _frontier_attention_mask(
        *,
        stable_length: int,
        cache_length_before: int,
        ancestor_cache_slots: torch.Tensor,
        query_length: int,
        device: torch.device,
    ) -> object:

        transient_before = cache_length_before - stable_length
        local_key_value_length = transient_before + query_length
        local_allowed = torch.zeros(
            (query_length, local_key_value_length),
            dtype=torch.bool,
            device=device,
        )
        if ancestor_cache_slots.numel():
            ancestor_local_slots = ancestor_cache_slots - stable_length
            rows = torch.arange(
                query_length, dtype=torch.long, device=device
            ).unsqueeze(1)
            local_allowed[rows, ancestor_local_slots] = True
        rows = torch.arange(query_length, dtype=torch.long, device=device)
        local_allowed[rows, transient_before + rows] = True
        key_value_length = cache_length_before + query_length

        def mask_mod(
            _batch: torch.Tensor,
            _head: torch.Tensor,
            query_index: torch.Tensor,
            key_value_index: torch.Tensor,
        ) -> torch.Tensor:
            safe_query = query_index.clamp(max=query_length - 1)
            local_index = key_value_index - stable_length
            safe_local = local_index.clamp(
                min=0, max=local_key_value_length - 1
            )
            local_visible = local_allowed[safe_query, safe_local]
            return torch.logical_and(
                query_index < query_length,
                torch.logical_or(
                    key_value_index < stable_length,
                    torch.logical_and(
                        key_value_index >= stable_length,
                        torch.logical_and(
                            key_value_index < key_value_length,
                            local_visible,
                        ),
                    ),
                ),
            )

        return create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=query_length,
            KV_LEN=key_value_length,
            device=device,
        )

    def _run_transformer(
        self,
        *,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: object,
        cache: DynamicCache,
        cache_position_start: int,
    ) -> torch.Tensor:

        input_embeds = self.model.embed_tokens(input_ids)
        hidden_states = torch.cat((input_embeds, hidden_states), dim=-1)
        position_embeddings = self.model.rotary_emb(
            hidden_states, position_ids
        )
        cache_position = torch.arange(
            cache_position_start,
            cache_position_start + input_ids.shape[1],
            dtype=torch.long,
            device=input_ids.device,
        )
        for decoder_layer in self.model.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        return hidden_states

    def _next_log_probabilities(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        logits = self.model.lm_head(self.model.norm(hidden_states))
        return torch.log_softmax(logits[0].to(torch.float32), dim=-1)

    def _sync_stable_prefix(
        self,
        *,
        draft_input_ids: torch.Tensor,
        aux_history: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int]:

        total_length = int(draft_input_ids.shape[1])
        if self._stable_cache is None:
            self._stable_cache = DynamicCache(
                config=self.model.config.transformer_layer_config
            )
        if total_length < self._stable_length:
            raise RuntimeError("draft committed prefix became shorter")
        if self._stable_cache.get_seq_length() != self._stable_length:
            raise RuntimeError("draft stable cache length is inconsistent")

        query_length = total_length - self._stable_length
        if query_length == 0:
            if self._stable_last_hidden is None:
                raise RuntimeError(
                    "draft stable cache missed its last hidden state"
                )
            return self._stable_last_hidden, 0, 0

        suffix_ids = draft_input_ids[:, self._stable_length :]
        suffix_aux = aux_history[:, self._stable_length :]
        if self.model.input_norm is not None:
            suffix_aux = self.model.input_norm(suffix_aux)
        suffix_aux = self.model.fc(suffix_aux)
        position_ids = 1 + torch.arange(
            self._stable_length,
            total_length,
            dtype=torch.long,
            device=draft_input_ids.device,
        ).unsqueeze(0)
        attention_mask = self._causal_incremental_mask(
            query_start=self._stable_length,
            query_length=query_length,
            key_value_length=total_length,
            device=draft_input_ids.device,
        )
        hidden_states = self._run_transformer(
            input_ids=suffix_ids,
            hidden_states=suffix_aux,
            position_ids=position_ids,
            attention_mask=attention_mask,
            cache=self._stable_cache,
            cache_position_start=self._stable_length,
        )
        self._stable_length = total_length
        self._stable_last_hidden = hidden_states[:, -1:]
        return self._stable_last_hidden, 1, query_length

    def _forward_frontier(
        self,
        frontier: _TensorFrontier,
        *,
        depth: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self._stable_cache is None:
            raise RuntimeError("draft frontier requires a stable cache")
        query_length = int(frontier.target_token_ids.shape[0])
        cache_length_before = self._stable_cache.get_seq_length()
        attention_mask = self._frontier_attention_mask(
            stable_length=self._stable_length,
            cache_length_before=cache_length_before,
            ancestor_cache_slots=frontier.ancestor_cache_slots,
            query_length=query_length,
            device=frontier.target_token_ids.device,
        )
        position_ids = torch.full(
            (1, query_length),
            self._stable_length + depth,
            dtype=torch.long,
            device=frontier.target_token_ids.device,
        )
        hidden_states = self._run_transformer(
            input_ids=frontier.target_token_ids.unsqueeze(0),
            hidden_states=frontier.parent_hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            cache=self._stable_cache,
            cache_position_start=cache_length_before,
        )
        current_slots = torch.arange(
            cache_length_before,
            cache_length_before + query_length,
            dtype=torch.long,
            device=frontier.target_token_ids.device,
        ).unsqueeze(1)
        lineage = torch.cat(
            (frontier.ancestor_cache_slots, current_slots), dim=1
        )
        return hidden_states, lineage

    def _expand_level(
        self,
        *,
        parent_raw_ids: torch.Tensor,
        parent_cumulative_log_probabilities: torch.Tensor,
        log_probabilities: torch.Tensor,
        top_k: int,
        first_raw_id: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        conditional, draft_ids = torch.topk(
            log_probabilities,
            k=top_k,
            dim=-1,
            largest=True,
            sorted=True,
        )
        target_ids = self._map_topk_ids(draft_ids)
        candidate_count = int(target_ids.numel())
        raw_ids = torch.arange(
            first_raw_id,
            first_raw_id + candidate_count,
            dtype=torch.long,
            device=target_ids.device,
        )
        parent_ids = parent_raw_ids.repeat_interleave(top_k)
        conditional = conditional.reshape(-1)
        cumulative = (
            parent_cumulative_log_probabilities.unsqueeze(1)
            + conditional.view(-1, top_k)
        ).reshape(-1)
        return (
            raw_ids,
            parent_ids,
            target_ids.reshape(-1),
            conditional,
            cumulative,
        )

    def _map_topk_ids(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return self.adapter.map_draft_to_target(draft_ids)

    @staticmethod
    def _materialize_candidates(
        *,
        raw_id_levels: list[torch.Tensor],
        parent_id_levels: list[torch.Tensor],
        token_id_levels: list[torch.Tensor],
        conditional_levels: list[torch.Tensor],
        cumulative_levels: list[torch.Tensor],
        depth_levels: list[torch.Tensor],
        prefix: tuple[Eagle3DraftCandidate, ...] = (),
    ) -> tuple[Eagle3DraftCandidate, ...]:

        raw_ids = torch.cat(raw_id_levels).tolist()
        parent_ids = torch.cat(parent_id_levels).tolist()
        token_ids = torch.cat(token_id_levels).tolist()
        conditional = torch.cat(conditional_levels).tolist()
        cumulative = torch.cat(cumulative_levels).tolist()
        depths = torch.cat(depth_levels).tolist()
        paths: dict[int, tuple[int, ...]] = {0: ()}
        paths.update((node.raw_id, node.path) for node in prefix)
        candidates = list(prefix)
        for ordinal, values in enumerate(
            zip(
                raw_ids,
                parent_ids,
                token_ids,
                conditional,
                cumulative,
                depths,
            ),
            start=len(prefix),
        ):
            raw_id, parent_id, token_id, conditional_score, score, depth = (
                values
            )
            raw_id = int(raw_id)
            parent_id = int(parent_id)
            token_id = int(token_id)
            if raw_id != ordinal + 1 or parent_id not in paths:
                raise RuntimeError(
                    "incremental EAGLE produced invalid topology"
                )
            path = paths[parent_id] + (token_id,)
            paths[raw_id] = path
            candidates.append(
                Eagle3DraftCandidate(
                    raw_id=raw_id,
                    parent_raw_id=parent_id,
                    depth=int(depth),
                    target_token_id=token_id,
                    conditional_log_probability=float(conditional_score),
                    cumulative_log_probability=float(score),
                    generation_ordinal=ordinal,
                    path=path,
                )
            )
        return tuple(candidates)

    def build_candidate_pool(
        self,
        *,
        fed_ids: torch.Tensor,
        aux_history: torch.Tensor,
        pending_seed: torch.Tensor,
        top_k: int = EAGLE3_DEFAULT_TREE_TOP_K,
        max_depth: int = EAGLE3_DEFAULT_TREE_DEPTH,
        stop_after_level: (
            Callable[[Eagle3DraftCandidatePool], bool] | None
        ) = None,
    ) -> Eagle3DraftCandidatePool:

        if fed_ids.ndim != 2 or fed_ids.shape[0] != 1:
            raise ValueError("EAGLE-3 tree currently requires batch size 1")
        if aux_history.shape[:2] != fed_ids.shape:
            raise ValueError("aux_history must align with fed_ids")
        if pending_seed.shape != (1, 1):
            raise ValueError("pending_seed must have shape [1, 1]")
        if not 1 <= top_k <= self.adapter.draft_vocab_size:
            raise ValueError("top_k is outside the draft vocabulary")
        if max_depth <= 0:
            raise ValueError("tree depth must be positive")
        if max_depth > EAGLE3_VALIDATED_TREE_DEPTH:
            raise ValueError(
                "tree depth exceeds the validated "
                f"depth-{EAGLE3_VALIDATED_TREE_DEPTH} contract"
            )
        if (
            fed_ids.ndim == 2
            and fed_ids.shape[0] == 1
            and fed_ids.shape[1] + 1 + max_depth
            > _committed_token_limit()
        ):
            raise NotImplementedError(
                "EAGLE-3 mask extension exceeds the validated committed-token "
                "contract"
            )
        draft_input_ids = torch.cat((fed_ids[:, 1:], pending_seed), dim=1)
        persistent = self._generation_active
        if not persistent:
            self.reset_state()

        try:
            root_hidden, prefix_forward_calls, prefix_query_tokens = (
                self._sync_stable_prefix(
                    draft_input_ids=draft_input_ids,
                    aux_history=aux_history,
                )
            )
            stable_length = self._stable_length
            log_probabilities = self._next_log_probabilities(root_hidden)
            parent_raw_ids = torch.zeros(
                1, dtype=torch.long, device=fed_ids.device
            )
            parent_scores = torch.zeros(
                1, dtype=torch.float32, device=fed_ids.device
            )
            parent_hidden_states = root_hidden
            parent_lineage = torch.empty(
                (1, 0), dtype=torch.long, device=fed_ids.device
            )
            raw_id_levels: list[torch.Tensor] = []
            parent_id_levels: list[torch.Tensor] = []
            token_id_levels: list[torch.Tensor] = []
            conditional_levels: list[torch.Tensor] = []
            cumulative_levels: list[torch.Tensor] = []
            depth_levels: list[torch.Tensor] = []
            first_raw_id = 1
            draft_forward_calls = prefix_forward_calls
            draft_query_tokens = prefix_query_tokens
            seed_token_id: int | None = None
            materialized: tuple[Eagle3DraftCandidate, ...] = ()
            materialized_levels = 0

            for depth in range(1, max_depth + 1):
                (
                    level_raw_ids,
                    level_parent_ids,
                    level_token_ids,
                    level_conditional,
                    level_cumulative,
                ) = self._expand_level(
                    parent_raw_ids=parent_raw_ids,
                    parent_cumulative_log_probabilities=parent_scores,
                    log_probabilities=log_probabilities,
                    top_k=top_k,
                    first_raw_id=first_raw_id,
                )
                raw_id_levels.append(level_raw_ids)
                parent_id_levels.append(level_parent_ids)
                token_id_levels.append(level_token_ids)
                conditional_levels.append(level_conditional)
                cumulative_levels.append(level_cumulative)
                depth_levels.append(
                    torch.full_like(level_raw_ids, depth, dtype=torch.long)
                )
                first_raw_id += int(level_raw_ids.numel())
                if depth == max_depth:
                    break
                if stop_after_level is not None:
                    if seed_token_id is None:
                        seed_token_id = int(pending_seed[0, 0].item())
                    materialized = self._materialize_candidates(
                        raw_id_levels=raw_id_levels[-1:],
                        parent_id_levels=parent_id_levels[-1:],
                        token_id_levels=token_id_levels[-1:],
                        conditional_levels=conditional_levels[-1:],
                        cumulative_levels=cumulative_levels[-1:],
                        depth_levels=depth_levels[-1:],
                        prefix=materialized,
                    )
                    materialized_levels = depth
                    if stop_after_level(
                        _layered_candidate_pool(
                            seed_token_id=seed_token_id,
                            top_k=top_k,
                            depth=depth,
                            candidates=materialized,
                            draft_forward_calls=draft_forward_calls,
                            draft_query_tokens=draft_query_tokens,
                        )
                    ):
                        break

                _, selected_positions = torch.topk(
                    level_cumulative,
                    k=top_k,
                    dim=0,
                    largest=True,
                    sorted=True,
                )
                selected_parent_rows = torch.div(
                    selected_positions,
                    top_k,
                    rounding_mode="floor",
                )
                frontier = _TensorFrontier(
                    target_token_ids=level_token_ids.index_select(
                        0, selected_positions
                    ),
                    cumulative_log_probabilities=(
                        level_cumulative.index_select(0, selected_positions)
                    ),
                    raw_ids=level_raw_ids.index_select(0, selected_positions),
                    parent_hidden_states=parent_hidden_states.index_select(
                        1, selected_parent_rows
                    ),
                    ancestor_cache_slots=parent_lineage.index_select(
                        0, selected_parent_rows
                    ),
                )
                parent_hidden_states, parent_lineage = self._forward_frontier(
                    frontier, depth=depth
                )
                draft_forward_calls += 1
                draft_query_tokens += top_k
                log_probabilities = self._next_log_probabilities(
                    parent_hidden_states
                )
                parent_raw_ids = frontier.raw_ids
                parent_scores = frontier.cumulative_log_probabilities

            built_depth = len(raw_id_levels)
            shallow = materialized_levels
            candidates = materialized
            if shallow < built_depth:
                candidates = self._materialize_candidates(
                    raw_id_levels=raw_id_levels[shallow:],
                    parent_id_levels=parent_id_levels[shallow:],
                    token_id_levels=token_id_levels[shallow:],
                    conditional_levels=conditional_levels[shallow:],
                    cumulative_levels=cumulative_levels[shallow:],
                    depth_levels=depth_levels[shallow:],
                    prefix=materialized,
                )
            return _layered_candidate_pool(
                seed_token_id=(
                    int(pending_seed[0, 0].item())
                    if seed_token_id is None
                    else seed_token_id
                ),
                top_k=top_k,
                depth=built_depth,
                candidates=candidates,
                draft_forward_calls=draft_forward_calls,
                draft_query_tokens=draft_query_tokens,
            )
        finally:
            if self._stable_cache is not None:
                self._stable_cache.crop(self._stable_length)
            if not persistent:
                self.reset_state()

    def propose_tree(
        self,
        *,
        fed_ids: torch.Tensor,
        aux_history: torch.Tensor,
        pending_seed: torch.Tensor,
        top_k: int = EAGLE3_DEFAULT_TREE_TOP_K,
        max_depth: int = EAGLE3_DEFAULT_TREE_DEPTH,
        draft_node_budget: int = EAGLE3_DEFAULT_TREE_NODE_BUDGET,
        committed_offset: int = 0,
    ) -> Eagle3DraftTree:

        if draft_node_budget <= 0:
            raise ValueError("draft node budget must be positive")
        pool = self.build_candidate_pool(
            fed_ids=fed_ids,
            aux_history=aux_history,
            pending_seed=pending_seed,
            top_k=top_k,
            max_depth=max_depth,
        )
        return self.selector.select(
            pool,
            draft_node_budget=draft_node_budget,
            committed_offset=committed_offset,
        )


def _compact_tree_cache(
    cache: DynamicCache,
    *,
    base_kv_length: int,
    verification_positions: tuple[int, ...],
    verification_node_count: int,
) -> None:

    expected_length = base_kv_length + verification_node_count
    source_indices = None
    destination = slice(base_kv_length, base_kv_length + len(verification_positions))
    for layer in cache.layers:
        if layer.get_seq_length() != expected_length:
            raise RuntimeError("target cache layer has an unexpected length")
        if source_indices is None or source_indices.device != layer.keys.device:
            source_indices = torch.tensor(
                tuple(base_kv_length + value for value in verification_positions),
                dtype=torch.long,
                device=layer.keys.device,
            )
        # index_select owns its result, so overlapping suffix writes are safe.
        selected_keys = layer.keys.index_select(-2, source_indices)
        selected_values = layer.values.index_select(-2, source_indices)
        layer.keys[..., destination, :].copy_(selected_keys)
        layer.values[..., destination, :].copy_(selected_values)
    cache.crop(base_kv_length + len(verification_positions))


@contextmanager
def _drafter_generation_scope(drafter: TreeDrafter):

    begin = getattr(drafter, "begin_generation", None)
    end = getattr(drafter, "end_generation", None)
    if begin is None:
        yield
        return
    if not callable(begin) or not callable(end):
        raise TypeError(
            "stateful tree drafter must define begin/end generation"
        )
    begin()
    try:
        yield
    finally:
        end()


@dataclass
class _TreeGenerationState:

    cache: object
    aux_history: torch.Tensor
    pending_seed: torch.Tensor
    fed_ids: torch.Tensor
    fed_attention: torch.Tensor
    generated: list[torch.Tensor]
    emitted: int
    proposed_total: int
    accepted_per_cycle: list[int]
    commit_audits: list[TreeCommitAudit]
    tree_cycle_metrics: list[Eagle3TreeCycleMetrics]
    drafter_state: tuple | None
    rng_state: torch.Tensor
    cuda_rng_state: torch.Tensor | None


@dataclass(frozen=True)
class _TreeGenerationCheckpoint:
    target: object
    drafter_model: object
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    config: tuple
    state: _TreeGenerationState

    def clone(self) -> _TreeGenerationCheckpoint:
        return replace(self, input_ids=self.input_ids.clone(),
            attention_mask=self.attention_mask.clone(), state=deepcopy(self.state))


class ShallowTreeSpeculator:

    def __init__(
        self,
        target: nn.Module,
        drafter: TreeDrafter,
        *,
        layer_indices: tuple[int, ...] | None = None,
    ) -> None:
        if getattr(target.config, "_attn_implementation", None) != "sdpa":
            raise ValueError("shallow tree verification currently requires SDPA")
        self.target = target
        self.drafter = drafter
        resolved_layer_indices = (
            getattr(drafter, "layer_indices", QWEN3_EAGLE3_LAYER_INDICES)
            if layer_indices is None
            else layer_indices
        )
        if (
            not resolved_layer_indices
            or len(set(resolved_layer_indices)) != len(resolved_layer_indices)
        ):
            raise ValueError("layer_indices must be non-empty and unique")
        self.layer_indices = tuple(resolved_layer_indices)

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int,
        top_k: int = EAGLE3_DEFAULT_TREE_TOP_K,
        max_depth: int = EAGLE3_DEFAULT_TREE_DEPTH,
        draft_node_budget: int = EAGLE3_DEFAULT_TREE_NODE_BUDGET,
        trace: GreedyChainTraceSink | None = None,
        on_prefill_complete: Callable[[], None] | None = None,
        stop_on_eos: bool = True,
        audit_commits: bool = False,
        selector: TreeCandidateSelector | None = None,
        collect_cycle_metrics: bool = False,
        max_cycles: int | None = None,
    ) -> Eagle3GenerationResult:
        return self._generate(
            input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
            top_k=top_k, max_depth=max_depth, draft_node_budget=draft_node_budget,
            trace=trace, on_prefill_complete=on_prefill_complete,
            stop_on_eos=stop_on_eos, audit_commits=audit_commits, selector=selector,
            collect_cycle_metrics=collect_cycle_metrics, max_cycles=max_cycles,
        )

    def _generate(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int,
        top_k: int = EAGLE3_DEFAULT_TREE_TOP_K,
        max_depth: int = EAGLE3_DEFAULT_TREE_DEPTH,
        draft_node_budget: int = EAGLE3_DEFAULT_TREE_NODE_BUDGET,
        trace: GreedyChainTraceSink | None = None,
        on_prefill_complete: Callable[[], None] | None = None,
        stop_on_eos: bool = True,
        audit_commits: bool = False,
        selector: TreeCandidateSelector | None = None,
        collect_cycle_metrics: bool = False,
        max_cycles: int | None = None,
        stop_after_level: Callable[..., bool] | None = None,
        _checkpoint: _TreeGenerationCheckpoint | None = None,
        _on_boundary: Callable[[_TreeGenerationCheckpoint], None] | None = None,
    ) -> Eagle3GenerationResult:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("tree speculation currently requires batch size 1")
        if stop_after_level is not None and selector is None:
            raise ValueError(
                "per-level stopping requires an explicit candidate-pool selector"
            )
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if max_cycles is not None and (
            isinstance(max_cycles, bool)
            or not isinstance(max_cycles, int)
            or max_cycles <= 0
        ):
            raise ValueError("max_cycles must be a positive integer or None")
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer")
        if (
            isinstance(max_depth, bool)
            or not isinstance(max_depth, int)
            or not 1 <= max_depth <= EAGLE3_VALIDATED_TREE_DEPTH
        ):
            raise ValueError(
                "max_depth must be an integer within the validated "
                f"1..{EAGLE3_VALIDATED_TREE_DEPTH} range"
            )
        if (
            isinstance(draft_node_budget, bool)
            or not isinstance(draft_node_budget, int)
            or draft_node_budget <= 0
        ):
            raise ValueError("draft_node_budget must be a positive integer")
        _validate_committed_token_budget(input_ids, max_new_tokens)
        if collect_cycle_metrics and selector is None:
            raise ValueError(
                "cycle metrics require an explicit candidate-pool selector"
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        elif attention_mask.shape != input_ids.shape or not attention_mask.all():
            raise ValueError("shallow EAGLE tree requires an unpadded input")

        eos_token_ids = (
            _normalized_eos_token_ids(self.target) if stop_on_eos else ()
        )
        stopped_on_eos = False
        stop_token_id = None
        accepted_per_cycle: list[int] = []
        proposed_total = 0
        emitted = 0
        generated: list[torch.Tensor] = []
        commit_audits: list[TreeCommitAudit] = []
        tree_cycle_metrics: list[Eagle3TreeCycleMetrics] = []

        with _drafter_generation_scope(self.drafter), AuxHiddenStateCollector(
            self.target, layer_indices=self.layer_indices
        ) as collector:
            checkpoint_config = (max_new_tokens, top_k, max_depth,
                draft_node_budget, stop_on_eos, self.layer_indices)
            if _checkpoint is None:
                outputs = self.target(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    logits_to_keep=1,
                )
                aux_history = collector.take()
                cache = outputs.past_key_values
                pending_seed = outputs.logits.argmax(dim=-1)
                if on_prefill_complete is not None:
                    on_prefill_complete()
                fed_ids = input_ids
                fed_attention = attention_mask
            else:
                if (_checkpoint.target is not self.target
                        or _checkpoint.drafter_model is not getattr(self.drafter, "model", self.drafter)
                        or _checkpoint.config != checkpoint_config
                        or not torch.equal(_checkpoint.input_ids, input_ids)
                        or not torch.equal(_checkpoint.attention_mask, attention_mask)):
                    raise ValueError("generation checkpoint source or configuration mismatch")
                state = deepcopy(_checkpoint.state)
                cache, aux_history = state.cache, state.aux_history
                pending_seed, fed_ids = state.pending_seed, state.fed_ids
                fed_attention, generated = state.fed_attention, state.generated
                emitted, proposed_total = state.emitted, state.proposed_total
                accepted_per_cycle = state.accepted_per_cycle
                commit_audits, tree_cycle_metrics = state.commit_audits, state.tree_cycle_metrics
                if state.drafter_state is not None:
                    (self.drafter._stable_cache, self.drafter._stable_length,
                     self.drafter._stable_last_hidden) = state.drafter_state
                torch.set_rng_state(state.rng_state)
                if state.cuda_rng_state is not None:
                    torch.cuda.set_rng_state(state.cuda_rng_state, input_ids.device)

            while emitted < max_new_tokens and (
                max_cycles is None or len(accepted_per_cycle) < max_cycles
            ):
                remaining = max_new_tokens - emitted
                if _first_eos_position(pending_seed, eos_token_ids) is not None:
                    generated.append(pending_seed)
                    emitted += 1
                    stopped_on_eos = True
                    stop_token_id = int(pending_seed[0, 0].item())
                    break
                if remaining == 1:
                    generated.append(pending_seed)
                    emitted += 1
                    break

                if _on_boundary is not None:
                    drafter_state = None
                    if isinstance(self.drafter, Eagle3TreeDrafter):
                        drafter_state = (self.drafter._stable_cache,
                            self.drafter._stable_length, self.drafter._stable_last_hidden)
                    elif hasattr(self.drafter, "begin_generation"):
                        raise TypeError("checkpoint requires a supported stateful drafter")
                    _on_boundary(_TreeGenerationCheckpoint(
                        target=self.target,
                        drafter_model=getattr(self.drafter, "model", self.drafter),
                        input_ids=input_ids, attention_mask=attention_mask,
                        config=checkpoint_config,
                        state=_TreeGenerationState(cache, aux_history, pending_seed,
                            fed_ids, fed_attention, generated, emitted, proposed_total,
                            accepted_per_cycle, commit_audits, tree_cycle_metrics,
                            drafter_state, torch.get_rng_state(),
                            torch.cuda.get_rng_state(input_ids.device) if input_ids.is_cuda else None),
                    ))

                iteration = len(accepted_per_cycle)
                committed_output_offset = emitted
                effective_max_depth = min(max_depth, remaining - 1)
                cycle_wall_start_ns = (
                    perf_counter_ns() if collect_cycle_metrics else 0
                )
                proposal_start_ns = (
                    perf_counter_ns() if collect_cycle_metrics else 0
                )
                pool: Eagle3DraftCandidatePool | None = None
                selector_wall_ns = 0
                if selector is None:
                    tree = self.drafter.propose_tree(
                        fed_ids=fed_ids,
                        aux_history=aux_history,
                        pending_seed=pending_seed,
                        top_k=top_k,
                        max_depth=effective_max_depth,
                        draft_node_budget=draft_node_budget,
                        committed_offset=committed_output_offset,
                    )
                else:
                    build_candidate_pool = getattr(
                        self.drafter, "build_candidate_pool", None
                    )
                    if build_candidate_pool is None:
                        raise TypeError(
                            "explicit selection requires a candidate-pool drafter"
                        )
                    level_stop_ns = 0
                    level_options = {}
                    if stop_after_level is not None:

                        def level_stop(
                            partial: Eagle3DraftCandidatePool,
                            offset: int = committed_output_offset,
                        ) -> bool:
                            nonlocal level_stop_ns
                            start_ns = perf_counter_ns()
                            try:
                                return bool(
                                    stop_after_level(
                                        partial, committed_offset=offset
                                    )
                                )
                            finally:
                                level_stop_ns += perf_counter_ns() - start_ns

                        level_options["stop_after_level"] = level_stop
                    pool = build_candidate_pool(
                        fed_ids=fed_ids,
                        aux_history=aux_history,
                        pending_seed=pending_seed,
                        top_k=top_k,
                        max_depth=effective_max_depth,
                        **level_options,
                    )
                    selection_start_ns = (
                        perf_counter_ns() if collect_cycle_metrics else 0
                    )
                    tree = selector.select(
                        pool,
                        draft_node_budget=draft_node_budget,
                        committed_offset=committed_output_offset,
                    )
                    try:
                        replayed_tree = pool.to_tree(tree.source_raw_ids)
                    except ValueError as error:
                        raise RuntimeError(
                            "explicit selector returned raw IDs outside the "
                            "candidate pool"
                        ) from error
                    if replayed_tree != tree:
                        raise RuntimeError(
                            "explicit selector tree cannot be reconstructed "
                            "from its source raw IDs"
                        )
                    if collect_cycle_metrics:
                        selector_wall_ns = (
                            perf_counter_ns()
                            - selection_start_ns
                            + level_stop_ns
                        )
                proposal_wall_ns = (
                    perf_counter_ns() - proposal_start_ns
                    if collect_cycle_metrics
                    else 0
                )
                verify_ids = tree.verification_token_ids(pending_seed)
                base_length = cache.get_seq_length()
                verify_attention = tree.verification_attention_mask(
                    base_kv_length=base_length,
                    device=verify_ids.device,
                )
                position_ids = tree.verification_position_ids(
                    base_kv_length=base_length,
                    device=verify_ids.device,
                )
                kv_prefix_guards = ()
                aux_prefix_guard = None
                if audit_commits:
                    kv_prefix_guards = tuple(
                        (
                            layer.keys[..., :base_length, :].clone(),
                            layer.values[..., :base_length, :].clone(),
                        )
                        for layer in cache.layers
                    )
                    aux_prefix_guard = aux_history.clone()
                if trace is not None:
                    trace.begin_cycle(
                        iteration=iteration,
                        base_kv_length=base_length,
                        verification_token_ids=verify_ids,
                        draft_tree=tree,
                    )
                try:
                    target_verification_start_ns = (
                        perf_counter_ns() if collect_cycle_metrics else 0
                    )
                    target_cuda_start = None
                    target_cuda_end = None
                    if collect_cycle_metrics and verify_ids.is_cuda:
                        target_cuda_start = torch.cuda.Event(enable_timing=True)
                        target_cuda_end = torch.cuda.Event(enable_timing=True)
                        verification_stream = torch.cuda.current_stream(
                            device=verify_ids.device
                        )
                        target_cuda_start.record(verification_stream)
                    outputs = self.target(
                        input_ids=verify_ids,
                        attention_mask=verify_attention,
                        position_ids=position_ids,
                        past_key_values=cache,
                        use_cache=True,
                        logits_to_keep=0,
                    )
                    query_aux = collector.take()
                    verifier_tokens = outputs.logits.argmax(dim=-1)
                    if target_cuda_end is not None:
                        target_cuda_end.record(verification_stream)
                    accepted_node_ids = tree.accepted_path(verifier_tokens)
                    verification_positions = (0,) + accepted_node_ids
                    committed = verify_ids[:, verification_positions]
                    eos_position = _first_eos_position(
                        committed, eos_token_ids
                    )
                    if eos_position is not None:
                        verification_positions = verification_positions[
                            : eos_position + 1
                        ]
                        accepted_node_ids = verification_positions[1:]
                        committed = verify_ids[:, verification_positions]
                        stopped_on_eos = True
                        stop_token_id = int(committed[0, -1].item())
                    target_verification_cuda_ns = None
                    if target_cuda_end is not None:
                        if target_cuda_start is None:
                            raise RuntimeError("target CUDA timing missed start event")
                        target_cuda_end.synchronize()
                        target_verification_cuda_ns = max(
                            0,
                            round(
                                target_cuda_start.elapsed_time(target_cuda_end)
                                * 1_000_000
                            ),
                        )
                    target_verification_wall_ns = (
                        perf_counter_ns() - target_verification_start_ns
                        if collect_cycle_metrics
                        else 0
                    )

                    position_tensor = torch.tensor(
                        verification_positions,
                        dtype=torch.long,
                        device=verify_ids.device,
                    )
                    selected_kv_guards = ()
                    selected_aux_guard = None
                    expected_pending_seed = None
                    if audit_commits:
                        source_indices = torch.tensor(
                            tuple(
                                base_length + value
                                for value in verification_positions
                            ),
                            dtype=torch.long,
                            device=verify_ids.device,
                        )
                        selected_kv_guards = tuple(
                            (
                                layer.keys.index_select(
                                    -2, source_indices
                                ).clone(),
                                layer.values.index_select(
                                    -2, source_indices
                                ).clone(),
                            )
                            for layer in cache.layers
                        )
                        selected_aux_guard = query_aux.index_select(
                            1, position_tensor
                        ).clone()
                        last_position = verification_positions[-1]
                        expected_pending_seed = verifier_tokens[
                            :, last_position : last_position + 1
                        ].clone()
                    _compact_tree_cache(
                        cache,
                        base_kv_length=base_length,
                        verification_positions=verification_positions,
                        verification_node_count=verify_ids.shape[1],
                    )
                    generated.append(committed)
                    fed_ids = torch.cat((fed_ids, committed), dim=1)
                    aux_history = torch.cat(
                        (aux_history, query_aux.index_select(1, position_tensor)),
                        dim=1,
                    )
                    fed_attention = torch.cat(
                        (fed_attention, torch.ones_like(committed)), dim=1
                    )
                    last_position = verification_positions[-1]
                    pending_seed = verifier_tokens[
                        :, last_position : last_position + 1
                    ]
                    if audit_commits:
                        expected_length = base_length + len(
                            verification_positions
                        )
                        kv_prefix_exact = all(
                            torch.equal(
                                layer.keys[..., :base_length, :],
                                prefix_keys,
                            )
                            and torch.equal(
                                layer.values[..., :base_length, :],
                                prefix_values,
                            )
                            for layer, (prefix_keys, prefix_values) in zip(
                                cache.layers, kv_prefix_guards
                            )
                        )
                        kv_selected_suffix_exact = all(
                            torch.equal(
                                layer.keys[..., base_length:, :],
                                selected_keys,
                            )
                            and torch.equal(
                                layer.values[..., base_length:, :],
                                selected_values,
                            )
                            for layer, (selected_keys, selected_values) in zip(
                                cache.layers, selected_kv_guards
                            )
                        )
                        kv_length_exact = (
                            cache.get_seq_length() == expected_length
                            and all(
                                layer.keys.shape[-2] == expected_length
                                and layer.values.shape[-2] == expected_length
                                for layer in cache.layers
                            )
                        )
                        if aux_prefix_guard is None or selected_aux_guard is None:
                            raise RuntimeError("tree commit audit missed aux guards")
                        aux_prefix_length = int(aux_prefix_guard.shape[1])
                        aux_prefix_exact = torch.equal(
                            aux_history[:, :aux_prefix_length],
                            aux_prefix_guard,
                        )
                        aux_selected_suffix_exact = torch.equal(
                            aux_history[:, aux_prefix_length:],
                            selected_aux_guard,
                        )
                        if expected_pending_seed is None:
                            raise RuntimeError(
                                "tree commit audit missed pending seed"
                            )
                        commit_audits.append(
                            TreeCommitAudit(
                                iteration=iteration,
                                base_kv_length=base_length,
                                verification_node_count=int(
                                    verify_ids.shape[1]
                                ),
                                verification_positions=tuple(
                                    int(value)
                                    for value in verification_positions
                                ),
                                rejected_node_ids=tuple(
                                    node_id
                                    for node_id in range(verify_ids.shape[1])
                                    if node_id not in verification_positions
                                ),
                                kv_layer_count=len(cache.layers),
                                kv_prefix_exact=kv_prefix_exact,
                                kv_selected_suffix_exact=(
                                    kv_selected_suffix_exact
                                ),
                                kv_length_exact=kv_length_exact,
                                aux_prefix_length=aux_prefix_length,
                                aux_length_after=int(aux_history.shape[1]),
                                aux_length_exact=(
                                    int(aux_history.shape[1])
                                    == aux_prefix_length
                                    + len(verification_positions)
                                ),
                                aux_prefix_exact=aux_prefix_exact,
                                aux_selected_suffix_exact=(
                                    aux_selected_suffix_exact
                                ),
                                pending_seed_exact=torch.equal(
                                    pending_seed, expected_pending_seed
                                ),
                            )
                        )
                    accepted_count = len(accepted_node_ids)
                    accepted_per_cycle.append(accepted_count)
                    proposed_total += tree.draft_node_count
                    emitted += committed.shape[1]
                    if trace is not None:
                        trace.end_cycle(
                            verifier_tokens=verifier_tokens,
                            accepted_draft_tokens=accepted_count,
                            committed_token_ids=committed,
                            kv_length_after=cache.get_seq_length(),
                            accepted_draft_node_ids=accepted_node_ids,
                        )
                    if collect_cycle_metrics:
                        if pool is None:
                            raise RuntimeError(
                                "cycle metrics missed the candidate pool"
                            )
                        tree_cycle_metrics.append(
                            Eagle3TreeCycleMetrics(
                                iteration=iteration,
                                committed_output_offset=(
                                    committed_output_offset
                                ),
                                actual_tree_depth=pool.actual_depth,
                                raw_candidate_nodes=(
                                    pool.raw_candidate_count
                                ),
                                selected_draft_nodes=(
                                    tree.draft_node_count
                                ),
                                accepted_draft_nodes=accepted_count,
                                path_score_calls=pool.path_score_calls,
                                draft_transformer_steps=(
                                    pool.draft_transformer_steps
                                ),
                                proposal_wall_ns=proposal_wall_ns,
                                selector_wall_ns=selector_wall_ns,
                                target_verification_wall_ns=(
                                    target_verification_wall_ns
                                ),
                                cycle_wall_ns=(
                                    perf_counter_ns()
                                    - cycle_wall_start_ns
                                ),
                                draft_forward_calls=(pool.draft_forward_calls),
                                draft_query_tokens=(pool.draft_query_tokens),
                                verification_raw_ids=(
                                    tree.verification_raw_ids
                                ),
                                accepted_raw_ids=(
                                    tree.raw_ids_for_verification_positions(
                                        accepted_node_ids
                                    )
                                ),
                                target_verification_cuda_ns=(
                                    target_verification_cuda_ns
                                ),
                                next_pending_seed_token_id=int(
                                    pending_seed[0, 0].item()
                                ),
                            )
                        )
                    if stopped_on_eos:
                        break
                except BaseException:
                    cache.crop(base_length)
                    if trace is not None:
                        trace.abort_cycle()
                    raise

        accepted_total = sum(accepted_per_cycle)
        stopped_on_cycle_limit = (
            max_cycles is not None
            and not stopped_on_eos
            and emitted < max_new_tokens
            and len(accepted_per_cycle) >= max_cycles
        )
        result = Eagle3GenerationResult(
            tokens=torch.cat(generated, dim=1)[:, :max_new_tokens],
            cycles=len(accepted_per_cycle),
            proposed_draft_tokens=proposed_total,
            accepted_draft_tokens=accepted_total,
            target_forward_calls=1 + len(accepted_per_cycle),
            accepted_per_cycle=tuple(accepted_per_cycle),
            stopped_on_eos=stopped_on_eos,
            stopped_on_cycle_limit=stopped_on_cycle_limit,
            stop_token_id=stop_token_id,
            commit_audits=tuple(commit_audits),
            tree_cycle_metrics=tuple(tree_cycle_metrics),
        )
        if trace is not None:
            finish_kwargs = {
                "output_tokens": result.tokens,
                "target_forward_calls": result.target_forward_calls,
                "stopped_on_eos": result.stopped_on_eos,
                "stop_token_id": result.stop_token_id,
            }
            if result.stopped_on_cycle_limit:
                finish_kwargs["stopped_on_cycle_limit"] = True
            trace.finish_generation(**finish_kwargs)
        return result


__all__ = [
    "CumulativeProbabilityTreeSelector",
    "Eagle3TreeDrafter",
    "PoolHindsightTreeSelector",
    "ShallowTreeSpeculator",
]
