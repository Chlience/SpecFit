
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch

from .eagle3_chain import (
    EAGLE3_DEFAULT_TREE_DEPTH,
    EAGLE3_DEFAULT_TREE_NODE_BUDGET,
    EAGLE3_DEFAULT_TREE_TOP_K,
    _committed_token_limit,
    Eagle3ChainDrafter,
)


class Eagle3TreeWarmupDrafter(Protocol):

    def propose_tree(
        self,
        *,
        fed_ids: torch.Tensor,
        aux_history: torch.Tensor,
        pending_seed: torch.Tensor,
        top_k: int,
        max_depth: int,
        draft_node_budget: int,
    ) -> object: ...


def eagle3_tree_temporary_kv_tokens(
    *,
    sequence_length: int,
    max_depth: int,
    top_k: int = EAGLE3_DEFAULT_TREE_TOP_K,
) -> int:

    if sequence_length <= 0 or max_depth <= 0 or top_k <= 0:
        raise ValueError("tree warmup lengths must be positive")
    if sequence_length + 1 + max_depth > _committed_token_limit():
        raise ValueError("tree warmup exceeds the committed-length contract")
    return sequence_length + top_k * (max_depth - 1)


def eagle3_tree_warmup_anchor_lengths(
    *,
    max_sequence_length: int,
    max_depth: int,
    top_k: int = EAGLE3_DEFAULT_TREE_TOP_K,
    sequence_anchors: Sequence[int] | None = None,
) -> tuple[int, ...]:

    eagle3_tree_temporary_kv_tokens(
        sequence_length=max_sequence_length,
        max_depth=max_depth,
        top_k=top_k,
    )
    if sequence_anchors is None:
        return (max_sequence_length,)
    if isinstance(sequence_anchors, (str, bytes)) or not sequence_anchors:
        raise ValueError("tree warmup sequence anchors must be non-empty")

    anchors: list[int] = []
    previous = 0
    minimum_incremental_anchor = sum(range(1, max_depth + 2)) + 1
    for index, value in enumerate(sequence_anchors):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"tree warmup sequence anchor {index} must be an integer"
            )
        if value <= previous:
            raise ValueError(
                "tree warmup sequence anchors must be strictly increasing"
            )
        if value < minimum_incremental_anchor:
            raise ValueError(
                "tree warmup sequence anchor is too short for incremental "
                "suffix coverage"
            )
        eagle3_tree_temporary_kv_tokens(
            sequence_length=value,
            max_depth=max_depth,
            top_k=top_k,
        )
        anchors.append(value)
        previous = value
    if anchors[-1] != max_sequence_length:
        raise ValueError(
            "the last tree warmup sequence anchor must equal "
            "max_sequence_length"
        )
    return tuple(anchors)


def _synthetic_inputs(
    drafter: Eagle3ChainDrafter,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight = drafter.model.fc.weight
    draft_zero = torch.zeros(1, dtype=torch.long, device=weight.device)
    target_zero = drafter.map_draft_to_target(draft_zero).reshape(1, 1)
    fed_ids = target_zero.expand(1, sequence_length)
    aux_history = weight.new_zeros(
        (1, sequence_length, drafter.model.fc.in_features)
    )
    return fed_ids, aux_history, target_zero


def eagle3_warmup_sequence_lengths(
    *,
    max_sequence_length: int,
    draft_length: int,
) -> tuple[int, ...]:

    if max_sequence_length <= 0 or draft_length <= 0:
        raise ValueError("warmup token lengths must be positive")
    max_supported = _committed_token_limit() - 1 - draft_length
    if max_sequence_length > max_supported:
        raise ValueError("warmup sequence length exceeds the EAGLE-3 contract")

    first = min(64, max_sequence_length)
    lengths = [first]
    if first > 1:
        lengths.append(first - 1)
    if max_sequence_length >= 129:
        lengths.append(129)
    return tuple(lengths)


def warmup_eagle3_drafter(
    drafter: Eagle3ChainDrafter,
    *,
    max_sequence_length: int,
    draft_length: int,
) -> tuple[int, ...]:

    lengths = eagle3_warmup_sequence_lengths(
        max_sequence_length=max_sequence_length,
        draft_length=draft_length,
    )
    with torch.inference_mode():
        for length in lengths:
            fed_ids, aux_history, target_zero = _synthetic_inputs(
                drafter, length
            )
            drafter.propose(
                fed_ids=fed_ids,
                aux_history=aux_history,
                pending_seed=target_zero,
                draft_length=draft_length,
            )
    return lengths


def warmup_eagle3_tree_drafter(
    drafter: Eagle3ChainDrafter,
    tree_drafter: Eagle3TreeWarmupDrafter,
    *,
    max_sequence_length: int,
    max_depth: int = EAGLE3_DEFAULT_TREE_DEPTH,
    top_k: int = EAGLE3_DEFAULT_TREE_TOP_K,
    draft_node_budget: int = EAGLE3_DEFAULT_TREE_NODE_BUDGET,
    sequence_anchors: Sequence[int] | None = None,
) -> tuple[int, ...]:

    anchors = eagle3_tree_warmup_anchor_lengths(
        max_sequence_length=max_sequence_length,
        max_depth=max_depth,
        top_k=top_k,
        sequence_anchors=sequence_anchors,
    )
    if top_k <= 0 or draft_node_budget <= 0:
        raise ValueError("tree warmup topology must be positive")

    begin_generation = getattr(tree_drafter, "begin_generation", None)
    end_generation = getattr(tree_drafter, "end_generation", None)
    stateful = callable(begin_generation) and callable(end_generation)
    if sequence_anchors is not None and not stateful:
        raise ValueError(
            "explicit tree warmup sequence anchors require a stateful drafter"
        )

    fed_ids, aux_history, pending_seed = _synthetic_inputs(
        drafter, max_sequence_length
    )
    with torch.inference_mode():
        tree_drafter.propose_tree(
            fed_ids=fed_ids,
            aux_history=aux_history,
            pending_seed=pending_seed,
            top_k=top_k,
            max_depth=max_depth,
            draft_node_budget=draft_node_budget,
        )
        if stateful:
            suffix_lengths = tuple(range(1, max_depth + 2))
            max_fed_length = _committed_token_limit() - 1 - max_depth
            for anchor in anchors:
                base_length = (
                    min(anchor, max_fed_length) - sum(suffix_lengths)
                )
                if base_length <= 0:
                    if sequence_anchors is not None:
                        raise ValueError(
                            "tree warmup sequence anchor is too short for "
                            "incremental suffix coverage"
                        )
                    continue
                fed_ids, aux_history, pending_seed = _synthetic_inputs(
                    drafter, base_length
                )
                begin_generation()
                try:
                    tree_drafter.propose_tree(
                        fed_ids=fed_ids,
                        aux_history=aux_history,
                        pending_seed=pending_seed,
                        top_k=top_k,
                        max_depth=max_depth,
                        draft_node_budget=draft_node_budget,
                    )
                    for suffix_length in suffix_lengths:
                        fed_ids = torch.cat(
                            (
                                fed_ids,
                                pending_seed.expand(1, suffix_length),
                            ),
                            dim=1,
                        )
                        aux_history = torch.cat(
                            (
                                aux_history,
                                aux_history.new_zeros(
                                    (
                                        1,
                                        suffix_length,
                                        aux_history.shape[-1],
                                    )
                                ),
                            ),
                            dim=1,
                        )
                        tree_drafter.propose_tree(
                            fed_ids=fed_ids,
                            aux_history=aux_history,
                            pending_seed=pending_seed,
                            top_k=top_k,
                            max_depth=max_depth,
                            draft_node_budget=draft_node_budget,
                        )
                finally:
                    end_generation()
    return anchors
