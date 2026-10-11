
from __future__ import annotations

import torch
from speculators.models.attention import (
    ALL_ATTENTION_FUNCTIONS,
    flex_attention_forward as _speculators_flex_attention_forward,
)
from speculators.models.eagle3 import core as eagle3_core
from speculators.models.eagle3.attention import (
    extend_mask_for_draft_tokens as _speculators_extend_mask,
)
from torch.nn.attention.flex_attention import BlockMask, create_block_mask


@torch.compiler.disable
def _materialize_query(query: torch.Tensor) -> torch.Tensor:

    return query.contiguous()


def _adjusted_flex_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: object,
    scaling: float | None = None,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    query = _materialize_query(query)
    query_length = query.shape[-2]
    key_length = key.shape[-2]
    adjusted_mask = attention_mask
    if attention_mask.shape[-2:] != (query_length, key_length):
        adjusted_mask = attention_mask._adjust(query_length, key_length)
        adjusted_mask.seq_lengths = (query_length, key_length)
    return _speculators_flex_attention_forward(
        module,
        query,
        key,
        value,
        adjusted_mask,
        scaling=scaling,
        **kwargs,
    )


def extend_eagle3_mask_for_draft_tokens(block_mask: BlockMask) -> BlockMask:

    query_length, key_length = block_mask.seq_lengths
    query_block_size, _ = block_mask.BLOCK_SIZE
    if query_length <= query_block_size:
        return _speculators_extend_mask(block_mask)
    return create_block_mask(
        block_mask.mask_mod,
        B=None,
        H=None,
        Q_LEN=query_length,
        KV_LEN=key_length + query_length,
        device=block_mask.kv_indices.device,
        BLOCK_SIZE=block_mask.BLOCK_SIZE,
    )


def install_eagle3_flex_attention_compatibility() -> None:
    ALL_ATTENTION_FUNCTIONS.register(
        "simple_flex_attention",
        _adjusted_flex_attention_forward,
    )
    eagle3_core.extend_mask_for_draft_tokens = (
        extend_eagle3_mask_for_draft_tokens
    )
