
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from specfit.state import ExpertKey

from .gpu_expert_cache import GpuExpertCache
from .lease import ExpertLease, NoEvictableSlotError


class ReductionMode(str, Enum):

    CANONICAL = "canonical"
    ARRIVAL_ORDER = "arrival_order"


class ExpertExecutionPolicy(str, Enum):

    EXPERT_ID_ASCENDING = "expert_id_ascending"
    RESIDENT_FIRST = "resident_first"


@dataclass(frozen=True, slots=True)
class _ExpertContribution:
    expert_id: int
    token_indices: torch.Tensor
    values: torch.Tensor


def _expert_forward(
    hidden_states: torch.Tensor,
    packed_weights: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
) -> torch.Tensor:
    projection_elements = hidden_size * intermediate_size
    expected_elements = 3 * projection_elements
    if packed_weights.numel() != expected_elements:
        raise ValueError(
            f"packed expert has {packed_weights.numel()} elements, "
            f"expected {expected_elements}"
        )
    gate_up = packed_weights[: 2 * projection_elements].view(
        2 * intermediate_size,
        hidden_size,
    )
    down = packed_weights[2 * projection_elements :].view(
        hidden_size,
        intermediate_size,
    )
    gate, up = F.linear(hidden_states, gate_up).chunk(2, dim=-1)
    activated = F.silu(gate) * up
    return F.linear(activated, down)


class Qwen3OffloadedExperts(nn.Module):

    def __init__(
        self,
        *,
        layer_id: int,
        cache: GpuExpertCache,
        hidden_size: int,
        intermediate_size: int,
        reduction_mode: ReductionMode | str = ReductionMode.CANONICAL,
        execution_policy: ExpertExecutionPolicy | str = (
            ExpertExecutionPolicy.EXPERT_ID_ASCENDING
        ),
        h2d_prefetch: bool = True,
    ) -> None:
        super().__init__()
        if layer_id < 0:
            raise ValueError("layer_id must be non-negative")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if type(h2d_prefetch) is not bool:
            raise ValueError("h2d_prefetch must be a boolean")
        self.layer_id = layer_id
        self.cache = cache
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.reduction_mode = ReductionMode(reduction_mode)
        self.execution_policy = ExpertExecutionPolicy(execution_policy)
        self.h2d_prefetch = h2d_prefetch
        self._prediction_scheduler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        *,
        execution_order: Sequence[int] | None = None,
    ) -> torch.Tensor:
        self._validate_inputs(hidden_states, top_k_index, top_k_weights)
        if self._prediction_scheduler is not None:
            if execution_order is not None:
                raise ValueError("explicit execution_order conflicts with dynamic predicted prefetch")
            return self._forward_prefetched(hidden_states, top_k_index, top_k_weights)
        active_experts = tuple(
            sorted(int(value) for value in torch.unique(top_k_index).tolist())
        )
        order = self._resolve_execution_order(active_experts, execution_order)
        contributions = []
        final_hidden_states = torch.zeros_like(hidden_states)

        with closing(self._leased_experts(order)) as scheduled:
            for expert_id, lease in scheduled:
                token_indices, top_k_positions = torch.where(
                    top_k_index == expert_id
                )
                if lease is None:
                    lease = self.cache.acquire(ExpertKey(self.layer_id, expert_id))
                with lease:
                    current_hidden_states = _expert_forward(
                        hidden_states[token_indices],
                        lease.wait(),
                        hidden_size=self.hidden_size,
                        intermediate_size=self.intermediate_size,
                    )
                    current_hidden_states.mul_(
                        top_k_weights[
                            token_indices,
                            top_k_positions,
                            None,
                        ]
                    )
                contribution = _ExpertContribution(
                    expert_id=expert_id,
                    token_indices=token_indices,
                    values=current_hidden_states,
                )
                if self.reduction_mode is ReductionMode.ARRIVAL_ORDER:
                    final_hidden_states.index_add_(
                        0,
                        contribution.token_indices,
                        contribution.values.to(final_hidden_states.dtype),
                    )
                else:
                    contributions.append(contribution)

        if self.reduction_mode is ReductionMode.ARRIVAL_ORDER:
            return final_hidden_states
        for contribution in sorted(
            contributions,
            key=lambda item: item.expert_id,
        ):
            final_hidden_states.index_add_(
                0,
                contribution.token_indices,
                contribution.values.to(final_hidden_states.dtype),
            )
        return final_hidden_states

    def _forward_prefetched(self, hidden_states, top_k_index, top_k_weights):
        """Group routes once, then submit whichever actual expert is ready.

        Stable sorting preserves the row-major token/Top-K position order of
        each historical ``where``. All experts still use their original MLP
        batch shape, and weighted contributions retain canonical reduction.
        """
        sorted_ids, positions = torch.sort(top_k_index.reshape(-1), stable=True)
        unique, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
        expert_ids, sizes = torch.stack((unique, counts)).cpu().tolist()
        width = top_k_index.shape[1]
        route_tokens = torch.div(positions, width, rounding_mode="floor")
        token_groups = route_tokens.split(sizes)
        groups = None
        contributions = []
        final_hidden_states = torch.zeros_like(hidden_states)
        scheduled = self._prediction_scheduler.experts(self.layer_id, tuple(expert_ids))
        with closing(scheduled):
            for expert_id, lease in scheduled:
                with lease:
                    if groups is None:
                        # Let the scheduler submit real misses before gathering
                        # this layer's inputs once in stable expert order.
                        hidden_groups = hidden_states[route_tokens].split(sizes)
                        weight_groups = top_k_weights.reshape(-1)[positions, None].split(sizes)
                        groups = dict(zip(expert_ids,
                            zip(token_groups, hidden_groups, weight_groups, strict=True), strict=True))
                    token_indices, expert_inputs, expert_weights = groups[expert_id]
                    values = _expert_forward(
                        expert_inputs, lease.wait(),
                        hidden_size=self.hidden_size, intermediate_size=self.intermediate_size,
                    )
                    values.mul_(expert_weights)
                contributions.append(_ExpertContribution(expert_id, token_indices, values))
        for contribution in sorted(contributions, key=lambda item: item.expert_id):
            final_hidden_states.index_add_(0, contribution.token_indices,
                contribution.values.to(final_hidden_states.dtype))
        return final_hidden_states

    def _leased_experts(
        self, order: tuple[int, ...]
    ) -> Iterator[tuple[int, ExpertLease | None]]:
        if not self.h2d_prefetch:
            for expert_id in order:
                yield expert_id, None
            return
        pending = None
        try:
            for index, expert_id in enumerate(order):
                current = pending
                pending = None
                if current is None:
                    current = self.cache.acquire(ExpertKey(self.layer_id, expert_id))
                with current:
                    if index + 1 < len(order):
                        try:
                            pending = self.cache._prefetch(
                                ExpertKey(self.layer_id, order[index + 1])
                            )
                        except NoEvictableSlotError:
                            pass
                    yield expert_id, current
        finally:
            if pending is not None:
                pending.release()

    def _validate_inputs(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> None:
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must have shape [tokens, hidden]")
        if hidden_states.shape[1] != self.hidden_size:
            raise ValueError(
                f"hidden_states width is {hidden_states.shape[1]}, "
                f"expected {self.hidden_size}"
            )
        if top_k_index.ndim != 2:
            raise ValueError("top_k_index must have shape [tokens, top_k]")
        if top_k_weights.shape != top_k_index.shape:
            raise ValueError("top_k_weights must match top_k_index shape")
        if top_k_index.shape[0] != hidden_states.shape[0]:
            raise ValueError("routing tensors must match the token count")
        if top_k_index.dtype is not torch.long:
            raise ValueError("top_k_index must use torch.long")
        if hidden_states.device != self.cache.device:
            raise ValueError("hidden_states must be on the cache device")
        if top_k_index.device != hidden_states.device:
            raise ValueError("top_k_index must be on the hidden-state device")
        if top_k_weights.device != hidden_states.device:
            raise ValueError(
                "top_k_weights must be on the hidden-state device"
            )

    def _resolve_execution_order(
        self,
        active_experts: tuple[int, ...],
        execution_order: Sequence[int] | None,
    ) -> tuple[int, ...]:
        if execution_order is not None:
            order = tuple(int(expert_id) for expert_id in execution_order)
            if len(order) != len(set(order)):
                raise ValueError("execution_order must not contain duplicates")
            if set(order) != set(active_experts):
                raise ValueError(
                    "execution_order must contain every active expert "
                    "exactly once"
                )
            return order
        if (
            self.execution_policy
            is ExpertExecutionPolicy.EXPERT_ID_ASCENDING
        ):
            return active_experts

        resident_keys = frozenset(self.cache.resident_expert_ids())
        return tuple(
            expert_id
            for expert_id in active_experts
            if ExpertKey(self.layer_id, expert_id) in resident_keys
        ) + tuple(
            expert_id
            for expert_id in active_experts
            if ExpertKey(self.layer_id, expert_id) not in resident_keys
        )
