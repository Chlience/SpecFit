
from __future__ import annotations

from dataclasses import dataclass
from math import exp, isclose, isfinite, log
from typing import Protocol

import torch


class ChainDrafter(Protocol):
    def propose(
        self,
        *,
        fed_ids: torch.Tensor,
        aux_history: torch.Tensor,
        pending_seed: torch.Tensor,
        draft_length: int,
    ) -> torch.Tensor: ...


class TreeDrafter(Protocol):
    def propose_tree(
        self,
        *,
        fed_ids: torch.Tensor,
        aux_history: torch.Tensor,
        pending_seed: torch.Tensor,
        top_k: int,
        max_depth: int,
        draft_node_budget: int,
        committed_offset: int = 0,
    ) -> Eagle3DraftTree: ...


class GreedyChainTraceSink(Protocol):

    def begin_cycle(
        self,
        *,
        iteration: int,
        base_kv_length: int,
        verification_token_ids: torch.Tensor,
        draft_tree: Eagle3DraftTree | None = None,
    ) -> None: ...

    def end_cycle(
        self,
        *,
        verifier_tokens: torch.Tensor,
        accepted_draft_tokens: int,
        committed_token_ids: torch.Tensor,
        kv_length_after: int,
        accepted_draft_node_ids: tuple[int, ...] | None = None,
    ) -> None: ...

    def abort_cycle(self) -> None: ...

    def finish_generation(
        self,
        *,
        output_tokens: torch.Tensor,
        target_forward_calls: int,
        stopped_on_eos: bool = False,
        stopped_on_cycle_limit: bool = False,
        stop_token_id: int | None = None,
    ) -> None: ...


def acceptance_prefix_length(
    verifier_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
) -> int:

    if verifier_tokens.shape != draft_tokens.shape:
        raise ValueError("verifier and draft token shapes must match")
    if verifier_tokens.ndim != 2 or verifier_tokens.shape[0] != 1:
        raise ValueError("acceptance currently requires batch size 1")
    prefix = torch.cumprod(
        verifier_tokens.eq(draft_tokens).to(torch.int64),
        dim=1,
    )
    return int(prefix.sum().item())


@dataclass(frozen=True, slots=True)
class Eagle3DraftCandidate:

    raw_id: int
    parent_raw_id: int
    depth: int
    target_token_id: int
    conditional_log_probability: float
    cumulative_log_probability: float
    generation_ordinal: int
    path: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.raw_id, bool)
            or not isinstance(self.raw_id, int)
            or self.raw_id <= 0
        ):
            raise ValueError("candidate raw IDs must be positive integers")
        if (
            isinstance(self.parent_raw_id, bool)
            or not isinstance(self.parent_raw_id, int)
            or not 0 <= self.parent_raw_id < self.raw_id
        ):
            raise ValueError("candidate parents must precede their children")
        if (
            isinstance(self.depth, bool)
            or not isinstance(self.depth, int)
            or self.depth <= 0
        ):
            raise ValueError("candidate depths must be positive integers")
        if (
            isinstance(self.target_token_id, bool)
            or not isinstance(self.target_token_id, int)
            or self.target_token_id < 0
        ):
            raise ValueError("candidate token IDs must be non-negative integers")
        if (
            not isfinite(self.conditional_log_probability)
            or self.conditional_log_probability > 0.0
        ):
            raise ValueError("conditional log probabilities must be finite and <= 0")
        if not isfinite(self.cumulative_log_probability):
            raise ValueError("cumulative log probabilities must be finite")
        if (
            isinstance(self.generation_ordinal, bool)
            or not isinstance(self.generation_ordinal, int)
            or self.generation_ordinal < 0
        ):
            raise ValueError("generation ordinals must be non-negative integers")
        if not isinstance(self.path, tuple):
            raise ValueError("candidate paths must be immutable tuples")
        if len(self.path) != self.depth or self.path[-1] != self.target_token_id:
            raise ValueError("candidate path must end at the candidate token")

    @property
    def conditional_probability(self) -> float:
        return exp(self.conditional_log_probability)


@dataclass(frozen=True, slots=True)
class Eagle3DraftCandidatePool:

    seed_token_id: int
    top_k: int
    max_depth: int
    candidates: tuple[Eagle3DraftCandidate, ...]
    path_score_calls: int
    draft_transformer_steps: int
    draft_forward_calls: int = 0
    draft_query_tokens: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed_token_id, bool)
            or not isinstance(self.seed_token_id, int)
            or self.seed_token_id < 0
        ):
            raise ValueError("candidate-pool seed must be a non-negative token ID")
        if (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or self.top_k <= 0
        ):
            raise ValueError("candidate-pool top_k must be a positive integer")
        if (
            isinstance(self.max_depth, bool)
            or not isinstance(self.max_depth, int)
            or self.max_depth <= 0
        ):
            raise ValueError("candidate-pool depth must be a positive integer")
        if not isinstance(self.candidates, tuple) or not self.candidates:
            raise ValueError("candidate pool must contain at least one node")
        if (
            isinstance(self.path_score_calls, bool)
            or not isinstance(self.path_score_calls, int)
            or self.path_score_calls <= 0
        ):
            raise ValueError("path_score_calls must be a positive integer")
        if (
            isinstance(self.draft_transformer_steps, bool)
            or not isinstance(self.draft_transformer_steps, int)
            or self.draft_transformer_steps < self.path_score_calls
        ):
            raise ValueError(
                "draft_transformer_steps must cover every path score call"
            )
        for name in ("draft_forward_calls", "draft_query_tokens"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")

        by_raw_id: dict[int, Eagle3DraftCandidate] = {}
        sibling_tokens: dict[int, set[int]] = {}
        for expected_raw_id, candidate in enumerate(self.candidates, start=1):
            if candidate.raw_id != expected_raw_id:
                raise ValueError("candidate raw IDs must be consecutive")
            if candidate.generation_ordinal != expected_raw_id - 1:
                raise ValueError("candidate ordinals must follow generation order")
            if candidate.depth > self.max_depth:
                raise ValueError("candidate depth exceeds the pool depth")
            if candidate.parent_raw_id == 0:
                parent_depth = 0
                parent_path: tuple[int, ...] = ()
                parent_score = 0.0
            else:
                parent = by_raw_id.get(candidate.parent_raw_id)
                if parent is None:
                    raise ValueError("candidate parent is absent from the pool")
                parent_depth = parent.depth
                parent_path = parent.path
                parent_score = parent.cumulative_log_probability
            if candidate.depth != parent_depth + 1:
                raise ValueError("candidate depth differs from its parent")
            if candidate.path != parent_path + (candidate.target_token_id,):
                raise ValueError("candidate path differs from its parent path")
            if not isclose(
                candidate.cumulative_log_probability,
                parent_score + candidate.conditional_log_probability,
                rel_tol=1e-5,
                abs_tol=1e-6,
            ):
                raise ValueError("candidate cumulative score is inconsistent")
            tokens = sibling_tokens.setdefault(candidate.parent_raw_id, set())
            if candidate.target_token_id in tokens:
                raise ValueError("candidate siblings must use distinct token IDs")
            tokens.add(candidate.target_token_id)
            by_raw_id[candidate.raw_id] = candidate

    @property
    def raw_candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def actual_depth(self) -> int:

        return max(candidate.depth for candidate in self.candidates)

    def to_tree(
        self,
        selected_raw_ids: tuple[int, ...],
    ) -> Eagle3DraftTree:

        if not isinstance(selected_raw_ids, tuple):
            raise ValueError("selected raw IDs must be an immutable tuple")
        if any(
            isinstance(raw_id, bool)
            or not isinstance(raw_id, int)
            or raw_id <= 0
            for raw_id in selected_raw_ids
        ):
            raise ValueError("selected raw IDs must be positive integers")
        if len(set(selected_raw_ids)) != len(selected_raw_ids):
            raise ValueError("selected raw IDs must be unique")
        by_raw_id = {node.raw_id: node for node in self.candidates}
        try:
            selected = [by_raw_id[raw_id] for raw_id in selected_raw_ids]
        except KeyError as error:
            raise ValueError("selected raw ID is absent from the pool") from error
        selected_raw_id_set = set(selected_raw_ids)
        if any(
            node.parent_raw_id != 0
            and node.parent_raw_id not in selected_raw_id_set
            for node in selected
        ):
            raise ValueError("selected candidates must be ancestor-closed")
        selected.sort(key=lambda node: node.generation_ordinal)
        raw_to_node_id = {
            node.raw_id: node_id
            for node_id, node in enumerate(selected, start=1)
        }
        return Eagle3DraftTree(
            token_ids=tuple(node.target_token_id for node in selected),
            parent_ids=tuple(
                0
                if node.parent_raw_id == 0
                else raw_to_node_id[node.parent_raw_id]
                for node in selected
            ),
            depths=tuple(node.depth for node in selected),
            conditional_probabilities=tuple(
                node.conditional_probability for node in selected
            ),
            cumulative_log_probabilities=tuple(
                node.cumulative_log_probability for node in selected
            ),
            source_raw_ids=tuple(node.raw_id for node in selected),
        )


class TreeCandidateSelector(Protocol):

    def select(
        self,
        pool: Eagle3DraftCandidatePool,
        *,
        draft_node_budget: int,
        committed_offset: int = 0,
    ) -> Eagle3DraftTree: ...


@dataclass(frozen=True, slots=True)
class Eagle3DraftTree:

    token_ids: tuple[int, ...]
    parent_ids: tuple[int, ...]
    depths: tuple[int, ...]
    conditional_probabilities: tuple[float, ...]
    cumulative_log_probabilities: tuple[float, ...]
    source_raw_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        size = len(self.token_ids)
        if not self.source_raw_ids and size:
            object.__setattr__(
                self,
                "source_raw_ids",
                tuple(range(1, size + 1)),
            )
        fields = (
            self.parent_ids,
            self.depths,
            self.conditional_probabilities,
            self.cumulative_log_probabilities,
            self.source_raw_ids,
        )
        if any(len(field) != size for field in fields):
            raise ValueError("draft tree fields must have equal lengths")
        if any(
            isinstance(raw_id, bool)
            or not isinstance(raw_id, int)
            or raw_id <= 0
            for raw_id in self.source_raw_ids
        ):
            raise ValueError("source raw IDs must be positive integers")
        if len(set(self.source_raw_ids)) != size:
            raise ValueError("source raw IDs must be unique")
        children: dict[int, set[int]] = {}
        cumulative_by_id = {0: 0.0}
        for node_id, (
            token_id,
            parent_id,
            depth,
            probability,
            cumulative_log_probability,
        ) in enumerate(
            zip(
                self.token_ids,
                self.parent_ids,
                self.depths,
                self.conditional_probabilities,
                self.cumulative_log_probabilities,
            ),
            start=1,
        ):
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
            ):
                raise ValueError("tree token IDs must be non-negative integers")
            if (
                isinstance(parent_id, bool)
                or not isinstance(parent_id, int)
                or not 0 <= parent_id < node_id
            ):
                raise ValueError("tree parents must precede their children")
            if isinstance(depth, bool) or not isinstance(depth, int):
                raise ValueError("tree depths must contain integers")
            parent_depth = 0 if parent_id == 0 else self.depths[parent_id - 1]
            if depth != parent_depth + 1:
                raise ValueError("tree node depth differs from its parent")
            if not isfinite(probability) or not 0.0 < probability <= 1.0:
                raise ValueError("conditional probabilities must be in (0, 1]")
            if not isfinite(cumulative_log_probability):
                raise ValueError("cumulative log probabilities must be finite")
            expected_cumulative = cumulative_by_id[parent_id] + log(probability)
            if not isclose(
                cumulative_log_probability,
                expected_cumulative,
                rel_tol=1e-5,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    "cumulative score differs from its parent and conditional "
                    "probability"
                )
            sibling_tokens = children.setdefault(parent_id, set())
            if token_id in sibling_tokens:
                raise ValueError("siblings must use distinct token IDs")
            sibling_tokens.add(token_id)
            cumulative_by_id[node_id] = cumulative_log_probability

    @property
    def draft_node_count(self) -> int:
        return len(self.token_ids)

    @property
    def verification_raw_ids(self) -> tuple[int, ...]:

        return (0,) + self.source_raw_ids

    def raw_ids_for_verification_positions(
        self,
        verification_positions: tuple[int, ...],
    ) -> tuple[int, ...]:

        if any(
            isinstance(position, bool)
            or not isinstance(position, int)
            or not 0 <= position <= self.draft_node_count
            for position in verification_positions
        ):
            raise ValueError("verification positions are outside the draft tree")
        return tuple(
            self.verification_raw_ids[position]
            for position in verification_positions
        )

    def verification_token_ids(self, pending_seed: torch.Tensor) -> torch.Tensor:
        if pending_seed.shape != (1, 1):
            raise ValueError("pending_seed must have shape [1, 1]")
        drafts = pending_seed.new_tensor(self.token_ids).unsqueeze(0)
        return torch.cat((pending_seed, drafts), dim=1)

    def verification_position_ids(
        self,
        *,
        base_kv_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        if base_kv_length < 0:
            raise ValueError("base_kv_length must be non-negative")
        return torch.tensor(
            (base_kv_length,)
            + tuple(base_kv_length + depth for depth in self.depths),
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

    def verification_attention_mask(
        self,
        *,
        base_kv_length: int,
        device: torch.device,
    ) -> torch.Tensor:

        if base_kv_length < 0:
            raise ValueError("base_kv_length must be non-negative")
        query_count = 1 + self.draft_node_count
        mask = torch.zeros(
            (1, 1, query_count, base_kv_length + query_count),
            dtype=torch.bool,
            device=device,
        )
        mask[..., :base_kv_length] = True
        for node_id in range(query_count):
            ancestor = node_id
            while True:
                mask[0, 0, node_id, base_kv_length + ancestor] = True
                if ancestor == 0:
                    break
                ancestor = self.parent_ids[ancestor - 1]
        return mask

    def accepted_path(
        self,
        verifier_token_ids: torch.Tensor,
    ) -> tuple[int, ...]:

        expected_shape = (1, 1 + self.draft_node_count)
        if verifier_token_ids.shape != expected_shape:
            raise ValueError("verifier tokens do not match the draft tree")
        verifier = tuple(int(value) for value in verifier_token_ids[0].tolist())
        children: dict[int, list[int]] = {}
        for node_id, parent_id in enumerate(self.parent_ids, start=1):
            children.setdefault(parent_id, []).append(node_id)
        accepted = []
        parent_id = 0
        while True:
            predicted = verifier[parent_id]
            matching = tuple(
                node_id
                for node_id in children.get(parent_id, ())
                if self.token_ids[node_id - 1] == predicted
            )
            if not matching:
                break
            if len(matching) != 1:
                raise RuntimeError("target token matches multiple sibling nodes")
            parent_id = matching[0]
            accepted.append(parent_id)
        return tuple(accepted)


@dataclass(frozen=True, slots=True)
class TreeCommitAudit:

    iteration: int
    base_kv_length: int
    verification_node_count: int
    verification_positions: tuple[int, ...]
    rejected_node_ids: tuple[int, ...]
    kv_layer_count: int
    kv_prefix_exact: bool
    kv_selected_suffix_exact: bool
    kv_length_exact: bool
    aux_prefix_length: int
    aux_length_after: int
    aux_length_exact: bool
    aux_prefix_exact: bool
    aux_selected_suffix_exact: bool
    pending_seed_exact: bool

    @property
    def passed(self) -> bool:

        return (
            self.kv_prefix_exact
            and self.kv_selected_suffix_exact
            and self.kv_length_exact
            and self.aux_length_exact
            and self.aux_prefix_exact
            and self.aux_selected_suffix_exact
            and self.pending_seed_exact
        )


@dataclass(frozen=True, slots=True)
class Eagle3TreeCycleMetrics:

    iteration: int
    committed_output_offset: int
    actual_tree_depth: int
    raw_candidate_nodes: int
    selected_draft_nodes: int
    accepted_draft_nodes: int
    path_score_calls: int
    draft_transformer_steps: int
    proposal_wall_ns: int
    selector_wall_ns: int
    target_verification_wall_ns: int
    cycle_wall_ns: int
    draft_forward_calls: int = 0
    draft_query_tokens: int = 0
    verification_raw_ids: tuple[int, ...] = ()
    accepted_raw_ids: tuple[int, ...] = ()
    target_verification_cuda_ns: int | None = None
    next_pending_seed_token_id: int | None = None

    def __post_init__(self) -> None:
        nonnegative_fields = (
            "iteration",
            "committed_output_offset",
            "selected_draft_nodes",
            "accepted_draft_nodes",
            "proposal_wall_ns",
            "selector_wall_ns",
            "target_verification_wall_ns",
            "cycle_wall_ns",
            "draft_forward_calls",
            "draft_query_tokens",
        )
        positive_fields = (
            "actual_tree_depth",
            "raw_candidate_nodes",
            "path_score_calls",
            "draft_transformer_steps",
        )
        for name in nonnegative_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in positive_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.selected_draft_nodes > self.raw_candidate_nodes:
            raise ValueError("selected nodes cannot exceed the candidate pool")
        if self.accepted_draft_nodes > self.selected_draft_nodes:
            raise ValueError("accepted nodes cannot exceed selected nodes")
        if self.accepted_draft_nodes > self.actual_tree_depth:
            raise ValueError("accepted nodes cannot exceed the actual tree depth")
        if self.selector_wall_ns > self.proposal_wall_ns:
            raise ValueError("selector time cannot exceed proposal time")
        if self.proposal_wall_ns > self.cycle_wall_ns:
            raise ValueError("proposal time cannot exceed cycle time")
        if self.target_verification_wall_ns > self.cycle_wall_ns:
            raise ValueError("target verification time cannot exceed cycle time")
        if (
            self.proposal_wall_ns + self.target_verification_wall_ns
            > self.cycle_wall_ns
        ):
            raise ValueError(
                "proposal and target verification time cannot exceed cycle time"
            )
        if self.target_verification_cuda_ns is not None and (
            isinstance(self.target_verification_cuda_ns, bool)
            or not isinstance(self.target_verification_cuda_ns, int)
            or self.target_verification_cuda_ns < 0
        ):
            raise ValueError(
                "target_verification_cuda_ns must be a non-negative integer or None"
            )
        if self.next_pending_seed_token_id is not None and (
            isinstance(self.next_pending_seed_token_id, bool)
            or not isinstance(self.next_pending_seed_token_id, int)
            or self.next_pending_seed_token_id < 0
        ):
            raise ValueError(
                "next_pending_seed_token_id must be a non-negative integer or None"
            )
        if self.verification_raw_ids:
            if (
                len(self.verification_raw_ids) != self.selected_draft_nodes + 1
                or self.verification_raw_ids[0] != 0
                or any(
                    isinstance(raw_id, bool)
                    or not isinstance(raw_id, int)
                    or raw_id <= 0
                    for raw_id in self.verification_raw_ids[1:]
                )
                or len(set(self.verification_raw_ids[1:]))
                != self.selected_draft_nodes
            ):
                raise ValueError(
                    "verification raw IDs must map seed and every selected node"
                )
            if (
                len(self.accepted_raw_ids) != self.accepted_draft_nodes
                or any(
                    raw_id not in self.verification_raw_ids[1:]
                    for raw_id in self.accepted_raw_ids
                )
            ):
                raise ValueError(
                    "accepted raw IDs must map every accepted draft node"
                )
        elif self.accepted_raw_ids:
            raise ValueError(
                "accepted raw IDs require the verification raw-ID mapping"
            )


@dataclass(frozen=True, slots=True)
class Eagle3GenerationResult:
    tokens: torch.Tensor
    cycles: int
    proposed_draft_tokens: int
    accepted_draft_tokens: int
    target_forward_calls: int
    accepted_per_cycle: tuple[int, ...]
    stopped_on_eos: bool = False
    stop_token_id: int | None = None
    commit_audits: tuple[TreeCommitAudit, ...] = ()
    tree_cycle_metrics: tuple[Eagle3TreeCycleMetrics, ...] = ()
    stopped_on_cycle_limit: bool = False

    @property
    def acceptance_rate(self) -> float:
        if self.proposed_draft_tokens == 0:
            return 0.0
        return self.accepted_draft_tokens / self.proposed_draft_tokens

    @property
    def mean_commit_length(self) -> float:
        if self.cycles == 0:
            return 1.0
        return 1.0 + self.accepted_draft_tokens / self.cycles
