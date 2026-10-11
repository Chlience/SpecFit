
from __future__ import annotations

import hashlib
import json
import math
import types
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from functools import cache
from typing import Any, Mapping, Union, get_args, get_origin, get_type_hints


TRACE_SCHEMA_VERSION = 1


def _require_schema_version(version: int) -> None:
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != TRACE_SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported trace schema {version}")


def _require_tuple(name: str, value: object) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be an immutable tuple")


def _require_nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty")


def _require_string(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")


def _require_bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


def _require_nonnegative(**values: int) -> None:
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")


def _require_probability(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be in [0, 1]")


def _require_nonnegative_int_tuple(name: str, value: tuple[int, ...]) -> None:
    _require_tuple(name, value)
    for item in value:
        _require_nonnegative(**{name: item})


def stable_trace_id(kind: str, *components: object) -> str:

    _require_nonempty("trace ID kind", kind)
    encoded = json.dumps(
        (kind, *components),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{kind}-{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True, slots=True)
class ManifestEntry:

    key: str
    value: str
    schema_version: int = TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_nonempty("manifest key", self.key)
        _require_string("manifest value", self.value)


@dataclass(frozen=True, slots=True)
class WorkloadManifestTrace:

    run_id: str
    request_id: str
    git_commit: str
    image_id: str
    target_model: str
    target_revision: str
    drafter_model: str
    drafter_revision: str
    tokenizer_revision: str
    device: str
    dtype: str
    reduction_mode: str
    drafter_mode: str
    slots_per_layer: int
    num_moe_layers: int
    num_experts_per_layer: int
    experts_per_token: int
    draft_length: int
    max_new_tokens: int
    seed: int
    raw_prompt: str
    rendered_prompt: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    metadata: tuple[ManifestEntry, ...] = ()
    schema_version: int = TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        for name in (
            "run_id",
            "request_id",
            "git_commit",
            "image_id",
            "target_model",
            "target_revision",
            "drafter_model",
            "drafter_revision",
            "tokenizer_revision",
            "device",
            "dtype",
            "reduction_mode",
            "drafter_mode",
        ):
            _require_nonempty(name, getattr(self, name))
        _require_string("raw_prompt", self.raw_prompt)
        _require_string("rendered_prompt", self.rendered_prompt)
        _require_nonnegative(
            slots_per_layer=self.slots_per_layer,
            num_moe_layers=self.num_moe_layers,
            num_experts_per_layer=self.num_experts_per_layer,
            experts_per_token=self.experts_per_token,
            draft_length=self.draft_length,
            max_new_tokens=self.max_new_tokens,
            seed=self.seed,
        )
        if (
            self.slots_per_layer == 0
            or self.num_moe_layers == 0
            or self.num_experts_per_layer == 0
            or self.experts_per_token == 0
            or self.max_new_tokens == 0
        ):
            raise ValueError("model dimensions and output length must be positive")
        if self.experts_per_token > self.num_experts_per_layer:
            raise ValueError("experts_per_token exceeds the expert count")
        for name in ("input_ids", "attention_mask", "metadata"):
            _require_tuple(name, getattr(self, name))
        if not self.input_ids:
            raise ValueError("input_ids must be non-empty")
        if len(self.input_ids) != len(self.attention_mask):
            raise ValueError("attention_mask must match input_ids")
        _require_nonnegative_int_tuple("input_ids", self.input_ids)
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value not in (0, 1)
            for value in self.attention_mask
        ):
            raise ValueError("attention_mask values must be zero or one")
        ordered = tuple(sorted(self.metadata, key=lambda item: item.key))
        if len({item.key for item in ordered}) != len(ordered):
            raise ValueError("manifest metadata keys must be unique")
        object.__setattr__(self, "metadata", ordered)

    @property
    def trace_id(self) -> str:
        return self.run_id


class VerificationNodeRole(StrEnum):
    SEED = "seed"
    DRAFT = "draft"


@dataclass(frozen=True, slots=True)
class DraftNodeTrace:

    tree_id: str
    node_id: int
    parent_id: int | None
    depth: int
    role: VerificationNodeRole
    token_id: int
    conditional_probability: float | None
    verification_position: int | None
    selected: bool
    verified: bool
    accepted: bool
    schema_version: int = TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_nonempty("tree_id", self.tree_id)
        _require_nonnegative(
            node_id=self.node_id,
            depth=self.depth,
            token_id=self.token_id,
        )
        if not isinstance(self.role, VerificationNodeRole):
            object.__setattr__(self, "role", VerificationNodeRole(self.role))
        if self.parent_id is not None:
            _require_nonnegative(parent_id=self.parent_id)
        if self.conditional_probability is not None:
            _require_probability(
                "conditional_probability", self.conditional_probability
            )
        if self.verification_position is not None:
            _require_nonnegative(
                verification_position=self.verification_position
            )
        for name in ("selected", "verified", "accepted"):
            _require_bool(name, getattr(self, name))
        if self.verified and not self.selected:
            raise ValueError("a verified node must be selected")
        if self.accepted and not self.verified:
            raise ValueError("an accepted node must be verified")
        if self.verified != (self.verification_position is not None):
            raise ValueError(
                "verification_position must be present exactly for verified nodes"
            )

    @property
    def trace_id(self) -> str:
        return stable_trace_id("node", self.tree_id, self.node_id)


@dataclass(frozen=True, slots=True)
class DraftTreeTrace:

    tree_id: str
    request_id: str
    iteration: int
    nodes: tuple[DraftNodeTrace, ...]
    schema_version: int = TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_nonempty("tree_id", self.tree_id)
        _require_nonempty("request_id", self.request_id)
        _require_nonnegative(iteration=self.iteration)
        _require_tuple("nodes", self.nodes)
        if not self.nodes:
            raise ValueError("nodes must be non-empty")
        if any(node.tree_id != self.tree_id for node in self.nodes):
            raise ValueError("every node must reference its containing tree")

        ordered = tuple(
            sorted(
                self.nodes,
                key=lambda node: (
                    node.depth,
                    node.verification_position
                    if node.verification_position is not None
                    else 2**63,
                    node.node_id,
                ),
            )
        )
        object.__setattr__(self, "nodes", ordered)
        by_id = {node.node_id: node for node in ordered}
        if len(by_id) != len(ordered):
            raise ValueError("node IDs must be unique")
        roots = [node for node in ordered if node.parent_id is None]
        if len(roots) != 1 or roots[0].depth != 0:
            raise ValueError("a trace tree must contain one depth-zero root")
        root = roots[0]
        if (
            root.role is not VerificationNodeRole.SEED
            or root.conditional_probability is not None
            or root.verification_position != 0
            or not root.selected
            or not root.verified
            or not root.accepted
        ):
            raise ValueError(
                "the root must be the selected, verified, committed seed"
            )
        positions = [
            node.verification_position
            for node in ordered
            if node.verification_position is not None
        ]
        if len(positions) != len(set(positions)):
            raise ValueError("verification positions must be unique")
        if sorted(positions) != list(range(len(positions))):
            raise ValueError("verification positions must be contiguous from zero")

        sibling_mass: dict[int, float] = {}
        accepted_children: dict[int, int] = {}
        for node in ordered:
            if node.parent_id is None:
                continue
            if node.role is not VerificationNodeRole.DRAFT:
                raise ValueError("only the root may have the seed role")
            parent = by_id.get(node.parent_id)
            if parent is None:
                raise ValueError(
                    f"node {node.node_id} has missing parent {node.parent_id}"
                )
            if node.depth != parent.depth + 1:
                raise ValueError("child depth must equal parent depth plus one")
            if node.selected and not parent.selected:
                raise ValueError("selected nodes must be ancestor-closed")
            if node.verified and not parent.verified:
                raise ValueError("verified nodes must be ancestor-closed")
            if node.accepted and not parent.accepted:
                raise ValueError("accepted nodes must be ancestor-closed")
            if (
                node.verified
                and node.verification_position is not None
                and parent.verification_position is not None
                and node.verification_position <= parent.verification_position
            ):
                raise ValueError(
                    "a verified child must follow its parent in verification order"
                )
            if node.conditional_probability is not None:
                sibling_mass[node.parent_id] = (
                    sibling_mass.get(node.parent_id, 0.0)
                    + node.conditional_probability
                )
            if node.accepted:
                accepted_children[node.parent_id] = (
                    accepted_children.get(node.parent_id, 0) + 1
                )
        if any(total > 1.0 + 1e-6 for total in sibling_mass.values()):
            raise ValueError("sibling conditional probabilities exceed one")
        if any(count > 1 for count in accepted_children.values()):
            raise ValueError("accepted draft nodes must form one path")

    @property
    def trace_id(self) -> str:
        return self.tree_id


@dataclass(frozen=True, slots=True)
class CycleTrace:

    run_id: str
    tree_id: str
    request_id: str
    iteration: int
    base_kv_length: int
    pending_seed_token_id: int
    verification_token_ids: tuple[int, ...]
    verifier_token_ids: tuple[int, ...]
    accepted_draft_node_ids: tuple[int, ...]
    committed_token_ids: tuple[int, ...]
    kv_length_after: int
    pre_snapshot_id: str | None = None
    post_snapshot_id: str | None = None
    schema_version: int = TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        for name in ("run_id", "tree_id", "request_id"):
            _require_nonempty(name, getattr(self, name))
        _require_nonnegative(
            iteration=self.iteration,
            base_kv_length=self.base_kv_length,
            pending_seed_token_id=self.pending_seed_token_id,
            kv_length_after=self.kv_length_after,
        )
        for name in (
            "verification_token_ids",
            "verifier_token_ids",
            "accepted_draft_node_ids",
            "committed_token_ids",
        ):
            value = getattr(self, name)
            _require_nonnegative_int_tuple(name, value)
        if not self.verification_token_ids:
            raise ValueError("verification_token_ids must be non-empty")
        if len(self.verification_token_ids) != len(self.verifier_token_ids):
            raise ValueError("verifier tokens must match verification positions")
        if len(set(self.accepted_draft_node_ids)) != len(
            self.accepted_draft_node_ids
        ):
            raise ValueError("accepted_draft_node_ids must be unique")
        if not self.committed_token_ids:
            raise ValueError("a cycle must commit at least the pending seed")
        if self.verification_token_ids[0] != self.pending_seed_token_id:
            raise ValueError("the first verification token must be the pending seed")
        expected_kv_length = self.base_kv_length + len(self.committed_token_ids)
        if self.kv_length_after != expected_kv_length:
            raise ValueError("KV length must advance by the committed token count")
        for name in ("pre_snapshot_id", "post_snapshot_id"):
            value = getattr(self, name)
            if value is not None:
                _require_nonempty(name, value)

    @property
    def trace_id(self) -> str:
        return stable_trace_id(
            "cycle", self.run_id, self.request_id, self.iteration
        )


@dataclass(frozen=True, slots=True)
class GenerationTrace:

    run_id: str
    request_id: str
    output_token_ids: tuple[int, ...]
    cycle_ids: tuple[str, ...]
    residual_token_ids: tuple[int, ...]
    target_forward_calls: int
    stopped_on_eos: bool = False
    stop_token_id: int | None = None
    schema_version: int = TRACE_SCHEMA_VERSION
    stopped_on_cycle_limit: bool = False

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_nonempty("run_id", self.run_id)
        _require_nonempty("request_id", self.request_id)
        _require_nonnegative_int_tuple(
            "output_token_ids", self.output_token_ids
        )
        _require_nonnegative_int_tuple(
            "residual_token_ids", self.residual_token_ids
        )
        _require_tuple("cycle_ids", self.cycle_ids)
        if not self.output_token_ids:
            raise ValueError("generation output must be non-empty")
        if any(
            not isinstance(cycle_id, str) or not cycle_id
            for cycle_id in self.cycle_ids
        ):
            raise ValueError("generation cycle IDs must be non-empty strings")
        if len(self.cycle_ids) != len(set(self.cycle_ids)):
            raise ValueError("generation cycle IDs must be unique")
        _require_nonnegative(
            target_forward_calls=self.target_forward_calls
        )
        if self.target_forward_calls == 0:
            raise ValueError("generation must contain a target forward")
        if not isinstance(self.stopped_on_eos, bool):
            raise TypeError("stopped_on_eos must be a boolean")
        if not isinstance(self.stopped_on_cycle_limit, bool):
            raise TypeError("stopped_on_cycle_limit must be a boolean")
        if self.stopped_on_eos and self.stopped_on_cycle_limit:
            raise ValueError("generation cannot stop on EOS and cycle limit")
        if self.stopped_on_cycle_limit and not self.cycle_ids:
            raise ValueError("cycle-limit termination requires a cycle")
        if self.stopped_on_eos:
            if (
                isinstance(self.stop_token_id, bool)
                or not isinstance(self.stop_token_id, int)
                or self.stop_token_id < 0
            ):
                raise ValueError(
                    "EOS termination requires a non-negative stop token"
                )
            if self.output_token_ids[-1] != self.stop_token_id:
                raise ValueError(
                    "EOS stop token must equal the last output token"
                )
        elif self.stop_token_id is not None:
            raise ValueError("stop token requires EOS termination")

    @property
    def trace_id(self) -> str:
        return stable_trace_id("generation", self.run_id, self.request_id)


@dataclass(frozen=True, slots=True)
class LayerRouteTrace:

    tree_id: str
    node_id: int
    layer_id: int
    expert_ids: tuple[int, ...]
    expert_weights: tuple[float, ...]
    schema_version: int = TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_nonempty("tree_id", self.tree_id)
        _require_nonnegative(node_id=self.node_id, layer_id=self.layer_id)
        _require_tuple("expert_ids", self.expert_ids)
        _require_tuple("expert_weights", self.expert_weights)
        if not self.expert_ids or len(self.expert_ids) != len(
            self.expert_weights
        ):
            raise ValueError("target route IDs and weights must be non-empty")
        _require_nonnegative_int_tuple("expert_ids", self.expert_ids)
        if len(set(self.expert_ids)) != len(self.expert_ids):
            raise ValueError("target route expert IDs must be unique")
        if any(
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0.0
            for weight in self.expert_weights
        ):
            raise ValueError("target route weights must be finite and non-negative")

    @property
    def trace_id(self) -> str:
        return stable_trace_id(
            "target-route", self.tree_id, self.node_id, self.layer_id
        )


@dataclass(frozen=True, slots=True)
class ExpertProbabilityTrace:

    expert_id: int
    probability: float
    schema_version: int = TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_nonnegative(expert_id=self.expert_id)
        _require_probability("probability", self.probability)


@dataclass(frozen=True, slots=True)
class TargetRouterTopKTrace:

    tree_id: str
    node_id: int
    layer_id: int
    experts: tuple[ExpertProbabilityTrace, ...]
    schema_version: int = TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_probability_route(
            self.schema_version,
            self.tree_id,
            self.node_id,
            self.layer_id,
            self.experts,
        )

    @property
    def trace_id(self) -> str:
        return stable_trace_id(
            "target-router-topk", self.tree_id, self.node_id, self.layer_id
        )


@dataclass(frozen=True, slots=True)
class PredictorRouteTrace:

    tree_id: str
    node_id: int
    layer_id: int
    experts: tuple[ExpertProbabilityTrace, ...]
    schema_version: int = TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_probability_route(
            self.schema_version,
            self.tree_id,
            self.node_id,
            self.layer_id,
            self.experts,
        )

    @property
    def trace_id(self) -> str:
        return stable_trace_id(
            "predictor-route", self.tree_id, self.node_id, self.layer_id
        )


def _validate_probability_route(
    schema_version: int,
    tree_id: str,
    node_id: int,
    layer_id: int,
    experts: tuple[ExpertProbabilityTrace, ...],
) -> None:
    _require_schema_version(schema_version)
    _require_nonempty("tree_id", tree_id)
    _require_nonnegative(node_id=node_id, layer_id=layer_id)
    _require_tuple("experts", experts)
    if not 1 <= len(experts) <= 32:
        raise ValueError("router Top-K must contain 1 to 32 experts")
    expert_ids = [expert.expert_id for expert in experts]
    if len(expert_ids) != len(set(expert_ids)):
        raise ValueError("router Top-K expert IDs must be unique")
    probabilities = [expert.probability for expert in experts]
    if probabilities != sorted(probabilities, reverse=True):
        raise ValueError(
            "router Top-K experts must be sorted by descending probability"
        )
    if sum(probabilities) > 1.0 + 1e-6:
        raise ValueError("router Top-K probability mass exceeds one")


class SnapshotPhase(StrEnum):
    BEFORE_VERIFY = "before_verify"
    AFTER_VERIFY = "after_verify"


@dataclass(frozen=True, slots=True)
class CacheStatsTrace:

    h2d_submissions: int
    resident_hits: int
    inflight_joins: int
    evictions: int
    pending_use_guards: int
    schema_version: int = TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_nonnegative(
            h2d_submissions=self.h2d_submissions,
            resident_hits=self.resident_hits,
            inflight_joins=self.inflight_joins,
            evictions=self.evictions,
            pending_use_guards=self.pending_use_guards,
        )


@dataclass(frozen=True, slots=True)
class CacheSlotTrace:
    slot_id: int
    expert_id: int | None
    generation: int
    active_leases: int
    last_used_sequence: int
    ready: bool
    pending_use_guards: int
    schema_version: int = TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_nonnegative(
            slot_id=self.slot_id,
            generation=self.generation,
            active_leases=self.active_leases,
            last_used_sequence=self.last_used_sequence,
            pending_use_guards=self.pending_use_guards,
        )
        _require_bool("ready", self.ready)
        if self.expert_id is not None:
            _require_nonnegative(expert_id=self.expert_id)
        if self.expert_id is None and (
            self.ready or self.active_leases or self.pending_use_guards
        ):
            raise ValueError("an empty slot cannot be ready or in use")


@dataclass(frozen=True, slots=True)
class PendingTransferTrace:
    transfer_id: str
    expert_id: int
    slot_id: int
    generation: int
    submission_sequence: int
    ready_event_id: str | None
    ready_event_completed: bool
    schema_version: int = TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_nonempty("transfer_id", self.transfer_id)
        if self.ready_event_id is not None:
            _require_nonempty("ready_event_id", self.ready_event_id)
        _require_nonnegative(
            expert_id=self.expert_id,
            slot_id=self.slot_id,
            generation=self.generation,
            submission_sequence=self.submission_sequence,
        )
        _require_bool(
            "ready_event_completed", self.ready_event_completed
        )


@dataclass(frozen=True, slots=True)
class LayerCacheSnapshotTrace:

    layer_id: int
    queue_id: str
    capacity: int
    submission_sequence: int
    slots: tuple[CacheSlotTrace, ...]
    pending_transfers: tuple[PendingTransferTrace, ...] = ()
    cache_id: str | None = None
    host_state_version: int | None = None
    lru_sequence: int | None = None
    stats: CacheStatsTrace | None = None
    schema_version: int = TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_nonempty("queue_id", self.queue_id)
        _require_nonnegative(
            layer_id=self.layer_id,
            capacity=self.capacity,
            submission_sequence=self.submission_sequence,
        )
        extended_state = (
            self.cache_id,
            self.host_state_version,
            self.lru_sequence,
            self.stats,
        )
        if any(value is None for value in extended_state) and not all(
            value is None for value in extended_state
        ):
            raise ValueError(
                "extended cache snapshot fields must be present together"
            )
        if self.cache_id is not None:
            _require_nonempty("cache_id", self.cache_id)
            _require_nonnegative(
                host_state_version=self.host_state_version,
                lru_sequence=self.lru_sequence,
            )
        if self.capacity == 0:
            raise ValueError("cache capacity must be positive")
        _require_tuple("slots", self.slots)
        _require_tuple("pending_transfers", self.pending_transfers)
        slots = tuple(sorted(self.slots, key=lambda slot: slot.slot_id))
        pending = tuple(
            sorted(
                self.pending_transfers,
                key=lambda item: item.submission_sequence,
            )
        )
        object.__setattr__(self, "slots", slots)
        object.__setattr__(self, "pending_transfers", pending)
        if len(slots) != self.capacity:
            raise ValueError("snapshot must contain every cache slot")
        if tuple(slot.slot_id for slot in slots) != tuple(range(self.capacity)):
            raise ValueError("cache slot IDs must cover range(capacity)")
        slots_by_id = {slot.slot_id: slot for slot in slots}
        if len(slots_by_id) != len(slots):
            raise ValueError("slot IDs must be unique within a layer")
        resident_expert_ids = [
            slot.expert_id
            for slot in slots
            if slot.expert_id is not None
        ]
        if len(resident_expert_ids) != len(set(resident_expert_ids)):
            raise ValueError(
                "resident experts must be unique within a layer"
            )
        if self.stats is not None:
            if any(
                slot.last_used_sequence > self.lru_sequence
                for slot in slots
            ):
                raise ValueError("slot last-use exceeds the LRU sequence")
            if self.stats.pending_use_guards != sum(
                slot.pending_use_guards for slot in slots
            ):
                raise ValueError(
                    "cache stats pending guards must match its slots"
                )
            if self.stats.h2d_submissions != self.submission_sequence:
                raise ValueError(
                    "cache H2D counter must match its submission sequence"
                )
        transfer_ids = [item.transfer_id for item in pending]
        sequences = [item.submission_sequence for item in pending]
        pending_generations = [
            (item.slot_id, item.generation) for item in pending
        ]
        if len(transfer_ids) != len(set(transfer_ids)):
            raise ValueError("pending transfer IDs must be unique")
        if len(sequences) != len(set(sequences)):
            raise ValueError("pending submission sequences must be unique")
        if len(pending_generations) != len(set(pending_generations)):
            raise ValueError(
                "pending slot generations must be unique"
            )
        if any(
            sequence >= self.submission_sequence
            for sequence in sequences
        ):
            raise ValueError(
                "pending sequence must precede the snapshot submission counter"
            )
        for transfer in pending:
            slot = slots_by_id.get(transfer.slot_id)
            if slot is None:
                raise ValueError("pending transfer references an unknown slot")
            if transfer.generation > slot.generation:
                raise ValueError(
                    "pending transfer generation exceeds its current slot"
                )
            if (
                transfer.generation == slot.generation
                and slot.expert_id != transfer.expert_id
            ):
                raise ValueError(
                    "current pending transfer does not match its slot"
                )
            if transfer.generation == slot.generation and slot.ready:
                raise ValueError("a current pending transfer cannot be ready")


@dataclass(frozen=True, slots=True)
class CacheSnapshotTrace:
    run_id: str
    tree_id: str
    cycle_id: str
    phase: SnapshotPhase
    runtime_state_version: int
    captured_at_ns: int
    layers: tuple[LayerCacheSnapshotTrace, ...]
    schema_version: int = TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        for name in ("run_id", "tree_id", "cycle_id"):
            _require_nonempty(name, getattr(self, name))
        _require_nonnegative(
            runtime_state_version=self.runtime_state_version,
            captured_at_ns=self.captured_at_ns,
        )
        if not isinstance(self.phase, SnapshotPhase):
            object.__setattr__(self, "phase", SnapshotPhase(self.phase))
        _require_tuple("layers", self.layers)
        layers = tuple(sorted(self.layers, key=lambda item: item.layer_id))
        object.__setattr__(self, "layers", layers)
        layer_ids = [layer.layer_id for layer in layers]
        if len(layer_ids) != len(set(layer_ids)):
            raise ValueError("snapshot layer IDs must be unique")
        layer_versions = [
            layer.host_state_version
            for layer in layers
            if layer.host_state_version is not None
        ]
        if layer_versions and len(layer_versions) == len(layers) and (
            self.runtime_state_version != sum(layer_versions)
        ):
            raise ValueError(
                "snapshot runtime state version must sum layer versions"
            )

    @property
    def trace_id(self) -> str:
        return stable_trace_id(
            "cache-snapshot", self.cycle_id, self.phase.value
        )


class CudaEventRole(StrEnum):
    H2D_START = "h2d_start"
    H2D_END = "h2d_end"
    CONSUME_START = "consume_start"
    CONSUME_END = "consume_end"
    CYCLE_START = "cycle_start"
    CYCLE_END = "cycle_end"
    LAYER_START = "layer_start"
    LAYER_END = "layer_end"


@dataclass(frozen=True, slots=True)
class CudaEventTrace:
    run_id: str
    sequence: int
    stream_id: str
    role: CudaEventRole
    elapsed_from_origin_ns: int
    schema_version: int = TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_nonempty("run_id", self.run_id)
        _require_nonempty("stream_id", self.stream_id)
        _require_nonnegative(
            sequence=self.sequence,
            elapsed_from_origin_ns=self.elapsed_from_origin_ns,
        )
        if not isinstance(self.role, CudaEventRole):
            object.__setattr__(self, "role", CudaEventRole(self.role))

    @property
    def trace_id(self) -> str:
        return stable_trace_id("cuda-event", self.run_id, self.sequence)


class RuntimeEventKind(StrEnum):
    RESIDENT_HIT = "resident_hit"
    INFLIGHT_JOIN = "inflight_join"
    H2D_SUBMIT = "h2d_submit"
    EVICT = "evict"
    CONSUME_WAIT = "consume_wait"
    RELEASE = "release"


@dataclass(frozen=True, slots=True)
class RuntimeEventTrace:
    run_id: str
    sequence: int
    kind: RuntimeEventKind
    host_time_ns: int
    layer_id: int
    expert_id: int
    slot_id: int
    generation: int
    cache_id: str | None = None
    queue_id: str | None = None
    submission_sequence: int | None = None
    tree_id: str | None = None
    cycle_id: str | None = None
    transfer_id: str | None = None
    victim_expert_id: int | None = None
    stream_ids: tuple[int, ...] = ()
    cuda_event_ids: tuple[str, ...] = ()
    schema_version: int = TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_nonempty("run_id", self.run_id)
        if (self.cache_id is None) != (self.queue_id is None):
            raise ValueError("cache_id and queue_id must be present together")
        if self.cache_id is not None:
            _require_nonempty("cache_id", self.cache_id)
            _require_nonempty("queue_id", self.queue_id)
        _require_nonnegative(
            sequence=self.sequence,
            host_time_ns=self.host_time_ns,
            layer_id=self.layer_id,
            expert_id=self.expert_id,
            slot_id=self.slot_id,
            generation=self.generation,
        )
        if not isinstance(self.kind, RuntimeEventKind):
            object.__setattr__(self, "kind", RuntimeEventKind(self.kind))
        if (self.tree_id is None) != (self.cycle_id is None):
            raise ValueError("tree_id and cycle_id must be present together")
        for name in ("tree_id", "cycle_id", "transfer_id"):
            value = getattr(self, name)
            if value is not None:
                _require_nonempty(name, value)
        if self.victim_expert_id is not None:
            _require_nonnegative(victim_expert_id=self.victim_expert_id)
        if self.submission_sequence is not None:
            _require_nonnegative(
                submission_sequence=self.submission_sequence
            )
        _require_tuple("stream_ids", self.stream_ids)
        if len(self.stream_ids) != len(set(self.stream_ids)):
            raise ValueError("stream IDs must be unique")
        _require_nonnegative_int_tuple("stream_ids", self.stream_ids)
        _require_tuple("cuda_event_ids", self.cuda_event_ids)
        if any(not event_id for event_id in self.cuda_event_ids):
            raise ValueError("CUDA event IDs must be non-empty")
        if self.kind in (
            RuntimeEventKind.H2D_SUBMIT,
            RuntimeEventKind.INFLIGHT_JOIN,
        ):
            if self.transfer_id is None:
                raise ValueError(
                    "H2D submit and in-flight join require a logical transfer"
                )
            if self.submission_sequence is not None:
                expected_transfer_id = stable_trace_id(
                    "h2d",
                    self.run_id,
                    self.layer_id,
                    self.submission_sequence,
                )
                if self.transfer_id != expected_transfer_id:
                    raise ValueError(
                        "runtime event has an invalid transfer ID"
                    )
        if self.kind is RuntimeEventKind.EVICT:
            if self.victim_expert_id is None:
                raise ValueError("eviction requires a victim expert")
        elif self.victim_expert_id is not None:
            raise ValueError("only eviction may name a victim expert")

    @property
    def trace_id(self) -> str:
        return stable_trace_id("runtime-event", self.run_id, self.sequence)


@dataclass(frozen=True, slots=True)
class H2DTrace:
    run_id: str
    submission_sequence: int
    layer_id: int
    expert_id: int
    slot_id: int
    generation: int
    bytes: int
    host_submit_ns: int
    start_event_id: str
    end_event_id: str
    tree_id: str | None = None
    cycle_id: str | None = None
    schema_version: int = TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_nonempty("run_id", self.run_id)
        _require_nonempty("start_event_id", self.start_event_id)
        _require_nonempty("end_event_id", self.end_event_id)
        _require_nonnegative(
            submission_sequence=self.submission_sequence,
            layer_id=self.layer_id,
            expert_id=self.expert_id,
            slot_id=self.slot_id,
            generation=self.generation,
            bytes=self.bytes,
            host_submit_ns=self.host_submit_ns,
        )
        if self.bytes == 0:
            raise ValueError("H2D byte count must be positive")
        if (self.tree_id is None) != (self.cycle_id is None):
            raise ValueError("tree_id and cycle_id must be present together")
        for name in ("tree_id", "cycle_id"):
            value = getattr(self, name)
            if value is not None:
                _require_nonempty(name, value)

    @property
    def trace_id(self) -> str:
        return stable_trace_id(
            "h2d", self.run_id, self.layer_id, self.submission_sequence
        )


@dataclass(frozen=True, slots=True)
class TraceBundle:

    manifest: WorkloadManifestTrace
    trees: tuple[DraftTreeTrace, ...]
    cycles: tuple[CycleTrace, ...]
    generations: tuple[GenerationTrace, ...] = ()
    target_routes: tuple[LayerRouteTrace, ...] = ()
    target_router_topk: tuple[TargetRouterTopKTrace, ...] = ()
    predictor_routes: tuple[PredictorRouteTrace, ...] = ()
    snapshots: tuple[CacheSnapshotTrace, ...] = ()
    cuda_events: tuple[CudaEventTrace, ...] = ()
    h2d_transfers: tuple[H2DTrace, ...] = ()
    runtime_events: tuple[RuntimeEventTrace, ...] = ()
    schema_version: int = TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        collection_keys = {
            "trees": lambda item: (
                item.request_id,
                item.iteration,
                item.tree_id,
            ),
            "cycles": lambda item: item.trace_id,
            "generations": lambda item: item.trace_id,
            "target_routes": lambda item: item.trace_id,
            "target_router_topk": lambda item: item.trace_id,
            "predictor_routes": lambda item: item.trace_id,
            "snapshots": lambda item: item.trace_id,
            "cuda_events": lambda item: item.sequence,
            "h2d_transfers": lambda item: item.trace_id,
            "runtime_events": lambda item: item.sequence,
        }
        for name, key in collection_keys.items():
            value = getattr(self, name)
            _require_tuple(name, value)
            object.__setattr__(self, name, tuple(sorted(value, key=key)))
        self._validate_references()

    @property
    def run_id(self) -> str:
        return self.manifest.run_id

    def _validate_references(self) -> None:
        trees = {tree.tree_id: tree for tree in self.trees}
        if len(trees) != len(self.trees):
            raise ValueError("tree IDs must be unique")
        cycles = {cycle.trace_id: cycle for cycle in self.cycles}
        if len(cycles) != len(self.cycles):
            raise ValueError("cycle IDs must be unique")
        if self.manifest.draft_length == 0 and (self.trees or self.cycles):
            raise ValueError(
                "target-only trace cannot contain speculative trees or cycles"
            )
        if {cycle.tree_id for cycle in self.cycles} != set(trees):
            raise ValueError("every tree must have exactly one cycle")

        for tree in self.trees:
            if tree.request_id != self.manifest.request_id:
                raise ValueError("tree request_id must match the manifest")
        nodes = {
            (tree.tree_id, node.node_id): node
            for tree in self.trees
            for node in tree.nodes
        }

        early_stop_parents: set[tuple[str, int]] = set()
        cycles_by_id = {cycle.trace_id: cycle for cycle in self.cycles}
        for generation in self.generations:
            if (
                not generation.stopped_on_eos
                or not generation.cycle_ids
            ):
                continue
            final_cycle = cycles_by_id.get(generation.cycle_ids[-1])
            if final_cycle is None:
                continue
            final_tree = trees.get(final_cycle.tree_id)
            if final_tree is None:
                continue
            if final_cycle.accepted_draft_node_ids:
                final_node_id = final_cycle.accepted_draft_node_ids[-1]
            else:
                final_node_id = next(
                    node.node_id
                    for node in final_tree.nodes
                    if node.role is VerificationNodeRole.SEED
                )
            final_node = nodes[(final_cycle.tree_id, final_node_id)]
            if generation.output_token_ids[-1] == final_node.token_id:
                early_stop_parents.add((final_cycle.tree_id, final_node_id))

        cycles_by_tree = {cycle.tree_id: cycle for cycle in self.cycles}
        if len(cycles_by_tree) != len(self.cycles):
            raise ValueError("each tree may have only one cycle")
        for tree_id, cycle in cycles_by_tree.items():
            tree = trees.get(tree_id)
            if tree is None:
                raise ValueError("cycle references an unknown tree")
            if (
                cycle.run_id != self.run_id
                or cycle.request_id != tree.request_id
                or cycle.iteration != tree.iteration
            ):
                raise ValueError("cycle identity does not match its tree")
            verified = tuple(
                sorted(
                    (node for node in tree.nodes if node.verified),
                    key=lambda node: node.verification_position,
                )
            )
            if tuple(node.token_id for node in verified) != (
                cycle.verification_token_ids
            ):
                raise ValueError(
                    "cycle verification tokens do not match its tree"
                )
            root = next(
                node
                for node in tree.nodes
                if node.role is VerificationNodeRole.SEED
            )
            if root.token_id != cycle.pending_seed_token_id:
                raise ValueError("cycle pending seed does not match its tree")
            accepted_drafts = tuple(
                node.node_id
                for node in sorted(
                    (
                        node
                        for node in tree.nodes
                        if node.role is VerificationNodeRole.DRAFT
                        and node.accepted
                    ),
                    key=lambda node: node.depth,
                )
            )
            if cycle.accepted_draft_node_ids != accepted_drafts:
                raise ValueError(
                    "cycle accepted draft nodes do not match its tree"
                )
            committed = (root.token_id,) + tuple(
                nodes[(tree.tree_id, node_id)].token_id
                for node_id in accepted_drafts
            )
            if cycle.committed_token_ids != committed:
                raise ValueError("cycle committed tokens do not match its tree")
            by_id = {node.node_id: node for node in tree.nodes}
            for node in tree.nodes:
                if node.role is VerificationNodeRole.SEED or not node.verified:
                    continue
                parent = by_id[node.parent_id]
                parent_position = parent.verification_position
                if parent_position is None:
                    raise ValueError(
                        "verified nodes must be ancestor-closed"
                    )
                predicted = cycle.verifier_token_ids[
                    parent_position
                ]
                should_accept = parent.accepted and predicted == node.token_id
                terminal_truncation = (
                    should_accept
                    and not node.accepted
                    and (tree.tree_id, parent.node_id) in early_stop_parents
                )
                if node.accepted != should_accept and not terminal_truncation:
                    raise ValueError(
                        "draft acceptance does not match parent verifier output"
                    )

        generation_ids = [
            generation.trace_id for generation in self.generations
        ]
        if len(generation_ids) != len(set(generation_ids)):
            raise ValueError("generation IDs must be unique")
        for generation in self.generations:
            if (
                generation.run_id != self.run_id
                or generation.request_id != self.manifest.request_id
            ):
                raise ValueError(
                    "generation identity does not match the manifest"
                )
            output_length = len(generation.output_token_ids)
            if (
                generation.stopped_on_cycle_limit
                and generation.residual_token_ids
            ):
                raise ValueError(
                    "cycle-limit termination cannot contain residual tokens"
                )
            if not 1 <= output_length <= self.manifest.max_new_tokens:
                raise ValueError(
                    "generation output length exceeds the manifest budget"
                )
            if (
                not generation.stopped_on_eos
                and not generation.stopped_on_cycle_limit
                and output_length != self.manifest.max_new_tokens
            ):
                raise ValueError(
                    "generation shorter than its budget requires explicit "
                    "EOS or cycle limit"
                )
            if (
                generation.stopped_on_cycle_limit
                and output_length == self.manifest.max_new_tokens
            ):
                raise ValueError(
                    "cycle-limit termination requires output below budget"
                )
            cycle_by_id = {
                cycle.trace_id: cycle for cycle in self.cycles
            }
            unknown_cycle_ids = set(generation.cycle_ids) - set(cycle_by_id)
            if unknown_cycle_ids:
                raise ValueError(
                    "generation references an unknown cycle"
                )
            if set(generation.cycle_ids) != set(cycle_by_id):
                raise ValueError(
                    "generation must reference every cycle exactly once"
                )
            ordered_cycles = tuple(
                cycle_by_id[cycle_id]
                for cycle_id in generation.cycle_ids
            )
            if tuple(cycle.iteration for cycle in ordered_cycles) != tuple(
                range(len(ordered_cycles))
            ):
                raise ValueError(
                    "generation cycle order must follow contiguous iterations"
                )
            reconstructed = tuple(
                token_id
                for cycle in ordered_cycles
                for token_id in cycle.committed_token_ids
            ) + generation.residual_token_ids
            if reconstructed != generation.output_token_ids:
                raise ValueError(
                    "cycles and residual tokens do not reconstruct generation"
                )
            if self.manifest.draft_length == 0:
                if generation.cycle_ids:
                    raise ValueError(
                        "target-only generation cannot contain speculative cycles"
                    )
                expected_target_forwards = len(
                    generation.output_token_ids
                )
            else:
                if len(generation.residual_token_ids) > 1:
                    raise ValueError(
                        "speculative generation has more than one residual token"
                    )
                if generation.residual_token_ids and ordered_cycles:
                    final_cycle = ordered_cycles[-1]
                    final_tree = trees[final_cycle.tree_id]
                    final_node_id = (
                        final_cycle.accepted_draft_node_ids[-1]
                        if final_cycle.accepted_draft_node_ids
                        else next(
                            node.node_id
                            for node in final_tree.nodes
                            if node.role is VerificationNodeRole.SEED
                        )
                    )
                    final_node = nodes[(final_tree.tree_id, final_node_id)]
                    final_position = final_node.verification_position
                    if final_position is None:
                        raise ValueError(
                            "final accepted node has no verification position"
                        )
                    pending_seed = final_cycle.verifier_token_ids[
                        final_position
                    ]
                    if generation.residual_token_ids != (pending_seed,):
                        raise ValueError(
                            "generation residual does not match the final "
                            "pending seed"
                        )
                expected_target_forwards = 1 + len(ordered_cycles)
            if generation.target_forward_calls != expected_target_forwards:
                raise ValueError(
                    "generation target forward count is inconsistent"
                )

        target_ids: set[str] = set()
        for route in self.target_routes:
            if route.trace_id in target_ids:
                raise ValueError("target route IDs must be unique")
            target_ids.add(route.trace_id)
            node = nodes.get((route.tree_id, route.node_id))
            if node is None:
                raise ValueError("target route references an unknown node")
            if not node.verified:
                raise ValueError("target route references an unverified node")
            if route.layer_id >= self.manifest.num_moe_layers:
                raise ValueError("target route layer exceeds the model layout")
            if len(route.expert_ids) != self.manifest.experts_per_token:
                raise ValueError("target route Top-K differs from the model layout")
            if any(
                expert_id >= self.manifest.num_experts_per_layer
                for expert_id in route.expert_ids
            ):
                raise ValueError("target route expert exceeds the model layout")
        target_topk_ids = [
            route.trace_id for route in self.target_router_topk
        ]
        if len(target_topk_ids) != len(set(target_topk_ids)):
            raise ValueError("target router Top-K IDs must be unique")
        for route in self.target_router_topk:
            node = nodes.get((route.tree_id, route.node_id))
            if node is None or not node.verified:
                raise ValueError(
                    "target router Top-K references an unverified node"
                )
            self._validate_probability_route_layout(route.layer_id, route.experts)
        predictor_ids = [route.trace_id for route in self.predictor_routes]
        if len(predictor_ids) != len(set(predictor_ids)):
            raise ValueError("predictor route IDs must be unique")
        if any(
            (route.tree_id, route.node_id) not in nodes
            for route in self.predictor_routes
        ):
            raise ValueError("predictor route references an unknown node")
        for route in self.predictor_routes:
            self._validate_probability_route_layout(route.layer_id, route.experts)

        snapshots = {snapshot.trace_id: snapshot for snapshot in self.snapshots}
        if len(snapshots) != len(self.snapshots):
            raise ValueError("snapshot IDs must be unique")
        for snapshot in self.snapshots:
            self._validate_scope(
                snapshot.run_id,
                snapshot.tree_id,
                snapshot.cycle_id,
                trees,
                cycles,
            )
            queue_ids = [layer.queue_id for layer in snapshot.layers]
            if len(queue_ids) != len(set(queue_ids)):
                raise ValueError("snapshot queue IDs must be unique")
            cache_ids = [
                layer.cache_id
                for layer in snapshot.layers
                if layer.cache_id is not None
            ]
            if len(cache_ids) != len(set(cache_ids)):
                raise ValueError("snapshot cache IDs must be unique")
            for layer in snapshot.layers:
                if layer.layer_id >= self.manifest.num_moe_layers:
                    raise ValueError("snapshot layer exceeds the model layout")
                if layer.capacity != self.manifest.slots_per_layer:
                    raise ValueError(
                        "snapshot capacity differs from the manifest"
                    )
                if any(
                    slot.expert_id is not None
                    and slot.expert_id
                    >= self.manifest.num_experts_per_layer
                    for slot in layer.slots
                ):
                    raise ValueError(
                        "snapshot expert exceeds the model layout"
                    )
        for cycle in self.cycles:
            for snapshot_id, expected_phase in (
                (cycle.pre_snapshot_id, SnapshotPhase.BEFORE_VERIFY),
                (cycle.post_snapshot_id, SnapshotPhase.AFTER_VERIFY),
            ):
                if snapshot_id is None:
                    continue
                snapshot = snapshots.get(snapshot_id)
                if snapshot is None:
                    raise ValueError("cycle references an unknown snapshot")
                if (
                    snapshot.run_id != cycle.run_id
                    or snapshot.tree_id != cycle.tree_id
                    or snapshot.cycle_id != cycle.trace_id
                    or snapshot.phase is not expected_phase
                ):
                    raise ValueError(
                        "cycle snapshot has the wrong scope or phase"
                    )

        cuda_events = {event.trace_id: event for event in self.cuda_events}
        if len(cuda_events) != len(self.cuda_events):
            raise ValueError("CUDA event IDs must be unique")
        if any(event.run_id != self.run_id for event in self.cuda_events):
            raise ValueError("CUDA event belongs to a different run")
        event_sequences = [event.sequence for event in self.cuda_events]
        if len(event_sequences) != len(set(event_sequences)):
            raise ValueError("CUDA event sequences must be unique")
        queue_by_layer: dict[int, str] = {}
        cache_by_layer: dict[int, str] = {}
        for snapshot in self.snapshots:
            for layer in snapshot.layers:
                previous = queue_by_layer.setdefault(
                    layer.layer_id, layer.queue_id
                )
                if previous != layer.queue_id:
                    raise ValueError("layer queue_id changes across snapshots")
                if layer.cache_id is not None:
                    previous_cache = cache_by_layer.setdefault(
                        layer.layer_id, layer.cache_id
                    )
                    if previous_cache != layer.cache_id:
                        raise ValueError(
                            "layer cache_id changes across snapshots"
                        )
        transfers = {
            transfer.trace_id: transfer for transfer in self.h2d_transfers
        }
        if len(transfers) != len(self.h2d_transfers):
            raise ValueError("H2D transfer IDs must be unique")
        logical_submits: dict[str, RuntimeEventTrace] = {}
        logical_submits_by_layer: dict[
            tuple[int, str | None], list[tuple[RuntimeEventTrace, int]]
        ] = {}
        for event in self.runtime_events:
            if event.kind is not RuntimeEventKind.H2D_SUBMIT:
                continue
            if event.transfer_id is None:
                raise ValueError("H2D submit has no logical transfer ID")
            previous_submit = logical_submits.setdefault(
                event.transfer_id, event
            )
            if previous_submit is not event:
                raise ValueError("logical H2D transfer is submitted twice")
            resolved_submission_sequence = event.submission_sequence
            if resolved_submission_sequence is None:
                timed_transfer = transfers.get(event.transfer_id)
                if timed_transfer is None:
                    raise ValueError("logical H2D submit has no sequence")
                resolved_submission_sequence = (
                    timed_transfer.submission_sequence
                )
            logical_submits_by_layer.setdefault(
                (event.layer_id, event.cache_id), []
            ).append((event, resolved_submission_sequence))
        for layer_submits in logical_submits_by_layer.values():
            sequences = [sequence for _, sequence in layer_submits]
            if any(
                current != previous + 1
                for previous, current in zip(sequences, sequences[1:])
            ):
                raise ValueError(
                    "logical H2D submissions violate layer order"
                )
            generation_by_slot: dict[int, int] = {}
            for event, _ in layer_submits:
                previous_generation = generation_by_slot.get(event.slot_id)
                if (
                    previous_generation is not None
                    and event.generation <= previous_generation
                ):
                    raise ValueError(
                        "logical H2D slot generation must increase"
                    )
                generation_by_slot[event.slot_id] = event.generation
        event_owners: dict[str, str] = {}
        transfer_events_by_layer: dict[
            int,
            list[tuple[H2DTrace, CudaEventTrace, CudaEventTrace]],
        ] = {}
        transfer_stream_by_layer: dict[int, str] = {}
        transfer_layer_by_stream: dict[str, int] = {}
        for transfer in self.h2d_transfers:
            self._validate_optional_scope(
                transfer.run_id,
                transfer.tree_id,
                transfer.cycle_id,
                trees,
                cycles,
            )
            if (
                transfer.layer_id >= self.manifest.num_moe_layers
                or transfer.expert_id >= self.manifest.num_experts_per_layer
                or transfer.slot_id >= self.manifest.slots_per_layer
            ):
                raise ValueError("H2D transfer exceeds the model/cache layout")
            if (
                transfer.start_event_id not in cuda_events
                or transfer.end_event_id not in cuda_events
            ):
                raise ValueError("H2D references an unknown CUDA event")
            start = cuda_events[transfer.start_event_id]
            end = cuda_events[transfer.end_event_id]
            if (
                start.run_id != transfer.run_id
                or end.run_id != transfer.run_id
                or start.role is not CudaEventRole.H2D_START
                or end.role is not CudaEventRole.H2D_END
                or start.stream_id != end.stream_id
                or start.sequence >= end.sequence
            ):
                raise ValueError("H2D CUDA events have the wrong identity")
            for event_id in (transfer.start_event_id, transfer.end_event_id):
                owner = event_owners.setdefault(event_id, transfer.trace_id)
                if owner != transfer.trace_id:
                    raise ValueError(
                        "a CUDA event must belong to one unique H2D transfer"
                    )
            observed_stream = transfer_stream_by_layer.setdefault(
                transfer.layer_id,
                start.stream_id,
            )
            if observed_stream != start.stream_id:
                raise ValueError("H2D CUDA stream changes within one layer")
            observed_layer = transfer_layer_by_stream.setdefault(
                start.stream_id,
                transfer.layer_id,
            )
            if observed_layer != transfer.layer_id:
                raise ValueError(
                    "one H2D CUDA stream cannot serve multiple layers"
                )
            expected_queue = queue_by_layer.get(transfer.layer_id)
            if expected_queue is not None and start.stream_id != expected_queue:
                raise ValueError("H2D CUDA stream differs from the layer queue")
            if start.elapsed_from_origin_ns > end.elapsed_from_origin_ns:
                raise ValueError("H2D end event must follow its start event")
            transfer_events_by_layer.setdefault(transfer.layer_id, []).append(
                (transfer, start, end)
            )
        for layer_transfers in transfer_events_by_layer.values():
            ordered_transfers = sorted(
                layer_transfers,
                key=lambda item: item[0].submission_sequence,
            )
            generation_by_slot: dict[int, int] = {}
            for transfer, _, _ in ordered_transfers:
                previous_generation = generation_by_slot.get(
                    transfer.slot_id
                )
                if (
                    previous_generation is not None
                    and transfer.generation <= previous_generation
                ):
                    raise ValueError(
                        "H2D slot generation must increase with submission order"
                    )
                generation_by_slot[transfer.slot_id] = transfer.generation
            for previous, current in zip(
                ordered_transfers,
                ordered_transfers[1:],
            ):
                previous_transfer, _, previous_end = previous
                current_transfer, current_start, _ = current
                if (
                    previous_transfer.host_submit_ns
                    > current_transfer.host_submit_ns
                    or previous_end.sequence >= current_start.sequence
                    or previous_end.elapsed_from_origin_ns
                    > current_start.elapsed_from_origin_ns
                ):
                    raise ValueError(
                        "H2D CUDA events violate layer submission order"
                    )
        initial_pending: dict[
            str,
            tuple[
                CacheSnapshotTrace,
                LayerCacheSnapshotTrace,
                PendingTransferTrace,
            ],
        ] = {}
        for snapshot in self.snapshots:
            if snapshot.phase is not SnapshotPhase.BEFORE_VERIFY:
                continue
            for layer in snapshot.layers:
                for pending in layer.pending_transfers:
                    previous = initial_pending.get(pending.transfer_id)
                    if (
                        previous is None
                        or snapshot.captured_at_ns
                        < previous[0].captured_at_ns
                    ):
                        initial_pending[pending.transfer_id] = (
                            snapshot,
                            layer,
                            pending,
                        )
        pending_identity_by_id: dict[str, tuple[object, ...]] = {}
        for snapshot in self.snapshots:
            for layer in snapshot.layers:
                for pending in layer.pending_transfers:
                    identity = (
                        snapshot.run_id,
                        layer.layer_id,
                        layer.cache_id,
                        layer.queue_id,
                        pending.expert_id,
                        pending.slot_id,
                        pending.generation,
                        pending.submission_sequence,
                    )
                    previous_identity = pending_identity_by_id.setdefault(
                        pending.transfer_id, identity
                    )
                    if previous_identity != identity:
                        raise ValueError(
                            "pending transfer changes identity across snapshots"
                        )
                    expected_transfer_id = stable_trace_id(
                        "h2d",
                        snapshot.run_id,
                        layer.layer_id,
                        pending.submission_sequence,
                    )
                    if pending.transfer_id != expected_transfer_id:
                        raise ValueError(
                            "pending transfer has an invalid transfer ID"
                        )
                    transfer = transfers.get(pending.transfer_id)
                    if transfer is None:
                        if pending.ready_event_id is not None:
                            raise ValueError(
                                "logical pending transfer cannot reference "
                                "a CUDA event"
                            )
                        logical_submit = logical_submits.get(pending.transfer_id)
                        if logical_submit is not None and (
                            logical_submit.host_time_ns
                            > snapshot.captured_at_ns
                            or logical_submit.run_id != snapshot.run_id
                            or logical_submit.layer_id != layer.layer_id
                            or logical_submit.expert_id != pending.expert_id
                            or logical_submit.slot_id != pending.slot_id
                            or logical_submit.generation != pending.generation
                            or logical_submit.submission_sequence
                            != pending.submission_sequence
                        ):
                            raise ValueError(
                                "pending transfer does not match its logical "
                                "H2D submit"
                            )
                        continue
                    if (
                        pending.ready_event_id is not None
                        and pending.ready_event_id not in cuda_events
                    ):
                        raise ValueError(
                            "timed pending transfer references an unknown "
                            "CUDA event"
                        )
                    if (
                        transfer.host_submit_ns > snapshot.captured_at_ns
                        or transfer.run_id != snapshot.run_id
                        or transfer.layer_id != layer.layer_id
                        or transfer.expert_id != pending.expert_id
                        or transfer.slot_id != pending.slot_id
                        or transfer.generation != pending.generation
                        or transfer.submission_sequence
                        != pending.submission_sequence
                        or (
                            pending.ready_event_id is not None
                            and transfer.end_event_id
                            != pending.ready_event_id
                        )
                    ):
                        raise ValueError(
                            "pending transfer does not match its H2D record"
                        )
        event_ids: set[str] = set()
        previous_host_time_ns: int | None = None
        for event in self.runtime_events:
            if event.trace_id in event_ids:
                raise ValueError("runtime event IDs must be unique")
            event_ids.add(event.trace_id)
            if (
                previous_host_time_ns is not None
                and event.host_time_ns < previous_host_time_ns
            ):
                raise ValueError(
                    "runtime event host time violates global sequence"
                )
            previous_host_time_ns = event.host_time_ns
            self._validate_optional_scope(
                event.run_id,
                event.tree_id,
                event.cycle_id,
                trees,
                cycles,
            )
            if (
                event.layer_id >= self.manifest.num_moe_layers
                or event.expert_id >= self.manifest.num_experts_per_layer
                or event.slot_id >= self.manifest.slots_per_layer
                or (
                    event.victim_expert_id is not None
                    and event.victim_expert_id
                    >= self.manifest.num_experts_per_layer
                )
            ):
                raise ValueError(
                    "runtime event exceeds the model/cache layout"
                )
            expected_queue = queue_by_layer.get(event.layer_id)
            if expected_queue is not None and (
                event.queue_id is not None
                and event.queue_id != expected_queue
            ):
                raise ValueError(
                    "runtime event queue differs from its snapshot layer"
                )
            expected_cache = cache_by_layer.get(event.layer_id)
            if expected_cache is not None and (
                event.cache_id is not None
                and event.cache_id != expected_cache
            ):
                raise ValueError(
                    "runtime event cache differs from its snapshot layer"
                )
            if event.transfer_id is not None:
                transfer = transfers.get(event.transfer_id)
                if transfer is None:
                    if event.submission_sequence is None:
                        raise ValueError(
                            "logical runtime transfer has no submission sequence"
                        )
                    expected_transfer_id = stable_trace_id(
                        "h2d",
                        event.run_id,
                        event.layer_id,
                        event.submission_sequence,
                    )
                    if event.transfer_id != expected_transfer_id:
                        raise ValueError(
                            "runtime event has an invalid transfer ID"
                        )
                elif (
                    transfer.run_id != event.run_id
                    or transfer.layer_id != event.layer_id
                    or transfer.expert_id != event.expert_id
                    or transfer.slot_id != event.slot_id
                    or transfer.generation != event.generation
                    or (
                        event.kind is RuntimeEventKind.H2D_SUBMIT
                        and (
                            transfer.tree_id != event.tree_id
                            or transfer.cycle_id != event.cycle_id
                        )
                    )
                    or (
                        event.submission_sequence is not None
                        and transfer.submission_sequence
                        != event.submission_sequence
                    )
                ):
                    raise ValueError(
                        "runtime event does not match its H2D record"
                    )
                if event.kind is RuntimeEventKind.INFLIGHT_JOIN:
                    submit = logical_submits.get(event.transfer_id)
                    if submit is not None:
                        if (
                            submit.run_id != event.run_id
                            or submit.layer_id != event.layer_id
                            or submit.cache_id != event.cache_id
                            or submit.queue_id != event.queue_id
                            or submit.expert_id != event.expert_id
                            or submit.slot_id != event.slot_id
                            or submit.generation != event.generation
                            or submit.submission_sequence
                            != event.submission_sequence
                        ):
                            raise ValueError(
                                "in-flight join does not match its logical "
                                "H2D submit"
                            )
                        if submit.sequence >= event.sequence:
                            raise ValueError(
                                "in-flight join must follow its logical "
                                "H2D submit"
                            )
                    elif transfer is None and event.cycle_id is not None:
                        baseline = initial_pending.get(event.transfer_id)
                        if baseline is None:
                            raise ValueError(
                                "in-flight join references an unknown "
                                "logical H2D"
                            )
                        snapshot, layer, pending = baseline
                        if (
                            snapshot.captured_at_ns > event.host_time_ns
                            or snapshot.run_id != event.run_id
                            or layer.layer_id != event.layer_id
                            or (
                                layer.cache_id is not None
                                and event.cache_id is not None
                                and layer.cache_id != event.cache_id
                            )
                            or (
                                event.queue_id is not None
                                and layer.queue_id != event.queue_id
                            )
                            or pending.expert_id != event.expert_id
                            or pending.slot_id != event.slot_id
                            or pending.generation != event.generation
                            or pending.submission_sequence
                            != event.submission_sequence
                        ):
                            raise ValueError(
                                "in-flight join does not match its initial "
                                "pending transfer"
                            )
            if any(
                event_id not in cuda_events
                for event_id in event.cuda_event_ids
            ):
                raise ValueError("runtime event references an unknown CUDA event")

    def _validate_scope(
        self,
        run_id: str,
        tree_id: str,
        cycle_id: str,
        trees: Mapping[str, DraftTreeTrace],
        cycles: Mapping[str, CycleTrace],
    ) -> None:
        if run_id != self.run_id or tree_id not in trees or cycle_id not in cycles:
            raise ValueError("trace record has an unknown run/tree/cycle scope")
        if cycles[cycle_id].tree_id != tree_id:
            raise ValueError("cycle does not belong to the referenced tree")

    def _validate_optional_scope(
        self,
        run_id: str,
        tree_id: str | None,
        cycle_id: str | None,
        trees: Mapping[str, DraftTreeTrace],
        cycles: Mapping[str, CycleTrace],
    ) -> None:
        if run_id != self.run_id:
            raise ValueError("trace record belongs to a different run")
        if tree_id is not None and cycle_id is not None:
            self._validate_scope(run_id, tree_id, cycle_id, trees, cycles)

    def _validate_probability_route_layout(
        self,
        layer_id: int,
        experts: tuple[ExpertProbabilityTrace, ...],
    ) -> None:
        if layer_id >= self.manifest.num_moe_layers:
            raise ValueError("router Top-K layer exceeds the model layout")
        if any(
            expert.expert_id >= self.manifest.num_experts_per_layer
            for expert in experts
        ):
            raise ValueError("router Top-K expert exceeds the model layout")

    def to_dict(self) -> dict[str, Any]:
        return _encode_record(self)

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_jsonl(self) -> str:
        records: tuple[object, ...] = (
            self.manifest,
            *self.trees,
            *self.cycles,
            *self.generations,
            *self.target_routes,
            *self.target_router_topk,
            *self.predictor_routes,
            *self.snapshots,
            *self.cuda_events,
            *self.h2d_transfers,
            *self.runtime_events,
        )
        return "\n".join(_canonical_json(_encode_record(item)) for item in records)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TraceBundle:
        decoded = _decode_record(value)
        if not isinstance(decoded, cls):
            raise ValueError("trace JSON root must be a TraceBundle")
        return decoded

    @classmethod
    def from_json(cls, value: str) -> TraceBundle:
        parsed = _strict_json_loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("trace JSON root must be an object")
        return cls.from_dict(parsed)

    @classmethod
    def from_jsonl(cls, value: str) -> TraceBundle:
        records = [
            _decode_record(parsed)
            for parsed in (
                _strict_json_loads(line)
                for line in value.split("\n")
                if line.strip()
            )
        ]
        manifests = [
            item for item in records if isinstance(item, WorkloadManifestTrace)
        ]
        if len(manifests) != 1:
            raise ValueError("trace JSONL must contain one workload manifest")

        def collect(record_type: type[Any]) -> tuple[Any, ...]:
            return tuple(item for item in records if isinstance(item, record_type))

        known = (
            WorkloadManifestTrace,
            DraftTreeTrace,
            CycleTrace,
            GenerationTrace,
            LayerRouteTrace,
            TargetRouterTopKTrace,
            PredictorRouteTrace,
            CacheSnapshotTrace,
            CudaEventTrace,
            H2DTrace,
            RuntimeEventTrace,
        )
        if any(not isinstance(item, known) for item in records):
            raise ValueError("trace JSONL contains a non-top-level record")
        return cls(
            manifest=manifests[0],
            trees=collect(DraftTreeTrace),
            cycles=collect(CycleTrace),
            generations=collect(GenerationTrace),
            target_routes=collect(LayerRouteTrace),
            target_router_topk=collect(TargetRouterTopKTrace),
            predictor_routes=collect(PredictorRouteTrace),
            snapshots=collect(CacheSnapshotTrace),
            cuda_events=collect(CudaEventTrace),
            h2d_transfers=collect(H2DTrace),
            runtime_events=collect(RuntimeEventTrace),
        )


_RECORD_NAMES: dict[type[Any], str] = {
    ManifestEntry: "manifest_entry",
    WorkloadManifestTrace: "workload_manifest",
    DraftNodeTrace: "draft_node",
    DraftTreeTrace: "draft_tree",
    CycleTrace: "cycle",
    GenerationTrace: "generation",
    LayerRouteTrace: "target_route",
    ExpertProbabilityTrace: "expert_probability",
    TargetRouterTopKTrace: "target_router_topk",
    PredictorRouteTrace: "predictor_route",
    CacheStatsTrace: "cache_stats",
    CacheSlotTrace: "cache_slot",
    PendingTransferTrace: "pending_transfer",
    LayerCacheSnapshotTrace: "layer_cache_snapshot",
    CacheSnapshotTrace: "cache_snapshot",
    CudaEventTrace: "cuda_event",
    RuntimeEventTrace: "runtime_event",
    H2DTrace: "h2d",
    TraceBundle: "trace_bundle",
}
_RECORD_TYPES = {name: record_type for record_type, name in _RECORD_NAMES.items()}

_V1_ADDITIVE_FIELDS: dict[type[Any], frozenset[str]] = {
    GenerationTrace: frozenset(
        {"stopped_on_eos", "stopped_on_cycle_limit", "stop_token_id"}
    ),
    LayerCacheSnapshotTrace: frozenset(
        {"cache_id", "host_state_version", "lru_sequence", "stats"}
    ),
    RuntimeEventTrace: frozenset(
        {"cache_id", "queue_id", "submission_sequence", "stream_ids"}
    ),
}


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _strict_json_loads(value: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = item
        return result

    return json.loads(value, object_pairs_hook=reject_duplicates)


def _encode_value(value: object) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _encode_record(value)
    if isinstance(value, tuple):
        return [_encode_value(item) for item in value]
    return value


def _encode_record(value: object) -> dict[str, Any]:
    record_type = _RECORD_NAMES.get(type(value))
    if record_type is None:
        raise TypeError(f"unsupported trace record {type(value).__name__}")
    encoded = {
        field.name: _encode_value(getattr(value, field.name))
        for field in fields(value)
    }
    encoded["record_type"] = record_type
    trace_id = getattr(value, "trace_id", None)
    if trace_id is not None:
        encoded["trace_id"] = trace_id
    return encoded


@cache
def _field_types(record_type: type[Any]) -> dict[str, Any]:
    return get_type_hints(record_type)


def _decode_record(value: Mapping[str, Any]) -> object:
    if not isinstance(value, Mapping):
        raise TypeError("trace record must be an object")
    name = value.get("record_type")
    record_type = _RECORD_TYPES.get(str(name))
    if record_type is None:
        raise ValueError(f"unknown trace record type {name!r}")
    allowed = {field.name for field in fields(record_type)}
    extras = set(value) - allowed - {"record_type", "trace_id"}
    if extras:
        raise ValueError(f"unknown trace fields: {sorted(extras)}")
    missing = {
        field.name
        for field in fields(record_type)
        if field.name not in value
    }
    required_missing = missing - _V1_ADDITIVE_FIELDS.get(
        record_type, frozenset()
    )
    if required_missing:
        raise ValueError(
            f"missing trace fields: {sorted(required_missing)}"
        )
    hints = _field_types(record_type)
    decoded = record_type(
        **{
            field.name: _decode_value(hints[field.name], value[field.name])
            for field in fields(record_type)
            if field.name in value
        }
    )
    expected = getattr(decoded, "trace_id", None)
    if expected is None:
        if "trace_id" in value:
            raise ValueError("record type does not define a stable ID")
    elif value.get("trace_id") != expected:
        raise ValueError("trace record has an invalid stable ID")
    return decoded


def _decode_value(annotation: Any, value: Any) -> Any:
    origin = get_origin(annotation)
    if origin is tuple:
        if not isinstance(value, list):
            raise TypeError("tuple fields must be encoded as JSON arrays")
        arguments = get_args(annotation)
        if len(arguments) != 2 or arguments[1] is not Ellipsis:
            raise TypeError("only variable-length tuples are supported")
        return tuple(_decode_value(arguments[0], item) for item in value)
    if origin in (Union, types.UnionType):
        arguments = get_args(annotation)
        if value is None and type(None) in arguments:
            return None
        candidates = [item for item in arguments if item is not type(None)]
        if len(candidates) != 1:
            raise TypeError("unsupported union in trace schema")
        return _decode_value(candidates[0], value)
    if isinstance(annotation, type) and issubclass(annotation, StrEnum):
        return annotation(value)
    if isinstance(annotation, type) and is_dataclass(annotation):
        if not isinstance(value, Mapping):
            raise TypeError("nested trace records must be JSON objects")
        decoded = _decode_record(value)
        if not isinstance(decoded, annotation):
            raise TypeError("nested trace record has the wrong type")
        return decoded
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("integer trace field has the wrong type")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("floating trace field has the wrong type")
        return float(value)
    if annotation is bool:
        if not isinstance(value, bool):
            raise TypeError("boolean trace field has the wrong type")
        return value
    if annotation is str:
        if not isinstance(value, str):
            raise TypeError("string trace field has the wrong type")
        return value
    raise TypeError(f"unsupported trace field type {annotation!r}")
