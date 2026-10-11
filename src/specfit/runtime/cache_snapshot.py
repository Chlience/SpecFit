
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import torch

from .lease import ExpertCacheStats


def _require_nonnegative(**values: int) -> None:
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class CacheSlotSnapshot:

    slot_id: int
    expert_id: Hashable | None
    generation: int
    active_leases: int
    last_used_sequence: int
    ready: bool
    pending_use_guards: int

    def __post_init__(self) -> None:
        _require_nonnegative(
            slot_id=self.slot_id,
            generation=self.generation,
            active_leases=self.active_leases,
            last_used_sequence=self.last_used_sequence,
            pending_use_guards=self.pending_use_guards,
        )
        if not isinstance(self.ready, bool):
            raise TypeError("ready must be a boolean")
        if self.expert_id is None and (
            self.ready or self.active_leases or self.pending_use_guards
        ):
            raise ValueError("an empty slot cannot be ready or in use")


@dataclass(frozen=True, slots=True)
class PendingTransferSnapshot:

    expert_id: Hashable
    slot_id: int
    generation: int
    submission_sequence: int
    ready_event: torch.cuda.Event
    ready_event_completed: bool

    def __post_init__(self) -> None:
        _require_nonnegative(
            slot_id=self.slot_id,
            generation=self.generation,
            submission_sequence=self.submission_sequence,
        )
        if not isinstance(self.ready_event_completed, bool):
            raise TypeError("ready_event_completed must be a boolean")


@dataclass(frozen=True, slots=True)
class GpuExpertCacheSnapshot:

    cache_id: str
    queue_id: str
    capacity: int
    host_state_version: int
    lru_sequence: int
    submission_sequence: int
    slots: tuple[CacheSlotSnapshot, ...]
    pending_transfers: tuple[PendingTransferSnapshot, ...]
    stats: ExpertCacheStats

    def __post_init__(self) -> None:
        if not self.cache_id or not self.queue_id:
            raise ValueError("cache_id and queue_id must be non-empty")
        _require_nonnegative(
            capacity=self.capacity,
            host_state_version=self.host_state_version,
            lru_sequence=self.lru_sequence,
            submission_sequence=self.submission_sequence,
        )
        if self.capacity == 0:
            raise ValueError("cache capacity must be positive")
        if not isinstance(self.slots, tuple):
            raise TypeError("slots must be an immutable tuple")
        if not isinstance(self.pending_transfers, tuple):
            raise TypeError(
                "pending_transfers must be an immutable tuple"
            )
        if len(self.slots) != self.capacity:
            raise ValueError("snapshot must contain every cache slot")
        if tuple(slot.slot_id for slot in self.slots) != tuple(
            range(self.capacity)
        ):
            raise ValueError("slot IDs must cover range(capacity)")
        assigned = [
            slot.expert_id
            for slot in self.slots
            if slot.expert_id is not None
        ]
        if len(assigned) != len(set(assigned)):
            raise ValueError("current expert assignments must be unique")

        sequences = [
            transfer.submission_sequence
            for transfer in self.pending_transfers
        ]
        if sequences != sorted(sequences):
            raise ValueError(
                "pending transfers must follow submission order"
            )
        if len(sequences) != len(set(sequences)):
            raise ValueError("pending submission sequences must be unique")
        if any(
            sequence >= self.submission_sequence for sequence in sequences
        ):
            raise ValueError(
                "pending sequence must precede the submission counter"
            )
        slots_by_id = {slot.slot_id: slot for slot in self.slots}
        for transfer in self.pending_transfers:
            slot = slots_by_id.get(transfer.slot_id)
            if slot is None:
                raise ValueError(
                    "pending transfer references an unknown slot"
                )
            if transfer.generation > slot.generation:
                raise ValueError(
                    "pending transfer generation exceeds its current slot"
                )
            if (
                transfer.generation == slot.generation
                and transfer.expert_id != slot.expert_id
            ):
                raise ValueError(
                    "current pending transfer does not match its slot"
                )
