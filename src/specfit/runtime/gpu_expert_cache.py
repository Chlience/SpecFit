
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from threading import RLock
from time import perf_counter_ns
from typing import Hashable, Mapping, Protocol

import torch

from specfit.trace import RuntimeEventKind

from .cache_observer import CacheTransitionObserver
from .cache_snapshot import (
    CacheSlotSnapshot,
    GpuExpertCacheSnapshot,
    PendingTransferSnapshot,
)
from .lease import ExpertCacheStats, ExpertLease, NoEvictableSlotError


@dataclass(slots=True)
class _MutableStats:
    h2d_submissions: int = 0
    resident_hits: int = 0
    inflight_joins: int = 0
    evictions: int = 0


@dataclass(slots=True)
class _Slot:
    slot_id: int
    tensor: torch.Tensor
    expert_id: Hashable | None = None
    generation: int = 0
    active_leases: int = 0
    last_used_sequence: int = 0
    ready_event: torch.cuda.Event | None = None
    use_events: dict[int, torch.cuda.Event] | None = None

    def __post_init__(self) -> None:
        if self.use_events is None:
            self.use_events = {}


@dataclass(frozen=True, slots=True)
class _Transfer:
    expert_id: Hashable
    slot_id: int
    generation: int
    submission_sequence: int
    ready_event: torch.cuda.Event


class _SourceOwner(Protocol):
    def close(self) -> None: ...


@dataclass(frozen=True)
class _ExpertCacheCheckpoint:

    owner: object
    weights: torch.Tensor
    slots: tuple[_Slot, ...]
    resident: dict
    inflight: dict
    transfers: tuple[_Transfer, ...]
    stats: _MutableStats
    sequence: int
    submission_sequence: int
    state_version: int


class GpuExpertCache:

    def __init__(
        self,
        cpu_sources: Mapping[Hashable, torch.Tensor],
        *,
        slot_count: int,
        device: torch.device | str,
        adopt_pinned_sources: bool = False,
        cache_id: str | None = None,
        queue_id: str | None = None,
        observer: CacheTransitionObserver | None = None,
        source_owner: _SourceOwner | None = None,
    ) -> None:
        if not cpu_sources:
            raise ValueError("cpu_sources must contain at least one expert")
        if slot_count <= 0:
            raise ValueError("slot_count must be positive")
        if source_owner is not None and not adopt_pinned_sources:
            raise ValueError(
                "source_owner requires adopt_pinned_sources=True"
            )

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("GpuExpertCache requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

        source_items = tuple(cpu_sources.items())
        first_shape = source_items[0][1].shape
        first_dtype = source_items[0][1].dtype
        self._sources: dict[Hashable, torch.Tensor] = {}
        self._source_order: dict[Hashable, int] = {}
        for source_index, (expert_id, source) in enumerate(source_items):
            if source.device.type != "cpu":
                raise ValueError(f"expert {expert_id!r} source must be on CPU")
            if source.shape != first_shape:
                raise ValueError("all expert sources must have the same shape")
            if source.dtype != first_dtype:
                raise ValueError("all expert sources must have the same dtype")
            if (
                adopt_pinned_sources
                and source.is_pinned()
                and source.is_contiguous()
            ):
                pinned = source.detach()
            else:
                if source_owner is not None:
                    raise ValueError(
                        "owned expert sources must be contiguous pinned "
                        "CPU tensors"
                    )
                pinned = torch.empty(
                    source.shape,
                    dtype=source.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                pinned.copy_(source.detach())
            self._sources[expert_id] = pinned
            self._source_order[expert_id] = source_index

        with torch.cuda.device(self.device):
            self.copy_stream = torch.cuda.Stream(device=self.device)
            storage = torch.empty(
                (slot_count, *first_shape),
                dtype=first_dtype,
                device=self.device,
            )
            # A recycled allocation can still have readers on its allocation
            # stream. Order our first side-stream write after those readers.
            self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))

        for name, value in (
            ("cache_id", cache_id),
            ("queue_id", queue_id),
        ):
            if value is not None and (
                not isinstance(value, str) or not value
            ):
                raise ValueError(f"{name} must be a non-empty string")
        self.cache_id = cache_id or (
            f"cuda-{self.device.index}-stream-"
            f"{int(self.copy_stream.cuda_stream)}"
        )
        self.queue_id = queue_id or f"{self.cache_id}-copy"
        self._storage = storage
        self._slots = [
            _Slot(slot_id=slot_id, tensor=storage[slot_id])
            for slot_id in range(slot_count)
        ]
        self._resident: dict[Hashable, int] = {}
        self._inflight: dict[Hashable, _Transfer] = {}
        self._outstanding_transfers: deque[_Transfer] = deque()
        self._stats = _MutableStats()
        self._sequence = 0
        self._submission_sequence = 0
        self._state_version = 0
        self._observer = observer
        self._source_owner = source_owner
        self._closed = False
        self._closing = False
        self._lock = RLock()

    @property
    def sources_are_pinned(self) -> bool:
        return not (self._closed or self._closing) and all(
            source.is_pinned() for source in self._sources.values()
        )

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def active_lease_count(self) -> int:
        with self._lock:
            return sum(slot.active_leases for slot in self._slots)

    def acquire(self, expert_id: Hashable) -> ExpertLease:
        return self._acquire(expert_id, defer_touch=False)

    def _prefetch(self, expert_id: Hashable) -> ExpertLease:
        return self._acquire(expert_id, defer_touch=True)

    def _prefetch_nonblocking(
        self, expert_id: Hashable, *, resident_only: bool = False,
        eviction_probabilities: Mapping[Hashable, float] | None = None,
    ) -> ExpertLease | None:
        with self._lock:
            if self._closed or self._closing:
                raise RuntimeError("expert cache is closed")
            if expert_id not in self._sources:
                raise KeyError(f"unknown expert {expert_id!r}")
            self._reap_completed_transfers()
            if expert_id in self._resident or expert_id in self._inflight:
                return self._acquire(expert_id, defer_touch=True)
            if resident_only:
                return None
            slot = (self._choose_lru_slot(ready_only=True)
                    if eviction_probabilities is None else
                    self._choose_probability_slot(expert_id, eviction_probabilities))
            if slot is None:
                return None
            self._assign_and_submit(slot, expert_id)
            return self._new_lease(slot, defer_touch=True)

    def _acquire(self, expert_id: Hashable, *, defer_touch: bool) -> ExpertLease:
        with self._lock:
            if self._closed or self._closing:
                raise RuntimeError("expert cache is closed")
            if expert_id not in self._sources:
                raise KeyError(f"unknown expert {expert_id!r}")
            self._reap_completed_transfers()

            transfer = self._inflight.get(expert_id)
            if transfer is not None:
                slot = self._slots[transfer.slot_id]
                if self._observer is not None:
                    self._record_transition(
                        kind=RuntimeEventKind.INFLIGHT_JOIN,
                        expert_id=expert_id,
                        slot=slot,
                        submission_sequence=(
                            transfer.submission_sequence
                        ),
                        ready_event=transfer.ready_event,
                    )
                self._stats.inflight_joins += 1
                return self._new_lease(slot, defer_touch=defer_touch)

            resident_slot_id = self._resident.get(expert_id)
            if resident_slot_id is not None:
                slot = self._slots[resident_slot_id]
                if self._observer is not None:
                    self._record_transition(
                        kind=RuntimeEventKind.RESIDENT_HIT,
                        expert_id=expert_id,
                        slot=slot,
                    )
                self._stats.resident_hits += 1
                return self._new_lease(slot, defer_touch=defer_touch)

            slot = self._choose_lru_slot()
            if slot is None:
                raise NoEvictableSlotError(
                    "all GPU expert slots have active leases"
                )
            self._assign_and_submit(slot, expert_id)
            return self._new_lease(slot, defer_touch=defer_touch)

    def snapshot(self) -> GpuExpertCacheSnapshot:

        with self._lock:
            completion = {
                transfer.submission_sequence: transfer.ready_event.query()
                for transfer in self._outstanding_transfers
            }
            pending_guard_counts = {
                slot.slot_id: sum(
                    not event.query()
                    for event in (slot.use_events or {}).values()
                )
                for slot in self._slots
            }
            current_transfer = {
                (transfer.slot_id, transfer.generation): transfer
                for transfer in self._inflight.values()
            }
            slots = []
            for slot in self._slots:
                transfer = current_transfer.get(
                    (slot.slot_id, slot.generation)
                )
                if slot.expert_id is None:
                    ready = False
                elif transfer is not None:
                    ready = completion[transfer.submission_sequence]
                else:
                    ready = slot.expert_id in self._resident
                slots.append(
                    CacheSlotSnapshot(
                        slot_id=slot.slot_id,
                        expert_id=slot.expert_id,
                        generation=slot.generation,
                        active_leases=slot.active_leases,
                        last_used_sequence=slot.last_used_sequence,
                        ready=ready,
                        pending_use_guards=pending_guard_counts[
                            slot.slot_id
                        ],
                    )
                )
            pending = tuple(
                PendingTransferSnapshot(
                    expert_id=transfer.expert_id,
                    slot_id=transfer.slot_id,
                    generation=transfer.generation,
                    submission_sequence=transfer.submission_sequence,
                    ready_event=transfer.ready_event,
                    ready_event_completed=False,
                )
                for transfer in self._outstanding_transfers
                if not completion[transfer.submission_sequence]
            )
            return GpuExpertCacheSnapshot(
                cache_id=self.cache_id,
                queue_id=self.queue_id,
                capacity=len(self._slots),
                host_state_version=self._state_version,
                lru_sequence=self._sequence,
                submission_sequence=self._submission_sequence,
                slots=tuple(slots),
                pending_transfers=pending,
                stats=self._stats_snapshot(
                    pending_use_guards=sum(
                        pending_guard_counts.values()
                    )
                ),
            )

    def cold_reset(self) -> None:

        with self._lock:
            if self._closed or self._closing:
                raise RuntimeError("expert cache is closed")
            if any(slot.active_leases for slot in self._slots):
                raise RuntimeError(
                    "cold reset cannot run while an active lease exists"
                )
            incomplete_transfer = any(
                not transfer.ready_event.query()
                for transfer in self._outstanding_transfers
            )
            incomplete_guard = any(
                not event.query()
                for slot in self._slots
                for event in (slot.use_events or {}).values()
            )
            if incomplete_transfer or incomplete_guard:
                raise RuntimeError(
                    "cold reset requires explicit CUDA synchronization"
                )

            self._resident.clear()
            self._inflight.clear()
            self._outstanding_transfers.clear()
            for slot in self._slots:
                slot.expert_id = None
                slot.generation = 0
                slot.active_leases = 0
                slot.last_used_sequence = 0
                slot.ready_event = None
                if slot.use_events is not None:
                    slot.use_events.clear()
            self._stats = _MutableStats()
            self._sequence = 0
            self._submission_sequence = 0
            self._state_version = 0

    def _require_checkpoint_boundary(self) -> None:
        if self._closed or self._closing:
            raise RuntimeError("expert cache is closed")
        if self._observer is not None:
            raise RuntimeError("checkpoint restoration requires no active cache observer")
        if any(slot.active_leases for slot in self._slots):
            raise RuntimeError("checkpoint boundary cannot contain active leases")
        if any(not t.ready_event.query() for t in self._outstanding_transfers) or any(
            not event.query()
            for slot in self._slots
            for event in (slot.use_events or {}).values()
        ):
            raise RuntimeError("checkpoint boundary requires explicit CUDA synchronization")

    def _capture_checkpoint(self) -> _ExpertCacheCheckpoint:
        with self._lock:
            self._require_checkpoint_boundary()
            weights = torch.empty_like(self._storage, device="cpu", pin_memory=True)
            weights.copy_(self._storage.detach())
            return _ExpertCacheCheckpoint(
                owner=self,
                weights=weights,
                slots=tuple(replace(slot, tensor=weights[slot.slot_id],
                    use_events=dict(slot.use_events or {})) for slot in self._slots),
                resident=dict(self._resident), inflight=dict(self._inflight),
                transfers=tuple(self._outstanding_transfers), stats=replace(self._stats),
                sequence=self._sequence, submission_sequence=self._submission_sequence,
                state_version=self._state_version,
            )

    def _restore_checkpoint(self, checkpoint: _ExpertCacheCheckpoint) -> dict:
        started = perf_counter_ns()
        with self._lock:
            self._require_checkpoint_boundary()
            if checkpoint.owner is not self:
                raise ValueError("expert checkpoint belongs to a different cache instance")
            checked = perf_counter_ns()
            self._storage.copy_(checkpoint.weights)
            copied = perf_counter_ns()
            torch.cuda.current_stream(self.device).synchronize()
            synchronized = perf_counter_ns()
            self._slots = [replace(slot, tensor=self._storage[slot.slot_id],
                use_events=dict(slot.use_events or {})) for slot in checkpoint.slots]
            self._resident = dict(checkpoint.resident)
            self._inflight = dict(checkpoint.inflight)
            self._outstanding_transfers = deque(checkpoint.transfers)
            self._stats = replace(checkpoint.stats)
            self._sequence = checkpoint.sequence
            self._submission_sequence = checkpoint.submission_sequence
            self._state_version = checkpoint.state_version
            finished = perf_counter_ns()
            return dict(weight_bytes=checkpoint.weights.numel() * checkpoint.weights.element_size(),
                weights_pinned=checkpoint.weights.is_pinned(), total_wall_ns=finished - started,
                timings_ns=dict(boundary_check=checked - started, weight_copy_call=copied - checked,
                    stream_synchronize=synchronized - copied, metadata_restore=finished - synchronized))

    def close(self) -> None:

        with self._lock:
            if self._closed:
                return
            if not self._closing and any(
                slot.active_leases for slot in self._slots
            ):
                raise RuntimeError(
                    "expert cache cannot close while an active lease exists"
                )

            self._closing = True
            torch.cuda.synchronize(self.device)
            self._resident.clear()
            self._inflight.clear()
            self._outstanding_transfers.clear()
            for slot in self._slots:
                slot.expert_id = None
                slot.ready_event = None
                if slot.use_events is not None:
                    slot.use_events.clear()
            self._sources.clear()
            source_owner = self._source_owner
            if source_owner is not None:
                source_owner.close()
            self._source_owner = None
            self._closed = True
            self._closing = False

    def stats(self) -> ExpertCacheStats:
        with self._lock:
            self._reap_completed_use_guards()
            return self._stats_snapshot()

    def resident_expert_ids(self) -> tuple[Hashable, ...]:
        with self._lock:
            self._reap_completed_transfers()
            return tuple(
                sorted(
                    self._resident,
                    key=self._source_order.__getitem__,
                )
            )

    def _new_lease(self, slot: _Slot, *, defer_touch: bool) -> ExpertLease:
        if slot.expert_id is None or slot.ready_event is None:
            raise RuntimeError("cannot lease an unassigned GPU slot")
        slot.active_leases += 1
        if defer_touch:
            self._state_version += 1
        else:
            self._touch(slot)
        return ExpertLease(
            cache=self,
            expert_id=slot.expert_id,
            slot_id=slot.slot_id,
            generation=slot.generation,
            ready_event=slot.ready_event,
            _defer_touch=defer_touch,
        )

    def _choose_lru_slot(self, *, ready_only: bool = False) -> _Slot | None:
        available = [slot for slot in self._slots if slot.active_leases == 0]
        if ready_only:
            # Probe in the same eviction order and stop at the first writable
            # slot. Resident metadata already proves this generation's H2D
            # completed; its later GPU readers still require their guards.
            for slot in sorted(available, key=lambda slot: (
                    slot.expert_id is not None, slot.last_used_sequence, slot.slot_id)):
                copied = (slot.ready_event is None
                          or self._resident.get(slot.expert_id) == slot.slot_id
                          or slot.ready_event.query())
                if copied and all(event.query() for event in (slot.use_events or {}).values()):
                    return slot
            return None
        if not available:
            return None
        return min(
            available,
            key=lambda slot: (
                slot.expert_id is not None,
                slot.last_used_sequence,
                slot.slot_id,
            ),
        )

    def _choose_probability_slot(
        self, expert_id: Hashable, probabilities: Mapping[Hashable, float],
    ) -> _Slot | None:
        """Replace only a writable, unleased expert with a lower probability.

        Empty slots precede occupied slots; equal victim probabilities retain
        the historical LRU and slot-ID tie breakers. Demand acquisition does
        not use this speculative admission rule.
        """
        incoming = probabilities.get(expert_id, 0.0)
        available = (slot for slot in self._slots if slot.active_leases == 0)
        for slot in sorted(available, key=lambda slot: (
                slot.expert_id is not None, probabilities.get(slot.expert_id, 0.0),
                slot.last_used_sequence, slot.slot_id)):
            if slot.expert_id is not None and probabilities.get(slot.expert_id, 0.0) >= incoming:
                break
            copied = (slot.ready_event is None
                      or self._resident.get(slot.expert_id) == slot.slot_id
                      or slot.ready_event.query())
            if copied and all(event.query() for event in (slot.use_events or {}).values()):
                return slot
        return None

    def _assign_and_submit(self, slot: _Slot, expert_id: Hashable) -> None:
        previous_expert = slot.expert_id
        if previous_expert is not None:
            self._resident.pop(previous_expert, None)
            self._inflight.pop(previous_expert, None)
            self._stats.evictions += 1

        slot.generation += 1
        slot.expert_id = expert_id
        guards = tuple((slot.use_events or {}).values())
        if slot.use_events is not None:
            slot.use_events.clear()

        with torch.cuda.stream(self.copy_stream):
            for guard in guards:
                self.copy_stream.wait_event(guard)
            slot.tensor.copy_(self._sources[expert_id], non_blocking=True)
            ready_event = torch.cuda.Event(blocking=False)
            ready_event.record(self.copy_stream)

        slot.ready_event = ready_event
        transfer = _Transfer(
            expert_id=expert_id,
            slot_id=slot.slot_id,
            generation=slot.generation,
            submission_sequence=self._submission_sequence,
            ready_event=ready_event,
        )
        self._inflight[expert_id] = transfer
        self._outstanding_transfers.append(transfer)
        self._stats.h2d_submissions += 1
        self._submission_sequence += 1
        if self._observer is not None:
            if previous_expert is not None:
                self._record_transition(
                    kind=RuntimeEventKind.EVICT,
                    expert_id=expert_id,
                    slot=slot,
                    victim_expert_id=previous_expert,
                )
            self._record_transition(
                kind=RuntimeEventKind.H2D_SUBMIT,
                expert_id=expert_id,
                slot=slot,
                submission_sequence=transfer.submission_sequence,
                ready_event=ready_event,
            )

    def _reap_completed_transfers(self) -> None:
        changed = False
        while (
            self._outstanding_transfers
            and self._outstanding_transfers[0].ready_event.query()
        ):
            transfer = self._outstanding_transfers.popleft()
            slot = self._slots[transfer.slot_id]
            if (
                slot.expert_id == transfer.expert_id
                and slot.generation == transfer.generation
            ):
                self._resident[transfer.expert_id] = transfer.slot_id
            current = self._inflight.get(transfer.expert_id)
            if current == transfer:
                del self._inflight[transfer.expert_id]
            changed = True
        if changed:
            self._state_version += 1

    def _reap_completed_use_guards(self) -> None:
        changed = False
        for slot in self._slots:
            if slot.use_events is None:
                continue
            for stream_id, event in tuple(slot.use_events.items()):
                if event.query():
                    del slot.use_events[stream_id]
                    changed = True
        if changed:
            self._state_version += 1

    def _touch(self, slot: _Slot) -> None:
        self._sequence += 1
        self._state_version += 1
        slot.last_used_sequence = self._sequence

    def _stats_snapshot(
        self,
        *,
        pending_use_guards: int | None = None,
    ) -> ExpertCacheStats:
        if pending_use_guards is None:
            pending_use_guards = sum(
                len(slot.use_events or ()) for slot in self._slots
            )
        return ExpertCacheStats(
            h2d_submissions=self._stats.h2d_submissions,
            resident_hits=self._stats.resident_hits,
            inflight_joins=self._stats.inflight_joins,
            evictions=self._stats.evictions,
            pending_use_guards=pending_use_guards,
        )

    def _lease_tensor(
        self, expert_id: Hashable, slot_id: int, generation: int,
        *, touch: bool = False,
    ) -> torch.Tensor:
        with self._lock:
            slot = self._slots[slot_id]
            if (
                slot.expert_id != expert_id
                or slot.generation != generation
                or slot.active_leases <= 0
            ):
                raise RuntimeError("expert lease refers to a reused GPU slot")
            if touch:
                self._touch(slot)
            return slot.tensor

    def _record_consume_wait(
        self,
        *,
        expert_id: Hashable,
        slot_id: int,
        generation: int,
        ready_event: torch.cuda.Event,
        stream: torch.cuda.Stream,
    ) -> None:
        observer = self._observer
        if observer is None:
            return
        observer.record(
            kind=RuntimeEventKind.CONSUME_WAIT,
            cache_id=self.cache_id,
            queue_id=self.queue_id,
            expert_id=expert_id,
            slot_id=slot_id,
            generation=generation,
            ready_event=ready_event,
            stream_ids=(int(stream.cuda_stream),),
        )

    def _release(
        self,
        expert_id: Hashable,
        slot_id: int,
        generation: int,
        used_streams: tuple[torch.cuda.Stream, ...],
    ) -> None:
        with self._lock:
            slot = self._slots[slot_id]
            if (
                slot.expert_id != expert_id
                or slot.generation != generation
                or slot.active_leases <= 0
            ):
                raise RuntimeError("expert lease refers to a reused GPU slot")
            for stream in used_streams:
                use_event = torch.cuda.Event(blocking=False)
                use_event.record(stream)
                if slot.use_events is None:
                    slot.use_events = {}
                slot.use_events[stream.cuda_stream] = use_event
            if self._observer is not None:
                self._record_transition(
                    kind=RuntimeEventKind.RELEASE,
                    expert_id=expert_id,
                    slot=slot,
                    stream_ids=tuple(
                        sorted(
                            int(stream.cuda_stream)
                            for stream in used_streams
                        )
                    ),
                )
            slot.active_leases -= 1
            self._touch(slot)

    def _record_transition(
        self,
        *,
        kind: RuntimeEventKind,
        expert_id: Hashable,
        slot: _Slot,
        submission_sequence: int | None = None,
        victim_expert_id: Hashable | None = None,
        ready_event: torch.cuda.Event | None = None,
        stream_ids: tuple[int, ...] = (),
    ) -> None:
        observer = self._observer
        if observer is None:
            return
        observer.record(
            kind=kind,
            cache_id=self.cache_id,
            queue_id=self.queue_id,
            expert_id=expert_id,
            slot_id=slot.slot_id,
            generation=slot.generation,
            submission_sequence=submission_sequence,
            victim_expert_id=victim_expert_id,
            ready_event=ready_event,
            stream_ids=stream_ids,
        )
