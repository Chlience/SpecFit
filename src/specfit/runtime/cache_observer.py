
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from time import perf_counter_ns
from typing import Hashable, Protocol

import torch

from specfit.trace import RuntimeEventKind


class CacheTransitionObserver(Protocol):

    def record(
        self,
        *,
        kind: RuntimeEventKind,
        cache_id: str,
        queue_id: str,
        expert_id: Hashable,
        slot_id: int,
        generation: int,
        submission_sequence: int | None = None,
        victim_expert_id: Hashable | None = None,
        ready_event: torch.cuda.Event | None = None,
        stream_ids: tuple[int, ...] = (),
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class CacheTransition:

    run_id: str
    sequence: int
    host_time_ns: int
    kind: RuntimeEventKind
    cache_id: str
    queue_id: str
    expert_id: Hashable
    slot_id: int
    generation: int
    submission_sequence: int | None = None
    victim_expert_id: Hashable | None = None
    ready_event: torch.cuda.Event | None = None
    stream_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.run_id or not self.cache_id or not self.queue_id:
            raise ValueError(
                "run_id, cache_id and queue_id must be non-empty"
            )
        for name in ("sequence", "host_time_ns", "slot_id", "generation"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.kind, RuntimeEventKind):
            object.__setattr__(
                self,
                "kind",
                RuntimeEventKind(self.kind),
            )
        if self.submission_sequence is not None and (
            isinstance(self.submission_sequence, bool)
            or not isinstance(self.submission_sequence, int)
            or self.submission_sequence < 0
        ):
            raise ValueError(
                "submission_sequence must be non-negative"
            )
        if not isinstance(self.stream_ids, tuple):
            raise TypeError("stream_ids must be an immutable tuple")
        if (
            any(
                isinstance(stream_id, bool)
                or not isinstance(stream_id, int)
                or stream_id < 0
                for stream_id in self.stream_ids
            )
            or len(self.stream_ids) != len(set(self.stream_ids))
        ):
            raise ValueError(
                "stream_ids must contain unique non-negative integers"
            )
        if self.kind is RuntimeEventKind.H2D_SUBMIT and (
            self.submission_sequence is None or self.ready_event is None
        ):
            raise ValueError(
                "H2D submit requires a sequence and ready event"
            )
        if (
            self.kind is RuntimeEventKind.EVICT
            and self.victim_expert_id is None
        ):
            raise ValueError("eviction requires a victim expert")


@dataclass(frozen=True, slots=True)
class CacheTransitionCounters:
    h2d_submissions: int = 0
    resident_hits: int = 0
    inflight_joins: int = 0
    evictions: int = 0


class RecordingCacheTransitionObserver:

    def __init__(self, *, run_id: str) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")
        self.run_id = run_id
        self._events: list[CacheTransition] = []
        self._next_sequence = 0
        self._lock = Lock()

    def record(
        self,
        *,
        kind: RuntimeEventKind,
        cache_id: str,
        queue_id: str,
        expert_id: Hashable,
        slot_id: int,
        generation: int,
        submission_sequence: int | None = None,
        victim_expert_id: Hashable | None = None,
        ready_event: torch.cuda.Event | None = None,
        stream_ids: tuple[int, ...] = (),
    ) -> None:
        with self._lock:
            event = CacheTransition(
                run_id=self.run_id,
                sequence=self._next_sequence,
                host_time_ns=perf_counter_ns(),
                kind=kind,
                cache_id=cache_id,
                queue_id=queue_id,
                expert_id=expert_id,
                slot_id=slot_id,
                generation=generation,
                submission_sequence=submission_sequence,
                victim_expert_id=victim_expert_id,
                ready_event=ready_event,
                stream_ids=stream_ids,
            )
            self._events.append(event)
            self._next_sequence += 1

    def events(self) -> tuple[CacheTransition, ...]:
        with self._lock:
            return tuple(self._events)

    def cursor(self) -> int:

        with self._lock:
            return self._next_sequence

    def reset(self, *, run_id: str | None = None) -> None:

        if run_id is not None and (
            not isinstance(run_id, str) or not run_id
        ):
            raise ValueError("run_id must be a non-empty string")
        with self._lock:
            if run_id is not None:
                self.run_id = run_id
            self._events.clear()
            self._next_sequence = 0


def aggregate_cache_transition_counters(
    events: tuple[CacheTransition, ...],
) -> CacheTransitionCounters:

    if not isinstance(events, tuple):
        raise TypeError("events must be an immutable tuple")
    return CacheTransitionCounters(
        h2d_submissions=sum(
            event.kind is RuntimeEventKind.H2D_SUBMIT
            for event in events
        ),
        resident_hits=sum(
            event.kind is RuntimeEventKind.RESIDENT_HIT
            for event in events
        ),
        inflight_joins=sum(
            event.kind is RuntimeEventKind.INFLIGHT_JOIN
            for event in events
        ),
        evictions=sum(
            event.kind is RuntimeEventKind.EVICT for event in events
        ),
    )
