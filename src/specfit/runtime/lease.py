
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Hashable

import torch

if TYPE_CHECKING:
    from .gpu_expert_cache import GpuExpertCache


class NoEvictableSlotError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ExpertCacheStats:
    h2d_submissions: int
    resident_hits: int
    inflight_joins: int
    evictions: int
    pending_use_guards: int


class ExpertLease:

    __slots__ = (
        "_cache",
        "_expert_id",
        "_generation",
        "_released",
        "_used_streams",
        "_deferred_touch",
        "_ready_confirmed",
        "ready_event",
        "slot_id",
    )

    def __init__(
        self,
        cache: GpuExpertCache,
        expert_id: Hashable,
        slot_id: int,
        generation: int,
        ready_event: torch.cuda.Event,
        *,
        _defer_touch: bool = False,
    ) -> None:
        self._cache = cache
        self._expert_id = expert_id
        self._generation = generation
        self._released = False
        self._used_streams: dict[int, torch.cuda.Stream] = {}
        self._deferred_touch = _defer_touch
        # Private demand-session fast path: only an already completed, immutable
        # transfer event may set this. The lease still protects this generation.
        self._ready_confirmed = False
        self.slot_id = slot_id
        self.ready_event = ready_event

    def wait(self, stream: torch.cuda.Stream | None = None) -> torch.Tensor:

        if self._released:
            raise RuntimeError("cannot wait on a released expert lease")
        if stream is None:
            stream = torch.cuda.current_stream(self._cache.device)
        if not self._ready_confirmed:
            stream.wait_event(self.ready_event)
        self._used_streams[stream.cuda_stream] = stream
        tensor = self._cache._lease_tensor(
            self._expert_id,
            self.slot_id,
            self._generation,
            touch=self._deferred_touch,
        )
        self._deferred_touch = False
        if self._cache._observer is not None:
            self._cache._record_consume_wait(
                expert_id=self._expert_id,
                slot_id=self.slot_id,
                generation=self._generation,
                ready_event=self.ready_event,
                stream=stream,
            )
        return tensor

    def release(self) -> None:
        if self._released:
            return
        self._cache._release(
            self._expert_id,
            self.slot_id,
            self._generation,
            tuple(self._used_streams.values()),
        )
        self._released = True

    def __enter__(self) -> ExpertLease:
        if self._released:
            raise RuntimeError("cannot re-enter a released expert lease")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()
