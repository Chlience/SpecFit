"""Private cache-off intervention for the controlled comparison only."""
from contextlib import ExitStack

import torch


def discard_completed_experts(loaded):
    """Invalidate reusable weights while retaining counters, allocation and lease safety.

    Called once before a target forward, including prefill. Synchronization and
    host work are charged to that request. Within-forward batching is unchanged.
    """
    torch.cuda.synchronize()
    caches = loaded.expert_caches
    with ExitStack() as stack:
        for cache in caches:
            stack.enter_context(cache._lock)
        # Validate every cache before changing any; no stale active/in-flight users.
        for cache in caches:
            cache._require_checkpoint_boundary()
        for cache in caches:
            cache._resident.clear()
            cache._inflight.clear()
            cache._outstanding_transfers.clear()
            for slot in cache._slots:
                slot.expert_id = None
                slot.generation += 1
                slot.last_used_sequence = 0
                slot.ready_event = None
                if slot.use_events is not None:
                    slot.use_events.clear()
            cache._state_version += 1
