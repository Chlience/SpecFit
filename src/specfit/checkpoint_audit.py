"""Bitwise checks shared by restoration validation and selection measurements."""
from __future__ import annotations

from dataclasses import asdict
import hashlib

from .experiment import canonical_json_sha256
from .state import ExpertKey


def generation_tensors(checkpoint):
    state = checkpoint.state
    yield checkpoint.input_ids
    yield checkpoint.attention_mask
    for name in ("aux_history", "pending_seed", "fed_ids", "fed_attention", "rng_state", "cuda_rng_state"):
        yield getattr(state, name)
    yield from state.generated
    for cache in (state.cache, None if state.drafter_state is None else state.drafter_state[0]):
        if cache is not None:
            for layer in cache.layers:
                yield layer.keys
                yield layer.values
    if state.drafter_state is not None:
        yield state.drafter_state[2]


def generation_scalars(checkpoint):
    state = checkpoint.state
    return (checkpoint.config, state.emitted, state.proposed_total, state.accepted_per_cycle,
        state.cache.get_seq_length(), None if state.drafter_state is None else state.drafter_state[1])


def checkpoint_digest(checkpoint):
    import torch

    def expert_id(value):
        return asdict(value) if isinstance(value, ExpertKey) else value

    digest = hashlib.sha256(canonical_json_sha256(generation_scalars(checkpoint.generation)).encode())
    metadata = [dict(sequence=c.sequence, submission_sequence=c.submission_sequence,
        state_version=c.state_version, stats=asdict(c.stats),
        resident=[(expert_id(key), slot) for key, slot in c.resident.items()],
        inflight=[(expert_id(t.expert_id), t.slot_id, t.generation, t.submission_sequence)
            for t in c.inflight.values()],
        transfers=[(expert_id(t.expert_id), t.slot_id, t.generation, t.submission_sequence)
            for t in c.transfers],
        slots=[(s.slot_id, expert_id(s.expert_id), s.generation, s.active_leases,
            s.last_used_sequence, sorted(s.use_events or {})) for s in c.slots])
        for c in checkpoint.expert_caches]
    digest.update(canonical_json_sha256(metadata).encode())
    tensors = [*generation_tensors(checkpoint.generation), *(c.weights for c in checkpoint.expert_caches)]
    for tensor in tensors:
        if tensor is None:
            digest.update(b"None")
            continue
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
        digest.update(memoryview(raw).cast("B"))
    return digest.hexdigest()


def audit_boundary(actual, expected, loaded):
    import torch

    if generation_scalars(actual) != generation_scalars(expected.generation):
        raise RuntimeError("restored generation position or budget differs from reference")
    for left, right in zip(generation_tensors(actual), generation_tensors(expected.generation), strict=True):
        equal = left is right if left is None or right is None else (
            left.shape == right.shape and left.dtype == right.dtype and torch.equal(
                left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)))
        if not equal:
            raise RuntimeError("restored KV, aux, seed, tokens or RNG differs bitwise")
    for cache, saved in zip(loaded.expert_caches, expected.expert_caches, strict=True):
        occupied = {slot.slot_id for slot in saved.slots if slot.expert_id is not None}
        for start in sorted(occupied):
            if start - 1 in occupied:
                continue
            stop = start + 1
            while stop in occupied:
                stop += 1
            # Compare complete occupied ranges on the cache device. The temporary
            # reference and equality buffers are released after each comparison.
            weights = cache._storage[start:stop].detach().view(torch.uint8)
            if not torch.equal(weights, saved.weights[start:stop].to(
                    weights.device, non_blocking=True).view(torch.uint8)):
                raise RuntimeError("restored expert slot weights differ bitwise")
    return dict(generation_state_exact=True, expert_weights_exact=True)
