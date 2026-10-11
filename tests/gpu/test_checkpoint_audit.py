from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch

from specfit.checkpoint_audit import audit_boundary
from specfit.runtime import GpuExpertCache


def generation(device="cpu"):
    def cache():
        return SimpleNamespace(get_seq_length=lambda: 2, layers=[SimpleNamespace(
            keys=torch.zeros((1, 1, 2, 4), device=device),
            values=torch.ones((1, 1, 2, 4), device=device))])

    return SimpleNamespace(config=(12, 2, 5, 18),
        input_ids=torch.tensor([[1, 2]], device=device),
        attention_mask=torch.ones((1, 2), dtype=torch.long, device=device),
        state=SimpleNamespace(emitted=2, proposed_total=3, accepted_per_cycle=[1],
            cache=cache(), drafter_state=(cache(), 2, torch.zeros((1, 1, 4), device=device)),
            aux_history=torch.zeros((1, 2, 4), device=device),
            pending_seed=torch.tensor([[3]], device=device),
            fed_ids=torch.tensor([[1, 2]], device=device),
            fed_attention=torch.ones((1, 2), dtype=torch.long, device=device),
            generated=[torch.tensor([[1, 2]], device=device)],
            rng_state=torch.tensor([0, 1, 2], dtype=torch.uint8),
            cuda_rng_state=torch.tensor([3, 4, 5], dtype=torch.uint8)))


def fixture(occupied, *, device="cpu", pinned=False):
    # Include distinct NaN payloads, both zero signs, infinity and a subnormal.
    words = torch.tensor([0x0000, 0x8000, 0x7FC1, 0xFFC2, 0x3F80, 0xBF80, 0x0001, 0x7F80],
                         dtype=torch.uint16)
    weights = words.repeat(8).reshape(4, 2, 8).view(torch.bfloat16)
    reference = weights.pin_memory() if pinned else weights.clone()
    saved = SimpleNamespace(weights=reference, slots=tuple(SimpleNamespace(
        slot_id=i, expert_id=i if i in occupied else None) for i in (3, 1, 0, 2)))
    actual = generation(device)
    expected = SimpleNamespace(generation=deepcopy(actual), expert_caches=(saved,))
    loaded = SimpleNamespace(expert_caches=(SimpleNamespace(_storage=weights.to(device).clone()),))
    return actual, expected, loaded


class CheckpointAuditCpuTest(unittest.TestCase):
    def test_generation_corruption_is_rejected_without_changing_the_reference(self):
        original = generation()
        expected = SimpleNamespace(generation=deepcopy(original), expert_caches=())
        loaded = SimpleNamespace(expert_caches=())
        for field in ("config", "keys", "aux_history", "pending_seed", "rng_state", "cuda_rng_state"):
            actual = deepcopy(original)
            with self.subTest(field=field):
                if field == "config":
                    actual.config = (13, 2, 5, 18)
                elif field == "keys":
                    actual.state.cache.layers[0].keys.view(torch.uint8).view(-1)[0] = 1
                else:
                    value = getattr(actual.state, field)
                    value.view(torch.uint8).view(-1)[0] ^= 1
                with self.assertRaisesRegex(RuntimeError, "differs from reference|differs bitwise"):
                    audit_boundary(actual, expected, loaded)
                self.assertEqual(audit_boundary(original, expected, loaded),
                    dict(generation_state_exact=True, expert_weights_exact=True))

    def test_cpu_weights_keep_the_same_occupied_slot_scope(self):
        actual, expected, loaded = fixture({0, 2, 3})
        storage = loaded.expert_caches[0]._storage
        storage[1].zero_()
        self.assertTrue(audit_boundary(actual, expected, loaded)["expert_weights_exact"])
        storage[3].view(torch.uint8).view(-1)[-1] ^= 1
        with self.assertRaisesRegex(RuntimeError, "expert slot weights differ bitwise"):
            audit_boundary(actual, expected, loaded)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class CheckpointAuditGpuTest(unittest.TestCase):
    def test_equal_bf16_bytes_include_nan_payloads_and_signed_zero(self):
        for occupied in (set(), {0}, {3}, {0, 2, 3}, set(range(4))):
            with self.subTest(occupied=sorted(occupied)):
                actual, expected, loaded = fixture(occupied, device="cuda:0", pinned=True)
                before = expected.expert_caches[0].weights.view(torch.uint8).clone()
                self.assertEqual(audit_boundary(actual, expected, loaded),
                    dict(generation_state_exact=True, expert_weights_exact=True))
                self.assertTrue(torch.equal(expected.expert_caches[0].weights.view(torch.uint8), before))
                self.assertTrue(torch.equal(loaded.expert_caches[0]._storage.view(torch.uint8).cpu(), before))

    def test_changed_bytes_in_each_occupied_slot_are_rejected(self):
        for occupied in ({0, 2, 3}, set(range(4))):
            actual, expected, loaded = fixture(occupied, device="cuda:0", pinned=True)
            storage = loaded.expert_caches[0]._storage
            reference = expected.expert_caches[0].weights
            for slot in sorted(occupied):
                # Include a zero sign bit and a NaN payload, not just finite values.
                for byte in (0, 1, 4, storage[slot].numel() * storage.element_size() - 1):
                    with self.subTest(occupied=sorted(occupied), slot=slot, byte=byte):
                        storage.copy_(reference)
                        storage[slot].view(torch.uint8).view(-1)[byte] ^= 0x80 if byte == 1 else 1
                        with self.assertRaisesRegex(RuntimeError, "expert slot weights differ bitwise"):
                            audit_boundary(actual, expected, loaded)
                        storage.copy_(reference)
                        self.assertTrue(audit_boundary(actual, expected, loaded)["expert_weights_exact"])

    def test_unoccupied_slot_contents_are_not_part_of_the_audit(self):
        for occupied in (set(), {0, 2}, {1, 2}):
            with self.subTest(occupied=sorted(occupied)):
                actual, expected, loaded = fixture(occupied, device="cuda:0", pinned=True)
                for slot in set(range(4)) - occupied:
                    loaded.expert_caches[0]._storage[slot].zero_()
                self.assertTrue(audit_boundary(actual, expected, loaded)["expert_weights_exact"])

    def test_current_stream_order_and_pageable_reference_are_supported(self):
        original_stream = torch.cuda.current_stream()
        for pinned in (False, True):
            with self.subTest(pinned=pinned):
                actual, expected, loaded = fixture({0, 1, 3}, device="cuda:0", pinned=pinned)
                stream = torch.cuda.Stream()
                stream.wait_stream(original_stream)
                with torch.cuda.stream(stream):
                    loaded.expert_caches[0]._storage.zero_()
                    loaded.expert_caches[0]._storage.copy_(expected.expert_caches[0].weights,
                                                         non_blocking=True)
                    self.assertTrue(audit_boundary(actual, expected, loaded)["expert_weights_exact"])
                self.assertEqual(torch.cuda.current_stream(), original_stream)

    def test_real_cache_restore_preserves_metadata_and_checks_cold_storage(self):
        cache = GpuExpertCache({i: torch.full((8, 16), float(i), dtype=torch.bfloat16)
            for i in range(5)}, slot_count=4, device="cuda:0")
        self.addCleanup(cache.close)
        for expert in (0, 1, 2):
            with cache.acquire(expert) as lease:
                lease.wait()
        torch.cuda.synchronize()
        saved = cache._capture_checkpoint()
        actual = generation("cuda:0")
        expected = SimpleNamespace(generation=deepcopy(actual), expert_caches=(saved,))
        loaded = SimpleNamespace(expert_caches=(cache,))
        reference = saved.weights.view(torch.uint8).clone()
        for cold in (False, True):
            with self.subTest(cold=cold):
                cache._restore_checkpoint(saved)
                if cold:
                    cache.cold_reset()
                before = cache.snapshot()
                self.assertTrue(audit_boundary(actual, expected, loaded)["expert_weights_exact"])
                self.assertEqual(cache.snapshot(), before)
                cache._storage[0].view(torch.uint8).view(-1)[-1] ^= 1
                with self.assertRaisesRegex(RuntimeError, "expert slot weights differ bitwise"):
                    audit_boundary(actual, expected, loaded)
                self.assertEqual(cache.snapshot(), before)
                self.assertTrue(torch.equal(saved.weights.view(torch.uint8), reference))
        torch.cuda.synchronize()
