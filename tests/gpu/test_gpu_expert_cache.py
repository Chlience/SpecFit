from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from specfit.runtime import GpuExpertCache, NoEvictableSlotError


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class GpuExpertCacheSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda:0")
        if torch.cuda.get_device_capability(cls.device) < (8, 0):
            raise unittest.SkipTest("BF16 requires compute capability 8.0+")

    def make_sources(self) -> dict[int, torch.Tensor]:
        base = torch.arange(64 * 96, dtype=torch.float32).reshape(64, 96)
        return {
            expert_id: ((base.remainder(29) - 14) / (expert_id + 17)).to(
                torch.bfloat16
            )
            for expert_id in range(3)
        }

    def test_one_slot_survives_102_switches_and_matches_reference(self) -> None:
        sources = self.make_sources()
        inputs = torch.linspace(
            -1.0,
            1.0,
            steps=4 * 96,
            device=self.device,
            dtype=torch.bfloat16,
        ).reshape(4, 96)
        expected = {
            expert_id: F.linear(inputs, source.to(self.device))
            for expert_id, source in sources.items()
        }
        cache = GpuExpertCache(sources, slot_count=1, device=self.device)

        self.assertTrue(cache.sources_are_pinned)
        self.assertNotEqual(
            cache.copy_stream.cuda_stream,
            torch.cuda.default_stream(self.device).cuda_stream,
        )

        actual: list[tuple[int, torch.Tensor]] = []
        switch_count = 102
        for iteration in range(switch_count):
            expert_id = iteration % 3
            with cache.acquire(expert_id) as lease:
                output = F.linear(inputs, lease.wait()).clone()
                actual.append((expert_id, output))

        torch.cuda.synchronize(self.device)
        for expert_id, output in actual:
            torch.testing.assert_close(
                output,
                expected[expert_id],
                rtol=1.6e-2,
                atol=1.6e-2,
            )

        stats = cache.stats()
        self.assertEqual(stats.h2d_submissions, switch_count)
        self.assertEqual(stats.evictions, switch_count - 1)
        self.assertEqual(cache.resident_expert_ids(), (2,))

    def test_initial_copy_preserves_pending_readers_of_recycled_storage(self) -> None:
        reused = 0
        for attempt in range(8):
            with self.subTest(attempt=attempt):
                previous = torch.full((1, 64, 96), 17, dtype=torch.bfloat16, device=self.device)
                pointer = previous.data_ptr()
                torch.cuda._sleep(100_000_000)
                observed = previous.clone()
                del previous
                cache = GpuExpertCache({0: torch.full((64, 96), 3, dtype=torch.bfloat16)},
                                       slot_count=1, device=self.device)
                try:
                    reused += int(cache._slots[0].tensor.data_ptr() == pointer)
                    with cache.acquire(0) as lease:
                        copied = lease.wait().clone()
                    torch.cuda.synchronize(self.device)
                    self.assertTrue(bool(torch.all(observed == 17)))
                    self.assertTrue(bool(torch.all(copied == 3)))
                finally:
                    cache.close()
        if not reused:
            self.skipTest("allocator did not reuse the pending allocation")

    def test_two_acquires_join_one_inflight_copy(self) -> None:
        cache = GpuExpertCache(
            self.make_sources(),
            slot_count=1,
            device=self.device,
        )
        with torch.cuda.stream(cache.copy_stream):
            torch.cuda._sleep(10_000_000)

        first = cache.acquire(0)
        second = cache.acquire(0)
        try:
            self.assertEqual(cache.stats().h2d_submissions, 1)
            self.assertEqual(cache.stats().inflight_joins, 1)
            self.assertEqual(first.slot_id, second.slot_id)
            self.assertIs(first.ready_event, second.ready_event)
            self.assertEqual(first.wait().data_ptr(), second.wait().data_ptr())
        finally:
            second.release()
            first.release()

        torch.cuda.synchronize(self.device)
        with cache.acquire(0) as resident:
            resident.wait()
        self.assertEqual(cache.stats().resident_hits, 1)
        self.assertEqual(cache.stats().inflight_joins, 1)

    def test_active_lease_protects_lru_slot_from_reuse(self) -> None:
        cache = GpuExpertCache(
            self.make_sources(),
            slot_count=1,
            device=self.device,
        )

        lease = cache.acquire(0)
        try:
            with self.assertRaises(NoEvictableSlotError):
                cache.acquire(1)
        finally:
            lease.release()

        with cache.acquire(1) as next_lease:
            next_lease.wait()
        torch.cuda.synchronize(self.device)
        self.assertEqual(cache.resident_expert_ids(), (1,))

    def test_lru_reuses_the_least_recently_released_slot(self) -> None:
        cache = GpuExpertCache(
            self.make_sources(),
            slot_count=2,
            device=self.device,
        )

        for expert_id in (0, 1, 0, 2):
            with cache.acquire(expert_id) as lease:
                lease.wait()

        torch.cuda.synchronize(self.device)
        self.assertEqual(cache.resident_expert_ids(), (0, 2))
        self.assertEqual(cache.stats().evictions, 1)

    def test_completed_hot_hits_leave_no_pending_use_guards(self) -> None:
        cache = GpuExpertCache(
            self.make_sources(),
            slot_count=1,
            device=self.device,
        )

        for _ in range(64):
            with cache.acquire(0) as lease:
                F.linear(
                    torch.ones(
                        1,
                        96,
                        dtype=torch.bfloat16,
                        device=self.device,
                    ),
                    lease.wait(),
                )

        torch.cuda.synchronize(self.device)
        self.assertEqual(cache.stats().pending_use_guards, 0)

    def test_snapshot_tracks_cold_pending_resident_and_eviction(self) -> None:
        cache = GpuExpertCache(
            self.make_sources(),
            slot_count=1,
            device=self.device,
            cache_id="test-layer-0",
        )

        cold = cache.snapshot()
        self.assertEqual(cold.cache_id, "test-layer-0")
        self.assertEqual(cold.queue_id, "test-layer-0-copy")
        self.assertEqual(cold.capacity, 1)
        self.assertEqual(cold.host_state_version, 0)
        self.assertEqual(cold.lru_sequence, 0)
        self.assertEqual(cold.submission_sequence, 0)
        self.assertEqual(cold.pending_transfers, ())
        self.assertIsNone(cold.slots[0].expert_id)
        self.assertEqual(cold.stats.h2d_submissions, 0)

        with torch.cuda.stream(cache.copy_stream):
            torch.cuda._sleep(10_000_000)
        first = cache.acquire(0)
        second = cache.acquire(0)
        pending = cache.snapshot()
        self.assertEqual(pending.slots[0].expert_id, 0)
        self.assertEqual(pending.slots[0].active_leases, 2)
        self.assertEqual(len(pending.pending_transfers), 1)
        transfer = pending.pending_transfers[0]
        self.assertEqual(transfer.expert_id, 0)
        self.assertEqual(transfer.slot_id, 0)
        self.assertEqual(transfer.generation, 1)
        self.assertEqual(transfer.submission_sequence, 0)
        self.assertIs(transfer.ready_event, first.ready_event)
        self.assertEqual(
            pending.slots[0].ready,
            transfer.ready_event_completed,
        )
        self.assertEqual(pending.stats.h2d_submissions, 1)
        self.assertEqual(pending.stats.inflight_joins, 1)

        unchanged = cache.snapshot()
        self.assertEqual(
            unchanged.host_state_version,
            pending.host_state_version,
        )
        self.assertEqual(unchanged.lru_sequence, pending.lru_sequence)
        self.assertEqual(unchanged.submission_sequence, 1)
        self.assertEqual(unchanged.stats, pending.stats)

        second.release()
        first.release()
        torch.cuda.synchronize(self.device)
        with cache.acquire(0) as resident:
            resident.wait()
        torch.cuda.synchronize(self.device)
        resident_snapshot = cache.snapshot()
        self.assertEqual(resident_snapshot.pending_transfers, ())
        self.assertTrue(resident_snapshot.slots[0].ready)
        self.assertEqual(resident_snapshot.stats.resident_hits, 1)

        with cache.acquire(1) as replacement:
            replacement.wait()
        torch.cuda.synchronize(self.device)
        evicted = cache.snapshot()
        self.assertEqual(evicted.slots[0].expert_id, 1)
        self.assertEqual(evicted.slots[0].generation, 2)
        self.assertEqual(evicted.submission_sequence, 2)
        self.assertEqual(evicted.stats.evictions, 1)

    def test_cold_reset_requires_quiescence_and_restores_initial_state(
        self,
    ) -> None:
        cache = GpuExpertCache(
            self.make_sources(),
            slot_count=1,
            device=self.device,
            cache_id="reset-layer",
        )
        with torch.cuda.stream(cache.copy_stream):
            torch.cuda._sleep(10_000_000)
        lease = cache.acquire(0)

        with self.assertRaisesRegex(RuntimeError, "active lease"):
            cache.cold_reset()
        lease.release()
        with self.assertRaisesRegex(RuntimeError, "CUDA synchronization"):
            cache.cold_reset()

        torch.cuda.synchronize(self.device)
        cache.cold_reset()
        cold = cache.snapshot()
        self.assertEqual(cold.host_state_version, 0)
        self.assertEqual(cold.lru_sequence, 0)
        self.assertEqual(cold.submission_sequence, 0)
        self.assertEqual(cold.pending_transfers, ())
        self.assertEqual(cold.stats.h2d_submissions, 0)
        self.assertEqual(cold.stats.inflight_joins, 0)
        self.assertEqual(cold.stats.evictions, 0)
        self.assertIsNone(cold.slots[0].expert_id)
        self.assertEqual(cold.slots[0].generation, 0)

        with cache.acquire(0) as fresh:
            fresh.wait()
        after_reset = cache.snapshot()
        self.assertEqual(after_reset.slots[0].generation, 1)
        self.assertEqual(after_reset.submission_sequence, 1)
        self.assertEqual(after_reset.stats.h2d_submissions, 1)

    def test_snapshot_preserves_reused_slot_submission_queue(self) -> None:
        cache = GpuExpertCache(
            self.make_sources(),
            slot_count=1,
            device=self.device,
            cache_id="queue-layer",
        )
        with torch.cuda.stream(cache.copy_stream):
            torch.cuda._sleep(10_000_000)

        for expert_id in (0, 1, 0):
            cache.acquire(expert_id).release()
        queued = cache.snapshot()

        self.assertEqual(
            tuple(
                transfer.submission_sequence
                for transfer in queued.pending_transfers
            ),
            (0, 1, 2),
        )
        self.assertEqual(
            tuple(
                transfer.expert_id
                for transfer in queued.pending_transfers
            ),
            (0, 1, 0),
        )
        self.assertEqual(
            tuple(
                transfer.generation
                for transfer in queued.pending_transfers
            ),
            (1, 2, 3),
        )
        self.assertTrue(
            all(
                transfer.slot_id == 0
                for transfer in queued.pending_transfers
            )
        )
        self.assertEqual(queued.slots[0].expert_id, 0)
        self.assertEqual(queued.slots[0].generation, 3)
        self.assertEqual(queued.stats.h2d_submissions, 3)
        self.assertEqual(queued.stats.evictions, 2)

        torch.cuda.synchronize(self.device)
        with cache.acquire(0) as resident:
            resident.wait()
        torch.cuda.synchronize(self.device)
        reaped = cache.snapshot()
        self.assertEqual(reaped.pending_transfers, ())
        self.assertEqual(reaped.slots[0].expert_id, 0)
        self.assertEqual(reaped.slots[0].generation, 3)
        self.assertEqual(reaped.stats.h2d_submissions, 3)
        self.assertEqual(reaped.stats.resident_hits, 1)
        cache.cold_reset()

    def test_snapshot_observes_use_guard_without_reaping_it(self) -> None:
        cache = GpuExpertCache(
            self.make_sources(),
            slot_count=1,
            device=self.device,
            cache_id="guard-layer",
        )
        torch.cuda.synchronize(self.device)
        compute_stream = torch.cuda.Stream(device=self.device)
        lease = cache.acquire(0)
        with torch.cuda.stream(compute_stream):
            lease.wait(compute_stream)
            torch.cuda._sleep(10_000_000)
        lease.release()

        guarded = cache.snapshot()
        repeated = cache.snapshot()
        self.assertEqual(guarded.slots[0].pending_use_guards, 1)
        self.assertEqual(guarded.stats.pending_use_guards, 1)
        self.assertEqual(repeated.slots[0].pending_use_guards, 1)
        self.assertEqual(
            repeated.host_state_version,
            guarded.host_state_version,
        )
        with self.assertRaisesRegex(RuntimeError, "CUDA synchronization"):
            cache.cold_reset()

        torch.cuda.synchronize(self.device)
        completed = cache.snapshot()
        self.assertEqual(completed.slots[0].pending_use_guards, 0)
        self.assertEqual(completed.stats.pending_use_guards, 0)
        self.assertEqual(
            completed.host_state_version,
            guarded.host_state_version,
        )
        cache.cold_reset()

    def test_cold_reset_replays_the_same_lru_state(self) -> None:
        cache = GpuExpertCache(
            self.make_sources(),
            slot_count=2,
            device=self.device,
            cache_id="replay-layer",
        )

        def run_accesses() -> tuple[object, ...]:
            for expert_id in (0, 1, 0, 2):
                with cache.acquire(expert_id) as lease:
                    lease.wait()
            torch.cuda.synchronize(self.device)
            snapshot = cache.snapshot()
            return (
                tuple(
                    (
                        slot.slot_id,
                        slot.expert_id,
                        slot.generation,
                        slot.last_used_sequence,
                    )
                    for slot in snapshot.slots
                ),
                snapshot.lru_sequence,
                snapshot.submission_sequence,
                snapshot.stats,
            )

        first = run_accesses()
        cache.cold_reset()
        second = run_accesses()

        self.assertEqual(second, first)
        torch.cuda.synchronize(self.device)
        cache.cold_reset()

    def test_close_rejects_active_lease_and_releases_source_owner(
        self,
    ) -> None:
        class SourceOwner:
            def __init__(self) -> None:
                self.close_calls = 0
                self.ready_event: torch.cuda.Event | None = None

            def close(self) -> None:
                if self.ready_event is None or not self.ready_event.query():
                    raise AssertionError(
                        "source owner closed before pending H2D completed"
                    )
                self.close_calls += 1

        owner = SourceOwner()
        sources = {
            expert_id: source.pin_memory()
            for expert_id, source in self.make_sources().items()
        }
        cache = GpuExpertCache(
            sources,
            slot_count=1,
            device=self.device,
            adopt_pinned_sources=True,
            source_owner=owner,
        )
        with torch.cuda.stream(cache.copy_stream):
            torch.cuda._sleep(10_000_000)
        lease = cache.acquire(0)
        owner.ready_event = lease.ready_event

        with self.assertRaisesRegex(RuntimeError, "active lease"):
            cache.close()
        self.assertFalse(cache.closed)
        self.assertEqual(owner.close_calls, 0)

        lease.release()
        cache.close()
        cache.close()

        self.assertTrue(cache.closed)
        self.assertEqual(owner.close_calls, 1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            cache.acquire(0)

    def test_adopt_unpinned_sources_preserves_copy_fallback(self) -> None:
        sources = self.make_sources()
        self.assertTrue(
            all(not source.is_pinned() for source in sources.values())
        )

        cache = GpuExpertCache(
            sources,
            slot_count=1,
            device=self.device,
            adopt_pinned_sources=True,
        )

        self.assertTrue(cache.sources_are_pinned)
        with cache.acquire(0) as lease:
            expert = lease.wait()
            self.assertTrue(torch.equal(expert.cpu(), sources[0]))
        cache.close()


if __name__ == "__main__":
    unittest.main()
