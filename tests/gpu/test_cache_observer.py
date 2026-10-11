from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from specfit.runtime import (
    GpuExpertCache,
    RecordingCacheTransitionObserver,
    RuntimeEventKind,
    aggregate_cache_transition_counters,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class CacheTransitionObserverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda:0")

    def make_sources(self) -> dict[int, torch.Tensor]:
        return {
            expert_id: torch.full(
                (32, 32),
                float(expert_id + 1),
                dtype=torch.bfloat16,
            )
            for expert_id in range(3)
        }

    def test_known_sequence_matches_runtime_counters(self) -> None:
        observer = RecordingCacheTransitionObserver(run_id="run-0")
        self.assertEqual(observer.cursor(), 0)
        cache = GpuExpertCache(
            self.make_sources(),
            slot_count=1,
            device=self.device,
            cache_id="layer-0",
            queue_id="layer-0-copy",
            observer=observer,
        )
        with torch.cuda.stream(cache.copy_stream):
            torch.cuda._sleep(10_000_000)

        first = cache.acquire(0)
        second = cache.acquire(0)
        first.wait()
        second.wait()
        second.release()
        first.release()
        torch.cuda.synchronize(self.device)

        with cache.acquire(0) as resident:
            resident.wait()
        with cache.acquire(1) as replacement:
            replacement.wait()
        torch.cuda.synchronize(self.device)

        events = observer.events()
        self.assertEqual(observer.cursor(), len(events))
        self.assertEqual(
            tuple(event.kind for event in events),
            (
                RuntimeEventKind.H2D_SUBMIT,
                RuntimeEventKind.INFLIGHT_JOIN,
                RuntimeEventKind.CONSUME_WAIT,
                RuntimeEventKind.CONSUME_WAIT,
                RuntimeEventKind.RELEASE,
                RuntimeEventKind.RELEASE,
                RuntimeEventKind.RESIDENT_HIT,
                RuntimeEventKind.CONSUME_WAIT,
                RuntimeEventKind.RELEASE,
                RuntimeEventKind.EVICT,
                RuntimeEventKind.H2D_SUBMIT,
                RuntimeEventKind.CONSUME_WAIT,
                RuntimeEventKind.RELEASE,
            ),
        )
        self.assertEqual(
            tuple(event.sequence for event in events),
            tuple(range(len(events))),
        )
        self.assertEqual(
            [event.submission_sequence for event in events
             if event.kind is RuntimeEventKind.H2D_SUBMIT],
            [0, 1],
        )
        submit_events = [
            event
            for event in events
            if event.kind is RuntimeEventKind.H2D_SUBMIT
        ]
        self.assertIs(submit_events[0].ready_event, first.ready_event)
        eviction = next(
            event
            for event in events
            if event.kind is RuntimeEventKind.EVICT
        )
        self.assertEqual(eviction.victim_expert_id, 0)
        self.assertEqual(eviction.expert_id, 1)
        self.assertTrue(
            all(event.cache_id == "layer-0" for event in events)
        )
        self.assertTrue(
            all(event.queue_id == "layer-0-copy" for event in events)
        )
        self.assertEqual(
            tuple(event.host_time_ns for event in events),
            tuple(sorted(event.host_time_ns for event in events)),
        )

        observed = aggregate_cache_transition_counters(events)
        runtime = cache.stats()
        self.assertEqual(
            (
                observed.h2d_submissions,
                observed.resident_hits,
                observed.inflight_joins,
                observed.evictions,
            ),
            (
                runtime.h2d_submissions,
                runtime.resident_hits,
                runtime.inflight_joins,
                runtime.evictions,
            ),
        )

    def test_observer_does_not_change_cache_result(self) -> None:
        observed = RecordingCacheTransitionObserver(run_id="run-0")
        caches = (
            GpuExpertCache(
                self.make_sources(),
                slot_count=2,
                device=self.device,
                cache_id="plain",
            ),
            GpuExpertCache(
                self.make_sources(),
                slot_count=2,
                device=self.device,
                cache_id="observed",
                observer=observed,
            ),
        )

        states = []
        outputs = []
        inputs = torch.eye(
            32,
            dtype=torch.bfloat16,
            device=self.device,
        )
        for cache in caches:
            cache_outputs = []
            for expert_id in (0, 1, 0, 2):
                with cache.acquire(expert_id) as lease:
                    cache_outputs.append(
                        F.linear(inputs, lease.wait()).clone()
                    )
            torch.cuda.synchronize(self.device)
            snapshot = cache.snapshot()
            outputs.append(cache_outputs)
            states.append(
                (
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
            )

        self.assertEqual(states[1], states[0])
        for plain, traced in zip(outputs[0], outputs[1]):
            self.assertTrue(torch.equal(traced, plain))
        self.assertGreater(len(observed.events()), 0)

    def test_two_slot_lru_event_names_the_actual_victim(self) -> None:
        observer = RecordingCacheTransitionObserver(run_id="run-0")
        cache = GpuExpertCache(
            self.make_sources(),
            slot_count=2,
            device=self.device,
            cache_id="layer-0",
            observer=observer,
        )

        for expert_id in (0, 1, 0, 2):
            with cache.acquire(expert_id) as lease:
                lease.wait()
            torch.cuda.synchronize(self.device)

        events = observer.events()
        evictions = tuple(
            event
            for event in events
            if event.kind is RuntimeEventKind.EVICT
        )
        self.assertEqual(len(evictions), 1)
        self.assertEqual(evictions[0].expert_id, 2)
        self.assertEqual(evictions[0].victim_expert_id, 1)
        self.assertEqual(cache.resident_expert_ids(), (0, 2))

    def test_one_observer_orders_multiple_layer_queues(self) -> None:
        observer = RecordingCacheTransitionObserver(run_id="run-0")
        caches = tuple(
            GpuExpertCache(
                self.make_sources(),
                slot_count=1,
                device=self.device,
                cache_id=f"layer-{layer_id}",
                queue_id=f"layer-{layer_id}-copy",
                observer=observer,
            )
            for layer_id in range(2)
        )

        for cache in caches:
            cache.acquire(0).release()
        events = observer.events()

        self.assertEqual(
            tuple(event.sequence for event in events),
            (0, 1, 2, 3),
        )
        self.assertEqual(
            tuple(event.cache_id for event in events),
            ("layer-0", "layer-0", "layer-1", "layer-1"),
        )
        self.assertEqual(
            tuple(
                event.submission_sequence
                for event in events
                if event.kind is RuntimeEventKind.H2D_SUBMIT
            ),
            (0, 0),
        )
        torch.cuda.synchronize(self.device)
        for cache in caches:
            cache.cold_reset()
        observer.reset(run_id="run-1")
        caches[0].acquire(1).release()
        restarted = observer.events()
        self.assertEqual(
            tuple(event.sequence for event in restarted),
            (0, 1),
        )
        self.assertTrue(
            all(event.run_id == "run-1" for event in restarted)
        )
        torch.cuda.synchronize(self.device)


if __name__ == "__main__":
    unittest.main()
