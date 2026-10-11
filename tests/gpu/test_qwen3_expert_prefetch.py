from __future__ import annotations

from contextlib import ExitStack
from types import MappingProxyType, SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import Qwen3MoeForCausalLM

from specfit import ExpertKey, RuntimeEventKind
from specfit.runtime import (GpuExpertCache, LoadedOffloadedQwen3, Qwen3OffloadedExperts,
    RecordingCacheTransitionObserver, load_offloaded_qwen3)
from specfit.runtime import qwen3_offloaded_experts as implementation
from tests.gpu.test_qwen3_offloaded_model import (_save_qwen3_style_checkpoint,
                                                  _two_layer_tiny_config)


class _Attention(nn.Module):
    def forward(self, hidden):
        return hidden.clone()


class _Model(nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()
        self.config = SimpleNamespace(num_hidden_layers=len(experts))
        for module in experts:
            layer = nn.Module()
            layer.self_attn = _Attention()
            layer.mlp = nn.Module()
            layer.mlp.experts = module
            self.model.layers.append(layer)

    def forward(self, hidden, indices, weights):
        for layer in self.model.layers:
            hidden = layer.self_attn(hidden)
            hidden = layer.mlp.experts(hidden, indices, weights)
        return hidden


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class Qwen3ExpertPrefetchTest(unittest.TestCase):
    def make_case(self, slots=3, layers=2):
        generator = torch.Generator().manual_seed(909)
        caches, experts = [], []
        observer = RecordingCacheTransitionObserver(run_id="predicted-prefetch")
        for lid in range(layers):
            sources = {ExpertKey(lid, e): torch.randn(3 * 16 * 12,
                generator=generator, dtype=torch.bfloat16) / 16 for e in range(6)}
            cache = GpuExpertCache(sources, slot_count=slots, device="cuda:0",
                cache_id=f"qwen3-layer-{lid}", queue_id=f"layer-{lid}-copy", observer=observer)
            caches.append(cache)
            self.addCleanup(cache.close)
            experts.append(Qwen3OffloadedExperts(layer_id=lid, cache=cache,
                hidden_size=16, intermediate_size=12, h2d_prefetch=False))
        loaded = LoadedOffloadedQwen3(_Model(experts), tuple(caches), None)
        inputs = (torch.randn(4, 16, generator=generator, dtype=torch.bfloat16).cuda(),
            torch.tensor([[4, 0, 2], [1, 4, 3], [2, 0, 1], [4, 3, 0]], device="cuda:0"),
            torch.rand(4, 3, generator=generator, dtype=torch.bfloat16).cuda())
        return loaded, observer, inputs

    def assert_restored(self, loaded, streams):
        torch.cuda.synchronize()
        for layer, cache, stream in zip(loaded.model.model.layers, loaded.expert_caches, streams):
            self.assertIs(cache.copy_stream, stream)
            self.assertIsNone(layer.mlp.experts._prediction_scheduler)
            snapshot = cache.snapshot()
            self.assertFalse(snapshot.pending_transfers)
            self.assertTrue(all(not s.active_leases and not s.pending_use_guards
                                for s in snapshot.slots))

    def test_bitwise_outputs_with_correct_missing_wrong_and_duplicate_predictions(self):
        for slots in (1, 2, 3, 6):
            loaded, _, inputs = self.make_case(slots)
            expected = loaded.model(*inputs)
            streams = tuple(c.copy_stream for c in loaded.expert_caches)
            for prediction in ((), (0, 1, 2, 3, 4), (5,), (4, 0, 4, 2, 5)):
                for depth in (0, 1):
                    with self.subTest(slots=slots, prediction=prediction, depth=depth):
                        loaded.cold_reset_expert_caches()
                        with loaded.prefetch_experts({i: prediction for i in range(2)},
                                lookahead_layers=depth, max_inflight=2) as scheduler:
                            actual = loaded.model(*inputs)
                        self.assertTrue(torch.equal(actual, expected))
                        self.assert_restored(loaded, streams)
                        stats = scheduler.stats()
                        self.assertTrue(stats["complete"])
                        self.assertLessEqual(stats["peak_inflight"], 2)
                        for lid, row in enumerate(stats["layers"]):
                            self.assertEqual(row["computed_experts"], 5)
                            self.assertEqual(row["actual_experts"], 5)
                            self.assertEqual(row["ready_at_router"] + row["inflight_at_router"]
                                             + row["missing_at_router"], 5)
                            self.assertEqual(row["prediction_copies"] + row["demand_copies"],
                                             loaded.expert_caches[lid].stats().h2d_submissions)

    def test_grouped_routes_preserve_noncontiguous_repeated_and_empty_inputs(self):
        for tokens in (0, 1, 4):
            for slots in (1, 6):
                with self.subTest(tokens=tokens, slots=slots):
                    loaded, _, (hidden, indices, weights) = self.make_case(slots)
                    indices[0, 2] = indices[0, 0]
                    inputs = tuple(t[:tokens].t().contiguous().t()
                                   for t in (hidden, indices, weights))
                    originals = tuple(t.clone() for t in inputs)
                    expected = loaded.model(*inputs)
                    streams = tuple(c.copy_stream for c in loaded.expert_caches)
                    for depth in (0, 1):
                        loaded.cold_reset_expert_caches()
                        with loaded.prefetch_experts({0: (0, 4, 5), 1: (0, 4, 5)},
                                lookahead_layers=depth, max_inflight=2):
                            actual = loaded.model(*inputs)
                        self.assertTrue(torch.equal(actual, expected))
                        for actual_input, original in zip(inputs, originals, strict=True):
                            self.assertTrue(torch.equal(actual_input, original))
                        self.assert_restored(loaded, streams)

    def test_real_misses_start_before_ready_experts_and_ready_work_precedes_wait(self):
        loaded, observer, inputs = self.make_case(slots=4, layers=1)
        cache = loaded.expert_caches[0]
        for e in (3, 4):
            with cache.acquire(ExpertKey(0, e)) as lease:
                lease.wait()
        torch.cuda.synchronize()
        with torch.cuda.stream(cache.copy_stream):
            torch.cuda._sleep(50_000_000)
        with loaded.prefetch_experts({}, trace=True) as scheduler:
            loaded.model(*inputs)
        events = scheduler.events()
        copies = [e for e in events if e["action"] == "copy"]
        computed = [e for e in events if e["action"] == "compute"]
        self.assertLess(copies[0]["sequence"], computed[0]["sequence"])
        self.assertEqual([e["expert_id"] for e in computed[:2]], [3, 4])
        self.assertTrue(all(not e["waited"] for e in computed[:2]))
        self.assertEqual({e["expert_id"] for e in computed}, set(range(5)))
        torch.cuda.synchronize()

    def test_id_predictions_snapshot_inputs_and_preserve_order_without_truncation(self):
        loaded, _, inputs = self.make_case(slots=3)
        expected = loaded.model(*inputs)
        streams = tuple(cache.copy_stream for cache in loaded.expert_caches)
        loaded.cold_reset_expert_caches()
        predictions = {layer: [4, 0, 4, 2, 1, 3] for layer in range(2)}
        scheduler = loaded.prefetch_experts(MappingProxyType(predictions),
            lookahead_layers=0, trace=True)
        self.assertEqual(scheduler.predictions, ((0, 1, 2, 3, 4),) * 2)
        for ids in predictions.values():
            ids[:] = [5]
        with scheduler:
            copies = [row["expert_id"] for row in scheduler.events()
                      if row["action"] == "copy"]
            self.assertEqual(copies, [0, 1])
            actual = loaded.model(*inputs)
        self.assertTrue(torch.equal(actual, expected))
        self.assert_restored(loaded, streams)

    def test_resident_only_forward_has_bounded_queries_and_no_copies(self):
        loaded, _, inputs = self.make_case(slots=6)
        expected = loaded.model(*inputs)
        torch.cuda.synchronize()
        for cache in loaded.expert_caches:
            self.assertEqual(len(cache.resident_expert_ids()), 5)
        before = [cache.stats().h2d_submissions for cache in loaded.expert_caches]
        queries = []
        original_query = torch.cuda.Event.query

        def query(event):
            queries.append(event)
            return original_query(event)

        with patch.object(torch.cuda.Event, "query", query):
            with loaded.prefetch_experts({0: tuple(range(5)), 1: tuple(range(5))}) as scheduler:
                actual = loaded.model(*inputs)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(actual, expected))
        # Allow one check per executed expert; a scan of every remaining
        # resident on every iteration exceeds this bound even in two tiny layers.
        self.assertLessEqual(len(queries), 10)
        self.assertEqual([cache.stats().h2d_submissions for cache in loaded.expert_caches], before)
        self.assertEqual([row["ready_at_router"] for row in scheduler.stats()["layers"]], [5, 5])

    def test_nonblocking_prefetch_stops_after_first_writable_lru_slot(self):
        loaded, _, _ = self.make_case(slots=4, layers=1)
        cache = loaded.expert_caches[0]
        for expert in range(4):
            with cache.acquire(ExpertKey(0, expert)) as lease:
                lease.wait()
        torch.cuda.synchronize()
        cache.resident_expert_ids()
        queries = []
        original_query = torch.cuda.Event.query

        def query(event):
            queries.append(event)
            return original_query(event)

        with patch.object(torch.cuda.Event, "query", query):
            lease = cache._prefetch_nonblocking(ExpertKey(0, 4))
        self.assertIsNotNone(lease)
        with lease:
            self.assertEqual(lease.slot_id, 0)
            actual = lease.wait().clone()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(actual.cpu(), cache._sources[ExpertKey(0, 4)]))
        self.assertLessEqual(len(queries), 1)

    def test_nonblocking_prefetch_skips_busy_lru_resident_and_preserves_readers(self):
        loaded, _, _ = self.make_case(slots=2, layers=1)
        cache = loaded.expert_caches[0]
        for expert in (0, 1):
            with cache.acquire(ExpertKey(0, expert)) as lease:
                lease.wait()
        torch.cuda.synchronize()
        cache.resident_expert_ids()
        use_stream = torch.cuda.Stream()
        with cache.acquire(ExpertKey(0, 0)) as lease:
            with torch.cuda.stream(use_stream):
                torch.cuda._sleep(100_000_000)
                preserved = lease.wait(use_stream).clone()
        # The busy slot is LRU; the later release makes slot 1 the next choice.
        with cache.acquire(ExpertKey(0, 1)) as lease:
            lease.wait()
        torch.cuda.current_stream().synchronize()
        lease = cache._prefetch_nonblocking(ExpertKey(0, 2))
        self.assertIsNotNone(lease)
        with lease:
            self.assertEqual(lease.slot_id, 1)
            actual = lease.wait().clone()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(preserved.cpu(), cache._sources[ExpertKey(0, 0)]))
        self.assertTrue(torch.equal(actual.cpu(), cache._sources[ExpertKey(0, 2)]))

    def test_attention_and_cross_layer_submission_are_observable(self):
        for depth in (0, 1):
            loaded, _, inputs = self.make_case(slots=6)
            for e in range(5):
                with loaded.expert_caches[0].acquire(ExpertKey(0, e)) as lease:
                    lease.wait()
            torch.cuda.synchronize()
            with loaded.prefetch_experts({0: list(range(5)), 1: list(range(5))},
                    lookahead_layers=depth, trace=True) as scheduler:
                loaded.model(*inputs)
            events = scheduler.events()
            attention = next(e["sequence"] for e in events
                             if e["action"] == "attention" and e["layer_id"] == 1)
            router = next(e["sequence"] for e in events
                          if e["action"] == "router" and e["layer_id"] == 1)
            copies = [e for e in events if e["action"] == "copy" and e["layer_id"] == 1]
            self.assertTrue(any(e["sequence"] < router for e in copies))
            self.assertEqual(any(e["sequence"] < attention for e in copies), bool(depth))
            torch.cuda.synchronize()

    def test_predictions_start_when_available_before_attention(self):
        loaded, _, inputs = self.make_case()
        with loaded.prefetch_experts({0: (0, 1)}, trace=True) as scheduler:
            self.assertGreater(loaded.expert_caches[0].stats().h2d_submissions, 0)
            self.assertFalse(any(e["action"] == "attention" for e in scheduler.events()))
            loaded.model(*inputs)
        torch.cuda.synchronize()

    def test_predicted_residents_survive_earlier_missing_predictions(self):
        for depth in (0, 1):
            for limit in (None, 1):
                with self.subTest(depth=depth, limit=limit):
                    loaded, observer, original = self.make_case(slots=3)
                    inputs = (original[0], torch.tensor([[0, 2]] * 4, device="cuda:0"),
                              original[2][:, :2])
                    expected = loaded.model(*inputs)
                    loaded.cold_reset_expert_caches()
                    # Expert 2 is needed and is the LRU resident. The earlier
                    # prediction 0 misses, so copying it before protecting 2
                    # would evict a weight that this same query still needs.
                    for lid, cache in enumerate(loaded.expert_caches):
                        for expert in (2, 1, 3):
                            with cache.acquire(ExpertKey(lid, expert)) as lease:
                                lease.wait()
                    torch.cuda.synchronize()
                    before = [c.stats().h2d_submissions for c in loaded.expert_caches]
                    event_start = len(observer.events())
                    streams = tuple(c.copy_stream for c in loaded.expert_caches)
                    with loaded.prefetch_experts({0: (0, 2), 1: (0, 2)},
                            lookahead_layers=depth, max_prefetch_per_layer=limit):
                        actual = loaded.model(*inputs)
                    self.assertTrue(torch.equal(actual, expected))
                    self.assert_restored(loaded, streams)
                    for lid, cache in enumerate(loaded.expert_caches):
                        self.assertEqual(cache.stats().h2d_submissions - before[lid], 1)
                    self.assertFalse(any(e.kind is RuntimeEventKind.EVICT
                        and e.victim_expert_id.expert_id == 2
                        for e in observer.events()[event_start:]))

    def test_entry_failure_releases_partially_acquired_predictions(self):
        loaded, _, inputs = self.make_case()
        streams = tuple(c.copy_stream for c in loaded.expert_caches)
        cache = loaded.expert_caches[0]
        with cache.acquire(ExpertKey(0, 0)) as lease:
            lease.wait()
        torch.cuda.synchronize()
        original = cache._prefetch_nonblocking
        def fail(key, **kwargs):
            if key.expert_id == 1:
                raise RuntimeError("injected entry failure")
            return original(key, **kwargs)
        with patch.object(cache, "_prefetch_nonblocking", fail):
            with self.assertRaisesRegex(RuntimeError, "injected entry failure"):
                with loaded.prefetch_experts({0: (0, 1)}):
                    pass
        self.assert_restored(loaded, streams)
        loaded.model(*inputs)
        torch.cuda.synchronize()

    def test_current_demand_precedes_new_future_predictions(self):
        loaded, _, inputs = self.make_case(slots=6)
        with loaded.prefetch_experts({1: (0, 1, 2, 3, 4)}, trace=True) as scheduler:
            loaded.model(*inputs)
        events = scheduler.events()
        router = next(e["sequence"] for e in events
                      if e["action"] == "router" and e["layer_id"] == 0)
        current = [e["sequence"] for e in events if e["action"] == "copy" and e["layer_id"] == 0]
        future = [e["sequence"] for e in events if e["action"] == "copy"
                  and e["layer_id"] == 1 and e["sequence"] > router]
        self.assertTrue(current)
        if future:
            self.assertLess(max(current), min(future))
        torch.cuda.synchronize()

    def test_speculation_never_queues_a_busy_slot_wait(self):
        loaded, _, _ = self.make_case(slots=1, layers=1)
        cache = loaded.expert_caches[0]
        use_stream = torch.cuda.Stream()
        with cache.acquire(ExpertKey(0, 0)) as lease:
            with torch.cuda.stream(use_stream):
                torch.cuda._sleep(50_000_000)
                preserved = lease.wait(use_stream).clone()
        before = cache.stats().h2d_submissions
        self.assertIsNone(cache._prefetch_nonblocking(ExpertKey(0, 1)))
        self.assertEqual(cache.stats().h2d_submissions, before)
        with cache._prefetch(ExpertKey(0, 1)) as lease:
            lease.wait()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(preserved.cpu(), cache._sources[ExpertKey(0, 0)]))

    def test_prior_inflight_prediction_is_joined_without_duplicate_copy(self):
        loaded, observer, inputs = self.make_case(slots=3, layers=1)
        cache = loaded.expert_caches[0]
        with torch.cuda.stream(cache.copy_stream):
            torch.cuda._sleep(50_000_000)
        cache._prefetch(ExpertKey(0, 0)).release()
        with loaded.prefetch_experts({0: (0, 1)}, trace=True):
            loaded.model(*inputs)
        submits = [e for e in observer.events() if e.kind is RuntimeEventKind.H2D_SUBMIT
                   and e.expert_id == ExpertKey(0, 0)]
        self.assertEqual(len(submits), 1)
        torch.cuda.synchronize()

    def test_exceptions_restore_streams_hooks_and_all_leases(self):
        for owner, name in ((torch, "sort"), (implementation, "_expert_forward")):
            loaded, _, inputs = self.make_case()
            streams = tuple(c.copy_stream for c in loaded.expert_caches)
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with loaded.prefetch_experts({0: (0, 1), 1: (0, 1)}) as scheduler:
                    with patch.object(owner, name, side_effect=RuntimeError("injected")):
                        loaded.model(*inputs)
            self.assertFalse(scheduler.stats()["complete"])
            self.assert_restored(loaded, streams)
            loaded.model(*inputs)
            torch.cuda.synchronize()

    def test_invalid_configuration_and_active_context_changes_fail_before_mutation(self):
        loaded, _, inputs = self.make_case()
        before = [cache.stats() for cache in loaded.expert_caches]
        for predictions, kwargs in (({2: (0,)}, {}), ({0: (6,)}, {}), ({0: (True,)}, {}),
                ({0: (0,), 1: (6,)}, {}), ({0: (-1,)}, {}), ({0: (0.,)}, {}),
                ({0: ("0",)}, {}),
                ({}, {"lookahead_layers": -1}), ({}, {"max_inflight": 0}),
                ({}, {"max_inflight": True}), ({}, {"max_prefetch_per_layer": -1})):
            with self.subTest(predictions=predictions, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    loaded.prefetch_experts(predictions, **kwargs)
                self.assertEqual([cache.stats() for cache in loaded.expert_caches], before)
                self.assertTrue(all(layer.mlp.experts._prediction_scheduler is None
                                    for layer in loaded.model.model.layers))
        with loaded.prefetch_experts({}) as scheduler:
            with self.assertRaisesRegex(RuntimeError, "context is active"):
                with loaded.prefetch_experts({}):
                    pass
            for operation in (lambda: loaded.set_expert_h2d_prefetch(False),
                    lambda: loaded.set_expert_execution_policy("resident_first"), loaded.close):
                with self.assertRaisesRegex(RuntimeError, "predicted prefetch"):
                    operation()
            loaded.model(*inputs)
        with self.assertRaisesRegex(RuntimeError, "reused"):
            with scheduler:
                pass
        torch.cuda.synchronize()

    def test_empty_or_repeated_forward_context_cannot_reuse_stale_predictions(self):
        loaded, _, inputs = self.make_case()
        streams = tuple(c.copy_stream for c in loaded.expert_caches)
        with self.assertRaisesRegex(RuntimeError, "complete target forward"):
            with loaded.prefetch_experts({}):
                pass
        with self.assertRaisesRegex(RuntimeError, "ordered eager"):
            with loaded.prefetch_experts({}):
                loaded.model(*inputs)
                loaded.model(*inputs)
        self.assert_restored(loaded, streams)

    def test_real_qwen3_attention_hooks_and_logits_match_legacy_paths(self):
        torch.manual_seed(909)
        source = Qwen3MoeForCausalLM(_two_layer_tiny_config()).to(torch.bfloat16)
        with tempfile.TemporaryDirectory() as root:
            _save_qwen3_style_checkpoint(source, root)
            with load_offloaded_qwen3(root, device="cuda:0", slots_per_layer=2,
                                      h2d_prefetch=False) as loaded:
                ids = torch.tensor([[1, 7, 11, 19]], device="cuda:0")
                predictions = {}
                def hook(lid):
                    def record(module, inputs, output):
                        predictions[lid] = torch.unique(output[2]).cpu().tolist()
                    return record
                with torch.inference_mode(), ExitStack() as stack:
                    for lid, layer in enumerate(loaded.model.model.layers):
                        stack.callback(layer.mlp.gate.register_forward_hook(hook(lid)).remove)
                    expected = loaded.model(ids, use_cache=False).logits
                for depth in (0, 1):
                    loaded.cold_reset_expert_caches()
                    with torch.inference_mode(), loaded.prefetch_experts(predictions,
                            lookahead_layers=depth) as scheduler:
                        actual = loaded.model(ids, use_cache=False).logits
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertTrue(scheduler.stats()["complete"])


if __name__ == "__main__":
    unittest.main()
