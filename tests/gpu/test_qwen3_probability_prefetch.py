from __future__ import annotations

from contextlib import ExitStack
from fractions import Fraction
import tempfile
from types import MappingProxyType
import unittest

import torch
from transformers import Qwen3MoeForCausalLM

from specfit import ExpertKey, RuntimeEventKind
from specfit.runtime import load_offloaded_qwen3
from tests.gpu import test_qwen3_expert_prefetch as fixtures


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class Qwen3ProbabilityPrefetchTest(unittest.TestCase):
    make_case = fixtures.Qwen3ExpertPrefetchTest.make_case
    assert_restored = fixtures.Qwen3ExpertPrefetchTest.assert_restored

    def fill(self, cache, experts):
        for expert in experts:
            with cache.acquire(ExpertKey(0, expert)) as lease:
                lease.wait()
        torch.cuda.synchronize()

    def test_probabilities_rank_and_limit_attention_prefetch_for_both_policies(self):
        loaded, _, inputs = self.make_case(slots=3, layers=1)
        expected = loaded.model(*inputs)
        for policy in ("lru", "probability"):
            with self.subTest(policy=policy):
                loaded.cold_reset_expert_caches()
                with loaded.prefetch_experts({0: (0, 1, 2, 3, 4)},
                        prediction_probabilities={0: {0: .7, 1: .8, 2: 0., 3: .1, 4: .9}},
                        eviction_policy=policy, lookahead_layers=0, trace=True) as scheduler:
                    copies = [row["expert_id"] for row in scheduler.events() if row["action"] == "copy"]
                    self.assertEqual(copies, [4, 1])
                    actual = loaded.model(*inputs)
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(actual, expected))
                self.assertEqual(scheduler.stats()["layers"][0]["predicted_experts"], 2)
                self.assertLessEqual(scheduler.stats()["peak_inflight"], 2)

    def test_probability_eviction_chooses_lower_probability_instead_of_lru(self):
        loaded, observer, inputs = self.make_case(slots=3, layers=1)
        cache = loaded.expert_caches[0]
        expected = loaded.model(*inputs)
        for policy, victim in (("lru", 0), ("probability", 1)):
            with self.subTest(policy=policy):
                loaded.cold_reset_expert_caches()
                self.fill(cache, (0, 1, 2))
                begin = len(observer.events())
                with loaded.prefetch_experts({0: (4,)},
                        prediction_probabilities={0: {0: .6, 1: .1, 2: .4, 4: .9}},
                        eviction_policy=policy, lookahead_layers=0):
                    evictions = [row.victim_expert_id.expert_id for row in observer.events()[begin:]
                                 if row.kind is RuntimeEventKind.EVICT]
                    self.assertEqual(evictions, [victim])
                    actual = loaded.model(*inputs)
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(actual, expected))

    def test_equal_or_lower_probability_skips_speculation_but_real_demand_progresses(self):
        loaded, _, inputs = self.make_case(slots=3, layers=1)
        cache = loaded.expert_caches[0]
        expected = loaded.model(*inputs)
        for incoming in (.5, .4):
            with self.subTest(incoming=incoming):
                loaded.cold_reset_expert_caches()
                self.fill(cache, (0, 1, 2))
                before = cache.stats().h2d_submissions
                streams = (cache.copy_stream,)
                with loaded.prefetch_experts({0: (4,)},
                        prediction_probabilities={0: {0: .8, 1: .6, 2: .5, 4: incoming}},
                        eviction_policy="probability", lookahead_layers=0) as scheduler:
                    self.assertEqual(cache.stats().h2d_submissions, before)
                    actual = loaded.model(*inputs)
                self.assert_restored(loaded, streams)
                self.assertTrue(torch.equal(actual, expected))
                self.assertEqual(scheduler.stats()["layers"][0]["prediction_copies"], 0)
                self.assertEqual(scheduler.stats()["layers"][0]["demand_copies"], 2)

    def test_probability_eviction_skips_busy_and_leased_victims(self):
        loaded, _, _ = self.make_case(slots=3, layers=1)
        cache = loaded.expert_caches[0]
        probabilities = {ExpertKey(0, e): p for e, p in ((0, .2), (1, .1), (2, .3), (4, .9))}
        for protected in ("leased", "busy"):
            with self.subTest(protected=protected):
                loaded.cold_reset_expert_caches()
                self.fill(cache, (0, 1, 2))
                use_stream = torch.cuda.Stream()
                held = cache.acquire(ExpertKey(0, 1))
                try:
                    with torch.cuda.stream(use_stream):
                        torch.cuda._sleep(100_000_000)
                        preserved = held.wait(use_stream).clone()
                    if protected == "busy":
                        held.release()
                    copied = cache._prefetch_nonblocking(ExpertKey(0, 4),
                        eviction_probabilities=probabilities)
                    self.assertIsNotNone(copied)
                    with copied:
                        self.assertEqual(copied.slot_id, 0)
                        actual = copied.wait().clone()
                finally:
                    held.release()
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(preserved.cpu(), cache._sources[ExpertKey(0, 1)]))
                self.assertTrue(torch.equal(actual.cpu(), cache._sources[ExpertKey(0, 4)]))

    def test_bad_probabilities_and_policy_fail_before_cache_mutation(self):
        loaded, _, _ = self.make_case(slots=3, layers=1)
        cache = loaded.expert_caches[0]
        cases = [dict(eviction_policy="unknown"), dict(eviction_policy="probability"),
                 dict(prediction_probabilities=[]), dict(prediction_probabilities={0: []}),
                 dict(prediction_probabilities={1: {0: .5}}),
                 dict(prediction_probabilities={0: {6: .5}}),
                 dict(prediction_probabilities={0: {0: .5, 6: 0.}}),
                 dict(prediction_probabilities={0: {True: .5}}),
                 dict(prediction_probabilities={0: {1: .5}})]
        cases += [dict(prediction_probabilities={0: {0: p}})
                  for p in (float("nan"), float("inf"), -.1, 1.1, True, "0.5")]
        before = cache.stats()
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((ValueError, TypeError)):
                    loaded.prefetch_experts({0: (0,)}, **kwargs)
                self.assertEqual(cache.stats(), before)
                self.assertIsNone(loaded.model.model.layers[0].mlp.experts._prediction_scheduler)
        for expert in (True, -1, 6, "0", 0.):
            with self.subTest(predicted_expert=expert):
                with self.assertRaises((ValueError, TypeError)):
                    loaded.prefetch_experts({0: (expert,)},
                        prediction_probabilities={0: {0: .5}})
                self.assertEqual(cache.stats(), before)
                self.assertIsNone(loaded.model.model.layers[0].mlp.experts._prediction_scheduler)

    def test_scored_context_snapshots_inputs_and_preserves_ties_and_non_candidates(self):
        loaded, _, inputs = self.make_case(slots=3, layers=2)
        expected = loaded.model(*inputs)
        streams = tuple(cache.copy_stream for cache in loaded.expert_caches)
        for policy in ("lru", "probability"):
            with self.subTest(policy=policy):
                loaded.cold_reset_expert_caches()
                predictions = {lid: [4, 0, 4, 1, 2] for lid in range(2)}
                scores = {lid: {0: Fraction(1, 2), 1: Fraction(1, 2), 2: 0., 4: .9, 5: 1.}
                          for lid in range(2)}
                scheduler = loaded.prefetch_experts(MappingProxyType(predictions),
                    prediction_probabilities=MappingProxyType({
                        lid: MappingProxyType(row) for lid, row in scores.items()}),
                    eviction_policy=policy, lookahead_layers=0, trace=True)
                self.assertEqual(scheduler.predictions, ((4, 0), (4, 0)))
                for lid in range(2):
                    predictions[lid][:] = [5]
                    scores[lid].clear()
                with scheduler:
                    copies = [row["expert_id"] for row in scheduler.events()
                              if row["action"] == "copy"]
                    self.assertEqual(copies, [4, 0])
                    actual = loaded.model(*inputs)
                self.assertTrue(torch.equal(actual, expected))
                self.assert_restored(loaded, streams)

    def test_real_qwen3_scored_predictions_preserve_complete_logits(self):
        torch.manual_seed(909)
        source = Qwen3MoeForCausalLM(fixtures._two_layer_tiny_config()).to(torch.bfloat16)
        with tempfile.TemporaryDirectory() as root:
            fixtures._save_qwen3_style_checkpoint(source, root)
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
                probabilities = {lid: {expert: 1 / (expert + 1) for expert in experts}
                                 for lid, experts in predictions.items()}
                for policy in ("lru", "probability"):
                    for depth in (0, 1):
                        with self.subTest(policy=policy, depth=depth):
                            loaded.cold_reset_expert_caches()
                            with torch.inference_mode(), loaded.prefetch_experts(predictions,
                                    prediction_probabilities=probabilities, eviction_policy=policy,
                                    lookahead_layers=depth) as scheduler:
                                actual = loaded.model(ids, use_cache=False).logits
                            self.assertTrue(torch.equal(actual, expected))
                            self.assertTrue(scheduler.stats()["complete"])


if __name__ == "__main__":
    unittest.main()
