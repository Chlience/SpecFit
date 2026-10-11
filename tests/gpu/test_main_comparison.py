import unittest
import torch

from specfit import ExpertKey
from specfit.main_comparison_runtime import discard_completed_experts
from tests.gpu import test_qwen3_expert_prefetch as fixture


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class MainComparisonGpuTest(unittest.TestCase):
    def make_case(self):
        loaded, _, inputs = fixture.Qwen3ExpertPrefetchTest.make_case(self, slots=6)
        for cache in loaded.expert_caches: cache._observer = None
        return loaded, inputs

    def query(self, loaded, inputs):
        with loaded.prefetch_experts({}, lookahead_layers=0, max_inflight=2):
            output = loaded.model(*inputs)
        torch.cuda.synchronize()
        return output

    def test_disabled_cache_retransfers_union_without_changing_batch_output_or_allocation(self):
        loaded, inputs = self.make_case()
        pointers = [c._storage.data_ptr() for c in loaded.expert_caches]
        original = self.query(loaded, inputs)
        self.assertEqual(sum(c._stats.h2d_submissions for c in loaded.expert_caches), 10)
        cached = self.query(loaded, inputs)
        self.assertEqual(sum(c._stats.h2d_submissions for c in loaded.expert_caches), 10)
        for expected in (20, 30):
            before = [c.snapshot() for c in loaded.expert_caches]
            discard_completed_experts(loaded)
            after = [c.snapshot() for c in loaded.expert_caches]
            self.assertTrue(all(all(s.expert_id is None for s in x.slots) for x in after))
            self.assertEqual([x.stats for x in before], [x.stats for x in after])
            actual = self.query(loaded, inputs)
            self.assertTrue(torch.equal(actual.view(torch.uint8), original.view(torch.uint8)))
            self.assertEqual(sum(c._stats.h2d_submissions for c in loaded.expert_caches), expected)
        self.assertTrue(torch.equal(cached, original))
        self.assertEqual([c._storage.data_ptr() for c in loaded.expert_caches], pointers)

    def test_active_lease_rejection_is_atomic_across_layers(self):
        loaded, inputs = self.make_case()
        self.query(loaded, inputs)
        with loaded.expert_caches[1].acquire(ExpertKey(1, 0)) as lease:
            lease.wait(); torch.cuda.synchronize()
            before = loaded.cache_snapshots()
            with self.assertRaisesRegex(RuntimeError, 'active leases'): discard_completed_experts(loaded)
            self.assertEqual(loaded.cache_snapshots(), before)
        discard_completed_experts(loaded)

    def test_observer_boundary_is_not_silently_bypassed(self):
        loaded, _, inputs = fixture.Qwen3ExpertPrefetchTest.make_case(self, slots=6)
        self.query(loaded, inputs)
        with self.assertRaisesRegex(RuntimeError, 'observer'): discard_completed_experts(loaded)


if __name__ == '__main__': unittest.main()
