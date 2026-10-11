from types import SimpleNamespace
import unittest

import torch

from specfit import ExpertKey
from specfit.runtime import candidate_pool_fingerprint
from specfit.selection_refinement import (CacheCostModel, VARIANTS, resident_bitmap, select_refined)
from specfit.selection_refinement_runtime import RefinementSelector, live_resident_bitmap
from tests.gpu import test_qwen3_expert_prefetch as gpu_fixtures
from tests.unit import test_selection_refinement as cpu_fixtures


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class SelectionRefinementGpuTest(unittest.TestCase):
    def case(self):
        loaded, _, _ = gpu_fixtures.Qwen3ExpertPrefetchTest.make_case(self, slots=6, layers=48)
        pool, atlas = cpu_fixtures.SelectionRefinementTest.fixture()
        live_pool = SimpleNamespace(seed_token_id=pool.seed_token_id, top_k=2, max_depth=5,
                                    candidates=pool.nodes)
        self.assertEqual(candidate_pool_fingerprint(live_pool), pool.fingerprint)
        legacy = cpu_fixtures.SelectionRefinementTest.model()
        cache = CacheCostModel(legacy.size_intercept_ns, legacy.expert_ns, 500000.0)
        return loaded, pool, live_pool, atlas, legacy, cache

    def populate(self, loaded):
        for lid, cache in enumerate(loaded.expert_caches):
            with cache.acquire(ExpertKey(lid, lid % 6)) as lease:
                lease.wait()
        torch.cuda.synchronize()
        return tuple(cache.snapshot() for cache in loaded.expert_caches)

    def test_selector_reads_live_restored_residents_without_modifying_cache_state(self):
        loaded, pool, live, atlas, legacy, cache = self.case()
        self.assertEqual(live_resident_bitmap(loaded), 0)
        selectors = [RefinementSelector(method, loaded=loaded, atlas=atlas, model=legacy,
            cache_model=cache, acceptance_exponent=2.0, expected_pool=pool.fingerprint)
            for method in VARIANTS]
        before = self.populate(loaded)  # Populate after constructing selectors.
        mask = resident_bitmap(tuple((lid % 6,) for lid in range(48)))
        self.assertEqual(live_resident_bitmap(loaded), mask)
        for selector in selectors:
            expected = select_refined(pool, atlas, selector.method, legacy_model=legacy,
                cache_model=cache, resident_mask=mask, acceptance_exponent=2.0)
            self.assertEqual(selector.select_raw_ids(live, draft_node_budget=18), expected)
            self.assertEqual(selector.observed_resident_mask,
                             None if selector.method == "expert_fast_dynamic" else mask)
        self.assertEqual(tuple(c.snapshot() for c in loaded.expert_caches), before)

    def test_changed_cache_input_is_rejected_and_original_probability_never_reads_it(self):
        loaded, pool, live, atlas, legacy, cache = self.case()
        self.populate(loaded)
        selector = RefinementSelector("expert_cache_dynamic", loaded=loaded, atlas=atlas,
            model=legacy, cache_model=cache, expected_resident_mask=0, expected_pool=pool.fingerprint)
        with self.assertRaisesRegex(RuntimeError, "resident experts"):
            selector.select_raw_ids(live, draft_node_budget=18)
        with loaded.expert_caches[0].acquire(ExpertKey(0, 0)) as lease:
            lease.wait()
            with self.assertRaisesRegex(RuntimeError, "lease-free"):
                live_resident_bitmap(loaded)
        original = RefinementSelector("probability_dynamic", loaded=None, atlas=None,
                                     model=legacy, expected_pool=pool.fingerprint)
        expected = select_refined(pool, atlas, "probability_dynamic", legacy_model=legacy,
                                  cache_model=cache)
        self.assertEqual(original.select_raw_ids(live, draft_node_budget=18), expected)
        self.assertIsNone(original.observed_resident_mask)


if __name__ == "__main__":
    unittest.main()
