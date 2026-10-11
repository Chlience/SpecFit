from contextlib import nullcontext
from dataclasses import asdict
import unittest

import torch

from specfit import ExpertKey
from specfit.cross_layer_prefetch import CompactTransitions, replay_events, VARIANTS
from tests.gpu import test_qwen3_expert_prefetch as cases


def logical(caches):
    layers = []
    for lid, cache in enumerate(caches):
        snapshot = cache.snapshot()
        slots = []
        for slot in snapshot.slots:
            row = asdict(slot)
            row["expert_id"] = None if slot.expert_id is None else slot.expert_id.expert_id
            slots.append(row)
        layers.append(dict(layer_id=lid, capacity=snapshot.capacity,
            lru_sequence=snapshot.lru_sequence, submission_sequence=snapshot.submission_sequence,
            counters=asdict(snapshot.stats), slots=slots))
    return dict(layers=layers)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class PrefetchAuditGpuTest(unittest.TestCase):
    def test_observed_events_reconstruct_real_cache_for_all_variants(self):
        for slots in (2, 6):
            loaded, _, inputs = cases.Qwen3ExpertPrefetchTest.make_case(self, slots=slots)
            for variant in VARIANTS:
                for natural in (False, True):
                    with self.subTest(slots=slots, variant=variant, natural=natural):
                        loaded.cold_reset_expert_caches()
                        if natural:
                            for lid, cache in enumerate(loaded.expert_caches):
                                for e in (0, 5):
                                    with cache.acquire(ExpertKey(lid, e)) as lease:
                                        lease.wait()
                        torch.cuda.synchronize()
                        pre = logical(loaded.expert_caches)
                        observer = CompactTransitions()
                        for cache in loaded.expert_caches:
                            cache._observer = observer
                        loaded.set_expert_h2d_prefetch(variant == "one_ahead")
                        predictions = ({} if variant == "ready_demand" else
                            {lid: [5] if variant == "wrong_cross_layer" else list(range(5)) for lid in range(2)})
                        context = (nullcontext() if variant in VARIANTS[:2] else
                            loaded.prefetch_experts(predictions, lookahead_layers=int(variant in VARIANTS[4:])))
                        with torch.inference_mode(), context:
                            loaded.model(*inputs)
                        torch.cuda.synchronize()
                        actual = {(lid, e) for lid in range(2) for e in range(5)}
                        self.assertEqual(replay_events(pre, observer.events, actual, variant),
                                         logical(loaded.expert_caches))
