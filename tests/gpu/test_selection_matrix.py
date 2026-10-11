from hashlib import sha256
from types import SimpleNamespace
import unittest

import torch

from specfit.cross_layer_prefetch import replay_events
from specfit.selection_evaluation_runtime import observe_execution
from specfit.selection_matrix import EXECUTORS, SCORED_CONTROLS
from tests.gpu.test_cross_layer_prefetch_audit import logical
from tests.gpu.test_qwen3_expert_prefetch import Qwen3ExpertPrefetchTest


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class SelectionMatrixGpuTest(unittest.TestCase):
    def case(self):
        loaded, _, inputs = Qwen3ExpertPrefetchTest.make_case(self, slots=6)
        original = loaded.model.forward
        def forward(*args, **kwargs):
            return SimpleNamespace(logits=original(*args, **kwargs).unsqueeze(0))
        loaded.model.forward=forward
        for cache in loaded.expert_caches:
            cache._observer=None
        return loaded,inputs,forward

    def test_all_executors_record_full_bytes_and_reconstruct_real_cache(self):
        loaded, inputs, forward = self.case()
        for tokens in (1,4):
            selected = tuple(value[:tokens] for value in inputs)
            expected=None
            actual_experts=set(selected[1].cpu().flatten().tolist())
            for execution in (*EXECUTORS, *SCORED_CONTROLS):
                with self.subTest(tokens=tokens, execution=execution):
                    loaded.cold_reset_expert_caches()
                    pre=logical(loaded.expert_caches)
                    predictions={lid:sorted(actual_experts) for lid in range(2)} if execution in ("predicted_same_layer", *SCORED_CONTROLS) else None
                    with torch.inference_mode(), observe_execution(loaded,execution,predictions) as info:
                        output=loaded.model(*selected).logits
                        torch.cuda.synchronize()
                    raw=output.contiguous().view(torch.uint8).cpu().numpy().tobytes()
                    expected=raw if expected is None else expected
                    self.assertEqual(raw,expected)
                    self.assertEqual(info["logits_sha256"],sha256(raw).hexdigest())
                    self.assertEqual(replay_events(pre,info["cache_events"],
                        {(lid,e) for lid in range(2) for e in actual_experts},
                        "ready_demand" if execution in SCORED_CONTROLS else execution),logical(loaded.expert_caches))
                    self.assertIs(loaded.model.forward,forward)
                    self.assertTrue(all(cache._observer is None for cache in loaded.expert_caches))

    def test_target_failure_restores_forward_observers_and_scheduler(self):
        loaded,inputs,_=self.case()
        def failure(*args,**kwargs):
            raise RuntimeError("target fixture failed")
        loaded.model.forward=failure
        with self.assertRaisesRegex(RuntimeError,"target fixture failed"):
            with observe_execution(loaded,"ready_demand"):
                loaded.model(*inputs)
        self.assertIs(loaded.model.forward,failure)
        self.assertTrue(all(cache._observer is None for cache in loaded.expert_caches))
        self.assertTrue(all(layer.mlp.experts._prediction_scheduler is None for layer in loaded.model.model.layers))


if __name__ == "__main__":
    unittest.main()
