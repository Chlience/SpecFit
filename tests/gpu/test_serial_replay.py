from pathlib import Path
import sys
import unittest

import torch

from tests.gpu import test_qwen3_expert_prefetch as fixtures

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from run_qwen3_serial_replay import cache_boundary


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class SerialReplayGpuTest(unittest.TestCase):
    def test_boundary_read_preserves_completed_use_guards_and_counters(self):
        loaded, _, inputs = fixtures.Qwen3ExpertPrefetchTest.make_case(self, slots=6)
        loaded.cold_reset_expert_caches()
        self.assertEqual(cache_boundary(loaded)["occupied"], 0)
        with loaded.prefetch_experts({}, lookahead_layers=0, max_inflight=2):
            loaded.model(*inputs)
        torch.cuda.synchronize()
        before = loaded.cache_snapshots()
        first = cache_boundary(loaded)
        second = cache_boundary(loaded)
        self.assertEqual(first, second)
        self.assertEqual(loaded.cache_snapshots(), before)
        self.assertEqual(first["occupied"], 10)
        self.assertEqual(first["counters"]["h2d_submissions"], 10)
        # The legacy stats accessor does reap completed use guards. It cannot
        # be mixed with an earlier snapshot to represent one request boundary.
        loaded.cache_stats()
        self.assertNotEqual(loaded.cache_snapshots(), before)
        self.assertEqual(cache_boundary(loaded)["counters"], first["counters"])


if __name__ == "__main__":
    unittest.main()
