"""Opt-in length boundaries, real Qwen attention and committed-state audits."""
from contextvars import Context
import unittest

import torch

from specfit.runtime import (
    Eagle3ChainDrafter, Eagle3DraftTree, GreedyChainSpeculator,
    ShallowTreeSpeculator, target_greedy_generate,
)
from specfit.runtime.eagle3_chain import (
    EAGLE3_MAX_COMMITTED_TOKENS, _committed_token_limit,
    _experimental_2048_token_budget,
)
from specfit.runtime.eagle3_warmup import (
    eagle3_tree_temporary_kv_tokens, eagle3_tree_warmup_anchor_lengths,
)
from tests.gpu import test_eagle3_chain as chain
from tests.gpu import test_eagle3_tree as tree


class ExperimentalLengthContractTest(unittest.TestCase):
    def test_scope_restores_default_after_nested_failure(self):
        self.assertEqual(_committed_token_limit(), 512)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with _experimental_2048_token_budget():
                self.assertEqual(_committed_token_limit(), 2048)
                self.assertEqual(EAGLE3_MAX_COMMITTED_TOKENS, 512)
                self.assertEqual(Context().run(_committed_token_limit), 512)
                with _experimental_2048_token_budget():
                    self.assertEqual(_committed_token_limit(), 2048)
                self.assertEqual(_committed_token_limit(), 2048)
                raise RuntimeError("injected")
        self.assertEqual(_committed_token_limit(), 512)

    def test_generation_boundary_and_overflow(self):
        ids = torch.zeros((1, 2047), dtype=torch.long)
        target = chain._IncrementingTarget(eos_token_id=7)
        spec = GreedyChainSpeculator(target, chain._IncrementingDrafter(), layer_indices=(0,))
        for generate in (lambda **kw: target_greedy_generate(target, ids, **kw),
                         lambda **kw: spec.generate(ids, **kw)):
            with self.assertRaisesRegex(ValueError, "512-token"):
                generate(max_new_tokens=1, stop_on_eos=False)
            with _experimental_2048_token_budget():
                result = generate(max_new_tokens=1, stop_on_eos=False)
                self.assertEqual(result.tokens.shape[1], 1)
                with self.assertRaisesRegex(ValueError, "2048-token"):
                    generate(max_new_tokens=2, stop_on_eos=False)

    def test_drafter_and_warmup_share_the_length_scope(self):
        adapter = Eagle3ChainDrafter(chain._FakeEagleModel())
        builder = tree._IncrementalScriptedTreeBuilder()
        def proposal(length):
            return dict(fed_ids=torch.zeros(1, length, dtype=torch.long),
                aux_history=torch.zeros(1, length, 6), pending_seed=torch.tensor([[13]]))
        with _experimental_2048_token_budget():
            self.assertEqual(eagle3_tree_temporary_kv_tokens(
                sequence_length=2042, max_depth=5), 2050)
            self.assertEqual(eagle3_tree_warmup_anchor_lengths(
                max_sequence_length=2042, max_depth=5,
                sequence_anchors=(506, 1024, 1536, 2042)), (506, 1024, 1536, 2042))
            adapter.propose(**proposal(2044), draft_length=3)
            builder.build_candidate_pool(**proposal(2044), max_depth=3)
            for call in (lambda: adapter.propose(**proposal(2045), draft_length=3),
                         lambda: builder.build_candidate_pool(**proposal(2045), max_depth=3)):
                with self.assertRaises(NotImplementedError):
                    call()
            with self.assertRaises(ValueError):
                eagle3_tree_temporary_kv_tokens(sequence_length=2043, max_depth=5)
        with self.assertRaises(ValueError):
            eagle3_tree_temporary_kv_tokens(sequence_length=2042, max_depth=5)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class QwenLongSequenceTest(unittest.TestCase):
    def test_real_attention_commits_and_tail_at_2048(self):
        torch.manual_seed(17)
        target = chain._tiny_target().cuda()
        target.set_attn_implementation("sdpa")
        draft = Eagle3DraftTree(token_ids=(0,)*5, parent_ids=(0,1,2,3,4),
            depths=(1,2,3,4,5), conditional_probabilities=(1.0,)*5,
            cumulative_log_probabilities=(0.0,)*5)
        spec = ShallowTreeSpeculator(target, tree._StaticTreeDrafter(draft), layer_indices=(0,))
        with torch.inference_mode(), _experimental_2048_token_budget():
            for length in (2040, 2041):
                ids = torch.ones((1, length), dtype=torch.long, device="cuda")
                result = spec.generate(ids, max_new_tokens=2048-length, max_depth=5,
                    stop_on_eos=False, audit_commits=True)
                self.assertEqual(result.tokens.shape[1]+length, 2048)
                self.assertTrue(result.commit_audits)
                self.assertTrue(all(a.passed for a in result.commit_audits))
                last = result.commit_audits[-1]
                self.assertEqual(last.aux_length_after, 2048 if length == 2040 else 2047)
                with self.assertRaisesRegex(ValueError, "2048-token"):
                    spec.generate(ids, max_new_tokens=2049-length)
            target.generation_config.eos_token_id = 0
            eos = spec.generate(ids, max_new_tokens=7, stop_on_eos=True, audit_commits=True)
            self.assertTrue(eos.stopped_on_eos)
            self.assertEqual(eos.tokens.tolist(), [[0]])
            self.assertEqual(eos.target_forward_calls, 1)


if __name__ == "__main__":
    unittest.main()
