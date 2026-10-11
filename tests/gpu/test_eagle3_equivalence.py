from __future__ import annotations

import math
import unittest

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from specfit.runtime import (
    Eagle3DraftTree,
    compare_tree_verification_to_causal_paths,
    diagnose_tree_node_zero_numerics,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class Qwen3TreeVerificationEquivalenceTest(unittest.TestCase):
    def test_branched_tree_matches_independent_causal_paths(self) -> None:
        torch.manual_seed(17)
        config = Qwen3MoeConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=48,
            moe_intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=64,
            bos_token_id=1,
            eos_token_id=63,
            pad_token_id=62,
            use_cache=True,
            attn_implementation="sdpa",
        )
        target = Qwen3MoeForCausalLM(config).cuda().eval()
        target.config._experts_implementation = "eager"
        input_ids = torch.tensor([[1, 7, 11]], device="cuda")
        attention_mask = torch.ones_like(input_ids)

        tree = Eagle3DraftTree(
            token_ids=(5, 6, 9, 10, 12, 13, 14),
            parent_ids=(0, 0, 1, 2, 2, 4, 5),
            depths=(1, 1, 2, 2, 2, 3, 3),
            conditional_probabilities=(
                0.55,
                0.45,
                0.7,
                0.65,
                0.35,
                0.8,
                0.7,
            ),
            cumulative_log_probabilities=(
                math.log(0.55),
                math.log(0.45),
                math.log(0.55 * 0.7),
                math.log(0.45 * 0.65),
                math.log(0.45 * 0.35),
                math.log(0.45 * 0.65 * 0.8),
                math.log(0.45 * 0.35 * 0.7),
            ),
        )
        node_zero_mask = tree.verification_attention_mask(
            base_kv_length=input_ids.shape[1],
            device=input_ids.device,
        )
        self.assertEqual(
            node_zero_mask[0, 0, 0, input_ids.shape[1] :].tolist(),
            [True, False, False, False, False, False, False, False],
        )

        with torch.inference_mode():
            prefill = target(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
            )
            pending_seed = prefill.logits.argmax(dim=-1)
            prompt_prefill_calls = 0

            def count_prompt_prefill(
                module: torch.nn.Module,
                args: tuple[torch.Tensor, ...],
                kwargs: dict[str, object],
            ) -> None:
                nonlocal prompt_prefill_calls
                del module, args
                if kwargs.get("position_ids") is None:
                    prompt_prefill_calls += 1

            handle = target.register_forward_pre_hook(
                count_prompt_prefill,
                with_kwargs=True,
            )
            try:
                report = compare_tree_verification_to_causal_paths(
                    target,
                    input_ids,
                    tree,
                    attention_mask=attention_mask,
                    expected_pending_seed=pending_seed,
                )
            finally:
                handle.remove()

        expected_node_count = 1 + tree.draft_node_count
        expected_route_comparisons = (
            expected_node_count * config.num_hidden_layers
        )
        self.assertTrue(report.passed)
        self.assertTrue(report.semantic_passed)
        self.assertTrue(report.diagnostic_complete)
        self.assertTrue(report.decision_and_route_passed)
        self.assertEqual(prompt_prefill_calls, 1)
        self.assertEqual(
            report.reference_mode,
            "exact_cloned_prefix_cached_linear_suffix",
        )
        self.assertEqual(report.node_count, expected_node_count)
        self.assertEqual(report.layer_count, config.num_hidden_layers)
        self.assertEqual(report.argmax_equal_count, expected_node_count)
        self.assertEqual(report.logits_finite_count, expected_node_count)
        self.assertEqual(report.logits_close_count, expected_node_count)
        self.assertEqual(
            report.route_comparison_count,
            expected_route_comparisons,
        )
        self.assertEqual(report.route_equal_count, expected_route_comparisons)
        self.assertEqual(
            report.ordered_route_equal_count,
            expected_route_comparisons,
        )
        self.assertEqual(
            report.set_route_equal_count,
            expected_route_comparisons,
        )
        self.assertEqual(
            report.routing_weights_finite_count,
            expected_route_comparisons,
        )
        self.assertEqual(
            report.routing_weights_close_count,
            expected_route_comparisons,
        )
        self.assertEqual(report.logits_atol, 2e-4)
        self.assertEqual(report.logits_rtol, 1e-2)
        self.assertEqual(report.routing_weights_atol, 2e-4)
        self.assertEqual(report.routing_weights_rtol, 1e-2)
        self.assertGreaterEqual(report.tree_min_top1_margin, 0.0)
        self.assertGreaterEqual(report.reference_min_top1_margin, 0.0)
        self.assertGreaterEqual(report.top1_margin_max_abs_difference, 0.0)
        self.assertTrue(report.prefix_cache_clone_exact)
        self.assertTrue(report.tree_prefix_cache_clone_exact)
        self.assertTrue(report.pristine_prefix_cache_unchanged)
        self.assertEqual(
            report.reference_prefix_cache_clone_exact_count,
            expected_node_count,
        )
        self.assertEqual(report.mask_passed_count, expected_node_count)
        self.assertEqual(report.position_passed_count, expected_node_count)
        self.assertEqual(
            report.tree_attention_shape,
            (1, 1, expected_node_count, input_ids.shape[1] + expected_node_count),
        )
        self.assertEqual(report.tree_attention_dtype, "bool")
        self.assertEqual(
            report.tree_verifier_token_ids,
            report.reference_verifier_token_ids,
        )
        self.assertEqual(
            report.tree_accepted_node_ids,
            report.reference_accepted_node_ids,
        )
        self.assertTrue(report.accepted_path_equal)
        self.assertEqual(
            report.tree_committed_token_ids,
            report.reference_committed_token_ids,
        )
        self.assertTrue(report.committed_tokens_equal)
        self.assertEqual(
            report.tree_next_pending_seed_token_id,
            report.reference_next_pending_seed_token_id,
        )
        self.assertTrue(report.next_pending_seed_equal)
        self.assertEqual(
            tuple(node.node_id for node in report.nodes),
            tuple(range(expected_node_count)),
        )
        self.assertEqual(report.nodes[6].parent_id, 4)
        self.assertEqual(report.nodes[6].depth, 3)
        self.assertEqual(
            report.nodes[6].causal_suffix_token_ids,
            (int(pending_seed.item()), 6, 10, 13),
        )
        for node in report.nodes:
            self.assertTrue(node.argmax_equal)
            self.assertTrue(node.logits_finite)
            self.assertTrue(node.prefix_cache_clone_exact)
            self.assertTrue(node.mask_passed)
            self.assertEqual(
                node.visible_key_positions,
                node.expected_key_positions,
            )
            self.assertTrue(node.position_passed)
            self.assertEqual(node.position_id, node.expected_position_id)
            self.assertEqual(
                node.tree_argmax_token_id,
                node.reference_argmax_token_id,
            )
            self.assertTrue(node.logits_close)
            self.assertGreaterEqual(node.logits_max_abs, 0.0)
            self.assertGreaterEqual(node.logits_mean_abs, 0.0)
            self.assertGreaterEqual(node.tree_top1_margin, 0.0)
            self.assertGreaterEqual(node.reference_top1_margin, 0.0)
            self.assertGreaterEqual(node.top1_margin_abs_difference, 0.0)
            self.assertEqual(
                node.route_comparison_count,
                config.num_hidden_layers,
            )
            self.assertEqual(
                node.route_equal_count,
                config.num_hidden_layers,
            )
            self.assertEqual(
                node.ordered_route_equal_count,
                config.num_hidden_layers,
            )
            self.assertEqual(
                node.set_route_equal_count,
                config.num_hidden_layers,
            )
            self.assertEqual(
                node.routing_weights_finite_count,
                config.num_hidden_layers,
            )
            self.assertEqual(
                node.routing_weights_close_count,
                config.num_hidden_layers,
            )
            self.assertEqual(node.route_mismatches, ())
            self.assertEqual(len(node.routes), config.num_hidden_layers)
            for route in node.routes:
                self.assertTrue(route.ordered_equal)
                self.assertTrue(route.set_equal)
                self.assertTrue(route.routing_weights_finite)
                self.assertTrue(route.routing_weights_close)
                self.assertEqual(
                    route.aligned_expert_ids,
                    tuple(sorted(route.aligned_expert_ids)),
                )
                self.assertEqual(
                    len(route.tree_aligned_routing_weights),
                    len(route.aligned_expert_ids),
                )
                self.assertEqual(
                    len(route.reference_aligned_routing_weights),
                    len(route.aligned_expert_ids),
                )

        numerical = diagnose_tree_node_zero_numerics(
            target,
            input_ids,
            tree,
            attention_mask=attention_mask,
            expected_pending_seed=pending_seed,
        )
        self.assertTrue(numerical.diagnostic_complete)
        self.assertTrue(numerical.prefix_cache_clone_exact)
        self.assertTrue(numerical.node_zero_mask_passed)
        self.assertEqual(
            numerical.tree_attention_shape,
            (1, 1, 8, input_ids.shape[1] + 8),
        )
        self.assertEqual(numerical.tree_attention_dtype, "bool")
        self.assertEqual(
            numerical.node_zero_visible_key_positions,
            tuple(range(input_ids.shape[1] + 1)),
        )
        self.assertTrue(numerical.node_zero_position_passed)
        self.assertEqual(
            numerical.node_zero_position_id,
            input_ids.shape[1],
        )
        self.assertEqual(numerical.verification_node_count, 8)
        self.assertEqual(numerical.layer_count, config.num_hidden_layers)
        expected_stage_order = (
            "layer_input",
            "attention_input",
            "q_projection",
            "q_normalized",
            "k_projection",
            "k_normalized",
            "v_projection",
            "attention_core",
            "attention_output",
            "router_input",
            "router_logits",
            "routing_weights",
            "expert_output",
            "layer_output",
        )
        self.assertEqual(numerical.stage_order, expected_stage_order)
        self.assertEqual(
            tuple((item.layer_id, item.stage) for item in numerical.stages),
            tuple(
                (layer_id, stage)
                for layer_id in range(config.num_hidden_layers)
                for stage in expected_stage_order
            ),
        )
        self.assertEqual(
            numerical.stage_comparison_count,
            config.num_hidden_layers * len(numerical.stage_order),
        )
        self.assertEqual(
            numerical.finite_stage_count,
            numerical.stage_comparison_count,
        )
        tolerance_differences = [
            item for item in numerical.stages if not item.close
        ]
        self.assertEqual(
            numerical.close_stage_count,
            numerical.stage_comparison_count - len(tolerance_differences),
        )
        expected_first = tolerance_differences[0] if tolerance_differences else None
        if expected_first is None:
            self.assertIsNone(numerical.first_tolerance_difference)
        else:
            self.assertEqual(
                numerical.first_tolerance_difference.layer_id,
                expected_first.layer_id,
            )
            self.assertEqual(
                numerical.first_tolerance_difference.stage,
                expected_first.stage,
            )
        self.assertEqual(
            numerical.ordered_route_equal_count,
            config.num_hidden_layers,
        )
        self.assertEqual(
            numerical.set_route_equal_count,
            config.num_hidden_layers,
        )
        self.assertTrue(numerical.argmax_equal)
        layer_zero_input = numerical.stages[0]
        self.assertEqual(layer_zero_input.stage, "layer_input")
        self.assertTrue(layer_zero_input.exact)
        self.assertEqual(layer_zero_input.different_elements, 0)
        for stage_name in (
            "attention_output",
            "router_logits",
            "expert_output",
        ):
            stage = next(
                item
                for item in numerical.stages
                if item.layer_id == 0 and item.stage == stage_name
            )
            self.assertEqual(stage.tree_row_count, 8)
            self.assertEqual(stage.reference_row_count, 1)
            self.assertTrue(stage.finite)

        class LeakyNodeZeroTree(Eagle3DraftTree):
            def verification_attention_mask(
                self,
                *,
                base_kv_length: int,
                device: torch.device,
            ) -> torch.Tensor:
                mask = super().verification_attention_mask(
                    base_kv_length=base_kv_length,
                    device=device,
                )
                mask[0, 0, 0, base_kv_length + 1] = True
                return mask

        leaky_tree = LeakyNodeZeroTree(
            token_ids=tree.token_ids,
            parent_ids=tree.parent_ids,
            depths=tree.depths,
            conditional_probabilities=tree.conditional_probabilities,
            cumulative_log_probabilities=tree.cumulative_log_probabilities,
        )
        leaky = diagnose_tree_node_zero_numerics(
            target,
            input_ids,
            leaky_tree,
            attention_mask=attention_mask,
            expected_pending_seed=pending_seed,
            atol=1e-3,
        )
        self.assertFalse(leaky.node_zero_mask_passed)
        self.assertFalse(leaky.diagnostic_complete)
        self.assertIn(
            input_ids.shape[1] + 1,
            leaky.node_zero_visible_key_positions,
        )
        self.assertIsNotNone(leaky.first_tolerance_difference)
        self.assertEqual(leaky.first_tolerance_difference.layer_id, 0)
        self.assertEqual(
            leaky.first_tolerance_difference.stage,
            "attention_core",
        )

        def reverse_tree_route_order(
            module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
            del module, inputs
            router_logits, routing_weights, selected_experts = output
            if selected_experts.shape[0] != expected_node_count:
                return None
            return (
                router_logits,
                routing_weights.flip(-1),
                selected_experts.flip(-1),
            )

        handles = [
            layer.mlp.gate.register_forward_hook(reverse_tree_route_order)
            for layer in target.model.layers
        ]
        try:
            reordered = compare_tree_verification_to_causal_paths(
                target,
                input_ids,
                tree,
                attention_mask=attention_mask,
                expected_pending_seed=pending_seed,
            )
        finally:
            for handle in handles:
                handle.remove()

        self.assertTrue(reordered.semantic_passed)
        self.assertTrue(reordered.passed)
        self.assertTrue(reordered.diagnostic_complete)
        self.assertFalse(reordered.decision_and_route_passed)
        self.assertLess(
            reordered.ordered_route_equal_count,
            reordered.route_comparison_count,
        )
        self.assertEqual(
            reordered.set_route_equal_count,
            reordered.route_comparison_count,
        )
        self.assertEqual(
            reordered.routing_weights_close_count,
            reordered.route_comparison_count,
        )


if __name__ == "__main__":
    unittest.main()
