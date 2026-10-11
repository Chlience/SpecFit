from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeExperts,
)
from specfit import RuntimeEventKind
from specfit.runtime import (
    ExpertExecutionPolicy,
    GpuExpertCache,
    Qwen3OffloadedExperts,
    RecordingCacheTransitionObserver,
    ReductionMode,
)
from specfit.state import ExpertKey

def _pack_expert(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    return torch.cat((gate.flatten(), up.flatten(), down.flatten()))

def _hf_reference(
    gate_up: torch.Tensor,
    down: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
) -> Qwen3MoeExperts:
    config = Qwen3MoeConfig(
        hidden_size=hidden_size,
        moe_intermediate_size=intermediate_size,
        num_experts=gate_up.shape[0],
        hidden_act="silu",
    )
    reference = Qwen3MoeExperts(config).to(
        device=device,
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        reference.gate_up_proj.copy_(gate_up)
        reference.down_proj.copy_(down)
    return reference


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class Qwen3OffloadedExpertsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda:0")
        if torch.cuda.get_device_capability(cls.device) < (8, 0):
            raise unittest.SkipTest("BF16 requires compute capability 8.0+")

    def test_canonical_output_is_invariant_to_execution_order(self) -> None:
        hidden_size = 4
        intermediate_size = 3
        generator = torch.Generator().manual_seed(29)
        gate_up = torch.randn(
            3,
            2 * intermediate_size,
            hidden_size,
            dtype=torch.bfloat16,
            generator=generator,
        )
        down = torch.randn(
            3,
            hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
            generator=generator,
        )
        sources = {
            ExpertKey(0, expert_id): _pack_expert(
                gate_up[expert_id, :intermediate_size],
                gate_up[expert_id, intermediate_size:],
                down[expert_id],
            )
            for expert_id in range(3)
        }
        hidden_states = torch.randn(
            2,
            hidden_size,
            dtype=torch.bfloat16,
            generator=generator,
        ).to(self.device)
        selected_experts = torch.tensor(
            [[0, 1, 2], [2, 0, 1]],
            dtype=torch.long,
            device=self.device,
        )
        routing_weights = torch.tensor(
            [[0.50, 0.30, 0.20], [0.45, 0.35, 0.20]],
            dtype=torch.bfloat16,
            device=self.device,
        )
        cache = GpuExpertCache(
            sources,
            slot_count=1,
            device=self.device,
        )
        experts = Qwen3OffloadedExperts(
            layer_id=0,
            cache=cache,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            reduction_mode=ReductionMode.CANONICAL,
        )

        forward_order = experts(
            hidden_states,
            selected_experts,
            routing_weights,
            execution_order=(0, 1, 2),
        )
        torch.cuda.synchronize(self.device)
        self.assertEqual(
            cache.resident_expert_ids(),
            (ExpertKey(0, 2),),
        )
        reverse_order = experts(
            hidden_states,
            selected_experts,
            routing_weights,
            execution_order=(2, 1, 0),
        )
        torch.cuda.synchronize(self.device)
        self.assertEqual(
            cache.resident_expert_ids(),
            (ExpertKey(0, 0),),
        )
        reference_model = _hf_reference(
            gate_up,
            down,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            device=self.device,
        )
        reference = reference_model(
            hidden_states,
            selected_experts,
            routing_weights,
        )

        self.assertTrue(torch.equal(forward_order, reverse_order))
        torch.testing.assert_close(
            forward_order,
            reference,
            rtol=1.6e-2,
            atol=1.6e-2,
        )

    def test_resident_first_avoids_eviction_before_first_use(self) -> None:
        hidden_size = 4
        intermediate_size = 3
        generator = torch.Generator().manual_seed(101)
        sources = {
            ExpertKey(0, expert_id): torch.randn(
                3 * hidden_size * intermediate_size,
                dtype=torch.bfloat16,
                generator=generator,
            )
            for expert_id in range(3)
        }
        hidden_states = torch.randn(
            2,
            hidden_size,
            dtype=torch.bfloat16,
            generator=generator,
        ).to(self.device)
        selected_experts = torch.tensor(
            [[0, 1, 2], [2, 0, 1]],
            dtype=torch.long,
            device=self.device,
        )
        routing_weights = torch.tensor(
            [[0.50, 0.30, 0.20], [0.45, 0.35, 0.20]],
            dtype=torch.bfloat16,
            device=self.device,
        )

        def build_case(
            policy: ExpertExecutionPolicy | None,
        ) -> tuple[
            GpuExpertCache,
            Qwen3OffloadedExperts,
            RecordingCacheTransitionObserver,
        ]:
            observer = RecordingCacheTransitionObserver(
                run_id=f"resident-first-{policy}"
            )
            cache = GpuExpertCache(
                sources,
                slot_count=2,
                device=self.device,
                observer=observer,
            )
            self.addCleanup(cache.close)
            for expert_id in (1, 2):
                with cache.acquire(ExpertKey(0, expert_id)) as lease:
                    lease.wait()
            torch.cuda.synchronize(self.device)
            observer.reset()
            expert_kwargs = {}
            if policy is not None:
                expert_kwargs["execution_policy"] = policy
            experts = Qwen3OffloadedExperts(
                layer_id=0,
                cache=cache,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                reduction_mode=ReductionMode.CANONICAL,
                **expert_kwargs,
            )
            return cache, experts, observer

        legacy_cache, legacy, legacy_observer = build_case(None)
        resident_cache, resident_first, resident_observer = build_case(
            ExpertExecutionPolicy.RESIDENT_FIRST
        )
        explicit_cache, explicit, explicit_observer = build_case(
            ExpertExecutionPolicy.RESIDENT_FIRST
        )
        self.assertIs(
            legacy.execution_policy,
            ExpertExecutionPolicy.EXPERT_ID_ASCENDING,
        )
        legacy_before = legacy_cache.stats()
        resident_before = resident_cache.stats()
        explicit_before = explicit_cache.stats()

        legacy_output = legacy(
            hidden_states,
            selected_experts,
            routing_weights,
        )
        resident_output = resident_first(
            hidden_states,
            selected_experts,
            routing_weights,
        )
        explicit_output = explicit(
            hidden_states,
            selected_experts,
            routing_weights,
            execution_order=(0, 1, 2),
        )
        torch.cuda.synchronize(self.device)

        legacy_after = legacy_cache.stats()
        resident_after = resident_cache.stats()
        explicit_after = explicit_cache.stats()
        acquisition_kinds = {
            RuntimeEventKind.H2D_SUBMIT,
            RuntimeEventKind.RESIDENT_HIT,
        }
        self.assertEqual(
            tuple(
                event.expert_id.expert_id
                for event in legacy_observer.events()
                if event.kind in acquisition_kinds
            ),
            (0, 1, 2),
        )
        self.assertEqual(
            tuple(
                (event.kind, event.expert_id.expert_id)
                for event in resident_observer.events()
                if event.kind in acquisition_kinds
            ),
            (
                (RuntimeEventKind.RESIDENT_HIT, 1),
                (RuntimeEventKind.RESIDENT_HIT, 2),
                (RuntimeEventKind.H2D_SUBMIT, 0),
            ),
        )
        self.assertEqual(
            tuple(
                event.expert_id.expert_id
                for event in explicit_observer.events()
                if event.kind in acquisition_kinds
            ),
            (0, 1, 2),
        )

        def stats_delta(before, after) -> tuple[int, int, int, int]:
            return (
                after.h2d_submissions - before.h2d_submissions,
                after.resident_hits - before.resident_hits,
                after.inflight_joins - before.inflight_joins,
                after.evictions - before.evictions,
            )

        self.assertEqual(
            stats_delta(legacy_before, legacy_after),
            (3, 0, 0, 3),
        )
        self.assertEqual(
            stats_delta(resident_before, resident_after),
            (1, 2, 0, 1),
        )
        self.assertEqual(
            stats_delta(explicit_before, explicit_after),
            (3, 0, 0, 3),
        )
        self.assertTrue(torch.equal(legacy_output, resident_output))
        self.assertTrue(torch.equal(legacy_output, explicit_output))

    def test_expert_math_matches_hf_fused_gate_up_path_bitwise(self) -> None:
        hidden_size = 2048
        intermediate_size = 768
        generator = torch.Generator().manual_seed(41)
        gate_up = torch.randn(
            1,
            2 * intermediate_size,
            hidden_size,
            dtype=torch.bfloat16,
            generator=generator,
        )
        down = torch.randn(
            1,
            hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
            generator=generator,
        )
        source = _pack_expert(
            gate_up[0, :intermediate_size],
            gate_up[0, intermediate_size:],
            down[0],
        )
        hidden_states = torch.randn(
            8,
            hidden_size,
            dtype=torch.bfloat16,
            generator=generator,
        ).to(self.device)
        top_k_index = torch.zeros(
            8,
            1,
            dtype=torch.long,
            device=self.device,
        )
        top_k_weights = torch.ones(
            8,
            1,
            dtype=torch.bfloat16,
            device=self.device,
        )
        cache = GpuExpertCache(
            {ExpertKey(0, 0): source},
            slot_count=1,
            device=self.device,
        )
        experts = Qwen3OffloadedExperts(
            layer_id=0,
            cache=cache,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        reference = _hf_reference(
            gate_up,
            down,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            device=self.device,
        )

        actual = experts(hidden_states, top_k_index, top_k_weights)
        expected = reference(hidden_states, top_k_index, top_k_weights)

        self.assertTrue(torch.equal(actual, expected))

    def test_arrival_order_preserves_execution_order_reduction(self) -> None:
        silu_one = F.silu(
            torch.tensor(1.0, dtype=torch.bfloat16)
        ).item()
        desired = (1.0, 2.0**-8, -1.0)
        sources = {
            ExpertKey(0, expert_id): _pack_expert(
                torch.tensor([[1.0]], dtype=torch.bfloat16),
                torch.tensor([[1.0]], dtype=torch.bfloat16),
                torch.tensor(
                    [[value / silu_one]],
                    dtype=torch.bfloat16,
                ),
            )
            for expert_id, value in enumerate(desired)
        }
        hidden_states = torch.ones(
            1,
            1,
            dtype=torch.bfloat16,
            device=self.device,
        )
        top_k_index = torch.tensor(
            [[0, 1, 2]],
            dtype=torch.long,
            device=self.device,
        )
        top_k_weights = torch.ones(
            1,
            3,
            dtype=torch.bfloat16,
            device=self.device,
        )
        caches = {
            mode: GpuExpertCache(
                sources,
                slot_count=1,
                device=self.device,
            )
            for mode in ReductionMode
        }
        experts = {
            mode: Qwen3OffloadedExperts(
                layer_id=0,
                cache=caches[mode],
                hidden_size=1,
                intermediate_size=1,
                reduction_mode=mode,
            )
            for mode in ReductionMode
        }

        rounded_away = experts[ReductionMode.ARRIVAL_ORDER](
            hidden_states,
            top_k_index,
            top_k_weights,
            execution_order=(0, 1, 2),
        )
        experts[ReductionMode.CANONICAL](
            hidden_states,
            top_k_index,
            top_k_weights,
            execution_order=(0, 1, 2),
        )
        # pending_use_guards is an asynchronous observation; compare the
        # completed executions at the same boundary.
        torch.cuda.synchronize(self.device)
        self.assertEqual(
            caches[ReductionMode.ARRIVAL_ORDER].stats(),
            caches[ReductionMode.CANONICAL].stats(),
        )
        retained = experts[ReductionMode.ARRIVAL_ORDER](
            hidden_states,
            top_k_index,
            top_k_weights,
            execution_order=(0, 2, 1),
        )

        self.assertEqual(float(rounded_away.item()), 0.0)
        self.assertEqual(float(retained.item()), 2.0**-8)


if __name__ == "__main__":
    unittest.main()
