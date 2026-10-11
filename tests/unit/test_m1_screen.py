from __future__ import annotations

import math
import unittest
from dataclasses import replace

from specfit.m1_screen import (
    CandidateScore,
    GateResult,
    StateScreen,
    evaluate_gate,
    pareto_frontier_indices,
    screen_state,
)
from specfit.r6a_experiment import FrozenStateRecord, PoolRecord
from specfit.r6a_route_observation import RouteObservation
from tests.unit.r6a_fixture import R6AFixture


class M1ScreenTest(unittest.TestCase):
    @staticmethod
    def _atlas(
        pool: PoolRecord,
        *,
        root2_uses_seed: bool = False,
    ) -> RouteObservation:
        parent = {node.raw_id: node.parent_raw_id for node in pool.nodes}

        def root(raw_id: int) -> int:
            while parent.get(raw_id, 0):
                raw_id = parent[raw_id]
            return raw_id

        rows = []
        for raw_id in (0, *(node.raw_id for node in pool.nodes)):
            experts = (
                tuple(range(8))
                if raw_id == 0 or root2_uses_seed and root(raw_id) == 2
                else tuple(range(8, 16))
                if root(raw_id) == 1
                else tuple(range(16, 24))
            )
            rows.extend((raw_id, layer_id, experts) for layer_id in range(48))
        return RouteObservation(
            position_to_raw=(0, *(node.raw_id for node in pool.nodes)),
            digest="d" * 64,
            expert_ids_by_raw_layer=tuple(rows),
        )

    @staticmethod
    def _frozen(experts: tuple[int, ...]) -> FrozenStateRecord:
        return FrozenStateRecord("a" * 64, (experts,) * 48)

    def test_state_screen_uses_actual_topology_and_draft_only_utility(self) -> None:
        default_pool = PoolRecord.read(R6AFixture.pool())
        alternate_pool = PoolRecord.read(
            R6AFixture.pool(R6AFixture.alternate_parents)
        )
        default = screen_state(
            state_id="q406-c0",
            question_id=406,
            cycle_index=0,
            pool=default_pool,
            atlas=self._atlas(default_pool),
            frozen=self._frozen(tuple(range(8, 16))),
        )
        alternate = screen_state(
            state_id="q406-c1",
            question_id=406,
            cycle_index=1,
            pool=alternate_pool,
            atlas=self._atlas(alternate_pool),
            frozen=self._frozen(tuple(range(8, 16))),
        )

        self.assertEqual((len(default.rows), len(alternate.rows)), (104, 133))
        row = default.ratio_greedy
        by_id = {node.raw_id: node for node in default_pool.nodes}
        self.assertAlmostEqual(
            row.draft_utility,
            sum(math.exp(by_id[raw_id].cumulative_log_probability)
                for raw_id in row.selected_raw_ids),
        )
        self.assertEqual((row.F_atlas, row.N_atlas), (768, 384))

    def test_ratio_greedy_is_seed_aware_and_uses_admissible_frontier(self) -> None:
        pool = PoolRecord.read(R6AFixture.pool())
        result = screen_state(
            state_id="q406-c0",
            question_id=406,
            cycle_index=0,
            pool=pool,
            atlas=self._atlas(pool, root2_uses_seed=True),
            frozen=self._frozen(()),
        )

        self.assertEqual(
            result.ratio_greedy.selected_raw_ids,
            (2, 5, 6, 9, 10, 13, 14),
        )
        parents = {node.raw_id: node.parent_raw_id for node in pool.nodes}
        self.assertTrue(all(
            parents[raw_id] == 0
            or parents[raw_id] in result.ratio_greedy.selected_raw_ids
            for raw_id in result.ratio_greedy.selected_raw_ids
        ))

    def test_matched_oracles_isolate_residency_and_pareto(self) -> None:
        pool = PoolRecord.read(R6AFixture.pool())
        result = screen_state(
            state_id="q406-c0",
            question_id=406,
            cycle_index=0,
            pool=pool,
            atlas=self._atlas(pool),
            frozen=self._frozen(tuple(range(16, 24))),
        )

        self.assertEqual(result.V_F, result.ratio_greedy)
        self.assertNotEqual(result.V_F.selected_raw_ids, result.V_R.selected_raw_ids)
        self.assertEqual(result.V_F.F_atlas, result.V_R.F_atlas)
        self.assertGreaterEqual(
            result.V_R.draft_utility,
            0.99 * max(result.ratio_greedy.draft_utility,
                       result.V_F.draft_utility),
        )
        self.assertEqual(result.delta_N_atlas, 384)
        self.assertIn(result.rows.index(result.V_F), result.pareto_frontier_indices)
        self.assertIn(result.rows.index(result.V_R), result.pareto_frontier_indices)

    def test_three_dimensional_pareto_keeps_tradeoffs_and_ties(self) -> None:
        rows = (
            CandidateScore((1,), 3.0, 2, 4),
            CandidateScore((2,), 3.0, 3, 3),
            CandidateScore((3,), 4.0, 6, 6),
            CandidateScore((4,), 3.0, 2, 4),
            CandidateScore((5,), 2.0, 4, 5),
        )
        self.assertEqual(pareto_frontier_indices(rows), (0, 1, 2, 3))

    def test_state_screen_rejects_incomplete_full_atlas(self) -> None:
        pool = PoolRecord.read(R6AFixture.pool())
        atlas = self._atlas(pool)
        with self.assertRaises(ValueError):
            screen_state(
                state_id="q406-c0",
                question_id=406,
                cycle_index=0,
                pool=pool,
                atlas=replace(
                    atlas,
                    expert_ids_by_raw_layer=atlas.expert_ids_by_raw_layer[:-1],
                ),
                frozen=self._frozen(()),
            )

    @staticmethod
    def _gate_states(positive: set[tuple[int, int]]) -> tuple[StateScreen, ...]:
        states = []
        for question_id in range(406, 477, 10):
            for cycle_index in range(4):
                V_F = CandidateScore((1,), 1.0, 1, 100)
                delta = 10 if (question_id, cycle_index) in positive else 0
                V_R = CandidateScore((1,), 1.0, 1, 100 - delta)
                states.append(StateScreen(
                    state_id=f"q{question_id}-c{cycle_index}",
                    question_id=question_id,
                    cycle_index=cycle_index,
                    rows=(V_F,),
                    ratio_greedy=V_F,
                    V_F=V_F,
                    V_R=V_R,
                    pareto_frontier_indices=(0,),
                ))
        return tuple(states)

    def test_gate_requires_ratio_state_prompt_and_validation_thresholds(self) -> None:
        positive = {
            *((406, index) for index in range(4)),
            *((416, index) for index in range(4)),
            *((426, index) for index in range(3)),
            *((436, index) for index in range(2)),
            *((446, index) for index in range(2)),
            *((456, index) for index in range(2)),
        }
        states = self._gate_states(positive)
        gate = evaluate_gate(states, validation_passed=True)

        self.assertIsInstance(gate, GateResult)
        self.assertAlmostEqual(gate.G_atlas, 170 / 3_200)
        self.assertEqual((gate.positive_state_count,
                          gate.positive_question_count), (17, 6))
        self.assertTrue(gate.passed)
        self.assertFalse(
            evaluate_gate(states, validation_passed=False).passed
        )
        self.assertFalse(evaluate_gate(
            self._gate_states(set(sorted(positive)[:-1])),
            validation_passed=True,
        ).passed)
        zero_states = tuple(
            replace(
                state,
                ratio_greedy=CandidateScore((1,), 1.0, 1, 0),
                V_F=CandidateScore((1,), 1.0, 1, 0),
                V_R=CandidateScore((1,), 1.0, 1, 0),
            )
            for state in states
        )
        zero_gate = evaluate_gate(zero_states, validation_passed=True)
        self.assertIsNone(zero_gate.G_atlas)
        self.assertFalse(zero_gate.passed)

    def test_gate_rejects_non_preregistered_state_distribution(self) -> None:
        states = list(self._gate_states(set()))
        states[-1] = StateScreen(
            state_id="q480-c3",
            question_id=480,
            cycle_index=3,
            rows=states[-1].rows,
            ratio_greedy=states[-1].ratio_greedy,
            V_F=states[-1].V_F,
            V_R=states[-1].V_R,
            pareto_frontier_indices=(0,),
        )
        with self.assertRaises(ValueError):
            evaluate_gate(tuple(states), validation_passed=True)


if __name__ == "__main__":
    unittest.main()
