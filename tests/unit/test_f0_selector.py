from dataclasses import replace
from itertools import combinations
import random
import unittest

from specfit.f0_direct import COEFFICIENTS, _bitset_scores, select_ids
from specfit.j1_calibration import CostModel
from specfit.j1_screen import select_joint_candidate
from specfit.m1_capture_replay import read_routes
from specfit.m1_screen import score_candidate_subsets
from specfit.r6a_experiment import PoolRecord, FrozenStateRecord
from specfit.r6a_selection_contract import enumerate_ancestor_closed_subsets
from tests.unit.r6a_fixture import R6AFixture


def brute_force(pool, budget=7):
    parent = {n.raw_id: n.parent_raw_id for n in pool.nodes}
    return tuple(ids for ids in combinations(parent, budget)
                 if all(parent[i] == 0 or parent[i] in ids for i in ids))


class F0SelectorTest(unittest.TestCase):
    def pools(self):
        rng = random.Random(480)
        yield PoolRecord.read(R6AFixture.pool())
        yield PoolRecord.read(R6AFixture.pool(R6AFixture.alternate_parents))
        for _ in range(8):
            parents = {1: 0, 2: 0}
            previous = [1, 2]
            raw = 3
            for _depth in range(2, 6):
                for parent in sorted(rng.sample(previous, 2)):
                    parents[raw] = parents[raw + 1] = parent
                    raw += 2
                previous = list(range(raw - 4, raw))
            yield PoolRecord.read(R6AFixture.pool(parents))

    def test_exact_order_and_membership_for_all_budgets_and_topologies(self):
        for pool in self.pools():
            for budget in range(20):
                with self.subTest(fingerprint=pool.fingerprint, budget=budget):
                    self.assertEqual(enumerate_ancestor_closed_subsets(pool, draft_node_budget=budget),
                                     brute_force(pool, budget))
        for budget in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                enumerate_ancestor_closed_subsets(pool, draft_node_budget=budget)

    def test_bitmaps_match_all_set_scores_and_selected_ids(self):
        for pool in self.pools():
            atlas = read_routes(R6AFixture.routes(pool.fingerprint, tuple(range(19))),
                                "fixture atlas", pool, tuple(range(19)))
            variants = (atlas, replace(atlas, expert_ids_by_raw_layer=tuple(
                (raw, layer, tuple(range(8))) for raw, layer, _ in atlas.expert_ids_by_raw_layer)),
                replace(atlas, expert_ids_by_raw_layer=tuple(reversed(tuple(
                    (raw, layer, tuple(range(120, 128)) if raw == 0 else experts)
                    for raw, layer, experts in atlas.expert_ids_by_raw_layer)))))
            subsets = brute_force(pool)
            for routes in variants:
                reference = score_candidate_subsets(pool=pool, atlas=routes,
                    frozen=FrozenStateRecord("0" * 64, ((),) * 48), subsets=subsets)
                actual = _bitset_scores(pool, routes, subsets)
                self.assertEqual(actual, reference)
                self.assertEqual([r.draft_utility.hex() for r in actual],
                                 [r.draft_utility.hex() for r in reference])
                for policy in ("f0", "hbm7"):
                    slope = COEFFICIENTS["preloaded_ns_per_expert"]
                    if policy == "f0":
                        slope += COEFFICIENTS["h2d_ns_per_expert"]
                    expected = select_joint_candidate(reference, CostModel(COEFFICIENTS["fixed_ns"], slope, 0))
                    self.assertEqual(select_ids(pool, routes, policy), expected.selected_raw_ids)

    def test_equal_utility_and_footprint_keep_lexicographic_tie_break(self):
        pool = next(self.pools())
        pool = replace(pool, nodes=tuple(replace(n, conditional_log_probability=0.0,
            cumulative_log_probability=0.0) for n in pool.nodes))
        atlas = read_routes(R6AFixture.routes(pool.fingerprint, tuple(range(19))),
                            "fixture atlas", pool, tuple(range(19)))
        atlas = replace(atlas, expert_ids_by_raw_layer=tuple(
            (raw, layer, tuple(range(8))) for raw, layer, _ in atlas.expert_ids_by_raw_layer))
        for policy in ("f0", "hbm7"):
            self.assertEqual(select_ids(pool, atlas, policy), brute_force(pool)[0])

    def test_rejects_incomplete_duplicate_or_invalid_routes(self):
        pool = next(self.pools())
        atlas = read_routes(R6AFixture.routes(pool.fingerprint, tuple(range(19))),
                            "fixture atlas", pool, tuple(range(19)))
        rows = atlas.expert_ids_by_raw_layer
        invalid = (rows[:-1], (rows[1], *rows[1:]), ((0, 0, (0,) * 8), *rows[1:]),
                   ((0, 0, tuple(range(121, 129))), *rows[1:]),
                   ((0, 48, tuple(range(8))), *rows[1:]),
                   ((0, 0, (False, *range(1, 8))), *rows[1:]))
        for entries in invalid:
            with self.assertRaises(ValueError):
                select_ids(pool, replace(atlas, expert_ids_by_raw_layer=entries), "f0")
