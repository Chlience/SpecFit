from dataclasses import replace
from itertools import combinations
from math import exp, log
import unittest

from specfit.m1_capture_replay import read_routes
from specfit.r6a_experiment import PoolRecord, compute_r6a_pool_fingerprint
from specfit.selection_methods import SelectionCostModel, probability_ids, select_nodes
from specfit.selection_refinement import (CacheCostModel, acceptance_observations,
    acceptance_probabilities, fast_expert_selection, fit_acceptance, fit_cache_cost_model,
    query_features, resident_bitmap, select_refined, temper)
from tests.unit.r6a_fixture import R6AFixture


class SelectionRefinementTest(unittest.TestCase):
    @staticmethod
    def fixture(parents=None, probabilities=(0.65, 0.2)):
        raw = R6AFixture.pool(parents)
        cumulative = {0: 0.0}
        for node in raw["candidates"]:
            conditional = log(probabilities[0] if node["raw_id"] % 2 else probabilities[1])
            cumulative[node["raw_id"]] = cumulative[node["parent_raw_id"]] + conditional
            node.update(conditional_log_probability=conditional,
                        cumulative_log_probability=cumulative[node["raw_id"]])
        raw["fingerprint"] = compute_r6a_pool_fingerprint(raw)
        pool = PoolRecord.read(raw)
        atlas = read_routes(R6AFixture.routes(pool.fingerprint, tuple(range(19))),
                            "refinement fixture", pool, tuple(range(19)))
        return pool, atlas

    @staticmethod
    def model():
        return SelectionCostModel(tuple(250e6 + 30e6 * k for k in range(8)),
                                  tuple(90e6 + 1e6 * k for k in range(8)), 250000.0)

    def test_incremental_search_preserves_all_original_decisions_and_ties(self):
        for parents in (None, R6AFixture.alternate_parents):
            pool, atlas = self.fixture(parents)
            shared = replace(atlas, expert_ids_by_raw_layer=tuple(
                (raw, layer, tuple(range(8))) for raw, layer, _ in atlas.expert_ids_by_raw_layer))
            for routes in (atlas, shared):
                for model in (self.model(), SelectionCostModel((1.0,) * 8, (1.0,) * 8, 0.0),
                              SelectionCostModel((1.0, *(1e12 for _ in range(7))),
                                                 (1.0, *(1e12 for _ in range(7))), 0.0)):
                    with self.subTest(parents=parents, shared=routes is shared, model=model):
                        self.assertEqual(fast_expert_selection(pool, routes, model),
                                         select_nodes(pool, routes, "expert_cost_dynamic", model))

    def brute(self, pool, atlas, cost, residents, exponent):
        probabilities = acceptance_probabilities(pool, exponent)
        parents = {n.raw_id: n.parent_raw_id for n in pool.nodes}
        resident_pairs = {(lid, expert) for lid, ids in enumerate(residents) for expert in ids}
        rows = []
        for k in range(8):
            for ids in combinations(parents, k):
                if any(parents[i] not in (0, *ids) for i in ids):
                    continue
                experts = {(layer, expert) for raw, layer, values in atlas.expert_ids_by_raw_layer
                           if raw in (0, *ids) for expert in values}
                utility = 1 + sum(probabilities[i] for i in ids)
                elapsed = cost.predict_ns(k, len(experts), len(experts - resident_pairs))
                rows.append(((-utility / elapsed, -utility, len(experts), k, ids), ids))
        return min(rows)[1]

    def test_search_preserves_objective_ties_with_compensated_probability_sums(self):
        pool, atlas = self.fixture(probabilities=(0.47454250539060727, 0.35255387827663837))
        atlas = replace(atlas, expert_ids_by_raw_layer=tuple(
            (raw, layer, tuple(range(8))) for raw, layer, _ in atlas.expert_ids_by_raw_layer))
        probabilities = {n.raw_id: exp(n.cumulative_log_probability) for n in pool.nodes}
        offsets = tuple(1.0 + sum((probabilities[i] for i in probability_ids(pool, k)), start=0.0)
                        for k in range(8))
        model = SelectionCostModel(offsets, offsets, 0.0)
        expected = select_nodes(pool, atlas, "expert_cost_dynamic", model)
        self.assertEqual(expected, (1, 2, 3, 4, 5, 6, 7))
        # Naive incremental addition gives six nodes a spurious score above 1.
        self.assertEqual(fast_expert_selection(pool, atlas, model), expected)

    def test_cache_and_calibrated_choices_match_independent_subset_set_reference(self):
        pool, atlas = self.fixture()
        cost = CacheCostModel((80e6,) * 8, 100000.0, 500000.0)
        for residents in ([()] * 48, [tuple(range(48))] * 48):
            for exponent in (1.0, 2.0):
                self.assertEqual(fast_expert_selection(pool, atlas, cost,
                    resident_mask=resident_bitmap(residents), exponent=exponent),
                    self.brute(pool, atlas, cost, residents, exponent))

    def test_residency_changes_only_missing_work_and_seed_is_always_charged(self):
        pool, atlas = self.fixture()
        empty = query_features(pool, atlas, (), 0)
        self.assertEqual(empty, dict(draft_node_count=0, expert_union=384, initial_missing=384))
        seed_residents = [tuple(values) for raw, _, values in atlas.expert_ids_by_raw_layer if raw == 0]
        resident = resident_bitmap(seed_residents)
        warm = query_features(pool, atlas, (), resident)
        self.assertEqual(warm, {**empty, "initial_missing": 0})
        cost = CacheCostModel((1e6,) * 8, 100000.0, 1e6)
        self.assertGreater(cost.predict_ns(0, 384, 0), 0)
        for invalid in ([()] * 47, [(1, 1)] * 48, [(128,)] * 48, [tuple(range(49))] * 48):
            with self.assertRaises(ValueError):
                resident_bitmap(invalid)
        with self.assertRaises(ValueError):
            query_features(pool, atlas, (3,), resident)

    def test_fit_recovers_compute_and_missing_coefficients_without_test_labels(self):
        points = []
        for k in range(8):
            for j in range(4):
                for z in range(3):
                    f, n = 384 + k * (45 + 15 * j), 30 + 45 * z + 7 * k
                    points.append(dict(question_id=j, draft_node_count=k,
                        expert_union=f, initial_missing=n,
                        cycle_without_selector_ns=80e6 + 2e6 * k + 125000 * f + 275000 * n))
        model, diagnostic = fit_cache_cost_model(points, mode="natural")
        self.assertLess(diagnostic["mean_absolute_relative_error"], 1e-7)
        self.assertAlmostEqual(model.expert_ns, 125000, delta=0.2)
        self.assertAlmostEqual(model.missing_ns, 275000, delta=0.2)
        self.assertEqual(CacheCostModel.from_dict(model.to_dict()), model)
        cold = [{**p, "initial_missing": p["expert_union"]} for p in points]
        model, diagnostic = fit_cache_cost_model(cold, mode="cold")
        self.assertEqual(model.missing_ns, 0)
        self.assertTrue(diagnostic["cold_collinear_terms_collapsed"])
        with self.assertRaises(ValueError):
            fit_cache_cost_model(points, mode="cold")
        with self.assertRaises(ValueError):
            fit_cache_cost_model(points[:-12], mode="natural")

    def test_acceptance_calibration_keeps_categorical_mass_and_ancestor_consistency(self):
        pool, _ = self.fixture()
        for exponent in (0.5, 1.0, 3.0):
            self.assertAlmostEqual(sum(temper((0.65, 0.2, 0.15), exponent)), 1.0)
            values = acceptance_probabilities(pool, exponent)
            for node in pool.nodes:
                parent = 1.0 if node.parent_raw_id == 0 else values[node.parent_raw_id]
                self.assertLessEqual(values[node.raw_id], parent)
            for parent in {n.parent_raw_id for n in pool.nodes}:
                self.assertLessEqual(sum(values[n.raw_id] for n in pool.nodes if n.parent_raw_id == parent),
                                     (1.0 if parent == 0 else values[parent]) + 1e-14)
        self.assertEqual(acceptance_probabilities(pool, 1.0),
                         {n.raw_id: exp(n.cumulative_log_probability) for n in pool.nodes})

    def test_training_observations_use_parent_verifier_and_grouped_calibration(self):
        pool, _ = self.fixture()
        verifier = [-1] * 19
        for node in pool.nodes:
            if node.raw_id % 2:
                verifier[node.parent_raw_id] = node.target_token_id
        observations = [row for qid in (401, 402, 403)
                        for row in acceptance_observations(pool, verifier, qid)]
        self.assertEqual(len(observations), 27)
        self.assertTrue(all(row["label"] == 0 for row in observations))
        exponent, diagnostics = fit_acceptance(observations)
        self.assertEqual(exponent, 3.0)
        self.assertTrue(diagnostics["accepted_by_grouped_validation"])
        with self.assertRaises(ValueError):
            fit_acceptance(observations[:9])

    def test_guard_falls_back_without_changing_old_method_semantics(self):
        pool, atlas = self.fixture()
        legacy = self.model()
        cache = CacheCostModel(legacy.size_intercept_ns, legacy.expert_ns, 100000.0)
        expected = select_nodes(pool, atlas, "probability_dynamic", legacy)
        actual = select_refined(pool, atlas, "expert_guarded_dynamic", legacy_model=legacy,
            cache_model=cache, acceptance_exponent=1.0, guard_margin=1.0)
        self.assertEqual(actual, expected)
        for method in ("target_only", "probability_fixed", "expert_cost_fixed",
                       "probability_dynamic", "expert_cost_dynamic"):
            self.assertEqual(select_refined(pool, atlas, method, legacy_model=legacy,
                cache_model=cache, resident_mask=(1 << 6144) - 1, acceptance_exponent=3.0),
                select_nodes(pool, atlas, method, legacy))


if __name__ == "__main__":
    unittest.main()
