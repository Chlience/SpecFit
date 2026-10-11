from dataclasses import replace
from itertools import combinations
from math import exp
import unittest

from specfit.m1_capture_replay import read_routes
from specfit.r6a_experiment import PoolRecord
from specfit.selection_methods import (METHODS, METHOD_NAMES, SelectionCostModel,
    fit_cost_model, minimum_expert_ids, probability_ids, select_nodes, selection_features)
from tests.unit.r6a_fixture import R6AFixture


class SelectionMethodsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = PoolRecord.read(R6AFixture.pool())
        cls.atlas = read_routes(R6AFixture.routes(cls.pool.fingerprint, tuple(range(19))),
                               "test atlas", cls.pool, tuple(range(19)))

    def model(self):
        return SelectionCostModel(tuple(500_000_000 + 20_000_000 * k for k in range(8)),
                                  (200_000_000,) * 8, 300_000)

    def test_names_and_fixed_probability_baseline(self):
        self.assertEqual(set(METHODS), set(METHOD_NAMES))
        self.assertEqual(select_nodes(self.pool, None, "target_only"), ())
        self.assertEqual(select_nodes(self.pool, None, "probability_fixed"), probability_ids(self.pool, 7))
        for old_name in ("p7", "p0", "f0", "unknown"):
            with self.assertRaises(ValueError):
                select_nodes(self.pool, self.atlas, old_name, self.model())

    def brute_rows(self):
        parent = {n.raw_id: n.parent_raw_id for n in self.pool.nodes}
        probabilities = {n.raw_id: exp(n.cumulative_log_probability) for n in self.pool.nodes}
        for count in range(8):
            for ids in combinations(parent, count):
                if any(parent[i] != 0 and parent[i] not in ids for i in ids):
                    continue
                experts = {(layer, expert) for raw, layer, values in self.atlas.expert_ids_by_raw_layer
                           if raw in (0, *ids) for expert in values}
                yield ids, 1 + sum(probabilities[i] for i in ids), len(experts)

    def test_expert_methods_match_independent_set_and_subset_reference(self):
        model = self.model()
        for method, counts in (("expert_cost_fixed", (7,)), ("expert_cost_dynamic", range(8))):
            rows = [r for r in self.brute_rows() if len(r[0]) in counts]
            expected = min(rows, key=lambda r:
                (-r[1] / model.predict_ns(len(r[0]), r[2]), -r[1], r[2], len(r[0]), r[0]))[0]
            self.assertEqual(select_nodes(self.pool, self.atlas, method, model), expected)
        for count in range(8):
            expected = min((r for r in self.brute_rows() if len(r[0]) == count),
                           key=lambda r: (r[2], -r[1], r[0]))[0]
            self.assertEqual(minimum_expert_ids(self.pool, self.atlas, count), expected)

    def test_dynamic_probability_never_reads_expert_signals_and_can_select_zero(self):
        expensive = SelectionCostModel((1.0, *(1e12 for _ in range(7))), (1.0,) * 8, 0.0)
        self.assertEqual(select_nodes(self.pool, object(), "probability_dynamic", expensive), ())
        cheap = SelectionCostModel((1.0,) * 8, (1.0,) * 8, 0.0)
        self.assertEqual(select_nodes(self.pool, object(), "probability_dynamic", cheap),
                         probability_ids(self.pool, 7))

    def test_zero_and_shared_experts_keep_mandatory_seed_cost(self):
        shared = replace(self.atlas, expert_ids_by_raw_layer=tuple(
            (raw, layer, tuple(range(8))) for raw, layer, _ in self.atlas.expert_ids_by_raw_layer))
        for count in range(8):
            features = selection_features(self.pool, shared, probability_ids(self.pool, count))
            self.assertEqual(features["expert_union_proxy"], 384)
        self.assertEqual(selection_features(self.pool, shared, ())["expected_committed_tokens_proxy"], 1)
        with self.assertRaises(ValueError):
            selection_features(self.pool, shared, (3,))

    def test_calibration_recovers_known_shape_offsets_and_expert_cost(self):
        offsets = tuple(100_000_000 + 800_000 * k * k for k in range(8))
        slope = 250_000.0
        points = []
        for k in range(8):
            for j in range(3):
                footprint = 384 + k * (48 + 12 * j)
                points.append(dict(draft_node_count=k, expert_union=footprint,
                    cycle_without_selector_ns=offsets[k] + slope * footprint))
        model, diagnostics = fit_cost_model(points)
        self.assertAlmostEqual(model.expert_ns, slope)
        for k in range(8):
            self.assertAlmostEqual(model.size_intercept_ns[k], offsets[k])
        self.assertLess(diagnostics["rmse_ns"], 1e-6)
        self.assertEqual(SelectionCostModel.from_dict(model.to_dict()), model)

    def test_calibration_handles_zero_slope_and_nonnegative_boundaries(self):
        points = [dict(draft_node_count=k, expert_union=384 + k * (j + 1),
                       cycle_without_selector_ns=1000 + 100 * k)
                  for k in range(8) for j in range(3)]
        model, diagnostic = fit_cost_model(points)
        self.assertEqual(model.expert_ns, 0)
        self.assertTrue(diagnostic["expert_slope_at_zero"])
        for row in points:
            row["cycle_without_selector_ns"] = 5 * row["expert_union"] + (0 if row["draft_node_count"] == 0 else 200)
        model, diagnostic = fit_cost_model(points)
        self.assertGreaterEqual(min(model.size_intercept_ns), 0)
        self.assertGreaterEqual(model.expert_ns, 0)
        self.assertLess(diagnostic["rmse_ns"], 1e-6)

    def test_rejects_missing_or_invalid_calibration(self):
        points = [dict(draft_node_count=k, expert_union=384, cycle_without_selector_ns=1000)
                  for k in range(8)]
        with self.assertRaises(ValueError):
            fit_cost_model(points[:-1])
        for field, value in (("draft_node_count", True), ("expert_union", 0),
                             ("cycle_without_selector_ns", float("nan")),
                             ("cycle_without_selector_ns", 0)):
            changed = [dict(x) for x in points]
            changed[0][field] = value
            with self.assertRaises(ValueError):
                fit_cost_model(changed)
        with self.assertRaises(ValueError):
            select_nodes(self.pool, self.atlas, "expert_cost_fixed")
        with self.assertRaises(ValueError):
            SelectionCostModel.from_dict({**self.model().to_dict(), "extra": 1})


if __name__ == "__main__":
    unittest.main()
