from collections import Counter
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from specfit import selection_evaluation as base, selection_matrix as matrix
from specfit.selection_methods import SelectionCostModel
from tests.unit import test_f0_direct as fixture


class SelectionMatrixTest(unittest.TestCase):
    def test_schedule_has_every_combination_and_an_audit_before_two_samples(self):
        jobs = matrix.schedule("q416-c37")
        self.assertEqual(jobs, matrix.schedule("q416-c37"))
        self.assertEqual(len(jobs), 120)
        self.assertEqual([j["execution_index"] for j in jobs], list(range(120)))
        groups = Counter((j["execution"], j["method"], j["mode"], j["phase"]) for j in jobs)
        for executor in matrix.EXECUTORS:
            for method in matrix.METHODS:
                for mode in ("cold", "natural"):
                    self.assertEqual(groups[executor, method, mode, "audit"], 1)
                    self.assertEqual(groups[executor, method, mode, "measurement"], 2)
        self.assertTrue(all(j["phase"] == "audit" for j in jobs[:40]))
        self.assertEqual(len(base.evaluation_schedule("q416-c37")), 50)
        self.assertNotIn("predicted_cross_layer", matrix.EXECUTORS)

    def test_calibration_uses_all_executors_and_original_disjoint_probes(self):
        jobs = matrix.schedule(401, calibration=True)
        self.assertEqual(len(jobs), 192)
        self.assertEqual({j["mode"] for j in jobs}, {"natural"})
        self.assertEqual({j["draft_node_count"] for j in jobs}, set(range(8)))
        counts = Counter((j["execution"], j["method"], j["draft_node_count"]) for j in jobs if j["phase"] == "measurement")
        self.assertEqual(len(counts), 64)
        self.assertEqual(set(counts.values()), {2})

    def test_optional_scored_controls_add_only_the_two_fixed_tree_comparisons(self):
        jobs = matrix.schedule("q416-c37", scored_controls=True)
        self.assertEqual(len(jobs), 132)
        extra = [j for j in jobs if j["execution"] in matrix.SCORED_CONTROLS]
        self.assertEqual(len(extra), 12)
        self.assertEqual({j["method"] for j in extra}, {"expert_cost_dynamic"})

    def test_cost_models_are_fit_per_executor_without_cross_group_samples(self):
        shards = []
        for qid in base.CALIBRATION_QUESTIONS:
            choices = {m: {str(k): dict(expert_union_proxy=400 + 20*k + (30 if m.endswith("experts") else 0))
                for k in range(8)} for m in matrix.CALIBRATION_CONFIG["selection_probes"]}
            records = []
            for job in matrix.schedule(qid, calibration=True):
                factor = 1 + matrix.EXECUTORS.index(job["execution"])
                footprint = choices[job["method"]][str(job["draft_node_count"] )]["expert_union_proxy"]
                records.append(dict(job, timings=dict(cycle_wall_ns=factor*(1000+job["draft_node_count"]*10+footprint)+7,
                    explicit_selector_ns=7)))
            shards.append(dict(question_id=qid, choices=choices, records=records))
        models, diagnostics = matrix.fit_models(shards)
        self.assertEqual(set(models), set(matrix.EXECUTORS))
        for i, execution in enumerate(matrix.EXECUTORS):
            self.assertAlmostEqual(models[execution].expert_ns, i+1)
            self.assertEqual(diagnostics[execution]["point_count"], 48)
        shards[0]["records"].pop()
        with self.assertRaisesRegex(ValueError, "exactly two"):
            matrix.fit_models(shards)

    def test_manifest_freezes_executor_specific_choices_and_preserves_population(self):
        fixture.F0DirectTest.setUpClass()
        original = fixture.F0DirectTest().state
        states = [SimpleNamespace(state_id=f"q{q}-c{c}", question_id=q, cycle_index=c,
            pool=original.pool, atlas=original.atlas) for q in range(406, 477, 10) for c in range(4)]
        models = {e: SelectionCostModel((1000,)*8, (1000,)*8, 1.0) for e in matrix.EXECUTORS}
        models["ready_demand"] = SelectionCostModel((1.,)*(1)+(1e9,)*7, (1.,)+(1e9,)*7, 0.)
        result = matrix.evaluation_manifest(SimpleNamespace(projection=SimpleNamespace(states=states)),
            git_commit="frozen", measurement_identity={}, calibration_descriptor={}, models=models)
        self.assertEqual(result["counts"]["cycles"], 3840)
        self.assertEqual(result["counts"]["measurements"], 2560)
        self.assertEqual([len(b) for b in result["execution_batches"]], [8]*4)
        self.assertEqual(sum(result["execution_batches"], []), result["state_order"])
        for state in result["states"]:
            self.assertEqual(state["choices"]["ready_demand"]["probability_dynamic"]["selected_raw_ids"], [])
            self.assertTrue(state["choices"]["demand"]["probability_dynamic"]["selected_raw_ids"])
            for e in matrix.EXECUTORS:
                self.assertEqual(state["choices"][e]["target_only"]["selected_raw_ids"], [])
                self.assertEqual(len(state["choices"][e]["probability_fixed"]["selected_raw_ids"]), 7)
        controls = matrix.evaluation_manifest(SimpleNamespace(projection=SimpleNamespace(states=states)),
            git_commit="frozen", measurement_identity={}, calibration_descriptor={}, models=models, scored_controls=True)
        self.assertEqual(controls["counts"]["cycles"], 4224)
        for state in controls["states"]:
            for execution in matrix.SCORED_CONTROLS:
                self.assertEqual(state["choices"][execution]["expert_cost_dynamic"],
                                 state["choices"]["predicted_same_layer"]["expert_cost_dynamic"])

    def test_references_bind_the_exact_query_before_formal_samples(self):
        reference = dict(method="target_only", execution="demand", selected_raw_ids=[],
            pool_fingerprint="pool", routes={}, acceptance={}, logits_shape=[1,1,8],
            logits_dtype="torch.bfloat16", logits_sha256="a"*64)
        jobs = [dict(execution="ready_demand", method="target_only", mode="cold", phase=p,
                     repetition=0, execution_index=i) for i, p in enumerate(("audit", "measurement"))]
        rows = [dict(reference, **{k:v for k,v in j.items()}, query_reference_sha256=matrix.canonical_json_sha256(reference),
            query_collection_finished_ns=10, cycle_wall_interval_ns=[20,30]) for j in jobs]
        arguments = dict(state=None, source_state={}, choices={("ready_demand","target_only",None):()},
            references={"target:":reference})
        with patch.object(matrix,"validate_cycle"), patch.object(base,"validate_audit"):
            matrix.validate_records(rows, jobs, **arguments)
            for change in (lambda r:r[1].update(logits_sha256="b"*64),
                           lambda r:r[1].update(query_reference_sha256="b"*64),
                           lambda r:r[1].update(query_collection_finished_ns=21),
                           lambda r:r[1].update(execution="demand")):
                bad=deepcopy(rows); change(bad)
                with self.assertRaises(ValueError):
                    matrix.validate_records(bad, jobs, **arguments)
            with self.assertRaisesRegex(ValueError,"matching audited warmup"):
                matrix.validate_records(rows[1:],jobs[1:],**arguments)

    def test_query_keys_distinguish_target_only_from_draft_with_zero_nodes(self):
        self.assertNotEqual(matrix.query_key("target_only",()), matrix.query_key("probability_dynamic",()))
        self.assertEqual(matrix.query_key("probability_fixed",(1,2)),matrix.query_key("expert_cost_fixed",(1,2)))

    def test_combining_rejects_missing_duplicate_or_smoke_batches(self):
        with self.assertRaisesRegex(ValueError,"four separate"):
            matrix.combine([], {}, None)
        with TemporaryDirectory() as directory:
            paths=[Path(directory)/f"batch-{i}.json" for i in range(4)]
            for p in paths:
                base.write_artifact(p,dict(batch=0,run_scope="batch"))
            with self.assertRaisesRegex(ValueError,"unique and ordered"):
                matrix.combine(paths,{},None)


if __name__ == "__main__":
    unittest.main()
