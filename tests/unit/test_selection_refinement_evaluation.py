from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from specfit import selection_refinement_evaluation as experiment
from specfit import selection_evaluation as base, selection_matrix as matrix
from specfit.experiment import canonical_json_sha256
from specfit.m1_capture_replay import logical_state_sha256
from specfit.selection_refinement import (CacheCostModel, VARIANTS, resident_bitmap,
                                           residents_from_logical_state)
from tests.unit import test_selection_refinement as fixtures
from tests.unit import test_m1_artifacts as m1fixtures
from tests.unit.r6a_fixture import R6AFixture


class SelectionRefinementEvaluationTest(unittest.TestCase):
    def fixture(self):
        pool, atlas = fixtures.SelectionRefinementTest.fixture()
        state = SimpleNamespace(state_id="q404-c0", question_id=404, cycle_index=0, pool=pool, atlas=atlas)
        legacy = fixtures.SelectionRefinementTest.model()
        cache = CacheCostModel(legacy.size_intercept_ns, legacy.expert_ns, 100000.0)
        models = dict(by_mode={mode: dict(legacy=legacy.to_dict(), cache=cache.to_dict())
                              for mode in ("cold", "natural")}, acceptance_exponent=1.0,
                      training_questions=[401, 402, 403], guard_margin=0.02)
        return state, m1fixtures.M1ArtifactTest._initial_state(), models

    def test_cohorts_are_disjoint_and_schedules_preserve_audit_and_repetition_coverage(self):
        all_questions = [q for qs in experiment.SPLITS.values() for q in qs]
        self.assertEqual(len(set(all_questions)), len(all_questions))
        for stage, count in (("calibration", 96), ("development", 72), ("holdout", 108)):
            jobs = experiment.schedule(stage, "state")
            self.assertEqual(len(jobs), count)
            self.assertEqual(jobs, experiment.schedule(stage, "state"))
            self.assertEqual([r["execution_index"] for r in jobs], list(range(count)))
            groups = Counter((r["method"], r.get("draft_node_count"), r["mode"], r["phase"]) for r in jobs)
            for (method, nodes, mode, phase), number in groups.items():
                self.assertEqual(number, 1 if phase == "audit" else experiment.REPETITIONS[stage])
            first_measurement = next(i for i, r in enumerate(jobs) if r["phase"] == "measurement")
            self.assertTrue(all(r["phase"] == "measurement" for r in jobs[first_measurement:]))
        smoke = experiment.schedule("calibration", "state", smoke=True)
        self.assertEqual(len(smoke), 36)
        self.assertEqual({r["draft_node_count"] for r in smoke}, {0, 4, 7})
        with self.assertRaises(ValueError):
            experiment.schedule("holdout", "state", smoke=True)

    def test_manifest_binds_dataset_models_and_prompt_split_before_any_measurement(self):
        _, _, models = self.fixture()
        lock = json.loads(Path("workloads/gsm8k_specbench.lock.json").read_text())["dataset"]
        descriptor = dict(basename=lock["filename"], size_bytes=lock["size_bytes"], sha256=lock["sha256"])
        examples = [SimpleNamespace(question_id=q, prompt=f"prompt-{q}") for q in range(401, 481)]
        with patch.object(experiment, "load_gsm8k_specbench", return_value=examples), \
             patch.object(base, "file_identity", return_value=descriptor):
            value = experiment.build_manifest(stage="holdout", git_commit="source", measurement_identity={},
                dataset_path=Path("unused"), calibration_descriptor={"sha256": "frozen"}, models=models)
            self.assertEqual(len(value["states"]), 32)
            self.assertEqual([len(b) for b in value["batches"]], [4] * 8)
            self.assertEqual({s["question_id"] for s in value["states"]}, set(experiment.SPLITS["holdout"]))
            models["acceptance_exponent"] = 3.0
            self.assertEqual(value["models"]["acceptance_exponent"], 1.0)
            smoke = experiment.build_manifest(stage="calibration", git_commit="source", measurement_identity={},
                dataset_path=Path("unused"), smoke=True)
            self.assertEqual(smoke["batches"], [["q401-c0", "q401-c4"]])
            self.assertEqual(smoke["scope"], "smoke")
        with patch.object(base, "file_identity", return_value={**descriptor, "sha256": "wrong"}):
            with self.assertRaisesRegex(ValueError, "dataset lock"):
                experiment.build_manifest(stage="calibration", git_commit="source", measurement_identity={},
                    dataset_path=Path("unused"))

    def test_choices_use_mode_specific_residency_and_keep_the_fast_ablation_identical(self):
        state, logical, models = self.fixture()
        choices = experiment.choices_for(state, stage="development", logical_cache=logical, models=models)
        for mode in ("cold", "natural"):
            self.assertEqual(set(choices[mode]), set(experiment.REFINEMENT_METHODS))
            self.assertEqual(choices[mode]["expert_fast_dynamic"], choices[mode]["expert_cost_dynamic"])
            for row in choices[mode].values():
                self.assertLessEqual(len(row["selected_raw_ids"]), 7)
                self.assertLessEqual(row["initial_missing"], row["expert_union"])
                if mode == "cold":
                    self.assertEqual(row["initial_missing"], row["expert_union"])

    def test_fit_rejects_evaluation_prompts_or_incomplete_training_coverage(self):
        with self.assertRaisesRegex(ValueError, "training cohort"):
            experiment.fit_models([])
        shards = [dict(question_id=q, cycle_index=c) for q in experiment.SPLITS["calibration"]
                  for c in experiment.BOUNDARIES["calibration"]]
        shards[0]["question_id"] = experiment.SPLITS["holdout"][0]
        with self.assertRaisesRegex(ValueError, "training cohort"):
            experiment.fit_models(shards)

    def test_cost_fits_are_separate_by_cache_mode_and_use_only_formal_training_samples(self):
        state, _, _ = self.fixture()
        raw_pool = dict(seed_token_id=state.pool.seed_token_id, top_k=2, max_depth=5,
            raw_candidate_count=18, fingerprint=state.pool.fingerprint,
            candidates=[asdict(n) for n in state.pool.nodes])
        shards = []
        for qid in experiment.SPLITS["calibration"]:
            for cycle in experiment.BOUNDARIES["calibration"]:
                sid = f"q{qid}-c{cycle}"
                choices = {mode: {} for mode in ("cold", "natural")}
                for mode in choices:
                    for method in experiment.PROBES:
                        for k in range(8):
                            f = 384 + k * (40 + qid % 10 * 4 + cycle + 9 * method.endswith("experts"))
                            n = f if mode == "cold" else 50 + cycle * 5 + (qid % 10) * 11
                            choices[mode][experiment.choice_key(method, k)] = dict(expert_union=f, initial_missing=n)
                rows = []
                for job in experiment.schedule("calibration", sid):
                    f = choices[job["mode"]][experiment.choice_key(job["method"], job["draft_node_count"])]
                    elapsed = 50e6 + job["draft_node_count"] * 1e6 + 100000 * f["expert_union"] + 200000 * f["initial_missing"]
                    if job["phase"] == "audit":
                        elapsed *= 100  # Audits must not contaminate the fit.
                    rows.append(dict(job, timings=dict(cycle_wall_ns=elapsed + 17, explicit_selector_ns=17)))
                shards.append(dict(state_id=sid, question_id=qid, cycle_index=cycle, choices=choices, records=rows,
                    pool=raw_pool, route_probe=dict(acceptance=dict(verifier_token_ids=[-1] * 19))))
        models, diagnostics = experiment.fit_models(shards)
        self.assertEqual(models["training_questions"], [401, 402, 403])
        self.assertEqual(diagnostics["cost"]["cold"]["legacy"]["point_count"], 144)
        self.assertEqual(models["by_mode"]["cold"]["cache"]["missing_ns"], 0)
        self.assertAlmostEqual(models["by_mode"]["natural"]["cache"]["missing_ns"], 200000, delta=1)
        self.assertLess(diagnostics["cost"]["natural"]["cache"]["mean_absolute_relative_error"], 1e-7)
        shards[0]["records"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete calibration"):
            experiment.fit_models(shards)

    def make_bound_shard(self, directory):
        state, logical, models = self.fixture()
        choices = experiment.choices_for(state, stage="development", logical_cache=logical, models=models)
        jobs = experiment.schedule("development", state.state_id)
        manifest = dict(protocol=experiment.PROTOCOL, stage="development", models=models, guard_margin=0.02,
            prompts=[dict(question_id=404, raw_prompt="prompt")], config=experiment.CONFIG,
            states=[dict(state_id=state.state_id, question_id=404, cycle_index=0, schedule=jobs)])
        digest = canonical_json_sha256(manifest)
        raw_pool = dict(seed_token_id=state.pool.seed_token_id, top_k=2, max_depth=5, raw_candidate_count=18,
                        fingerprint=state.pool.fingerprint, candidates=[asdict(n) for n in state.pool.nodes])
        probe = dict(method="route_probe", execution="demand", mode="natural", selected_raw_ids=list(range(1, 19)),
            routes=R6AFixture.routes(state.pool.fingerprint, tuple(range(19))), pre_logical_state=logical,
            cycle_wall_interval_ns=[1, 2])
        frozen = dict(state_id=state.state_id, manifest_sha256=digest, choices=choices,
            models_sha256=canonical_json_sha256(models), route_probe_sha256=canonical_json_sha256(probe), frozen_at_ns=10)
        choice_descriptor = base.write_artifact(directory / "choices.json", frozen)
        references = {}
        for job in jobs:
            ids = choices[job["mode"]][job["method"]]["selected_raw_ids"]
            key = matrix.query_key(job["method"], ids)
            references[key] = dict(method="target_only" if job["method"] == "target_only" else "frozen_selection",
                execution="demand", mode="natural", selected_raw_ids=ids, pool_fingerprint=state.pool.fingerprint,
                routes={}, acceptance={}, logits_shape=[1, len(ids) + 1, 8], logits_dtype="torch.bfloat16",
                logits_sha256="a" * 64, cycle_wall_interval_ns=[11, 19])
        reference_descriptor = base.write_artifact(directory / "queries.json", dict(state_id=state.state_id,
            manifest_sha256=digest, collection_finished_ns=20, references=references))
        cold = deepcopy(logical)
        for layer in cold["layers"]:
            for slot in layer["slots"]:
                slot["expert_id"] = None
        records = []
        for i, job in enumerate(jobs):
            ids = choices[job["mode"]][job["method"]]["selected_raw_ids"]
            reference = references[matrix.query_key(job["method"], ids)]
            pre = logical if job["mode"] == "natural" else cold
            records.append(dict(reference, **{**job, "cycle_wall_interval_ns": [30 + 2*i, 31 + 2*i]},
                pre_logical_state=pre, query_reference_sha256=canonical_json_sha256(reference),
                choice_manifest_sha256=choice_descriptor["sha256"], query_collection_finished_ns=20,
                selector_resident_sha256=canonical_json_sha256(hex(resident_bitmap(residents_from_logical_state(pre))))
                if job["method"] in VARIANTS[1:] else None))
        shard = dict(status="complete", state_id=state.state_id, question_id=404, cycle_index=0,
            manifest_sha256=digest, prepared_prompt=dict(question_id=404, raw_prompt="prompt",
                input_ids=[1, 2], attention_mask=[1, 1]), reference_history=dict(commit_audits=[],
                reference_probability_nodes=7, base_kv_length=2, committed_token_ids=[], selected_raw_ids=[]),
            source_state=dict(base_kv_length=2, pre_logical_state_sha256=logical_state_sha256(logical)),
            pool=raw_pool, route_probe=probe, choices=choices, choice_manifest=choice_descriptor,
            query_references=reference_descriptor, records=records, checkpoint={})
        return json.loads(json.dumps(shard)), manifest

    def test_new_binding_checks_reject_changed_decisions_cache_inputs_and_future_references(self):
        # The existing matrix tests cover CUDA/cache-event and commit audits.
        # Isolate this protocol's additional decision/input/timestamp binding.
        with TemporaryDirectory() as temporary, patch.object(matrix, "validate_cycle"), \
             patch.object(base, "validate_audit"), patch.object(base, "validate_checkpoint"):
            directory = Path(temporary)
            shard, manifest = self.make_bound_shard(directory)
            experiment.validate_shard(shard, manifest, directory)
            for mutation in (lambda s: s["records"][1].update(logits_sha256="b" * 64),
                lambda s: s["records"][1].update(query_collection_finished_ns=10000),
                lambda s: s["choices"]["cold"]["target_only"].update(selected_raw_ids=[1]),
                lambda s: s["records"][1].update(choice_manifest_sha256="b" * 64),
                lambda s: s["records"].pop(),
                lambda s: next(r for r in s["records"] if r["method"] == "expert_cache_dynamic").update(
                    selector_resident_sha256="b" * 64)):
                changed = deepcopy(shard)
                mutation(changed)
                with self.assertRaises(ValueError):
                    experiment.validate_shard(changed, manifest, directory)

    def test_combine_rejects_missing_or_reordered_batches(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            experiment.combine([], {"batches": [["a"]]}, Path("unused"))
        with self.assertRaisesRegex(ValueError, "order"):
            experiment.combine([dict(batch=1)], {"batches": [["a"]]}, Path("unused"))
        environments = [dict(gpu_name="NVIDIA A100-PCIE-40GB", server="a100-host", gpu_index=0),
            dict(gpu_name="NVIDIA GeForce RTX 4090", server="rtx4090-host", gpu_index=1, slots_per_layer=32)]
        with patch.object(experiment, "validate_result", return_value=[]), \
             self.assertRaisesRegex(ValueError, "one device"):
            experiment.combine([dict(batch=b, environment=e) for b, e in enumerate(environments)],
                               {"batches": [["a"], ["b"]]}, Path("unused"))


if __name__ == "__main__":
    unittest.main()
