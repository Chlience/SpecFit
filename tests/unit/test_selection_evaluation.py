from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from specfit.f0_direct_analysis import validate_trial
from specfit.m1_capture_replay import same_json_value
from specfit.selection_evaluation import (CALIBRATION_QUESTIONS, METHODS,
    calibration_schedule, evaluation_schedule, prompt_bootstrap, read_artifact,
    read_shards, summarize_measurements, validate_records, write_artifact)
from tests.unit import test_f0_direct as direct_fixture
from tests.unit import test_trace_schema as trace_fixture


class SelectionEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        direct_fixture.F0DirectTest.setUpClass()
        cls.fixture = direct_fixture.F0DirectTest()

    def test_fixed_schedule_covers_all_methods_modes_and_repetitions(self):
        schedule = evaluation_schedule("q406-c15")
        self.assertEqual(schedule, evaluation_schedule("q406-c15"))
        self.assertEqual(len(schedule), 50)
        counts = Counter((r["method"], r["mode"], r["phase"]) for r in schedule)
        for method in METHODS:
            for mode in ("cold", "natural"):
                self.assertEqual([counts[method, mode, p] for p in ("warmup", "audit", "measurement")],
                                 [1, 1, 3])
        self.assertEqual([r["execution_index"] for r in schedule], list(range(50)))

    def test_calibration_covers_zero_through_seven_without_evaluation_prompts(self):
        self.assertFalse(set(CALIBRATION_QUESTIONS) & set(range(406, 477, 10)))
        for qid in CALIBRATION_QUESTIONS:
            rows = calibration_schedule(qid)
            self.assertEqual(len(rows), 80)
            self.assertEqual(set(r["mode"] for r in rows), {"natural"})
            counts = Counter((r["method"], r["draft_node_count"]) for r in rows if r["phase"] == "measurement")
            self.assertEqual(len(counts), 16)
            self.assertEqual(set(counts.values()), {3})

    def test_calibration_tokenization_satisfies_the_immutable_trace_interface(self):
        from scripts.benchmark_qwen3_selection import prepare_calibration

        encoded = dict(input_ids=[10, 11], attention_mask=[1, 1])

        class Tokenizer:
            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
                return "<user>" + messages[0]["content"]

            def __call__(self, rendered):
                return encoded

        prepared = prepare_calibration(Tokenizer(), dict(prompts=[dict(question_id=401, raw_prompt="Question")]))[401]
        trace = replace(trace_fixture.TraceSchemaTest().manifest(), input_ids=prepared["input_ids"],
            attention_mask=prepared["attention_mask"], raw_prompt=prepared["raw_prompt"],
            rendered_prompt=prepared["rendered"], max_new_tokens=prepared["max_new_tokens"])
        encoded["input_ids"].append(12)
        encoded["attention_mask"].append(1)
        self.assertEqual(trace.input_ids, (10, 11))
        self.assertEqual(trace.attention_mask, (1, 1))
        self.assertEqual(trace.raw_prompt, "Question")

    def test_variable_node_shapes_preserve_cache_routes_and_serial_zero_semantics(self):
        original = self.fixture.selected
        try:
            for selected in ((), (1,), original):
                self.fixture.selected = selected
                record = self.fixture.trial()
                record["acceptance"]["verifier_token_ids"] = [0] * (1 + len(selected))
                record.update(method="expert_cost_dynamic", draft_forward_calls=5)
                job = dict(job_id="job", selected_raw_ids=list(selected))
                validate_trial(record, job=job, state=self.fixture.state,
                    source_state=self.fixture.source_state, variable_budget=True)
                if not selected:
                    with self.assertRaises(ValueError):
                        validate_trial({**record, "method": "target_only"}, job=job,
                            state=self.fixture.state, source_state=self.fixture.source_state, variable_budget=True)
                    record.update(method="target_only", draft_forward_calls=0)
                    record["timings"].update(explicit_selector_ns=0, proposal_wall_ns=0)
                    validate_trial(record, job=job, state=self.fixture.state,
                        source_state=self.fixture.source_state, variable_budget=True)
        finally:
            self.fixture.selected = original

    def test_result_files_are_exclusive_and_bound_to_bytes(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "result.json"
            descriptor = write_artifact(path, dict(status="complete"))
            self.assertEqual(read_artifact(path, descriptor), dict(status="complete"))
            with self.assertRaises(FileExistsError):
                write_artifact(path, {})
            path.write_text('{"status":"changed"}\n')
            with self.assertRaises(ValueError):
                read_artifact(path, descriptor)
            for raw in ('{"a":1,"a":2}', '{"x":NaN}', '{"x":1e1000}'):
                path.write_text(raw)
                with self.assertRaises(ValueError):
                    read_artifact(path)
            with self.assertRaises(ValueError):
                list(read_shards(dict(shards=[dict(basename="../result.json")]), Path(root)))

    def test_audited_records_reject_missing_cycles_gc_changes_and_false_commit_checks(self):
        schedule = [dict(execution_index=i, method="probability_fixed", mode="natural",
                         phase=phase, repetition=0)
                    for i, phase in enumerate(("warmup", "audit", "measurement"))]
        records = []
        for job in schedule:
            record = self.fixture.trial()
            record.update(job, question_id=406, draft_forward_calls=5,
                preparation_method="checkpoint_restore", gc_enabled_during_cycle=False,
                cycle_wall_interval_ns=[100, 10100],
                boundary_checks=dict(generation_state_exact=True, expert_weights_exact=True))
            if job["phase"] == "audit":
                record["commit_audit"] = dict(kv_layer_count=48, iteration=15, base_kv_length=128,
                    verification_node_count=8, verification_positions=[0], rejected_node_ids=list(range(1, 8)),
                    aux_prefix_length=128, aux_length_after=129,
                    **{name: True for name in ("kv_prefix_exact", "kv_selected_suffix_exact", "kv_length_exact",
                        "aux_length_exact", "aux_prefix_exact", "aux_selected_suffix_exact", "pending_seed_exact")})
            records.append(record)
        kwargs = dict(state=self.fixture.state, source_state=self.fixture.source_state,
                      expected_choices={("probability_fixed", None): self.fixture.selected})
        validate_records(records, schedule, **kwargs)
        for mutation in (lambda rs: rs.pop(), lambda rs: rs[1].update(gc_enabled_during_cycle=True),
                lambda rs: rs[1]["commit_audit"].update(kv_prefix_exact=False),
                lambda rs: rs[2].update(cycle_wall_interval_ns=[100, 10101])):
            changed = deepcopy(records)
            mutation(changed)
            with self.assertRaises(ValueError):
                validate_records(changed, schedule, **kwargs)

    def test_summary_uses_ratio_of_sums_and_rejects_missing_baselines(self):
        rows = []
        for mode in ("cold", "natural"):
            for method in METHODS:
                for qid, length, elapsed in ((406, 1, 1e9), (416, 3, 2e9)):
                    for repetition in range(3):
                        rows.append(dict(phase="measurement", state_id=f"q{qid}-c0", question_id=qid,
                            repetition=repetition, mode=mode, method=method, selected_raw_ids=[], F_own=384,
                            acceptance=dict(committed_token_ids=[42] * length),
                            timings=dict(cycle_wall_ns=elapsed, explicit_selector_ns=0,
                                proposal_wall_ns=0, target_verification_cuda_ns=elapsed),
                            cache_counter_delta=dict(h2d_submissions=384)))
        summary = summarize_measurements(rows)
        for mode in ("cold", "natural"):
            for method in METHODS:
                self.assertAlmostEqual(summary["by_mode"][mode]["methods"][method]["tokens_per_second"], 4 / 3)
        with self.assertRaises(ValueError):
            summarize_measurements([r for r in rows if r["method"] != "target_only"])

    def test_prompt_bootstrap_keeps_paired_states(self):
        baseline = {f"q{q}-c{c}": dict(question_id=q, committed_tokens=1 + c, cycle_ns=100 + q)
                    for q in (406, 416) for c in range(4)}
        method = deepcopy(baseline)
        for row in method.values():
            row["cycle_ns"] *= 0.8
        interval = prompt_bootstrap(method, baseline)["percentile_95_interval"]
        self.assertAlmostEqual(interval[0], 0.25)
        self.assertAlmostEqual(interval[1], 0.25)

    def test_summary_reproduces_after_artifact_serialization(self):
        rows = []
        for mode in ("cold", "natural"):
            for method in METHODS:
                count = 0 if method == "target_only" else 5 if method.endswith("dynamic") else 7
                rows.append(dict(phase="measurement", state_id="q406-c0", question_id=406,
                    repetition=0, mode=mode, method=method, selected_raw_ids=list(range(1, count + 1)),
                    F_own=384, acceptance=dict(committed_token_ids=[42]),
                    timings=dict(cycle_wall_ns=1000, explicit_selector_ns=0,
                        proposal_wall_ns=0, target_verification_cuda_ns=1000),
                    cache_counter_delta=dict(h2d_submissions=384)))
        expected = summarize_measurements(rows)
        with TemporaryDirectory() as root:
            path = Path(root) / "summary.json"
            descriptor = write_artifact(path, expected)
            observed = read_artifact(path, descriptor)
        self.assertTrue(same_json_value(observed, summarize_measurements(rows)))


if __name__ == "__main__":
    unittest.main()
