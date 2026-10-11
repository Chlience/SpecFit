from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from specfit.experiment import canonical_json_sha256
from specfit.f0_direct import (CONFIG, PACKED_SOURCE, PROTOCOL, COEFFICIENTS,
    DirectSources, build_manifest, build_schedules, expert_union, pilot_state_ids,
    select_ids, unique_jobs, validate_manifest, write_json_exclusive)
from specfit.f0_direct_analysis import (COUNTERS, EXPERT_BYTES, analyze_result,
    cache_totals, summarize_rows, validate_outcome, validate_trial)
from specfit.m1_capture_replay import (advance_logical_state, logical_state_sha256,
    read_logical_state, read_routes)
from specfit.m1_capture_validation import M1StateProjection, M1CaptureProjection, M1_QUESTION_IDS
from specfit.m1_screen import score_candidate_subsets
from specfit.r6a_experiment import FrozenStateRecord, PoolRecord
from specfit.r6a_identity import R6A_MODEL_IDENTITIES
from specfit.r6a_selection_contract import enumerate_ancestor_closed_subsets
from specfit.r6a_tb2_contract import R6A_TB2_IMAGE_ID, R6A_TB2_NUMERIC_SETTINGS
from tests.unit.r6a_fixture import R6AFixture
from tests.unit import test_m1_artifacts as m1fixture


class F0DirectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = PoolRecord.read(R6AFixture.pool())
        cls.atlas = read_routes(R6AFixture.routes(cls.pool.fingerprint, tuple(range(19))),
                               "atlas", cls.pool, tuple(range(19)))
        cls.selected = select_ids(cls.pool, cls.atlas, "p7")
        cls.pre = m1fixture.M1ArtifactTest._initial_state()
        cls.state = M1StateProjection("q406-c15", 406, 15, cls.pool,
            FrozenStateRecord("a" * 64, (tuple(range(48)),) * 48), cls.atlas)
        cls.source_state = {"state_id": cls.state.state_id, "base_kv_length": 128,
            "pre_logical_state_sha256": logical_state_sha256(cls.pre)}
        cls.job = {"job_id": "job", "state_id": cls.state.state_id,
                   "selected_raw_ids": list(cls.selected), "policies": ["p7", "f0"]}

    def trial(self, mode="natural", capacity=48):
        pre = deepcopy(self.pre)
        if mode == "cold":
            for layer in pre["layers"]:
                layer.update(lru_sequence=0, submission_sequence=0, capacity=capacity, slots=layer["slots"][:capacity])
                layer["counters"] = {k: 0 for k in layer["counters"]}
                for slot in layer["slots"]:
                    slot.update(expert_id=None, generation=0, last_used_sequence=0, ready=False)
        pre, snapshot = read_logical_state(pre, "test pre", expected_capacity=capacity)
        routes = R6AFixture.routes(self.pool.fingerprint, (0, *self.selected))
        route = read_routes(routes, "own", self.pool, (0, *self.selected))
        post, _ = advance_logical_state(pre, snapshot, route)
        counts = {k: cache_totals(post)[k] - cache_totals(pre)[k] for k in COUNTERS}
        union = expert_union(route, self.selected)
        return {"execution_index": 0, "job_id": "job", "state_id": self.state.state_id,
            "mode": mode, "selected_raw_ids": list(self.selected), "pool_fingerprint": self.pool.fingerprint,
            "reference_cache_sha256": self.source_state["pre_logical_state_sha256"],
            "pre_logical_state": pre, "post_logical_state_sha256": logical_state_sha256(post),
            "routes": routes, "F_own": sum(map(len, union)), "max_own_layer_union": max(map(len, union)),
            "cache_counter_delta": counts, "logical_h2d_bytes": counts["h2d_submissions"] * EXPERT_BYTES,
            "acceptance": {"verifier_token_ids": [0] * 8, "committed_token_ids": [42],
                "accepted_raw_ids": [], "accepted_draft_tokens": 0,
                "next_pending_seed_token_id": 0, "kv_length_after": 129},
            "prepare_wall_ns": 1000, "timings": {"cycle_wall_ns": 10000,
                "explicit_selector_ns": 100, "proposal_wall_ns": 1000,
                "target_verification_cuda_ns": 8000}, "commit_audit": None}

    def validate(self, record):
        return validate_trial(record, job=self.job, state=self.state, source_state=self.source_state)

    def test_cold_and_natural_over_capacity_are_measured(self):
        for mode in ("cold", "natural"):
            record = self.trial(mode)
            self.assertGreater(self.validate(record)["max_own_layer_union"], 48)
            if mode == "cold":
                self.assertEqual(record["cache_counter_delta"]["h2d_submissions"], record["F_own"])

    def test_state_slots_fix_the_cache_capacity(self):
        # A second device measures at fewer expert slots; the cycle state carries that capacity.
        record = self.trial("cold", capacity=32)
        device = SimpleNamespace(state_id=self.state.state_id, pool=self.pool, slots_per_layer=32)
        result = validate_trial(record, job=self.job, state=device, source_state=self.source_state)
        self.assertEqual(result["F_own"], record["cache_counter_delta"]["h2d_submissions"])
        self.assertGreater(record["cache_counter_delta"]["evictions"], 0)
        with self.assertRaises(ValueError):
            self.validate(record)

    def test_restoration_measurements_cannot_enter_pilot_analysis(self):
        for phase in ("warmups", "audits", "trials"):
            with self.assertRaisesRegex(ValueError, "cannot enter the F0 pilot"):
                analyze_result({phase: [{"preparation_method": "checkpoint_restore"}]}, {}, None)

    def test_reject_changed_source_cache_route_counter_and_tokens(self):
        for field, value in (("reference_cache_sha256", "b" * 64),
                             ("post_logical_state_sha256", "b" * 64),
                             ("logical_h2d_bytes", 1), ("selected_raw_ids", [1]),
                             ("F_own", 1)):
            with self.subTest(field=field):
                trial = self.trial()
                trial[field] = value
                with self.assertRaises(ValueError):
                    self.validate(trial)
        for mutation in (
            lambda r: r["pre_logical_state"]["layers"][0].update(lru_sequence=999),
            lambda r: r["routes"]["rows"][0].update(expert_ids=list(range(1, 9))),
            lambda r: r["cache_counter_delta"].update(resident_hits=True),
            lambda r: r["acceptance"].update(committed_token_ids=[43]),
            lambda r: r["acceptance"].update(accepted_draft_tokens=False),
            lambda r: r["timings"].update(cycle_wall_ns=100),
        ):
            trial = self.trial()
            mutation(trial)
            with self.assertRaises(ValueError):
                self.validate(trial)
        trial = self.trial("cold")
        trial["pre_logical_state"] = deepcopy(self.pre)
        with self.assertRaises(ValueError):
            self.validate(trial)

    def test_maximal_acceptance_and_budget_truncation(self):
        trial = self.trial()
        verifier = [0] * 8
        verifier[0] = 1001
        verifier[self.selected.index(1) + 1] = 1003
        trial["acceptance"].update(verifier_token_ids=verifier, committed_token_ids=[42, 1001, 1003],
            accepted_raw_ids=[1, 3], accepted_draft_tokens=2, kv_length_after=131)
        self.validate(trial)
        trial["acceptance"].update(committed_token_ids=[42, 1001], accepted_raw_ids=[1],
            accepted_draft_tokens=1, kv_length_after=512, next_pending_seed_token_id=1003)
        validate_outcome(trial, self.pool, base_length=510, selected=self.selected)
        with self.assertRaises(ValueError):
            validate_outcome(trial, self.pool, base_length=128, selected=self.selected)

    def test_optional_prepare_timings_conserve_total_and_reject_invalid_values(self):
        trial = self.trial()
        self.validate(trial)  # Historical v2/v3 records have no breakdown.
        parts = {"initialization_and_prefill": 100, "reference_history": 700,
                 "boundary_validation_and_cache_setup": 200}
        trial["prepare_timings_ns"] = parts
        self.validate(trial)
        for invalid in (None, {**parts, "extra": 0}, {**parts, "reference_history": 699},
                        {**parts, "reference_history": -1}, {**parts, "reference_history": True}):
            with self.subTest(parts=invalid):
                trial["prepare_timings_ns"] = invalid
                with self.assertRaises(ValueError):
                    self.validate(trial)

    def test_second_sampled_cycle_and_all_samples_preserved(self):
        states = [{"question_id": q, "cycle_index": c, "state_id": f"q{q}-c{c}"}
                  for q in M1_QUESTION_IDS for c in (40, 3, 21, 15)]
        self.assertEqual(pilot_state_ids(states), [f"q{q}-c15" for q in M1_QUESTION_IDS])
        with self.assertRaises(ValueError):
            pilot_state_ids(states[:-1])

    def test_dedup_and_balanced_pair_schedule(self):
        for count in (1, 3, 14, 16):
            jobs = [{"job_id": f"job{i}"} for i in range(count)]
            schedule = build_schedules(jobs)["paired"]
            self.assertEqual(schedule, build_schedules(jobs)["paired"])
            self.assertEqual(len(schedule), count * 6)
            self.assertEqual(len({(r["job_id"], r["mode"], r["repetition"]) for r in schedule}), count * 6)
            for i in range(0, len(schedule), 2):
                a, b = schedule[i:i+2]
                self.assertEqual((a["job_id"], a["repetition"]), (b["job_id"], b["repetition"]))
                self.assertEqual((a["mode"], b["mode"]), ("cold", "natural") if i % 4 == 0 else ("natural", "cold"))
        states = [{"state_id": "s", "decisions": {p: {"selected_raw_ids": [1]} for p in ("p7", "f0")}}]
        self.assertEqual(unique_jobs(states, ["s"])[0]["policies"], ["p7", "f0"])

    def test_manifest_rebuild_rejects_choice_and_order_tampering(self):
        rows = score_candidate_subsets(pool=self.pool, atlas=self.atlas, frozen=self.state.frozen,
            subsets=tuple(sorted(enumerate_ancestor_closed_subsets(self.pool))))
        # These fixture rows are observations from the same pure M1 screen model.
        screen_rows = [{"selected_raw_ids": list(r.selected_raw_ids), "F_atlas": r.F_atlas,
                        "draft_utility": r.draft_utility} for r in rows]
        projections, raw, screens = [], [], []
        for q in M1_QUESTION_IDS:
            for cycle in (3, 15, 21, 40):
                sid = f"q{q}-c{cycle}"
                projections.append(replace(self.state, state_id=sid, question_id=q, cycle_index=cycle))
                raw.append({**self.source_state, "state_id": sid,
                            "full_atlas_routes": {"digest": self.atlas.digest}})
                screens.append({"state_id": sid, "candidate_sets": screen_rows})
        sources = DirectSources({"states": raw}, M1CaptureProjection("a" * 40, None, tuple(projections)),
                                {"states": screens})
        with patch("specfit.f0_direct.CANDIDATE_COUNT", len(rows) * 32):
            manifest = build_manifest(sources, git_commit="a" * 40)
            validate_manifest(manifest, sources, expected_git_commit="a" * 40)
            self.assertTrue(manifest["gate"]["proxy_capacity_diagnostics"])
            for key, value in (("coefficients", {}), ("pilot_state_ids", []), ("schedule", {})):
                invalid = deepcopy(manifest)
                invalid[key] = value
                with self.assertRaises(ValueError):
                    validate_manifest(invalid, sources, expected_git_commit="a" * 40)

    def result_fixture(self):
        sid = self.state.state_id
        decision = {"selected_raw_ids": list(self.selected), "F_atlas": 3000, "expected_committed_tokens": 4.5}
        manifest = {"git_commit": "a" * 40, "protocol": PROTOCOL, "states": [{"state_id": sid, "decisions": {
            "p7": decision, "f0": decision}}], "jobs": [self.job], "pilot_state_ids": [sid],
            "schedule": build_schedules([self.job]), "gate": {"decision_difference_count": 0},
            "coefficients": COEFFICIENTS}
        result = {"artifact_type": "f0_direct_result", "protocol": PROTOCOL, "status": "complete",
            "manifest_sha256": canonical_json_sha256(manifest), "config": CONFIG,
            "environment": {"git_commit": "a" * 40, "image_id": R6A_TB2_IMAGE_ID,
                "gpu_name": "NVIDIA A100-PCIE-40GB", "server": "a100-host", "device": "cuda:0",
                "numeric_settings": R6A_TB2_NUMERIC_SETTINGS, "packed_descriptor": PACKED_SOURCE,
                "models": R6A_MODEL_IDENTITIES}, "warmups": [], "audits": [], "trials": [],
            "selector_measurements": [{"state_id": sid, "policy": p, "selected_raw_ids": list(self.selected),
                "samples_ns": [t] * 6} for p, t in (("p7", 100), ("f0", 1100))]}
        for mode in ("cold", "natural"):
            result["warmups"].append(self.trial(mode))
            audit = self.trial(mode)
            audit["commit_audit"] = {**{k: True for k in ("kv_prefix_exact", "kv_selected_suffix_exact",
                "kv_length_exact", "aux_length_exact", "aux_prefix_exact", "aux_selected_suffix_exact", "pending_seed_exact")},
                "kv_layer_count": 48, "iteration": 15, "base_kv_length": 128, "verification_node_count": 8,
                "verification_positions": [0], "rejected_node_ids": list(range(1, 8)),
                "aux_prefix_length": 128, "aux_length_after": 129}
            result["audits"].append(audit)
        for i, scheduled in enumerate(manifest["schedule"]["paired"]):
            result["trials"].append({**self.trial(scheduled["mode"]), **scheduled, "execution_index": i})
        sources = SimpleNamespace(projection=SimpleNamespace(states=[self.state]), capture={"states": [self.source_state]})
        return result, manifest, sources

    def test_shared_execution_charges_each_selector_and_actual_tokens(self):
        result, manifest, sources = self.result_fixture()
        analysis = analyze_result(result, manifest, sources)
        natural = analysis["by_mode"]["natural"]
        self.assertAlmostEqual(natural["gain_vs_p7"], 10000 / 11000 - 1)
        self.assertEqual(natural["policies"]["f0"]["committed_tokens"], 3)
        self.assertEqual(natural["predictor_total_budget_ns"], -3000)

    def test_result_rejects_missing_trials_audits_order_and_environment(self):
        for mutate in (lambda r: r["trials"].pop(), lambda r: r["audits"].pop(),
                       lambda r: r["trials"].reverse(), lambda r: r["environment"].update(image_id="other"),
                       lambda r: r["audits"][0]["commit_audit"].update(aux_length_after=130),
                       lambda r: r["selector_measurements"][0].update(samples_ns=[1])):
            result, manifest, sources = self.result_fixture()
            mutate(result)
            with self.assertRaises(ValueError):
                analyze_result(result, manifest, sources)

    def test_legacy_measurements_remain_independently_recomputable(self):
        result, manifest, sources = self.result_fixture()
        expected = analyze_result(result, manifest, sources)
        result["protocol"] = manifest["protocol"] = "f0-direct-v2"
        result["manifest_sha256"] = canonical_json_sha256(manifest)
        self.assertEqual(analyze_result(result, manifest, sources), expected)
        result["protocol"] = PROTOCOL
        with self.assertRaises(ValueError):
            analyze_result(result, manifest, sources)

    def test_gate_uses_natural_prompt_directions(self):
        def rows(positive):
            return [{"question_id": q, "mode": m, "policy": p, "committed_tokens": 1,
                "reported_cycle_ns": 100 if p == "p7" else (90 if q < positive else 101),
                "h2d_submissions": 1, "logical_h2d_bytes": EXPERT_BYTES}
                for q in range(8) for m in ("cold", "natural") for p in ("p7", "f0")]
        for positive, expected in ((6, "enter_d3"), (2, "stop_current_configuration"),
                                   (4, "one_bounded_diagnostic")):
            self.assertEqual(summarize_rows(rows(positive), pilot_all_same=False,
                all_difference_count=22)["decision"], expected)
        self.assertEqual(summarize_rows(rows(8), pilot_all_same=True,
            all_difference_count=22)["decision"], "diagnose_remaining_states")

    def test_artifact_write_never_overwrites_existing_result(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            write_json_exclusive(path, {"first": True})
            with self.assertRaises(FileExistsError):
                write_json_exclusive(path, {"first": False})
            self.assertIn("true", path.read_text())
