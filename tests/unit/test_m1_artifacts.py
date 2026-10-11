from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from specfit.f0_direct import load_sources
from specfit.experiment import canonical_json_sha256
from specfit.m1_capture_replay import (
    advance_logical_state,
    logical_state_sha256,
    read_logical_state,
    read_pool,
    read_routes,
    resident_experts_by_layer,
)
from specfit.m1_capture_validation import (
    M1_CAPTURE_CONFIG, M1_CAPTURE_SCHEMA_VERSION, M1_CAPTURE_TYPE,
    M1_LIMITATIONS,
    M1_QUESTION_IDS,
    M1_SAMPLING_CONTRACT,
    M1_TOKEN_LOCK_TYPE,
    load_validated_m1_capture,
    output_token_diagnostic,
    validate_m1_capture_data,
)
from specfit.m1_screen_artifact import (
    build_m1_screen_artifact,
    validate_m1_screen_artifact,
)
from specfit.r6a_identity import R6A_DATASET_IDENTITY, R6A_MODEL_IDENTITIES
from specfit.r6a_tb2_contract import R6A_TB2_IMAGE_ID, R6A_TB2_NUMERIC_SETTINGS
from tests.unit.r6a_fixture import R6AFixture
class M1ArtifactTest(unittest.TestCase):
    commit = "a" * 40
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "workloads").mkdir()
        (self.root / "docker").mkdir()
        self.workload = self.root / "math_reasoning.jsonl"
        records = [
            {"question_id": qid, "category": "math_reasoning",
             "reference": ["answer"], "prompt": f"question {qid}"}
            for qid in range(401, 481)
        ]
        self.workload.write_text("".join(
            json.dumps(item) + "\n" for item in records), encoding="utf-8")
        self.token_lock = self.root / "workloads/m1_gsm8k_prompt_tokens.lock.json"
        self.prompt_ids = {qid: [qid, 1] for qid in M1_QUESTION_IDS}
        self.token_lock.write_text(json.dumps({
            "schema_version": 1,
            "artifact_type": M1_TOKEN_LOCK_TYPE,
            "tokenizer": R6A_MODEL_IDENTITIES["target"],
            "chat_template": {"messages": ["single_user_prompt"],
                              "tokenize": False,
                              "add_generation_prompt": True},
            "prompts": [{"question_id": qid, "input_ids": ids}
                        for qid, ids in self.prompt_ids.items()],
        }, sort_keys=True), encoding="utf-8")
        self.workload_lock = self.root / "workloads/gsm8k_specbench.lock.json"
        self.workload_lock.write_text(json.dumps({"schema_version": 1, "dataset": {
            **R6A_DATASET_IDENTITY, "size_bytes": self.workload.stat().st_size,
            "sha256": hashlib.sha256(self.workload.read_bytes()).hexdigest(),
            "expected_count": 80, "question_id_start": 401,
            "question_id_end": 480}}, sort_keys=True), encoding="utf-8")
        self.models_lock = self.root / "docker/models.lock.json"
        self.models_lock.write_text(json.dumps(R6A_MODEL_IDENTITIES, sort_keys=True),
                                    encoding="utf-8")
        self.authority = patch("specfit.m1_capture_sources._ROOT", self.root)
        self.authority.start()
        self.capture_path = self.root / "capture.json"
    def tearDown(self) -> None:
        self.authority.stop()
        self.temporary.cleanup()
    @staticmethod
    def _descriptor(path: Path) -> dict[str, object]:
        raw = path.read_bytes()
        return {"basename": path.name, "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest()}
    @staticmethod
    def _initial_state() -> dict[str, object]:
        return {"layers": [{
            "layer_id": layer, "capacity": 48, "lru_sequence": 96,
            "submission_sequence": 48,
            "counters": {"h2d_submissions": 48, "resident_hits": 0,
                         "inflight_joins": 0, "evictions": 0,
                         "pending_use_guards": 0},
            "slots": [{"slot_id": slot, "expert_id": slot,
                       "generation": 1, "active_leases": 0,
                       "last_used_sequence": slot + 1, "ready": True,
                       "pending_use_guards": 0}
                      for slot in range(48)],
        } for layer in range(48)]}
    @staticmethod
    def _pool() -> dict[str, object]:
        pool = deepcopy(R6AFixture.pool())
        for node in pool["candidates"]:
            node["conditional_log_probability_hex"] = float(
                node["conditional_log_probability"]).hex()
            node["cumulative_log_probability_hex"] = float(
                node["cumulative_log_probability"]).hex()
        return pool
    def _capture(self) -> dict[str, object]:
        pool_data = self._pool()
        pool = read_pool(pool_data, "fixture pool")
        selected = tuple(sorted(node.raw_id for node in sorted(
            pool.nodes, key=lambda node: (-node.cumulative_log_probability,
                                          node.depth, node.generation_ordinal)
        )[:7]))
        selected_routes_data = R6AFixture.routes(pool.fingerprint, (0, *selected))
        selected_routes = read_routes(
            selected_routes_data, "selected routes", pool, (0, *selected))
        atlas_data = R6AFixture.routes(pool.fingerprint, tuple(range(19)))
        references, states, target_only, diagnostics = [], [], [], []
        for qid in M1_QUESTION_IDS:
            context = list(self.prompt_ids[qid])
            output = []
            logical, snapshot = read_logical_state(
                deepcopy(self._initial_state()), "initial")
            cycles = []
            boundaries = []
            for cycle_index in range(4):
                pre = deepcopy(logical)
                cycle = {
                    "cycle_index": cycle_index,
                    "context_token_ids": list(context),
                    "context_token_count": len(context),
                    "base_kv_length": len(context),
                    "context_token_ids_sha256": canonical_json_sha256(context),
                    "pending_seed_token_id": 42,
                    "committed_offset": len(output),
                    "pool": deepcopy(pool_data),
                    "selected_raw_ids": list(selected),
                    "selected_routes": deepcopy(selected_routes_data),
                    "verifier_token_ids": [42] * (1 + len(selected)),
                    "transition": {"accepted_raw_ids": [],
                                   "committed_token_ids": [42],
                                   "next_pending_seed_token_id": 42},
                    "pre_logical_state_sha256": logical_state_sha256(pre),
                }
                boundaries.append((deepcopy(cycle), pre))
                logical, snapshot = advance_logical_state(
                    logical, snapshot, selected_routes)
                cycle["post_logical_state_sha256"] = logical_state_sha256(logical)
                cycles.append(cycle)
                context.append(42); output.append(42)
            eos = 151645
            cycles[-1]["verifier_token_ids"][0] = eos
            cycles[-1]["transition"]["next_pending_seed_token_id"] = eos
            output.append(eos)
            identity = {"question_id": qid, "output_token_count": len(output),
                "output_token_ids_sha256": canonical_json_sha256(output),
                "max_new_tokens": 512 - len(self.prompt_ids[qid]),
                "stopped_on_eos": True, "stop_token_id": output[-1]}
            target_only.append({**identity, "output_token_ids": list(output)})
            references.append({**identity, "tail_token_ids": [eos],
                "cycle_count": 4, "sampled_cycle_indices": [0, 1, 2, 3],
                "initial_logical_state": deepcopy(self._initial_state()),
                "cycles": cycles})
            diagnostics.append({"question_id": qid,
                **output_token_diagnostic(tuple(output), tuple(output))})
            for cycle_index, (cycle, pre) in enumerate(boundaries):
                resident = resident_experts_by_layer(pre)
                frozen = {"cache_state_fingerprint": logical_state_sha256(pre),
                    "quiescent": True,
                    "resident_experts_by_layer": [
                        {"layer_id": layer, "expert_ids": list(experts)}
                        for layer, experts in enumerate(resident)],
                    "logical_state": pre}
                states.append({"state_id": f"q{qid}-c{cycle_index}",
                    "question_id": qid, "cycle_index": cycle_index,
                    "context_token_count": cycle["context_token_count"],
                    "base_kv_length": cycle["context_token_count"],
                    "context_token_ids_sha256": cycle["context_token_ids_sha256"],
                    "pending_seed_token_id": 42,
                    "committed_offset": cycle["committed_offset"],
                    "pool": deepcopy(pool_data),
                    "pre_logical_state_sha256": logical_state_sha256(pre),
                    "frozen_state": frozen,
                    "verification_position_to_raw_id": list(range(19)),
                    "full_atlas_routes": deepcopy(atlas_data)})
        prompts = [{"question_id": qid, "input_token_ids": ids,
                    "input_token_ids_sha256": canonical_json_sha256(ids)}
                   for qid, ids in self.prompt_ids.items()]
        return {"schema_version": M1_CAPTURE_SCHEMA_VERSION,
            "artifact_type": M1_CAPTURE_TYPE,
            "status": "complete", "environment": {
                "git_commit": self.commit, "image_id": R6A_TB2_IMAGE_ID,
                "server": "a100-host", "device": "cuda:0",
                "numeric_settings": R6A_TB2_NUMERIC_SETTINGS},
            "config": M1_CAPTURE_CONFIG,
            "dataset": {**R6A_DATASET_IDENTITY,
                "workload_descriptor": self._descriptor(self.workload),
                "token_lock_descriptor": self._descriptor(self.token_lock),
                "prompts": prompts},
            "models": {**R6A_MODEL_IDENTITIES,
                "lock_descriptor": self._descriptor(self.models_lock)},
            "sampling_contract": M1_SAMPLING_CONTRACT,
            "target_only": target_only, "reference_rollouts": references,
            "output_diagnostics": diagnostics,
            "states": states, "limitations": M1_LIMITATIONS}
    def _write_capture(self, artifact: dict[str, object]) -> None:
        self.capture_path.write_text(json.dumps(
            artifact, sort_keys=True, allow_nan=False), encoding="utf-8")
    def _validate_data(self, artifact: dict[str, object], *, workload: Path | None = None,
                       token_lock: Path | None = None) -> object:
        return validate_m1_capture_data(
            artifact, workload_path=workload or self.workload,
            token_lock_path=token_lock or self.token_lock,
            expected_git_commit=self.commit)
    def test_valid_capture_projects_exactly_32_screen_inputs(self) -> None:
        artifact = self._capture(); self._write_capture(artifact)
        projection = load_validated_m1_capture(
            self.capture_path, workload_path=self.workload,
            token_lock_path=self.token_lock, expected_git_commit=self.commit)
        self.assertEqual(len(projection.states), 32)
        self.assertFalse(hasattr(projection.states[0], "accepted_raw_ids"))
    def test_capture_rejects_source_unknown_outcome_and_chain_tampering(self) -> None:
        artifact = self._capture()
        cases = []
        value = deepcopy(artifact); value["dataset"]["workload_descriptor"]["sha256"] = "0" * 64; cases.append(value)
        value = deepcopy(artifact); value["unknown"] = 1; cases.append(value)
        value = deepcopy(artifact); value["states"][0]["accepted_raw_ids"] = []; cases.append(value)
        value = deepcopy(artifact); value["reference_rollouts"][0]["cycles"][0]["post_logical_state_sha256"] = "0" * 64; cases.append(value)
        value = deepcopy(artifact); value["config"]["batch_size"] = True; cases.append(value)
        value = deepcopy(artifact); value["target_only"][0]["output_token_ids"][0] += 1; cases.append(value)
        value = deepcopy(artifact); value["target_only"][0]["output_token_ids"][0] = 151643; value["target_only"][0]["output_token_ids_sha256"] = canonical_json_sha256(value["target_only"][0]["output_token_ids"]); cases.append(value)
        value = deepcopy(artifact); value["reference_rollouts"][0]["cycles"][0]["base_kv_length"] += 1; cases.append(value)
        value = deepcopy(artifact); value["states"][0]["base_kv_length"] += 1; cases.append(value)
        value = deepcopy(artifact); slots = value["reference_rollouts"][0]["initial_logical_state"]["layers"][0]["slots"]; slots[0], slots[1] = slots[1], slots[0]; cases.append(value)
        value = deepcopy(artifact); value["reference_rollouts"][0]["cycles"][0]["pool"]["candidates"][0]["conditional_log_probability"] = 0.1; value["reference_rollouts"][0]["cycles"][0]["pool"]["candidates"][0]["conditional_log_probability_hex"] = float(0.1).hex(); cases.append(value)
        value = deepcopy(artifact); target, ref = value["target_only"][0], value["reference_rollouts"][0]; target.update(stopped_on_eos=False, stop_token_id=None); ref.update(stopped_on_eos=False, stop_token_id=None); cases.append(value)
        for poisoned in cases:
            with self.assertRaises(ValueError):
                self._validate_data(poisoned)
        value = deepcopy(artifact); cycle = value["reference_rollouts"][0]["cycles"][-1]
        cycle["verifier_token_ids"][0] = 1001; cycle["transition"]["next_pending_seed_token_id"] = 1001
        with self.assertRaises(ValueError):
            self._validate_data(value)
        value = deepcopy(artifact); target, ref = value["target_only"][0], value["reference_rollouts"][0]
        ref["cycles"][-1]["verifier_token_ids"][0] = 42; ref["cycles"][-1]["transition"]["next_pending_seed_token_id"] = 42; ref["tail_token_ids"] = [42]
        digest = canonical_json_sha256([42] * 5); ref.update(output_token_ids_sha256=digest, stop_token_id=42); target.update(output_token_ids_sha256=digest, stop_token_id=42)
        with self.assertRaisesRegex(ValueError, "EOS"):
            self._validate_data(value)
        alternate = self.root / "alternate"; alternate.mkdir()
        other_workload = alternate / self.workload.name
        other_workload.write_bytes(self.workload.read_bytes() + b"\n")
        value = deepcopy(artifact)
        value["dataset"]["workload_descriptor"] = self._descriptor(other_workload)
        with self.assertRaises(ValueError):
            self._validate_data(value, workload=other_workload)
        other_lock = alternate / self.token_lock.name
        lock = json.loads(self.token_lock.read_text()); lock["prompts"][0]["input_ids"][0] += 1
        other_lock.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")
        value = deepcopy(artifact); value["dataset"]["token_lock_descriptor"] = self._descriptor(other_lock)
        with self.assertRaises(ValueError):
            self._validate_data(value, token_lock=other_lock)
    def test_capture_accepts_verified_target_reference_divergence(self) -> None:
        artifact = self._capture()
        target = artifact["target_only"][0]
        target["output_token_ids"][0] = 99
        target["output_token_ids_sha256"] = canonical_json_sha256(
            target["output_token_ids"])
        reference = (42, 42, 42, 42, 151645)
        artifact["output_diagnostics"][0] = {"question_id": 406,
            **output_token_diagnostic(tuple(target["output_token_ids"]), reference)}
        self.assertEqual(len(self._validate_data(artifact).states), 32)
        artifact["output_diagnostics"][0]["exact"] = True
        with self.assertRaises(ValueError):
            self._validate_data(artifact)
    def test_screen_validator_rebuilds_and_rejects_coherent_tampering(self) -> None:
        self._write_capture(self._capture())
        screen = build_m1_screen_artifact(
            capture_path=self.capture_path, workload_path=self.workload,
            token_lock_path=self.token_lock,
            expected_capture_git_commit=self.commit, git_commit="b" * 40)
        validate_m1_screen_artifact(
            screen, capture_path=self.capture_path, workload_path=self.workload,
            token_lock_path=self.token_lock,
            expected_capture_git_commit=self.commit,
            expected_git_commit="b" * 40)
        poisoned = deepcopy(screen)
        poisoned["states"][0]["candidate_sets"][0]["N_atlas"] += 1
        poisoned["states"][0]["delta_N_atlas"] += 1
        poisoned["summary"]["delta_N_atlas"] += 1
        with self.assertRaises(ValueError):
            validate_m1_screen_artifact(
                poisoned, capture_path=self.capture_path,
                workload_path=self.workload, token_lock_path=self.token_lock,
                expected_capture_git_commit=self.commit,
                expected_git_commit="b" * 40)

    def _direct_sources_fixture(self):
        self._write_capture(self._capture())
        screen = build_m1_screen_artifact(
            capture_path=self.capture_path, workload_path=self.workload,
            token_lock_path=self.token_lock,
            expected_capture_git_commit=self.commit, git_commit="b" * 40)
        screen_path = self.root / "screen.json"
        screen_path.write_text(json.dumps(screen, sort_keys=True), encoding="utf-8")
        return screen, screen_path

    def _load_direct_sources(self, screen_path, *, capture_descriptor=None):
        with patch("specfit.f0_direct.CAPTURE_SOURCE",
                   capture_descriptor or self._descriptor(self.capture_path)), patch(
                   "specfit.f0_direct.F0_M1_SCREEN_SOURCE", self._descriptor(screen_path)):
            return load_sources(capture_path=self.capture_path, screen_path=screen_path,
                workload_path=self.workload, token_lock_path=self.token_lock)

    def test_direct_sources_read_and_validate_capture_once(self) -> None:
        screen, screen_path = self._direct_sources_fixture()
        descriptor = self._descriptor(self.capture_path)
        read_bytes = Path.read_bytes
        with patch("specfit.m1_capture_validation.validate_m1_capture_data",
                   wraps=validate_m1_capture_data) as validate, patch(
                   "specfit.f0_direct.validate_m1_capture_data", validate), patch.object(
                   Path, "read_bytes", autospec=True, side_effect=read_bytes) as reads:
            sources = self._load_direct_sources(screen_path, capture_descriptor=descriptor)
        self.assertEqual(validate.call_count, 1)
        self.assertEqual(sum(call.args[0] == self.capture_path for call in reads.call_args_list), 1)
        self.assertEqual(sources.screen, screen)
        self.assertEqual(len(sources.projection.states), 32)
        self.assertEqual(sources.projection.source, (
            descriptor["basename"], descriptor["size_bytes"], descriptor["sha256"]))

    def test_direct_sources_still_reject_bound_source_and_content_tampering(self) -> None:
        screen, screen_path = self._direct_sources_fixture()
        capture_raw = self.capture_path.read_bytes()
        descriptor = self._descriptor(self.capture_path)
        self.capture_path.write_bytes(capture_raw + b"\n")
        with self.assertRaises(ValueError):
            self._load_direct_sources(screen_path, capture_descriptor=descriptor)

        # Even when the outer descriptor is bound to these bytes, reference replay
        # and the independent screen reconstruction must still reject corruption.
        capture = json.loads(capture_raw)
        capture["reference_rollouts"][0]["cycles"][0]["post_logical_state_sha256"] = "0" * 64
        self._write_capture(capture)
        with self.assertRaises(ValueError):
            self._load_direct_sources(screen_path)

        self.capture_path.write_bytes(capture_raw)
        screen["states"][0]["candidate_sets"][0]["N_atlas"] += 1
        screen["states"][0]["delta_N_atlas"] += 1
        screen["summary"]["delta_N_atlas"] += 1
        screen_path.write_text(json.dumps(screen, sort_keys=True), encoding="utf-8")
        with self.assertRaises(ValueError):
            self._load_direct_sources(screen_path)
if __name__ == "__main__":
    unittest.main()
