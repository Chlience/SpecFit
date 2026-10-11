from __future__ import annotations
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from .experiment import canonical_json_sha256
from .m1_capture_pool import ReferencePool, read_reference_pool
from .m1_capture_replay import (advance_logical_state, logical_state_sha256, read_logical_state,
    read_pool, read_routes, resident_experts_by_layer, same_json_value)
from .m1_capture_sources import (file_descriptor, read_strict_json_file,
    validate_capture_sources)
from .r6a_experiment import FrozenStateRecord, PoolRecord
from .r6a_route_observation import RouteObservation
from .r6a_tb2_contract import R6A_TB2_IMAGE_ID, R6A_TB2_NUMERIC_SETTINGS
M1_CAPTURE_TYPE = "m1_full_atlas_capture"
M1_CAPTURE_SCHEMA_VERSION = 2
M1_TOKEN_LOCK_TYPE = "m1_gsm8k_prompt_tokens"
M1_EOS_TOKEN_IDS = frozenset((151645, 151643))
M1_QUESTION_IDS = tuple(range(406, 477, 10))
M1_CAPTURE_CONFIG = {
    "batch_size": 1, "max_total_tokens": 512, "tree_top_k": 2, "tree_max_depth": 5,
    "pool_draft_nodes": 18, "selected_draft_nodes": 7, "num_moe_layers": 48,
    "num_routed_experts": 128, "experts_per_token": 8, "slots_per_layer": 48,
    "dtype": "bfloat16", "numeric_mode": "math_bf16_full_reduction",
    "prefill_execution_order": "expert_id_ascending", "decode_execution_order": "resident_first",
    "cache_policy": "lru", "commit_audit": True}
M1_SAMPLING_CONTRACT = {"states_per_question": 4, "total_states": 32,
    "cycle_index_formula": "floor((2*j+1)*C/8), j=0..3", "requires_cycle_count_at_least": 4}
M1_LIMITATIONS = {"route_semantics": "full_atlas_proxy", "batch_size": 1,
    "actual_h2d": False, "timing": False, "accepted_path_in_screen": False,
    "m2_implemented": False, "token_identity": "diagnostic_only"}
@dataclass(frozen=True, slots=True)
class M1StateProjection:
    state_id: str; question_id: int; cycle_index: int
    pool: PoolRecord; frozen: FrozenStateRecord; atlas: RouteObservation
@dataclass(frozen=True, slots=True)
class M1CaptureProjection:
    git_commit: str; source: tuple[str, int, str] | None
    states: tuple[M1StateProjection, ...]
def _obj(name: str, value: object, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{name} field set inconsistent")
    return value
def _items(name: str, value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an array")
    return value
def _integer(name: str, value: object, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be no less than {minimum} as an integer")
    return value
def _sha(name: str, value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value
def _boolean(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a bool")
    return value
def output_token_diagnostic(target: tuple[int, ...], reference: tuple[int, ...]
                            ) -> dict[str, object]:
    common = next((index for index, pair in enumerate(zip(target, reference))
                   if pair[0] != pair[1]), min(len(target), len(reference)))
    different = sum(left != right for left, right in zip(target, reference))
    different += abs(len(target) - len(reference))
    exact = target == reference
    return {"exact": exact, "common_prefix_token_count": common,
            "first_difference_offset": None if exact else common,
            "different_token_count": different,
            "target_token_id": None if exact or common >= len(target) else target[common],
            "reference_token_id": (None if exact or common >= len(reference)
                                   else reference[common])}
def _outcome(value: Mapping[str, object], prompt_length: int, name: str
             ) -> tuple[int, str, int, bool, int | None]:
    count = _integer(f"{name}.output_token_count", value["output_token_count"], 1)
    digest = _sha(f"{name}.output_token_ids_sha256", value["output_token_ids_sha256"])
    budget = _integer(f"{name}.max_new_tokens", value["max_new_tokens"], 1)
    stopped = _boolean(f"{name}.stopped_on_eos", value["stopped_on_eos"])
    raw_stop = value["stop_token_id"]
    stop = None if raw_stop is None else _integer(f"{name}.stop_token_id", raw_stop)
    if stopped and stop not in M1_EOS_TOKEN_IDS:
        raise ValueError(f"{name}.stop_token_id is not in the fixed EOS set")
    if (budget != 512 - prompt_length or count > budget
            or (not stopped and (count != budget or stop is not None))):
        raise ValueError(f"{name} termination semantics inconsistent with fixed token budget")
    return count, digest, budget, stopped, stop
def _validate_output_tokens(tokens: tuple[int, ...], outcome: tuple[int, str, int, bool, int | None], name: str) -> None:
    eos = tuple(index for index, token in enumerate(tokens) if token in M1_EOS_TOKEN_IDS)
    if ((len(tokens), canonical_json_sha256(tokens)) != outcome[:2]
            or eos != ((len(tokens) - 1,) if outcome[3] else ()) or outcome[3] and tokens[-1] != outcome[4]):
        raise ValueError(f"{name} output token identity/EOS inconsistent")
def _selected(value: object, pool: PoolRecord | ReferencePool,
              name: str) -> tuple[int, ...]:
    selected = tuple(_integer(name, item, 1) for item in _items(name, value))
    wanted = tuple(sorted(node.raw_id for node in sorted(
        pool.nodes, key=lambda node: (-node.cumulative_log_probability,
                                      node.depth, node.generation_ordinal)
    )[:min(7, len(pool.nodes))]))
    if selected != wanted:
        raise ValueError(f"{name} is not probability fixed-7")
    return selected
def validate_m1_capture_data(artifact: Mapping[str, object], *, workload_path: Path,
    token_lock_path: Path, expected_git_commit: str,
    source: tuple[str, int, str] | None = None) -> M1CaptureProjection:
    root = _obj("capture", artifact, {"schema_version", "artifact_type", "status",
        "environment", "config", "dataset", "models", "sampling_contract",
        "target_only", "reference_rollouts", "output_diagnostics", "states",
        "limitations"})
    env = _obj("environment", root["environment"],
               {"git_commit", "image_id", "server", "device", "numeric_settings"})
    if re.fullmatch(r"[0-9a-f]{40}", expected_git_commit or "") is None or not same_json_value(env, {
        "git_commit": expected_git_commit, "image_id": R6A_TB2_IMAGE_ID,
        "server": "a100-host", "device": "cuda:0",
        "numeric_settings": R6A_TB2_NUMERIC_SETTINGS}):
        raise ValueError("capture environment does not match the external Git/fixed environment")
    if (not same_json_value(root["schema_version"], M1_CAPTURE_SCHEMA_VERSION)
        or root["artifact_type"] != M1_CAPTURE_TYPE or root["status"] != "complete"
        or not same_json_value(root["config"], M1_CAPTURE_CONFIG)
        or not same_json_value(root["sampling_contract"], M1_SAMPLING_CONTRACT)
        or not same_json_value(root["limitations"], M1_LIMITATIONS)):
        raise ValueError("capture root identity or fixed contract mismatch")
    prompts = validate_capture_sources(dataset=root["dataset"], models=root["models"],
        workload_path=workload_path, token_lock_path=token_lock_path)
    target_rows = [_obj("target_only", item,
        {"question_id", "output_token_count", "output_token_ids_sha256",
         "output_token_ids", "max_new_tokens", "stopped_on_eos", "stop_token_id"})
        for item in _items("target_only", root["target_only"])]
    if tuple(_integer("target question", item["question_id"], 1) for item in target_rows) != M1_QUESTION_IDS:
        raise ValueError("target-only prompt distribution inconsistent")
    targets = {}
    for item in target_rows:
        qid = int(item["question_id"])
        outcome = _outcome(item, len(prompts[qid]["input_token_ids"]), "target-only")
        tokens = tuple(_integer("target token", token) for token in
                       _items("target output", item["output_token_ids"]))
        _validate_output_tokens(tokens, outcome, "target-only")
        targets[qid] = (outcome, tokens)
    boundaries: dict[tuple[int, int], tuple[Mapping[str, object], ReferencePool,
                                            dict[str, object]]] = {}
    references = _items("reference_rollouts", root["reference_rollouts"])
    diagnostics = []
    if len(references) != 8:
        raise ValueError("reference rollout count inconsistent")
    for ref_index, raw_ref in enumerate(references):
        ref = _obj(f"reference[{ref_index}]", raw_ref, {"question_id", "output_token_count",
            "output_token_ids_sha256", "max_new_tokens", "stopped_on_eos", "stop_token_id",
            "tail_token_ids", "cycle_count", "sampled_cycle_indices",
            "initial_logical_state", "cycles"})
        qid = _integer("reference question", ref["question_id"], 1)
        if qid != M1_QUESTION_IDS[ref_index]:
            raise ValueError("reference prompt order is inconsistent")
        outcome = _outcome(ref, len(prompts[qid]["input_token_ids"]), f"reference q{qid}")
        count = _integer("cycle_count", ref["cycle_count"], 4)
        sampled = tuple((2 * j + 1) * count // 8 for j in range(4))
        if tuple(_integer("sampled cycle", x) for x in _items(
                "sampled cycles", ref["sampled_cycle_indices"])) != sampled:
            raise ValueError("sampled cycle does not match the pre-registered formula")
        state, snapshot = read_logical_state(ref["initial_logical_state"], f"q{qid}.initial")
        context = list(prompts[qid]["input_token_ids"])
        output: list[int] = []
        expected_pending: int | None = None
        cycles = _items("reference cycles", ref["cycles"])
        if len(cycles) != count:
            raise ValueError("reference cycle_count mismatch")
        for cycle_index, raw_cycle in enumerate(cycles):
            cycle = _obj(f"q{qid}.cycles[{cycle_index}]", raw_cycle, {"cycle_index",
                "context_token_ids", "context_token_count", "context_token_ids_sha256", "base_kv_length",
                "pending_seed_token_id", "committed_offset", "pool", "selected_raw_ids",
                "selected_routes", "verifier_token_ids", "transition", "pre_logical_state_sha256",
                "post_logical_state_sha256"})
            context_ids = [_integer("context_token_ids", token) for token in
                           _items("context_token_ids", cycle["context_token_ids"])]
            pre_sha = logical_state_sha256(state)
            if (_integer("cycle_index", cycle["cycle_index"]), context_ids,
                _integer("context_token_count", cycle["context_token_count"]),
                _integer("base_kv_length", cycle["base_kv_length"]),
                _sha("context_token_ids_sha256", cycle["context_token_ids_sha256"]),
                _integer("committed_offset", cycle["committed_offset"]),
                _sha("pre_logical_state_sha256", cycle["pre_logical_state_sha256"])) != (
                cycle_index, context, len(context), len(context), canonical_json_sha256(context),
                len(output), pre_sha):
                raise ValueError("reference context/offset/pre-state chain inconsistent")
            pool = read_reference_pool(cycle["pool"], f"q{qid}.cycle{cycle_index}.pool")
            expected_depth = min(5, outcome[2] - len(output) - 1)
            if expected_depth < 1 or pool.max_depth != expected_depth:
                raise ValueError("reference pool effective depth does not match remaining budget")
            pending = _integer("pending_seed_token_id", cycle["pending_seed_token_id"])
            if expected_pending is not None and pending != expected_pending:
                raise ValueError("reference next-pending seed chain is inconsistent")
            if cycle_index in sampled and (pool.max_depth != 5 or len(pool.nodes) != 18):
                raise ValueError("sampled reference boundary missing the complete 18-node pool")
            selected = _selected(cycle["selected_raw_ids"], pool, "selected_raw_ids")
            routes = read_routes(cycle["selected_routes"], "selected_routes", pool, (0, *selected))
            verifier = tuple(_integer("verifier token", token) for token in
                             _items("verifier_token_ids", cycle["verifier_token_ids"]))
            mapping = (0, *selected)
            if len(verifier) != len(mapping):
                raise ValueError("reference verifier token coverage is incomplete")
            transition = _obj("transition", cycle["transition"],
                {"accepted_raw_ids", "committed_token_ids", "next_pending_seed_token_id"})
            accepted = tuple(_integer("accepted", x, 1) for x in _items("accepted", transition["accepted_raw_ids"]))
            by_id = {node.raw_id: node for node in pool.nodes}
            position = {raw_id: index for index, raw_id in enumerate(mapping)}
            parent = 0; maximal = []
            while True:
                match = next((raw_id for raw_id in selected
                    if by_id[raw_id].parent_raw_id == parent
                    and by_id[raw_id].target_token_id == verifier[position[parent]]), None)
                if match is None:
                    break
                maximal.append(match); parent = match
                if by_id[match].target_token_id in M1_EOS_TOKEN_IDS:
                    break
            if accepted != tuple(maximal):
                raise ValueError("reference accepted raw IDs are not the verifier maximum selected path")
            committed = tuple(_integer("committed", x) for x in _items("committed", transition["committed_token_ids"]))
            if pool.seed_token_id != pending or committed != (
                pool.seed_token_id, *(by_id[raw_id].target_token_id for raw_id in accepted)):
                raise ValueError("reference seed/commit transition inconsistent")
            boundaries[qid, cycle_index] = (cycle, pool, state)
            output.extend(committed); context.extend(committed)
            expected_pending = _integer("next pending", transition["next_pending_seed_token_id"])
            if expected_pending != verifier[position[parent]]:
                raise ValueError("reference next pending does not match verifier output")
            state, snapshot = advance_logical_state(state, snapshot, routes)
            if logical_state_sha256(state) != _sha("post logical state", cycle["post_logical_state_sha256"]):
                raise ValueError("reference post-state fingerprint chain mismatch")
        tail = tuple(_integer("tail token", x) for x in _items("tail_token_ids", ref["tail_token_ids"]))
        if len(tail) > 1 or (tail and tail[0] != expected_pending):
            raise ValueError("reference residual tail does not start from the final pending seed")
        output.extend(tail)
        _validate_output_tokens(tuple(output), outcome, f"reference q{qid}")
        diagnostics.append({"question_id": qid, **output_token_diagnostic(
            targets[qid][1], tuple(output))})
    if not same_json_value(root["output_diagnostics"], diagnostics):
        raise ValueError("target-only/reference token difference diagnostics inconsistent")
    projected = []
    states = _items("states", root["states"])
    expected_pairs = tuple((qid, (2*j+1)*len(_items("cycles", references[i]["cycles"]))//8)
                           for i, qid in enumerate(M1_QUESTION_IDS) for j in range(4))
    if len(states) != 32:
        raise ValueError("probe state count inconsistent")
    for index, (raw_state, pair) in enumerate(zip(states, expected_pairs, strict=True)):
        probe = _obj(f"states[{index}]", raw_state, {"state_id", "question_id", "cycle_index",
            "context_token_count", "context_token_ids_sha256", "base_kv_length", "pending_seed_token_id",
            "committed_offset", "pool", "pre_logical_state_sha256", "frozen_state",
            "verification_position_to_raw_id", "full_atlas_routes"})
        qid, cycle_index = pair; reference, ref_pool, ref_state = boundaries[pair]
        if (probe["state_id"] != f"q{qid}-c{cycle_index}"
            or _integer("probe question_id", probe["question_id"], 1) != qid
            or _integer("probe cycle_index", probe["cycle_index"]) != cycle_index):
            raise ValueError("probe state identity/order is inconsistent")
        pool = read_pool(probe["pool"], f"states[{index}].pool")
        frozen_raw = _obj("frozen_state", probe["frozen_state"],
            {"cache_state_fingerprint", "quiescent", "resident_experts_by_layer", "logical_state"})
        logical, _ = read_logical_state(frozen_raw["logical_state"], "probe logical state")
        state_sha = logical_state_sha256(logical)
        if (pool.fingerprint != ref_pool.fingerprint or logical != ref_state
                or state_sha != _sha("probe pre-state SHA", probe["pre_logical_state_sha256"])):
            raise ValueError("probe pool/BEFORE state does not match the reference boundary")
        frozen = FrozenStateRecord.read({key: frozen_raw[key] for key in
            ("cache_state_fingerprint", "quiescent", "resident_experts_by_layer")},
            slots=48, num_experts=128)
        if frozen.fingerprint != state_sha or frozen.resident_by_layer != resident_experts_by_layer(logical):
            raise ValueError("probe frozen residency is inconsistent with the full logical state")
        if (_integer("probe context count", probe["context_token_count"]),
            _integer("probe base KV length", probe["base_kv_length"]),
            _sha("probe context SHA", probe["context_token_ids_sha256"]),
            _integer("probe pending seed", probe["pending_seed_token_id"]),
            _integer("probe committed offset", probe["committed_offset"])) != (
            _integer("reference context count", reference["context_token_count"]),
            _integer("reference context count", reference["context_token_count"]),
            _sha("reference context SHA", reference["context_token_ids_sha256"]),
            _integer("reference pending seed", reference["pending_seed_token_id"]),
            _integer("reference committed offset", reference["committed_offset"])):
            raise ValueError("probe context/seed/offset inconsistent with reference boundary")
        positions = tuple(_integer("atlas mapping", x) for x in
                          _items("atlas mapping", probe["verification_position_to_raw_id"]))
        if positions != tuple(range(19)):
            raise ValueError("full atlas mapping must be raw ID 0..18")
        atlas = read_routes(probe["full_atlas_routes"], "full_atlas_routes", pool, positions)
        projected.append(M1StateProjection(str(probe["state_id"]), qid, cycle_index,
                                           pool, frozen, atlas))
    return M1CaptureProjection(expected_git_commit, source, tuple(projected))
def load_validated_m1_capture(capture_path: Path, *, workload_path: Path,
    token_lock_path: Path, expected_git_commit: str) -> M1CaptureProjection:
    artifact, raw = read_strict_json_file(capture_path, "capture")
    descriptor = file_descriptor(capture_path, raw)
    source = (str(descriptor["basename"]), int(descriptor["size_bytes"]),
              str(descriptor["sha256"]))
    return validate_m1_capture_data(artifact, workload_path=workload_path,
        token_lock_path=token_lock_path, expected_git_commit=expected_git_commit,
        source=source)
__all__ = ["M1_CAPTURE_CONFIG", "M1_CAPTURE_SCHEMA_VERSION", "M1_CAPTURE_TYPE", "M1_EOS_TOKEN_IDS", "M1_LIMITATIONS", "M1_QUESTION_IDS",
    "M1_SAMPLING_CONTRACT", "M1_TOKEN_LOCK_TYPE", "M1CaptureProjection",
    "M1StateProjection", "load_validated_m1_capture", "output_token_diagnostic",
    "validate_m1_capture_data"]
