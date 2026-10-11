
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from .experiment import canonical_json_sha256
from .gsm8k import load_gsm8k_specbench
from .m1_capture_replay import same_json_value
from .r6a_identity import R6A_DATASET_IDENTITY, R6A_MODEL_IDENTITIES

_ROOT = Path(__file__).resolve().parents[2]
_WORKLOAD_LOCK = "workloads/gsm8k_specbench.lock.json"
_TOKEN_LOCK = "workloads/m1_gsm8k_prompt_tokens.lock.json"
_MODEL_LOCK = "docker/models.lock.json"
_TOKEN_LOCK_TYPE = "m1_gsm8k_prompt_tokens"
_QUESTION_IDS = tuple(range(406, 477, 10))


def _object(name: str, value: object, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{name} field set inconsistent")
    return value


def _sequence(name: str, value: object) -> Sequence[object]:
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


def _strict_json(raw: bytes, name: str) -> Mapping[str, object]:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"{name} contains duplicate field {key}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"{name} contains {item}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not strict JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} root node must be an object")
    return value


def read_strict_json_file(path: Path, name: str) -> tuple[Mapping[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")
    raw = path.read_bytes()
    return _strict_json(raw, name), raw


def file_descriptor(path: Path, raw: bytes) -> dict[str, object]:
    return {"basename": path.name, "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def _prompt(value: object, name: str) -> dict[str, object]:
    item = _object(name, value, {"question_id", "input_ids"})
    ids = tuple(_integer(f"{name}.input_ids", token)
                for token in _sequence(name, item["input_ids"]))
    if not ids:
        raise ValueError(f"{name} input_ids must not be empty")
    return {"question_id": _integer(f"{name}.question_id", item["question_id"], 1),
            "input_token_ids": list(ids),
            "input_token_ids_sha256": canonical_json_sha256(ids)}


def _workload_authority(workload_path: Path) -> tuple[dict[str, object], object]:
    lock_path = _ROOT / _WORKLOAD_LOCK
    lock, _ = read_strict_json_file(lock_path, "workload lock")
    root = _object("workload lock", lock, {"schema_version", "dataset"})
    dataset = _object("workload lock dataset", root["dataset"], {
        "repo_id", "revision", "filename", "size_bytes", "sha256", "category",
        "expected_count", "question_id_start", "question_id_end"})
    fixed = {**R6A_DATASET_IDENTITY, "expected_count": 80,
             "question_id_start": 401, "question_id_end": 480}
    if not same_json_value(root["schema_version"], 1) or any(
        not same_json_value(dataset[key], value) for key, value in fixed.items()
    ):
        raise ValueError("workload lock identity is inconsistent")
    expected = {"basename": dataset["filename"],
                "size_bytes": _integer("workload size", dataset["size_bytes"], 1),
                "sha256": _sha("workload SHA", dataset["sha256"])}
    if workload_path.is_symlink() or not workload_path.is_file():
        raise ValueError("workload must be a regular non-symlink file")
    raw = workload_path.read_bytes()
    if not same_json_value(file_descriptor(workload_path, raw), expected):
        raise ValueError("workload is inconsistent with the repository authoritative lock")
    return expected, load_gsm8k_specbench(workload_path, lock_path=lock_path)


def _token_authority(token_lock_path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    authority_path = _ROOT / _TOKEN_LOCK
    authority, authority_raw = read_strict_json_file(authority_path, "repository token lock")
    supplied, supplied_raw = read_strict_json_file(token_lock_path, "token lock")
    descriptor = file_descriptor(authority_path, authority_raw)
    if (not same_json_value(file_descriptor(token_lock_path, supplied_raw), descriptor)
            or not same_json_value(supplied, authority)):
        raise ValueError("token lock does not match the repo's authoritative file")
    root = _object("token lock", authority, {"schema_version", "artifact_type",
        "tokenizer", "chat_template", "prompts"})
    expected_template = {"messages": ["single_user_prompt"], "tokenize": False,
                         "add_generation_prompt": True}
    if not same_json_value({key: root[key] for key in (
        "schema_version", "artifact_type", "tokenizer", "chat_template")}, {
        "schema_version": 1, "artifact_type": _TOKEN_LOCK_TYPE,
        "tokenizer": R6A_MODEL_IDENTITIES["target"],
        "chat_template": expected_template}):
        raise ValueError("token lock identity is inconsistent")
    prompts = [_prompt(item, f"token_lock.prompts[{index}]") for index, item in
               enumerate(_sequence("token lock prompts", root["prompts"]))]
    if tuple(item["question_id"] for item in prompts) != _QUESTION_IDS:
        raise ValueError("token lock prompt distribution mismatch")
    return descriptor, prompts


def _model_authority() -> dict[str, object]:
    path = _ROOT / _MODEL_LOCK
    lock, raw = read_strict_json_file(path, "models lock")
    root = _object("models lock", lock, {"target", "drafter"})
    for name, identity in R6A_MODEL_IDENTITIES.items():
        model = root[name]
        if not isinstance(model, Mapping) or any(
            not same_json_value(model.get(key), value) for key, value in identity.items()
        ):
            raise ValueError(f"models lock {name} identity mismatch")
    return {"target": R6A_MODEL_IDENTITIES["target"],
            "drafter": R6A_MODEL_IDENTITIES["drafter"],
            "lock_descriptor": file_descriptor(path, raw)}


def validate_capture_sources(*, dataset: object, models: object,
    workload_path: Path, token_lock_path: Path) -> dict[int, dict[str, object]]:

    workload_descriptor, examples = _workload_authority(workload_path)
    token_descriptor, prompts = _token_authority(token_lock_path)
    expected_dataset = {**R6A_DATASET_IDENTITY,
        "workload_descriptor": workload_descriptor,
        "token_lock_descriptor": token_descriptor, "prompts": prompts}
    if not same_json_value(dataset, expected_dataset):
        raise ValueError("capture dataset inconsistent with repository authoritative source")
    if not same_json_value(models, _model_authority()):
        raise ValueError("capture models does not match the repo's authoritative models lock")
    if not set(_QUESTION_IDS) <= {item.question_id for item in examples}:
        raise ValueError("workload is missing the preregistered prompt")
    return {int(item["question_id"]): item for item in prompts}


__all__ = ["file_descriptor", "read_strict_json_file", "validate_capture_sources"]
