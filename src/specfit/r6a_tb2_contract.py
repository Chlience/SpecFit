
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .r6a_identity import R6A_DATASET_IDENTITY, R6A_MODEL_IDENTITIES
from .r6a_tb2 import (
    R6A_TB2_ARTIFACT_TYPE,
    R6A_TB2_POLICIES,
    R6A_TB2_SOURCE_SHA256,
    replay_contract_record,
)

R6A_TB2_IMAGE_ID = (
    "sha256:64f6847e922677a269c82e23f140cca5ceafd5899e17002d52bc05035eec6567"
)
R6A_TB2_FIXED_CONFIG = {
    "batch_size": 1,
    "tree_top_k": 2,
    "tree_max_depth": 5,
    "draft_node_budget": 7,
    "verification_positions": 8,
    "num_moe_layers": 48,
    "num_routed_experts": 128,
    "experts_per_token": 8,
    "slots_per_layer": 48,
    "cycles_per_trial": 1,
    "dtype": "bfloat16",
    "numeric_mode": "math_bf16_full_reduction",
}
R6A_TB2_NUMERIC_SETTINGS = {
    "allow_bf16_reduced_precision_reduction": False,
    "cuda_matmul_fp32_precision": "tf32",
    "cudnn_sdp_enabled": False,
    "deterministic_algorithms": False,
    "deterministic_warn_only": False,
    "flash_sdp_enabled": False,
    "forced_sdpa_backend": "math",
    "math_sdp_enabled": True,
    "memory_efficient_sdp_enabled": False,
}
R6A_TB2_DATASET = {**R6A_DATASET_IDENTITY, "question_id": 480}
R6A_TB2_MODELS = {
    name: dict(identity) for name, identity in R6A_MODEL_IDENTITIES.items()
}
R6A_TB2_GOLDEN = {
    "question_id": 480,
    "output_token_ids": [1654, 525, 2661, 429, 1447, 12, 220],
}
R6A_TB2_SOURCES = {
    "e019": {
        "path": "r6a-tb1-q480-48slots-7f3b979.json",
        "sha256": R6A_TB2_SOURCE_SHA256["e019"],
        "git_commit": "7f3b979738a82504c39757bdae990f3dad8efa74",
    },
    "e020": {
        "path": "r6a-tb1b-q480-48slots-5b94eda.json",
        "sha256": R6A_TB2_SOURCE_SHA256["e020"],
        "git_commit": "5b94edaf9f941fc3e4ce033eea4701b83ee8d362",
    },
}


def require_mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def require_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def validate_r6a_tb2_root(
    artifact: Mapping[str, object],
) -> tuple[Mapping[str, Any], Sequence[Any]]:

    root = require_mapping("artifact", artifact)
    if set(root) != {
        "schema_version", "artifact_type", "status", "environment", "config",
        "source_artifacts", "dataset", "models", "golden",
        "replay_contract", "trials",
    }:
        raise ValueError("artifact root field set inconsistent")
    if (
        type(root.get("schema_version")) is not int
        or root.get("schema_version") != 1
        or root.get("artifact_type") != R6A_TB2_ARTIFACT_TYPE
        or root.get("status") != "complete"
    ):
        raise ValueError("artifact must be a complete R6a-TB2 schema v1")
    if root.get("replay_contract") != replay_contract_record():
        raise ValueError("replay contract is inconsistent")
    if dict(require_mapping("source_artifacts", root.get("source_artifacts"))) != (
        R6A_TB2_SOURCES
    ):
        raise ValueError("E019/E020 source artifact identity mismatch")
    environment = require_mapping("environment", root.get("environment"))
    if set(environment) != {
        "git_commit", "image_id", "server", "device", "numeric_settings",
    }:
        raise ValueError("environment field set is inconsistent")
    git_commit = environment.get("git_commit")
    if not isinstance(git_commit, str) or re.fullmatch(r"[0-9a-f]{40}", git_commit) is None:
        raise ValueError("environment.git_commit must be a full Git commit")
    expected_environment = {
        "image_id": R6A_TB2_IMAGE_ID,
        "server": "a100-host",
        "device": "cuda:0",
        "numeric_settings": R6A_TB2_NUMERIC_SETTINGS,
    }
    if any(environment.get(key) != value for key, value in expected_environment.items()):
        raise ValueError("environment does not match the TB2-A fixed run environment")
    if dict(require_mapping("config", root.get("config"))) != R6A_TB2_FIXED_CONFIG:
        raise ValueError("config does not match the q480/48-slot TB2-A contract")
    if dict(require_mapping("dataset", root.get("dataset"))) != R6A_TB2_DATASET:
        raise ValueError("dataset does not match q480 source identity")
    if dict(require_mapping("models", root.get("models"))) != R6A_TB2_MODELS:
        raise ValueError("models are inconsistent with the fixed target/drafter identity")
    if dict(require_mapping("golden", root.get("golden"))) != R6A_TB2_GOLDEN:
        raise ValueError("golden inconsistent with historical target-only result")
    trials = root.get("trials")
    if isinstance(trials, (str, bytes)) or not isinstance(trials, Sequence):
        raise ValueError("trials must be an array")
    if len(trials) != len(R6A_TB2_POLICIES):
        raise ValueError("TB2-A must contain three distinct subtrees")
    return environment, trials


__all__ = [
    "R6A_TB2_DATASET",
    "R6A_TB2_FIXED_CONFIG",
    "R6A_TB2_GOLDEN",
    "R6A_TB2_IMAGE_ID",
    "R6A_TB2_MODELS",
    "R6A_TB2_NUMERIC_SETTINGS",
    "R6A_TB2_SOURCES",
    "require_integer",
    "require_mapping",
    "validate_r6a_tb2_root",
]
