
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

R6A_SCHEMA_VERSION = 1
R6A_ARTIFACT_TYPE = "r6a_frozen_cycle_subtree"
R6A_TREE_TOP_K = 2
R6A_TREE_MAX_DEPTH = 5
R6A_DRAFT_NODE_BUDGET = 7
R6A_VERIFICATION_POSITIONS = 8
R6A_MOE_LAYER_COUNT = 48
R6A_EXPERTS_PER_TOKEN = 8

def _map(name: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value

def _seq(name: str, value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an array")
    return value

def _int(name: str, value: object, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be no less than {minimum} as an integer")
    return value

def _ints(
    name: str,
    value: object,
    minimum: int = 0,
) -> tuple[int, ...]:
    return tuple(
        _int(f"{name}[{index}]", item, minimum)
        for index, item in enumerate(_seq(name, value))
    )


def _sha(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value.lower() != value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest") from error
    return value


def _digest(payload: object) -> str:
    data = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class CandidateRecord:

    raw_id: int
    parent_raw_id: int
    depth: int
    target_token_id: int
    conditional_log_probability: float
    cumulative_log_probability: float
    generation_ordinal: int
    path: tuple[int, ...]

    @classmethod
    def read(cls, value: object, name: str) -> CandidateRecord:
        item = _map(name, value)
        conditional = item.get("conditional_log_probability")
        cumulative = item.get("cumulative_log_probability")
        for field, number in (
            ("conditional_log_probability", conditional),
            ("cumulative_log_probability", cumulative),
        ):
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(number)
            ):
                raise ValueError(f"{name}.{field} must be a finite number")
        return cls(
            raw_id=_int(f"{name}.raw_id", item.get("raw_id"), 1),
            parent_raw_id=_int(
                f"{name}.parent_raw_id", item.get("parent_raw_id")
            ),
            depth=_int(f"{name}.depth", item.get("depth"), 1),
            target_token_id=_int(
                f"{name}.target_token_id", item.get("target_token_id")
            ),
            conditional_log_probability=float(conditional),
            cumulative_log_probability=float(cumulative),
            generation_ordinal=_int(
                f"{name}.generation_ordinal",
                item.get("generation_ordinal"),
            ),
            path=_ints(f"{name}.path", item.get("path")),
        )

    def fingerprint_payload(self) -> dict[str, object]:
        return {
            "raw_id": self.raw_id,
            "parent_raw_id": self.parent_raw_id,
            "depth": self.depth,
            "target_token_id": self.target_token_id,
            "conditional_log_probability": (
                self.conditional_log_probability.hex()
            ),
            "cumulative_log_probability": (
                self.cumulative_log_probability.hex()
            ),
            "generation_ordinal": self.generation_ordinal,
            "path": list(self.path),
        }


def compute_r6a_pool_fingerprint(pool: Mapping[str, object]) -> str:

    nodes = tuple(
        CandidateRecord.read(item, f"pool.candidates[{index}]")
        for index, item in enumerate(
            _seq("pool.candidates", pool.get("candidates"))
        )
    )
    return _digest(
        {
            "schema_version": 1,
            "seed_token_id": _int(
                "pool.seed_token_id", pool.get("seed_token_id")
            ),
            "top_k": _int("pool.top_k", pool.get("top_k"), 1),
            "max_depth": _int(
                "pool.max_depth", pool.get("max_depth"), 1
            ),
            "candidates": [node.fingerprint_payload() for node in nodes],
        }
    )


@dataclass(frozen=True, slots=True)
class PoolRecord:

    seed_token_id: int
    fingerprint: str
    nodes: tuple[CandidateRecord, ...]

    @classmethod
    def read(cls, value: object) -> PoolRecord:
        pool = _map("pool", value)
        fixed = (
            pool.get("top_k"),
            pool.get("max_depth"),
            pool.get("raw_candidate_count"),
        )
        if fixed != (2, 5, 18):
            raise ValueError("full pool must be the 18 nodes of Top-2/depth-5")
        nodes = tuple(
            CandidateRecord.read(item, f"pool.candidates[{index}]")
            for index, item in enumerate(
                _seq("pool.candidates", pool.get("candidates"))
            )
        )
        if tuple(node.raw_id for node in nodes) != tuple(range(1, 19)):
            raise ValueError("pool raw IDs must contiguously cover 1..18")
        by_id: dict[int, CandidateRecord] = {}
        for node in nodes:
            parent = by_id.get(node.parent_raw_id)
            if node.parent_raw_id and parent is None:
                raise ValueError("pool parent must precede child")
            parent_depth = parent.depth if parent else 0
            parent_path = parent.path if parent else ()
            parent_score = parent.cumulative_log_probability if parent else 0.0
            if (
                node.generation_ordinal != node.raw_id - 1
                or node.depth != parent_depth + 1
                or node.path != parent_path + (node.target_token_id,)
            ):
                raise ValueError("pool raw topology is inconsistent")
            if not math.isclose(
                node.cumulative_log_probability,
                parent_score + node.conditional_log_probability,
                rel_tol=1e-5,
                abs_tol=1e-6,
            ):
                raise ValueError("pool cumulative log probability mismatch")
            by_id[node.raw_id] = node
        depth_counts = tuple(
            sum(node.depth == depth for node in nodes)
            for depth in range(1, 6)
        )
        if depth_counts != (2, 4, 4, 4, 4):
            raise ValueError("pool per-depth node counts must be 2/4/4/4/4")
        fingerprint = _sha("pool.fingerprint", pool.get("fingerprint"))
        if fingerprint != compute_r6a_pool_fingerprint(pool):
            raise ValueError("pool fingerprint inconsistent with content")
        return cls(
            seed_token_id=_int(
                "pool.seed_token_id", pool.get("seed_token_id")
            ),
            fingerprint=fingerprint,
            nodes=nodes,
        )


@dataclass(frozen=True, slots=True)
class FrozenStateRecord:

    fingerprint: str
    resident_by_layer: tuple[tuple[int, ...], ...]

    @classmethod
    def read(
        cls,
        value: object,
        *,
        slots: int,
        num_experts: int,
    ) -> FrozenStateRecord:
        frozen = _map("frozen_state", value)
        fingerprint = _sha(
            "frozen_state.cache_state_fingerprint",
            frozen.get("cache_state_fingerprint"),
        )
        if frozen.get("quiescent") is not True:
            raise ValueError("frozen state must be quiescent")
        layers = _seq(
            "resident_experts_by_layer",
            frozen.get("resident_experts_by_layer"),
        )
        if len(layers) != R6A_MOE_LAYER_COUNT:
            raise ValueError("frozen state must record residency for 48 layers")
        resident = []
        for layer_id, item in enumerate(layers):
            layer = _map("resident layer", item)
            experts = _ints("resident experts", layer.get("expert_ids"))
            invalid = (
                layer.get("layer_id") != layer_id
                or experts != tuple(sorted(set(experts)))
                or len(experts) > slots
                or any(expert >= num_experts for expert in experts)
            )
            if invalid:
                raise ValueError("frozen residency mismatch")
            resident.append(experts)
        return cls(fingerprint, tuple(resident))


def compute_r6a_route_digest(
    *,
    pool_fingerprint: str,
    verification_position_to_raw_id: Sequence[int],
    rows: Sequence[Mapping[str, object]],
) -> str:

    from .r6a_route_observation import compute_r6a_route_digest as compute

    return compute(
        pool_fingerprint=pool_fingerprint,
        verification_position_to_raw_id=verification_position_to_raw_id,
        rows=rows,
    )


def validate_r6a_artifact(artifact: Mapping[str, object]) -> None:

    from .r6a_artifact_validation import validate_r6a_artifact as validate

    validate(artifact)


__all__ = [name for name in globals() if name.startswith("R6A_")] + [
    "CandidateRecord",
    "FrozenStateRecord",
    "PoolRecord",
    "compute_r6a_pool_fingerprint",
    "compute_r6a_route_digest",
    "validate_r6a_artifact",
]
