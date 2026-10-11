
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .r6a_experiment import CandidateRecord, PoolRecord, compute_r6a_pool_fingerprint

_POOL_KEYS = {"seed_token_id", "top_k", "max_depth", "raw_candidate_count",
              "fingerprint", "candidates"}
_NODE_KEYS = {"raw_id", "parent_raw_id", "depth", "target_token_id",
              "conditional_log_probability", "conditional_log_probability_hex",
              "cumulative_log_probability", "cumulative_log_probability_hex",
              "generation_ordinal", "path"}


@dataclass(frozen=True, slots=True)
class ReferencePool:
    seed_token_id: int
    top_k: int
    max_depth: int
    fingerprint: str
    nodes: tuple[CandidateRecord, ...]


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


def read_reference_pool(value: object, name: str) -> ReferencePool:

    item = _object(name, value, _POOL_KEYS)
    top_k = _integer(f"{name}.top_k", item["top_k"], 1)
    max_depth = _integer(f"{name}.max_depth", item["max_depth"], 1)
    expected_count = 2 + 4 * (max_depth - 1)
    count = _integer(f"{name}.raw_candidate_count", item["raw_candidate_count"], 1)
    raw_nodes = _sequence(f"{name}.candidates", item["candidates"])
    if top_k != 2 or max_depth > 5 or count != expected_count or len(raw_nodes) != count:
        raise ValueError(f"{name} is not a valid Top-2 effective-depth pool")
    nodes = []
    clean_nodes = []
    for index, raw in enumerate(raw_nodes):
        node = _object(f"{name}.candidates[{index}]", raw, _NODE_KEYS)
        clean = {key: node[key] for key in _NODE_KEYS if not key.endswith("_hex")}
        for field in ("conditional_log_probability", "cumulative_log_probability"):
            number, encoded = node[field], node[f"{field}_hex"]
            if (isinstance(number, bool) or not isinstance(number, (int, float))
                    or not isinstance(encoded, str)):
                raise ValueError(f"{name}.{field} type invalid")
            decoded = float.fromhex(encoded)
            if (not math.isfinite(decoded) or decoded.hex() != encoded
                    or float(number).hex() != encoded):
                raise ValueError(f"{name}.{field} float.hex mismatch")
        candidate = CandidateRecord.read(clean, f"{name}.candidates[{index}]")
        if candidate.conditional_log_probability > 0:
            raise ValueError(f"{name} conditional log probability must be <= 0")
        nodes.append(candidate); clean_nodes.append(clean)
    if tuple(node.raw_id for node in nodes) != tuple(range(1, count + 1)):
        raise ValueError(f"{name} raw IDs must be contiguous")
    by_id: dict[int, CandidateRecord] = {}
    sibling_tokens: dict[int, set[int]] = {}
    for node in nodes:
        parent = by_id.get(node.parent_raw_id) if node.parent_raw_id else None
        parent_depth = parent.depth if parent else 0
        parent_path = parent.path if parent else ()
        parent_score = parent.cumulative_log_probability if parent else 0.0
        if (node.generation_ordinal != node.raw_id - 1
                or (node.parent_raw_id and parent is None)
                or node.depth != parent_depth + 1
                or node.path != parent_path + (node.target_token_id,)
                or not math.isclose(node.cumulative_log_probability,
                    parent_score + node.conditional_log_probability,
                    rel_tol=1e-5, abs_tol=1e-6)):
            raise ValueError(f"{name} topology/probability is inconsistent")
        siblings = sibling_tokens.setdefault(node.parent_raw_id, set())
        if node.target_token_id in siblings:
            raise ValueError(f"{name} sibling token must be unique")
        siblings.add(node.target_token_id); by_id[node.raw_id] = node
    depth_counts = tuple(sum(node.depth == depth for node in nodes)
                         for depth in range(1, max_depth + 1))
    if depth_counts != (2,) + (4,) * (max_depth - 1):
        raise ValueError(f"{name} effective-depth topology is incomplete")
    clean_pool = {"seed_token_id": _integer(f"{name}.seed_token_id", item["seed_token_id"]),
        "top_k": top_k, "max_depth": max_depth, "raw_candidate_count": count,
        "fingerprint": _sha(f"{name}.fingerprint", item["fingerprint"]),
        "candidates": clean_nodes}
    if clean_pool["fingerprint"] != compute_r6a_pool_fingerprint(clean_pool):
        raise ValueError(f"{name} fingerprint inconsistent with content")
    return ReferencePool(clean_pool["seed_token_id"], top_k, max_depth,
                         clean_pool["fingerprint"], tuple(nodes))


def read_full_pool(value: object, name: str) -> PoolRecord:
    pool = read_reference_pool(value, name)
    if pool.max_depth != 5 or len(pool.nodes) != 18:
        raise ValueError(f"{name} must be the complete 18-node pool")
    return PoolRecord(pool.seed_token_id, pool.fingerprint, pool.nodes)


__all__ = ["ReferencePool", "read_full_pool", "read_reference_pool"]
