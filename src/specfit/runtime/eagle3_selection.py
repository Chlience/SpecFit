
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Protocol

from .eagle3_types import Eagle3DraftCandidatePool, Eagle3DraftTree


class RawIdSelectionPolicy(Protocol):

    def select_raw_ids(
        self,
        pool: Eagle3DraftCandidatePool,
        *,
        draft_node_budget: int,
        committed_offset: int = 0,
    ) -> tuple[int, ...]: ...


def candidate_pool_fingerprint(pool: Eagle3DraftCandidatePool) -> str:

    payload = {
        "schema_version": 1,
        "seed_token_id": pool.seed_token_id,
        "top_k": pool.top_k,
        "max_depth": pool.max_depth,
        "candidates": [
            {
                "raw_id": node.raw_id,
                "parent_raw_id": node.parent_raw_id,
                "depth": node.depth,
                "target_token_id": node.target_token_id,
                "conditional_log_probability": (
                    float(node.conditional_log_probability).hex()
                ),
                "cumulative_log_probability": (
                    float(node.cumulative_log_probability).hex()
                ),
                "generation_ordinal": node.generation_ordinal,
                "path": list(node.path),
            }
            for node in pool.candidates
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class TreeSelectionDecision:

    pool_fingerprint: str
    draft_node_budget: int
    committed_offset: int
    selected_raw_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            len(self.pool_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.pool_fingerprint
            )
        ):
            raise ValueError("pool fingerprint must be a lowercase SHA-256 digest")
        if (
            isinstance(self.draft_node_budget, bool)
            or not isinstance(self.draft_node_budget, int)
            or self.draft_node_budget <= 0
        ):
            raise ValueError("draft node budget must be a positive integer")
        if (
            isinstance(self.committed_offset, bool)
            or not isinstance(self.committed_offset, int)
            or self.committed_offset < 0
        ):
            raise ValueError("committed offset must be a non-negative integer")
        if len(self.selected_raw_ids) > self.draft_node_budget:
            raise ValueError("selected raw IDs exceed the draft node budget")
        if any(
            isinstance(raw_id, bool)
            or not isinstance(raw_id, int)
            or raw_id <= 0
            for raw_id in self.selected_raw_ids
        ):
            raise ValueError("selected raw IDs must be positive integers")
        if len(set(self.selected_raw_ids)) != len(self.selected_raw_ids):
            raise ValueError("selected raw IDs must be unique")

    @classmethod
    def from_pool(
        cls,
        pool: Eagle3DraftCandidatePool,
        *,
        draft_node_budget: int,
        committed_offset: int,
        selected_raw_ids: tuple[int, ...],
    ) -> TreeSelectionDecision:

        if len(selected_raw_ids) > draft_node_budget:
            raise ValueError("selected raw IDs exceed the draft node budget")
        tree = pool.to_tree(selected_raw_ids)
        return cls(
            pool_fingerprint=candidate_pool_fingerprint(pool),
            draft_node_budget=draft_node_budget,
            committed_offset=committed_offset,
            selected_raw_ids=tree.source_raw_ids,
        )

    @property
    def verification_raw_ids(self) -> tuple[int, ...]:

        return (0,) + self.selected_raw_ids

    def materialize(self, pool: Eagle3DraftCandidatePool) -> Eagle3DraftTree:

        if candidate_pool_fingerprint(pool) != self.pool_fingerprint:
            raise ValueError("selection decision does not match the candidate pool")
        tree = pool.to_tree(self.selected_raw_ids)
        if tree.source_raw_ids != self.selected_raw_ids:
            raise RuntimeError("selection decision changed verification order")
        return tree


class ExplicitRawIdTreeSelector:

    def __init__(self, policy: RawIdSelectionPolicy) -> None:
        if not callable(getattr(policy, "select_raw_ids", None)):
            raise TypeError("raw-ID policy must define select_raw_ids")
        self.policy = policy

    def decide(
        self,
        pool: Eagle3DraftCandidatePool,
        *,
        draft_node_budget: int,
        committed_offset: int = 0,
    ) -> TreeSelectionDecision:
        selected_raw_ids = self.policy.select_raw_ids(
            pool,
            draft_node_budget=draft_node_budget,
            committed_offset=committed_offset,
        )
        return TreeSelectionDecision.from_pool(
            pool,
            draft_node_budget=draft_node_budget,
            committed_offset=committed_offset,
            selected_raw_ids=selected_raw_ids,
        )

    def select(
        self,
        pool: Eagle3DraftCandidatePool,
        *,
        draft_node_budget: int,
        committed_offset: int = 0,
    ) -> Eagle3DraftTree:
        return self.decide(
            pool,
            draft_node_budget=draft_node_budget,
            committed_offset=committed_offset,
        ).materialize(pool)


__all__ = [
    "ExplicitRawIdTreeSelector",
    "RawIdSelectionPolicy",
    "TreeSelectionDecision",
    "candidate_pool_fingerprint",
]
