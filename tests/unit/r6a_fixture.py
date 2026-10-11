
from __future__ import annotations

from specfit.r6a_experiment import (
    FrozenStateRecord,
    PoolRecord,
    R6A_ARTIFACT_TYPE,
    compute_r6a_pool_fingerprint,
    compute_r6a_route_digest,
)
from specfit.r6a_route_observation import RouteObservation
from specfit.r6a_identity import (
    R6A_DATASET_IDENTITY,
    R6A_MODEL_IDENTITIES,
)
from specfit.r6a_selection_contract import (
    enumerate_ancestor_closed_subsets,
    expected_policy_decisions,
)

class R6AFixture:
    parents = {
        1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2,
        7: 3, 8: 3, 9: 5, 10: 5, 11: 7, 12: 7,
        13: 9, 14: 9, 15: 11, 16: 11, 17: 13, 18: 13,
    }
    alternate_parents = {**parents, 9: 4, 10: 4}
    frozen_fingerprint = "a" * 64
    expert_bytes = 1_024

    @classmethod
    def pool(cls, parents: dict[int, int] | None = None) -> dict[str, object]:
        nodes: dict[int, dict[str, object]] = {}
        candidates = []
        topology = cls.parents if parents is None else parents
        for raw_id, parent_id in topology.items():
            parent = nodes.get(parent_id)
            depth = int(parent["depth"]) + 1 if parent else 1
            path = [*(parent["path"] if parent else []), 1_000 + raw_id]
            conditional = -0.1 - raw_id / 1_000
            cumulative = (
                float(parent["cumulative_log_probability"]) + conditional
                if parent
                else conditional
            )
            node = {
                "raw_id": raw_id,
                "parent_raw_id": parent_id,
                "depth": depth,
                "target_token_id": 1_000 + raw_id,
                "conditional_log_probability": conditional,
                "cumulative_log_probability": cumulative,
                "generation_ordinal": raw_id - 1,
                "path": path,
            }
            nodes[raw_id] = node
            candidates.append(node)
        pool = {
            "seed_token_id": 42,
            "top_k": 2,
            "max_depth": 5,
            "raw_candidate_count": 18,
            "candidates": candidates,
        }
        pool["fingerprint"] = compute_r6a_pool_fingerprint(pool)
        return pool
    @staticmethod
    def mapping(selected: tuple[int, ...]) -> dict[str, object]:
        positions = (0, *selected)
        return {
            "selected_raw_ids": list(selected),
            "verification_position_to_raw_id": list(positions),
            "raw_to_verification_position": {
                str(raw): position for position, raw in enumerate(positions)
            },
        }
    @staticmethod
    def rows(positions: tuple[int, ...]) -> list[dict[str, object]]:
        rows = []
        for position, raw_id in enumerate(positions):
            for layer_id in range(48):
                first = (raw_id * 11 + layer_id * 3) % 128
                rows.append(
                    {
                        "verification_position": position,
                        "raw_id": raw_id,
                        "layer_id": layer_id,
                        "expert_ids": [
                            (first + offset) % 128 for offset in range(8)
                        ],
                    }
                )
        return rows
    @classmethod
    def routes(
        cls,
        pool_fingerprint: str,
        positions: tuple[int, ...],
    ) -> dict[str, object]:
        rows = cls.rows(positions)
        return {
            "pool_fingerprint": pool_fingerprint,
            "coverage_rows": len(rows),
            "digest": compute_r6a_route_digest(
                pool_fingerprint=pool_fingerprint,
                verification_position_to_raw_id=positions,
                rows=rows,
            ),
            "rows": rows,
        }
    @staticmethod
    def frozen_state() -> dict[str, object]:
        return {
            "cache_state_fingerprint": "a" * 64,
            "quiescent": True,
            "resident_experts_by_layer": [
                {"layer_id": layer, "expert_ids": list(range(48))}
                for layer in range(48)
            ],
        }
    @staticmethod
    def acceptance(
        selected: tuple[int, ...],
        pool: PoolRecord,
        golden: tuple[int, ...],
    ) -> dict[str, object]:
        by_id = {node.raw_id: node for node in pool.nodes}
        accepted = []
        parent = 0
        for token_id in golden[1:6]:
            matches = [
                raw_id
                for raw_id in selected
                if by_id[raw_id].parent_raw_id == parent
                and by_id[raw_id].target_token_id == token_id
            ]
            if not matches:
                break
            parent = matches[0]
            accepted.append(parent)
        committed = [
            pool.seed_token_id,
            *(by_id[raw_id].target_token_id for raw_id in accepted),
        ]
        return {
            "accepted_draft_tokens": len(accepted),
            "accepted_raw_ids": accepted,
            "committed_token_ids": committed,
            "next_pending_seed_token_id": golden[len(committed)],
        }
    @staticmethod
    def runtime(post: str, acquisitions: int | None = None) -> dict[str, object]:
        h2d = min(300, acquisitions or 420)
        hits = (acquisitions - h2d) if acquisitions is not None else 120
        return {
            "cache_state": {
                "pre_fingerprint": "a" * 64,
                "post_fingerprint": post * 64,
            },
            "timings": {
                "proposal_wall_ns": 100,
                "selector_wall_ns": 10,
                "target_verification_wall_ns": 200,
                "cycle_wall_ns": 350,
                "target_verification_cuda_ns": 180,
            },
            "cache_counter_delta": {
                "h2d_submissions": h2d,
                "resident_hits": hits,
                "inflight_joins": 0,
                "evictions": min(h2d, 252),
                "unique_expert_acquisitions": acquisitions or 420,
                "expert_bytes": R6AFixture.expert_bytes,
                "logical_h2d_bytes": h2d * R6AFixture.expert_bytes,
            },
        }
    @classmethod
    def artifact(cls, parents: dict[int, int] | None = None) -> dict[str, object]:
        pool_data = cls.pool(parents)
        pool = PoolRecord.read(pool_data)
        fingerprint = pool.fingerprint
        golden = (42, 1_001, 1_003, 1_007, 1_011, 1_015, 9_999)
        frozen_data = cls.frozen_state()
        frozen = FrozenStateRecord.read(frozen_data, slots=48, num_experts=128)
        atlas_ids = tuple(range(1, 19))
        atlas_positions = (0, *atlas_ids)
        atlas_route_data = cls.routes(fingerprint, atlas_positions)
        atlas_routes = RouteObservation.read(
            atlas_route_data,
            name="atlas.routes",
            position_to_raw=atlas_positions,
            pool_fingerprint=fingerprint,
            num_experts=128,
        )
        subsets = enumerate_ancestor_closed_subsets(pool)
        scores = expected_policy_decisions(
            pool, atlas_routes, frozen, subsets=subsets
        )
        decisions = [
            {
                "policy": policy,
                "selected_raw_ids": list(score.selected_raw_ids),
                "draft_utility_proxy": score.draft_utility_proxy,
                "unique_expert_count": score.unique_expert_count,
                "initial_frozen_nonresident_expert_count": (
                    score.initial_frozen_nonresident_expert_count
                ),
            }
            for policy, score in scores.items()
        ]
        atlas = {
            "execution_index": 0,
            **cls.mapping(atlas_ids),
            "routes": atlas_route_data,
            **cls.runtime("b", len(atlas_routes.unique_expert_keys())),
            "acceptance": cls.acceptance(atlas_ids, pool, golden),
        }
        trials = []
        for execution_index, decision in enumerate(decisions, start=1):
            selected = tuple(decision["selected_raw_ids"])
            positions = (0, *selected)
            route_data = cls.routes(fingerprint, positions)
            observation = RouteObservation.read(
                route_data,
                name="trial.routes",
                position_to_raw=positions,
                pool_fingerprint=fingerprint,
                num_experts=128,
            )
            trials.append(
                {
                    "execution_index": execution_index,
                    "trial_id": f"q480-{decision['policy']}",
                    "selection": {
                        "policy": decision["policy"],
                        **cls.mapping(selected),
                    },
                    "routes": route_data,
                    **cls.runtime(
                        chr(ord("b") + execution_index),
                        len(observation.unique_expert_keys()),
                    ),
                    "acceptance": cls.acceptance(selected, pool, golden),
                }
            )
        return {
            "schema_version": 1,
            "artifact_type": R6A_ARTIFACT_TYPE,
            "status": "complete",
            "dataset": {
                **R6A_DATASET_IDENTITY,
                "question_id": 480,
            },
            "prompt": {
                "input_ids_length": 128,
                "input_ids_sha256": "d" * 64,
            },
            "models": {
                name: dict(identity)
                for name, identity in R6A_MODEL_IDENTITIES.items()
            },
            "environment": {
                "git_commit": "abc",
                "image_id": "ngc25.11",
                "server": "a100-host",
                "device": "cuda:0",
            },
            "config": {
                "batch_size": 1,
                "tree_top_k": 2,
                "tree_max_depth": 5,
                "draft_node_budget": 7,
                "verification_positions": 8,
                "num_moe_layers": 48,
                "experts_per_token": 8,
                "cycles_per_trial": 1,
                "num_routed_experts": 128,
                "slots_per_layer": 48,
                "expert_bytes": cls.expert_bytes,
                "dtype": "bfloat16",
                "numeric_mode": "math_bf16_full_reduction",
            },
            "golden": {
                "question_id": 480,
                "output_token_ids": list(golden),
            },
            "pool": pool_data,
            "frozen_state": frozen_data,
            "route_atlas_probe": atlas,
            "offline_selection": {
                "ancestor_closed_fixed_budget_subset_count": len(subsets),
                "decisions": decisions,
            },
            "trials": trials,
        }
__all__ = ["R6AFixture"]
