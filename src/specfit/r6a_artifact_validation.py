
from __future__ import annotations

from collections.abc import Mapping

from .r6a_experiment import (
    FrozenStateRecord,
    PoolRecord,
    R6A_ARTIFACT_TYPE,
    _int,
    _ints,
    _map,
    _seq,
    _sha,
)
from .r6a_identity import ExperimentIdentity
from .r6a_route_observation import (
    RouteObservation,
    validate_acquisition_counters,
)
from .r6a_selection_contract import (
    R6A_POLICIES,
    validate_policy_decisions,
)

_TIMINGS = (
    "proposal_wall_ns",
    "selector_wall_ns",
    "target_verification_wall_ns",
    "cycle_wall_ns",
    "target_verification_cuda_ns",
)
_COUNTERS = (
    "h2d_submissions",
    "resident_hits",
    "inflight_joins",
    "evictions",
    "unique_expert_acquisitions",
    "expert_bytes",
    "logical_h2d_bytes",
)

def _selected(
    value: object,
    pool: PoolRecord,
    *,
    budget: int,
) -> tuple[int, ...]:
    selected = _ints("selected_raw_ids", value, 1)
    if len(selected) != budget or selected != tuple(sorted(set(selected))):
        raise ValueError(f"selection must contain {budget} ordered raw IDs")
    by_id = {node.raw_id: node for node in pool.nodes}
    chosen = set(selected)
    invalid = any(
        raw_id not in by_id
        or (
            by_id[raw_id].parent_raw_id
            and by_id[raw_id].parent_raw_id not in chosen
        )
        for raw_id in selected
    )
    if invalid:
        raise ValueError("selection does not satisfy ancestor closure")
    return selected

def _mapping(
    value: Mapping[str, object],
    pool: PoolRecord,
    expected: tuple[int, ...],
) -> tuple[int, ...]:
    selected = _selected(
        value.get("selected_raw_ids"), pool, budget=len(expected)
    )
    if selected != expected:
        raise ValueError("trial selection inconsistent with offline decision")
    positions = _ints(
        "verification_position_to_raw_id",
        value.get("verification_position_to_raw_id"),
    )
    inverse = {
        str(raw_id): position for position, raw_id in enumerate(positions)
    }
    if positions != (0, *selected) or dict(
        _map(
            "raw_to_verification_position",
            value.get("raw_to_verification_position"),
        )
    ) != inverse:
        raise ValueError("raw ID to verification position mapping inconsistent")
    return positions

def _runtime(
    value: Mapping[str, object],
    *,
    name: str,
    frozen_fingerprint: str,
) -> dict[str, int]:
    cache = _map(f"{name}.cache_state", value.get("cache_state"))
    pre = _sha(f"{name}.pre_fingerprint", cache.get("pre_fingerprint"))
    if pre != frozen_fingerprint:
        raise ValueError("verification pre-state does not match frozen state")
    _sha(f"{name}.post_fingerprint", cache.get("post_fingerprint"))
    values: dict[str, int] = {}
    for group, fields in (
        ("timings", _TIMINGS),
        ("cache_counter_delta", _COUNTERS),
    ):
        record = _map(f"{name}.{group}", value.get(group))
        for field in fields:
            number = _int(f"{name}.{group}.{field}", record.get(field))
            if group == "cache_counter_delta":
                values[field] = number
    return values

def _acceptance(
    value: object,
    *,
    name: str,
    pool: PoolRecord,
    selected: tuple[int, ...],
    golden: tuple[int, ...],
) -> None:
    acceptance = _map(name, value)
    accepted = _ints(
        f"{name}.accepted_raw_ids",
        acceptance.get("accepted_raw_ids"),
        1,
    )
    if acceptance.get("accepted_draft_tokens") != len(accepted):
        raise ValueError("accepted draft count mismatch")
    by_id = {node.raw_id: node for node in pool.nodes}
    expected_accepted = []
    parent_raw_id = 0
    for token_id in golden[1:6]:
        matches = [
            raw_id
            for raw_id in selected
            if by_id[raw_id].parent_raw_id == parent_raw_id
            and by_id[raw_id].target_token_id == token_id
        ]
        if len(matches) > 1:
            raise ValueError("selected tree has duplicate tokens under the same parent")
        if not matches:
            break
        parent_raw_id = matches[0]
        expected_accepted.append(parent_raw_id)
    if accepted != tuple(expected_accepted):
        raise ValueError("accepted raw IDs are not the golden longest matching path")
    committed = _ints(
        f"{name}.committed_token_ids",
        acceptance.get("committed_token_ids"),
    )
    expected = (
        pool.seed_token_id,
        *(by_id[raw_id].target_token_id for raw_id in accepted),
    )
    pending = acceptance.get("next_pending_seed_token_id")
    if (
        committed != expected
        or committed != golden[: len(committed)]
        or len(golden) <= len(committed)
        or pending != golden[len(committed)]
    ):
        raise ValueError("committed golden prefix or next pending seed inconsistent")

def validate_r6a_artifact(artifact: Mapping[str, object]) -> None:

    root = _map("artifact", artifact)
    identity = (
        root.get("schema_version"),
        root.get("artifact_type"),
        root.get("status"),
    )
    if identity != (1, R6A_ARTIFACT_TYPE, "complete"):
        raise ValueError("artifact must be complete R6a schema v1")
    config = _map("config", root.get("config"))
    fixed = {
        "batch_size": 1,
        "tree_top_k": 2,
        "tree_max_depth": 5,
        "draft_node_budget": 7,
        "verification_positions": 8,
        "num_moe_layers": 48,
        "experts_per_token": 8,
        "cycles_per_trial": 1,
    }
    if any(config.get(key) != expected for key, expected in fixed.items()):
        raise ValueError("config does not match the fixed R6a-TB1 contract")
    identity_record = ExperimentIdentity.read(root, config)
    num_experts = _int(
        "num_routed_experts", config.get("num_routed_experts"), 1
    )
    slots = _int("slots_per_layer", config.get("slots_per_layer"), 1)
    expert_bytes = _int("expert_bytes", config.get("expert_bytes"), 1)
    if slots >= num_experts:
        raise ValueError("R6a-TB1 requires partial expert residency")
    environment = _map("environment", root.get("environment"))
    required_environment = ("git_commit", "image_id", "server", "device")
    if any(not environment.get(field) for field in required_environment):
        raise ValueError("environment identity incomplete")
    golden_record = _map("golden", root.get("golden"))
    if golden_record.get("question_id") != identity_record.question_id:
        raise ValueError("golden.question_id does not match the dataset")
    golden = _ints("golden.output_token_ids", golden_record.get("output_token_ids"))
    pool = PoolRecord.read(root.get("pool"))
    if len(golden) < 7 or golden[0] != pool.seed_token_id:
        raise ValueError("golden must contain at least 7 tokens and the first token must equal seed")
    frozen_state = FrozenStateRecord.read(
        root.get("frozen_state"),
        slots=slots,
        num_experts=num_experts,
    )
    frozen = frozen_state.fingerprint
    atlas = _map("route_atlas_probe", root.get("route_atlas_probe"))
    all_raw_ids = tuple(range(1, 19))
    atlas_positions = _mapping(atlas, pool, all_raw_ids)
    atlas_routes = RouteObservation.read(
        atlas.get("routes"),
        name="route_atlas_probe.routes",
        position_to_raw=atlas_positions,
        pool_fingerprint=pool.fingerprint,
        num_experts=num_experts,
    )
    atlas_counters = _runtime(atlas, name="route_atlas_probe", frozen_fingerprint=frozen)
    validate_acquisition_counters(
        atlas_counters,
        atlas_routes,
        name="route_atlas_probe",
        expert_bytes=expert_bytes,
    )
    _acceptance(
        atlas.get("acceptance"),
        name="route_atlas_probe.acceptance",
        pool=pool,
        selected=all_raw_ids,
        golden=golden,
    )
    decisions = validate_policy_decisions(
        root.get("offline_selection"),
        pool=pool,
        atlas=atlas_routes,
        frozen=frozen_state,
    )
    seen_indices = {_int("atlas.execution_index", atlas.get("execution_index"))}
    seen_ids: set[str] = set()
    trial_policies: set[str] = set()
    trials = _seq("trials", root.get("trials"))
    if len(trials) != len(R6A_POLICIES):
        raise ValueError("complete artifact must contain exactly three formal trials")
    for index, item in enumerate(trials):
        trial = _map(f"trials[{index}]", item)
        selection = _map("trial.selection", trial.get("selection"))
        policy = selection.get("policy")
        if not isinstance(policy, str) or policy not in decisions:
            raise ValueError("trial policy is missing an offline decision")
        positions = _mapping(selection, pool, decisions[policy])
        own_routes = RouteObservation.read(
            trial.get("routes"),
            name=f"trials[{index}].routes",
            position_to_raw=positions,
            pool_fingerprint=pool.fingerprint,
            num_experts=num_experts,
        )
        counters = _runtime(
            trial,
            name=f"trials[{index}]",
            frozen_fingerprint=frozen,
        )
        validate_acquisition_counters(
            counters,
            own_routes,
            name=f"trials[{index}]",
            expert_bytes=expert_bytes,
        )
        _acceptance(
            trial.get("acceptance"),
            name=f"trials[{index}].acceptance",
            pool=pool,
            selected=decisions[policy],
            golden=golden,
        )
        execution_index = _int(
            "trial.execution_index", trial.get("execution_index")
        )
        trial_id = trial.get("trial_id")
        if (
            execution_index in seen_indices
            or not isinstance(trial_id, str)
            or not trial_id
            or trial_id in seen_ids
        ):
            raise ValueError("trial execution_index/trial_id must be unique")
        seen_indices.add(execution_index)
        seen_ids.add(trial_id)
        trial_policies.add(policy)
    if trial_policies != set(R6A_POLICIES):
        raise ValueError("complete artifact must each run the formal policy exactly once")
__all__ = ["validate_r6a_artifact"]
