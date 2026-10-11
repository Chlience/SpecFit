
from __future__ import annotations
from dataclasses import asdict, dataclass

from .experiment import canonical_json_sha256
from .r6a_route_observation import compute_r6a_route_digest
from .runtime import (
    CumulativeProbabilityTreeSelector,
    Eagle3DraftCandidatePool,
    candidate_pool_fingerprint,
)
from .trace import CacheSnapshotTrace, SnapshotPhase, TraceBundle

_COUNTERS = ("h2d_submissions", "resident_hits", "inflight_joins", "evictions",
             "pending_use_guards")
_SLOTS = ("slot_id", "expert_id", "generation", "active_leases",
          "last_used_sequence", "ready", "pending_use_guards")
@dataclass(frozen=True, slots=True)
class M1Selection:
    pool: Eagle3DraftCandidatePool
    committed_offset: int
    selected_raw_ids: tuple[int, ...]
class ScheduledM1Policy:
    def __init__(self, *, probe_cycle: int | None = None) -> None:
        if probe_cycle is not None and (
            isinstance(probe_cycle, bool) or not isinstance(probe_cycle, int)
            or probe_cycle < 0
        ):
            raise ValueError("probe_cycle must be a non-negative integer")
        self.probe_cycle = probe_cycle
        self.records: list[M1Selection] = []
        self._probability = CumulativeProbabilityTreeSelector()

    def select_raw_ids(
        self,
        pool: Eagle3DraftCandidatePool,
        *,
        draft_node_budget: int,
        committed_offset: int = 0,
    ) -> tuple[int, ...]:
        cycle = len(self.records)
        raw_ids = tuple(node.raw_id for node in pool.candidates)
        expected_nodes = 2 + 4 * (pool.max_depth - 1)
        is_probe = cycle == self.probe_cycle
        if (draft_node_budget != 18 or pool.top_k != 2
                or not 1 <= pool.max_depth <= 5
                or raw_ids != tuple(range(1, expected_nodes + 1))
                or is_probe and expected_nodes != 18):
            raise RuntimeError("M1 pool does not match reference/probe fixed topology")
        if self.probe_cycle is not None and cycle > self.probe_cycle:
            raise RuntimeError("generation must not continue after M1 full-atlas probe")
        if is_probe:
            selected = raw_ids
        else:
            selected = self._probability.select_raw_ids(
                pool,
                draft_node_budget=min(7, expected_nodes),
                committed_offset=committed_offset,
            )
        normalized = pool.to_tree(selected).source_raw_ids
        expected = 18 if is_probe else min(7, expected_nodes)
        if len(normalized) != expected:
            raise RuntimeError("M1 selector did not fill the fixed node budget")
        self.records.append(M1Selection(pool, committed_offset, normalized))
        return normalized
def quartile_cycle_indices(cycle_count: int) -> tuple[int, int, int, int]:
    if (isinstance(cycle_count, bool) or not isinstance(cycle_count, int)
            or cycle_count < 4):
        raise ValueError("M1 reference requires at least 4 cycles")
    return tuple(((2 * index + 1) * cycle_count) // 8 for index in range(4))
def serialize_m1_pool(pool: Eagle3DraftCandidatePool) -> dict[str, object]:
    candidates = []
    for node in pool.candidates:
        item = asdict(node)
        item["path"] = list(node.path)
        item["conditional_log_probability_hex"] = float(
            node.conditional_log_probability).hex()
        item["cumulative_log_probability_hex"] = float(
            node.cumulative_log_probability).hex()
        candidates.append(item)
    return {
        "seed_token_id": pool.seed_token_id,
        "top_k": pool.top_k,
        "max_depth": pool.max_depth,
        "raw_candidate_count": pool.raw_candidate_count,
        "fingerprint": candidate_pool_fingerprint(pool),
        "candidates": candidates,
    }
def project_logical_cache_state(snapshot: CacheSnapshotTrace) -> dict[str, object]:
    layers = []
    for layer in snapshot.layers:
        if layer.lru_sequence is None or layer.stats is None:
            raise ValueError("M1 requires a fully expanded cache snapshot")
        layers.append({
            "layer_id": layer.layer_id,
            "capacity": layer.capacity,
            "lru_sequence": layer.lru_sequence,
            "submission_sequence": layer.submission_sequence,
            "counters": {name: getattr(layer.stats, name) for name in _COUNTERS},
            "slots": [
                {name: getattr(slot, name) for name in _SLOTS}
                for slot in layer.slots
            ],
        })
    return {"layers": layers}
def logical_cache_state_sha256(snapshot: CacheSnapshotTrace) -> str:
    return canonical_json_sha256(project_logical_cache_state(snapshot))
def require_quiescent(snapshot: CacheSnapshotTrace) -> None:
    if any(
        layer.pending_transfers
        or any(
            slot.active_leases or slot.pending_use_guards
            or (slot.expert_id is not None and not slot.ready)
            for slot in layer.slots
        )
        for layer in snapshot.layers
    ):
        raise RuntimeError("M1 selection boundary is not a quiescent ready state")
def project_frozen_state(snapshot: CacheSnapshotTrace) -> dict[str, object]:
    require_quiescent(snapshot)
    logical = project_logical_cache_state(snapshot)
    return {
        "cache_state_fingerprint": canonical_json_sha256(logical),
        "quiescent": True,
        "resident_experts_by_layer": [
            {"layer_id": layer.layer_id, "expert_ids": sorted(
                slot.expert_id for slot in layer.slots
                if slot.expert_id is not None and slot.ready
            )} for layer in snapshot.layers
        ],
        "logical_state": logical,
    }
def _cycles(trace: TraceBundle):
    return tuple(sorted(trace.cycles, key=lambda cycle: cycle.iteration))
def _snapshot(trace: TraceBundle, cycle_index: int, phase: SnapshotPhase):
    cycle = _cycles(trace)[cycle_index]
    matches = tuple(
        item for item in trace.snapshots
        if item.cycle_id == cycle.trace_id and item.phase is phase
    )
    if len(matches) != 1:
        raise RuntimeError("M1 cycle missing a unique cache snapshot")
    return matches[0]
def _routes(
    trace: TraceBundle,
    cycle_index: int,
    position_to_raw: tuple[int, ...],
    pool_fingerprint: str,
) -> dict[str, object]:
    cycle = _cycles(trace)[cycle_index]
    rows = [{
        "verification_position": route.node_id,
        "raw_id": position_to_raw[route.node_id],
        "layer_id": route.layer_id,
        "expert_ids": list(route.expert_ids),
    } for route in trace.target_routes if route.tree_id == cycle.tree_id]
    rows.sort(key=lambda row: (row["verification_position"], row["layer_id"]))
    expected = len(position_to_raw) * trace.manifest.num_moe_layers
    if len(rows) != expected:
        raise RuntimeError("M1 target route coverage is incomplete")
    return {
        "pool_fingerprint": pool_fingerprint,
        "coverage_rows": len(rows),
        "digest": compute_r6a_route_digest(
            pool_fingerprint=pool_fingerprint,
            verification_position_to_raw_id=position_to_raw,
            rows=rows,
        ),
        "rows": rows,
    }
def project_reference_rollout(
    *, question_id: int, input_token_ids: tuple[int, ...], trace: TraceBundle,
    selections: tuple[M1Selection, ...],
) -> dict[str, object]:
    ordered_cycles = _cycles(trace)
    if len(ordered_cycles) != len(selections) or len(trace.generations) != 1:
        raise RuntimeError("M1 reference cycle/selection counts do not close")
    context = list(input_token_ids)
    cycles = []
    previous_post = None
    for index, (cycle, selection) in enumerate(zip(ordered_cycles, selections, strict=True)):
        pool = selection.pool
        mapping = (0, *selection.selected_raw_ids)
        pre = _snapshot(trace, index, SnapshotPhase.BEFORE_VERIFY)
        post = _snapshot(trace, index, SnapshotPhase.AFTER_VERIFY)
        require_quiescent(pre)
        require_quiescent(post)
        tree = pool.to_tree(selection.selected_raw_ids)
        invalid = (
            cycle.iteration != index or cycle.base_kv_length != len(context)
            or selection.committed_offset != len(context) - len(input_token_ids)
            or cycle.pending_seed_token_id != pool.seed_token_id
            or cycle.verification_token_ids != (pool.seed_token_id, *tree.token_ids)
            or previous_post not in (None, logical_cache_state_sha256(pre))
        )
        if invalid:
            raise RuntimeError("M1 reference context, pool, or cache chain is not closed")
        accepted_raw = [mapping[position] for position in cycle.accepted_draft_node_ids]
        last_position = cycle.accepted_draft_node_ids[-1] if accepted_raw else 0
        record = {
            "cycle_index": index,
            "base_kv_length": cycle.base_kv_length,
            "context_token_ids": list(context),
            "context_token_count": len(context),
            "context_token_ids_sha256": canonical_json_sha256(context),
            "pending_seed_token_id": cycle.pending_seed_token_id,
            "committed_offset": selection.committed_offset,
            "pool": serialize_m1_pool(pool),
            "selected_raw_ids": list(selection.selected_raw_ids),
            "selected_routes": _routes(trace, index, mapping, candidate_pool_fingerprint(pool)),
            "verifier_token_ids": list(cycle.verifier_token_ids),
            "transition": {
                "accepted_raw_ids": accepted_raw,
                "committed_token_ids": list(cycle.committed_token_ids),
                "next_pending_seed_token_id": cycle.verifier_token_ids[last_position],
            },
            "pre_logical_state_sha256": logical_cache_state_sha256(pre),
            "post_logical_state_sha256": logical_cache_state_sha256(post),
        }
        cycles.append(record)
        context.extend(cycle.committed_token_ids)
        previous_post = record["post_logical_state_sha256"]
    generation = trace.generations[0]
    output = list(generation.output_token_ids)
    committed = [token for item in cycles
                 for token in item["transition"]["committed_token_ids"]]
    if committed + list(generation.residual_token_ids) != output:
        raise RuntimeError("M1 reference transition cannot reconstruct full output")
    indices = quartile_cycle_indices(len(cycles))
    if any(selections[index].pool.raw_candidate_count != 18 for index in indices):
        raise RuntimeError("M1 sampled reference cycle missing the complete 18-node pool")
    return {
        "question_id": question_id,
        "max_new_tokens": trace.manifest.max_new_tokens,
        "output_token_count": len(output),
        "output_token_ids_sha256": canonical_json_sha256(output),
        "tail_token_ids": list(generation.residual_token_ids),
        "stopped_on_eos": generation.stopped_on_eos,
        "stop_token_id": generation.stop_token_id,
        "cycle_count": len(cycles),
        "sampled_cycle_indices": list(indices),
        "initial_logical_state": project_logical_cache_state(
            _snapshot(trace, 0, SnapshotPhase.BEFORE_VERIFY)
        ),
        "cycles": cycles,
    }
def project_probe_state(
    *, question_id: int, probe_cycle: int, input_token_ids: tuple[int, ...],
    trace: TraceBundle, selections: tuple[M1Selection, ...],
    reference_cycle: dict[str, object],
) -> dict[str, object]:
    ordered_cycles = _cycles(trace)
    if len(ordered_cycles) != probe_cycle + 1 or len(selections) != probe_cycle + 1:
        raise RuntimeError("M1 probe did not stop after the specified full cycle")
    context = [*input_token_ids]
    for cycle in ordered_cycles[:probe_cycle]:
        context.extend(cycle.committed_token_ids)
    cycle, selection = ordered_cycles[probe_cycle], selections[probe_cycle]
    mapping = (0, *selection.selected_raw_ids)
    tree = selection.pool.to_tree(selection.selected_raw_ids)
    if (mapping != tuple(range(19)) or cycle.iteration != probe_cycle
            or cycle.verification_token_ids != (selection.pool.seed_token_id,
                                                  *tree.token_ids)):
        raise RuntimeError("M1 full-atlas must cover raw ID 0..18")
    pre = _snapshot(trace, probe_cycle, SnapshotPhase.BEFORE_VERIFY)
    frozen = project_frozen_state(pre)
    pool = serialize_m1_pool(selection.pool)
    boundary = (
        cycle.base_kv_length, canonical_json_sha256(context), cycle.pending_seed_token_id,
        selection.committed_offset, pool["fingerprint"],
        frozen["cache_state_fingerprint"],
    )
    expected = (
        reference_cycle["base_kv_length"],
        reference_cycle["context_token_ids_sha256"],
        reference_cycle["pending_seed_token_id"],
        reference_cycle["committed_offset"],
        reference_cycle["pool"]["fingerprint"],
        reference_cycle["pre_logical_state_sha256"],
    )
    if boundary != expected:
        raise RuntimeError("M1 probe does not reproduce the reference selection boundary")
    record = {
        "state_id": f"q{question_id}-c{probe_cycle}",
        "question_id": question_id,
        "cycle_index": probe_cycle,
        "base_kv_length": cycle.base_kv_length,
        "context_token_count": len(context),
        "context_token_ids_sha256": boundary[1],
        "pending_seed_token_id": cycle.pending_seed_token_id,
        "committed_offset": selection.committed_offset,
        "pool": pool,
        "verification_position_to_raw_id": list(mapping),
        "pre_logical_state_sha256": boundary[-1],
        "frozen_state": frozen,
        "full_atlas_routes": _routes(trace, probe_cycle, mapping, pool["fingerprint"]),
    }
    return record
