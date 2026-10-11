from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal
import torch

from .replay import replay_qwen3_chain
from .r6a_experiment import compute_r6a_route_digest
from .r6a_frozen_cycle import (
    FrozenRouteTable,
    cache_snapshot_state_fingerprint,
    live_cache_snapshots_state_fingerprint,
    resident_experts_from_snapshot,
    route_table_from_trace,
)
from .runtime import (
    CumulativeProbabilityTreeSelector,
    Eagle3ChainDrafter,
    Eagle3DraftCandidatePool,
    Eagle3TreeDrafter,
    ExplicitRawIdTreeSelector,
    GreedyChainTraceRecorder,
    LoadedOffloadedQwen3,
    RecordingCacheTransitionObserver,
    ShallowTreeSpeculator,
    candidate_pool_fingerprint,
)
from .state import ExpertKey
from .trace import CacheSnapshotTrace, SnapshotPhase, TraceBundle, WorkloadManifestTrace
_COUNTERS = ("h2d_submissions", "resident_hits", "inflight_joins", "evictions")
SelectionMode = Literal["all", "probability", "fixed"]
@dataclass(frozen=True, slots=True)
class FrozenCycleRun:
    record: dict[str, object]
    pool: Eagle3DraftCandidatePool
    routes: FrozenRouteTable
    initial_resident_experts: frozenset[ExpertKey]
    frozen_state: dict[str, object]
    trace: TraceBundle
class _CapturingPolicy:
    def __init__(
        self,
        mode: SelectionMode,
        selected_raw_ids: tuple[int, ...] = (),
        expected_pool_fingerprint: str | None = None,
    ) -> None:
        self.mode = mode
        self.selected_raw_ids = selected_raw_ids
        self.expected_pool_fingerprint = expected_pool_fingerprint
        self.pool: Eagle3DraftCandidatePool | None = None
    def select_raw_ids(
        self,
        pool: Eagle3DraftCandidatePool,
        *,
        draft_node_budget: int,
        committed_offset: int = 0,
    ) -> tuple[int, ...]:
        del committed_offset
        fingerprint = candidate_pool_fingerprint(pool)
        if self.expected_pool_fingerprint not in (None, fingerprint):
            raise RuntimeError("This candidate pool does not match the frozen atlas fingerprint")
        self.pool = pool
        if self.mode == "all":
            selected = tuple(node.raw_id for node in pool.candidates)
        elif self.mode == "probability":
            selected = CumulativeProbabilityTreeSelector().select_raw_ids(
                pool, draft_node_budget=draft_node_budget
            )
        elif self.mode == "fixed":
            selected = self.selected_raw_ids
        else:
            raise ValueError(f"unsupported selection mode: {self.mode}")
        if len(selected) != draft_node_budget:
            raise RuntimeError("selector did not fill this fixed node budget")
        return selected
def serialize_candidate_pool(pool: Eagle3DraftCandidatePool) -> dict[str, object]:
    return {
        "seed_token_id": pool.seed_token_id,
        "top_k": pool.top_k,
        "max_depth": pool.max_depth,
        "raw_candidate_count": pool.raw_candidate_count,
        "fingerprint": candidate_pool_fingerprint(pool),
        "candidates": [asdict(node) for node in pool.candidates],
    }
def selection_mapping(selected_raw_ids: tuple[int, ...]) -> dict[str, object]:
    positions = (0, *selected_raw_ids)
    return {
        "selected_raw_ids": list(selected_raw_ids),
        "verification_position_to_raw_id": list(positions),
        "raw_to_verification_position": {
            str(raw_id): position for position, raw_id in enumerate(positions)
        },
    }
def serialize_routes(
    routes: FrozenRouteTable,
    verification_raw_ids: tuple[int, ...],
) -> dict[str, object]:
    rows = [
        {
            "verification_position": position,
            "raw_id": raw_id,
            "layer_id": layer_id,
            "expert_ids": list(experts),
        }
        for position, raw_id in enumerate(verification_raw_ids)
        for layer_id, experts in enumerate(routes.for_raw_id(raw_id).experts_by_layer)
    ]
    return {
        "pool_fingerprint": routes.pool_fingerprint,
        "coverage_rows": len(rows),
        "digest": compute_r6a_route_digest(
            pool_fingerprint=routes.pool_fingerprint,
            verification_position_to_raw_id=verification_raw_ids,
            rows=rows,
        ),
        "rows": rows,
    }
def _snapshot(trace: TraceBundle, phase: SnapshotPhase) -> CacheSnapshotTrace:
    matches = tuple(item for item in trace.snapshots if item.phase is phase)
    if len(matches) != 1:
        raise RuntimeError(f"single-round trace should contain one {phase.value} snapshot")
    return matches[0]
def _counter_delta(pre: CacheSnapshotTrace, post: CacheSnapshotTrace) -> dict[str, int]:
    def totals(snapshot: CacheSnapshotTrace) -> dict[str, int]:
        return {
            field: sum(int(getattr(layer.stats, field)) for layer in snapshot.layers)
            for field in _COUNTERS
        }
    before, after = totals(pre), totals(post)
    delta = {field: after[field] - before[field] for field in _COUNTERS}
    if any(value < 0 for value in delta.values()):
        raise RuntimeError("verification cache counter has a negative increment")
    return delta
def frozen_state_record(snapshot: CacheSnapshotTrace, fingerprint: str) -> dict[str, object]:
    return {
        "cache_state_fingerprint": fingerprint,
        "quiescent": True,
        "resident_experts_by_layer": [
            {
                "layer_id": layer_id,
                "expert_ids": sorted(
                    slot.expert_id
                    for slot in layer.slots
                    if slot.expert_id is not None and slot.ready
                ),
            }
            for layer_id, layer in enumerate(snapshot.layers)
        ],
    }
def _require_quiescent(snapshots: tuple[Any, ...]) -> None:
    if any(
        layer.pending_transfers
        or any(slot.active_leases or slot.pending_use_guards for slot in layer.slots)
        for layer in snapshots
    ):
        raise RuntimeError("prefill frozen boundary still contains transfer, lease, or use guard")
def _require_empty_cache(snapshots: tuple[Any, ...]) -> None:
    counters = _COUNTERS + ("pending_use_guards",)
    invalid = any(
        layer.pending_transfers
        or any(slot.expert_id is not None for slot in layer.slots)
        or any(int(getattr(layer.stats, field)) for field in counters)
        for layer in snapshots
    )
    if invalid:
        raise RuntimeError("cold reset did not leave expert cache empty")
def unique_route_acquisition_count(
    routes: FrozenRouteTable, verification_raw_ids: tuple[int, ...]
) -> int:

    return len({
        (layer_id, expert_id)
        for raw_id in verification_raw_ids
        for layer_id, experts in enumerate(routes.for_raw_id(raw_id).experts_by_layer)
        for expert_id in experts
    })
def run_frozen_cycle(
    *,
    loaded: LoadedOffloadedQwen3,
    drafter: Eagle3ChainDrafter,
    observer: RecordingCacheTransitionObserver | None,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    manifest: WorkloadManifestTrace,
    selection_mode: SelectionMode,
    selected_raw_ids: tuple[int, ...] = (),
    expected_pool_fingerprint: str | None = None,
    execution_index: int = 0,
    prepare_verification_cache: Callable[[], None] | None = None,
) -> FrozenCycleRun:
    loaded.cold_reset_expert_caches()
    _require_empty_cache(loaded.cache_snapshots())
    if observer is not None:
        observer.reset(run_id=manifest.run_id)
    policy = _CapturingPolicy(
        selection_mode, selected_raw_ids, expected_pool_fingerprint
    )
    selector = ExplicitRawIdTreeSelector(policy)
    recorder = GreedyChainTraceRecorder(
        manifest,
        target=loaded.model,
        cache_snapshot_provider=loaded.cache_snapshots,
        transition_log=observer,
    )
    frozen_fingerprint: str | None = None
    def on_prefill_complete() -> None:
        nonlocal frozen_fingerprint
        torch.cuda.synchronize(input_ids.device)
        snapshots = loaded.cache_snapshots()
        _require_quiescent(snapshots)
        if prepare_verification_cache is not None:
            prepare_verification_cache()
            torch.cuda.synchronize(input_ids.device)
            snapshots = loaded.cache_snapshots()
            _require_quiescent(snapshots)
        frozen_fingerprint = live_cache_snapshots_state_fingerprint(snapshots)
    speculator = ShallowTreeSpeculator(loaded.model, Eagle3TreeDrafter(drafter))
    budget = 18 if selection_mode == "all" else 7
    with recorder, torch.inference_mode():
        result = speculator.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=7,
            top_k=2,
            max_depth=5,
            draft_node_budget=budget,
            trace=recorder,
            on_prefill_complete=on_prefill_complete,
            stop_on_eos=False,
            selector=selector,
            collect_cycle_metrics=True,
            max_cycles=1,
        )
    torch.cuda.synchronize(input_ids.device)
    trace = recorder.finalize()
    if observer is not None:
        replay_qwen3_chain(trace)
    pool = policy.pool
    if pool is None or result.cycles != 1 or result.target_forward_calls != 2:
        raise RuntimeError("frozen experiment must execute one cycle and two target forwards")
    if len(result.tree_cycle_metrics) != 1 or len(trace.cycles) != 1:
        raise RuntimeError("Frozen experiment did not produce exactly one candidate pool and cycle")
    metric, cycle = result.tree_cycle_metrics[0], trace.cycles[0]
    verification_raw_ids = metric.verification_raw_ids
    normalized = (0, *pool.to_tree(verification_raw_ids[1:]).source_raw_ids)
    if verification_raw_ids != normalized:
        raise RuntimeError("runtime verification raw-ID mapping malformed")
    pre = _snapshot(trace, SnapshotPhase.BEFORE_VERIFY)
    post = _snapshot(trace, SnapshotPhase.AFTER_VERIFY)
    pre_fingerprint = cache_snapshot_state_fingerprint(pre)
    if frozen_fingerprint != pre_fingerprint:
        raise RuntimeError("prefill callback does not match verification pre-state")
    routes = route_table_from_trace(
        trace, pool=pool, verification_position_to_raw_id=verification_raw_ids
    )
    accepted_positions = cycle.accepted_draft_node_ids
    last_position = accepted_positions[-1] if accepted_positions else 0
    next_pending_seed = cycle.verifier_token_ids[last_position]
    if metric.next_pending_seed_token_id != next_pending_seed:
        raise RuntimeError("cycle metric does not match trace's next pending seed")
    cuda_ns = metric.target_verification_cuda_ns
    if cuda_ns is None:
        raise RuntimeError("CUDA verification timing is missing")
    counters = _counter_delta(pre, post)
    unique_acquisitions = unique_route_acquisition_count(routes, verification_raw_ids)
    acquisition_events = sum(counters[field] for field in _COUNTERS[:3])
    if acquisition_events != unique_acquisitions:
        raise RuntimeError("own-route unique acquisitions do not close with H2D+hit+join")
    config = loaded.model.config
    expert_bytes = int(3 * config.hidden_size * config.moe_intermediate_size
                       * torch.bfloat16.itemsize)
    record: dict[str, object] = {
        "execution_index": execution_index,
        **selection_mapping(verification_raw_ids[1:]),
        "routes": serialize_routes(routes, verification_raw_ids),
        "cache_state": {
            "pre_fingerprint": pre_fingerprint,
            "post_fingerprint": cache_snapshot_state_fingerprint(post),
        },
        "timings": {
            "proposal_wall_ns": metric.proposal_wall_ns,
            "selector_wall_ns": metric.selector_wall_ns,
            "target_verification_wall_ns": metric.target_verification_wall_ns,
            "cycle_wall_ns": metric.cycle_wall_ns,
            "target_verification_cuda_ns": cuda_ns,
        },
        "cache_counter_delta": {
            **counters,
            "unique_expert_acquisitions": unique_acquisitions,
            "expert_bytes": expert_bytes,
            "logical_h2d_bytes": counters["h2d_submissions"] * expert_bytes,
        },
        "acceptance": {
            "accepted_draft_tokens": metric.accepted_draft_nodes,
            "accepted_raw_ids": list(metric.accepted_raw_ids),
            "committed_token_ids": list(cycle.committed_token_ids),
            "next_pending_seed_token_id": next_pending_seed,
        },
    }
    return FrozenCycleRun(
        record,
        pool,
        routes,
        resident_experts_from_snapshot(pre),
        frozen_state_record(pre, pre_fingerprint),
        trace,
    )
