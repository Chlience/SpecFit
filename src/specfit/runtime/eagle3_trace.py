
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Protocol

import torch
from torch import nn

from specfit.state import ExpertKey
from specfit.trace import (
    CacheSlotTrace,
    CacheSnapshotTrace,
    CacheStatsTrace,
    CycleTrace,
    DraftNodeTrace,
    DraftTreeTrace,
    ExpertProbabilityTrace,
    GenerationTrace,
    LayerCacheSnapshotTrace,
    LayerRouteTrace,
    PendingTransferTrace,
    RuntimeEventTrace,
    SnapshotPhase,
    TargetRouterTopKTrace,
    TraceBundle,
    VerificationNodeRole,
    WorkloadManifestTrace,
    stable_trace_id,
)

from .cache_observer import CacheTransition
from .cache_snapshot import GpuExpertCacheSnapshot
from .eagle3_types import Eagle3DraftTree


class CacheTransitionLog(Protocol):

    run_id: str

    def cursor(self) -> int: ...

    def events(self) -> tuple[CacheTransition, ...]: ...


@dataclass(slots=True)
class _ActiveCycle:
    iteration: int
    base_kv_length: int
    verification_token_ids: torch.Tensor
    draft_tree: Eagle3DraftTree | None
    routes_by_layer: dict[int, _LayerRouteCapture]
    pre_snapshot: CacheSnapshotTrace | None
    transition_start: int | None


@dataclass(slots=True)
class _LayerRouteCapture:
    layer_id: int
    router_logits: torch.Tensor
    routing_weights: torch.Tensor
    selected_experts: torch.Tensor


@dataclass(slots=True)
class _CompletedCycle:
    iteration: int
    base_kv_length: int
    verification_token_ids: torch.Tensor
    draft_tree: Eagle3DraftTree | None
    verifier_token_ids: torch.Tensor
    accepted_draft_tokens: int
    accepted_draft_node_ids: tuple[int, ...]
    committed_token_ids: torch.Tensor
    kv_length_after: int
    layer_routes: tuple[_LayerRouteCapture, ...]
    pre_snapshot: CacheSnapshotTrace | None
    post_snapshot: CacheSnapshotTrace | None
    transition_start: int | None
    transition_end: int | None


@dataclass(slots=True)
class _GenerationCapture:
    output_tokens: torch.Tensor
    target_forward_calls: int
    stopped_on_eos: bool
    stopped_on_cycle_limit: bool
    stop_token_id: int | None


def _token_tuple(value: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(token_id) for token_id in value.cpu().reshape(-1).tolist())


def _trace_expert_id(value: object, *, layer_id: int) -> int:

    if isinstance(value, ExpertKey):
        if value.layer_id != layer_id:
            raise ValueError("expert key belongs to a different layer")
        return value.expert_id
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError("expert ID must be a non-negative integer")
    return value


class GreedyChainTraceRecorder:

    def __init__(
        self,
        manifest: WorkloadManifestTrace,
        *,
        target: nn.Module | None = None,
        cache_snapshot_provider: Callable[
            [], tuple[GpuExpertCacheSnapshot, ...]
        ]
        | None = None,
        transition_log: CacheTransitionLog | None = None,
    ) -> None:
        if manifest.draft_length < 0:
            raise ValueError("trace draft length must be non-negative")
        self.manifest = manifest
        self._target = target
        if transition_log is not None and cache_snapshot_provider is None:
            raise ValueError(
                "transition log requires a cache snapshot provider"
            )
        if (
            transition_log is not None
            and transition_log.run_id != manifest.run_id
        ):
            raise ValueError("transition log run_id differs from manifest")
        self._cache_snapshot_provider = cache_snapshot_provider
        self._transition_log = transition_log
        self._active_cycle: _ActiveCycle | None = None
        self._completed_cycles: list[_CompletedCycle] = []
        self._generation: _GenerationCapture | None = None
        self._finalized: TraceBundle | None = None
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._entered = False
        self._transition_origin: int | None = None
        self._transition_final_cursor: int | None = None
        self._cache_layer_by_id: dict[str, int] = {}
        self._cache_identity_by_layer: dict[int, tuple[str, str]] = {}

    def __enter__(self) -> GreedyChainTraceRecorder:
        if self._entered:
            raise RuntimeError("trace recorder context cannot be re-entered")
        self._entered = True
        if self._transition_log is not None:
            self._transition_origin = self._transition_log.cursor()
        if self._target is None:
            return self
        layers = self._target.model.layers
        if len(layers) != self.manifest.num_moe_layers:
            raise ValueError("target MoE layer count differs from manifest")
        for layer_id, layer in enumerate(layers):
            gate = getattr(getattr(layer, "mlp", None), "gate", None)
            if gate is None:
                self._remove_hooks()
                raise ValueError(f"target layer {layer_id} has no MoE gate")
            self._hook_handles.append(
                gate.register_forward_hook(
                    self._make_gate_hook(layer_id)
                )
            )
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        self.abort_cycle()
        self._remove_hooks()
        self._target = None
        if exc_type is not None or self._generation is None:
            self._discard_runtime_capture()

    def _remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def _discard_runtime_capture(self) -> None:

        self._completed_cycles.clear()
        self._generation = None
        self._cache_snapshot_provider = None
        self._transition_log = None
        self._cache_layer_by_id.clear()
        self._cache_identity_by_layer.clear()

    def _make_gate_hook(self, layer_id: int):
        def capture(
            module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ) -> None:
            del module, inputs
            active = self._active_cycle
            if active is None:
                return
            if layer_id in active.routes_by_layer:
                raise RuntimeError(
                    f"target layer {layer_id} routed twice in one cycle"
                )
            if not isinstance(output, tuple) or len(output) != 3:
                raise RuntimeError("Qwen3 gate returned an unexpected output")
            router_logits, routing_weights, selected_experts = output
            rows = active.verification_token_ids.shape[1]
            if (
                router_logits.shape
                != (rows, self.manifest.num_experts_per_layer)
                or selected_experts.shape
                != (rows, self.manifest.experts_per_token)
                or routing_weights.shape != selected_experts.shape
            ):
                raise RuntimeError(
                    "Qwen3 gate routes do not match verification layout"
                )
            active.routes_by_layer[layer_id] = _LayerRouteCapture(
                layer_id=layer_id,
                router_logits=router_logits.detach(),
                routing_weights=routing_weights.detach(),
                selected_experts=selected_experts.detach(),
            )

        return capture

    def begin_cycle(
        self,
        *,
        iteration: int,
        base_kv_length: int,
        verification_token_ids: torch.Tensor,
        draft_tree: Eagle3DraftTree | None = None,
    ) -> None:
        if self._finalized is not None or self._generation is not None:
            raise RuntimeError("trace recorder has already finished")
        if self._active_cycle is not None:
            raise RuntimeError("a trace cycle is already active")
        if iteration != len(self._completed_cycles):
            raise ValueError("trace cycle iterations must be contiguous")
        if (
            verification_token_ids.ndim != 2
            or verification_token_ids.shape[0] != 1
            or verification_token_ids.shape[1] < 1
        ):
            raise ValueError(
                "verification tokens must have shape [1, length >= 1]"
            )
        if self._target is not None and not self._hook_handles:
            raise RuntimeError("target route capture requires recorder context")
        if self._transition_log is not None and self._transition_origin is None:
            raise RuntimeError(
                "transition capture requires the recorder context"
            )
        if draft_tree is not None:
            expected_drafts = tuple(
                int(value)
                for value in verification_token_ids[0, 1:].tolist()
            )
            if expected_drafts != draft_tree.token_ids:
                raise ValueError(
                    "verification tokens differ from the supplied draft tree"
                )
        tree_id = stable_trace_id(
            "tree",
            self.manifest.run_id,
            self.manifest.request_id,
            iteration,
        )
        cycle_id = stable_trace_id(
            "cycle",
            self.manifest.run_id,
            self.manifest.request_id,
            iteration,
        )
        pre_snapshot = self._capture_cache_snapshot(
            tree_id=tree_id,
            cycle_id=cycle_id,
            phase=SnapshotPhase.BEFORE_VERIFY,
        )
        transition_start = (
            self._transition_log.cursor()
            if self._transition_log is not None
            else None
        )
        self._active_cycle = _ActiveCycle(
            iteration=iteration,
            base_kv_length=base_kv_length,
            verification_token_ids=verification_token_ids.detach(),
            draft_tree=draft_tree,
            routes_by_layer={},
            pre_snapshot=pre_snapshot,
            transition_start=transition_start,
        )

    def end_cycle(
        self,
        *,
        verifier_tokens: torch.Tensor,
        accepted_draft_tokens: int,
        committed_token_ids: torch.Tensor,
        kv_length_after: int,
        accepted_draft_node_ids: tuple[int, ...] | None = None,
    ) -> None:
        active = self._active_cycle
        if active is None:
            raise RuntimeError("no trace cycle is active")
        if verifier_tokens.shape != active.verification_token_ids.shape:
            raise ValueError("verifier tokens must match verification tokens")
        draft_count = active.verification_token_ids.shape[1] - 1
        if not 0 <= accepted_draft_tokens <= draft_count:
            raise ValueError("accepted draft count is outside the chain")
        if accepted_draft_node_ids is None:
            accepted_draft_node_ids = tuple(
                range(1, 1 + accepted_draft_tokens)
            )
        if len(accepted_draft_node_ids) != accepted_draft_tokens:
            raise ValueError("accepted node IDs differ from acceptance count")
        if committed_token_ids.shape != (1, 1 + accepted_draft_tokens):
            raise ValueError("committed tokens do not match acceptance")
        positions = (0,) + accepted_draft_node_ids
        if any(not 1 <= node_id <= draft_count for node_id in positions[1:]):
            raise ValueError("accepted node ID is outside the draft tree")
        expected_committed = active.verification_token_ids[
            :, torch.tensor(
                positions,
                dtype=torch.long,
                device=active.verification_token_ids.device,
            )
        ]
        if not torch.equal(committed_token_ids, expected_committed):
            raise ValueError("committed tokens differ from accepted tree path")
        if active.draft_tree is not None:
            full_path = active.draft_tree.accepted_path(verifier_tokens)
            if full_path[: len(accepted_draft_node_ids)] != (
                accepted_draft_node_ids
            ):
                raise ValueError("accepted nodes do not follow target verification")
            for parent_id, node_id in zip(
                (0,) + accepted_draft_node_ids,
                accepted_draft_node_ids,
            ):
                if active.draft_tree.parent_ids[node_id - 1] != parent_id:
                    raise ValueError("accepted draft nodes do not form one path")
        if self._target is not None and set(active.routes_by_layer) != set(
            range(self.manifest.num_moe_layers)
        ):
            raise RuntimeError("verification did not capture every MoE layer")
        tree_id = stable_trace_id(
            "tree",
            self.manifest.run_id,
            self.manifest.request_id,
            active.iteration,
        )
        cycle_id = stable_trace_id(
            "cycle",
            self.manifest.run_id,
            self.manifest.request_id,
            active.iteration,
        )
        post_snapshot = self._capture_cache_snapshot(
            tree_id=tree_id,
            cycle_id=cycle_id,
            phase=SnapshotPhase.AFTER_VERIFY,
        )
        transition_end = (
            self._transition_log.cursor()
            if self._transition_log is not None
            else None
        )
        self._completed_cycles.append(
            _CompletedCycle(
                iteration=active.iteration,
                base_kv_length=active.base_kv_length,
                verification_token_ids=active.verification_token_ids,
                draft_tree=active.draft_tree,
                verifier_token_ids=verifier_tokens.detach(),
                accepted_draft_tokens=accepted_draft_tokens,
                accepted_draft_node_ids=accepted_draft_node_ids,
                committed_token_ids=committed_token_ids.detach(),
                kv_length_after=kv_length_after,
                layer_routes=tuple(
                    active.routes_by_layer[layer_id]
                    for layer_id in sorted(active.routes_by_layer)
                ),
                pre_snapshot=active.pre_snapshot,
                post_snapshot=post_snapshot,
                transition_start=active.transition_start,
                transition_end=transition_end,
            )
        )
        self._active_cycle = None

    def abort_cycle(self) -> None:
        self._active_cycle = None

    def finish_generation(
        self,
        *,
        output_tokens: torch.Tensor,
        target_forward_calls: int,
        stopped_on_eos: bool = False,
        stopped_on_cycle_limit: bool = False,
        stop_token_id: int | None = None,
    ) -> None:
        if self._active_cycle is not None:
            raise RuntimeError("cannot finish while a trace cycle is active")
        if self._generation is not None or self._finalized is not None:
            raise RuntimeError("trace recorder has already finished")
        if output_tokens.ndim != 2 or output_tokens.shape[0] != 1:
            raise ValueError("output tokens must have shape [1, length]")
        self._generation = _GenerationCapture(
            output_tokens=output_tokens.detach(),
            target_forward_calls=target_forward_calls,
            stopped_on_eos=stopped_on_eos,
            stopped_on_cycle_limit=stopped_on_cycle_limit,
            stop_token_id=stop_token_id,
        )
        if self._transition_log is not None:
            self._transition_final_cursor = self._transition_log.cursor()

    def _capture_cache_snapshot(
        self,
        *,
        tree_id: str,
        cycle_id: str,
        phase: SnapshotPhase,
    ) -> CacheSnapshotTrace | None:
        provider = self._cache_snapshot_provider
        if provider is None:
            return None
        live_layers = provider()
        if len(live_layers) != self.manifest.num_moe_layers:
            raise ValueError(
                "cache snapshot layer count differs from manifest"
            )
        layers = tuple(
            self._serialize_cache_layer(layer_id, live)
            for layer_id, live in enumerate(live_layers)
        )
        return CacheSnapshotTrace(
            run_id=self.manifest.run_id,
            tree_id=tree_id,
            cycle_id=cycle_id,
            phase=phase,
            runtime_state_version=sum(
                layer.host_state_version for layer in live_layers
            ),
            captured_at_ns=perf_counter_ns(),
            layers=layers,
        )

    def _serialize_cache_layer(
        self,
        layer_id: int,
        live: GpuExpertCacheSnapshot,
    ) -> LayerCacheSnapshotTrace:
        identity = (live.cache_id, live.queue_id)
        previous = self._cache_identity_by_layer.setdefault(
            layer_id, identity
        )
        if previous != identity:
            raise ValueError("cache identity changes across snapshots")
        previous_layer = self._cache_layer_by_id.setdefault(
            live.cache_id, layer_id
        )
        if previous_layer != layer_id:
            raise ValueError("one cache ID maps to multiple layers")
        return LayerCacheSnapshotTrace(
            layer_id=layer_id,
            cache_id=live.cache_id,
            queue_id=live.queue_id,
            capacity=live.capacity,
            host_state_version=live.host_state_version,
            lru_sequence=live.lru_sequence,
            submission_sequence=live.submission_sequence,
            slots=tuple(
                CacheSlotTrace(
                    slot_id=slot.slot_id,
                    expert_id=(
                        _trace_expert_id(slot.expert_id, layer_id=layer_id)
                        if slot.expert_id is not None
                        else None
                    ),
                    generation=slot.generation,
                    active_leases=slot.active_leases,
                    last_used_sequence=slot.last_used_sequence,
                    ready=slot.ready,
                    pending_use_guards=slot.pending_use_guards,
                )
                for slot in live.slots
            ),
            stats=CacheStatsTrace(
                h2d_submissions=live.stats.h2d_submissions,
                resident_hits=live.stats.resident_hits,
                inflight_joins=live.stats.inflight_joins,
                evictions=live.stats.evictions,
                pending_use_guards=live.stats.pending_use_guards,
            ),
            pending_transfers=tuple(
                PendingTransferTrace(
                    transfer_id=stable_trace_id(
                        "h2d",
                        self.manifest.run_id,
                        layer_id,
                        transfer.submission_sequence,
                    ),
                    expert_id=_trace_expert_id(
                        transfer.expert_id, layer_id=layer_id
                    ),
                    slot_id=transfer.slot_id,
                    generation=transfer.generation,
                    submission_sequence=transfer.submission_sequence,
                    ready_event_id=None,
                    ready_event_completed=(
                        transfer.ready_event_completed
                    ),
                )
                for transfer in live.pending_transfers
            ),
        )

    def finalize(self) -> TraceBundle:

        if self._finalized is not None:
            return self._finalized
        if self._active_cycle is not None:
            raise RuntimeError("cannot finalize while a trace cycle is active")
        if self._generation is None:
            raise RuntimeError("generation has not finished")

        materialized_topk: dict[
            tuple[int, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        for cycle in self._completed_cycles:
            for route in cycle.layer_routes:
                probabilities = torch.softmax(
                    route.router_logits,
                    dtype=torch.float32,
                    dim=-1,
                )
                values, indices = torch.topk(
                    probabilities,
                    k=min(32, probabilities.shape[-1]),
                    dim=-1,
                    sorted=True,
                )
                materialized_topk[(cycle.iteration, route.layer_id)] = (
                    values,
                    indices,
                )

        devices = {
            tensor.device
            for cycle in self._completed_cycles
            for tensor in (
                cycle.verification_token_ids,
                cycle.verifier_token_ids,
                cycle.committed_token_ids,
            )
            if tensor.device.type == "cuda"
        }
        devices.update(
            tensor.device
            for cycle in self._completed_cycles
            for route in cycle.layer_routes
            for tensor in (
                route.router_logits,
                route.routing_weights,
                route.selected_experts,
            )
            if tensor.device.type == "cuda"
        )
        if self._generation.output_tokens.device.type == "cuda":
            devices.add(self._generation.output_tokens.device)
        for device in devices:
            torch.cuda.synchronize(device)

        trees = []
        cycles = []
        target_routes = []
        target_router_topk = []
        committed_prefix: list[int] = []
        for captured in self._completed_cycles:
            verification_ids = _token_tuple(
                captured.verification_token_ids
            )
            verifier_ids = _token_tuple(captured.verifier_token_ids)
            committed_ids = _token_tuple(captured.committed_token_ids)
            committed_prefix.extend(committed_ids)
            tree_id = stable_trace_id(
                "tree",
                self.manifest.run_id,
                self.manifest.request_id,
                captured.iteration,
            )
            accepted_node_ids = {0, *captured.accepted_draft_node_ids}
            if captured.draft_tree is None:
                parent_ids = tuple(
                    position - 1 if position else None
                    for position in range(len(verification_ids))
                )
                depths = tuple(range(len(verification_ids)))
                probabilities: tuple[float | None, ...] = (None,) * len(
                    verification_ids
                )
            else:
                parent_ids = (None,) + captured.draft_tree.parent_ids
                depths = (0,) + captured.draft_tree.depths
                probabilities = (None,) + (
                    captured.draft_tree.conditional_probabilities
                )
            nodes = tuple(
                DraftNodeTrace(
                    tree_id=tree_id,
                    node_id=position,
                    parent_id=parent_ids[position],
                    depth=depths[position],
                    role=(
                        VerificationNodeRole.SEED
                        if position == 0
                        else VerificationNodeRole.DRAFT
                    ),
                    token_id=token_id,
                    conditional_probability=probabilities[position],
                    verification_position=position,
                    selected=True,
                    verified=True,
                    accepted=position in accepted_node_ids,
                )
                for position, token_id in enumerate(verification_ids)
            )
            tree = DraftTreeTrace(
                tree_id=tree_id,
                request_id=self.manifest.request_id,
                iteration=captured.iteration,
                nodes=nodes,
            )
            cycle = CycleTrace(
                run_id=self.manifest.run_id,
                tree_id=tree_id,
                request_id=self.manifest.request_id,
                iteration=captured.iteration,
                base_kv_length=captured.base_kv_length,
                pending_seed_token_id=verification_ids[0],
                verification_token_ids=verification_ids,
                verifier_token_ids=verifier_ids,
                accepted_draft_node_ids=captured.accepted_draft_node_ids,
                committed_token_ids=committed_ids,
                kv_length_after=captured.kv_length_after,
                pre_snapshot_id=(
                    captured.pre_snapshot.trace_id
                    if captured.pre_snapshot is not None
                    else None
                ),
                post_snapshot_id=(
                    captured.post_snapshot.trace_id
                    if captured.post_snapshot is not None
                    else None
                ),
            )
            trees.append(tree)
            cycles.append(cycle)
            for route in captured.layer_routes:
                topk_values, topk_indices = materialized_topk[
                    (captured.iteration, route.layer_id)
                ]
                selected = route.selected_experts.cpu()
                weights = route.routing_weights.cpu()
                topk_values = topk_values.cpu()
                topk_indices = topk_indices.cpu()
                for node_id in range(len(verification_ids)):
                    target_routes.append(
                        LayerRouteTrace(
                            tree_id=tree_id,
                            node_id=node_id,
                            layer_id=route.layer_id,
                            expert_ids=tuple(
                                int(value) for value in selected[node_id]
                            ),
                            expert_weights=tuple(
                                float(value) for value in weights[node_id]
                            ),
                        )
                    )
                    target_router_topk.append(
                        TargetRouterTopKTrace(
                            tree_id=tree_id,
                            node_id=node_id,
                            layer_id=route.layer_id,
                            experts=tuple(
                                ExpertProbabilityTrace(
                                    expert_id=int(expert_id),
                                    probability=float(probability),
                                )
                                for expert_id, probability in zip(
                                    topk_indices[node_id],
                                    topk_values[node_id],
                                )
                            ),
                        )
                    )

        output_ids = _token_tuple(self._generation.output_tokens)
        residual_ids = output_ids[len(committed_prefix) :]
        generation = GenerationTrace(
            run_id=self.manifest.run_id,
            request_id=self.manifest.request_id,
            output_token_ids=output_ids,
            cycle_ids=tuple(cycle.trace_id for cycle in cycles),
            residual_token_ids=residual_ids,
            target_forward_calls=self._generation.target_forward_calls,
            stopped_on_eos=self._generation.stopped_on_eos,
            stopped_on_cycle_limit=(
                self._generation.stopped_on_cycle_limit
            ),
            stop_token_id=self._generation.stop_token_id,
        )
        snapshots = tuple(
            snapshot
            for cycle in self._completed_cycles
            for snapshot in (cycle.pre_snapshot, cycle.post_snapshot)
            if snapshot is not None
        )
        runtime_events = self._serialize_runtime_events()
        bundle = TraceBundle(
            manifest=self.manifest,
            trees=tuple(trees),
            cycles=tuple(cycles),
            generations=(generation,),
            target_routes=tuple(target_routes),
            target_router_topk=tuple(target_router_topk),
            snapshots=snapshots,
            runtime_events=runtime_events,
        )
        self._completed_cycles.clear()
        self._generation = None
        self._cache_snapshot_provider = None
        self._transition_log = None
        self._cache_layer_by_id.clear()
        self._cache_identity_by_layer.clear()
        self._finalized = bundle
        return bundle

    def _serialize_runtime_events(self) -> tuple[RuntimeEventTrace, ...]:
        log = self._transition_log
        if log is None:
            return ()
        if (
            self._transition_origin is None
            or self._transition_final_cursor is None
        ):
            raise RuntimeError("transition capture has no complete boundary")
        origin = self._transition_origin
        final = self._transition_final_cursor
        events = tuple(
            event
            for event in log.events()
            if origin <= event.sequence < final
        )
        if tuple(event.sequence for event in events) != tuple(
            range(origin, final)
        ):
            raise RuntimeError("transition log changed before finalization")

        scopes = []
        for cycle in self._completed_cycles:
            if cycle.transition_start is None or cycle.transition_end is None:
                raise RuntimeError("cycle transition boundary is incomplete")
            if cycle.transition_start > cycle.transition_end:
                raise RuntimeError("cycle transition boundary is reversed")
            scopes.append(
                (
                    cycle.transition_start,
                    cycle.transition_end,
                    stable_trace_id(
                        "tree",
                        self.manifest.run_id,
                        self.manifest.request_id,
                        cycle.iteration,
                    ),
                    stable_trace_id(
                        "cycle",
                        self.manifest.run_id,
                        self.manifest.request_id,
                        cycle.iteration,
                    ),
                )
            )

        result = []
        for event in events:
            if event.run_id != self.manifest.run_id:
                raise ValueError("transition belongs to a different run")
            layer_id = self._cache_layer_by_id.get(event.cache_id)
            if layer_id is None:
                if not isinstance(event.expert_id, ExpertKey):
                    raise ValueError("transition references an unknown cache")
                layer_id = event.expert_id.layer_id
                if layer_id >= self.manifest.num_moe_layers:
                    raise ValueError("transition layer exceeds manifest")
                identity = (event.cache_id, event.queue_id)
                previous_identity = self._cache_identity_by_layer.setdefault(
                    layer_id, identity
                )
                if previous_identity != identity:
                    raise ValueError(
                        "transition cache identity changes within a layer"
                    )
                previous_layer = self._cache_layer_by_id.setdefault(
                    event.cache_id, layer_id
                )
                if previous_layer != layer_id:
                    raise ValueError("one cache ID maps to multiple layers")
            cache_id, queue_id = self._cache_identity_by_layer[layer_id]
            if event.cache_id != cache_id or event.queue_id != queue_id:
                raise ValueError("transition cache identity does not match")
            tree_id = None
            cycle_id = None
            for start, end, candidate_tree, candidate_cycle in scopes:
                if start <= event.sequence < end:
                    if cycle_id is not None:
                        raise RuntimeError(
                            "transition belongs to overlapping cycles"
                        )
                    tree_id = candidate_tree
                    cycle_id = candidate_cycle
            submission_sequence = event.submission_sequence
            transfer_id = (
                stable_trace_id(
                    "h2d",
                    self.manifest.run_id,
                    layer_id,
                    submission_sequence,
                )
                if submission_sequence is not None
                else None
            )
            result.append(
                RuntimeEventTrace(
                    run_id=event.run_id,
                    sequence=event.sequence,
                    kind=event.kind,
                    host_time_ns=event.host_time_ns,
                    layer_id=layer_id,
                    cache_id=cache_id,
                    queue_id=queue_id,
                    expert_id=_trace_expert_id(
                        event.expert_id, layer_id=layer_id
                    ),
                    slot_id=event.slot_id,
                    generation=event.generation,
                    submission_sequence=submission_sequence,
                    tree_id=tree_id,
                    cycle_id=cycle_id,
                    transfer_id=transfer_id,
                    victim_expert_id=(
                        _trace_expert_id(
                            event.victim_expert_id,
                            layer_id=layer_id,
                        )
                        if event.victim_expert_id is not None
                        else None
                    ),
                    stream_ids=event.stream_ids,
                )
            )
        return tuple(result)
