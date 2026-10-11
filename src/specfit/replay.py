
from __future__ import annotations

from dataclasses import dataclass

from .trace import (
    CacheSnapshotTrace,
    LayerCacheSnapshotTrace,
    RuntimeEventKind,
    RuntimeEventTrace,
    TraceBundle,
)


class LogicalReplayError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ReplayCounters:

    h2d_submissions: int = 0
    resident_hits: int = 0
    inflight_joins: int = 0
    evictions: int = 0
    consume_waits: int = 0
    releases: int = 0

    @property
    def acquisitions(self) -> int:
        return (
            self.h2d_submissions
            + self.resident_hits
            + self.inflight_joins
        )

    @property
    def event_count(self) -> int:
        return (
            self.h2d_submissions
            + self.resident_hits
            + self.inflight_joins
            + self.evictions
            + self.consume_waits
            + self.releases
        )

    def __add__(self, other: ReplayCounters) -> ReplayCounters:
        if not isinstance(other, ReplayCounters):
            return NotImplemented
        return ReplayCounters(
            h2d_submissions=(
                self.h2d_submissions + other.h2d_submissions
            ),
            resident_hits=self.resident_hits + other.resident_hits,
            inflight_joins=self.inflight_joins + other.inflight_joins,
            evictions=self.evictions + other.evictions,
            consume_waits=self.consume_waits + other.consume_waits,
            releases=self.releases + other.releases,
        )


@dataclass(frozen=True, slots=True)
class LogicalReplayEvent:

    kind: RuntimeEventKind
    layer_id: int
    expert_id: int
    slot_id: int
    generation: int
    submission_sequence: int | None = None
    victim_expert_id: int | None = None


@dataclass(frozen=True, slots=True)
class ReplaySlotState:

    layer_id: int
    slot_id: int
    expert_id: int | None
    generation: int
    last_used_sequence: int


@dataclass(frozen=True, slots=True)
class CycleReplayResult:

    iteration: int
    tree_id: str
    cycle_id: str
    accessed_experts_by_layer: tuple[tuple[int, ...], ...]
    counters: ReplayCounters
    events: tuple[LogicalReplayEvent, ...]
    occupancy_before: int
    occupancy_after: int


@dataclass(frozen=True, slots=True)
class LogicalReplayResult:

    run_id: str
    cycles: tuple[CycleReplayResult, ...]
    counters: ReplayCounters
    initial_occupancy: int
    final_occupancy: int
    final_slots: tuple[ReplaySlotState, ...]


@dataclass(frozen=True, slots=True)
class ExpertAccessReplayResult:

    accessed_experts_by_layer: tuple[tuple[int, ...], ...]
    counters: ReplayCounters
    events: tuple[LogicalReplayEvent, ...]
    initial_occupancy: int
    final_occupancy: int
    final_slots: tuple[ReplaySlotState, ...]
    final_submission_sequences: tuple[int, ...]
    final_lru_sequences: tuple[int, ...]
    initial_resident_reload_h2d: int


@dataclass(slots=True)
class _ReplaySlot:
    slot_id: int
    expert_id: int | None
    generation: int
    active_leases: int
    last_used_sequence: int


@dataclass(slots=True)
class _MutableCounters:
    h2d_submissions: int
    resident_hits: int
    inflight_joins: int
    evictions: int
    pending_use_guards: int

    def freeze(
        self,
        *,
        consume_waits: int = 0,
        releases: int = 0,
    ) -> ReplayCounters:
        return ReplayCounters(
            h2d_submissions=self.h2d_submissions,
            resident_hits=self.resident_hits,
            inflight_joins=self.inflight_joins,
            evictions=self.evictions,
            consume_waits=consume_waits,
            releases=releases,
        )


@dataclass(slots=True)
class _LayerState:
    layer_id: int
    cache_id: str
    queue_id: str
    capacity: int
    submission_sequence: int
    lru_sequence: int
    stats: _MutableCounters
    slots: list[_ReplaySlot]

    @classmethod
    def from_snapshot(
        cls,
        snapshot: LayerCacheSnapshotTrace,
        *,
        location: str,
    ) -> _LayerState:
        _require_quiescent_layer(snapshot, location=location)
        if (
            snapshot.cache_id is None
            or snapshot.lru_sequence is None
            or snapshot.stats is None
        ):
            raise LogicalReplayError(
                f"{location}: missing R3.4b extended cache state"
            )
        return cls(
            layer_id=snapshot.layer_id,
            cache_id=snapshot.cache_id,
            queue_id=snapshot.queue_id,
            capacity=snapshot.capacity,
            submission_sequence=snapshot.submission_sequence,
            lru_sequence=snapshot.lru_sequence,
            stats=_MutableCounters(
                h2d_submissions=snapshot.stats.h2d_submissions,
                resident_hits=snapshot.stats.resident_hits,
                inflight_joins=snapshot.stats.inflight_joins,
                evictions=snapshot.stats.evictions,
                pending_use_guards=snapshot.stats.pending_use_guards,
            ),
            slots=[
                _ReplaySlot(
                    slot_id=slot.slot_id,
                    expert_id=slot.expert_id,
                    generation=slot.generation,
                    active_leases=slot.active_leases,
                    last_used_sequence=slot.last_used_sequence,
                )
                for slot in snapshot.slots
            ],
        )

    @property
    def occupancy(self) -> int:
        return sum(slot.expert_id is not None for slot in self.slots)

    def _touch(self, slot: _ReplaySlot) -> None:
        self.lru_sequence += 1
        slot.last_used_sequence = self.lru_sequence

    def _choose_lru_slot(self) -> _ReplaySlot:
        available = [slot for slot in self.slots if slot.active_leases == 0]
        if not available:
            raise LogicalReplayError(
                f"layer {self.layer_id}: no replaceable expert slot"
            )
        return min(
            available,
            key=lambda slot: (
                slot.expert_id is not None,
                slot.last_used_sequence,
                slot.slot_id,
            ),
        )

    def access(self, expert_id: int) -> tuple[LogicalReplayEvent, ...]:
        resident = next(
            (slot for slot in self.slots if slot.expert_id == expert_id),
            None,
        )
        events: list[LogicalReplayEvent] = []
        if resident is not None:
            slot = resident
            self.stats.resident_hits += 1
            events.append(
                self._event(RuntimeEventKind.RESIDENT_HIT, expert_id, slot)
            )
        else:
            slot = self._choose_lru_slot()
            victim = slot.expert_id
            slot.generation += 1
            slot.expert_id = expert_id
            if victim is not None:
                self.stats.evictions += 1
                events.append(
                    self._event(
                        RuntimeEventKind.EVICT,
                        expert_id,
                        slot,
                        victim_expert_id=victim,
                    )
                )
            submission_sequence = self.submission_sequence
            self.submission_sequence += 1
            self.stats.h2d_submissions += 1
            events.append(
                self._event(
                    RuntimeEventKind.H2D_SUBMIT,
                    expert_id,
                    slot,
                    submission_sequence=submission_sequence,
                )
            )

        slot.active_leases += 1
        self._touch(slot)
        events.append(
            self._event(RuntimeEventKind.CONSUME_WAIT, expert_id, slot)
        )
        events.append(self._event(RuntimeEventKind.RELEASE, expert_id, slot))
        slot.active_leases -= 1
        self._touch(slot)
        return tuple(events)

    def _event(
        self,
        kind: RuntimeEventKind,
        expert_id: int,
        slot: _ReplaySlot,
        *,
        submission_sequence: int | None = None,
        victim_expert_id: int | None = None,
    ) -> LogicalReplayEvent:
        return LogicalReplayEvent(
            kind=kind,
            layer_id=self.layer_id,
            expert_id=expert_id,
            slot_id=slot.slot_id,
            generation=slot.generation,
            submission_sequence=submission_sequence,
            victim_expert_id=victim_expert_id,
        )


def replay_qwen3_accesses(
    before_snapshot: CacheSnapshotTrace,
    accessed_experts_by_layer: tuple[tuple[int, ...], ...],
) -> ExpertAccessReplayResult:

    return replay_qwen3_access_cycles(
        before_snapshot,
        (accessed_experts_by_layer,),
    )[0]


def replay_qwen3_access_cycles(
    before_snapshot: CacheSnapshotTrace,
    accessed_experts_by_cycle: tuple[
        tuple[tuple[int, ...], ...], ...
    ],
    *,
    resident_first: bool = False,
) -> tuple[ExpertAccessReplayResult, ...]:

    if not isinstance(resident_first, bool):
        raise LogicalReplayError("resident_first must be bool")
    if not accessed_experts_by_cycle:
        raise LogicalReplayError("counterfactual cycle sequence must not be empty")
    expected_layers = tuple(range(len(before_snapshot.layers)))
    observed_layers = tuple(layer.layer_id for layer in before_snapshot.layers)
    if observed_layers != expected_layers:
        raise LogicalReplayError("counterfactual BEFORE layer IDs must be contiguous")
    for accessed_experts_by_layer in accessed_experts_by_cycle:
        if len(accessed_experts_by_layer) != len(before_snapshot.layers):
            raise LogicalReplayError("counterfactual access layer count does not match BEFORE")
        for layer_id, experts in enumerate(accessed_experts_by_layer):
            invalid = any(
                isinstance(expert, bool)
                or not isinstance(expert, int)
                or expert < 0
                for expert in experts
            )
            if invalid or experts != tuple(sorted(set(experts))):
                raise LogicalReplayError(
                    f"counterfactual layer {layer_id} experts must be unique and in ascending order"
                )
    states = {
        layer.layer_id: _LayerState.from_snapshot(
            layer,
            location=f"counterfactual BEFORE layer {layer.layer_id}",
        )
        for layer in before_snapshot.layers
    }
    results: list[ExpertAccessReplayResult] = []
    for accessed_experts_by_layer in accessed_experts_by_cycle:
        execution_order = accessed_experts_by_layer
        if resident_first:
            execution_by_layer: list[tuple[int, ...]] = []
            for layer_id, experts in enumerate(accessed_experts_by_layer):
                resident = {
                    slot.expert_id
                    for slot in states[layer_id].slots
                    if slot.expert_id is not None
                }
                execution_by_layer.append(
                    tuple(expert for expert in experts if expert in resident)
                    + tuple(
                        expert for expert in experts if expert not in resident
                    )
                )
            execution_order = tuple(execution_by_layer)
        results.append(
            _replay_accesses(
                states,
                accessed_experts_by_layer,
                execution_order_by_layer=execution_order,
            )
        )
    return tuple(results)


def _replay_accesses(
    states: dict[int, _LayerState],
    accessed_experts_by_layer: tuple[tuple[int, ...], ...],
    *,
    execution_order_by_layer: tuple[tuple[int, ...], ...] | None = None,
) -> ExpertAccessReplayResult:
    initial_occupancy = sum(state.occupancy for state in states.values())
    initial_resident = {
        (layer_id, slot.expert_id)
        for layer_id, state in states.items()
        for slot in state.slots
        if slot.expert_id is not None
    }
    stats_before = {
        layer_id: state.stats.freeze()
        for layer_id, state in states.items()
    }
    events: list[LogicalReplayEvent] = []
    execution_order = (
        accessed_experts_by_layer
        if execution_order_by_layer is None
        else execution_order_by_layer
    )
    for layer_id, experts in enumerate(execution_order):
        for expert_id in experts:
            events.extend(states[layer_id].access(expert_id))

    counters = ReplayCounters()
    for layer_id, state in states.items():
        counters += _counter_delta(
            stats_before[layer_id],
            state.stats.freeze(),
            location=f"counterfactual layer {layer_id}",
        )
    counters = ReplayCounters(
        h2d_submissions=counters.h2d_submissions,
        resident_hits=counters.resident_hits,
        inflight_joins=counters.inflight_joins,
        evictions=counters.evictions,
        consume_waits=sum(
            event.kind is RuntimeEventKind.CONSUME_WAIT for event in events
        ),
        releases=sum(event.kind is RuntimeEventKind.RELEASE for event in events),
    )
    final_slots = _freeze_slots(states)
    final_occupancy = sum(slot.expert_id is not None for slot in final_slots)
    if final_occupancy - initial_occupancy != (
        counters.h2d_submissions - counters.evictions
    ):
        raise LogicalReplayError("counterfactual occupancy equation cannot close")
    if counters.event_count != len(events):
        raise LogicalReplayError("counterfactual event count cannot be reconciled")
    reload_h2d = sum(
        event.kind is RuntimeEventKind.H2D_SUBMIT
        and (event.layer_id, event.expert_id) in initial_resident
        for event in events
    )
    initial_nonresident = sum(
        (layer_id, expert_id) not in initial_resident
        for layer_id, experts in enumerate(accessed_experts_by_layer)
        for expert_id in experts
    )
    if counters.h2d_submissions != initial_nonresident + reload_h2d:
        raise LogicalReplayError("counterfactual H2D dynamic classification cannot be closed")
    ordered_states = tuple(state for _, state in sorted(states.items()))
    return ExpertAccessReplayResult(
        accessed_experts_by_layer=accessed_experts_by_layer,
        counters=counters,
        events=tuple(events),
        initial_occupancy=initial_occupancy,
        final_occupancy=final_occupancy,
        final_slots=final_slots,
        final_submission_sequences=tuple(
            state.submission_sequence for state in ordered_states
        ),
        final_lru_sequences=tuple(state.lru_sequence for state in ordered_states),
        initial_resident_reload_h2d=reload_h2d,
    )


def _freeze_slots(states: dict[int, _LayerState]) -> tuple[ReplaySlotState, ...]:
    return tuple(
        ReplaySlotState(
            layer_id=layer_id,
            slot_id=slot.slot_id,
            expert_id=slot.expert_id,
            generation=slot.generation,
            last_used_sequence=slot.last_used_sequence,
        )
        for layer_id, state in sorted(states.items())
        for slot in state.slots
    )


def replay_qwen3_chain(trace: TraceBundle) -> LogicalReplayResult:

    cycles = tuple(sorted(trace.cycles, key=lambda cycle: cycle.iteration))
    if not cycles:
        raise LogicalReplayError("trace has no verification cycle")
    if tuple(cycle.iteration for cycle in cycles) != tuple(range(len(cycles))):
        raise LogicalReplayError("cycle iteration must increment consecutively starting from 0")

    trees = {tree.tree_id: tree for tree in trace.trees}
    snapshots = {snapshot.trace_id: snapshot for snapshot in trace.snapshots}
    routes = {
        (route.tree_id, route.node_id, route.layer_id): route
        for route in trace.target_routes
    }
    scoped_events: dict[str, list[RuntimeEventTrace]] = {
        cycle.trace_id: [] for cycle in cycles
    }
    for event in trace.runtime_events:
        if event.cycle_id is None:
            continue
        if event.cycle_id not in scoped_events:
            raise LogicalReplayError(
                f"runtime event {event.sequence} references an unknown cycle"
            )
        scoped_events[event.cycle_id].append(event)

    first_pre = _resolve_snapshot(
        cycles[0].pre_snapshot_id,
        snapshots,
        expected_cycle_id=cycles[0].trace_id,
        location="iteration 0 BEFORE",
    )
    _require_layer_layout(trace, first_pre, location="iteration 0 BEFORE")
    states = {
        layer.layer_id: _LayerState.from_snapshot(
            layer,
            location=f"iteration 0 BEFORE layer {layer.layer_id}",
        )
        for layer in first_pre.layers
    }
    initial_occupancy = sum(state.occupancy for state in states.values())

    results: list[CycleReplayResult] = []
    totals = ReplayCounters()
    for cycle in cycles:
        tree = trees.get(cycle.tree_id)
        if tree is None:
            raise LogicalReplayError(
                f"iteration {cycle.iteration}: missing tree {cycle.tree_id}"
            )
        pre = _resolve_snapshot(
            cycle.pre_snapshot_id,
            snapshots,
            expected_cycle_id=cycle.trace_id,
            location=f"iteration {cycle.iteration} BEFORE",
        )
        post = _resolve_snapshot(
            cycle.post_snapshot_id,
            snapshots,
            expected_cycle_id=cycle.trace_id,
            location=f"iteration {cycle.iteration} AFTER",
        )
        _require_layer_layout(
            trace,
            pre,
            location=f"iteration {cycle.iteration} BEFORE",
        )
        _require_layer_layout(
            trace,
            post,
            location=f"iteration {cycle.iteration} AFTER",
        )
        pre_by_layer = {layer.layer_id: layer for layer in pre.layers}
        post_by_layer = {layer.layer_id: layer for layer in post.layers}
        for layer_id, state in states.items():
            _assert_state_matches_snapshot(
                state,
                pre_by_layer[layer_id],
                location=f"iteration {cycle.iteration} BEFORE layer {layer_id}",
            )

        verified_node_ids = tuple(
            node.node_id for node in tree.nodes if node.verified
        )
        accessed_by_layer: list[tuple[int, ...]] = []
        for layer_id in range(trace.manifest.num_moe_layers):
            expert_ids: set[int] = set()
            for node_id in verified_node_ids:
                route = routes.get((tree.tree_id, node_id, layer_id))
                if route is None:
                    raise LogicalReplayError(
                        f"iteration {cycle.iteration} layer {layer_id}: "
                        f"missing node {node_id} target route"
                    )
                expert_ids.update(route.expert_ids)
            accessed = tuple(sorted(expert_ids))
            accessed_by_layer.append(accessed)
        predicted = _replay_accesses(states, tuple(accessed_by_layer))
        expected_events = predicted.events

        actual_events = tuple(
            sorted(scoped_events[cycle.trace_id], key=lambda event: event.sequence)
        )
        _assert_events_match(
            tuple(expected_events),
            actual_events,
            states,
            iteration=cycle.iteration,
        )

        for layer_id, state in states.items():
            _assert_state_matches_snapshot(
                state,
                post_by_layer[layer_id],
                location=f"iteration {cycle.iteration} AFTER layer {layer_id}",
            )
        result = CycleReplayResult(
            iteration=cycle.iteration,
            tree_id=cycle.tree_id,
            cycle_id=cycle.trace_id,
            accessed_experts_by_layer=tuple(accessed_by_layer),
            counters=predicted.counters,
            events=expected_events,
            occupancy_before=predicted.initial_occupancy,
            occupancy_after=predicted.final_occupancy,
        )
        results.append(result)
        totals += predicted.counters

    final_slots = _freeze_slots(states)
    return LogicalReplayResult(
        run_id=trace.run_id,
        cycles=tuple(results),
        counters=totals,
        initial_occupancy=initial_occupancy,
        final_occupancy=sum(slot.expert_id is not None for slot in final_slots),
        final_slots=final_slots,
    )


def _resolve_snapshot(
    snapshot_id: str | None,
    snapshots: dict[str, CacheSnapshotTrace],
    *,
    expected_cycle_id: str,
    location: str,
) -> CacheSnapshotTrace:
    if snapshot_id is None:
        raise LogicalReplayError(f"{location}: cycle does not reference a cache snapshot")
    snapshot = snapshots.get(snapshot_id)
    if snapshot is None:
        raise LogicalReplayError(f"{location}: cache snapshot not found {snapshot_id}")
    if snapshot.cycle_id != expected_cycle_id:
        raise LogicalReplayError(f"{location}: snapshot belongs to another cycle")
    return snapshot


def _require_layer_layout(
    trace: TraceBundle,
    snapshot: CacheSnapshotTrace,
    *,
    location: str,
) -> None:
    expected = tuple(range(trace.manifest.num_moe_layers))
    actual = tuple(layer.layer_id for layer in snapshot.layers)
    if actual != expected:
        raise LogicalReplayError(
            f"{location}: layer layout is {actual}, expected {expected}"
        )


def _require_quiescent_layer(
    snapshot: LayerCacheSnapshotTrace,
    *,
    location: str,
) -> None:
    if snapshot.pending_transfers:
        raise LogicalReplayError(f"{location}: pending transfer remains at the cycle boundary")
    if snapshot.stats is not None and snapshot.stats.pending_use_guards:
        raise LogicalReplayError(f"{location}: cycle boundary still has pending use guard")
    for slot in snapshot.slots:
        if slot.active_leases:
            raise LogicalReplayError(f"{location}: active lease remains at cycle boundary")
        if slot.pending_use_guards:
            raise LogicalReplayError(f"{location}: slot still has a pending use guard")
        if slot.expert_id is not None and not slot.ready:
            raise LogicalReplayError(f"{location}: assigned slot not yet ready")


def _assert_state_matches_snapshot(
    state: _LayerState,
    snapshot: LayerCacheSnapshotTrace,
    *,
    location: str,
) -> None:
    _require_quiescent_layer(snapshot, location=location)
    if (
        snapshot.cache_id is None
        or snapshot.lru_sequence is None
        or snapshot.stats is None
    ):
        raise LogicalReplayError(f"{location}: missing R3.4b extended cache state")
    expected_identity = (
        state.layer_id,
        state.cache_id,
        state.queue_id,
        state.capacity,
    )
    actual_identity = (
        snapshot.layer_id,
        snapshot.cache_id,
        snapshot.queue_id,
        snapshot.capacity,
    )
    if actual_identity != expected_identity:
        raise LogicalReplayError(
            f"{location}: cache identity mismatch, "
            f"expected={expected_identity!r}, actual={actual_identity!r}"
        )
    expected_header = (
        state.submission_sequence,
        state.lru_sequence,
        state.stats.h2d_submissions,
        state.stats.resident_hits,
        state.stats.inflight_joins,
        state.stats.evictions,
        state.stats.pending_use_guards,
    )
    actual_header = (
        snapshot.submission_sequence,
        snapshot.lru_sequence,
        snapshot.stats.h2d_submissions,
        snapshot.stats.resident_hits,
        snapshot.stats.inflight_joins,
        snapshot.stats.evictions,
        snapshot.stats.pending_use_guards,
    )
    if actual_header != expected_header:
        raise LogicalReplayError(
            f"{location}: cache counters inconsistent, "
            f"expected={expected_header!r}, actual={actual_header!r}"
        )
    for expected, actual in zip(state.slots, snapshot.slots, strict=True):
        expected_slot = (
            expected.slot_id,
            expected.expert_id,
            expected.generation,
            expected.active_leases,
            expected.last_used_sequence,
        )
        actual_slot = (
            actual.slot_id,
            actual.expert_id,
            actual.generation,
            actual.active_leases,
            actual.last_used_sequence,
        )
        if actual_slot != expected_slot:
            raise LogicalReplayError(
                f"{location} slot {expected.slot_id}: state inconsistent, "
                f"expected={expected_slot!r}, actual={actual_slot!r}"
            )


def _assert_events_match(
    expected: tuple[LogicalReplayEvent, ...],
    actual: tuple[RuntimeEventTrace, ...],
    states: dict[int, _LayerState],
    *,
    iteration: int,
) -> None:
    if len(actual) != len(expected):
        raise LogicalReplayError(
            f"iteration {iteration}: runtime event count is {len(actual)}, "
            f"logical replay generated {len(expected)}"
        )
    for event_index, (expected_event, actual_event) in enumerate(
        zip(expected, actual, strict=True)
    ):
        state = states.get(actual_event.layer_id)
        if state is None:
            raise LogicalReplayError(
                f"iteration {iteration} event {event_index}: unknown layer"
            )
        if (
            actual_event.cache_id != state.cache_id
            or actual_event.queue_id != state.queue_id
        ):
            raise LogicalReplayError(
                f"iteration {iteration} layer {actual_event.layer_id} "
                f"event {event_index}: cache/queue identity inconsistent"
            )
        projected = LogicalReplayEvent(
            kind=actual_event.kind,
            layer_id=actual_event.layer_id,
            expert_id=actual_event.expert_id,
            slot_id=actual_event.slot_id,
            generation=actual_event.generation,
            submission_sequence=actual_event.submission_sequence,
            victim_expert_id=actual_event.victim_expert_id,
        )
        if projected != expected_event:
            raise LogicalReplayError(
                f"iteration {iteration} layer {expected_event.layer_id} "
                f"event {event_index}: expected={expected_event!r}, "
                f"actual={projected!r}"
            )


def _counter_delta(
    before: ReplayCounters,
    after: ReplayCounters,
    *,
    location: str,
) -> ReplayCounters:
    values = {
        "h2d_submissions": after.h2d_submissions - before.h2d_submissions,
        "resident_hits": after.resident_hits - before.resident_hits,
        "inflight_joins": after.inflight_joins - before.inflight_joins,
        "evictions": after.evictions - before.evictions,
    }
    if any(value < 0 for value in values.values()):
        raise LogicalReplayError(f"{location}: cumulative counter regressed")
    return ReplayCounters(**values)
