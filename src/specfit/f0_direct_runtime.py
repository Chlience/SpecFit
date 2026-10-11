from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, dataclass
from time import perf_counter_ns

import torch

from .f0_direct import select_ids
from .f0_direct_analysis import (COUNTERS, EXPERT_BYTES, cache_totals, validate_trial)
from .m1_capture import project_frozen_state
from .m1_capture_replay import logical_state_sha256
from .r6a_frozen_cycle import FrozenNodeRoute, FrozenRouteTable
from .r6a_runner import serialize_routes
from .runtime import (CumulativeProbabilityTreeSelector, Eagle3TreeDrafter,
    ExplicitRawIdTreeSelector, ExpertExecutionPolicy, GreedyChainTraceRecorder,
    ShallowTreeSpeculator, candidate_pool_fingerprint)
from .trace import SnapshotPhase


@dataclass(frozen=True)
class _TrialCheckpoint:
    generation: object
    expert_caches: tuple
    state_id: str
    pool_fingerprint: str
    reference_cache_sha256: str
    capture_wall_ns: int
    reference_prepare_wall_ns: int
    host_allocator: dict


class _CheckpointCaptured(Exception):
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint


def capture_trial_checkpoint(**kwargs):
    try:
        run_trial(**kwargs, mode="natural", _capture=True)
    except _CheckpointCaptured as captured:
        return captured.checkpoint
    raise RuntimeError("generation stopped before the requested checkpoint boundary")


def _capture_expert_checkpoints(caches):
    before = dict(torch.cuda.memory.host_memory_stats())
    checkpoints = tuple(cache._capture_checkpoint() for cache in caches)
    after = dict(torch.cuda.memory.host_memory_stats())
    return checkpoints, dict(before=before, after=after)


class TrialValidationError(RuntimeError):
    def __init__(self, record, error):
        super().__init__(str(error))
        self.record = record


def _tokens(value) -> list[int]:
    return [int(x) for x in value.detach().cpu().reshape(-1).tolist()]


class _LightTrace:
    def __init__(self):
        self.cycles = []
        self.active = None

    def begin_cycle(self, *, iteration, base_kv_length, verification_token_ids, draft_tree=None):
        self.active = dict(iteration=iteration, base_kv_length=base_kv_length,
            verification_token_ids=verification_token_ids.detach(), tree=draft_tree, routes={})

    def end_cycle(self, *, verifier_tokens, accepted_draft_tokens, committed_token_ids,
                  kv_length_after, accepted_draft_node_ids=None):
        self.active.update(verifier_tokens=verifier_tokens.detach(), accepted_draft_tokens=accepted_draft_tokens,
            committed_token_ids=committed_token_ids.detach(), kv_length_after=kv_length_after,
            accepted_draft_node_ids=accepted_draft_node_ids)
        self.cycles.append(self.active)
        self.active = None

    def abort_cycle(self):
        self.active = None

    def finish_generation(self, **kwargs):
        pass

    def hook(self, layer_id):
        def capture(module, inputs, output):
            if self.active is not None:
                if layer_id in self.active["routes"] or len(output) != 3:
                    raise RuntimeError("F0 gate hook duplicated or output invalid")
                self.active["routes"][layer_id] = output[2].detach()
        return capture


def _outcome(cycle) -> dict:
    positions = cycle["accepted_draft_node_ids"]
    tree = cycle["tree"]
    verifier = _tokens(cycle["verifier_tokens"])
    return {"verifier_token_ids": verifier,
        "accepted_raw_ids": list(tree.raw_ids_for_verification_positions(positions)),
        "accepted_draft_tokens": cycle["accepted_draft_tokens"],
        "committed_token_ids": _tokens(cycle["committed_token_ids"]),
        "next_pending_seed_token_id": verifier[positions[-1] if positions else 0],
        "kv_length_after": cycle["kv_length_after"]}


def _validate_history(light: _LightTrace, reference: dict, count: int) -> None:
    if len(light.cycles) != count:
        raise RuntimeError("F0 reference cycle count mismatch")
    for actual, expected in zip(light.cycles, reference["cycles"][:count], strict=True):
        outcome = _outcome(actual)
        transition = expected["transition"]
        if (outcome["verifier_token_ids"] != expected["verifier_token_ids"]
                or any(outcome[k] != transition[k] for k in
                       ("accepted_raw_ids", "committed_token_ids", "next_pending_seed_token_id"))
                or actual["base_kv_length"] != expected["base_kv_length"]):
            raise RuntimeError("F0 reference acceptance history trajectory diverged")


class _ReplayPolicy:
    def __init__(self, reference, cycle_index, selected):
        self.reference = reference
        self.cycle_index = cycle_index
        self.selected = selected
        self.index = 0
        self.pool = None

    def select_raw_ids(self, pool, *, draft_node_budget, committed_offset=0):
        i = self.index
        self.index += 1
        expected = self.reference["cycles"][i]
        if (candidate_pool_fingerprint(pool) != expected["pool"]["fingerprint"]
                or committed_offset != expected["committed_offset"]):
            raise RuntimeError("F0 failed to reproduce reference pool/offset")
        ids = self.selected if i == self.cycle_index else tuple(expected["selected_raw_ids"])
        if i == self.cycle_index:
            self.pool = pool
        return ids


class _BoundaryDrafter(Eagle3TreeDrafter):
    def __init__(self, drafter, *, cycle_index, prepare):
        super().__init__(drafter)
        self.cycle_index = cycle_index
        self.prepare = prepare
        self.index = 0
        self.boundary_prepare_ns = 0

    def build_candidate_pool(self, **kwargs):
        if self.index == self.cycle_index:
            prepare_start = perf_counter_ns()
            self.prepare(kwargs)
            self.boundary_prepare_ns = perf_counter_ns() - prepare_start
        self.index += 1
        return super().build_candidate_pool(**kwargs)


def run_trial(*, loaded, drafter, source_state, state, reference, prompt_ids,
              job, mode, manifest, execution_index, audit=False,
              _checkpoint=None, _capture=False, _on_boundary=None, _diagnostic_clock=False):
    device = torch.device("cuda:0")
    preparation_start = perf_counter_ns()
    loaded.set_expert_execution_policy(ExpertExecutionPolicy.EXPERT_ID_ASCENDING)
    if _checkpoint is None:
        loaded.cold_reset_expert_caches()
    elif (_checkpoint.state_id != state.state_id
            or _checkpoint.pool_fingerprint != state.pool.fingerprint
            or _checkpoint.reference_cache_sha256 != source_state["pre_logical_state_sha256"]
            or len(_checkpoint.expert_caches) != len(loaded.expert_caches)
            or any(saved.owner is not cache for saved, cache in
                zip(_checkpoint.expert_caches, loaded.expert_caches, strict=True))):
        raise ValueError("trial checkpoint source mismatch")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    light = _LightTrace()
    snapshotter = GreedyChainTraceRecorder(manifest, cache_snapshot_provider=loaded.cache_snapshots)
    pre = start = prepared_ns = None
    reference_cache_hash = None
    prepare_timings_ns = {}
    restore_timings_ns = {}
    restore_layers = []
    experts_restored_ns = None

    def snapshot():
        captured = snapshotter._capture_cache_snapshot(tree_id="f0", cycle_id="f0",
                                                       phase=SnapshotPhase.BEFORE_VERIFY)
        return project_frozen_state(captured)["logical_state"]

    def prefill_done():
        torch.cuda.synchronize(device)
        prepare_timings_ns["initialization_and_prefill"] = perf_counter_ns() - preparation_start
        loaded.set_expert_execution_policy(ExpertExecutionPolicy.RESIDENT_FIRST)

    def prepare(kwargs):
        nonlocal pre, start, prepared_ns, reference_cache_hash
        torch.cuda.synchronize(device)
        reference_done_ns = perf_counter_ns() - preparation_start
        if _checkpoint is None:
            prepare_timings_ns["reference_history"] = (
                reference_done_ns - prepare_timings_ns["initialization_and_prefill"])
            _validate_history(light, reference, state.cycle_index)
        else:
            restore_timings_ns["generation_state_restore"] = reference_done_ns - experts_restored_ns
        if (_tokens(kwargs["fed_ids"]) != reference["cycles"][state.cycle_index]["context_token_ids"]
                or _tokens(kwargs["pending_seed"]) != [source_state["pending_seed_token_id"]]
                or kwargs["aux_history"].shape[1] != source_state["base_kv_length"]):
            raise RuntimeError("F0 reference token/KV-aux boundary inconsistent")
        pre = snapshot()
        reference_cache_hash = logical_state_sha256(pre)
        if reference_cache_hash != source_state["pre_logical_state_sha256"]:
            raise RuntimeError("F0 full reference cache was not rebuilt")
        if mode == "cold":
            loaded.cold_reset_expert_caches()
            pre = snapshot()
        elif mode != "natural":
            raise ValueError("F0 cache mode invalid")
        torch.cuda.synchronize(device)
        prepared_ns = perf_counter_ns() - preparation_start
        parts = prepare_timings_ns if _checkpoint is None else restore_timings_ns
        parts["boundary_validation_and_cache_setup"] = prepared_ns - reference_done_ns
        start = perf_counter_ns()

    policy = _ReplayPolicy(reference, state.cycle_index, tuple(job["selected_raw_ids"]))
    tree_drafter = _BoundaryDrafter(drafter, cycle_index=state.cycle_index, prepare=prepare)
    def boundary(checkpoint):
        if len(checkpoint.state.accepted_per_cycle) != state.cycle_index:
            return
        if _on_boundary is not None:
            _on_boundary(checkpoint)
        if _capture:
            prepare(dict(fed_ids=checkpoint.state.fed_ids,
                pending_seed=checkpoint.state.pending_seed,
                aux_history=checkpoint.state.aux_history))
            capture_start = perf_counter_ns()
            generation = checkpoint.clone()
            caches, host_allocator = _capture_expert_checkpoints(loaded.expert_caches)
            torch.cuda.synchronize(device)
            raise _CheckpointCaptured(_TrialCheckpoint(generation, caches,
                state.state_id, state.pool.fingerprint, reference_cache_hash,
                perf_counter_ns() - capture_start, prepared_ns, host_allocator))

    if _checkpoint is not None:
        torch.cuda.synchronize(device)
        restore_timings_ns["initialization"] = perf_counter_ns() - preparation_start
        for layer_id, (cache, saved) in enumerate(zip(
                loaded.expert_caches, _checkpoint.expert_caches, strict=True)):
            restore_layers.append(dict(layer_id=layer_id, **cache._restore_checkpoint(saved)))
        torch.cuda.synchronize(device)
        experts_restored_ns = perf_counter_ns() - preparation_start
        restore_timings_ns["expert_cache_restore"] = (
            experts_restored_ns - restore_timings_ns["initialization"])
        loaded.set_expert_execution_policy(ExpertExecutionPolicy.RESIDENT_FIRST)
        policy.index = tree_drafter.index = state.cycle_index
    with ExitStack() as stack:
        for lid, layer in enumerate(loaded.model.model.layers):
            handle = layer.mlp.gate.register_forward_hook(light.hook(lid))
            stack.callback(handle.remove)
        with torch.inference_mode():
            speculator = ShallowTreeSpeculator(loaded.model, tree_drafter)
            private_options = {}
            generate = speculator.generate
            if _capture or _checkpoint is not None or _on_boundary is not None:
                generate = speculator._generate
                private_options = dict(_on_boundary=boundary,
                    _checkpoint=None if _checkpoint is None else _checkpoint.generation)
            result = generate(
                input_ids, attention_mask=attention_mask,
                max_new_tokens=reference["max_new_tokens"], top_k=2, max_depth=5,
                draft_node_budget=7, trace=light, on_prefill_complete=prefill_done,
                stop_on_eos=True, audit_commits=audit,
                selector=ExplicitRawIdTreeSelector(policy), collect_cycle_metrics=True,
                max_cycles=state.cycle_index + 1, **private_options)
        torch.cuda.synchronize(device)
        end = perf_counter_ns()
    if (start is None or pre is None or policy.pool is None
            or result.cycles != state.cycle_index + 1
            or len(light.cycles) != (result.cycles if _checkpoint is None else 1)):
        raise RuntimeError("F0 did not execute the complete target cycle")
    if audit and (len(result.commit_audits) != result.cycles
                  or not all(a.passed for a in result.commit_audits)):
        raise RuntimeError("F0 reference/selected KV/aux audit failed")
    post = snapshot()
    cycle = light.cycles[-1]
    metric = result.tree_cycle_metrics[-1]
    pool = policy.pool
    probability = tuple(sorted(CumulativeProbabilityTreeSelector().select_raw_ids(pool, draft_node_budget=7)))
    if probability != select_ids(state.pool, state.atlas, "p7"):
        raise RuntimeError("F0 P7 did not reproduce the production selector")
    mapping = (0, *job["selected_raw_ids"])
    if metric.verification_raw_ids != mapping or set(cycle["routes"]) != set(range(48)):
        raise RuntimeError("F0 own routes coverage is incomplete")
    matrices = {lid: tensor.cpu().tolist() for lid, tensor in cycle["routes"].items()}
    if any(len(rows) != 8 or any(len(ids) != 8 for ids in rows) for rows in matrices.values()):
        raise RuntimeError("F0 own-route query shape invalid")
    table = FrozenRouteTable(state.pool.fingerprint, tuple(FrozenNodeRoute(raw,
        tuple(tuple(matrices[lid][pos]) for lid in range(48))) for pos, raw in enumerate(mapping)))
    table.validate_selection(pool, tuple(job["selected_raw_ids"]))
    routes = serialize_routes(table, mapping)
    union = [set(e for ids in matrices[lid] for e in ids) for lid in range(48)]
    counters = {k: cache_totals(post)[k] - cache_totals(pre)[k] for k in COUNTERS}
    record = {"execution_index": execution_index, "job_id": job["job_id"], "state_id": state.state_id,
        "mode": mode, "selected_raw_ids": job["selected_raw_ids"], "pool_fingerprint": state.pool.fingerprint,
        "reference_cache_sha256": reference_cache_hash, "pre_logical_state": pre,
        "post_logical_state_sha256": logical_state_sha256(post), "routes": routes,
        "F_own": sum(map(len, union)), "max_own_layer_union": max(map(len, union)),
        "cache_counter_delta": counters, "logical_h2d_bytes": counters["h2d_submissions"] * EXPERT_BYTES,
        "acceptance": _outcome(cycle), "prepare_wall_ns": prepared_ns,
        "prepare_timings_ns": prepare_timings_ns,
        "timings": {"cycle_wall_ns": end - start, "explicit_selector_ns": metric.selector_wall_ns,
            "proposal_wall_ns": metric.proposal_wall_ns, "target_verification_cuda_ns": metric.target_verification_cuda_ns,
            "runtime_cycle_wall_ns_including_boundary_preparation": metric.cycle_wall_ns},
        "commit_audit": asdict(result.commit_audits[-1]) if audit else None}
    record["timings"]["proposal_wall_ns"] -= tree_drafter.boundary_prepare_ns
    if _diagnostic_clock:
        record["cycle_wall_interval_ns"] = [start, end]
    if _checkpoint is not None:
        record.pop("prepare_timings_ns")
        record["preparation_method"] = "checkpoint_restore"
        record["restore_timings_ns"] = restore_timings_ns
        record["expert_cache_restore_layers"] = restore_layers
        if sum(restore_timings_ns.values()) != prepared_ns or any(
                value < 0 for value in restore_timings_ns.values()):
            raise RuntimeError("checkpoint restore timing breakdown is inconsistent")
    try:
        validate_trial(record, job=job, state=state, source_state=source_state)
    except Exception as error:
        raise TrialValidationError(record, error) from error
    return record


def measure_selectors(states, manifest) -> list[dict]:
    from .runtime.eagle3_types import Eagle3DraftCandidate, Eagle3DraftCandidatePool
    measurements = []
    expected = {s["state_id"]: s for s in manifest["states"]}
    for state in states:
        if state.state_id not in manifest["pilot_state_ids"]:
            continue
        nodes = tuple(Eagle3DraftCandidate(**asdict(n)) for n in state.pool.nodes)
        pool = Eagle3DraftCandidatePool(state.pool.seed_token_id, 2, 5, nodes, 1, 1)
        samples = {p: [] for p in ("p7", "f0")}
        for rep in range(7):
            for policy in (("p7", "f0") if rep % 2 == 0 else ("f0", "p7")):
                start = perf_counter_ns()
                ids = select_ids(state.pool, state.atlas, policy)
                tree = pool.to_tree(ids)
                elapsed = perf_counter_ns() - start
                if tree.source_raw_ids != tuple(expected[state.state_id]["decisions"][policy]["selected_raw_ids"]):
                    raise RuntimeError("F0 selector timing altered the frozen selection")
                if rep:
                    samples[policy].append(elapsed)
        for policy in samples:
            measurements.append({"state_id": state.state_id, "policy": policy,
                "selected_raw_ids": expected[state.state_id]["decisions"][policy]["selected_raw_ids"],
                "samples_ns": samples[policy]})
    return measurements
