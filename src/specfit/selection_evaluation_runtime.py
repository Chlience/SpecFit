"""Execute named selection methods from a common, audited generation checkpoint."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
import gc
from hashlib import sha256
from time import perf_counter_ns
from types import SimpleNamespace

import torch

from .checkpoint_audit import audit_boundary
from .f0_direct_analysis import COUNTERS, EXPERT_BYTES, cache_totals, validate_trial
from .f0_direct_runtime import (_CheckpointCaptured, _LightTrace, _TrialCheckpoint,
    _capture_expert_checkpoints, _outcome, _tokens)
from .m1_capture import project_frozen_state
from .m1_capture_replay import logical_state_sha256
from .r6a_experiment import PoolRecord
from .r6a_frozen_cycle import FrozenNodeRoute, FrozenRouteTable
from .r6a_runner import serialize_candidate_pool, serialize_routes
from .runtime import (Eagle3TreeDrafter, ExplicitRawIdTreeSelector, ExpertExecutionPolicy,
    GreedyChainTraceRecorder, ShallowTreeSpeculator, candidate_pool_fingerprint)
from .selection_methods import minimum_expert_ids, probability_ids, select_nodes
from .trace import SnapshotPhase


def cache_state(loaded, manifest):
    recorder = GreedyChainTraceRecorder(manifest, cache_snapshot_provider=loaded.cache_snapshots)
    snapshot = recorder._capture_cache_snapshot(tree_id="selection", cycle_id="selection",
                                               phase=SnapshotPhase.BEFORE_VERIFY)
    return project_frozen_state(snapshot)["logical_state"]


def capture_initial_checkpoint(*, loaded, drafter, input_ids, manifest, state_id):
    """Capture a calibration prompt after prefill, before any draft work."""
    started = perf_counter_ns()
    loaded.cold_reset_expert_caches()
    loaded.set_expert_execution_policy(ExpertExecutionPolicy.EXPERT_ID_ASCENDING)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    ids = torch.tensor([input_ids], dtype=torch.long, device="cuda:0")
    attention = torch.ones_like(ids)

    def prefill_done():
        torch.cuda.synchronize()
        loaded.set_expert_execution_policy(ExpertExecutionPolicy.RESIDENT_FIRST)

    def capture(generation):
        torch.cuda.synchronize()
        pre = cache_state(loaded, manifest)
        prepared_ns = perf_counter_ns() - started
        begin = perf_counter_ns()
        saved = generation.clone()
        experts, allocator = _capture_expert_checkpoints(loaded.expert_caches)
        torch.cuda.synchronize()
        raise _CheckpointCaptured(_TrialCheckpoint(saved, experts, state_id, "",
            logical_state_sha256(pre), perf_counter_ns() - begin, prepared_ns, allocator))

    with torch.inference_mode():
        try:
            ShallowTreeSpeculator(loaded.model, Eagle3TreeDrafter(drafter))._generate(
                ids, attention_mask=attention, max_new_tokens=manifest.max_new_tokens,
                top_k=2, max_depth=5, draft_node_budget=18, stop_on_eos=True,
                _on_boundary=capture, on_prefill_complete=prefill_done)
        except _CheckpointCaptured as event:
            return event.checkpoint
    raise RuntimeError("calibration prompt did not reach a draft boundary")


def restore_experts(*, loaded, checkpoint, mode, manifest):
    if mode not in ("cold", "natural"):
        raise ValueError("unknown cache mode")
    if (len(checkpoint.expert_caches) != len(loaded.expert_caches)
            or any(saved.owner is not cache for saved, cache in
                   zip(checkpoint.expert_caches, loaded.expert_caches, strict=True))):
        raise ValueError("checkpoint belongs to a different expert-cache instance")
    begin = perf_counter_ns()
    layers = [dict(layer_id=i, **cache._restore_checkpoint(saved))
              for i, (cache, saved) in enumerate(zip(
                  loaded.expert_caches, checkpoint.expert_caches, strict=True))]
    torch.cuda.synchronize()
    copy_ns = perf_counter_ns() - begin
    loaded.set_expert_execution_policy(ExpertExecutionPolicy.RESIDENT_FIRST)
    reference = cache_state(loaded, manifest)
    if logical_state_sha256(reference) != checkpoint.reference_cache_sha256:
        raise RuntimeError("restoration did not reproduce the complete reference cache")
    if mode == "cold":
        loaded.cold_reset_expert_caches()
    pre = cache_state(loaded, manifest)
    return pre, copy_ns, layers


@contextmanager
def observe_execution(loaded, execution=None, predictions=None):
    """Observe the same target boundary for the explicit matrix executors."""
    if execution is None:
        yield None
        return
    from .cross_layer_prefetch import CompactTransitions
    from .selection_matrix import EXECUTORS, SCORED_CONTROLS
    from .experiment import canonical_json_sha256

    if execution not in (*EXECUTORS, *SCORED_CONTROLS):
        raise ValueError("unknown matrix executor")
    if execution not in ("predicted_same_layer", *SCORED_CONTROLS) and predictions:
        raise ValueError("this executor does not accept predicted experts")
    loaded.set_expert_h2d_prefetch(execution == "one_ahead")
    observer, outputs, stats = CompactTransitions(), [], {}
    original = loaded.model.forward
    information = dict(execution=execution, cache_events=observer.events, prefetch_stats=stats)
    score_options = {}
    if execution in SCORED_CONTROLS:
        probabilities = {lid: {key.expert_id: int(key.expert_id in (predictions or {}).get(lid, ()))
            for key in cache._sources} for lid, cache in enumerate(loaded.expert_caches)}
        predictions = {lid: sorted(row) for lid, row in probabilities.items()}
        information["score_input_sha256"] = canonical_json_sha256(
            [[row[e] for e in sorted(row)] for row in probabilities.values()])
        score_options = dict(prediction_probabilities=probabilities,
            eviction_policy="probability" if execution == "scored_probability" else "lru")

    def forward(*args, **kwargs):
        if outputs:
            raise RuntimeError("matrix cycle executed more than one target forward")
        if execution in ("demand", "one_ahead"):
            output = original(*args, **kwargs)
        else:
            with loaded.prefetch_experts(predictions or {}, lookahead_layers=0,
                                        max_inflight=2, **score_options) as scheduler:
                output = original(*args, **kwargs)
            stats.update(scheduler.stats())
        outputs.append(output.logits.detach())
        return output

    with ExitStack() as stack:
        loaded.model.forward = forward
        stack.callback(setattr, loaded.model, "forward", original)
        for cache in loaded.expert_caches:
            if cache._observer is not None:
                raise RuntimeError("matrix observation requires exclusive cache ownership")
            cache._observer = observer
            stack.callback(setattr, cache, "_observer", None)
        yield information
    if len(outputs) != 1:
        raise RuntimeError("matrix cycle did not execute its target forward")
    logits = outputs[0]
    information.update(logits_shape=list(logits.shape), logits_dtype=str(logits.dtype),
        logits_sha256=sha256(logits.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest())


class EvaluationSelector:
    """Run the complete selector on the live pool and check the frozen choice."""

    def __init__(self, method, *, atlas=None, model=None, expected_pool=None,
                 expected_ids=None, calibration_count=None):
        self.method, self.atlas, self.model = method, atlas, model
        self.expected_pool, self.expected_ids = expected_pool, expected_ids
        self.calibration_count = calibration_count
        self.pool = None

    def select_raw_ids(self, pool, *, draft_node_budget, committed_offset=0):
        fingerprint = candidate_pool_fingerprint(pool)
        if self.expected_pool is not None and fingerprint != self.expected_pool:
            raise RuntimeError("the live candidate pool differs from its frozen source")
        self.pool = pool
        view = SimpleNamespace(seed_token_id=pool.seed_token_id,
                               fingerprint=fingerprint, nodes=pool.candidates)
        if self.method == "route_probe":
            selected = tuple(n.raw_id for n in pool.candidates)
        elif self.method == "calibration_probability":
            selected = probability_ids(view, self.calibration_count)
        elif self.method == "calibration_minimum_experts":
            selected = minimum_expert_ids(view, self.atlas, self.calibration_count)
        elif self.method == "frozen_selection":
            selected = self.expected_ids
        else:
            selected = select_nodes(view, self.atlas, self.method, self.model)
        if len(selected) > draft_node_budget:
            raise RuntimeError("selection exceeds the runtime node budget")
        if self.expected_ids is not None and selected != self.expected_ids:
            raise RuntimeError("the complete selector did not reproduce its frozen choice")
        return selected


def _record(*, loaded, manifest, checkpoint, source_state, state, method, job_id,
            mode, execution_index, pre, mapping, matrices, outcome, timings,
            prepare_ns, restore_ns, layers, audit, boundary_checks,
            draft_forward_calls, interval, observation=None):
    if (set(matrices) != set(range(48)) or any(len(rows) != len(mapping)
            or any(len(ids) != 8 for ids in rows) for rows in matrices.values())):
        raise RuntimeError("own routes do not cover the executed query shape")
    table = FrozenRouteTable(state.pool.fingerprint, tuple(FrozenNodeRoute(raw,
        tuple(tuple(matrices[layer][position]) for layer in range(48)))
        for position, raw in enumerate(mapping)))
    routes = serialize_routes(table, mapping)
    post = cache_state(loaded, manifest)
    counters = {key: cache_totals(post)[key] - cache_totals(pre)[key] for key in COUNTERS}
    union = [set(e for ids in matrices[layer] for e in ids) for layer in range(48)]
    selected = list(mapping[1:])
    record = dict(method=method, job_id=job_id, state_id=state.state_id,
        question_id=state.question_id, execution_index=execution_index, mode=mode,
        selected_raw_ids=selected, pool_fingerprint=state.pool.fingerprint,
        reference_cache_sha256=checkpoint.reference_cache_sha256,
        pre_logical_state=pre, post_logical_state_sha256=logical_state_sha256(post),
        routes=routes, F_own=sum(map(len, union)), max_own_layer_union=max(map(len, union)),
        cache_counter_delta=counters, logical_h2d_bytes=counters["h2d_submissions"] * EXPERT_BYTES,
        acceptance=outcome, timings=timings, prepare_wall_ns=prepare_ns,
        preparation_method="checkpoint_restore", expert_restore_wall_ns=restore_ns,
        expert_cache_restore_layers=layers, commit_audit=audit, boundary_checks=boundary_checks,
        draft_forward_calls=draft_forward_calls, cycle_wall_interval_ns=interval,
        gc_enabled_during_cycle=gc.isenabled())
    if observation is None:
        validate_trial(record, job={"job_id": job_id, "selected_raw_ids": selected}, state=state,
                       source_state=source_state, variable_budget=True,
                       max_draft_nodes=checkpoint.generation.config[3])
    else:
        from .selection_matrix import validate_cycle
        record.update(observation)
        validate_cycle(record, state=state, source_state=source_state)
    return record


def run_tree_cycle(*, loaded, drafter, checkpoint, state, source_state, selector,
                   mode, manifest, job_id, execution_index, audit=False,
                   execution=None, predictions=None):
    prepared = perf_counter_ns()
    pre, restore_ns, layers = restore_experts(loaded=loaded, checkpoint=checkpoint,
                                             mode=mode, manifest=manifest)
    generation = checkpoint.generation
    tree_drafter = Eagle3TreeDrafter(drafter)
    light = _LightTrace()
    start = None
    checks = {}

    def begin(actual):
        nonlocal start
        if len(actual.state.accepted_per_cycle) != state.cycle_index:
            raise RuntimeError("measurement began at the wrong generation boundary")
        if audit:
            checks.update(audit_boundary(actual, checkpoint, loaded))
        torch.cuda.synchronize()
        start = perf_counter_ns()

    with observe_execution(loaded, execution, predictions) as observation, ExitStack() as stack:
        for lid, layer in enumerate(loaded.model.model.layers):
            handle = layer.mlp.gate.register_forward_hook(light.hook(lid))
            stack.callback(handle.remove)
        with torch.inference_mode():
            result = ShallowTreeSpeculator(loaded.model, tree_drafter)._generate(
                generation.input_ids, attention_mask=generation.attention_mask,
                max_new_tokens=generation.config[0], top_k=2, max_depth=5,
                draft_node_budget=generation.config[3], stop_on_eos=True,
                selector=ExplicitRawIdTreeSelector(selector), trace=light,
                audit_commits=audit, collect_cycle_metrics=True,
                max_cycles=state.cycle_index + 1, _checkpoint=generation, _on_boundary=begin)
            torch.cuda.synchronize()
            end = perf_counter_ns()
    if start is None or len(light.cycles) != 1 or selector.pool is None:
        raise RuntimeError("the requested complete tree cycle was not executed")
    cycle, metric = light.cycles[0], result.tree_cycle_metrics[-1]
    pool = PoolRecord.read(serialize_candidate_pool(selector.pool))
    if state.pool is None:
        state = SimpleNamespace(**{**vars(state), "pool": pool})
    if pool.fingerprint != state.pool.fingerprint:
        raise RuntimeError("measurement pool identity changed")
    record = _record(loaded=loaded, manifest=manifest, checkpoint=checkpoint,
        source_state=source_state, state=state, method=selector.method, job_id=job_id,
        mode=mode, execution_index=execution_index, pre=pre,
        mapping=metric.verification_raw_ids,
        matrices={lid: tensor.cpu().tolist() for lid, tensor in cycle["routes"].items()},
        outcome=_outcome(cycle), timings=dict(cycle_wall_ns=end - start,
            explicit_selector_ns=metric.selector_wall_ns, proposal_wall_ns=metric.proposal_wall_ns,
            target_verification_cuda_ns=metric.target_verification_cuda_ns),
        prepare_ns=start - prepared, restore_ns=restore_ns, layers=layers,
        audit=asdict(result.commit_audits[-1]) if audit else None,
        boundary_checks=checks, draft_forward_calls=metric.draft_forward_calls, interval=[start, end],
        observation=observation)
    return record, pool


def run_target_only_cycle(*, loaded, checkpoint, state, source_state, mode, manifest,
                          job_id, execution_index, audit=False, execution=None, predictions=None):
    """One genuine serial target step: no drafter, candidate pool or tree selector."""
    prepared = perf_counter_ns()
    pre, restore_ns, layers = restore_experts(loaded=loaded, checkpoint=checkpoint,
                                             mode=mode, manifest=manifest)
    generation = replace(checkpoint.generation, state=deepcopy(checkpoint.generation.state))
    saved = generation.state
    torch.set_rng_state(saved.rng_state)
    if saved.cuda_rng_state is not None:
        torch.cuda.set_rng_state(saved.cuda_rng_state)
    checks = audit_boundary(generation, checkpoint, loaded) if audit else {}
    prefix = tuple((layer.keys.clone(), layer.values.clone()) for layer in saved.cache.layers) if audit else ()
    base_length = saved.cache.get_seq_length()
    route_tensors = {}

    def capture(layer):
        def hook(module, inputs, output):
            if layer in route_tensors:
                raise RuntimeError("serial target executed a layer more than once")
            route_tensors[layer] = output[2].detach()
        return hook

    with observe_execution(loaded, execution, predictions) as observation, ExitStack() as stack:
        for lid, layer in enumerate(loaded.model.model.layers):
            handle = layer.mlp.gate.register_forward_hook(capture(lid))
            stack.callback(handle.remove)
        with torch.inference_mode():
            torch.cuda.synchronize()
            start = perf_counter_ns()
            attention = torch.cat((saved.fed_attention, torch.ones_like(saved.pending_seed)), dim=1)
            begin_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin_event.record()
            output = loaded.model(input_ids=saved.pending_seed, attention_mask=attention,
                past_key_values=saved.cache, use_cache=True, logits_to_keep=1)
            verifier = output.logits.argmax(dim=-1)
            end_event.record()
            committed = saved.pending_seed
            # Maintain the serial prefix; no auxiliary draft state is needed.
            saved.fed_ids = torch.cat((saved.fed_ids, committed), dim=1)
            saved.fed_attention = attention
            saved.pending_seed = verifier
            torch.cuda.synchronize()
            end = perf_counter_ns()
    cuda_ns = max(0, round(begin_event.elapsed_time(end_event) * 1_000_000))
    audit_record = None
    if audit:
        audit_record = dict(kind="target_only", kv_layer_count=len(saved.cache.layers),
            kv_prefix_exact=all(torch.equal(layer.keys[..., :base_length, :], keys)
                and torch.equal(layer.values[..., :base_length, :], values)
                for layer, (keys, values) in zip(saved.cache.layers, prefix, strict=True)),
            kv_length_exact=saved.cache.get_seq_length() == base_length + 1 and all(
                layer.keys.shape[-2] == base_length + 1 and layer.values.shape[-2] == base_length + 1
                for layer in saved.cache.layers), verification_node_count=1,
            pending_seed_exact=torch.equal(saved.pending_seed, verifier))
        if not all(audit_record[k] for k in ("kv_prefix_exact", "kv_length_exact", "pending_seed_exact")):
            raise RuntimeError("serial target KV/commit audit failed")
    outcome = dict(verifier_token_ids=_tokens(verifier), committed_token_ids=_tokens(committed),
        accepted_raw_ids=[], accepted_draft_tokens=0, next_pending_seed_token_id=int(verifier[0, 0].item()),
        kv_length_after=saved.cache.get_seq_length())
    return _record(loaded=loaded, manifest=manifest, checkpoint=checkpoint,
        source_state=source_state, state=state, method="target_only", job_id=job_id, mode=mode,
        execution_index=execution_index, pre=pre, mapping=(0,),
        matrices={lid: value.cpu().tolist() for lid, value in route_tensors.items()}, outcome=outcome,
        timings=dict(cycle_wall_ns=end - start, explicit_selector_ns=0, proposal_wall_ns=0,
                     target_verification_cuda_ns=cuda_ns), prepare_ns=start - prepared,
        restore_ns=restore_ns, layers=layers, audit=audit_record, boundary_checks=checks,
        draft_forward_calls=0, interval=[start, end], observation=observation)
