"""Private live-selector and new-prompt checkpoint adapters for refinement runs."""
from __future__ import annotations

from dataclasses import asdict
from time import perf_counter_ns
from types import SimpleNamespace

import torch

from .f0_direct_runtime import _CheckpointCaptured, _TrialCheckpoint, _capture_expert_checkpoints, _tokens
from .m1_capture_replay import logical_state_sha256
from .runtime import (Eagle3TreeDrafter, ExplicitRawIdTreeSelector, ExpertExecutionPolicy,
                      ShallowTreeSpeculator, candidate_pool_fingerprint)
from .selection_evaluation_runtime import EvaluationSelector, cache_state
from .selection_methods import probability_ids
from .selection_refinement import VARIANTS, resident_bitmap, select_refined


def live_resident_bitmap(loaded):
    # resident_expert_ids() reaps completed transfers and changes state_version.
    # Checkpoint restoration deliberately preserves those completed events. Read
    # the occupied slots under the same lock without advancing cache metadata.
    layers = []
    for lid, cache in enumerate(loaded.expert_caches):
        with cache._lock:
            if (cache._closed or cache._closing or any(slot.active_leases for slot in cache._slots)
                    or any(not transfer.ready_event.query() for transfer in cache._outstanding_transfers)):
                raise RuntimeError("selection requires a synchronized, lease-free cache boundary")
            keys = tuple(slot.expert_id for slot in cache._slots if slot.expert_id is not None)
            if any(key.layer_id != lid for key in keys):
                raise RuntimeError("resident expert belongs to another layer")
            layers.append(tuple(key.expert_id for key in keys))
    return resident_bitmap(tuple(layers))


class RefinementSelector(EvaluationSelector):
    def __init__(self, method, *, loaded, cache_model=None, acceptance_exponent=1.0,
                 guard_margin=0.02, expected_resident_mask=None, **kwargs):
        super().__init__(method, **kwargs)
        self.loaded = loaded
        self.cache_model = cache_model
        self.acceptance_exponent = acceptance_exponent
        self.guard_margin = guard_margin
        self.expected_resident_mask = expected_resident_mask
        self.observed_resident_mask = None

    def select_raw_ids(self, pool, *, draft_node_budget, committed_offset=0):
        if self.method not in VARIANTS:
            return super().select_raw_ids(pool, draft_node_budget=draft_node_budget,
                                          committed_offset=committed_offset)
        fingerprint = candidate_pool_fingerprint(pool)
        if self.expected_pool is not None and fingerprint != self.expected_pool:
            raise RuntimeError("live refinement pool differs from its frozen input")
        self.pool = pool
        view = SimpleNamespace(seed_token_id=pool.seed_token_id, fingerprint=fingerprint, nodes=pool.candidates)
        mask = 0
        if self.method in VARIANTS[1:]:
            # Read inside the timed selection call, after restore and before the
            # target query. No query reference or acceptance is available here.
            mask = live_resident_bitmap(self.loaded)
            self.observed_resident_mask = mask
            if self.expected_resident_mask is not None and mask != self.expected_resident_mask:
                raise RuntimeError("live resident experts differ from the selection input")
        selected = select_refined(view, self.atlas, self.method, legacy_model=self.model,
            cache_model=self.cache_model, resident_mask=mask,
            acceptance_exponent=self.acceptance_exponent, guard_margin=self.guard_margin)
        if len(selected) > min(7, draft_node_budget):
            raise RuntimeError("refinement choice exceeds the query budget")
        if self.expected_ids is not None and selected != self.expected_ids:
            raise RuntimeError("live refinement choice differs from the frozen decision")
        return selected


def capture_boundary_checkpoint(*, loaded, drafter, input_ids, manifest, state_id, cycle_index):
    """Reach a predetermined boundary using fixed probability-7 reference cycles."""
    if type(cycle_index) is not int or cycle_index < 0:
        raise ValueError("invalid reference boundary")
    started = perf_counter_ns()
    loaded.cold_reset_expert_caches()
    loaded.set_expert_h2d_prefetch(False)
    loaded.set_expert_execution_policy(ExpertExecutionPolicy.EXPERT_ID_ASCENDING)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    ids = torch.tensor([input_ids], dtype=torch.long, device="cuda:0")
    attention = torch.ones_like(ids)
    selected_history = []
    history = None

    class ReferenceSelector:
        def select_raw_ids(self, pool, *, draft_node_budget, committed_offset=0):
            selected = probability_ids(SimpleNamespace(nodes=pool.candidates), 7)
            selected_history.append(list(selected))
            return selected

    def prefill_done():
        torch.cuda.synchronize()
        loaded.set_expert_execution_policy(ExpertExecutionPolicy.RESIDENT_FIRST)

    def capture(generation):
        nonlocal history
        if len(generation.state.accepted_per_cycle) != cycle_index:
            return
        if (len(generation.state.commit_audits) != cycle_index
                or len(selected_history) != cycle_index
                or any(not audit.passed for audit in generation.state.commit_audits)):
            raise RuntimeError("reference prefix did not pass every commit audit")
        torch.cuda.synchronize()
        pre = cache_state(loaded, manifest)
        prepared_ns = perf_counter_ns() - started
        began = perf_counter_ns()
        saved = generation.clone()
        experts, allocator = _capture_expert_checkpoints(loaded.expert_caches)
        torch.cuda.synchronize()
        captured_ns = perf_counter_ns() - began
        history = dict(reference_probability_nodes=7, selected_raw_ids=selected_history,
            committed_token_ids=[token for tensor in saved.state.generated for token in _tokens(tensor)],
            base_kv_length=saved.state.cache.get_seq_length(),
            commit_audits=[asdict(audit) for audit in saved.state.commit_audits])
        raise _CheckpointCaptured(_TrialCheckpoint(saved, experts, state_id, "",
            logical_state_sha256(pre), captured_ns, prepared_ns, allocator))

    with torch.inference_mode():
        try:
            ShallowTreeSpeculator(loaded.model, Eagle3TreeDrafter(drafter))._generate(
                ids, attention_mask=attention, max_new_tokens=manifest.max_new_tokens,
                top_k=2, max_depth=5, draft_node_budget=18, stop_on_eos=True,
                selector=ExplicitRawIdTreeSelector(ReferenceSelector()), audit_commits=True,
                _on_boundary=capture, on_prefill_complete=prefill_done, max_cycles=cycle_index + 1)
        except _CheckpointCaptured as event:
            return event.checkpoint, history
    raise RuntimeError(f"reference generation ended before predetermined boundary {state_id}")
