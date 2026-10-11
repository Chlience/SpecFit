"""One-forward predicted expert prefetch with a shared, bounded H2D queue."""
from __future__ import annotations

from collections import deque
from contextlib import ExitStack
from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Mapping, Sequence

import torch

from specfit.state import ExpertKey

from .lease import ExpertLease
from .qwen3_offloaded_experts import Qwen3OffloadedExperts, ReductionMode


@dataclass(slots=True)
class _LayerStats:
    layer_id: int
    predicted_experts: int
    actual_experts: int = 0
    ready_at_router: int = 0
    inflight_at_router: int = 0
    missing_at_router: int = 0
    prediction_copies: int = 0
    demand_copies: int = 0
    unused_prediction_copies: int = 0
    computed_experts: int = 0
    compute_waits: int = 0


class Qwen3ExpertPrefetch:
    """Opt-in scheduling for exactly one complete eager target forward.

    Predictions are per-layer expert IDs known before entering this context.
    Actual router outputs exclusively determine computation. A shared copy
    stream services current demand before future predictions; at most
    ``max_inflight`` new copies remain incomplete. Speculation reserves one
    slot per layer for demand, and may be further limited by
    ``max_prefetch_per_layer``. ``lookahead_layers=0`` confines speculation to
    the current Attention; 1 also uses preceding-layer expert computation.

    Enter/exit at a host-quiescent boundary without external expert leases.
    GPU work is ordered with events, without a device-wide synchronization.
    Original copy streams and the legacy expert paths are restored on exit,
    including exceptions. This context is neither reentrant nor reusable.
    """

    def __init__(
        self, loaded, predictions: Mapping[int, Sequence[int]], *,
        lookahead_layers: int = 1, max_inflight: int = 2,
        max_prefetch_per_layer: int | None = None, trace: bool = False,
        prediction_probabilities: Mapping[int, Mapping[int, float]] | None = None,
        eviction_policy: str = "lru",
    ) -> None:
        for name, value, minimum in (
            ("lookahead_layers", lookahead_layers, 0),
            ("max_inflight", max_inflight, 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if (max_prefetch_per_layer is not None
                and (type(max_prefetch_per_layer) is not int
                     or max_prefetch_per_layer < 0)):
            raise ValueError("max_prefetch_per_layer must be a non-negative integer or None")
        if type(trace) is not bool:
            raise ValueError("trace must be a boolean")
        if not isinstance(predictions, Mapping):
            raise TypeError("predictions must map layer IDs to expert IDs")
        if eviction_policy not in ("lru", "probability"):
            raise ValueError("eviction_policy must be 'lru' or 'probability'")
        if prediction_probabilities is None and eviction_policy != "lru":
            raise ValueError("probability eviction requires prediction_probabilities")
        if prediction_probabilities is not None and not isinstance(prediction_probabilities, Mapping):
            raise TypeError("prediction_probabilities must map layers to expert probabilities")
        self._loaded = loaded
        self._layers = tuple(loaded.model.model.layers)
        self._caches = tuple(loaded.expert_caches)
        if not self._layers or len(self._layers) != len(self._caches):
            raise ValueError("decoder layers and expert caches must match")
        if any(type(layer) is not int or not 0 <= layer < len(self._layers)
               for layer in predictions):
            raise ValueError("prediction layer is outside the loaded model")
        if prediction_probabilities is not None and any(
                type(layer) is not int or not 0 <= layer < len(self._layers)
                for layer in prediction_probabilities):
            raise ValueError("probability layer is outside the loaded model")
        self._limits = tuple(min(len(cache._slots) - 1,
            len(cache._slots) - 1 if max_prefetch_per_layer is None
            else max_prefetch_per_layer) for cache in self._caches)
        self._probabilities = []
        self._requests = []
        normalized = []
        for layer, cache in enumerate(self._caches):
            ids = tuple(predictions.get(layer, ()))
            probabilities = None
            # Reuse source keys for validation and requests. Empty predictions
            # need no source scan unless scores are supplied for eviction.
            keys = ({key.expert_id: key for key in cache._sources
                     if type(key) is ExpertKey and key.layer_id == layer}
                    if ids or prediction_probabilities is not None else {})
            if any(type(expert) is not int or expert not in keys for expert in ids):
                raise ValueError(f"invalid predicted expert in layer {layer}")
            if prediction_probabilities is not None:
                supplied = prediction_probabilities.get(layer, {})
                if not isinstance(supplied, Mapping):
                    raise TypeError("each probability layer must map expert IDs to probabilities")
                scores = {}
                probabilities = {}
                for expert, value in supplied.items():
                    value_type = type(value)
                    if (type(expert) is not int or expert not in keys
                            or value_type is bool
                            or (value_type is not int and value_type is not float
                                and not isinstance(value, Real))
                            or not math.isfinite(value) or not 0 <= value <= 1):
                        raise ValueError(f"invalid expert probability in layer {layer}")
                    scores[expert] = float(value)
                    probabilities[keys[expert]] = scores[expert]
                if any(expert not in scores for expert in ids):
                    raise ValueError("every predicted expert needs an explicit probability")
                ids = tuple(sorted((expert for expert in set(ids) if scores[expert] > 0),
                                   key=lambda expert: (-scores[expert], expert)))[:self._limits[layer]]
            else:
                ids = tuple(sorted(set(ids)))
            self._probabilities.append(probabilities)
            normalized.append(ids)
            self._requests.append(deque(keys[expert] for expert in ids))
        if len({cache.device for cache in self._caches}) != 1:
            raise ValueError("prefetch requires all layers on one GPU")
        self.predictions = tuple(normalized)
        self.lookahead_layers = lookahead_layers
        self.max_inflight = max_inflight
        self.eviction_policy = eviction_policy
        self._trace = trace
        self._events: list[dict] = []
        self._stack = ExitStack()
        self._leases: dict[ExpertKey, ExpertLease] = {}
        self._held_counts = [0] * len(self._layers)
        self._speculative_leases = 0
        self._ranks: dict[ExpertKey, tuple[int, int]] = {}
        self._pending: list[torch.cuda.Event] = []
        # Every transfer records a fresh event. Completion is monotone for
        # that event, including when its old slot later receives new weights.
        self._ready_events: set[torch.cuda.Event] = set()
        self._prediction_copies: set[ExpertKey] = set()
        self._prediction_residents_checked: set[int] = set()
        self._actual: dict[int, set[int]] = {}
        self._layer_stats = [_LayerStats(i, len(ids)) for i, ids in enumerate(self.predictions)]
        self._copy_sequence = 0
        self._next_layer = 0
        self._open_layer: int | None = None
        self._current_layer = 0
        self._entered = self._used = self._complete = False
        self._peak_inflight = self._peak_speculative_leases = 0
        self._queue_capacity_waits = self._slot_wait_copies = 0

    def __enter__(self):
        if self._used:
            raise RuntimeError("a prefetch context cannot be reused")
        modules = tuple(layer.mlp.experts for layer in self._layers)
        if any(not isinstance(module, Qwen3OffloadedExperts)
               or module.cache is not cache
               or module.layer_id != layer
               for layer, (module, cache) in enumerate(zip(modules, self._caches, strict=True))):
            raise RuntimeError("every decoder layer must contain its offloaded expert cache")
        if any(module.reduction_mode is not ReductionMode.CANONICAL for module in modules):
            raise ValueError("predicted prefetch requires canonical reduction")
        if any(module._prediction_scheduler is not None for module in modules):
            raise RuntimeError("another predicted prefetch context is active")
        if any(cache.closed or cache.active_lease_count for cache in self._caches):
            raise RuntimeError("prefetch requires open caches without active leases")
        self._used = True
        try:
            self._copy_stream = torch.cuda.Stream(device=self._caches[0].device)
            for cache in self._caches:
                old_stream = cache.copy_stream
                self._copy_stream.wait_stream(old_stream)
                # Prior asynchronous copies are ordered before our queue and
                # consume credits until complete; no new copies bypass them.
                self._pending.extend(t.ready_event for t in cache._outstanding_transfers
                                     if not t.ready_event.query())
                cache.copy_stream = self._copy_stream
                self._stack.callback(self._restore_stream, cache, old_stream)
            for lid, (layer, module) in enumerate(zip(self._layers, modules, strict=True)):
                module._prediction_scheduler = self
                self._stack.callback(setattr, module, "_prediction_scheduler", None)
                handle = layer.self_attn.register_forward_pre_hook(self._attention_hook(lid))
                self._stack.callback(handle.remove)
            self._entered = True
            self._record("prediction_available")
            self._pump()
            return self
        except BaseException:
            for key in tuple(self._leases):
                self._take_lease(key).release()
            self._entered = False
            self._stack.close()
            raise

    def _restore_stream(self, cache, original) -> None:
        original.wait_stream(self._copy_stream)
        cache.copy_stream = original

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            for key in tuple(self._leases):
                self._take_lease(key).release()
        finally:
            self._entered = False
            self._stack.close()
        self._complete = exc_type is None and self._next_layer == len(self._layers)
        if exc_type is None and not self._complete:
            raise RuntimeError("prefetch context requires exactly one complete target forward")

    def _record(self, action, *, layer=None, expert=None, **values) -> None:
        if self._trace:
            self._events.append(dict(sequence=len(self._events), action=action,
                current_layer=self._current_layer, layer_id=layer, expert_id=expert, **values))

    def _attention_hook(self, layer):
        def hook(module, inputs):
            del module, inputs
            self._begin_layer(layer)
        return hook

    def _begin_layer(self, layer) -> None:
        if not self._entered or self._open_layer is not None or layer != self._next_layer:
            raise RuntimeError("predicted prefetch requires one ordered eager layer traversal")
        self._current_layer = self._open_layer = layer
        self._record("attention", layer=layer)
        self._pump()

    def _event_ready(self, event) -> bool:
        if event in self._ready_events:
            return True
        if event.query():
            self._ready_events.add(event)
            return True
        return False

    def _reap(self) -> None:
        self._pending = [event for event in self._pending if not self._event_ready(event)]

    def _take_lease(self, key):
        lease = self._leases.pop(key)
        self._ranks.pop(key, None)
        self._held_counts[key.layer_id] -= 1
        if key.layer_id not in self._actual:
            self._speculative_leases -= 1
        return lease

    def _hold(self, key, *, resident_only=False, force=False, prediction=False):
        cache = self._caches[key.layer_id]
        before = cache._submission_sequence
        if force:
            lease = cache._prefetch(key)
        else:
            if prediction and self.eviction_policy == "probability":
                lease = cache._prefetch_nonblocking(key, resident_only=resident_only,
                    eviction_probabilities=self._probabilities[key.layer_id])
            else:
                lease = cache._prefetch_nonblocking(key, resident_only=resident_only)
        if lease is None:
            return False
        self._leases[key] = lease
        self._held_counts[key.layer_id] += 1
        if key.layer_id not in self._actual:
            self._speculative_leases += 1
        # The cache only publishes residents after confirming their transfer.
        if key in cache._resident:
            self._ready_events.add(lease.ready_event)
        if cache._submission_sequence != before:
            self._pending.append(lease.ready_event)
            self._ranks[key] = (0, self._copy_sequence)
            self._copy_sequence += 1
            stats = self._layer_stats[key.layer_id]
            if prediction:
                self._prediction_copies.add(key)
                stats.prediction_copies += 1
            else:
                stats.demand_copies += 1
            self._peak_inflight = max(self._peak_inflight, len(self._pending))
            self._record("copy", layer=key.layer_id, expert=key.expert_id,
                         prediction=prediction, slot_wait=force)
        else:
            transfer = cache._inflight.get(key)
            self._ranks[key] = (-1, transfer.submission_sequence if transfer else -1)
        self._peak_speculative_leases = max(self._peak_speculative_leases,
                                            self._speculative_leases)
        return True

    def _pump(self) -> None:
        self._reap()
        last = min(len(self._layers), self._current_layer + self.lookahead_layers + 1)
        for layer in range(self._current_layer, last):
            demand = layer in self._actual
            if not demand and layer not in self._prediction_residents_checked:
                # Protect predicted residents before an earlier-ID miss can
                # evict them. Do this once per layer to avoid rescanning the
                # prediction set at every expert submission boundary.
                cache = self._caches[layer]
                if cache._resident or cache._inflight:
                    for key in self._requests[layer]:
                        if self._held_counts[layer] >= self._limits[layer]:
                            break
                        if key not in self._leases:
                            self._hold(key, resident_only=True)
                self._prediction_residents_checked.add(layer)
            requests = self._requests[layer]
            while requests:
                key = requests[0]
                if key in self._leases:
                    requests.popleft()
                    continue
                if not demand and self._held_counts[layer] >= self._limits[layer]:
                    return
                if self._hold(key, resident_only=True):
                    requests.popleft()
                    continue
                if len(self._pending) >= self.max_inflight:
                    return
                if not self._hold(key, prediction=not demand):
                    return
                requests.popleft()
            # Only move farther ahead after nearer requests have been placed.

    def experts(self, layer: int, active_experts: tuple[int, ...]):
        """Yield each real expert once, preferring completed weight transfers."""
        if not self._entered or self._open_layer != layer or layer in self._actual:
            raise RuntimeError("expert scheduling must follow this layer's Attention exactly once")
        actual = set(active_experts)
        self._speculative_leases -= self._held_counts[layer]
        self._actual[layer] = actual
        remaining = dict.fromkeys(ExpertKey(layer, expert) for expert in sorted(actual))
        stats = self._layer_stats[layer]
        stats.actual_experts = len(actual)
        self._record("router", layer=layer, experts=list(active_experts))
        # Drop incorrect predictions before protecting all actual residents.
        # Existing correct in-flight copies keep their original lease/event.
        for key in tuple(self._leases):
            if key.layer_id == layer and key.expert_id not in actual:
                self._take_lease(key).release()
        stats.unused_prediction_copies = sum(key.layer_id == layer and key.expert_id not in actual
                                            for key in self._prediction_copies)
        for key in remaining:
            if key not in self._leases:
                self._hold(key, resident_only=True)
            lease = self._leases.get(key)
            if lease is None:
                stats.missing_at_router += 1
            elif self._event_ready(lease.ready_event):
                stats.ready_at_router += 1
            else:
                stats.inflight_at_router += 1
        # Actual routing replaces the prediction queue once. Held requests
        # need no further acquisition scans, and consumed keys stay removed.
        self._requests[layer] = deque(key for key in remaining if key not in self._leases)
        while remaining:
            self._pump()
            ready = oldest = None
            for candidate in remaining:
                lease = self._leases.get(candidate)
                if lease is None:
                    continue
                if self._event_ready(lease.ready_event):
                    ready = candidate
                    break
                if oldest is None or self._ranks[candidate] < self._ranks[oldest]:
                    oldest = candidate
            if ready is not None:
                key = ready
                waited = False
            elif oldest is not None:
                key = oldest
                stats.compute_waits += 1
                waited = True
            else:
                self._reap()
                if len(self._pending) >= self.max_inflight:
                    # Only speculative/prior work is in flight, with no real
                    # computation available. Waiting here frees a queue credit.
                    self._queue_capacity_waits += 1
                    self._pending[0].synchronize()
                    continue
                # All reusable slots may still be read by previously submitted
                # kernels. Only this demand fallback may queue a use-event wait.
                key = next(iter(remaining))
                self._hold(key, force=True)
                self._slot_wait_copies += 1
                continue
            lease = self._take_lease(key)
            del remaining[key]
            self._record("compute", layer=layer, expert=key.expert_id, waited=waited)
            try:
                yield key.expert_id, lease
                stats.computed_experts += 1
            finally:
                lease.release()
        self._pump()
        self._open_layer = None
        self._next_layer = layer + 1
        self._record("layer_complete", layer=layer)

    def stats(self) -> dict:
        """Read recorded host statistics without synchronizing the GPU."""
        return dict(complete=self._complete, lookahead_layers=self.lookahead_layers,
            max_inflight=self.max_inflight, prefetch_slot_limits=list(self._limits),
            peak_inflight=self._peak_inflight,
            peak_speculative_leases=self._peak_speculative_leases,
            queue_capacity_waits=self._queue_capacity_waits,
            slot_wait_copies=self._slot_wait_copies,
            layers=[asdict(row) for row in self._layer_stats])

    def events(self) -> tuple[dict, ...]:
        return tuple(dict(row) for row in self._events)
