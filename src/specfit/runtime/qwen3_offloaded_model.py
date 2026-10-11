
from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from transformers import GenerationConfig, Qwen3MoeConfig, Qwen3MoeForCausalLM

from specfit.adapters.qwen3_checkpoint import (
    CheckpointLayout,
    ExpertSpec,
    ModelCensus,
)
from specfit.adapters.qwen3_packed_checkpoint import PackedQwen3Checkpoint
from specfit.state import ExpertKey

from .cache_observer import CacheTransitionObserver
from .cache_snapshot import GpuExpertCacheSnapshot
from .gpu_expert_cache import GpuExpertCache
from .lease import ExpertCacheStats
from .qwen3_offloaded_experts import (
    ExpertExecutionPolicy,
    Qwen3OffloadedExperts,
    ReductionMode,
)
from .registered_mmap import RegisteredMmapExpertLayer

_ROUTED_EXPERT_WEIGHT = re.compile(
    r"^model\.layers\.\d+\.mlp\.experts\.\d+\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
_LOGGER = logging.getLogger(__name__)


class PackedExpertBackingMode(str, Enum):

    PINNED_COPY = "pinned_copy"
    REGISTERED_MMAP = "registered_mmap"


@dataclass(frozen=True, slots=True)
class Qwen3LoadTimings:

    checkpoint_format: str
    expert_backing_mode: str
    checkpoint_inspection_seconds: float
    model_initialization_seconds: float
    expert_pinned_allocation_seconds: float
    expert_weight_read_seconds: float
    expert_mmap_seconds: float
    expert_host_registration_seconds: float
    expert_view_seconds: float
    expert_cache_initialization_seconds: float
    nonexpert_read_seconds: float
    nonexpert_h2d_seconds: float
    finalization_seconds: float
    total_seconds: float
    expert_bytes_read: int
    expert_bytes_mapped: int
    expert_bytes_registered: int
    nonexpert_bytes_read: int


@dataclass(slots=True)
class _MutableLoadTimings:
    checkpoint_format: str = "original"
    expert_backing_mode: str = "original_pinned_copy"
    checkpoint_inspection_seconds: float = 0.0
    model_initialization_seconds: float = 0.0
    expert_pinned_allocation_seconds: float = 0.0
    expert_weight_read_seconds: float = 0.0
    expert_mmap_seconds: float = 0.0
    expert_host_registration_seconds: float = 0.0
    expert_view_seconds: float = 0.0
    expert_cache_initialization_seconds: float = 0.0
    nonexpert_read_seconds: float = 0.0
    nonexpert_h2d_seconds: float = 0.0
    finalization_seconds: float = 0.0
    expert_bytes_read: int = 0
    expert_bytes_mapped: int = 0
    expert_bytes_registered: int = 0
    nonexpert_bytes_read: int = 0

    def freeze(self, *, total_seconds: float) -> Qwen3LoadTimings:
        return Qwen3LoadTimings(
            checkpoint_format=self.checkpoint_format,
            expert_backing_mode=self.expert_backing_mode,
            checkpoint_inspection_seconds=(
                self.checkpoint_inspection_seconds
            ),
            model_initialization_seconds=self.model_initialization_seconds,
            expert_pinned_allocation_seconds=(
                self.expert_pinned_allocation_seconds
            ),
            expert_weight_read_seconds=self.expert_weight_read_seconds,
            expert_mmap_seconds=self.expert_mmap_seconds,
            expert_host_registration_seconds=(
                self.expert_host_registration_seconds
            ),
            expert_view_seconds=self.expert_view_seconds,
            expert_cache_initialization_seconds=(
                self.expert_cache_initialization_seconds
            ),
            nonexpert_read_seconds=self.nonexpert_read_seconds,
            nonexpert_h2d_seconds=self.nonexpert_h2d_seconds,
            finalization_seconds=self.finalization_seconds,
            total_seconds=total_seconds,
            expert_bytes_read=self.expert_bytes_read,
            expert_bytes_mapped=self.expert_bytes_mapped,
            expert_bytes_registered=self.expert_bytes_registered,
            nonexpert_bytes_read=self.nonexpert_bytes_read,
        )


@dataclass(frozen=True, slots=True)
class LoadedOffloadedQwen3:

    model: Qwen3MoeForCausalLM
    expert_caches: tuple[GpuExpertCache, ...]
    load_timings: Qwen3LoadTimings

    def layer_cache_stats(self) -> tuple[ExpertCacheStats, ...]:
        return tuple(cache.stats() for cache in self.expert_caches)

    def cache_snapshots(self) -> tuple[GpuExpertCacheSnapshot, ...]:

        snapshots = tuple(cache.snapshot() for cache in self.expert_caches)
        expected_layers = self.model.config.num_hidden_layers
        expected_cache_ids = tuple(
            f"qwen3-layer-{layer_id}"
            for layer_id in range(expected_layers)
        )
        if tuple(
            snapshot.cache_id for snapshot in snapshots
        ) != expected_cache_ids:
            raise RuntimeError(
                "expert cache order does not match Qwen3 decoder layers"
            )
        queue_ids = tuple(snapshot.queue_id for snapshot in snapshots)
        if len(queue_ids) != len(set(queue_ids)):
            raise RuntimeError("expert cache queue IDs must be unique")
        return snapshots

    def cold_reset_expert_caches(self) -> None:

        snapshots = self.cache_snapshots()
        if any(
            slot.active_leases
            for snapshot in snapshots
            for slot in snapshot.slots
        ):
            raise RuntimeError(
                "cold reset cannot run while an active lease exists"
            )
        devices = {cache.device for cache in self.expert_caches}
        for device in devices:
            torch.cuda.synchronize(device)
        for cache in self.expert_caches:
            cache.cold_reset()

    def cache_stats(self) -> ExpertCacheStats:
        stats = self.layer_cache_stats()
        return ExpertCacheStats(
            h2d_submissions=sum(item.h2d_submissions for item in stats),
            resident_hits=sum(item.resident_hits for item in stats),
            inflight_joins=sum(item.inflight_joins for item in stats),
            evictions=sum(item.evictions for item in stats),
            pending_use_guards=sum(item.pending_use_guards for item in stats),
        )

    def set_expert_execution_policy(
        self,
        execution_policy: ExpertExecutionPolicy | str,
    ) -> None:

        policy = ExpertExecutionPolicy(execution_policy)
        expert_modules = tuple(
            layer.mlp.experts for layer in self.model.model.layers
        )
        if not all(
            isinstance(module, Qwen3OffloadedExperts)
            for module in expert_modules
        ):
            raise RuntimeError(
                "loaded Qwen3 does not contain an offloaded expert module "
                "in every decoder layer"
            )
        if any(module._prediction_scheduler is not None for module in expert_modules):
            raise RuntimeError("cannot switch execution policy during predicted prefetch")
        for module in expert_modules:
            module.execution_policy = policy

    def set_expert_h2d_prefetch(self, enabled: bool) -> None:
        if type(enabled) is not bool:
            raise ValueError("h2d_prefetch must be a boolean")
        modules = tuple(layer.mlp.experts for layer in self.model.model.layers)
        if not all(isinstance(module, Qwen3OffloadedExperts) for module in modules):
            raise RuntimeError("every decoder layer must contain offloaded experts")
        if any(module._prediction_scheduler is not None for module in modules):
            raise RuntimeError("cannot switch H2D policy during predicted prefetch")
        if any(cache.closed or cache.active_lease_count for cache in self.expert_caches):
            raise RuntimeError("prefetch switching requires open caches without active leases")
        for module in modules:
            module.h2d_prefetch = enabled

    def prefetch_experts(
        self, predictions, *, lookahead_layers: int = 1, max_inflight: int = 2,
        max_prefetch_per_layer: int | None = None, trace: bool = False,
        prediction_probabilities=None, eviction_policy: str = "lru",
    ):
        from .qwen3_expert_prefetch import Qwen3ExpertPrefetch

        return Qwen3ExpertPrefetch(self, predictions,
            lookahead_layers=lookahead_layers, max_inflight=max_inflight,
            max_prefetch_per_layer=max_prefetch_per_layer, trace=trace,
            prediction_probabilities=prediction_probabilities, eviction_policy=eviction_policy)

    @property
    def closed(self) -> bool:
        return all(cache.closed for cache in self.expert_caches)

    def close(self) -> None:

        if any(getattr(layer.mlp.experts, "_prediction_scheduler", None) is not None
               for layer in self.model.model.layers):
            raise RuntimeError("cannot close the model during predicted prefetch")

        active_leases = sum(
            cache.active_lease_count for cache in self.expert_caches
        )
        if active_leases:
            raise RuntimeError(
                "loaded Qwen3 cannot close while an active lease exists"
            )
        for cache in reversed(self.expert_caches):
            cache.close()

    def __enter__(self) -> LoadedOffloadedQwen3:
        if self.closed:
            raise RuntimeError("loaded Qwen3 is closed")
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        self.close()


def _copy_expert(
    destination: torch.Tensor,
    expert: ExpertSpec,
    handles: dict[str, object],
) -> None:
    offset = 0
    for location in (
        expert.gate_proj,
        expert.up_proj,
        expert.down_proj,
    ):
        source = handles[location.shard].get_tensor(location.name)
        elements = source.numel()
        destination[offset : offset + elements].copy_(source.reshape(-1))
        offset += elements


def _build_layer_cache(
    packed: torch.Tensor,
    *,
    layer_id: int,
    slot_count: int,
    device: torch.device,
    cache_observer: CacheTransitionObserver | None,
    timings: _MutableLoadTimings,
    source_owner: object | None = None,
) -> GpuExpertCache:
    view_start = time.perf_counter()
    sources = {
        ExpertKey(layer_id, expert_id): packed[expert_id]
        for expert_id in range(packed.shape[0])
    }
    timings.expert_view_seconds += time.perf_counter() - view_start
    cache_start = time.perf_counter()
    cache = GpuExpertCache(
        sources,
        slot_count=slot_count,
        device=device,
        adopt_pinned_sources=True,
        cache_id=f"qwen3-layer-{layer_id}",
        queue_id=f"layer-{layer_id}-copy",
        observer=cache_observer,
        source_owner=source_owner,
    )
    timings.expert_cache_initialization_seconds += (
        time.perf_counter() - cache_start
    )
    return cache


def _load_original_layer_cache(
    layout: CheckpointLayout,
    *,
    layer_id: int,
    slot_count: int,
    device: torch.device,
    cache_observer: CacheTransitionObserver | None,
    timings: _MutableLoadTimings,
) -> GpuExpertCache:
    census = layout.census
    layer_experts = tuple(
        layout.expert(ExpertKey(layer_id, expert_id))
        for expert_id in range(census.num_experts)
    )
    if any(
        location.dtype != "BF16"
        for expert in layer_experts
        for location in (
            expert.gate_proj,
            expert.up_proj,
            expert.down_proj,
        )
    ):
        raise ValueError("P0 loader requires BF16 routed experts")

    packed_elements = census.expert_bytes // torch.bfloat16.itemsize
    allocation_start = time.perf_counter()
    packed = torch.empty(
        (census.num_experts, packed_elements),
        dtype=torch.bfloat16,
        device="cpu",
        pin_memory=True,
    )
    timings.expert_pinned_allocation_seconds += (
        time.perf_counter() - allocation_start
    )
    shard_names = {
        location.shard
        for expert in layer_experts
        for location in (
            expert.gate_proj,
            expert.up_proj,
            expert.down_proj,
        )
    }
    read_start = time.perf_counter()
    with ExitStack() as stack:
        handles = {
            shard: stack.enter_context(
                safe_open(
                    layout.model_root / shard,
                    framework="pt",
                    device="cpu",
                )
            )
            for shard in shard_names
        }
        for expert_id, expert in enumerate(layer_experts):
            _copy_expert(packed[expert_id], expert, handles)
    timings.expert_weight_read_seconds += time.perf_counter() - read_start
    timings.expert_bytes_read += packed.numel() * packed.element_size()

    return _build_layer_cache(
        packed,
        layer_id=layer_id,
        slot_count=slot_count,
        device=device,
        cache_observer=cache_observer,
        timings=timings,
    )


def _load_packed_copy_backings(
    packed_checkpoint: PackedQwen3Checkpoint,
    *,
    timings: _MutableLoadTimings,
) -> tuple[torch.Tensor, ...]:
    census = packed_checkpoint.census
    expected_shape = (
        census.num_experts,
        packed_checkpoint.packed_expert_elements,
    )
    backings = []
    for layer_id in range(census.num_hidden_layers):
        allocation_start = time.perf_counter()
        backing = torch.empty(
            expected_shape,
            dtype=torch.bfloat16,
            device="cpu",
            pin_memory=True,
        )
        timings.expert_pinned_allocation_seconds += (
            time.perf_counter() - allocation_start
        )

        read_start = time.perf_counter()
        with safe_open(
            packed_checkpoint.layer_path(layer_id),
            framework="pt",
            device="cpu",
        ) as handle:
            mapped = handle.get_tensor("experts")
            if mapped.shape != expected_shape:
                raise ValueError(
                    f"packed layer {layer_id} has shape "
                    f"{tuple(mapped.shape)}, expected {expected_shape}"
                )
            if mapped.dtype is not torch.bfloat16:
                raise ValueError(
                    f"packed layer {layer_id} must use BF16"
                )
            backing.copy_(mapped)
            del mapped
        timings.expert_weight_read_seconds += (
            time.perf_counter() - read_start
        )
        timings.expert_bytes_read += (
            backing.numel() * backing.element_size()
        )
        backings.append(backing)
        _LOGGER.info(
            "Loaded prepacked expert layer %d/%d into page-locked host memory",
            layer_id + 1,
            census.num_hidden_layers,
        )
    return tuple(backings)


def _open_registered_packed_layer(
    packed_checkpoint: PackedQwen3Checkpoint,
    *,
    layer_id: int,
    device: torch.device,
    timings: _MutableLoadTimings,
) -> RegisteredMmapExpertLayer:
    census = packed_checkpoint.census
    owner = RegisteredMmapExpertLayer(
        packed_checkpoint.layer_path(layer_id),
        expected_shape=(
            census.num_experts,
            packed_checkpoint.packed_expert_elements,
        ),
        device=device,
    )
    timings.expert_mmap_seconds += owner.mmap_seconds
    timings.expert_host_registration_seconds += (
        owner.registration_seconds
    )
    timings.expert_bytes_mapped += owner.mapped_bytes
    timings.expert_bytes_registered += owner.mapped_bytes
    _LOGGER.info(
        "Mapped and registered prepacked expert layer %d/%d",
        layer_id + 1,
        census.num_hidden_layers,
    )
    return owner


def _nonexpert_manifest(
    model: Qwen3MoeForCausalLM,
    *,
    model_root: Path,
) -> dict[Path, tuple[str, ...]]:
    index = json.loads(
        (model_root / "model.safetensors.index.json").read_text(
            encoding="utf-8"
        )
    )
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in index["weight_map"].items():
        if _ROUTED_EXPERT_WEIGHT.match(name) is None:
            by_shard[shard].append(name)

    by_file = {
        model_root / shard: tuple(sorted(names))
        for shard, names in by_shard.items()
    }
    _validate_nonexpert_files(model, by_file=by_file)
    return by_file


def _packed_nonexpert_manifest(
    model: Qwen3MoeForCausalLM,
    *,
    packed_checkpoint: PackedQwen3Checkpoint,
) -> dict[Path, tuple[str, ...]]:
    path = packed_checkpoint.nonexpert_path
    with safe_open(path, framework="pt", device="cpu") as handle:
        names = tuple(sorted(handle.keys()))
    by_file = {path: names}
    _validate_nonexpert_files(model, by_file=by_file)
    return by_file


def _validate_nonexpert_files(
    model: Qwen3MoeForCausalLM,
    *,
    by_file: dict[Path, tuple[str, ...]],
) -> None:
    parameters = dict(model.named_parameters())
    expected = set(parameters)
    provided = {name for names in by_file.values() for name in names}
    if provided != expected:
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        raise ValueError(
            "nonexpert checkpoint names do not match the HF model: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    for path, names in by_file.items():
        with safe_open(
            path,
            framework="pt",
            device="cpu",
        ) as handle:
            for name in names:
                view = handle.get_slice(name)
                if tuple(view.get_shape()) != tuple(parameters[name].shape):
                    raise ValueError(
                        f"checkpoint shape does not match parameter {name}"
                    )
                if view.get_dtype() != "BF16":
                    raise ValueError(
                        f"P0 loader requires BF16 parameter {name}"
                    )


def _read_nonexpert_parameters(
    *,
    by_file: dict[Path, tuple[str, ...]],
) -> dict[str, torch.Tensor]:
    values = {}
    for path, names in sorted(by_file.items()):
        with safe_open(
            path,
            framework="pt",
            device="cpu",
        ) as handle:
            for name in names:
                values[name] = handle.get_tensor(name).clone()
        _LOGGER.info(
            "Loaded non-expert shard %s (%d parameters)",
            path.name,
            len(names),
        )
    return values


def _install_nonexpert_parameters(
    model: Qwen3MoeForCausalLM,
    *,
    device: torch.device,
    values: dict[str, torch.Tensor],
) -> None:
    torch.cuda.synchronize(device)
    for name, value in values.items():
        set_module_tensor_to_device(
            model,
            name,
            device,
            value=value,
            dtype=torch.bfloat16,
            clear_cache=False,
        )
    torch.cuda.synchronize(device)


def _finish_offloaded_qwen3_load(
    *,
    model: Qwen3MoeForCausalLM,
    caches: list[GpuExpertCache],
    census: ModelCensus,
    nonexpert_files: dict[Path, tuple[str, ...]],
    root: Path,
    target_device: torch.device,
    slots_per_layer: int,
    reduction_mode: ReductionMode | str,
    expert_execution_policy: ExpertExecutionPolicy,
    h2d_prefetch: bool,
    timings: _MutableLoadTimings,
    total_start: float,
) -> LoadedOffloadedQwen3:

    for layer_id, cache in enumerate(caches):
        model.model.layers[layer_id].mlp.experts = Qwen3OffloadedExperts(
            layer_id=layer_id,
            cache=cache,
            hidden_size=census.hidden_size,
            intermediate_size=census.moe_intermediate_size,
            reduction_mode=reduction_mode,
            execution_policy=expert_execution_policy,
            h2d_prefetch=h2d_prefetch,
        )
        _LOGGER.info(
            "Built expert cache layer %d/%d (%d slots)",
            layer_id + 1,
            census.num_hidden_layers,
            slots_per_layer,
        )

    nonexpert_read_start = time.perf_counter()
    nonexpert_values = _read_nonexpert_parameters(by_file=nonexpert_files)
    timings.nonexpert_read_seconds += (
        time.perf_counter() - nonexpert_read_start
    )
    timings.nonexpert_bytes_read += sum(
        value.numel() * value.element_size()
        for value in nonexpert_values.values()
    )
    nonexpert_h2d_start = time.perf_counter()
    _install_nonexpert_parameters(
        model,
        device=target_device,
        values=nonexpert_values,
    )
    timings.nonexpert_h2d_seconds += (
        time.perf_counter() - nonexpert_h2d_start
    )
    del nonexpert_values

    finalization_start = time.perf_counter()
    model.model.rotary_emb.to(target_device)
    meta_parameters = [
        name for name, parameter in model.named_parameters() if parameter.is_meta
    ]
    if meta_parameters:
        raise RuntimeError(
            f"model still has meta parameters: {meta_parameters[:5]}"
        )

    generation_config = root / "generation_config.json"
    if generation_config.exists():
        model.generation_config = GenerationConfig.from_pretrained(
            root,
            local_files_only=True,
        )
    model.requires_grad_(False)
    model.eval()
    torch.cuda.synchronize(target_device)
    timings.finalization_seconds += (
        time.perf_counter() - finalization_start
    )
    frozen_timings = timings.freeze(
        total_seconds=time.perf_counter() - total_start
    )
    _LOGGER.info("Qwen3 loading stage duration: %s", frozen_timings)
    return LoadedOffloadedQwen3(
        model=model,
        expert_caches=tuple(caches),
        load_timings=frozen_timings,
    )


def load_offloaded_qwen3(
    model_root: str | Path,
    *,
    device: torch.device | str,
    slots_per_layer: int,
    packed_checkpoint_root: str | Path | None = None,
    packed_expert_backing: PackedExpertBackingMode | str | None = None,
    verify_packed_hashes: bool = False,
    reduction_mode: ReductionMode | str = ReductionMode.CANONICAL,
    expert_execution_policy: ExpertExecutionPolicy | str = (
        ExpertExecutionPolicy.EXPERT_ID_ASCENDING
    ),
    attn_implementation: str = "sdpa",
    cache_observer: CacheTransitionObserver | None = None,
    h2d_prefetch: bool = True,
) -> LoadedOffloadedQwen3:

    total_start = time.perf_counter()
    if type(h2d_prefetch) is not bool:
        raise ValueError("h2d_prefetch must be a boolean")
    timings = _MutableLoadTimings()
    resolved_execution_policy = ExpertExecutionPolicy(
        expert_execution_policy
    )
    root = Path(model_root).resolve()
    target_device = torch.device(device)
    if target_device.type != "cuda":
        raise ValueError("offloaded Qwen3 requires a CUDA device")
    if target_device.index is None:
        target_device = torch.device("cuda", torch.cuda.current_device())

    inspection_start = time.perf_counter()
    if packed_expert_backing is None:
        backing_mode = (
            PackedExpertBackingMode.REGISTERED_MMAP
            if packed_checkpoint_root is not None
            else PackedExpertBackingMode.PINNED_COPY
        )
    else:
        backing_mode = PackedExpertBackingMode(packed_expert_backing)
    layout: CheckpointLayout | None = None
    packed_checkpoint: PackedQwen3Checkpoint | None = None
    if packed_checkpoint_root is None:
        if backing_mode is not PackedExpertBackingMode.PINNED_COPY:
            raise ValueError(
                "registered_mmap requires packed_checkpoint_root"
            )
        layout = CheckpointLayout.from_pretrained(root)
        census = layout.census
    else:
        packed_checkpoint = PackedQwen3Checkpoint.open(
            packed_checkpoint_root,
            source_metadata_root=root,
            verify_hashes=verify_packed_hashes,
        )
        census = packed_checkpoint.census
        timings.checkpoint_format = "packed-v1"
        timings.expert_backing_mode = backing_mode.value
    if not 1 <= slots_per_layer <= census.num_experts:
        raise ValueError(
            f"slots_per_layer must be in [1, {census.num_experts}]"
        )

    config = Qwen3MoeConfig.from_pretrained(
        root,
        local_files_only=True,
        attn_implementation=attn_implementation,
    )
    timings.checkpoint_inspection_seconds += (
        time.perf_counter() - inspection_start
    )
    if config.hidden_act != "silu":
        raise ValueError("P0 loader requires hidden_act='silu'")
    initialization_start = time.perf_counter()
    with init_empty_weights(include_buffers=False):
        model = Qwen3MoeForCausalLM(config)

    for layer in model.model.layers:
        if not hasattr(layer.mlp, "experts"):
            raise ValueError("P0 loader requires an MoE block in every layer")
        layer.mlp.experts = torch.nn.Identity()
    timings.model_initialization_seconds += (
        time.perf_counter() - initialization_start
    )
    manifest_start = time.perf_counter()
    if packed_checkpoint is None:
        nonexpert_files = _nonexpert_manifest(model, model_root=root)
    else:
        nonexpert_files = _packed_nonexpert_manifest(
            model,
            packed_checkpoint=packed_checkpoint,
        )
    timings.checkpoint_inspection_seconds += (
        time.perf_counter() - manifest_start
    )

    caches = []
    try:
        if packed_checkpoint is None:
            if layout is None:
                raise RuntimeError("original checkpoint layout is missing")
            for layer_id in range(census.num_hidden_layers):
                caches.append(
                    _load_original_layer_cache(
                        layout,
                        layer_id=layer_id,
                        slot_count=slots_per_layer,
                        device=target_device,
                        cache_observer=cache_observer,
                        timings=timings,
                    )
                )
        elif backing_mode is PackedExpertBackingMode.PINNED_COPY:
            packed_backings = _load_packed_copy_backings(
                packed_checkpoint,
                timings=timings,
            )
            for layer_id, backing in enumerate(packed_backings):
                caches.append(
                    _build_layer_cache(
                        backing,
                        layer_id=layer_id,
                        slot_count=slots_per_layer,
                        device=target_device,
                        cache_observer=cache_observer,
                        timings=timings,
                    )
                )
        else:
            for layer_id in range(census.num_hidden_layers):
                owner = _open_registered_packed_layer(
                    packed_checkpoint,
                    layer_id=layer_id,
                    device=target_device,
                    timings=timings,
                )
                try:
                    cache = _build_layer_cache(
                        owner.tensor,
                        layer_id=layer_id,
                        slot_count=slots_per_layer,
                        device=target_device,
                        cache_observer=cache_observer,
                        timings=timings,
                        source_owner=owner,
                    )
                except BaseException:
                    owner.close()
                    raise
                caches.append(cache)
    except BaseException:
        for cache in reversed(caches):
            cache.close()
        raise

    try:
        return _finish_offloaded_qwen3_load(
            model=model,
            caches=caches,
            census=census,
            nonexpert_files=nonexpert_files,
            root=root,
            target_device=target_device,
            slots_per_layer=slots_per_layer,
            reduction_mode=reduction_mode,
            expert_execution_policy=resolved_execution_policy,
            h2d_prefetch=h2d_prefetch,
            timings=timings,
            total_start=total_start,
        )
    except BaseException:
        for cache in reversed(caches):
            cache.close()
        raise
