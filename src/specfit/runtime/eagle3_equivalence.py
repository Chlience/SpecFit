
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn

from .eagle3_types import Eagle3DraftTree


@dataclass(frozen=True, slots=True)
class LayerRouteMismatch:
    layer_id: int
    tree_expert_ids: tuple[int, ...]
    reference_expert_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LayerRoutingEquivalence:

    layer_id: int
    tree_expert_ids: tuple[int, ...]
    reference_expert_ids: tuple[int, ...]
    ordered_equal: bool
    set_equal: bool
    aligned_expert_ids: tuple[int, ...]
    tree_aligned_routing_weights: tuple[float, ...]
    reference_aligned_routing_weights: tuple[float, ...]
    routing_weights_finite: bool
    routing_weights_close: bool
    routing_weights_max_abs: float | None
    routing_weights_mean_abs: float | None


@dataclass(frozen=True, slots=True)
class NodeVerificationEquivalence:
    node_id: int
    parent_id: int | None
    depth: int
    token_id: int
    causal_suffix_token_ids: tuple[int, ...]
    tree_argmax_token_id: int
    reference_argmax_token_id: int
    argmax_equal: bool
    logits_finite: bool
    logits_close: bool
    logits_max_abs: float | None
    logits_mean_abs: float | None
    tree_top1_margin: float | None
    reference_top1_margin: float | None
    top1_margin_abs_difference: float | None
    route_equal_count: int
    route_comparison_count: int
    route_mismatches: tuple[LayerRouteMismatch, ...]
    prefix_cache_clone_exact: bool
    mask_passed: bool
    visible_key_positions: tuple[int, ...]
    expected_key_positions: tuple[int, ...]
    position_passed: bool
    position_id: int
    expected_position_id: int
    ordered_route_equal_count: int
    set_route_equal_count: int
    routing_weights_finite_count: int
    routing_weights_close_count: int
    routing_weights_max_abs: float | None
    routing_weights_mean_abs: float | None
    routes: tuple[LayerRoutingEquivalence, ...]


@dataclass(frozen=True, slots=True)
class TreeVerificationEquivalence:
    passed: bool
    decision_and_route_passed: bool
    semantic_passed: bool
    diagnostic_complete: bool
    reference_mode: str
    pending_seed_token_id: int
    verification_token_ids: tuple[int, ...]
    node_count: int
    layer_count: int
    experts_per_token: int
    argmax_equal_count: int
    logits_finite_count: int
    logits_close_count: int
    route_equal_count: int
    route_comparison_count: int
    logits_atol: float
    logits_rtol: float
    logits_max_abs: float | None
    logits_mean_abs: float | None
    tree_min_top1_margin: float | None
    reference_min_top1_margin: float | None
    top1_margin_max_abs_difference: float | None
    prefix_cache_clone_exact: bool
    tree_prefix_cache_clone_exact: bool
    reference_prefix_cache_clone_exact_count: int
    pristine_prefix_cache_unchanged: bool
    mask_passed_count: int
    position_passed_count: int
    tree_attention_shape: tuple[int, ...]
    tree_attention_dtype: str
    tree_verifier_token_ids: tuple[int, ...]
    reference_verifier_token_ids: tuple[int, ...]
    tree_accepted_node_ids: tuple[int, ...]
    reference_accepted_node_ids: tuple[int, ...]
    accepted_path_equal: bool
    tree_committed_token_ids: tuple[int, ...]
    reference_committed_token_ids: tuple[int, ...]
    committed_tokens_equal: bool
    tree_next_pending_seed_token_id: int
    reference_next_pending_seed_token_id: int
    next_pending_seed_equal: bool
    ordered_route_equal_count: int
    set_route_equal_count: int
    routing_weights_finite_count: int
    routing_weights_close_count: int
    routing_weights_atol: float
    routing_weights_rtol: float
    routing_weights_max_abs: float | None
    routing_weights_mean_abs: float | None
    nodes: tuple[NodeVerificationEquivalence, ...]


@dataclass(frozen=True, slots=True)
class StageLocation:
    layer_id: int
    stage: str


@dataclass(frozen=True, slots=True)
class TensorStageDifference:
    layer_id: int
    stage: str
    shape: tuple[int, ...]
    tree_dtype: str
    reference_dtype: str
    tree_row_count: int
    reference_row_count: int
    finite: bool
    exact: bool
    close: bool
    different_elements: int | None
    element_count: int
    max_abs: float | None
    mean_abs: float | None


@dataclass(frozen=True, slots=True)
class LayerRouteComparison:
    layer_id: int
    tree_row_count: int
    reference_row_count: int
    tree_expert_ids: tuple[int, ...]
    reference_expert_ids: tuple[int, ...]
    ordered_equal: bool
    set_equal: bool


@dataclass(frozen=True, slots=True)
class NodeZeroNumericalReport:

    reference_mode: str
    pending_seed_token_id: int
    verification_node_count: int
    layer_count: int
    stage_order: tuple[str, ...]
    prefix_cache_clone_exact: bool
    node_zero_mask_passed: bool
    tree_attention_shape: tuple[int, ...]
    tree_attention_dtype: str
    node_zero_visible_key_positions: tuple[int, ...]
    node_zero_expected_key_positions: tuple[int, ...]
    node_zero_position_passed: bool
    node_zero_position_id: int
    node_zero_expected_position_id: int
    stage_comparison_count: int
    finite_stage_count: int
    exact_stage_count: int
    close_stage_count: int
    first_exact_difference: StageLocation | None
    first_tolerance_difference: StageLocation | None
    route_comparison_count: int
    ordered_route_equal_count: int
    set_route_equal_count: int
    first_ordered_route_difference_layer: int | None
    first_set_route_difference_layer: int | None
    tree_argmax_token_id: int
    reference_argmax_token_id: int
    argmax_equal: bool
    logits_finite: bool
    logits_close: bool
    logits_max_abs: float | None
    logits_mean_abs: float | None
    diagnostic_complete: bool
    stages: tuple[TensorStageDifference, ...]
    routes: tuple[LayerRouteComparison, ...]


@dataclass(frozen=True, slots=True)
class _CapturedTensor:
    value: torch.Tensor
    row_count: int


_NODE_ZERO_STAGE_ORDER = (
    "layer_input",
    "attention_input",
    "q_projection",
    "q_normalized",
    "k_projection",
    "k_normalized",
    "v_projection",
    "attention_core",
    "attention_output",
    "router_input",
    "router_logits",
    "routing_weights",
    "expert_output",
    "layer_output",
)


class _Qwen3NodeZeroCapture:

    def __init__(self, target: nn.Module) -> None:
        self._target = target
        self._values: dict[tuple[int, str], _CapturedTensor] = {}
        self._routes: dict[int, _CapturedTensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> _Qwen3NodeZeroCapture:
        if self._handles:
            raise RuntimeError("node-zero capture is already active")
        try:
            for layer_id, layer in enumerate(self._target.model.layers):
                attention = getattr(layer, "self_attn", None)
                mlp = getattr(layer, "mlp", None)
                gate = getattr(mlp, "gate", None)
                experts = getattr(mlp, "experts", None)
                if attention is None or gate is None or experts is None:
                    raise ValueError(
                        f"target layer {layer_id} lacks a diagnostic boundary"
                    )
                self._handles.append(
                    layer.register_forward_pre_hook(
                        self._make_pre_hook(layer_id, "layer_input")
                    )
                )
                self._handles.append(
                    layer.register_forward_hook(
                        self._make_output_hook(layer_id, "layer_output")
                    )
                )
                self._handles.append(
                    attention.q_proj.register_forward_hook(
                        self._make_projection_hook(layer_id)
                    )
                )
                self._handles.append(
                    attention.q_norm.register_forward_hook(
                        self._make_output_hook(layer_id, "q_normalized")
                    )
                )
                self._handles.append(
                    attention.k_proj.register_forward_hook(
                        self._make_output_hook(layer_id, "k_projection")
                    )
                )
                self._handles.append(
                    attention.k_norm.register_forward_hook(
                        self._make_output_hook(layer_id, "k_normalized")
                    )
                )
                self._handles.append(
                    attention.v_proj.register_forward_hook(
                        self._make_output_hook(layer_id, "v_projection")
                    )
                )
                self._handles.append(
                    attention.o_proj.register_forward_hook(
                        self._make_attention_output_hook(layer_id)
                    )
                )
                self._handles.append(
                    gate.register_forward_hook(self._make_gate_hook(layer_id))
                )
                self._handles.append(
                    experts.register_forward_hook(
                        self._make_output_hook(
                            layer_id,
                            "expert_output",
                            flattened=True,
                        )
                    )
                )
        except Exception:
            self._remove_hooks()
            raise
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        self._remove_hooks()

    def _remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _capture(
        self,
        *,
        layer_id: int,
        stage: str,
        tensor: torch.Tensor,
        flattened: bool = False,
    ) -> None:
        key = (layer_id, stage)
        if key in self._values:
            raise RuntimeError(f"captured {stage} twice in layer {layer_id}")
        if flattened:
            if tensor.ndim < 2 or tensor.shape[0] < 1:
                raise RuntimeError(f"{stage} has an unexpected flat layout")
            row_count = int(tensor.shape[0])
            value = tensor[0]
        else:
            if tensor.ndim < 3 or tensor.shape[0] != 1 or tensor.shape[1] < 1:
                raise RuntimeError(f"{stage} has an unexpected batched layout")
            row_count = int(tensor.shape[1])
            value = tensor[0, 0]
        self._values[key] = _CapturedTensor(
            value=value.detach().clone(),
            row_count=row_count,
        )

    def _make_pre_hook(self, layer_id: int, stage: str):
        def capture(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            del module
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError(f"{stage} hook received no tensor")
            self._capture(layer_id=layer_id, stage=stage, tensor=inputs[0])

        return capture

    def _make_output_hook(
        self,
        layer_id: int,
        stage: str,
        *,
        flattened: bool = False,
    ):
        def capture(
            module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            del module, inputs
            if not isinstance(output, torch.Tensor):
                raise RuntimeError(f"{stage} hook received no tensor")
            self._capture(
                layer_id=layer_id,
                stage=stage,
                tensor=output,
                flattened=flattened,
            )

        return capture

    def _make_projection_hook(self, layer_id: int):
        def capture(
            module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            del module
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError("q projection hook received no input tensor")
            if not isinstance(output, torch.Tensor):
                raise RuntimeError("q projection hook received no output tensor")
            self._capture(
                layer_id=layer_id,
                stage="attention_input",
                tensor=inputs[0],
            )
            self._capture(
                layer_id=layer_id,
                stage="q_projection",
                tensor=output,
            )

        return capture

    def _make_attention_output_hook(self, layer_id: int):
        def capture(
            module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            del module
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError("o projection hook received no input tensor")
            if not isinstance(output, torch.Tensor):
                raise RuntimeError("o projection hook received no output tensor")
            self._capture(
                layer_id=layer_id,
                stage="attention_core",
                tensor=inputs[0],
            )
            self._capture(
                layer_id=layer_id,
                stage="attention_output",
                tensor=output,
            )

        return capture

    def _make_gate_hook(self, layer_id: int):
        def capture(
            module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ) -> None:
            del module
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError("gate hook received no input tensor")
            if not isinstance(output, tuple) or len(output) != 3:
                raise RuntimeError("Qwen3 gate returned an unexpected output")
            router_logits, routing_weights, selected_experts = output
            self._capture(
                layer_id=layer_id,
                stage="router_input",
                tensor=inputs[0],
                flattened=True,
            )
            self._capture(
                layer_id=layer_id,
                stage="router_logits",
                tensor=router_logits,
                flattened=True,
            )
            self._capture(
                layer_id=layer_id,
                stage="routing_weights",
                tensor=routing_weights,
                flattened=True,
            )
            if selected_experts.ndim != 2 or selected_experts.shape[0] < 1:
                raise RuntimeError("Qwen3 gate routes have an unexpected layout")
            if layer_id in self._routes:
                raise RuntimeError(f"captured routes twice in layer {layer_id}")
            self._routes[layer_id] = _CapturedTensor(
                value=selected_experts[0].detach().clone(),
                row_count=int(selected_experts.shape[0]),
            )

        return capture

    def take_cpu(
        self,
        *,
        expected_rows: int,
    ) -> tuple[
        dict[tuple[int, str], _CapturedTensor],
        dict[int, _CapturedTensor],
    ]:
        expected_layers = len(self._target.model.layers)
        expected_keys = {
            (layer_id, stage)
            for layer_id in range(expected_layers)
            for stage in _NODE_ZERO_STAGE_ORDER
        }
        if set(self._values) != expected_keys:
            raise RuntimeError("node-zero capture missed one or more stages")
        if set(self._routes) != set(range(expected_layers)):
            raise RuntimeError("node-zero capture missed one or more routes")
        if any(item.row_count != expected_rows for item in self._values.values()):
            raise RuntimeError("captured tensor rows do not match target input")
        if any(item.row_count != expected_rows for item in self._routes.values()):
            raise RuntimeError("captured route rows do not match target input")
        values = {
            key: _CapturedTensor(
                value=item.value.cpu(),
                row_count=item.row_count,
            )
            for key, item in self._values.items()
        }
        routes = {
            layer_id: _CapturedTensor(
                value=item.value.cpu(),
                row_count=item.row_count,
            )
            for layer_id, item in self._routes.items()
        }
        self._values.clear()
        self._routes.clear()
        return values, routes


class _Qwen3GateCapture:

    def __init__(self, target: nn.Module) -> None:
        self._target = target
        self._selected_by_layer: dict[int, torch.Tensor] = {}
        self._weights_by_layer: dict[int, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> _Qwen3GateCapture:
        if self._handles:
            raise RuntimeError("gate capture is already active")
        for layer_id, layer in enumerate(self._target.model.layers):
            gate = getattr(getattr(layer, "mlp", None), "gate", None)
            if gate is None:
                self._remove_hooks()
                raise ValueError(f"target layer {layer_id} has no MoE gate")
            self._handles.append(
                gate.register_forward_hook(self._make_hook(layer_id))
            )
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        self._remove_hooks()

    def _remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, layer_id: int):
        def capture(
            module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ) -> None:
            del module, inputs
            if layer_id in self._selected_by_layer:
                raise RuntimeError("target layer routed twice in one forward")
            if not isinstance(output, tuple) or len(output) != 3:
                raise RuntimeError("Qwen3 gate returned an unexpected output")
            routing_weights, selected_experts = output[1:]
            if (
                routing_weights.ndim != 2
                or selected_experts.ndim != 2
                or routing_weights.shape != selected_experts.shape
            ):
                raise RuntimeError("Qwen3 gate routes have an unexpected shape")
            self._selected_by_layer[layer_id] = (
                selected_experts.detach().clone()
            )
            self._weights_by_layer[layer_id] = routing_weights.detach().clone()

        return capture

    def selected_experts(
        self,
        *,
        expected_rows: int,
        experts_per_token: int,
    ) -> tuple[torch.Tensor, ...]:
        expected_layers = len(self._target.model.layers)
        if set(self._selected_by_layer) != set(range(expected_layers)):
            raise RuntimeError("target forward missed one or more MoE gates")
        result = tuple(
            self._selected_by_layer[layer_id]
            for layer_id in range(expected_layers)
        )
        if any(
            value.shape != (expected_rows, experts_per_token)
            for value in result
        ):
            raise RuntimeError("target routes do not match the input layout")
        return result

    def routes(
        self,
        *,
        expected_rows: int,
        experts_per_token: int,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        selected = self.selected_experts(
            expected_rows=expected_rows,
            experts_per_token=experts_per_token,
        )
        expected_layers = len(self._target.model.layers)
        if set(self._weights_by_layer) != set(range(expected_layers)):
            raise RuntimeError("target forward missed one or more routing weights")
        weights = tuple(
            self._weights_by_layer[layer_id]
            for layer_id in range(expected_layers)
        )
        if any(
            value.shape != (expected_rows, experts_per_token)
            for value in weights
        ):
            raise RuntimeError("target routing weights do not match input layout")
        return tuple(zip(selected, weights))


def _verification_path(
    tree: Eagle3DraftTree,
    node_id: int,
) -> tuple[int, ...]:
    if not 0 <= node_id <= tree.draft_node_count:
        raise ValueError("node ID is outside the draft tree")
    ancestors = []
    cursor = node_id
    while cursor != 0:
        ancestors.append(cursor)
        cursor = tree.parent_ids[cursor - 1]
    return (0,) + tuple(reversed(ancestors))


def _compare_layer_routing(
    *,
    layer_id: int,
    tree_experts: torch.Tensor,
    tree_weights: torch.Tensor,
    reference_experts: torch.Tensor,
    reference_weights: torch.Tensor,
    atol: float,
    rtol: float,
) -> LayerRoutingEquivalence:

    tree_expert_ids = tuple(int(value) for value in tree_experts.cpu())
    reference_expert_ids = tuple(
        int(value) for value in reference_experts.cpu()
    )
    if (
        len(set(tree_expert_ids)) != len(tree_expert_ids)
        or len(set(reference_expert_ids)) != len(reference_expert_ids)
    ):
        raise RuntimeError("a top-k route contains duplicate expert IDs")

    tree_values = tuple(
        float(value)
        for value in tree_weights.to(torch.float32).cpu().tolist()
    )
    reference_values = tuple(
        float(value)
        for value in reference_weights.to(torch.float32).cpu().tolist()
    )
    tree_by_expert = dict(zip(tree_expert_ids, tree_values))
    reference_by_expert = dict(zip(reference_expert_ids, reference_values))
    aligned_expert_ids = tuple(
        sorted(set(tree_expert_ids) | set(reference_expert_ids))
    )
    tree_aligned = torch.tensor(
        tuple(tree_by_expert.get(expert_id, 0.0) for expert_id in aligned_expert_ids),
        dtype=torch.float32,
    )
    reference_aligned = torch.tensor(
        tuple(
            reference_by_expert.get(expert_id, 0.0)
            for expert_id in aligned_expert_ids
        ),
        dtype=torch.float32,
    )
    finite = bool(
        torch.isfinite(tree_aligned).all().item()
        and torch.isfinite(reference_aligned).all().item()
    )
    difference = (tree_aligned - reference_aligned).abs()
    return LayerRoutingEquivalence(
        layer_id=layer_id,
        tree_expert_ids=tree_expert_ids,
        reference_expert_ids=reference_expert_ids,
        ordered_equal=tree_expert_ids == reference_expert_ids,
        set_equal=set(tree_expert_ids) == set(reference_expert_ids),
        aligned_expert_ids=aligned_expert_ids,
        tree_aligned_routing_weights=tuple(float(value) for value in tree_aligned),
        reference_aligned_routing_weights=tuple(
            float(value) for value in reference_aligned
        ),
        routing_weights_finite=finite,
        routing_weights_close=bool(
            finite
            and torch.allclose(
                tree_aligned,
                reference_aligned,
                atol=atol,
                rtol=rtol,
            )
        ),
        routing_weights_max_abs=(
            float(difference.max().item()) if finite else None
        ),
        routing_weights_mean_abs=(
            float(difference.mean().item()) if finite else None
        ),
    )


def _top1_margin(logits: torch.Tensor) -> float | None:

    if logits.numel() < 2 or not bool(torch.isfinite(logits).all().item()):
        return None
    top_two = torch.topk(logits.to(torch.float32), k=2).values
    return float((top_two[0] - top_two[1]).item())


def _compare_captured_tensor(
    *,
    layer_id: int,
    stage: str,
    tree: _CapturedTensor,
    reference: _CapturedTensor,
    atol: float,
    rtol: float,
) -> TensorStageDifference:
    if tree.value.shape != reference.value.shape:
        raise RuntimeError(f"{stage} values have different node-row shapes")
    tree_value = tree.value
    reference_value = reference.value
    finite = bool(
        torch.isfinite(tree_value).all().item()
        and torch.isfinite(reference_value).all().item()
    )
    difference = (
        tree_value.to(torch.float32) - reference_value.to(torch.float32)
    ).abs()
    return TensorStageDifference(
        layer_id=layer_id,
        stage=stage,
        shape=tuple(int(value) for value in tree_value.shape),
        tree_dtype=str(tree_value.dtype).removeprefix("torch."),
        reference_dtype=str(reference_value.dtype).removeprefix("torch."),
        tree_row_count=tree.row_count,
        reference_row_count=reference.row_count,
        finite=finite,
        exact=bool(torch.equal(tree_value, reference_value)),
        close=bool(
            finite
            and torch.allclose(
                tree_value,
                reference_value,
                atol=atol,
                rtol=rtol,
            )
        ),
        different_elements=(
            int(torch.count_nonzero(tree_value != reference_value).item())
            if finite
            else None
        ),
        element_count=tree_value.numel(),
        max_abs=float(difference.max().item()) if finite else None,
        mean_abs=float(difference.mean().item()) if finite else None,
    )


def _prompt_cache_clone_is_exact(
    original: object,
    cloned: object,
    *,
    expected_layers: int,
) -> bool:
    original_layers = getattr(original, "layers", None)
    cloned_layers = getattr(cloned, "layers", None)
    if (
        original is cloned
        or original_layers is None
        or cloned_layers is None
        or original_layers is cloned_layers
        or len(original_layers) != expected_layers
        or len(cloned_layers) != expected_layers
    ):
        return False
    for original_layer, cloned_layer in zip(original_layers, cloned_layers):
        if original_layer is cloned_layer:
            return False
        if bool(original_layer.is_initialized) != bool(
            cloned_layer.is_initialized
        ):
            return False
        if not original_layer.is_initialized:
            continue
        for original_tensor, cloned_tensor in (
            (original_layer.keys, cloned_layer.keys),
            (original_layer.values, cloned_layer.values),
        ):
            if (
                original_tensor.shape != cloned_tensor.shape
                or original_tensor.stride() != cloned_tensor.stride()
                or original_tensor.dtype != cloned_tensor.dtype
                or original_tensor.device != cloned_tensor.device
                or original_tensor.untyped_storage().data_ptr()
                == cloned_tensor.untyped_storage().data_ptr()
                or not torch.equal(original_tensor, cloned_tensor)
            ):
                return False
    return True


@torch.inference_mode()
def diagnose_tree_node_zero_numerics(
    target: nn.Module,
    input_ids: torch.Tensor,
    tree: Eagle3DraftTree,
    *,
    attention_mask: torch.Tensor | None = None,
    expected_pending_seed: torch.Tensor | None = None,
    atol: float = 2e-4,
    rtol: float = 1e-2,
) -> NodeZeroNumericalReport:

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("node-zero diagnostic currently requires batch size 1")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    elif attention_mask.shape != input_ids.shape or not attention_mask.all():
        raise ValueError("node-zero diagnostic requires an unpadded input")
    if atol < 0.0 or rtol < 0.0:
        raise ValueError("diagnostic tolerances must be non-negative")
    if getattr(target.config, "_attn_implementation", None) != "sdpa":
        raise ValueError("node-zero diagnostic currently requires SDPA")

    layer_count = len(target.model.layers)
    tree_prefill = target(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        logits_to_keep=1,
    )
    pending_seed = tree_prefill.logits.argmax(dim=-1)
    if expected_pending_seed is not None:
        if expected_pending_seed.shape != (1, 1):
            raise ValueError("expected_pending_seed must have shape [1, 1]")
        if not torch.equal(pending_seed, expected_pending_seed):
            raise RuntimeError("diagnostic prefill produced a different seed")

    verification_ids = tree.verification_token_ids(pending_seed)
    verification_rows = verification_ids.shape[1]
    base_kv_length = tree_prefill.past_key_values.get_seq_length()
    reference_cache = copy.deepcopy(tree_prefill.past_key_values)
    prefix_cache_clone_exact = _prompt_cache_clone_is_exact(
        tree_prefill.past_key_values,
        reference_cache,
        expected_layers=layer_count,
    )
    tree_attention = tree.verification_attention_mask(
        base_kv_length=base_kv_length,
        device=input_ids.device,
    )
    tree_positions = tree.verification_position_ids(
        base_kv_length=base_kv_length,
        device=input_ids.device,
    )
    node_zero_visible = tuple(
        int(value)
        for value in torch.nonzero(
            tree_attention[0, 0, 0],
            as_tuple=False,
        )
        .reshape(-1)
        .cpu()
    )
    expected_visible = tuple(range(base_kv_length + 1))
    expected_tree_attention_shape = (
        1,
        1,
        verification_rows,
        base_kv_length + verification_rows,
    )
    node_zero_mask_passed = (
        tuple(tree_attention.shape) == expected_tree_attention_shape
        and tree_attention.dtype is torch.bool
        and node_zero_visible == expected_visible
    )
    node_zero_position_id = int(tree_positions[0, 0].item())
    node_zero_position_passed = node_zero_position_id == base_kv_length

    tree_capture = _Qwen3NodeZeroCapture(target)
    with tree_capture:
        tree_outputs = target(
            input_ids=verification_ids,
            attention_mask=tree_attention,
            position_ids=tree_positions,
            past_key_values=tree_prefill.past_key_values,
            use_cache=True,
            logits_to_keep=0,
        )
    tree_values, tree_routes = tree_capture.take_cpu(
        expected_rows=verification_rows,
    )
    tree_logits = tree_outputs.logits[0, 0].detach().cpu()
    del tree_outputs, tree_prefill

    reference_attention = torch.ones(
        (1, 1, 1, base_kv_length + 1),
        dtype=torch.bool,
        device=input_ids.device,
    )
    reference_positions = torch.tensor(
        [[base_kv_length]],
        dtype=torch.long,
        device=input_ids.device,
    )
    reference_capture = _Qwen3NodeZeroCapture(target)
    with reference_capture:
        reference_outputs = target(
            input_ids=pending_seed,
            attention_mask=reference_attention,
            position_ids=reference_positions,
            past_key_values=reference_cache,
            use_cache=True,
            logits_to_keep=1,
        )
    reference_values, reference_routes = reference_capture.take_cpu(
        expected_rows=1,
    )
    reference_logits = reference_outputs.logits[0, -1].detach().cpu()
    del reference_outputs, reference_cache

    stage_results = tuple(
        _compare_captured_tensor(
            layer_id=layer_id,
            stage=stage,
            tree=tree_values[(layer_id, stage)],
            reference=reference_values[(layer_id, stage)],
            atol=atol,
            rtol=rtol,
        )
        for layer_id in range(layer_count)
        for stage in _NODE_ZERO_STAGE_ORDER
    )
    route_results = []
    for layer_id in range(layer_count):
        tree_route = tree_routes[layer_id]
        reference_route = reference_routes[layer_id]
        tree_experts = tuple(int(value) for value in tree_route.value.tolist())
        reference_experts = tuple(
            int(value) for value in reference_route.value.tolist()
        )
        route_results.append(
            LayerRouteComparison(
                layer_id=layer_id,
                tree_row_count=tree_route.row_count,
                reference_row_count=reference_route.row_count,
                tree_expert_ids=tree_experts,
                reference_expert_ids=reference_experts,
                ordered_equal=tree_experts == reference_experts,
                set_equal=set(tree_experts) == set(reference_experts),
            )
        )
    routes = tuple(route_results)

    first_exact = next((item for item in stage_results if not item.exact), None)
    first_tolerance = next(
        (item for item in stage_results if not item.close),
        None,
    )
    first_route = next((item for item in routes if not item.ordered_equal), None)
    first_set_route = next((item for item in routes if not item.set_equal), None)
    logits_finite = bool(
        torch.isfinite(tree_logits).all().item()
        and torch.isfinite(reference_logits).all().item()
    )
    logits_difference = (
        tree_logits.to(torch.float32) - reference_logits.to(torch.float32)
    ).abs()
    finite_stage_count = sum(item.finite for item in stage_results)
    diagnostic_complete = (
        prefix_cache_clone_exact
        and node_zero_mask_passed
        and node_zero_position_passed
        and len(stage_results) == layer_count * len(_NODE_ZERO_STAGE_ORDER)
        and finite_stage_count == len(stage_results)
        and len(routes) == layer_count
        and logits_finite
    )
    return NodeZeroNumericalReport(
        reference_mode="cached_single_seed_query",
        pending_seed_token_id=int(pending_seed[0, 0].item()),
        verification_node_count=verification_rows,
        layer_count=layer_count,
        stage_order=_NODE_ZERO_STAGE_ORDER,
        prefix_cache_clone_exact=prefix_cache_clone_exact,
        node_zero_mask_passed=node_zero_mask_passed,
        tree_attention_shape=tuple(
            int(value) for value in tree_attention.shape
        ),
        tree_attention_dtype=str(tree_attention.dtype).removeprefix("torch."),
        node_zero_visible_key_positions=node_zero_visible,
        node_zero_expected_key_positions=expected_visible,
        node_zero_position_passed=node_zero_position_passed,
        node_zero_position_id=node_zero_position_id,
        node_zero_expected_position_id=base_kv_length,
        stage_comparison_count=len(stage_results),
        finite_stage_count=finite_stage_count,
        exact_stage_count=sum(item.exact for item in stage_results),
        close_stage_count=sum(item.close for item in stage_results),
        first_exact_difference=(
            StageLocation(first_exact.layer_id, first_exact.stage)
            if first_exact is not None
            else None
        ),
        first_tolerance_difference=(
            StageLocation(first_tolerance.layer_id, first_tolerance.stage)
            if first_tolerance is not None
            else None
        ),
        route_comparison_count=len(routes),
        ordered_route_equal_count=sum(item.ordered_equal for item in routes),
        set_route_equal_count=sum(item.set_equal for item in routes),
        first_ordered_route_difference_layer=(
            first_route.layer_id if first_route is not None else None
        ),
        first_set_route_difference_layer=(
            first_set_route.layer_id if first_set_route is not None else None
        ),
        tree_argmax_token_id=int(tree_logits.argmax().item()),
        reference_argmax_token_id=int(reference_logits.argmax().item()),
        argmax_equal=int(tree_logits.argmax().item())
        == int(reference_logits.argmax().item()),
        logits_finite=logits_finite,
        logits_close=bool(
            logits_finite
            and torch.allclose(
                tree_logits,
                reference_logits,
                atol=atol,
                rtol=rtol,
            )
        ),
        logits_max_abs=(
            float(logits_difference.max().item()) if logits_finite else None
        ),
        logits_mean_abs=(
            float(logits_difference.mean().item()) if logits_finite else None
        ),
        diagnostic_complete=diagnostic_complete,
        stages=stage_results,
        routes=routes,
    )


@torch.inference_mode()
def compare_tree_verification_to_causal_paths(
    target: nn.Module,
    input_ids: torch.Tensor,
    tree: Eagle3DraftTree,
    *,
    attention_mask: torch.Tensor | None = None,
    expected_pending_seed: torch.Tensor | None = None,
    logits_atol: float = 2e-4,
    logits_rtol: float = 1e-2,
    routing_weights_atol: float = 2e-4,
    routing_weights_rtol: float = 1e-2,
) -> TreeVerificationEquivalence:

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("tree equivalence currently requires batch size 1")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    elif attention_mask.shape != input_ids.shape or not attention_mask.all():
        raise ValueError("tree equivalence requires an unpadded input")
    if (
        logits_atol < 0.0
        or logits_rtol < 0.0
        or routing_weights_atol < 0.0
        or routing_weights_rtol < 0.0
    ):
        raise ValueError("diagnostic tolerances must be non-negative")
    if getattr(target.config, "_attn_implementation", None) != "sdpa":
        raise ValueError("tree equivalence currently requires SDPA")

    experts_per_token = int(target.config.num_experts_per_tok)
    layer_count = len(target.model.layers)
    prefill = target(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        logits_to_keep=1,
    )
    pending_seed = prefill.logits.argmax(dim=-1)
    if expected_pending_seed is not None:
        if expected_pending_seed.shape != (1, 1):
            raise ValueError("expected_pending_seed must have shape [1, 1]")
        if not torch.equal(pending_seed, expected_pending_seed):
            raise RuntimeError("diagnostic prefill produced a different seed")

    verification_ids = tree.verification_token_ids(pending_seed)
    verification_rows = verification_ids.shape[1]
    base_kv_length = prefill.past_key_values.get_seq_length()
    pristine_prefix_cache = prefill.past_key_values
    pristine_prefix_guard = copy.deepcopy(pristine_prefix_cache)
    pristine_prefix_guard_exact = _prompt_cache_clone_is_exact(
        pristine_prefix_cache,
        pristine_prefix_guard,
        expected_layers=layer_count,
    )
    tree_prefix_cache = copy.deepcopy(pristine_prefix_cache)
    tree_prefix_cache_clone_exact = _prompt_cache_clone_is_exact(
        pristine_prefix_cache,
        tree_prefix_cache,
        expected_layers=layer_count,
    )
    tree_attention = tree.verification_attention_mask(
        base_kv_length=base_kv_length,
        device=input_ids.device,
    )
    tree_positions = tree.verification_position_ids(
        base_kv_length=base_kv_length,
        device=input_ids.device,
    )
    expected_tree_attention_shape = (
        1,
        1,
        verification_rows,
        base_kv_length + verification_rows,
    )
    tree_attention_layout_passed = (
        tuple(tree_attention.shape) == expected_tree_attention_shape
        and tree_attention.dtype is torch.bool
    )
    tree_position_layout_passed = (
        tuple(tree_positions.shape) == (1, verification_rows)
        and tree_positions.dtype is torch.long
    )

    with _Qwen3GateCapture(target) as tree_capture:
        tree_outputs = target(
            input_ids=verification_ids,
            attention_mask=tree_attention,
            position_ids=tree_positions,
            past_key_values=tree_prefix_cache,
            use_cache=True,
            logits_to_keep=0,
        )
    tree_logits = tree_outputs.logits[0].detach().clone()
    tree_routes = tree_capture.routes(
        expected_rows=verification_rows,
        experts_per_token=experts_per_token,
    )
    del tree_outputs, tree_prefix_cache
    pristine_after_tree_exact = _prompt_cache_clone_is_exact(
        pristine_prefix_guard,
        pristine_prefix_cache,
        expected_layers=layer_count,
    )

    node_results = []
    reference_prefix_cache_clone_exact_count = 0
    for node_id in range(verification_rows):
        verification_path = _verification_path(tree, node_id)
        suffix = verification_ids[:, verification_path]
        reference_prefix_cache = copy.deepcopy(pristine_prefix_cache)
        reference_prefix_cache_clone_exact = _prompt_cache_clone_is_exact(
            pristine_prefix_cache,
            reference_prefix_cache,
            expected_layers=layer_count,
        )
        reference_prefix_cache_clone_exact_count += int(
            reference_prefix_cache_clone_exact
        )
        suffix_length = suffix.shape[1]
        causal_attention = torch.zeros(
            (1, 1, suffix_length, base_kv_length + suffix_length),
            dtype=torch.bool,
            device=input_ids.device,
        )
        causal_attention[..., :base_kv_length] = True
        causal_attention[0, 0, :, base_kv_length:] = torch.ones(
            (suffix_length, suffix_length),
            dtype=torch.bool,
            device=input_ids.device,
        ).tril()
        causal_position_ids = torch.arange(
            base_kv_length,
            base_kv_length + suffix_length,
            dtype=torch.long,
            device=input_ids.device,
        ).unsqueeze(0)
        with _Qwen3GateCapture(target) as causal_capture:
            causal_outputs = target(
                input_ids=suffix,
                attention_mask=causal_attention,
                position_ids=causal_position_ids,
                past_key_values=reference_prefix_cache,
                use_cache=True,
                logits_to_keep=1,
            )
        causal_logits = causal_outputs.logits[0, -1].detach()
        causal_routes = causal_capture.routes(
            expected_rows=suffix.shape[1],
            experts_per_token=experts_per_token,
        )

        current_tree_logits = tree_logits[node_id]
        difference = (
            current_tree_logits.to(torch.float32)
            - causal_logits.to(torch.float32)
        ).abs()
        logits_finite = bool(
            torch.isfinite(current_tree_logits).all().item()
            and torch.isfinite(causal_logits).all().item()
        )
        tree_argmax = int(current_tree_logits.argmax().item())
        reference_argmax = int(causal_logits.argmax().item())
        tree_top1_margin = _top1_margin(current_tree_logits)
        reference_top1_margin = _top1_margin(causal_logits)
        layer_routes = tuple(
            _compare_layer_routing(
                layer_id=layer_id,
                tree_experts=tree_layer[0][node_id],
                tree_weights=tree_layer[1][node_id],
                reference_experts=causal_layer[0][-1],
                reference_weights=causal_layer[1][-1],
                atol=routing_weights_atol,
                rtol=routing_weights_rtol,
            )
            for layer_id, (tree_layer, causal_layer) in enumerate(
                zip(tree_routes, causal_routes)
            )
        )
        route_mismatches = []
        for route in layer_routes:
            if not route.ordered_equal:
                route_mismatches.append(
                    LayerRouteMismatch(
                        layer_id=route.layer_id,
                        tree_expert_ids=route.tree_expert_ids,
                        reference_expert_ids=route.reference_expert_ids,
                    )
                )
        parent_id = None if node_id == 0 else tree.parent_ids[node_id - 1]
        depth = 0 if node_id == 0 else tree.depths[node_id - 1]
        token_id = int(verification_ids[0, node_id].item())
        visible_key_positions = tuple(
            int(value)
            for value in torch.nonzero(
                tree_attention[0, 0, node_id],
                as_tuple=False,
            )
            .reshape(-1)
            .cpu()
        )
        expected_key_positions = tuple(range(base_kv_length)) + tuple(
            base_kv_length + path_node_id
            for path_node_id in verification_path
        )
        expected_position_id = base_kv_length + depth
        position_id = int(tree_positions[0, node_id].item())
        routing_weights_finite_count = sum(
            route.routing_weights_finite for route in layer_routes
        )
        finite_route_weights = (
            routing_weights_finite_count == len(layer_routes)
        )
        aligned_weight_count = sum(
            len(route.aligned_expert_ids) for route in layer_routes
        )
        node_results.append(
            NodeVerificationEquivalence(
                node_id=node_id,
                parent_id=parent_id,
                depth=depth,
                token_id=token_id,
                causal_suffix_token_ids=tuple(
                    int(value) for value in suffix[0].tolist()
                ),
                tree_argmax_token_id=tree_argmax,
                reference_argmax_token_id=reference_argmax,
                argmax_equal=tree_argmax == reference_argmax,
                logits_finite=logits_finite,
                logits_close=bool(
                    torch.allclose(
                        current_tree_logits,
                        causal_logits,
                        atol=logits_atol,
                        rtol=logits_rtol,
                    )
                ),
                logits_max_abs=(
                    float(difference.max().item())
                    if logits_finite
                    else None
                ),
                logits_mean_abs=(
                    float(difference.mean().item())
                    if logits_finite
                    else None
                ),
                tree_top1_margin=tree_top1_margin,
                reference_top1_margin=reference_top1_margin,
                top1_margin_abs_difference=(
                    abs(tree_top1_margin - reference_top1_margin)
                    if tree_top1_margin is not None
                    and reference_top1_margin is not None
                    else None
                ),
                route_equal_count=layer_count - len(route_mismatches),
                route_comparison_count=layer_count,
                route_mismatches=tuple(route_mismatches),
                prefix_cache_clone_exact=reference_prefix_cache_clone_exact,
                mask_passed=(
                    tree_attention_layout_passed
                    and visible_key_positions == expected_key_positions
                ),
                visible_key_positions=visible_key_positions,
                expected_key_positions=expected_key_positions,
                position_passed=(
                    tree_position_layout_passed
                    and position_id == expected_position_id
                ),
                position_id=position_id,
                expected_position_id=expected_position_id,
                ordered_route_equal_count=sum(
                    route.ordered_equal for route in layer_routes
                ),
                set_route_equal_count=sum(
                    route.set_equal for route in layer_routes
                ),
                routing_weights_finite_count=(
                    routing_weights_finite_count
                ),
                routing_weights_close_count=sum(
                    route.routing_weights_close for route in layer_routes
                ),
                routing_weights_max_abs=(
                    max(
                        route.routing_weights_max_abs
                        for route in layer_routes
                        if route.routing_weights_max_abs is not None
                    )
                    if finite_route_weights
                    else None
                ),
                routing_weights_mean_abs=(
                    sum(
                        route.routing_weights_mean_abs
                        * len(route.aligned_expert_ids)
                        for route in layer_routes
                        if route.routing_weights_mean_abs is not None
                    )
                    / aligned_weight_count
                    if finite_route_weights and aligned_weight_count
                    else None
                ),
                routes=layer_routes,
            )
        )
        del causal_outputs, reference_prefix_cache

    nodes = tuple(node_results)
    pristine_after_references_exact = _prompt_cache_clone_is_exact(
        pristine_prefix_guard,
        pristine_prefix_cache,
        expected_layers=layer_count,
    )
    pristine_prefix_cache_unchanged = (
        pristine_after_tree_exact and pristine_after_references_exact
    )
    del prefill, pristine_prefix_cache, pristine_prefix_guard
    argmax_equal_count = sum(node.argmax_equal for node in nodes)
    logits_finite_count = sum(node.logits_finite for node in nodes)
    logits_close_count = sum(node.logits_close for node in nodes)
    route_equal_count = sum(node.route_equal_count for node in nodes)
    route_comparison_count = sum(
        node.route_comparison_count for node in nodes
    )
    ordered_route_equal_count = sum(
        node.ordered_route_equal_count for node in nodes
    )
    set_route_equal_count = sum(
        node.set_route_equal_count for node in nodes
    )
    routing_weights_finite_count = sum(
        node.routing_weights_finite_count for node in nodes
    )
    routing_weights_close_count = sum(
        node.routing_weights_close_count for node in nodes
    )
    decision_and_route_passed = (
        argmax_equal_count == len(nodes)
        and route_equal_count == route_comparison_count
    )
    all_logits_finite = logits_finite_count == len(nodes)
    prefix_cache_clone_exact = (
        pristine_prefix_guard_exact
        and tree_prefix_cache_clone_exact
        and reference_prefix_cache_clone_exact_count == len(nodes)
        and pristine_prefix_cache_unchanged
    )
    mask_passed_count = sum(node.mask_passed for node in nodes)
    position_passed_count = sum(node.position_passed for node in nodes)
    tree_verifier_token_ids = tuple(
        node.tree_argmax_token_id for node in nodes
    )
    reference_verifier_token_ids = tuple(
        node.reference_argmax_token_id for node in nodes
    )
    tree_accepted_node_ids = tree.accepted_path(
        torch.tensor((tree_verifier_token_ids,), dtype=torch.long)
    )
    reference_accepted_node_ids = tree.accepted_path(
        torch.tensor((reference_verifier_token_ids,), dtype=torch.long)
    )
    accepted_path_equal = (
        tree_accepted_node_ids == reference_accepted_node_ids
    )
    tree_verification_positions = (0,) + tree_accepted_node_ids
    reference_verification_positions = (0,) + reference_accepted_node_ids
    verification_token_ids = tuple(
        int(value) for value in verification_ids[0].tolist()
    )
    tree_committed_token_ids = tuple(
        verification_token_ids[position]
        for position in tree_verification_positions
    )
    reference_committed_token_ids = tuple(
        verification_token_ids[position]
        for position in reference_verification_positions
    )
    committed_tokens_equal = (
        tree_committed_token_ids == reference_committed_token_ids
    )
    tree_next_pending_seed_token_id = tree_verifier_token_ids[
        tree_verification_positions[-1]
    ]
    reference_next_pending_seed_token_id = reference_verifier_token_ids[
        reference_verification_positions[-1]
    ]
    next_pending_seed_equal = (
        tree_next_pending_seed_token_id
        == reference_next_pending_seed_token_id
    )
    semantic_passed = (
        prefix_cache_clone_exact
        and mask_passed_count == len(nodes)
        and position_passed_count == len(nodes)
        and all_logits_finite
        and argmax_equal_count == len(nodes)
        and accepted_path_equal
        and committed_tokens_equal
        and next_pending_seed_equal
    )
    diagnostic_complete = (
        prefix_cache_clone_exact
        and mask_passed_count == len(nodes)
        and position_passed_count == len(nodes)
        and len(nodes) == verification_rows
        and route_comparison_count == len(nodes) * layer_count
        and routing_weights_finite_count == route_comparison_count
        and all_logits_finite
    )
    all_routing_weights_finite = (
        routing_weights_finite_count == route_comparison_count
    )
    aligned_weight_count = sum(
        len(route.aligned_expert_ids)
        for node in nodes
        for route in node.routes
    )
    return TreeVerificationEquivalence(
        passed=semantic_passed,
        decision_and_route_passed=decision_and_route_passed,
        semantic_passed=semantic_passed,
        diagnostic_complete=diagnostic_complete,
        reference_mode="exact_cloned_prefix_cached_linear_suffix",
        pending_seed_token_id=int(pending_seed[0, 0].item()),
        verification_token_ids=verification_token_ids,
        node_count=len(nodes),
        layer_count=layer_count,
        experts_per_token=experts_per_token,
        argmax_equal_count=argmax_equal_count,
        logits_finite_count=logits_finite_count,
        logits_close_count=logits_close_count,
        route_equal_count=route_equal_count,
        route_comparison_count=route_comparison_count,
        logits_atol=logits_atol,
        logits_rtol=logits_rtol,
        logits_max_abs=(
            max(
                node.logits_max_abs
                for node in nodes
                if node.logits_max_abs is not None
            )
            if all_logits_finite
            else None
        ),
        logits_mean_abs=(
            sum(
                node.logits_mean_abs
                for node in nodes
                if node.logits_mean_abs is not None
            )
            / len(nodes)
            if all_logits_finite
            else None
        ),
        tree_min_top1_margin=(
            min(
                node.tree_top1_margin
                for node in nodes
                if node.tree_top1_margin is not None
            )
            if all_logits_finite
            else None
        ),
        reference_min_top1_margin=(
            min(
                node.reference_top1_margin
                for node in nodes
                if node.reference_top1_margin is not None
            )
            if all_logits_finite
            else None
        ),
        top1_margin_max_abs_difference=(
            max(
                node.top1_margin_abs_difference
                for node in nodes
                if node.top1_margin_abs_difference is not None
            )
            if all_logits_finite
            else None
        ),
        prefix_cache_clone_exact=prefix_cache_clone_exact,
        tree_prefix_cache_clone_exact=tree_prefix_cache_clone_exact,
        reference_prefix_cache_clone_exact_count=(
            reference_prefix_cache_clone_exact_count
        ),
        pristine_prefix_cache_unchanged=(
            pristine_prefix_cache_unchanged
        ),
        mask_passed_count=mask_passed_count,
        position_passed_count=position_passed_count,
        tree_attention_shape=tuple(int(value) for value in tree_attention.shape),
        tree_attention_dtype=str(tree_attention.dtype).removeprefix("torch."),
        tree_verifier_token_ids=tree_verifier_token_ids,
        reference_verifier_token_ids=reference_verifier_token_ids,
        tree_accepted_node_ids=tree_accepted_node_ids,
        reference_accepted_node_ids=reference_accepted_node_ids,
        accepted_path_equal=accepted_path_equal,
        tree_committed_token_ids=tree_committed_token_ids,
        reference_committed_token_ids=reference_committed_token_ids,
        committed_tokens_equal=committed_tokens_equal,
        tree_next_pending_seed_token_id=(
            tree_next_pending_seed_token_id
        ),
        reference_next_pending_seed_token_id=(
            reference_next_pending_seed_token_id
        ),
        next_pending_seed_equal=next_pending_seed_equal,
        ordered_route_equal_count=ordered_route_equal_count,
        set_route_equal_count=set_route_equal_count,
        routing_weights_finite_count=routing_weights_finite_count,
        routing_weights_close_count=routing_weights_close_count,
        routing_weights_atol=routing_weights_atol,
        routing_weights_rtol=routing_weights_rtol,
        routing_weights_max_abs=(
            max(
                route.routing_weights_max_abs
                for node in nodes
                for route in node.routes
                if route.routing_weights_max_abs is not None
            )
            if all_routing_weights_finite
            else None
        ),
        routing_weights_mean_abs=(
            sum(
                route.routing_weights_mean_abs
                * len(route.aligned_expert_ids)
                for node in nodes
                for route in node.routes
                if route.routing_weights_mean_abs is not None
            )
            / aligned_weight_count
            if all_routing_weights_finite and aligned_weight_count
            else None
        ),
        nodes=nodes,
    )


__all__ = [
    "LayerRouteMismatch",
    "LayerRoutingEquivalence",
    "NodeZeroNumericalReport",
    "NodeVerificationEquivalence",
    "TreeVerificationEquivalence",
    "compare_tree_verification_to_causal_paths",
    "diagnose_tree_node_zero_numerics",
]
