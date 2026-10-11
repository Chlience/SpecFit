
from __future__ import annotations

import inspect
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import torch
from speculators.models.eagle3 import Eagle3DraftModel
from torch import nn
from transformers import AutoConfig

from .eagle3_attention import install_eagle3_flex_attention_compatibility
from .eagle3_types import (
    ChainDrafter,
    Eagle3GenerationResult,
    GreedyChainTraceSink,
    acceptance_prefix_length,
)

QWEN3_EAGLE3_LAYER_INDICES = (1, 23, 44)
EAGLE3_MAX_COMMITTED_TOKENS = 512
EAGLE3_VALIDATED_TREE_DEPTH = 5
EAGLE3_DEFAULT_TREE_TOP_K = 2
EAGLE3_DEFAULT_TREE_DEPTH = 3
EAGLE3_DEFAULT_TREE_NODE_BUDGET = 7
EAGLE3_DEPTH5_MAX_FED_TOKENS = (
    EAGLE3_MAX_COMMITTED_TOKENS - 1 - EAGLE3_VALIDATED_TREE_DEPTH
)
EAGLE3_DEPTH5_MAX_DRAFTER_KV_TOKENS = (
    EAGLE3_DEPTH5_MAX_FED_TOKENS
    + EAGLE3_DEFAULT_TREE_TOP_K * (EAGLE3_VALIDATED_TREE_DEPTH - 1)
)
EAGLE3_VALIDATED_TOTAL_TOKENS = EAGLE3_MAX_COMMITTED_TOKENS
_COMMITTED_TOKEN_LIMIT = ContextVar(
    "eagle3_committed_token_limit", default=EAGLE3_MAX_COMMITTED_TOKENS
)


def _committed_token_limit() -> int:
    return _COMMITTED_TOKEN_LIMIT.get()


@contextmanager
def _experimental_2048_token_budget():
    """Opt in only inside the independent long-sequence validation entry.

    Legacy constants and runners retain 512. Context-local state is restored on
    success or failure and covers target, drafter and warmup checks together.
    This opt-in alone is not evidence of full-model validation.
    """
    token = _COMMITTED_TOKEN_LIMIT.set(2048)
    try:
        yield
    finally:
        _COMMITTED_TOKEN_LIMIT.reset(token)


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def _validate_committed_token_budget(
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> None:
    total_tokens = int(input_ids.shape[1]) + max_new_tokens
    if total_tokens > _committed_token_limit():
        raise ValueError(
            "prompt and generation budget exceed the validated "
            f"{_committed_token_limit()}-token committed-length contract"
        )


def _normalized_eos_token_ids(target: nn.Module) -> tuple[int, ...]:

    generation_config = getattr(target, "generation_config", None)
    value = getattr(generation_config, "eos_token_id", None)
    if value is None:
        value = getattr(getattr(target, "config", None), "eos_token_id", None)
    if value is None:
        return ()
    if isinstance(value, int) and not isinstance(value, bool):
        values = (value,)
    elif isinstance(value, torch.Tensor):
        values = tuple(int(item) for item in value.reshape(-1).tolist())
    else:
        values = tuple(int(item) for item in value)
    if any(item < 0 for item in values):
        raise ValueError("EOS token IDs must be non-negative")
    return tuple(dict.fromkeys(values))


def _first_eos_position(
    token_ids: torch.Tensor,
    eos_token_ids: tuple[int, ...],
) -> int | None:
    if not eos_token_ids:
        return None
    matches = torch.zeros_like(token_ids, dtype=torch.bool)
    for token_id in eos_token_ids:
        matches.logical_or_(token_ids.eq(token_id))
    positions = matches[0].nonzero(as_tuple=False)
    if positions.numel() == 0:
        return None
    return int(positions[0, 0].item())


class AuxHiddenStateCollector:

    def __init__(
        self,
        target: nn.Module,
        *,
        layer_indices: tuple[int, ...] = QWEN3_EAGLE3_LAYER_INDICES,
    ) -> None:
        if not layer_indices or len(set(layer_indices)) != len(layer_indices):
            raise ValueError("layer_indices must be non-empty and unique")
        layers = target.model.layers
        if min(layer_indices) < 0 or max(layer_indices) >= len(layers):
            raise ValueError("layer_indices are outside the target model")
        self._layers = layers
        self._layer_indices = layer_indices
        self._outputs: dict[int, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> AuxHiddenStateCollector:
        if self._handles:
            raise RuntimeError("collector is already active")
        for layer_id in self._layer_indices:
            self._handles.append(
                self._layers[layer_id].register_forward_hook(
                    self._make_hook(layer_id)
                )
            )
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._outputs.clear()

    def _make_hook(self, layer_id: int):
        def capture(
            module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor | tuple[torch.Tensor, ...],
        ) -> None:
            del module, inputs
            value = output[0] if isinstance(output, tuple) else output
            self._outputs[layer_id] = value.detach()

        return capture

    def take(self) -> torch.Tensor:
        missing = sorted(set(self._layer_indices) - self._outputs.keys())
        if missing:
            raise RuntimeError(f"target forward missed hidden layers {missing}")
        result = torch.cat(
            [self._outputs[layer_id] for layer_id in self._layer_indices],
            dim=-1,
        )
        self._outputs.clear()
        return result


class Eagle3ChainDrafter:

    def __init__(
        self,
        model: nn.Module,
        *,
        use_torch_compile: bool = False,
        layer_indices: tuple[int, ...] = QWEN3_EAGLE3_LAYER_INDICES,
    ) -> None:
        if not layer_indices or len(set(layer_indices)) != len(layer_indices):
            raise ValueError("layer_indices must be non-empty and unique")
        if min(layer_indices) < 0:
            raise ValueError("layer_indices must be non-negative")
        install_eagle3_flex_attention_compatibility()
        self.model = model
        self.use_torch_compile = use_torch_compile
        self.layer_indices = layer_indices
        self._eager_forward = None
        if not use_torch_compile:
            class_forward = type(model).forward
            raw_forward = inspect.unwrap(class_forward)
            if raw_forward is class_forward:
                raw_forward = getattr(
                    class_forward,
                    "_torchdynamo_orig_callable",
                    class_forward,
                )
            self._eager_forward = raw_forward.__get__(model, type(model))
        self._validate_vocabulary_contract()

    @classmethod
    def from_pretrained(
        cls,
        draft_root: str | Path,
        *,
        target_root: str | Path,
        device: torch.device | str,
        use_torch_compile: bool = False,
        expected_target_repo_id: str | None = None,
    ) -> Eagle3ChainDrafter:
        config = Eagle3DraftModel.config_class.from_pretrained(
            draft_root,
            local_files_only=True,
        )
        declared_target = config.speculators_config.verifier.name_or_path
        if declared_target is not None:
            declared_name = Path(str(declared_target)).name
            actual_name = Path(target_root).resolve().name
            if declared_name != actual_name:
                raise ValueError(
                    "EAGLE-3 checkpoint declares a different target model: "
                    f"{declared_target!r} != {actual_name!r}"
                )
        if (
            expected_target_repo_id is not None
            and declared_target != expected_target_repo_id
        ):
            raise ValueError(
                "EAGLE-3 verifier identity differs from the model lock: "
                f"{declared_target!r} != {expected_target_repo_id!r}"
            )
        target_config = AutoConfig.from_pretrained(
            target_root,
            local_files_only=True,
        )
        draft_target_config = config.transformer_layer_config
        for field in ("vocab_size", "hidden_size"):
            if getattr(target_config, field) != getattr(
                draft_target_config, field
            ):
                raise ValueError(
                    f"EAGLE-3 {field} differs from the target config"
                )
        configured_layer_ids = config.eagle_aux_hidden_state_layer_ids
        layer_ids = tuple(
            configured_layer_ids
            if configured_layer_ids is not None
            else QWEN3_EAGLE3_LAYER_INDICES
        )
        if (
            not layer_ids
            or len(set(layer_ids)) != len(layer_ids)
            or min(layer_ids) < 0
            or max(layer_ids) >= target_config.num_hidden_layers
        ):
            raise ValueError("EAGLE-3 auxiliary layer IDs differ from target")
        config.speculators_config.verifier.name_or_path = str(
            Path(target_root).resolve()
        )
        model = Eagle3DraftModel.from_pretrained(
            draft_root,
            config=config,
            dtype=torch.bfloat16,
            local_files_only=True,
        )
        model.to(device)
        model.requires_grad_(False)
        model.eval()
        return cls(
            model,
            use_torch_compile=use_torch_compile,
            layer_indices=layer_ids,
        )

    def _validate_vocabulary_contract(self) -> None:

        if not hasattr(self.model, "embed_tokens") or not hasattr(
            self.model, "lm_head"
        ):
            raise ValueError("EAGLE-3 model has no vocabulary modules")
        self.target_vocab_size = int(self.model.embed_tokens.num_embeddings)
        self.draft_vocab_size = int(self.model.lm_head.out_features)
        t2d = self.model.t2d
        d2t = self.model.d2t
        if (t2d is None) != (d2t is None):
            raise ValueError("EAGLE-3 t2d and d2t must appear together")
        if t2d is None:
            if self.draft_vocab_size != self.target_vocab_size:
                raise ValueError(
                    "full-vocabulary EAGLE-3 head differs from target vocab"
                )
            self.vocabulary_mode = "full"
            self._target_to_draft = torch.arange(
                self.target_vocab_size,
                dtype=torch.long,
                device=self.model.lm_head.weight.device,
            )
            return
        if t2d.ndim != 1 or t2d.shape[0] != self.target_vocab_size:
            raise ValueError("EAGLE-3 t2d shape differs from target vocab")
        if t2d.dtype != torch.bool:
            raise ValueError("EAGLE-3 t2d mapping must be a boolean mask")
        if d2t.ndim != 1 or d2t.shape[0] != self.draft_vocab_size:
            raise ValueError("EAGLE-3 d2t shape differs from draft vocab")
        if d2t.dtype not in _INTEGER_DTYPES:
            raise ValueError("EAGLE-3 d2t mapping must contain integers")
        draft_ids = torch.arange(
            self.draft_vocab_size,
            dtype=torch.long,
            device=d2t.device,
        )
        mapped = draft_ids + d2t.to(torch.long)
        if bool(
            torch.logical_or(
                mapped.lt(0), mapped.ge(self.target_vocab_size)
            ).any().item()
        ):
            raise ValueError("EAGLE-3 d2t mapping exceeds target vocab")
        if mapped.unique().numel() != self.draft_vocab_size:
            raise ValueError("EAGLE-3 d2t mapping contains duplicates")
        expected_support = torch.zeros_like(t2d)
        expected_support[mapped] = True
        if not torch.equal(t2d, expected_support):
            raise ValueError("EAGLE-3 t2d and d2t mappings disagree")
        inverse = torch.full(
            (self.target_vocab_size,),
            -1,
            dtype=torch.long,
            device=mapped.device,
        )
        inverse[mapped] = draft_ids
        self._target_to_draft = inverse
        self.vocabulary_mode = "compressed"

    def target_token_support(self, target_ids: torch.Tensor) -> torch.Tensor:

        if target_ids.dtype not in _INTEGER_DTYPES:
            raise ValueError("target token IDs must contain integers")
        if target_ids.numel() and bool(
            torch.logical_or(
                target_ids.lt(0), target_ids.ge(self.target_vocab_size)
            ).any().item()
        ):
            raise ValueError("target token IDs exceed the target vocabulary")
        t2d = self.model.t2d
        if t2d is None:
            return torch.ones_like(target_ids, dtype=torch.bool)
        return t2d[target_ids]

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:

        if draft_ids.dtype not in _INTEGER_DTYPES:
            raise ValueError("draft token IDs must contain integers")
        if draft_ids.numel() and bool(
            torch.logical_or(
                draft_ids.lt(0), draft_ids.ge(self.draft_vocab_size)
            ).any().item()
        ):
            raise ValueError("draft token IDs exceed the draft vocabulary")
        d2t = self.model.d2t
        if d2t is None:
            return draft_ids
        return draft_ids + d2t[draft_ids]

    def map_target_to_draft(self, target_ids: torch.Tensor) -> torch.Tensor:

        support = self.target_token_support(target_ids)
        if not bool(support.all().item()):
            raise ValueError("target token is outside the draft vocabulary")
        return self._target_to_draft[target_ids]

    def propose(
        self,
        *,
        fed_ids: torch.Tensor,
        aux_history: torch.Tensor,
        pending_seed: torch.Tensor,
        draft_length: int,
    ) -> torch.Tensor:
        if fed_ids.ndim != 2 or fed_ids.shape[0] != 1:
            raise ValueError("EAGLE-3 chain currently requires batch size 1")
        if aux_history.shape[:2] != fed_ids.shape:
            raise ValueError("aux_history must align with fed_ids")
        if pending_seed.shape != (1, 1):
            raise ValueError("pending_seed must have shape [1, 1]")
        if draft_length <= 0:
            raise ValueError("draft_length must be positive")
        if (
            fed_ids.shape[1] + 1 + draft_length
            > _committed_token_limit()
        ):
            raise NotImplementedError(
                "EAGLE-3 draft request exceeds the validated "
                f"{_committed_token_limit()}-token committed-length contract"
            )

        draft_input_ids = torch.cat((fed_ids[:, 1:], pending_seed), dim=1)
        if self._eager_forward is None:
            steps = self.model(
                aux_history,
                draft_input_ids,
                ttt_steps=draft_length,
            )
        else:
            steps = self._eager_forward(
                aux_history,
                draft_input_ids,
                ttt_steps=draft_length,
            )
        if len(steps) != draft_length:
            raise RuntimeError("EAGLE-3 returned an unexpected step count")
        draft_ids = torch.stack([step[:, -1] for step in steps], dim=1)
        return self.map_draft_to_target(draft_ids)


def target_greedy_generate(
    target: nn.Module,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    max_new_tokens: int,
    trace: GreedyChainTraceSink | None = None,
    on_prefill_complete: Callable[[], None] | None = None,
    stop_on_eos: bool = True,
) -> Eagle3GenerationResult:

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("target greedy currently requires batch size 1")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    _validate_committed_token_budget(input_ids, max_new_tokens)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    elif attention_mask.shape != input_ids.shape or not attention_mask.all():
        raise ValueError("target greedy requires an unpadded attention mask")

    outputs = target(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        logits_to_keep=1,
    )
    eos_token_ids = _normalized_eos_token_ids(target) if stop_on_eos else ()
    next_token = outputs.logits.argmax(dim=-1)
    generated = [next_token]
    target_forward_calls = 1
    eos_position = _first_eos_position(next_token, eos_token_ids)
    if on_prefill_complete is not None:
        on_prefill_complete()

    for _ in range(max_new_tokens - 1 if eos_position is None else 0):
        attention_mask = torch.cat(
            (attention_mask, torch.ones_like(next_token)),
            dim=1,
        )
        outputs = target(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=outputs.past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
        next_token = outputs.logits.argmax(dim=-1)
        generated.append(next_token)
        target_forward_calls += 1
        eos_position = _first_eos_position(next_token, eos_token_ids)
        if eos_position is not None:
            break

    tokens = torch.cat(generated, dim=1)
    stopped_on_eos = _first_eos_position(tokens, eos_token_ids) is not None
    result = Eagle3GenerationResult(
        tokens=tokens,
        cycles=0,
        proposed_draft_tokens=0,
        accepted_draft_tokens=0,
        target_forward_calls=target_forward_calls,
        accepted_per_cycle=(),
        stopped_on_eos=stopped_on_eos,
        stop_token_id=(int(tokens[0, -1].item()) if stopped_on_eos else None),
    )
    if trace is not None:
        trace.finish_generation(
            output_tokens=result.tokens,
            target_forward_calls=result.target_forward_calls,
            stopped_on_eos=result.stopped_on_eos,
            stop_token_id=result.stop_token_id,
        )
    return result


class GreedyChainSpeculator:

    def __init__(
        self,
        target: nn.Module,
        drafter: ChainDrafter,
        *,
        layer_indices: tuple[int, ...] | None = None,
    ) -> None:
        self.target = target
        self.drafter = drafter
        resolved_layer_indices = (
            getattr(drafter, "layer_indices", QWEN3_EAGLE3_LAYER_INDICES)
            if layer_indices is None
            else layer_indices
        )
        if (
            not resolved_layer_indices
            or len(set(resolved_layer_indices)) != len(resolved_layer_indices)
        ):
            raise ValueError("layer_indices must be non-empty and unique")
        self.layer_indices = tuple(resolved_layer_indices)

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int,
        draft_length: int = 3,
        trace: GreedyChainTraceSink | None = None,
        on_prefill_complete: Callable[[], None] | None = None,
        stop_on_eos: bool = True,
    ) -> Eagle3GenerationResult:

        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("speculation currently requires batch size 1")
        if max_new_tokens <= 0 or draft_length <= 0:
            raise ValueError("token counts must be positive")
        _validate_committed_token_budget(input_ids, max_new_tokens)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        elif attention_mask.shape != input_ids.shape or not attention_mask.all():
            raise ValueError("P0 EAGLE-3 requires an unpadded attention mask")

        eos_token_ids = (
            _normalized_eos_token_ids(self.target) if stop_on_eos else ()
        )
        stopped_on_eos = False
        stop_token_id = None
        with AuxHiddenStateCollector(
            self.target, layer_indices=self.layer_indices
        ) as collector:
            outputs = self.target(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                logits_to_keep=1,
            )
            aux_history = collector.take()
            cache = outputs.past_key_values
            pending_seed = outputs.logits.argmax(dim=-1)
            if on_prefill_complete is not None:
                on_prefill_complete()
            fed_ids = input_ids
            fed_attention = attention_mask
            generated = []
            accepted_per_cycle = []
            emitted = 0
            proposed_total = 0

            while emitted < max_new_tokens:
                remaining = max_new_tokens - emitted
                if _first_eos_position(pending_seed, eos_token_ids) is not None:
                    generated.append(pending_seed)
                    emitted += 1
                    stopped_on_eos = True
                    stop_token_id = int(pending_seed[0, 0].item())
                    break
                if remaining == 1:
                    generated.append(pending_seed)
                    emitted += 1
                    break
                cycle_draft_length = min(draft_length, remaining - 1)
                drafts = self.drafter.propose(
                    fed_ids=fed_ids,
                    aux_history=aux_history,
                    pending_seed=pending_seed,
                    draft_length=cycle_draft_length,
                )
                verify_ids = torch.cat((pending_seed, drafts), dim=1)
                base_length = cache.get_seq_length()
                verify_attention = torch.cat(
                    (fed_attention, torch.ones_like(verify_ids)),
                    dim=1,
                )
                iteration = len(accepted_per_cycle)
                if trace is not None:
                    trace.begin_cycle(
                        iteration=iteration,
                        base_kv_length=base_length,
                        verification_token_ids=verify_ids,
                    )
                try:
                    outputs = self.target(
                        input_ids=verify_ids,
                        attention_mask=verify_attention,
                        past_key_values=cache,
                        use_cache=True,
                        logits_to_keep=0,
                    )
                    query_aux = collector.take()
                    verifier_tokens = outputs.logits.argmax(dim=-1)
                    accepted = acceptance_prefix_length(
                        verifier_tokens[:, :cycle_draft_length], drafts
                    )
                    commit_count = 1 + accepted
                    committed = verify_ids[:, :commit_count]
                    eos_position = _first_eos_position(
                        committed, eos_token_ids
                    )
                    if eos_position is not None:
                        commit_count = eos_position + 1
                        accepted = commit_count - 1
                        committed = verify_ids[:, :commit_count]
                        stopped_on_eos = True
                        stop_token_id = int(committed[0, -1].item())
                    generated.append(committed)
                    fed_ids = torch.cat((fed_ids, committed), dim=1)
                    aux_history = torch.cat(
                        (aux_history, query_aux[:, :commit_count]),
                        dim=1,
                    )
                    fed_attention = verify_attention[
                        :, : base_length + commit_count
                    ]
                    cache.crop(base_length + commit_count)
                    pending_seed = verifier_tokens[
                        :, accepted : accepted + 1
                    ]
                    accepted_per_cycle.append(accepted)
                    proposed_total += cycle_draft_length
                    emitted += commit_count
                    if trace is not None:
                        trace.end_cycle(
                            verifier_tokens=verifier_tokens,
                            accepted_draft_tokens=accepted,
                            committed_token_ids=committed,
                            kv_length_after=base_length + commit_count,
                        )
                    if stopped_on_eos:
                        break
                except BaseException:
                    if trace is not None:
                        trace.abort_cycle()
                    raise

        accepted_total = sum(accepted_per_cycle)
        cycles = len(accepted_per_cycle)
        result = Eagle3GenerationResult(
            tokens=torch.cat(generated, dim=1)[:, :max_new_tokens],
            cycles=cycles,
            proposed_draft_tokens=proposed_total,
            accepted_draft_tokens=accepted_total,
            target_forward_calls=1 + cycles,
            accepted_per_cycle=tuple(accepted_per_cycle),
            stopped_on_eos=stopped_on_eos,
            stop_token_id=stop_token_id,
        )
        if trace is not None:
            trace.finish_generation(
                output_tokens=result.tokens,
                target_forward_calls=result.target_forward_calls,
                stopped_on_eos=result.stopped_on_eos,
                stop_token_id=result.stop_token_id,
            )
        return result
