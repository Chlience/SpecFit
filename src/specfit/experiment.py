
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import Any

from .trace import RuntimeEventKind, RuntimeEventTrace


PERSISTENT_EXPERIMENT_SCHEMA_VERSION = 1
PERSISTENT_EXPERIMENT_ARTIFACT_TYPE = "r3.5_persistent_runner"

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_REQUIRED_ENVIRONMENT_FIELDS = (
    "git_commit",
    "image_id",
    "server",
    "device",
)
_DETERMINISTIC_RUN_FIELDS = (
    "output_token_ids",
    "accepted_per_cycle",
    "proposed_draft_tokens",
    "accepted_draft_tokens",
    "target_forward_calls",
    "stopped_on_eos",
    "stop_token_id",
    "h2d_submission_order_sha256",
    "lru_victim_order_sha256",
    "final_cache_state_sha256",
    "cache_stats",
    "prefill_cache_stats",
)
_CACHE_STAT_FIELDS = (
    "h2d_submissions",
    "resident_hits",
    "inflight_joins",
    "evictions",
    "pending_use_guards",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def canonical_runtime_event_digests(
    events: Sequence[RuntimeEventTrace],
) -> dict[str, str]:

    logical = tuple(
        (
            event.kind.value,
            event.layer_id,
            event.expert_id,
            event.slot_id,
            event.generation,
            event.submission_sequence,
            event.victim_expert_id,
        )
        for event in events
    )
    h2d = tuple(
        (
            event.layer_id,
            event.expert_id,
            event.slot_id,
            event.generation,
            event.submission_sequence,
        )
        for event in events
        if event.kind is RuntimeEventKind.H2D_SUBMIT
    )
    victims = tuple(
        (
            event.layer_id,
            event.expert_id,
            event.slot_id,
            event.generation,
            event.victim_expert_id,
        )
        for event in events
        if event.kind is RuntimeEventKind.EVICT
    )
    return {
        "logical_event_sha256": canonical_json_sha256(logical),
        "h2d_submission_order_sha256": canonical_json_sha256(h2d),
        "lru_victim_order_sha256": canonical_json_sha256(victims),
    }


def canonical_cache_state(layers: Sequence[object]) -> tuple[object, ...]:

    result = []
    for layer_index, layer in enumerate(layers):
        layer_id = int(getattr(layer, "layer_id", layer_index))
        slots = []
        for slot in getattr(layer, "slots"):
            expert = getattr(slot, "expert_id")
            if expert is not None and not isinstance(expert, int):
                expert = int(getattr(expert, "expert_id"))
            slots.append(
                (
                    int(getattr(slot, "slot_id")),
                    expert,
                    int(getattr(slot, "generation")),
                    int(getattr(slot, "last_used_sequence")),
                )
            )
        result.append(
            (
                layer_id,
                int(getattr(layer, "lru_sequence")),
                int(getattr(layer, "submission_sequence")),
                tuple(slots),
            )
        )
    return tuple(result)


def canonical_cache_state_sha256(layers: Sequence[object]) -> str:

    return canonical_json_sha256(canonical_cache_state(layers))


def _require_mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return value


def _require_sequence(name: str, value: object) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return value


def _require_nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_integer(name: str, value: object, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _require_boolean(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _require_sha256(name: str, value: object) -> str:
    digest = _require_nonempty_string(name, value)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return digest


def _require_identity(name: str, value: object) -> None:
    if isinstance(value, str):
        _require_nonempty_string(name, value)
        return
    identity = _require_mapping(name, value)
    if not identity:
        raise ValueError(f"{name} must be non-empty")
    for key, item in identity.items():
        _require_nonempty_string(f"{name}.{key}", item)


def _require_token_ids(name: str, value: object) -> tuple[int, ...]:
    values = _require_sequence(name, value)
    result = tuple(
        _require_integer(f"{name}[{index}]", item)
        for index, item in enumerate(values)
    )
    return result


def _require_json_mapping(name: str, value: object) -> Mapping[str, Any]:
    result = _require_mapping(name, value)
    if not result:
        raise ValueError(f"{name} must be non-empty")
    try:
        _canonical_json_bytes(result)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON values") from error
    return result


def _require_cache_stats(name: str, value: object) -> Mapping[str, Any]:
    result = _require_mapping(name, value)
    if set(result) != set(_CACHE_STAT_FIELDS):
        raise ValueError(f"{name} must contain the five cache counters")
    for field in _CACHE_STAT_FIELDS:
        _require_integer(f"{name}.{field}", result[field])
    return result


def validate_persistent_experiment_artifact(
    artifact: Mapping[str, object],
) -> None:

    root = _require_mapping("artifact", artifact)
    version = _require_integer("schema_version", root.get("schema_version"))
    if version != PERSISTENT_EXPERIMENT_SCHEMA_VERSION:
        raise ValueError(f"unsupported persistent experiment schema {version}")
    if root.get("artifact_type") != PERSISTENT_EXPERIMENT_ARTIFACT_TYPE:
        raise ValueError("artifact_type does not identify the R3.5 runner")
    _require_nonempty_string("session_id", root.get("session_id"))

    environment = _require_mapping("environment", root.get("environment"))
    for field in _REQUIRED_ENVIRONMENT_FIELDS:
        _require_nonempty_string(
            f"environment.{field}",
            environment.get(field),
        )

    models = _require_mapping("models", root.get("models"))
    for field in ("target", "drafter"):
        _require_identity(f"models.{field}", models.get(field))

    settings = _require_mapping("settings", root.get("settings"))
    draft_lengths = tuple(
        _require_integer(f"settings.draft_lengths[{index}]", item)
        for index, item in enumerate(
            _require_sequence(
                "settings.draft_lengths",
                settings.get("draft_lengths"),
            )
        )
    )
    if draft_lengths != (0, 3):
        raise ValueError("settings.draft_lengths must equal [0, 3]")
    repeats = _require_integer("settings.repeats", settings.get("repeats"))
    if repeats != 2:
        raise ValueError("settings.repeats must equal 2")
    new_tokens = _require_integer(
        "settings.new_tokens",
        settings.get("new_tokens"),
        positive=True,
    )
    slots_per_layer = _require_integer(
        "settings.slots_per_layer",
        settings.get("slots_per_layer"),
        positive=True,
    )

    prompts = _require_sequence("prompts", root.get("prompts"))
    if len(prompts) != 2:
        raise ValueError("prompts must contain exactly two entries")
    prompt_ids: list[str] = []
    for index, value in enumerate(prompts):
        prompt = _require_mapping(f"prompts[{index}]", value)
        prompt_ids.append(
            _require_nonempty_string(
                f"prompts[{index}].prompt_id",
                prompt.get("prompt_id"),
            )
        )
        _require_nonempty_string(
            f"prompts[{index}].raw_prompt",
            prompt.get("raw_prompt"),
        )
        _require_nonempty_string(
            f"prompts[{index}].rendered_prompt",
            prompt.get("rendered_prompt"),
        )
        input_ids = _require_token_ids(
            f"prompts[{index}].input_ids",
            prompt.get("input_ids"),
        )
        attention_mask = _require_token_ids(
            f"prompts[{index}].attention_mask",
            prompt.get("attention_mask"),
        )
        if not input_ids or len(attention_mask) != len(input_ids):
            raise ValueError("prompt input IDs and attention mask must align")
        if any(value != 1 for value in attention_mask):
            raise ValueError("R3.5 prompts must use an unpadded attention mask")
        if len(input_ids) + new_tokens > 128:
            raise ValueError("prompt and generated tokens exceed EAGLE-3 limit")
        draft_vocabulary = _require_mapping(
            f"prompts[{index}].draft_vocabulary",
            prompt.get("draft_vocabulary"),
        )
        supported = _require_integer(
            f"prompts[{index}].draft_vocabulary.supported_output_tokens",
            draft_vocabulary.get("supported_output_tokens"),
        )
        evaluated = _require_integer(
            f"prompts[{index}].draft_vocabulary.evaluated_output_tokens",
            draft_vocabulary.get("evaluated_output_tokens"),
            positive=True,
        )
        coverage = draft_vocabulary.get("coverage")
        if (
            isinstance(coverage, bool)
            or not isinstance(coverage, (int, float))
            or float(coverage) != supported / evaluated
            or float(coverage) < 0.95
        ):
            raise ValueError("prompt draft vocabulary coverage must be at least 95%")
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("prompt_id values must be unique")

    load_counts = _require_mapping("load_counts", root.get("load_counts"))
    for field in ("target", "drafter"):
        count = _require_integer(
            f"load_counts.{field}",
            load_counts.get(field),
        )
        if count != 1:
            raise ValueError(f"load_counts.{field} must equal 1")

    runs = _require_sequence("runs", root.get("runs"))
    expected_keys = set(product(prompt_ids, draft_lengths, range(repeats)))
    if len(runs) != len(expected_keys):
        raise ValueError("runs must contain the complete eight-run matrix")

    run_ids: set[str] = set()
    request_ids: set[str] = set()
    trace_paths: set[str] = set()
    run_by_key: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for index, value in enumerate(runs):
        run = _require_mapping(f"runs[{index}]", value)
        execution_index = _require_integer(
            f"runs[{index}].execution_index",
            run.get("execution_index"),
        )
        if execution_index != index:
            raise ValueError("run execution indices must be contiguous")
        run_id = _require_nonempty_string(
            f"runs[{index}].run_id",
            run.get("run_id"),
        )
        if run_id in run_ids:
            raise ValueError("run_id values must be unique")
        run_ids.add(run_id)
        request_id = _require_nonempty_string(
            f"runs[{index}].request_id",
            run.get("request_id"),
        )
        if request_id in request_ids:
            raise ValueError("request_id values must be unique")
        request_ids.add(request_id)

        prompt_id = _require_nonempty_string(
            f"runs[{index}].prompt_id",
            run.get("prompt_id"),
        )
        draft_length = _require_integer(
            f"runs[{index}].draft_length",
            run.get("draft_length"),
        )
        repeat_index = _require_integer(
            f"runs[{index}].repeat_index",
            run.get("repeat_index"),
        )
        key = (prompt_id, draft_length, repeat_index)
        if key not in expected_keys:
            raise ValueError(f"runs[{index}] has an unexpected matrix key {key}")
        if key in run_by_key:
            raise ValueError(f"duplicate run matrix key {key}")
        run_by_key[key] = run

        output_token_ids = _require_token_ids(
            f"runs[{index}].output_token_ids",
            run.get("output_token_ids"),
        )
        stopped_on_eos = _require_boolean(
            f"runs[{index}].stopped_on_eos",
            run.get("stopped_on_eos", False),
        )
        stop_token_id = run.get("stop_token_id")
        if not 1 <= len(output_token_ids) <= new_tokens:
            raise ValueError(
                f"runs[{index}].output_token_ids exceeds the token budget"
            )
        if stopped_on_eos:
            stop_token_id = _require_integer(
                f"runs[{index}].stop_token_id", stop_token_id
            )
            if output_token_ids[-1] != stop_token_id:
                raise ValueError(
                    f"runs[{index}].stop_token_id differs from output tail"
                )
        elif stop_token_id is not None:
            raise ValueError(
                f"runs[{index}].stop_token_id requires EOS termination"
            )
        elif len(output_token_ids) != new_tokens:
            raise ValueError(
                f"runs[{index}].short output requires EOS termination"
            )
        _require_token_ids(
            f"runs[{index}].accepted_per_cycle",
            run.get("accepted_per_cycle"),
        )
        for field in (
            "proposed_draft_tokens",
            "accepted_draft_tokens",
            "target_forward_calls",
        ):
            _require_integer(f"runs[{index}].{field}", run.get(field))
        for field in (
            "h2d_submission_order_sha256",
            "lru_victim_order_sha256",
            "logical_event_sha256",
            "final_cache_state_sha256",
        ):
            _require_sha256(f"runs[{index}].{field}", run.get(field))
        _require_cache_stats(
            f"runs[{index}].cache_stats",
            run.get("cache_stats"),
        )
        _require_cache_stats(
            f"runs[{index}].prefill_cache_stats",
            run.get("prefill_cache_stats"),
        )
        _require_integer(
            f"runs[{index}].prefill_event_count",
            run.get("prefill_event_count"),
        )
        for field in ("h2d_submission_count", "lru_victim_count"):
            _require_integer(f"runs[{index}].{field}", run.get(field))
        final_cache_state = _require_sequence(
            f"runs[{index}].final_cache_state",
            run.get("final_cache_state"),
        )
        if canonical_json_sha256(final_cache_state) != run.get(
            "final_cache_state_sha256"
        ):
            raise ValueError(
                "final_cache_state_sha256 differs from final_cache_state"
            )

        reset = _require_mapping(f"runs[{index}].reset", run.get("reset"))
        layer_count = _require_integer(
            f"runs[{index}].reset.layer_count",
            reset.get("layer_count"),
            positive=True,
        )
        empty_slots = _require_integer(
            f"runs[{index}].reset.empty_slots",
            reset.get("empty_slots"),
        )
        if empty_slots != layer_count * slots_per_layer:
            raise ValueError("reset must expose every cache slot as empty")
        for field in (
            "pending_transfers",
            "active_leases",
            "pending_use_guards",
            "host_state_version_sum",
            "lru_sequence_sum",
            "submission_sequence_sum",
            "observer_event_count",
            "observer_next_sequence",
        ):
            if _require_integer(
                f"runs[{index}].reset.{field}", reset.get(field)
            ) != 0:
                raise ValueError(f"runs[{index}].reset.{field} must be zero")
        reset_stats = _require_cache_stats(
            f"runs[{index}].reset.cache_stats",
            reset.get("cache_stats"),
        )
        if any(reset_stats.values()):
            raise ValueError("reset cache counters must be zero")

        memory = _require_mapping(f"runs[{index}].memory", run.get("memory"))
        for field in (
            "rss_bytes",
            "cuda_allocated_bytes",
            "cuda_reserved_bytes",
        ):
            _require_integer(f"runs[{index}].memory.{field}", memory.get(field))

        trace = _require_mapping(f"runs[{index}].trace", run.get("trace"))
        trace_path = _require_nonempty_string(
            f"runs[{index}].trace.path",
            trace.get("path"),
        )
        if not Path(trace_path).is_absolute():
            raise ValueError("trace paths must be absolute")
        if trace_path in trace_paths:
            raise ValueError("trace paths must be unique")
        trace_paths.add(trace_path)
        _require_integer(
            f"runs[{index}].trace.size_bytes",
            trace.get("size_bytes"),
            positive=True,
        )
        _require_sha256(
            f"runs[{index}].trace.sha256",
            trace.get("sha256"),
        )
        _require_integer(
            f"runs[{index}].trace.record_count",
            trace.get("record_count"),
            positive=True,
        )

    if set(run_by_key) != expected_keys:
        raise ValueError("runs do not cover the complete experiment matrix")

    for prompt_id, draft_length in product(prompt_ids, draft_lengths):
        first = run_by_key[(prompt_id, draft_length, 0)]
        second = run_by_key[(prompt_id, draft_length, 1)]
        for field in _DETERMINISTIC_RUN_FIELDS:
            if first.get(field) != second.get(field):
                raise ValueError(
                    f"repeated runs differ for {prompt_id}, k={draft_length}: "
                    f"{field}"
                )

    for prompt_id in prompt_ids:
        reference = run_by_key[(prompt_id, 0, 0)]
        speculative = run_by_key[(prompt_id, 3, 0)]
        if reference["output_token_ids"] != speculative["output_token_ids"]:
            raise ValueError(f"k=0 and k=3 outputs differ for {prompt_id}")
        prefill_reference = reference["prefill_cache_stats"]
        for draft_length, repeat_index in product(draft_lengths, range(repeats)):
            current = run_by_key[
                (prompt_id, draft_length, repeat_index)
            ]["prefill_cache_stats"]
            if current != prefill_reference:
                raise ValueError(
                    f"prefill cache stats differ across runs for {prompt_id}"
                )

    memory_after_load = _require_mapping(
        "memory_after_model_load", root.get("memory_after_model_load")
    )
    for field in (
        "rss_bytes",
        "cuda_allocated_bytes",
        "cuda_reserved_bytes",
    ):
        _require_integer(
            f"memory_after_model_load.{field}",
            memory_after_load.get(field),
        )
    memory_limits = _require_mapping(
        "memory_limits", root.get("memory_limits")
    )
    growth_limit = _require_integer(
        "memory_limits.repeat_growth_bytes",
        memory_limits.get("repeat_growth_bytes"),
    )
    comparisons = []
    for prompt_id, draft_length in product(prompt_ids, draft_lengths):
        first_memory = _require_mapping(
            "repeat 0 memory",
            run_by_key[(prompt_id, draft_length, 0)]["memory"],
        )
        second_memory = _require_mapping(
            "repeat 1 memory",
            run_by_key[(prompt_id, draft_length, 1)]["memory"],
        )
        rss_growth = max(
            0,
            second_memory["rss_bytes"] - first_memory["rss_bytes"],
        )
        cuda_growth = max(
            0,
            second_memory["cuda_allocated_bytes"]
            - first_memory["cuda_allocated_bytes"],
        )
        comparisons.append(
            {
                "prompt_id": prompt_id,
                "draft_length": draft_length,
                "rss_growth_bytes": rss_growth,
                "cuda_allocated_growth_bytes": cuda_growth,
            }
        )
    last_memory = _require_mapping("last run memory", runs[-1]["memory"])
    session_growth = {
        "rss_growth_bytes": max(
            0,
            last_memory["rss_bytes"] - memory_after_load["rss_bytes"],
        ),
        "cuda_allocated_growth_bytes": max(
            0,
            last_memory["cuda_allocated_bytes"]
            - memory_after_load["cuda_allocated_bytes"],
        ),
    }
    maximum_growth = max(
        max(item["rss_growth_bytes"], item["cuda_allocated_growth_bytes"])
        for item in (*comparisons, session_growth)
    )
    if maximum_growth > growth_limit:
        raise ValueError("repeat memory growth exceeds the declared limit")
    memory_validation = _require_mapping(
        "memory_validation", root.get("memory_validation")
    )
    if (
        memory_validation.get("limit_bytes") != growth_limit
        or memory_validation.get("maximum_observed_growth_bytes")
        != maximum_growth
        or memory_validation.get("comparisons") != comparisons
        or memory_validation.get("session_growth") != session_growth
        or memory_validation.get("passed") is not True
    ):
        raise ValueError("memory_validation differs from per-run measurements")
