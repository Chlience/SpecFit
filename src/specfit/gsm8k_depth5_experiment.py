
from __future__ import annotations

import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


GSM8K_DEPTH5_SCHEMA_VERSION = 1
GSM8K_DEPTH5_ARTIFACT_TYPE = "gsm8k_eagle3_depth5_selector_matrix"
GSM8K_DEPTH5_TOTAL_TOKEN_BUDGET = 512
GSM8K_DEPTH5_TREE_TOP_K = 2
GSM8K_DEPTH5_TREE_MAX_DEPTH = 5
GSM8K_DEPTH5_MODES = (
    "target_only",
    "eagle3_tree_top2_depth5_budget7",
    "eagle3_tree_top2_depth5_budget11",
    "eagle3_tree_top2_depth5_hindsight_path_oracle",
)
GSM8K_DEPTH5_SPECULATIVE_MODES = GSM8K_DEPTH5_MODES[1:]

_MODE_NODE_BUDGETS = {
    "eagle3_tree_top2_depth5_budget7": 7,
    "eagle3_tree_top2_depth5_budget11": 11,
    "eagle3_tree_top2_depth5_hindsight_path_oracle": 11,
}
_REQUIRED_DATASET_FIELDS = ("repo_id", "revision", "filename", "category")
_REQUIRED_ENVIRONMENT_FIELDS = ("git_commit", "image_id", "server", "device")
_CACHE_COUNTER_FIELDS = (
    "h2d_submissions",
    "resident_hits",
    "inflight_joins",
    "evictions",
)
_CACHE_GAUGE_FIELDS = ("pending_use_guards",)
_CACHE_FIELDS = _CACHE_COUNTER_FIELDS + _CACHE_GAUGE_FIELDS
_PHASE_TIMING_FIELDS = (
    "tree_build_seconds",
    "selection_seconds",
    "target_verification_seconds",
    "tree_cycle_seconds",
)
_CYCLE_WORK_FIELDS = (
    "actual_tree_depth_per_cycle",
    "pool_correct_depth_per_cycle",
    "path_score_calls_per_cycle",
    "draft_transformer_steps_per_cycle",
    "proposal_wall_ns_per_cycle",
    "selector_wall_ns_per_cycle",
    "target_verify_wall_ns_per_cycle",
    "cycle_wall_ns_per_cycle",
    "selection_regret_per_cycle",
    "proposal_regret_per_cycle",
)


def max_new_tokens_for_input(
    input_tokens: int,
    *,
    total_token_budget: int = GSM8K_DEPTH5_TOTAL_TOKEN_BUDGET,
) -> int:

    if isinstance(input_tokens, bool) or not isinstance(input_tokens, int):
        raise TypeError("input_tokens must be an integer")
    if isinstance(total_token_budget, bool) or not isinstance(
        total_token_budget, int
    ):
        raise TypeError("total_token_budget must be an integer")
    if input_tokens <= 0:
        raise ValueError("input_tokens must be a positive integer")
    if total_token_budget <= input_tokens:
        raise ValueError("Input length must be less than the total length limit")
    return total_token_budget - input_tokens


def speculative_mode_order(question_id: int) -> tuple[str, ...]:

    if question_id <= 0:
        raise ValueError("question_id must be a positive integer")
    offset = (question_id - 401) % len(GSM8K_DEPTH5_SPECULATIVE_MODES)
    return (
        GSM8K_DEPTH5_SPECULATIVE_MODES[offset:]
        + GSM8K_DEPTH5_SPECULATIVE_MODES[:offset]
    )


def acceptance_summary(
    accepted_per_cycle: Sequence[int],
) -> dict[str, object]:

    accepted = tuple(int(value) for value in accepted_per_cycle)
    if any(value < 0 or value > GSM8K_DEPTH5_TREE_MAX_DEPTH for value in accepted):
        raise ValueError("per-round accepted length must be in 0..5")
    histogram = Counter(accepted)
    return {
        "histogram_0_to_5": [
            histogram.get(depth, 0)
            for depth in range(GSM8K_DEPTH5_TREE_MAX_DEPTH + 1)
        ],
        "reached_depth_counts": [
            sum(value >= depth for value in accepted)
            for depth in range(1, GSM8K_DEPTH5_TREE_MAX_DEPTH + 1)
        ],
    }


def aggregate_depth5_runs(
    runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:

    aggregate: dict[str, object] = {}
    for mode in GSM8K_DEPTH5_MODES:
        selected = [run for run in runs if run.get("mode") == mode]
        generated_tokens = sum(int(run["generated_tokens"]) for run in selected)
        generation_seconds = sum(
            float(_require_mapping("run.timings", run["timings"])["generation_seconds"])
            for run in selected
        )
        wall_generation_seconds = sum(
            float(
                _require_mapping("run.timings", run["timings"])[
                    "wall_generation_seconds"
                ]
            )
            for run in selected
        )
        prefill_seconds = sum(
            float(_require_mapping("run.timings", run["timings"])["prefill_seconds"])
            for run in selected
        )
        decode_seconds = sum(
            float(_require_mapping("run.timings", run["timings"])["decode_seconds"])
            for run in selected
        )
        decode_tokens = sum(
            max(0, int(run["generated_tokens"]) - 1) for run in selected
        )
        cycles = sum(int(run["cycles"]) for run in selected)
        proposed = sum(int(run["proposed_draft_tokens"]) for run in selected)
        accepted = sum(int(run["accepted_draft_tokens"]) for run in selected)
        target_forward_calls = sum(
            int(run["target_forward_calls"]) for run in selected
        )
        verification_positions = sum(
            int(run["verification_positions"]) for run in selected
        )
        accepted_histogram = [0] * (GSM8K_DEPTH5_TREE_MAX_DEPTH + 1)
        for run in selected:
            for depth, count in enumerate(
                _require_mapping("run.acceptance", run["acceptance"])[
                    "histogram_0_to_5"
                ]
            ):
                accepted_histogram[depth] += int(count)
        cache_counters = {
            field: sum(
                int(
                    _require_mapping(
                        "run.cache_stats.end",
                        _require_mapping("run.cache_stats", run["cache_stats"])[
                            "end"
                        ],
                    )[field]
                )
                for run in selected
            )
            for field in _CACHE_COUNTER_FIELDS
        }
        decode_cache_counters = {
            field: sum(
                int(
                    _require_mapping(
                        "run.cache_stats.decode_counter_delta",
                        _require_mapping("run.cache_stats", run["cache_stats"])[
                            "decode_counter_delta"
                        ],
                    )[field]
                )
                for run in selected
            )
            for field in _CACHE_COUNTER_FIELDS
        }
        h2d_bytes = sum(
            int(
                _require_mapping(
                    "run.cache_stats.logical_h2d_bytes",
                    _require_mapping("run.cache_stats", run["cache_stats"])[
                        "logical_h2d_bytes"
                    ],
                )["total"]
            )
            for run in selected
        )
        cycle_work_totals = {
            field.removesuffix("_per_cycle"): sum(
                int(value)
                for run in selected
                for value in _require_sequence(
                    f"run.{field}", run[field]
                )
            )
            for field in _CYCLE_WORK_FIELDS
        }
        aggregate[mode] = {
            "samples": len(selected),
            "generated_tokens": generated_tokens,
            "decode_tokens_after_first": decode_tokens,
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "generation_seconds": generation_seconds,
            "wall_generation_seconds": wall_generation_seconds,
            "committed_tokens_per_second": (
                None
                if generation_seconds == 0
                else generated_tokens / generation_seconds
            ),
            "decode_tpot_seconds": (
                None if decode_tokens == 0 else decode_seconds / decode_tokens
            ),
            "cycles": cycles,
            "proposed_draft_tokens": proposed,
            "accepted_draft_tokens": accepted,
            "accepted_over_proposed": (
                None if proposed == 0 else accepted / proposed
            ),
            "mean_acceptance_length": (
                None if cycles == 0 else 1.0 + accepted / cycles
            ),
            "accepted_length_histogram_0_to_5": accepted_histogram,
            "target_forward_calls": target_forward_calls,
            "verification_positions": verification_positions,
            "cache_counters": cache_counters,
            "decode_cache_counters": decode_cache_counters,
            "logical_h2d_bytes": h2d_bytes,
            "logical_h2d_gib": h2d_bytes / 2**30,
            "cycle_work_totals": cycle_work_totals,
            "mean_selection_regret": (
                None
                if cycles == 0
                else cycle_work_totals["selection_regret"] / cycles
            ),
            "mean_proposal_regret": (
                None
                if cycles == 0
                else cycle_work_totals["proposal_regret"] / cycles
            ),
        }
    return aggregate


def write_depth5_checkpoint(path: Path, artifact: Mapping[str, object]) -> None:

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    serialized = json.dumps(
        artifact,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(serialized)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def assert_depth5_resume_compatible(
    existing: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:

    identity_fields = (
        "schema_version",
        "artifact_type",
        "session_id",
        "dataset",
        "selected_question_ids",
        "settings",
        "models",
        "environment",
    )
    for field in identity_fields:
        if existing.get(field) != expected.get(field):
            raise ValueError(f"resume checkpoint field {field} does not match the current config")


def validate_gsm8k_depth5_artifact(
    artifact: Mapping[str, object],
    *,
    allow_incomplete: bool = False,
) -> None:

    root = _require_mapping("artifact", artifact)
    version = _require_integer("schema_version", root.get("schema_version"))
    if version != GSM8K_DEPTH5_SCHEMA_VERSION:
        raise ValueError(f"unsupported GSM8K depth5 schema {version}")
    if root.get("artifact_type") != GSM8K_DEPTH5_ARTIFACT_TYPE:
        raise ValueError("artifact_type does not match the GSM8K depth5 experiment")
    _require_nonempty_string("session_id", root.get("session_id"))

    status = root.get("status")
    if status not in ("running", "complete"):
        raise ValueError("status must be running or complete")
    if status != "complete" and not allow_incomplete:
        raise ValueError("Experiment result is still in running state")

    dataset = _require_mapping("dataset", root.get("dataset"))
    for field in _REQUIRED_DATASET_FIELDS:
        _require_nonempty_string(f"dataset.{field}", dataset.get(field))

    settings = _require_mapping("settings", root.get("settings"))
    modes = tuple(
        _require_nonempty_string(f"settings.modes[{index}]", value)
        for index, value in enumerate(
            _require_sequence("settings.modes", settings.get("modes"))
        )
    )
    if modes != GSM8K_DEPTH5_MODES:
        raise ValueError("settings.modes must contain the fixed four generation modes")
    if _require_integer("settings.batch_size", settings.get("batch_size")) != 1:
        raise ValueError("settings.batch_size must equal 1")
    if (
        _require_integer(
            "settings.max_total_tokens",
            settings.get("max_total_tokens"),
            positive=True,
        )
        != GSM8K_DEPTH5_TOTAL_TOKEN_BUDGET
    ):
        raise ValueError("settings.max_total_tokens must equal 512")
    if (
        _require_integer(
            "settings.tree_top_k", settings.get("tree_top_k"), positive=True
        )
        != GSM8K_DEPTH5_TREE_TOP_K
    ):
        raise ValueError("settings.tree_top_k must equal 2")
    if (
        _require_integer(
            "settings.tree_max_depth",
            settings.get("tree_max_depth"),
            positive=True,
        )
        != GSM8K_DEPTH5_TREE_MAX_DEPTH
    ):
        raise ValueError("settings.tree_max_depth must equal 5")
    _require_integer(
        "settings.slots_per_layer",
        settings.get("slots_per_layer"),
        positive=True,
    )
    _require_integer(
        "settings.expert_bytes", settings.get("expert_bytes"), positive=True
    )
    if settings.get("numeric_mode") not in (
        "default",
        "bf16_full_reduction",
        "math_sdpa",
        "math_ieee",
        "math_bf16_full_reduction",
    ):
        raise ValueError("settings.numeric_mode is not supported")
    if settings.get("timing_scope") != "lightweight_summary":
        raise ValueError("settings.timing_scope must equal lightweight_summary")
    oracle_settings = _require_mapping(
        "settings.hindsight_oracle", settings.get("hindsight_oracle")
    )
    expected_oracle_settings = {
        "scope": "same_candidate_pool_acceptance_upper_bound",
        "teacher_information_cost_included": False,
        "optimizes_cross_cycle_cache_externality": False,
    }
    if dict(oracle_settings) != expected_oracle_settings:
        raise ValueError("settings.hindsight_oracle upper-bound convention mismatch")

    models = _require_mapping("models", root.get("models"))
    _require_model_identity("models.target", models.get("target"))
    _require_model_identity("models.drafter", models.get("drafter"))
    environment = _require_mapping("environment", root.get("environment"))
    for field in _REQUIRED_ENVIRONMENT_FIELDS:
        _require_nonempty_string(f"environment.{field}", environment.get(field))
    if not _require_mapping(
        "environment.numeric_settings",
        environment.get("numeric_settings"),
    ):
        raise ValueError("environment.numeric_settings cannot be empty")

    selected_question_ids = tuple(
        _require_integer(f"selected_question_ids[{index}]", value, positive=True)
        for index, value in enumerate(
            _require_sequence(
                "selected_question_ids", root.get("selected_question_ids")
            )
        )
    )
    if not selected_question_ids:
        raise ValueError("selected_question_ids cannot be empty")
    if len(set(selected_question_ids)) != len(selected_question_ids):
        raise ValueError("selected_question_ids must not contain duplicate question numbers")

    runs = _require_sequence("runs", root.get("runs"))
    expected_keys = {
        (question_id, mode)
        for question_id in selected_question_ids
        for mode in GSM8K_DEPTH5_MODES
    }
    if status == "complete" and len(runs) != len(expected_keys):
        raise ValueError("The number of runs in the full result must equal the number of questions times the four generation modes")
    if len(runs) > len(expected_keys):
        raise ValueError("runs count exceeds the experiment matrix range")

    run_by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    for index, value in enumerate(runs):
        run = _validate_run(
            index=index,
            value=value,
            expected_keys=expected_keys,
            settings=settings,
        )
        key = (int(run["question_id"]), str(run["mode"]))
        if key in run_by_key:
            raise ValueError(f"runs contains duplicate experiment entry {key}")
        run_by_key[key] = run

    if status == "complete" and set(run_by_key) != expected_keys:
        raise ValueError("runs do not cover all four generation modes for each question")
    for question_id in selected_question_ids:
        target = run_by_key.get((question_id, "target_only"))
        if target is None:
            if any(
                (question_id, mode) in run_by_key
                for mode in GSM8K_DEPTH5_SPECULATIVE_MODES
            ):
                raise ValueError("each question must record target_only output first")
            continue
        target_output = tuple(target["output_token_ids"])
        for mode in GSM8K_DEPTH5_SPECULATIVE_MODES:
            run = run_by_key.get((question_id, mode))
            if run is not None and tuple(run["output_token_ids"]) != target_output:
                raise ValueError(f"question {question_id} outputs from the four generation modes are inconsistent")

    aggregate = _require_mapping("aggregate", root.get("aggregate"))
    if set(aggregate) != set(GSM8K_DEPTH5_MODES):
        raise ValueError("aggregate must cover the four fixed generation modes")
    expected_aggregate = aggregate_depth5_runs(tuple(run_by_key.values()))
    for mode in GSM8K_DEPTH5_MODES:
        actual = _require_mapping(f"aggregate.{mode}", aggregate[mode])
        expected = _require_mapping(
            f"expected_aggregate.{mode}", expected_aggregate[mode]
        )
        if actual != expected:
            raise ValueError(f"aggregate.{mode} inconsistent with runs")


def _validate_run(
    *,
    index: int,
    value: object,
    expected_keys: set[tuple[int, str]],
    settings: Mapping[str, Any],
) -> Mapping[str, Any]:
    name = f"runs[{index}]"
    run = _require_mapping(name, value)
    question_id = _require_integer(
        f"{name}.question_id", run.get("question_id"), positive=True
    )
    mode = _require_nonempty_string(f"{name}.mode", run.get("mode"))
    if (question_id, mode) not in expected_keys:
        raise ValueError(f"{name} contains unselected questions or generation modes")
    _require_integer(f"{name}.execution_index", run.get("execution_index"))
    _require_nonempty_string(f"{name}.run_id", run.get("run_id"))

    input_tokens = _require_integer(
        f"{name}.input_tokens", run.get("input_tokens"), positive=True
    )
    max_new_tokens = _require_integer(
        f"{name}.max_new_tokens", run.get("max_new_tokens"), positive=True
    )
    if max_new_tokens != max_new_tokens_for_input(
        input_tokens,
        total_token_budget=int(settings["max_total_tokens"]),
    ):
        raise ValueError(f"{name}.max_new_tokens does not match the total length limit")
    output_token_ids = _require_token_ids(
        f"{name}.output_token_ids", run.get("output_token_ids")
    )
    generated_tokens = _require_integer(
        f"{name}.generated_tokens", run.get("generated_tokens"), positive=True
    )
    if generated_tokens != len(output_token_ids):
        raise ValueError(f"{name}.generated_tokens does not match the output")
    if generated_tokens > max_new_tokens:
        raise ValueError(f"{name}.output_token_ids exceeds the generation budget")
    if _require_integer(f"{name}.total_tokens", run.get("total_tokens")) != (
        input_tokens + generated_tokens
    ):
        raise ValueError(f"{name}.total_tokens inconsistent with input/output length")

    stopped_on_eos = run.get("stopped_on_eos")
    if not isinstance(stopped_on_eos, bool):
        raise TypeError(f"{name}.stopped_on_eos must be a boolean")
    termination_reason = run.get("termination_reason")
    expected_reason = "eos" if stopped_on_eos else "max_total_tokens"
    if termination_reason != expected_reason:
        raise ValueError(f"{name}.termination_reason does not match the stop state")
    if not stopped_on_eos and generated_tokens != max_new_tokens:
        raise ValueError(f"{name} must generate to the total length limit when EOS is not encountered")

    cycles = _require_integer(f"{name}.cycles", run.get("cycles"))
    proposed = _require_integer(
        f"{name}.proposed_draft_tokens", run.get("proposed_draft_tokens")
    )
    accepted = _require_integer(
        f"{name}.accepted_draft_tokens", run.get("accepted_draft_tokens")
    )
    accepted_per_cycle = tuple(
        _require_integer(f"{name}.accepted_per_cycle[{cycle}]", item)
        for cycle, item in enumerate(
            _require_sequence(
                f"{name}.accepted_per_cycle", run.get("accepted_per_cycle")
            )
        )
    )
    pool_per_cycle = tuple(
        _require_integer(f"{name}.tree_pool_nodes_per_cycle[{cycle}]", item)
        for cycle, item in enumerate(
            _require_sequence(
                f"{name}.tree_pool_nodes_per_cycle",
                run.get("tree_pool_nodes_per_cycle"),
            )
        )
    )
    selected_per_cycle = tuple(
        _require_integer(
            f"{name}.selected_tree_nodes_per_cycle[{cycle}]", item
        )
        for cycle, item in enumerate(
            _require_sequence(
                f"{name}.selected_tree_nodes_per_cycle",
                run.get("selected_tree_nodes_per_cycle"),
            )
        )
    )
    committed_offsets = tuple(
        _require_integer(
            f"{name}.committed_output_offsets_per_cycle[{cycle}]", item
        )
        for cycle, item in enumerate(
            _require_sequence(
                f"{name}.committed_output_offsets_per_cycle",
                run.get("committed_output_offsets_per_cycle"),
            )
        )
    )
    path_score_calls = tuple(
        _require_integer(
            f"{name}.path_score_calls_per_cycle[{cycle}]",
            item,
            positive=True,
        )
        for cycle, item in enumerate(
            _require_sequence(
                f"{name}.path_score_calls_per_cycle",
                run.get("path_score_calls_per_cycle"),
            )
        )
    )
    draft_steps = tuple(
        _require_integer(
            f"{name}.draft_transformer_steps_per_cycle[{cycle}]",
            item,
            positive=True,
        )
        for cycle, item in enumerate(
            _require_sequence(
                f"{name}.draft_transformer_steps_per_cycle",
                run.get("draft_transformer_steps_per_cycle"),
            )
        )
    )
    actual_depths = _require_integer_sequence(
        f"{name}.actual_tree_depth_per_cycle",
        run.get("actual_tree_depth_per_cycle"),
        positive=True,
    )
    correct_depths = _require_integer_sequence(
        f"{name}.pool_correct_depth_per_cycle",
        run.get("pool_correct_depth_per_cycle"),
    )
    proposal_wall_ns = _require_integer_sequence(
        f"{name}.proposal_wall_ns_per_cycle",
        run.get("proposal_wall_ns_per_cycle"),
    )
    selector_wall_ns = _require_integer_sequence(
        f"{name}.selector_wall_ns_per_cycle",
        run.get("selector_wall_ns_per_cycle"),
    )
    target_verify_wall_ns = _require_integer_sequence(
        f"{name}.target_verify_wall_ns_per_cycle",
        run.get("target_verify_wall_ns_per_cycle"),
    )
    cycle_wall_ns = _require_integer_sequence(
        f"{name}.cycle_wall_ns_per_cycle",
        run.get("cycle_wall_ns_per_cycle"),
    )
    selection_regrets = _require_integer_sequence(
        f"{name}.selection_regret_per_cycle",
        run.get("selection_regret_per_cycle"),
    )
    proposal_regrets = _require_integer_sequence(
        f"{name}.proposal_regret_per_cycle",
        run.get("proposal_regret_per_cycle"),
    )
    _validate_speculation_counts(
        name=name,
        mode=mode,
        cycles=cycles,
        proposed=proposed,
        accepted=accepted,
        accepted_per_cycle=accepted_per_cycle,
        pool_per_cycle=pool_per_cycle,
        selected_per_cycle=selected_per_cycle,
        actual_depths=actual_depths,
        correct_depths=correct_depths,
        path_score_calls=path_score_calls,
        draft_steps=draft_steps,
        proposal_wall_ns=proposal_wall_ns,
        selector_wall_ns=selector_wall_ns,
        target_verify_wall_ns=target_verify_wall_ns,
        cycle_wall_ns=cycle_wall_ns,
        selection_regrets=selection_regrets,
        proposal_regrets=proposal_regrets,
    )
    if len(committed_offsets) != cycles:
        raise ValueError("per-round commit position count must equal cycles")
    expected_offsets = []
    offset = 0
    for value in accepted_per_cycle:
        expected_offsets.append(offset)
        offset += 1 + value
    if committed_offsets != tuple(expected_offsets):
        raise ValueError("committed_output_offsets_per_cycle inconsistent with accepted length")
    if _require_mapping(f"{name}.acceptance", run.get("acceptance")) != (
        acceptance_summary(accepted_per_cycle)
    ):
        raise ValueError(f"{name}.acceptance does not match per-round statistics")

    target_forward_calls = _require_integer(
        f"{name}.target_forward_calls",
        run.get("target_forward_calls"),
        positive=True,
    )
    expected_forward_calls = generated_tokens if mode == "target_only" else 1 + cycles
    if target_forward_calls != expected_forward_calls:
        raise ValueError(f"{name}.target_forward_calls does not match the mode")
    verification_positions = _require_integer(
        f"{name}.verification_positions", run.get("verification_positions")
    )
    if verification_positions != cycles + proposed:
        raise ValueError(f"{name}.verification_positions does not match tree statistics")

    oracle = run.get("oracle")
    if mode == "eagle3_tree_top2_depth5_hindsight_path_oracle":
        oracle_mapping = _require_mapping(f"{name}.oracle", oracle)
        if {
            key: oracle_mapping.get(key)
            for key in (
                "scope",
                "teacher_information_cost_included",
                "optimizes_cross_cycle_cache_externality",
            )
        } != {
            "scope": "same_candidate_pool_acceptance_upper_bound",
            "teacher_information_cost_included": False,
            "optimizes_cross_cycle_cache_externality": False,
        }:
            raise ValueError(f"{name}.oracle upper-bound convention is inconsistent")
        if len(
            _require_sequence(
                f"{name}.oracle.selection_records",
                oracle_mapping.get("selection_records"),
            )
        ) != cycles:
            raise ValueError(f"{name}.oracle.selection_records count inconsistent")
    elif oracle is not None:
        raise ValueError(f"{name} only oracle mode may contain an oracle field")

    timings = _require_mapping(f"{name}.timings", run.get("timings"))
    if timings.get("phase_timing_scope") != "host_wall_partial":
        raise ValueError(
            f"{name}.timings.phase_timing_scope must equal host_wall_partial"
        )
    prefill_seconds = _require_nonnegative_number(
        f"{name}.timings.prefill_seconds", timings.get("prefill_seconds")
    )
    decode_seconds = _require_nonnegative_number(
        f"{name}.timings.decode_seconds", timings.get("decode_seconds")
    )
    generation_seconds = _require_nonnegative_number(
        f"{name}.timings.generation_seconds", timings.get("generation_seconds")
    )
    _require_nonnegative_number(
        f"{name}.timings.wall_generation_seconds",
        timings.get("wall_generation_seconds"),
    )
    if not math.isclose(
        prefill_seconds + decode_seconds,
        generation_seconds,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ValueError(f"{name}.timings stage sum mismatch")
    phase_seconds = _require_mapping(
        f"{name}.timings.phase_seconds", timings.get("phase_seconds")
    )
    phase_values = {
        field: _require_nonnegative_number(
            f"{name}.timings.phase_seconds.{field}",
            phase_seconds.get(field),
        )
        for field in _PHASE_TIMING_FIELDS
    }
    expected_phase_values = {
        "tree_build_seconds": (
            sum(proposal_wall_ns) - sum(selector_wall_ns)
        )
        / 1e9,
        "selection_seconds": sum(selector_wall_ns) / 1e9,
        "target_verification_seconds": sum(target_verify_wall_ns) / 1e9,
        "tree_cycle_seconds": sum(cycle_wall_ns) / 1e9,
    }
    if any(
        not math.isclose(
            phase_values[field],
            expected_phase_values[field],
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        for field in _PHASE_TIMING_FIELDS
    ):
        raise ValueError(f"{name}.timings.phase_seconds is inconsistent with per-cycle timing")

    _validate_cache_stats(
        name=name,
        cache_stats=run.get("cache_stats"),
        expert_bytes=int(settings["expert_bytes"]),
    )
    return run


def _validate_speculation_counts(
    *,
    name: str,
    mode: str,
    cycles: int,
    proposed: int,
    accepted: int,
    accepted_per_cycle: tuple[int, ...],
    pool_per_cycle: tuple[int, ...],
    selected_per_cycle: tuple[int, ...],
    actual_depths: tuple[int, ...],
    correct_depths: tuple[int, ...],
    path_score_calls: tuple[int, ...],
    draft_steps: tuple[int, ...],
    proposal_wall_ns: tuple[int, ...],
    selector_wall_ns: tuple[int, ...],
    target_verify_wall_ns: tuple[int, ...],
    cycle_wall_ns: tuple[int, ...],
    selection_regrets: tuple[int, ...],
    proposal_regrets: tuple[int, ...],
) -> None:
    cycle_sequences = (
        accepted_per_cycle,
        pool_per_cycle,
        selected_per_cycle,
        actual_depths,
        correct_depths,
        path_score_calls,
        draft_steps,
        proposal_wall_ns,
        selector_wall_ns,
        target_verify_wall_ns,
        cycle_wall_ns,
        selection_regrets,
        proposal_regrets,
    )
    if mode == "target_only":
        if cycles != 0 or proposed != 0 or accepted != 0 or any(
            cycle_sequences
        ):
            raise ValueError("target_only speculative stats must all be zero")
        return
    if any(len(values) != cycles for values in cycle_sequences):
        raise ValueError("P7/P11/O per-cycle array length must equal cycles")
    if any(value > GSM8K_DEPTH5_TREE_MAX_DEPTH for value in accepted_per_cycle):
        raise ValueError("depth=5 accepts at most five draft tokens per cycle")
    if any(value > GSM8K_DEPTH5_TREE_MAX_DEPTH for value in actual_depths):
        raise ValueError("actual_tree_depth cannot exceed depth=5")
    if sum(accepted_per_cycle) != accepted:
        raise ValueError("accepted_per_cycle sum must equal accepted_draft_tokens")
    if sum(selected_per_cycle) != proposed:
        raise ValueError("selected_tree_nodes_per_cycle sum must equal proposed")
    if any(
        selected > pool
        for selected, pool in zip(selected_per_cycle, pool_per_cycle)
    ):
        raise ValueError("selected nodes per round must not exceed candidate pool nodes")
    if any(
        accepted_value > selected
        for accepted_value, selected in zip(
            accepted_per_cycle, selected_per_cycle
        )
    ):
        raise ValueError("The number of accepted nodes per round cannot exceed the number of selected nodes")
    if any(
        correct > actual
        for correct, actual in zip(correct_depths, actual_depths)
    ):
        raise ValueError("pool_correct_depth cannot exceed actual_tree_depth")
    if mode != "eagle3_tree_top2_depth5_hindsight_path_oracle" and any(
        selected <= 0 for selected in selected_per_cycle
    ):
        raise ValueError("cumulative-probability tree cycle must select at least one node")
    if accepted > proposed:
        raise ValueError("accepted_draft_tokens cannot exceed proposed_draft_tokens")
    budget = _MODE_NODE_BUDGETS[mode]
    if any(value > budget for value in selected_per_cycle):
        raise ValueError(f"{mode} selected nodes per round exceed the budget {budget}")
    if any(steps < calls for calls, steps in zip(path_score_calls, draft_steps)):
        raise ValueError("Draft Transformer step count must cover the path scoring calls")
    if any(
        selector > proposal or proposal > cycle
        for selector, proposal, cycle in zip(
            selector_wall_ns, proposal_wall_ns, cycle_wall_ns
        )
    ):
        raise ValueError("per-cycle timing must satisfy selector <= proposal <= cycle")
    if any(
        proposal + target_verify > cycle
        for proposal, target_verify, cycle in zip(
            proposal_wall_ns,
            target_verify_wall_ns,
            cycle_wall_ns,
        )
    ):
        raise ValueError("proposal and target_verify timing sum cannot exceed the cycle")
    expected_selection_regrets = tuple(
        correct - accepted_value
        for correct, accepted_value in zip(correct_depths, accepted_per_cycle)
    )
    if selection_regrets != expected_selection_regrets:
        raise ValueError("selection_regret must equal pool_correct - accepted")
    expected_proposal_regrets = tuple(
        actual - correct
        for actual, correct in zip(actual_depths, correct_depths)
    )
    if proposal_regrets != expected_proposal_regrets:
        raise ValueError("proposal_regret must equal actual_depth - pool_correct")
    if mode == "eagle3_tree_top2_depth5_hindsight_path_oracle" and (
        accepted_per_cycle != correct_depths
    ):
        raise ValueError("Mode O must satisfy accepted == pool_correct")


def _validate_cache_stats(
    *, name: str, cache_stats: object, expert_bytes: int
) -> None:
    cache = _require_mapping(f"{name}.cache_stats", cache_stats)
    prefill = _require_mapping(f"{name}.cache_stats.prefill", cache.get("prefill"))
    end = _require_mapping(f"{name}.cache_stats.end", cache.get("end"))
    delta = _require_mapping(
        f"{name}.cache_stats.decode_counter_delta",
        cache.get("decode_counter_delta"),
    )
    for field in _CACHE_FIELDS:
        _require_integer(f"{name}.cache_stats.prefill.{field}", prefill.get(field))
        _require_integer(f"{name}.cache_stats.end.{field}", end.get(field))
    for field in _CACHE_COUNTER_FIELDS:
        value = _require_integer(
            f"{name}.cache_stats.decode_counter_delta.{field}", delta.get(field)
        )
        if int(prefill[field]) + value != int(end[field]):
            raise ValueError(f"{name}.cache_stats.{field} stage sum mismatch")
    logical = _require_mapping(
        f"{name}.cache_stats.logical_h2d_bytes",
        cache.get("logical_h2d_bytes"),
    )
    expected = {
        "prefill": int(prefill["h2d_submissions"]) * expert_bytes,
        "decode": int(delta["h2d_submissions"]) * expert_bytes,
        "total": int(end["h2d_submissions"]) * expert_bytes,
    }
    if dict(logical) != expected:
        raise ValueError(f"{name}.cache_stats.logical_h2d_bytes is inconsistent with the commit count")


def _require_mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _require_sequence(name: str, value: object) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return value


def _require_integer_sequence(
    name: str,
    value: object,
    *,
    positive: bool = False,
) -> tuple[int, ...]:
    values = _require_sequence(name, value)
    return tuple(
        _require_integer(
            f"{name}[{index}]",
            item,
            positive=positive,
        )
        for index, item in enumerate(values)
    )


def _require_nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_integer(
    name: str, value: object, *, positive: bool = False
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    minimum = 1 if positive else 0
    if value < minimum:
        qualifier = "positive integer" if positive else "non-negative integer"
        raise ValueError(f"{name} must be{qualifier}")
    return value


def _require_nonnegative_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a numeric value")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _require_token_ids(name: str, value: object) -> tuple[int, ...]:
    values = _require_sequence(name, value)
    result = tuple(
        _require_integer(f"{name}[{index}]", item)
        for index, item in enumerate(values)
    )
    if not result:
        raise ValueError(f"{name} cannot be empty")
    return result


def _require_model_identity(name: str, value: object) -> None:
    identity = _require_mapping(name, value)
    for field in ("path", "revision"):
        _require_nonempty_string(f"{name}.{field}", identity.get(field))


__all__ = [
    "GSM8K_DEPTH5_ARTIFACT_TYPE",
    "GSM8K_DEPTH5_MODES",
    "GSM8K_DEPTH5_SCHEMA_VERSION",
    "GSM8K_DEPTH5_SPECULATIVE_MODES",
    "GSM8K_DEPTH5_TOTAL_TOKEN_BUDGET",
    "GSM8K_DEPTH5_TREE_MAX_DEPTH",
    "GSM8K_DEPTH5_TREE_TOP_K",
    "acceptance_summary",
    "aggregate_depth5_runs",
    "assert_depth5_resume_compatible",
    "max_new_tokens_for_input",
    "speculative_mode_order",
    "validate_gsm8k_depth5_artifact",
    "write_depth5_checkpoint",
]
