"""CPU contracts for complete, ordered GSM8K request streams.

Offline atlas preparation and measured execution are separate artifacts. A
stream resets expert caches exactly once, and never between its requests.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from hashlib import sha256
import math
import re
from types import SimpleNamespace

from .experiment import canonical_json_sha256
from .r6a_experiment import compute_r6a_pool_fingerprint
from .r6a_route_observation import RouteObservation
from .selection_methods import SelectionCostModel, select_nodes
from .selection_refinement import fast_expert_selection
from .m1_capture_validation import M1_EOS_TOKEN_IDS

PROTOCOL = "gsm8k-serial-replay-v1"
METHODS = ("target_only", "expert_fast_dynamic", "probability_dynamic")
TOTAL_TOKENS = 512
COUNTERS = ("h2d_submissions", "resident_hits", "inflight_joins", "evictions")


def file_identity(path):
    data = path.read_bytes()
    return dict(basename=path.name, size_bytes=len(data), sha256=sha256(data).hexdigest())


def pool_view(value):
    """Read recorded candidates without imposing the historical 18-node limit."""
    if value["fingerprint"] != compute_r6a_pool_fingerprint(value):
        raise ValueError("candidate pool fingerprint mismatch")
    nodes = tuple(SimpleNamespace(**n) for n in value["candidates"])
    if (len(nodes) not in (2, 6, 10, 14, 18)
            or value["top_k"] != 2 or value["raw_candidate_count"] != len(nodes)
            or value["max_depth"] != (len(nodes) + 2) // 4
            or tuple(n.raw_id for n in nodes) != tuple(range(1, len(nodes) + 1))
            or any(not 0 <= n.parent_raw_id < n.raw_id for n in nodes)
            or any(not math.isfinite(n.cumulative_log_probability)
                   or n.cumulative_log_probability > 0 for n in nodes)):
        raise ValueError("invalid serial candidate pool")
    return SimpleNamespace(nodes=nodes, seed_token_id=value["seed_token_id"],
                           fingerprint=value["fingerprint"])


def route_atlas(pool, matrices):
    positions = tuple(range(len(pool.nodes) + 1))
    if len(matrices) != 48:
        raise ValueError("route atlas must cover 48 layers")
    rows = []
    for lid, matrix in enumerate(matrices):
        if len(matrix) != len(positions):
            raise ValueError("route atlas position coverage mismatch")
        for raw, values in enumerate(matrix):
            if (len(values) != 8 or len(set(values)) != 8
                    or any(type(e) is not int or not 0 <= e < 128 for e in values)):
                raise ValueError("invalid Top-8 expert route")
            rows.append((raw, lid, tuple(values)))
    return RouteObservation(positions, canonical_json_sha256(matrices), tuple(rows))


def choose(pool, method, model, matrices=None):
    if method == "probability_dynamic":
        return select_nodes(pool, None, method, model)
    if method == "expert_fast_dynamic":
        return fast_expert_selection(pool, route_atlas(pool, matrices), model)
    raise ValueError("method does not select a draft tree")


def numeric_answer(text, *, reference=False):
    """Require an explicit final answer; never silently pick a reasoning number."""
    patterns = [r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)"]
    if not reference:
        patterns += [r"\\boxed\{\s*(?:\\?\$)?\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}",
                     r"(?:final answer|answer is|answer:)\s*:?\s*(?:\*\*|__)?\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)"]
    for pattern in patterns:
        values = re.findall(pattern, text, re.IGNORECASE)
        if values:
            try:
                value = Decimal(values[-1].replace(",", ""))
                if value.is_finite():
                    return str(value.normalize())
            except InvalidOperation:
                pass
    return None


def validate_capture(value, model):
    if value.get("protocol") != PROTOCOL or value.get("method") != "expert_fast_dynamic":
        raise ValueError("unexpected atlas artifact")
    offset = 0
    committed = []
    for index, row in enumerate(value["cycles"]):
        pool = pool_view(row["pool"])
        ids = choose(pool, value["method"], model, row["atlas"])
        if (row["iteration"] != index or row["committed_offset"] != offset
                or list(ids) != row["selected_raw_ids"]):
            raise ValueError("atlas decision or sequence boundary mismatch")
        tokens = row["committed_tokens"]
        if not tokens or len(tokens) > 1 + len(ids) or tokens[0] != pool.seed_token_id:
            raise ValueError("invalid captured acceptance")
        nodes = {n.raw_id: n for n in pool.nodes}
        raw = 0
        for token in tokens[1:]:
            candidates = [i for i in ids if nodes[i].parent_raw_id == raw
                          and nodes[i].target_token_id == token]
            if len(candidates) != 1:
                raise ValueError("captured tokens do not form an accepted tree path")
            raw = candidates[0]
        offset += len(tokens)
        committed.extend(tokens)
    output = value["output_token_ids"]
    if output[:offset] != committed or not 0 <= len(output) - offset <= 1:
        raise ValueError("captured output does not match cycle commits")
    if not all(row["passed"] for row in value["commit_audits"]):
        raise ValueError("captured KV/aux commit audit failed")
    if len(value["commit_audits"]) != len(value["cycles"]):
        raise ValueError("captured audit coverage mismatch")
    return len(value["cycles"])


def validate_stream(rows, *, question_ids, method, repetition):
    if [r["question_id"] for r in rows] != list(question_ids):
        raise ValueError("stream must cover every question in the frozen order")
    for index, row in enumerate(rows):
        if (row["protocol"] != PROTOCOL or row["method"] != method
                or row["repetition"] != repetition or row["ordinal"] != index
                or row["status"] != "complete"):
            raise ValueError("stream identity or completion mismatch")
        if index == 0:
            if row["pre_cache_occupied"] != 0:
                raise ValueError("stream did not start with empty expert caches")
        elif row["pre_cache_sha256"] != rows[index - 1]["post_cache_sha256"]:
            raise ValueError("expert cache was reset or changed between requests")
        if row["input_tokens"] + len(row["output_token_ids"]) > TOTAL_TOKENS:
            raise ValueError("generation exceeded the validated token budget")
        if row["output_tokens"] != len(row["output_token_ids"]) or row["output_tokens"] < 1:
            raise ValueError("invalid output token count")
        eos = [i for i, token in enumerate(row["output_token_ids"]) if token in M1_EOS_TOKEN_IDS]
        if ((row["stop_reason"] == "eos" and (eos != [row["output_tokens"] - 1]
                or row["stop_token_id"] != row["output_token_ids"][-1]))
                or (row["stop_reason"] == "length" and (eos or row["stop_token_id"] is not None))):
            raise ValueError("termination does not match the actual output tokens")
        if row["stop_reason"] not in ("eos", "length"):
            raise ValueError("unknown generation termination")
        if row["stop_reason"] == "length" and row["input_tokens"] + row["output_tokens"] != TOTAL_TOKENS:
            raise ValueError("length-limited generation ended early")
        if not 0 < row["prefill_ns"] <= row["request_ns"] or row["decode_ns"] != row["request_ns"] - row["prefill_ns"]:
            raise ValueError("request timing does not close")
        if any(row["cache_delta"][k] < 0 for k in COUNTERS):
            raise ValueError("expert counters decreased within a request")
        for side in ("pre", "post"):
            layers = row[f"{side}_cache"]
            occupied = sum(s["expert"] is not None for layer in layers for s in layer["slots"])
            if (canonical_json_sha256(layers) != row[f"{side}_cache_sha256"]
                    or occupied != row[f"{side}_cache_occupied"]):
                raise ValueError("recorded cache snapshot differs from its digest or occupancy")
        if any(row["cache_delta"][k] != sum(b["stats"][k] - a["stats"][k]
                for a, b in zip(row["pre_cache"], row["post_cache"], strict=True)) for k in COUNTERS):
            raise ValueError("cache counter delta does not match snapshots")
        if not row["all_commits_passed"] or not row["reference_match"]:
            raise ValueError("same-query replay or commit audit failed")
        if row["max_gpu_reserved_bytes"] > 38 * 1024**3 or row["max_rss_bytes"] > 128 * 1024**3:
            raise ValueError("resource budget exceeded")
    return dict(questions=len(rows), output_tokens=sum(r["output_tokens"] for r in rows),
        decode_tokens=sum(r["output_tokens"] - 1 for r in rows),
        request_ns=sum(r["request_ns"] for r in rows),
        decode_ns=sum(r["decode_ns"] for r in rows),
        truncated=sum(r["stop_reason"] == "length" for r in rows),
        answer_correct=sum(r["answer_correct"] for r in rows),
        answer_unparsed=sum(r["predicted_answer"] is None for r in rows))
