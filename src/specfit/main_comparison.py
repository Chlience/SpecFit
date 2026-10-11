"""Frozen contracts for the original SpecFit and cache-controlled EAGLE comparison."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
from math import exp, isfinite
from pathlib import Path
import random
import re
from types import SimpleNamespace

from .acceptance_metrics import acceptance_lengths
from .checkpoint_comparison import CHAIN_METHODS, verify_chain_cycles
from .experiment import canonical_json_sha256
from .gsm8k import load_gsm8k_specbench
from .r6a_selection_contract import enumerate_ancestor_closed_subsets
from .selection_methods import probability_ids
from .serial_replay import choose as original_choose, file_identity, numeric_answer, pool_view
from .specfit_exact import _BITS, _score_packed_routes, _search_rows

PROTOCOL = "specfit-main-comparison-v1"
METHODS = ("ar", "eagle_no_cache", "eagle_cache", "specfit")
REFERENCE_METHODS = ("ar", "eagle_cache", "specfit")
# Same decoding rule as the cached counterpart; completed experts are discarded before every target forward.
NO_CACHE = {"ar_no_cache": "ar", "eagle_no_cache": "eagle_cache", "eagle_chain3_no_cache": "eagle_chain3_cache",
            "specfit_no_cache": "specfit", "specfit_adaptive_no_cache": "specfit_adaptive"}
# Adaptive-depth runs only; the frozen four-method comparison does not include them.
# specfit_packed: SpecFit choices from the packed exact search, sharing the SpecFit reference (pilot only).
# specfit_adaptive: stop drafting after level d when SpecFit on the depth<=d pool picks no depth-d node.
LEVEL_METHODS = ("specfit_packed", "specfit_adaptive")
TOTAL_TOKENS = {"gsm8k": 512, "humaneval": 2048, "aime25": 2048, "summarization": 2048}
SMOKE_IDS = {"gsm8k": (404, 439), "humaneval": (0,), "aime25": (10,), "summarization": (31, 54)}
# Input-length ranks 0, 11, 23, 34, 45, 56, 68, 79 of the audited 2048 budget: shortest to longest article.
PILOT_IDS = {"summarization": (7, 11, 19, 36, 52, 54, 58, 60)}
PROMPT_BUDGETS = ("docs/results/evaluation-prompt-budget-2048.json",
                  "docs/results/summarization-prompt-budget-2048.json")
GPU_UUIDS = ("GPU-00000000-0000-0000-0000-000000000001",
             "GPU-00000000-0000-0000-0000-000000000002")
GPU_NAME = "NVIDIA A100-PCIE-40GB"
# Second device: one RTX 4090 24GB (PCIe Gen3 x16) on another host, measuring every method itself.
DEVICES = {"rtx4090": dict(gpu_name="NVIDIA GeForce RTX 4090", gpu_uuid="GPU-00000000-0000-0000-0000-000000000003")}
ROUTE_SEED = "specfit-route-accuracy-v1"


def reference_method(method):
    if method not in (*METHODS, *CHAIN_METHODS, *NO_CACHE, *LEVEL_METHODS):
        raise ValueError("unknown comparison method")
    if method == 'specfit_packed':
        return 'specfit'
    return NO_CACHE.get(method, method)


def choose(pool, method, model, atlas=None):
    if method in ("eagle_cache", "eagle_no_cache"):
        return probability_ids(pool, min(7, len(pool.nodes)))
    if method in ("specfit", "specfit_no_cache"):
        return original_choose(pool, "expert_fast_dynamic", model, atlas)
    raise ValueError("AR does not construct or select a candidate pool")


class LevelSpecFit:
    """SpecFit over one cycle's level prefixes: pack each new level's routes once, then run the
    exact packed search, which keeps the original objective, builtin summation and tie order."""

    def __init__(self, model, atlas):
        if len(atlas) != 48:
            raise ValueError("route atlas must cover 48 layers")
        self.model, self.atlas, self.bitmaps = model, atlas, []

    def select(self, nodes):
        return _score_packed_routes(SimpleNamespace(nodes=nodes), self.pack(nodes), self.model, 0)

    def pack(self, nodes):
        count = len(nodes)
        if count not in (2, 6, 10, 14, 18) or any(n.raw_id != raw for raw, n in enumerate(nodes, 1)):
            raise ValueError("invalid level-prefix candidate pool")
        bits = _BITS
        for raw in range(len(self.bitmaps), count + 1):
            packed = 0
            for layer, matrix in enumerate(self.atlas):
                if len(matrix) <= raw or len(matrix[raw]) != 8:
                    raise ValueError("route atlas does not cover the candidate prefix")
                a, b, c, d, e, f, g, h = matrix[raw]
                if (type(a) is not int or type(b) is not int or type(c) is not int
                        or type(d) is not int or type(e) is not int or type(f) is not int
                        or type(g) is not int or type(h) is not int
                        or (a | b | c | d | e | f | g | h) & ~127):
                    raise ValueError("invalid expert ID")
                mask = bits[a] | bits[b] | bits[c] | bits[d] | bits[e] | bits[f] | bits[g] | bits[h]
                if mask.bit_count() != 8:
                    raise ValueError("duplicate expert ID")
                packed |= mask << (128 * layer)
            self.bitmaps.append(packed)
        return self.bitmaps

    def stop(self, nodes):
        """Adaptive-depth rule on the depth<=d prefix: (no depth-d node selected, selection)."""
        ids = self.select(nodes)
        depth = nodes[-1].depth
        return all(nodes[raw - 1].depth != depth for raw in ids), ids


@dataclass(frozen=True)
class RevisedCostModel:
    """Cost-revised cycle time: verification, selector and bookkeeping by node count plus one expert-union slope,
    refit on measured cycles, and the measured proposal time of each number of drafted levels. A tree that uses a
    node of the level just drafted commits the next level, so it is charged that level's drafting as well."""

    size_intercept_ns: tuple
    expert_ns: float
    draft_ns: tuple

    def __post_init__(self):
        if (len(self.size_intercept_ns) != 8 or len(self.draft_ns) != 5
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not isfinite(x) or x < 0
                       for x in (*self.size_intercept_ns, self.expert_ns, *self.draft_ns))
                or any(b + 384 * self.expert_ns <= 0 for b in self.size_intercept_ns)
                or any(a >= b for a, b in zip(self.draft_ns, self.draft_ns[1:]))):
            raise ValueError("invalid revised cost model")

    def predict_ns(self, node_count, expert_union, drafted_levels):
        if (type(node_count) is not int or not 0 <= node_count <= 7 or type(expert_union) is not int
                or expert_union < 384 or type(drafted_levels) is not int or not 1 <= drafted_levels <= 5):
            raise ValueError("tree is outside the revised cost model")
        return self.size_intercept_ns[node_count] + self.expert_ns * expert_union + self.draft_ns[drafted_levels - 1]

    @classmethod
    def from_dict(cls, value):
        if set(value) != {"size_intercept_ns", "expert_ns", "draft_ns"}:
            raise ValueError("unexpected revised cost-model fields")
        return cls(tuple(value["size_intercept_ns"]), value["expert_ns"], tuple(value["draft_ns"]))


def drafted_levels(nodes, ids, level):
    """Levels drafted if the depth<=level selection `ids` is kept: using a depth-`level` node continues drafting."""
    return min(level + 1, 5) if any(nodes[raw - 1].depth == level for raw in ids) else level


class RevisedLevelSpecFit(LevelSpecFit):
    """Adaptive SpecFit with the revised cost model; same packing, search order, reward and tie order."""

    def select(self, nodes):
        bitmaps, level = self.pack(nodes), nodes[-1].depth
        model = self.model
        rows = _search_rows(tuple(n.parent_raw_id for n in nodes))
        probabilities = (0.0, *(exp(n.cumulative_log_probability) for n in nodes))
        reaches = (False, *(n.depth == level for n in nodes))
        offsets, slope = model.size_intercept_ns, model.expert_ns
        stay, grow = model.draft_ns[level - 1], model.draft_ns[min(level + 1, 5) - 1]
        footprints, reached = [], []
        best_score, best_tie, best_ids = -1.0, None, None
        for parent, raw, ids, count, gather in rows:
            if parent < 0:
                footprint, reach = bitmaps[0], False
            else:
                footprint, reach = footprints[parent] | bitmaps[raw], reached[parent] or reaches[raw]
            footprints.append(footprint)
            reached.append(reach)
            experts = footprint.bit_count()
            elapsed = offsets[count] + slope * experts + (grow if reach else stay)
            reward = (1.0 + sum(gather(probabilities), start=0.0) if count > 1 else
                      1.0 + probabilities[raw] if count == 1 else 1.0)
            score = reward / elapsed
            if score >= best_score:
                tie = (-reward, experts, count, ids)
                if score > best_score or best_tie is None or tie < best_tie:
                    best_score, best_tie, best_ids = score, tie, ids
        return best_ids


def exhaustive_specfit_choice(pool, model, atlas):
    """Original objective, independent enumeration; also covers short generation tails."""
    from .serial_replay import route_atlas
    route_atlas(pool, atlas)  # Reject partial layers, repeated experts and bad IDs.
    masks = [sum(1 << (layer*128+expert) for layer in range(48) for expert in atlas[layer][raw])
             for raw in range(len(pool.nodes)+1)]
    probabilities = {n.raw_id: exp(n.cumulative_log_probability) for n in pool.nodes}
    best = None
    for count in range(min(7, len(pool.nodes))+1):
        for ids in enumerate_ancestor_closed_subsets(pool, draft_node_budget=count):
            union = masks[0]
            for raw in ids: union |= masks[raw]
            experts = union.bit_count()
            utility = 1.0 + sum((probabilities[raw] for raw in ids), start=0.0)
            key = (-utility/model.predict_ns(count, experts), -utility, experts, count, ids)
            if best is None or key < best: best = key
    return best[-1]


def exhaustive_revised_choice(pool, model, atlas):
    """Revised-cost objective on a depth<=d prefix by the same independent enumeration."""
    from .serial_replay import route_atlas
    route_atlas(pool, atlas)
    level = pool.nodes[-1].depth
    masks = [sum(1 << (layer*128+expert) for layer in range(48) for expert in atlas[layer][raw])
             for raw in range(len(pool.nodes)+1)]
    probabilities = {n.raw_id: exp(n.cumulative_log_probability) for n in pool.nodes}
    best = None
    for count in range(min(7, len(pool.nodes))+1):
        for ids in enumerate_ancestor_closed_subsets(pool, draft_node_budget=count):
            union = masks[0]
            for raw in ids: union |= masks[raw]
            experts = union.bit_count()
            utility = 1.0 + sum((probabilities[raw] for raw in ids), start=0.0)
            elapsed = model.predict_ns(count, experts, drafted_levels(pool.nodes, ids, level))
            key = (-utility/elapsed, -utility, experts, count, ids)
            if best is None or key < best: best = key
    return best[-1]


def verify_adaptive_depth(pool, model, atlas, limit, revised=None):
    """Recheck every level decision of one adaptive cycle by exhaustive search; return its selection."""
    depth = (len(pool.nodes) + 2) // 4
    if not 1 <= depth <= limit or len(atlas) != 48 or any(len(layer) != len(pool.nodes) + 1 for layer in atlas):
        raise ValueError("adaptive pool/routes exceed the generation tail or its prefix")
    for level in range(1, depth + 1):
        count = 4 * level - 2
        part = SimpleNamespace(nodes=pool.nodes[:count], seed_token_id=pool.seed_token_id)
        rows = [layer[:count + 1] for layer in atlas]
        ids = (exhaustive_specfit_choice(part, model, rows) if revised is None else
               exhaustive_revised_choice(part, revised, rows))
        picked = any(part.nodes[raw - 1].depth == level for raw in ids)
        if level < depth and not picked:
            raise ValueError("adaptive drafting continued past a level SpecFit left empty")
    if depth < limit and picked:
        raise ValueError("adaptive drafting stopped before an empty level or the tail limit")
    return ids


def predicted_routes(atlas, accuracy, key):
    """Simulated expert predictor: each routed expert of a node and layer is kept with probability `accuracy`,
    otherwise replaced by a uniformly drawn expert outside that node's true top-8. Every node and layer draws from
    its own stream keyed by `key`, so the prediction of a pool prefix is the prefix of the full-pool prediction."""
    if not 0 < accuracy <= 1:
        raise ValueError("route accuracy must be in (0, 1]")
    out = []
    for layer, rows in enumerate(atlas):
        predicted = []
        for raw, true in enumerate(rows):
            rng = random.Random(f"{ROUTE_SEED}|{'|'.join(map(str, key))}|{layer}|{raw}")
            kept = [e for e in true if rng.random() < accuracy]
            while len(kept) < len(true):
                expert = rng.randrange(128)
                if expert not in true and expert not in kept:
                    kept.append(expert)
            predicted.append(sorted(kept))
        out.append(predicted)
    return out


def load_prompt_budget(repository):
    """Merge audited 2048-token prompt budgets; tokenizer and length must agree, datasets must not overlap."""
    merged = None
    for relative in PROMPT_BUDGETS:
        budget = json.loads((repository / relative).read_text())
        if merged is None:
            merged = dict(budget, datasets=dict(budget["datasets"]))
            continue
        if (budget["total_tokens"] != merged["total_tokens"] or budget["tokenizer_files"] != merged["tokenizer_files"]
                or set(budget["datasets"]) & set(merged["datasets"])):
            raise ValueError("prompt budgets disagree")
        merged["datasets"].update(budget["datasets"])
    return merged


def load_workloads(gsm_path, prepared_root, repository, names=("humaneval", "aime25")):
    gsm_lock = json.loads((repository / "workloads/gsm8k_specbench.lock.json").read_text())["dataset"]
    actual = file_identity(gsm_path)
    if any(actual[k] != gsm_lock[k] for k in ("size_bytes", "sha256")):
        raise ValueError("GSM8K source identity changed")
    work = {"gsm8k": [dict(dataset="gsm8k", question_id=e.question_id, source_id=str(e.question_id),
        prompt=e.prompt, evaluation=dict(kind="gsm8k_explicit_number", answer=numeric_answer(e.reference[0], reference=True)))
        for e in load_gsm8k_specbench(gsm_path)]}
    identities = {"gsm8k": actual}
    budget = load_prompt_budget(repository)
    lock = json.loads((repository / "workloads/extended_benchmarks.lock.json").read_text())["datasets"]
    for name in names:
        spec = lock[name]
        path = prepared_root / name / spec["revision"] / "requests.jsonl"
        descriptor = file_identity(path)
        if descriptor["sha256"] != budget["datasets"][name]["prepared_sha256"]:
            raise ValueError("prepared dataset differs from audited inputs")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if [r["question_id"] for r in rows] != list(range(spec["count"])) or any(r["dataset"] != name for r in rows):
            raise ValueError("dataset coverage/order changed")
        work[name] = rows
        identities[name] = descriptor
    return work, identities


def numeric_grade(dataset, text, evaluation):
    if dataset == "humaneval":
        return dict(status="pending_isolated_tests", predicted=None, correct=None)
    if dataset == "summarization":
        return dict(status="not_scored", predicted=None, correct=None)
    value = numeric_answer(text)
    if dataset == "aime25" and value is not None:
        try:
            number = Decimal(value)
            value = str(int(number)) if number == number.to_integral_value() and 0 <= number <= 999 else None
        except (ValueError, InvalidOperation):
            value = None
    return dict(status="unparsed" if value is None else "parsed", predicted=value,
                correct=value is not None and value == evaluation["answer"])


def extract_python(text):
    """Use the single returned code block, or the complete raw response; no repair."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if len(blocks) > 1:
        raise ValueError("multiple code blocks are ambiguous")
    return blocks[0].strip() if blocks else text.strip()


def verify_record(row, reference, model, expected_total, route_accuracy=None, revised=None):
    """Recompute choices, termination, transfer conservation and recorded summaries."""
    method = row["method"]
    reference_method(method)
    if route_accuracy is not None and reference_method(method) != "specfit_adaptive":
        raise ValueError("simulated route prediction is limited to adaptive SpecFit")
    if revised is not None and reference_method(method) != "specfit_adaptive":
        raise ValueError("the revised cost model is limited to adaptive SpecFit")
    if row["status"] != "complete" or row["total_tokens"] != expected_total:
        raise ValueError("record status or total length changed")
    tokens = row["output_token_ids"]
    if row["output_tokens"] != len(tokens) or not 0 < len(tokens) <= expected_total - row["input_tokens"]:
        raise ValueError("invalid output length")
    eos = [i for i, t in enumerate(tokens) if t in (151643, 151645)]
    if row["stop_reason"] == "eos":
        if eos != [len(tokens) - 1] or row["stop_token_id"] != tokens[-1]:
            raise ValueError("EOS semantics changed")
    elif row["stop_reason"] != "length" or eos or len(tokens) + row["input_tokens"] != expected_total:
        raise ValueError("length termination changed")
    if not 0 < row["prefill_ns"] <= row["request_ns"] or row["decode_ns"] != row["request_ns"] - row["prefill_ns"]:
        raise ValueError("timing does not close")
    if reference is not None:
        if (reference["method"] != reference_method(method)
                or any(row[k] != reference[k] for k in ("dataset", "question_id", "input_ids_sha256", "output_token_ids", "query_proofs"))
                or row["reference_sha256"] != canonical_json_sha256(reference)):
            raise ValueError("generation differs from independent same-query reference")
    if not all(a["passed"] for a in row["commit_audits"]):
        raise ValueError("KV/aux commit audit failed")
    if reference is None and reference_method(method) not in ('ar', *CHAIN_METHODS) and len(row["commit_audits"]) != len(row["cycles"]):
        raise ValueError("reference commit audit coverage changed")
    if not all(p["finite"] for p in row["query_proofs"]):
        raise ValueError("nonfinite target output")
    committed = []
    if method in CHAIN_METHODS:
        verify_chain_cycles(row, reference)
    for i, cycle in enumerate(row["cycles"]):
        if cycle["iteration"] != i or cycle["committed_offset"] != len(committed):
            raise ValueError("cycle order/offset changed")
        if method in CHAIN_METHODS:
            committed.extend(cycle['committed_tokens'])
            continue
        pool = pool_view(cycle["pool"])
        atlas = None
        if method in ("specfit", "specfit_no_cache"):
            atlas = (reference or row)["cycles"][i]["atlas"]
            # Use the original exhaustive implementation independently of fast selection.
            ids = exhaustive_specfit_choice(pool, model, atlas)
        elif method == "specfit_packed":
            if reference is None:
                raise ValueError("packed SpecFit reuses the SpecFit reference")
            ids = exhaustive_specfit_choice(pool, model, reference["cycles"][i]["atlas"])
        elif reference_method(method) == "specfit_adaptive":
            limit = min(5, expected_total - row["input_tokens"] - cycle["committed_offset"] - 1)
            atlas = (reference or row)["cycles"][i]["atlas"]
            if route_accuracy is not None:
                # The run selected with simulated predictor routes; the stored atlas holds the true ones.
                atlas = predicted_routes(atlas, route_accuracy, (row["dataset"], row["question_id"], i))
            ids = verify_adaptive_depth(pool, model, atlas, limit, revised)
        else:
            ids = probability_ids(pool, min(7, len(pool.nodes)))
        if list(ids) != cycle["selected_raw_ids"]:
            raise ValueError("selected tree differs from frozen rule")
        ct = cycle["committed_tokens"]
        nodes = {n.raw_id: n for n in pool.nodes}
        if not ct or ct[0] != pool.seed_token_id or len(ct) > 1 + len(ids):
            raise ValueError("invalid accepted tree path")
        parent = 0
        for token in ct[1:]:
            children = [raw for raw in ids if nodes[raw].parent_raw_id == parent and nodes[raw].target_token_id == token]
            if len(children) != 1:
                raise ValueError("accepted path is not in selected tree")
            parent = children[0]
        committed.extend(ct)
    if reference_method(method) == "ar":
        if row["cycles"] or row["cycle_metrics"]:
            raise ValueError("AR must not call drafter")
    elif tokens[:len(committed)] != committed or not 0 <= len(tokens) - len(committed) <= 1:
        raise ValueError("tree commits do not reconstruct output")
    acceptance_lengths([dict(row, method='target_only' if reference_method(method) == 'ar' else method,
                            scheduler=dict(forwards=len(row['query_proofs'])-1))])
    before, after = row["pre_cache"], row["post_cache"]
    for side, layers in (("pre", before), ("post", after)):
        if canonical_json_sha256(layers) != row[side + "_cache_sha256"]:
            raise ValueError("cache boundary identity changed")
    for key, value in row["cache_delta"].items():
        if value < 0 or value != sum(b["stats"][key] - a["stats"][key] for a, b in zip(before, after, strict=True)):
            raise ValueError("cache counters do not close")
    if method in NO_CACHE:
        expected = sum(p["expert_union"] for p in row["query_proofs"])
        if row["cache_delta"]["h2d_submissions"] != expected or row["discard_calls"] != len(row["query_proofs"]):
            raise ValueError("no-cache H2D must equal every forward's actual expert union")
    elif row["discard_calls"] != 0 or row["discard_ns"] != 0:
        raise ValueError("cached method unexpectedly discarded experts")
    if row["max_gpu_reserved_bytes"] > 38 * 1024**3 or row["max_rss_bytes"] > 128 * 1024**3:
        raise ValueError("resource budget exceeded")


def summarize(rows):
    if not rows:
        raise ValueError("empty stream")
    lengths = acceptance_lengths([dict(r, method='target_only' if reference_method(r['method']) == 'ar' else r['method'],
        scheduler=dict(forwards=len(r['query_proofs'])-1)) for r in rows])
    return dict(requests=len(rows), output_tokens=sum(r["output_tokens"] for r in rows),
        decode_tokens=sum(r["output_tokens"] - 1 for r in rows),
        request_ns=sum(r["request_ns"] for r in rows), decode_ns=sum(r["decode_ns"] for r in rows),
        decode_tps=sum(r["output_tokens"] - 1 for r in rows) * 1e9 / sum(r["decode_ns"] for r in rows),
        request_tps=sum(r["output_tokens"] for r in rows) * 1e9 / sum(r["request_ns"] for r in rows),
        truncated=sum(r["stop_reason"] == "length" for r in rows),
        numeric_correct=sum(r["quality"]["correct"] is True for r in rows),
        quality_status=dict(Counter(r["quality"]["status"] for r in rows)),
        cache_delta={k: sum(r["cache_delta"][k] for r in rows) for k in rows[0]["cache_delta"]},
        discard_ns=sum(r["discard_ns"] for r in rows), acceptance_lengths=lengths)
