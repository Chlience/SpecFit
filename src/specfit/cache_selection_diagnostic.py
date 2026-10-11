"""Exact cache-cost selection and paired-diagnostic contracts (CPU only).

Only tree topology is cached. Probabilities, routes, residency and all scores
are rebuilt on every call; no observed acceptance or timing enters selection.
"""
from functools import lru_cache
from math import exp

from .selection_refinement import CacheCostModel, _atlas_bitmaps

METHODS = ("target_only", "probability_dynamic", "expert_fast_dynamic",
           "expert_cache_dynamic", "expert_cache_compact")
QUESTIONS = (405, 415, 425, 435, 445, 455, 465, 475)
BOUNDARIES = (0, 4)
REPETITIONS = 5
PROTOCOL = "a100-cache-selection-diagnostic-v1"


@lru_cache(maxsize=64)
def _topology(parents):
    """Return parent-row, added-node, and IDs for every legal <=7-node tree."""
    if (len(parents) not in (2, 6, 10, 14, 18)
            or any(type(p) is not int or not 0 <= p < raw
                   for raw, p in enumerate(parents, 1))):
        raise ValueError("invalid candidate topology")
    rows = [(-1, 0, ())]

    def visit(start, included, ids, row_index):
        if len(ids) == 7:
            return
        for raw in range(start, len(parents) + 1):
            if included & (1 << parents[raw - 1]):
                chosen = (*ids, raw)
                child = len(rows)
                rows.append((row_index, raw, chosen))
                visit(raw + 1, included | (1 << raw), chosen, child)

    visit(1, 1, (), 0)
    return tuple(rows)


def compact_cache_selection(pool, atlas, model, resident_mask):
    """Exactly preserve the original cache selector's objective and tie order."""
    if not isinstance(model, CacheCostModel):
        raise ValueError("a frozen cache cost model is required")
    if type(resident_mask) is not int or not 0 <= resident_mask < 1 << (48 * 128):
        raise ValueError("invalid resident bitmap")
    bitmaps = _atlas_bitmaps(pool, atlas)
    parents = tuple(n.parent_raw_id for n in pool.nodes)
    rows = _topology(parents)
    probabilities = [0.0, *(exp(n.cumulative_log_probability) for n in pool.nodes)]
    footprints = []
    missing_mask = ~resident_mask
    best_key = None
    best_ids = None
    for parent_row, raw, ids in rows:
        footprint = bitmaps[0] if parent_row < 0 else footprints[parent_row] | bitmaps[raw]
        footprints.append(footprint)
        experts = footprint.bit_count()
        elapsed = model.size_intercept_ns[len(ids)] + model.expert_ns * experts
        if model.missing_ns:
            elapsed += model.missing_ns * (footprint & missing_mask).bit_count()
        # Keep Python's builtin sum and increasing raw-ID order, including on
        # Python versions with compensated summation. Incremental sums can alter ties.
        reward = 1.0 + sum((probabilities[i] for i in ids), start=0.0)
        key = (-reward / elapsed, -reward, experts, len(ids), ids)
        if best_key is None or key < best_key:
            best_key, best_ids = key, ids
    return best_ids


def schedule(ordinal, repetitions=REPETITIONS):
    """Balanced rotation across one audit, formal repeats and diagnostic repeats."""
    rows = []
    for phase, count in (("audit", 1), ("measure", repetitions), ("dependency_trace", 2)):
        for repetition in range(count):
            shift = (ordinal + repetition) % len(METHODS)
            for method in METHODS[shift:] + METHODS[:shift]:
                rows.append(dict(execution_index=len(rows), phase=phase,
                                 repetition=repetition, method=method))
    return rows


def own_missing(record):
    """Count actual-query experts absent at the pre-verification boundary."""
    residents = {(layer["layer_id"], slot["expert_id"])
                 for layer in record["pre_logical_state"]["layers"]
                 for slot in layer["slots"] if slot["expert_id"] is not None}
    experts = {(row["layer_id"], e) for row in record["routes"]["rows"]
               for e in row["expert_ids"]}
    return len(experts - residents)
