"""Private, decision-equivalent SpecFit implementation for measured experiments.

Only topology and native tuple gatherers are cached. Every invocation validates
and packs fresh routes, reads fresh probabilities, and scores every legal tree.
"""
from functools import lru_cache
from math import exp
from operator import itemgetter

from .cache_selection_diagnostic import _topology
from .selection_methods import SelectionCostModel
from .selection_refinement import CacheCostModel

_BITS = tuple(1 << e for e in range(128))
_LAYERS = tuple(1 << lid for lid in range(48))


def _pack_routes(pool, atlas):
    """Validate complete Top-8 routes, then pack each node into one Python int."""
    count = len(pool.nodes)
    expected = tuple(range(count + 1))
    if (count not in (2, 6, 10, 14, 18)
            or (0, *(n.raw_id for n in pool.nodes)) != expected
            or atlas.position_to_raw != expected
            or len(atlas.expert_ids_by_raw_layer) != (count + 1) * 48):
        raise ValueError("atlas must cover every candidate and all 48 layers")
    observed, packed = [0] * (count + 1), [0] * (count + 1)
    bits, layers = _BITS, _LAYERS
    for raw, layer, experts in atlas.expert_ids_by_raw_layer:
        if (type(raw) is not int or not 0 <= raw <= count
                or type(layer) is not int or not 0 <= layer < 48 or len(experts) != 8):
            raise ValueError("invalid atlas raw/layer/Top-8")
        bit = layers[layer]
        if observed[raw] & bit:
            raise ValueError("duplicate atlas raw/layer")
        observed[raw] |= bit
        a, b, c, d, e, f, g, h = experts
        # Exact int checks retain rejection of bools and integer subclasses.
        # The combined high bits detect both negative and >=128 expert IDs.
        if (type(a) is not int or type(b) is not int or type(c) is not int
                or type(d) is not int or type(e) is not int or type(f) is not int
                or type(g) is not int or type(h) is not int
                or (a | b | c | d | e | f | g | h) & ~127):
            raise ValueError("invalid expert ID")
        mask = bits[a] | bits[b] | bits[c] | bits[d] | bits[e] | bits[f] | bits[g] | bits[h]
        if mask.bit_count() != 8:
            raise ValueError("duplicate expert ID")
        packed[raw] |= mask << (128 * layer)
    return tuple(packed)


@lru_cache(maxsize=64)
def _search_rows(parents):
    return tuple((parent, raw, ids, len(ids), itemgetter(*ids) if len(ids) > 1 else None)
                 for parent, raw, ids in _topology(parents))


def exact_specfit_selection(pool, atlas, model, resident_mask=0):
    """Preserve raw-probability objective, builtin summation and full tie order."""
    if not isinstance(model, (SelectionCostModel, CacheCostModel)):
        raise ValueError("a frozen SpecFit cost model is required")
    if type(resident_mask) is not int or not 0 <= resident_mask < 1 << (48 * 128):
        raise ValueError("invalid resident bitmap")
    bitmaps = _pack_routes(pool, atlas)
    return _score_packed_routes(pool, bitmaps, model, resident_mask)


def _score_packed_routes(pool, bitmaps, model, resident_mask, *, probabilities=None):
    """Score freshly validated routes; callers own input-boundary validation."""
    rows = _search_rows(tuple(n.parent_raw_id for n in pool.nodes))
    if probabilities is None:
        probabilities = (0.0, *(exp(n.cumulative_log_probability) for n in pool.nodes))
    offsets, slope = model.size_intercept_ns, model.expert_ns
    missing_slope = model.missing_ns if isinstance(model, CacheCostModel) else 0.0
    missing_mask = ~resident_mask
    footprints = []
    best_score, best_tie, best_ids = -1.0, None, None
    for parent, raw, ids, count, gather in rows:
        footprint = bitmaps[0] if parent < 0 else footprints[parent] | bitmaps[raw]
        footprints.append(footprint)
        experts = footprint.bit_count()
        elapsed = offsets[count] + slope * experts
        if missing_slope:
            elapsed += missing_slope * (footprint & missing_mask).bit_count()
        # itemgetter constructs the same ordered tuple in C, avoiding a Python
        # generator for every tree. Keep builtin sum (including compensated sum).
        reward = (1.0 + sum(gather(probabilities), start=0.0) if count > 1 else
                  1.0 + probabilities[raw] if count == 1 else 1.0)
        score = reward / elapsed
        if score >= best_score:
            tie = (-reward, experts, count, ids)
            if score > best_score or best_tie is None or tie < best_tie:
                best_score, best_tie, best_ids = score, tie, ids
    return best_ids
