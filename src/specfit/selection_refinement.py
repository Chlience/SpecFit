"""Independent cache-aware selector variants; the original five methods stay intact.

All online decisions consume only the live draft pool, an offline route proxy,
current resident expert IDs, and coefficients fitted on separate prompts.
Measured acceptance and elapsed times are inputs to fitting, never selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log
from statistics import mean

from .f0_direct import _atlas_bitmaps as _full_atlas_bitmaps
from .selection_methods import METHODS, METHOD_NAMES, SelectionCostModel, fit_cost_model, select_nodes

VARIANTS = ("expert_fast_dynamic", "expert_cache_dynamic",
            "expert_calibrated_dynamic", "expert_guarded_dynamic")
REFINEMENT_METHODS = (*METHODS, *VARIANTS)
REFINEMENT_NAMES = {**METHOD_NAMES,
    "expert_fast_dynamic": "dynamic expert cost + incremental search",
    "expert_cache_dynamic": "dynamic expert cost + cache-aware",
    "expert_calibrated_dynamic": "cache-aware + acceptance-gain calibration",
    "expert_guarded_dynamic": "cache-aware + gain-calibrated + conservative fallback"}
TEMPERATURE_EXPONENTS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def _atlas_bitmaps(pool, atlas):
    """Preserve full-pool behavior and support shorter generation-tail pools."""
    count = len(pool.nodes)
    if count == 18:
        return _full_atlas_bitmaps(pool, atlas)
    expected = tuple(range(count + 1))
    if (count not in (2, 6, 10, 14)
            or (0, *(n.raw_id for n in pool.nodes)) != expected
            or atlas.position_to_raw != expected
            or len(atlas.expert_ids_by_raw_layer) != (count + 1) * 48):
        raise ValueError("tail atlas must cover every candidate and all 48 layers")
    observed = set()
    bitmaps = [0] * (count + 1)
    for raw, layer, experts in atlas.expert_ids_by_raw_layer:
        if (type(raw) is not int or raw not in expected
                or type(layer) is not int or not 0 <= layer < 48
                or (raw, layer) in observed or len(experts) != 8
                or any(type(e) is not int or not 0 <= e < 128 for e in experts)
                or len(set(experts)) != 8):
            raise ValueError("invalid tail atlas raw/layer/Top-8 coverage")
        observed.add((raw, layer))
        bitmaps[raw] |= sum(1 << (128 * layer + expert) for expert in experts)
    return tuple(bitmaps)


def _finite_nonnegative(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and isfinite(value) and value >= 0)


def resident_bitmap(experts_by_layer) -> int:
    """Encode the observable resident set, preserving every expert's layer."""
    if len(experts_by_layer) != 48:
        raise ValueError("resident information must cover all 48 layers")
    bitmap = 0
    for lid, experts in enumerate(experts_by_layer):
        seen = set()
        for expert in experts:
            if type(expert) is not int or not 0 <= expert < 128 or expert in seen:
                raise ValueError("invalid or duplicate resident expert")
            seen.add(expert)
            bitmap |= 1 << (lid * 128 + expert)
        if len(seen) > 48:
            raise ValueError("resident set exceeds the fixed layer capacity")
    return bitmap


def residents_from_logical_state(logical):
    layers = logical["layers"]
    if [layer["layer_id"] for layer in layers] != list(range(48)):
        raise ValueError("cache layers are missing or reordered")
    return tuple(tuple(slot["expert_id"] for slot in layer["slots"]
                       if slot["expert_id"] is not None) for layer in layers)


@dataclass(frozen=True)
class CacheCostModel:
    """Size intercepts plus compute work and initially missing expert work."""

    size_intercept_ns: tuple[float, ...]
    expert_ns: float
    missing_ns: float

    def __post_init__(self):
        if (len(self.size_intercept_ns) != 8
                or any(not _finite_nonnegative(x) for x in
                       (*self.size_intercept_ns, self.expert_ns, self.missing_ns))
                or any(b + 384 * self.expert_ns <= 0 for b in self.size_intercept_ns)):
            raise ValueError("invalid cache-aware cost coefficients")

    def predict_ns(self, count, footprint, missing):
        if (type(count) is not int or not 0 <= count <= 7
                or type(footprint) is not int or footprint < 384
                or type(missing) is not int or not 0 <= missing <= footprint):
            raise ValueError("invalid cache-aware query features")
        return self.size_intercept_ns[count] + self.expert_ns * footprint + self.missing_ns * missing

    def to_dict(self):
        return dict(size_intercept_ns=list(self.size_intercept_ns),
                    expert_ns=self.expert_ns, missing_ns=self.missing_ns)

    @classmethod
    def from_dict(cls, value):
        if set(value) != {"size_intercept_ns", "expert_ns", "missing_ns"}:
            raise ValueError("unexpected cache model fields")
        return cls(tuple(value["size_intercept_ns"]), value["expert_ns"], value["missing_ns"])


def fit_cache_cost_model(points, *, mode):
    """Nonnegative least squares by enumerating the eight intercept active sets.

    Cold misses equal the union; collapse the two collinear terms to the existing
    one-slope model. Natural-cache coefficients are identified from independent
    prompt/boundary/size probes. No additional numerical dependency is needed.
    """
    if mode not in ("cold", "natural"):
        raise ValueError("unknown cache mode")
    legacy, _ = fit_cost_model(points)
    for row in points:
        f, n = row["expert_union"], row["initial_missing"]
        if type(n) is not int or not 0 <= n <= f or (mode == "cold" and n != f):
            raise ValueError("invalid initial miss count for the cache mode")
    if mode == "cold":
        model = CacheCostModel(legacy.size_intercept_ns, legacy.expert_ns, 0.0)
        candidate_count = 0
    else:
        groups = {k: [] for k in range(8)}
        for row in points:
            groups[row["draft_node_count"]].append((row["expert_union"] / 1024,
                row["initial_missing"] / 1024, row["cycle_without_selector_ns"] / 1e6))
        centers = {k: tuple(mean(row[j] for row in rows) for j in range(3))
                   for k, rows in groups.items()}
        # Given active intercepts the remaining problem is a 2x2 quadratic.
        # Include its two slope boundaries. This also covers singular columns
        # without iterative convergence or a platform-dependent solver.
        moments = {}
        for k, rows in groups.items():
            for active in (False, True):
                cx, cz, cy = centers[k] if active else (0.0, 0.0, 0.0)
                shifted = [(x - cx, z - cz, y - cy) for x, z, y in rows]
                moments[k, active] = (sum(x*x for x, z, y in shifted),
                    sum(x*z for x, z, y in shifted), sum(z*z for x, z, y in shifted),
                    sum(x*y for x, z, y in shifted), sum(z*y for x, z, y in shifted))
        candidates = {(0.0, 0.0)}
        for active_mask in range(256):
            xx, xz, zz, xy, zy = (sum(moments[k, bool(active_mask & (1 << k))][j]
                                    for k in range(8)) for j in range(5))
            if xx > 0:
                candidates.add((max(0.0, xy / xx), 0.0))
            if zz > 0:
                candidates.add((0.0, max(0.0, zy / zz)))
            determinant = xx * zz - xz * xz
            if determinant > 1e-12 * max(xx * zz, 1.0):
                a, c = (xy * zz - zy * xz) / determinant, (zy * xx - xy * xz) / determinant
                if a >= 0 and c >= 0:
                    candidates.add((a, c))
        def residual(slopes):
            a, c = slopes
            offsets = {k: max(0.0, y - a*x - c*z) for k, (x, z, y) in centers.items()}
            error = sum((y - offsets[k] - a*x - c*z) ** 2
                        for k, rows in groups.items() for x, z, y in rows)
            return error, offsets
        slopes = min(candidates, key=lambda ac: (residual(ac)[0], ac))
        _, offsets = residual(slopes)
        candidate_count = len(candidates)
        model = CacheCostModel(tuple(offsets[k] * 1e6 for k in range(8)),
                               slopes[0] * 1e6 / 1024, slopes[1] * 1e6 / 1024)
    errors = [model.predict_ns(r["draft_node_count"], r["expert_union"], r["initial_missing"])
              / r["cycle_without_selector_ns"] - 1 for r in points]
    return model, dict(point_count=len(points), solver="nonnegative_intercept_active_sets",
        slope_candidates=candidate_count,
        mean_absolute_relative_error=mean(abs(e) for e in errors),
        max_absolute_relative_error=max(abs(e) for e in errors),
        cold_collinear_terms_collapsed=mode == "cold")


def conditional_groups(pool):
    groups = {}
    for node in pool.nodes:
        groups.setdefault(node.parent_raw_id, []).append(node)
    result = []
    for parent, children in sorted(groups.items()):
        probabilities = tuple(exp(node.conditional_log_probability) for node in children)
        if len(children) != 2 or sum(probabilities) > 1.0 + 1e-6:
            raise ValueError("acceptance calibration requires the complete Top-2 pool")
        result.append((parent, tuple(children), (*probabilities, max(0.0, 1.0 - sum(probabilities)))))
    return tuple(result)


def temper(probabilities, exponent):
    if (isinstance(exponent, bool) or exponent not in TEMPERATURE_EXPONENTS
            or any(not _finite_nonnegative(p) or p > 1 for p in probabilities)
            or not 0.999999 <= sum(probabilities) <= 1.000001):
        raise ValueError("invalid categorical acceptance probabilities")
    values = tuple(p ** exponent for p in probabilities)
    total = sum(values)
    return tuple(v / total for v in values)


def acceptance_probabilities(pool, exponent):
    """Temper sibling/other probabilities, then multiply along each draft path."""
    if isinstance(exponent, bool) or exponent not in TEMPERATURE_EXPONENTS:
        raise ValueError("invalid acceptance temperature exponent")
    if exponent == 1.0:
        return {node.raw_id: exp(node.cumulative_log_probability) for node in pool.nodes}
    conditional = {}
    for _, children, probabilities in conditional_groups(pool):
        calibrated = temper(probabilities, exponent)
        conditional.update((node.raw_id, calibrated[i]) for i, node in enumerate(children))
    cumulative = {0: 1.0}
    for node in pool.nodes:
        cumulative[node.raw_id] = cumulative[node.parent_raw_id] * conditional[node.raw_id]
    cumulative.pop(0)
    return cumulative


def acceptance_observations(pool, verifier_token_ids, question_id):
    """Training-only categorical labels from an independent full-pool query."""
    if len(verifier_token_ids) != 19:
        raise ValueError("training labels require all full-pool verifier positions")
    rows = []
    for parent, children, probabilities in conditional_groups(pool):
        labels = [i for i, node in enumerate(children)
                  if node.target_token_id == verifier_token_ids[parent]]
        rows.append(dict(question_id=question_id, parent_raw_id=parent,
                         probabilities=list(probabilities), label=labels[0] if labels else 2))
    return rows


def acceptance_loss(rows, exponent):
    if not rows:
        raise ValueError("acceptance calibration is empty")
    scores = []
    for row in rows:
        label = row["label"]
        if type(label) is not int or not 0 <= label < 3:
            raise ValueError("invalid acceptance label")
        p = temper(row["probabilities"], exponent)
        scores.append(sum((value - int(i == label)) ** 2 for i, value in enumerate(p)))
    return mean(scores)


def fit_acceptance(rows):
    """Choose one scalar on training prompts; report prompt-held-out Brier loss."""
    questions = sorted({r["question_id"] for r in rows})
    if len(questions) < 3:
        raise ValueError("acceptance calibration needs at least three training prompts")
    def choose(training):
        return min(TEMPERATURE_EXPONENTS,
                   key=lambda x: (acceptance_loss(training, x), abs(log(x)), x))
    folds = []
    for qid in questions:
        exponent = choose([r for r in rows if r["question_id"] != qid])
        held_out = [r for r in rows if r["question_id"] == qid]
        folds.append(dict(question_id=qid, exponent=exponent,
                          raw_brier=acceptance_loss(held_out, 1.0),
                          calibrated_brier=acceptance_loss(held_out, exponent)))
    # Retain raw probabilities unless the grouped validation improves their loss.
    accepted = mean(r["calibrated_brier"] for r in folds) < mean(r["raw_brier"] for r in folds)
    exponent = choose(rows) if accepted else 1.0
    return exponent, dict(training_questions=questions, observations=len(rows), folds=folds,
        accepted_by_grouped_validation=accepted,
        raw_brier=acceptance_loss(rows, 1.0), fitted_brier=acceptance_loss(rows, exponent))


def query_features(pool, atlas, selected, resident_mask):
    if type(resident_mask) is not int or not 0 <= resident_mask < 1 << (48 * 128):
        raise ValueError("invalid resident bitmap")
    bitmaps = _atlas_bitmaps(pool, atlas)
    by_id = {n.raw_id: n for n in pool.nodes}
    if (tuple(sorted(set(selected))) != tuple(selected)
            or any(i not in by_id or by_id[i].parent_raw_id not in (0, *selected) for i in selected)):
        raise ValueError("query must be a sorted ancestor-closed subset")
    footprint = bitmaps[0]
    for raw in selected:
        footprint |= bitmaps[raw]
    return dict(draft_node_count=len(selected), expert_union=footprint.bit_count(),
                initial_missing=(footprint & ~resident_mask).bit_count())


def fast_expert_selection(pool, atlas, model, *, resident_mask=0, exponent=1.0,
                          fallback_ids=None, guard_margin=0.0):
    """Search every legal subset once, carrying its expert union.

    Reuse the original builtin sum for rewards: Python's compensated summation
    can differ from incremental addition and change exact objective ties.
    """
    if not isinstance(model, (SelectionCostModel, CacheCostModel)):
        raise ValueError("unsupported selection cost model")
    if type(resident_mask) is not int or not 0 <= resident_mask < 1 << (48 * 128):
        raise ValueError("invalid resident bitmap")
    if not _finite_nonnegative(guard_margin) or guard_margin > 1:
        raise ValueError("invalid relative guard margin")
    bitmaps = _atlas_bitmaps(pool, atlas)
    probabilities = acceptance_probabilities(pool, exponent)
    nodes = tuple((n.raw_id, 1 << n.raw_id, 1 << n.parent_raw_id) for n in pool.nodes)
    offsets, slope = model.size_intercept_ns, model.expert_ns
    missing_slope = model.missing_ns if isinstance(model, CacheCostModel) else 0.0
    selected = []
    best_key = None
    best_ids = None
    fallback_score = None

    def visit(start, included, footprint):
        nonlocal best_key, best_ids, fallback_score
        count, experts = len(selected), footprint.bit_count()
        elapsed = offsets[count] + slope * experts
        if missing_slope:
            elapsed += missing_slope * (footprint & ~resident_mask).bit_count()
        utility = 1.0 + sum((probabilities[raw] for raw in selected), start=0.0)
        ids = tuple(selected)
        key = (-utility / elapsed, -utility, experts, count, ids)
        if ids == fallback_ids:
            fallback_score = utility / elapsed
        if best_key is None or key < best_key:
            best_key, best_ids = key, ids
        if count == 7:
            return
        for index in range(start, len(nodes)):
            raw, node_bit, parent_bit = nodes[index]
            if included & parent_bit:
                selected.append(raw)
                visit(index + 1, included | node_bit, footprint | bitmaps[raw])
                selected.pop()

    visit(0, 1, bitmaps[0])
    if fallback_ids is not None:
        if fallback_score is None:
            raise ValueError("fallback must be an admissible candidate")
        if -best_key[0] <= fallback_score * (1.0 + guard_margin):
            return fallback_ids
    return best_ids


def select_refined(pool, atlas, method, *, legacy_model, cache_model,
                   resident_mask=0, acceptance_exponent=1.0, guard_margin=0.02):
    if method in METHODS:
        return select_nodes(pool, atlas, method, legacy_model)
    if method not in VARIANTS:
        raise ValueError("unknown refinement method")
    if not _finite_nonnegative(guard_margin) or guard_margin > 1:
        raise ValueError("invalid relative guard margin")
    if method == "expert_fast_dynamic":
        return fast_expert_selection(pool, atlas, legacy_model)
    exponent = 1.0 if method == "expert_cache_dynamic" else acceptance_exponent
    # The complete search remains charged to the guarded method, including when
    # it falls back. The margin controls decision risk, not already-paid work.
    baseline = (select_nodes(pool, atlas, "probability_dynamic", legacy_model)
                if method == "expert_guarded_dynamic" else None)
    return fast_expert_selection(pool, atlas, cache_model, resident_mask=resident_mask,
        exponent=exponent, fallback_ids=baseline, guard_margin=guard_margin)
