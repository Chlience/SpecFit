"""Five named tree-selection methods and a freshly calibrated cycle-cost proxy."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, isfinite, sqrt
from statistics import mean

from .f0_direct import _atlas_bitmaps
from .r6a_selection_contract import enumerate_ancestor_closed_subsets

METHODS = (
    "target_only", "probability_fixed", "expert_cost_fixed",
    "probability_dynamic", "expert_cost_dynamic",
)
METHOD_NAMES = {
    "target_only": "no draft generated",
    "probability_fixed": "fixed-probability tree selection",
    "expert_cost_fixed": "fixed expert-cost selection tree",
    "probability_dynamic": "dynamic probability tree selection",
    "expert_cost_dynamic": "dynamic expert-cost tree selection",
}
MAX_DRAFT_NODES = 7


@dataclass(frozen=True)
class SelectionCostModel:
    """A nonnegative intercept for each query size and one expert-union slope.

    The probability baseline uses only the observed size-specific mean. Expert
    methods share the same calibrated model for fixed and variable budgets.
    Times include the complete draft pool and exclude the selector itself.
    """

    size_mean_ns: tuple[float, ...]
    size_intercept_ns: tuple[float, ...]
    expert_ns: float

    def __post_init__(self):
        if (len(self.size_mean_ns) != MAX_DRAFT_NODES + 1
                or len(self.size_intercept_ns) != MAX_DRAFT_NODES + 1
                or any(isinstance(x, bool) or not isfinite(x) or x <= 0
                       for x in self.size_mean_ns)
                or any(isinstance(x, bool) or not isfinite(x) or x < 0
                       for x in (*self.size_intercept_ns, self.expert_ns))
                or any(b + 384 * self.expert_ns <= 0 for b in self.size_intercept_ns)):
            raise ValueError("invalid calibrated selection cost model")

    def predict_ns(self, node_count: int, expert_union: int | None = None) -> float:
        if type(node_count) is not int or not 0 <= node_count <= MAX_DRAFT_NODES:
            raise ValueError("draft node count is outside the calibrated range")
        if expert_union is None:
            return self.size_mean_ns[node_count]
        if type(expert_union) is not int or expert_union < 384:
            raise ValueError("expert union must include the mandatory seed in all layers")
        return self.size_intercept_ns[node_count] + self.expert_ns * expert_union

    def to_dict(self) -> dict:
        value = asdict(self)
        return {**value, "size_mean_ns": list(self.size_mean_ns),
                "size_intercept_ns": list(self.size_intercept_ns)}

    @classmethod
    def from_dict(cls, value: dict) -> SelectionCostModel:
        if set(value) != {"size_mean_ns", "size_intercept_ns", "expert_ns"}:
            raise ValueError("unexpected cost-model fields")
        return cls(tuple(value["size_mean_ns"]), tuple(value["size_intercept_ns"]), value["expert_ns"])


def fit_cost_model(points: list[dict]) -> tuple[SelectionCostModel, dict]:
    """Fit nonnegative size intercepts and an expert slope without a dependency.

    For a fixed slope, each optimal intercept is max(0, mean(y)-slope*mean(F)).
    Its eight breakpoints partition a one-dimensional convex least-squares
    problem. Solve the quadratic in each interval and compare the candidates.
    Each input point is one calibration-state/selected-tree repetition mean.
    """
    groups = {k: [] for k in range(MAX_DRAFT_NODES + 1)}
    for row in points:
        k, footprint, elapsed = (row[key] for key in
            ("draft_node_count", "expert_union", "cycle_without_selector_ns"))
        if (type(k) is not int or k not in groups or type(footprint) is not int
                or footprint < 384 or isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float)) or not isfinite(elapsed) or elapsed <= 0):
            raise ValueError("invalid calibration point")
        groups[k].append((footprint, float(elapsed)))
    if any(not rows for rows in groups.values()):
        raise ValueError("calibration must cover every draft node count 0 through 7")
    centers = {k: (mean(x for x, _ in rows), mean(y for _, y in rows))
               for k, rows in groups.items()}
    breaks = sorted({0.0, *(y / x for x, y in centers.values())})
    candidates = set(breaks)
    for low, high in zip(breaks, (*breaks[1:], float("inf")), strict=True):
        middle = low + 1.0 if high == float("inf") else (low + high) / 2
        numerator = denominator = 0.0
        for k, rows in groups.items():
            mx, my = centers[k]
            active_intercept = my - middle * mx > 0
            for x, y in rows:
                dx, dy = (x - mx, y - my) if active_intercept else (x, y)
                numerator += dx * dy
                denominator += dx * dx
        if denominator:
            candidates.add(max(low, min(high, numerator / denominator)))

    def residual(slope):
        offsets = {k: max(0.0, my - slope * mx) for k, (mx, my) in centers.items()}
        error = sum((y - offsets[k] - slope * x) ** 2
                    for k, rows in groups.items() for x, y in rows)
        return error, offsets

    slope = min(candidates, key=lambda candidate: (residual(candidate)[0], candidate))
    squared_error, offsets = residual(slope)
    model = SelectionCostModel(tuple(centers[k][1] for k in groups),
                               tuple(offsets[k] for k in groups), slope)
    errors = [(model.predict_ns(row["draft_node_count"], row["expert_union"])
               / row["cycle_without_selector_ns"] - 1) for row in points]
    return model, {"point_count": len(points), "rmse_ns": sqrt(squared_error / len(points)),
        "mean_absolute_relative_error": mean(abs(x) for x in errors),
        "max_absolute_relative_error": max(abs(x) for x in errors),
        "expert_slope_at_zero": slope == 0.0,
        "fit_scope": "calibration_only_no_evaluation_timing_or_acceptance"}


def probability_ids(pool, node_count: int) -> tuple[int, ...]:
    if type(node_count) is not int or not 0 <= node_count <= len(pool.nodes):
        raise ValueError("invalid probability-selection node count")
    ranked = sorted(pool.nodes, key=lambda n:
                    (-n.cumulative_log_probability, n.depth, n.generation_ordinal))
    return tuple(sorted(n.raw_id for n in ranked[:node_count]))


def _candidate_scores(pool, atlas, counts):
    bitmaps = _atlas_bitmaps(pool, atlas)
    probabilities = {n.raw_id: exp(n.cumulative_log_probability) for n in pool.nodes}
    for count in counts:
        for ids in enumerate_ancestor_closed_subsets(pool, draft_node_budget=count):
            footprint = bitmaps[0]
            for raw_id in ids:
                footprint |= bitmaps[raw_id]
            yield ids, 1.0 + sum((probabilities[i] for i in ids), start=0.0), footprint.bit_count()


def selection_features(pool, atlas, selected: tuple[int, ...]) -> dict:
    bitmaps = _atlas_bitmaps(pool, atlas)
    by_id = {n.raw_id: n for n in pool.nodes}
    if (len(set(selected)) != len(selected) or any(i not in by_id for i in selected)
            or any(by_id[i].parent_raw_id not in (0, *selected) for i in selected)):
        raise ValueError("selected nodes must be unique, present and ancestor closed")
    footprint = bitmaps[0]
    for raw_id in selected:
        footprint |= bitmaps[raw_id]
    return {"draft_node_count": len(selected), "expert_union_proxy": footprint.bit_count(),
        "expected_committed_tokens_proxy": 1.0 + sum(
            (exp(by_id[i].cumulative_log_probability) for i in selected), start=0.0)}


def minimum_expert_ids(pool, atlas, node_count: int) -> tuple[int, ...]:
    """A calibration probe, independent of accepted tokens and measured times."""
    return min(_candidate_scores(pool, atlas, (node_count,)),
               key=lambda row: (row[2], -row[1], row[0]))[0]


def select_nodes(pool, atlas, method: str, model: SelectionCostModel | None = None) -> tuple[int, ...]:
    if method not in METHODS:
        raise ValueError("unknown selection method; use a descriptive method identifier")
    if method == "target_only":
        return ()
    if method == "probability_fixed":
        return probability_ids(pool, MAX_DRAFT_NODES)
    if model is None:
        raise ValueError("this method requires a frozen calibration model")
    if method == "probability_dynamic":
        probabilities = {n.raw_id: exp(n.cumulative_log_probability) for n in pool.nodes}
        candidates = []
        # A generation tail can have fewer than seven candidates. Its admissible
        # sizes are bounded by the actual pool, with the same calibrated costs.
        for count in range(min(MAX_DRAFT_NODES, len(pool.nodes)) + 1):
            ids = probability_ids(pool, count)
            utility = 1.0 + sum((probabilities[i] for i in ids), start=0.0)
            candidates.append((ids, utility, model.predict_ns(count)))
        return min(candidates, key=lambda row:
                   (-row[1] / row[2], -row[1], len(row[0]), row[0]))[0]
    counts = (MAX_DRAFT_NODES,) if method == "expert_cost_fixed" else range(MAX_DRAFT_NODES + 1)
    return min(_candidate_scores(pool, atlas, counts), key=lambda row:
        (-row[1] / model.predict_ns(len(row[0]), row[2]), -row[1], row[2], len(row[0]), row[0]))[0]
