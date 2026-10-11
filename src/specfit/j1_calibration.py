
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from .m1_screen import CandidateScore


J1_G0_SOURCE = {
    "basename": "g0-resident-first-q480-48slots-e890614.json",
    "size_bytes": 943_367,
    "sha256": "8e59a2dd3f2993a6f29bcc2c1027554eaa1e7d46cc25251e35646ab8e7c6bc17",
    "git_commit": "e890614b982091faba105a26e13067fc8b106db1",
}


@dataclass(frozen=True, slots=True)
class CalibrationPoint:

    selection_policy: str
    expert_acquisitions: int
    h2d_submissions: int
    median_cycle_wall_ns: int


@dataclass(frozen=True, slots=True)
class CostModel:

    fixed_ns: float
    expert_ns: float
    h2d_ns: float

    def predict_ns(self, row: CandidateScore) -> float:
        value = (
            self.fixed_ns
            + self.expert_ns * row.F_atlas
            + self.h2d_ns * row.N_atlas
        )
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("J1 cycle time prediction must be a positive finite value")
        return value


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be no less than {minimum} as an integer")
    return value


def _mapping(name: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _sequence(name: str, value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an array")
    return value


def _read_fixed_json(path: Path) -> dict[str, object]:
    if (
        path.name != J1_G0_SOURCE["basename"]
        or path.is_symlink()
        or not path.is_file()
    ):
        raise ValueError("calibration source basename, symlink, or file type violates the contract")
    raw = path.read_bytes()
    if (
        len(raw) != J1_G0_SOURCE["size_bytes"]
        or hashlib.sha256(raw).hexdigest() != J1_G0_SOURCE["sha256"]
    ):
        raise ValueError("calibration source size/SHA-256 does not match the contract")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Calibration source is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("calibration source root node must be an object")
    return value


def read_g0_calibration_points(path: Path) -> tuple[CalibrationPoint, ...]:

    artifact = _read_fixed_json(path)
    environment = _mapping("G0 environment", artifact.get("environment"))
    config = _mapping("G0 config", artifact.get("config"))
    if (
        artifact.get("artifact_type") != "g0_resident_first_q480"
        or artifact.get("status") != "complete"
        or environment.get("git_commit") != J1_G0_SOURCE["git_commit"]
        or config.get("selection_order")
        != ["probability", "unique_efficiency", "joint_hindsight"]
        or config.get("formal_repetitions") != 5
        or config.get("execution_policy_order")
        != ["expert_id_ascending", "resident_first"]
        or config.get("draft_node_budget") != 7
    ):
        raise ValueError("G0 environment or fixed-7 calibration contract is inconsistent")

    pairs = _sequence("G0 formal_pairs", artifact.get("formal_pairs"))
    policies = ("probability", "unique_efficiency", "joint_hindsight")
    expected_order = tuple(
        (policy, repetition)
        for policy in policies
        for repetition in range(5)
    )
    observed_order = tuple(
        (
            _mapping(f"G0 pair[{index}]", pair).get("selection_policy"),
            _mapping(f"G0 pair[{index}]", pair).get("repetition_index"),
        )
        for index, pair in enumerate(pairs)
    )
    if observed_order != expected_order:
        raise ValueError("G0 formal pair order violates the fixed contract")

    points = []
    for policy in policies:
        rows = []
        for index, pair_value in enumerate(pairs):
            pair = _mapping(f"G0 pair[{index}]", pair_value)
            if pair.get("selection_policy") != policy:
                continue
            runs = _sequence(f"G0 pair[{index}].runs", pair.get("runs"))
            if len(runs) != 2:
                raise ValueError("G0 pair must contain two fixed-order runs")
            resident = _mapping(f"G0 pair[{index}].resident_first", runs[1])
            counters = _mapping(
                "G0 resident counters", resident.get("cache_counter_delta")
            )
            timings = _mapping("G0 resident timings", resident.get("timings"))
            schedule = _mapping("G0 resident schedule", resident.get("schedule"))
            if (
                resident.get("selection_policy") != policy
                or resident.get("execution_policy") != "resident_first"
                or resident.get("phase") != "formal"
                or schedule.get("resident_first_exact") is not True
                or schedule.get("initial_resident_reload_h2d") != 0
            ):
                raise ValueError("G0 resident-first run label or scheduling contract inconsistent")
            footprint = _integer(
                "G0 unique_expert_acquisitions",
                counters.get("unique_expert_acquisitions"),
                minimum=1,
            )
            nonresident = _integer(
                "G0 h2d_submissions", counters.get("h2d_submissions")
            )
            hits = _integer("G0 resident_hits", counters.get("resident_hits"))
            inflight = _integer("G0 inflight_joins", counters.get("inflight_joins"))
            cycle = _integer(
                "G0 cycle_wall_ns", timings.get("cycle_wall_ns"), minimum=1
            )
            if footprint != nonresident + hits + inflight:
                raise ValueError("G0 expert acquisition does not reconcile with H2D/hit/in-flight")
            rows.append((footprint, nonresident, cycle))
        if len(rows) != 5 or len({row[:2] for row in rows}) != 1:
            raise ValueError("G0 each selection must have five formal same-cost repeats")
        footprint, nonresident = rows[0][:2]
        points.append(CalibrationPoint(
            policy,
            footprint,
            nonresident,
            int(median(row[2] for row in rows)),
        ))
    return tuple(points)


def fit_cost_models(
    points: Sequence[CalibrationPoint],
) -> tuple[CostModel, CostModel]:

    if len(points) != 3:
        raise ValueError("J1 pilot calibration must contain exactly three shapes")
    p0, p1, p2 = points
    df1 = p1.expert_acquisitions - p0.expert_acquisitions
    dn1 = p1.h2d_submissions - p0.h2d_submissions
    df2 = p2.expert_acquisitions - p0.expert_acquisitions
    dn2 = p2.h2d_submissions - p0.h2d_submissions
    dt1 = p1.median_cycle_wall_ns - p0.median_cycle_wall_ns
    dt2 = p2.median_cycle_wall_ns - p0.median_cycle_wall_ns
    determinant = df1 * dn2 - df2 * dn1
    if determinant == 0:
        raise ValueError("G0 three calibration points cannot identify the $F/N$ coefficients")
    expert_ns = (dt1 * dn2 - dt2 * dn1) / determinant
    h2d_ns = (df1 * dt2 - df2 * dt1) / determinant
    fixed_ns = (
        p0.median_cycle_wall_ns
        - expert_ns * p0.expert_acquisitions
        - h2d_ns * p0.h2d_submissions
    )
    full = CostModel(fixed_ns, expert_ns, h2d_ns)

    mean_f = sum(point.expert_acquisitions for point in points) / len(points)
    mean_t = sum(point.median_cycle_wall_ns for point in points) / len(points)
    variance = sum((point.expert_acquisitions - mean_f) ** 2 for point in points)
    if variance == 0:
        raise ValueError("G0 calibration point footprint unchanged")
    unique_expert_ns = sum(
        (point.expert_acquisitions - mean_f)
        * (point.median_cycle_wall_ns - mean_t)
        for point in points
    ) / variance
    unique = CostModel(mean_t - unique_expert_ns * mean_f, unique_expert_ns, 0.0)
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in (
            full.fixed_ns,
            full.expert_ns,
            full.h2d_ns,
            unique.fixed_ns,
            unique.expert_ns,
        )
    ):
        raise ValueError("J1 pilot fit produced a non-positive or non-finite coefficient")
    return full, unique


__all__ = [
    "CalibrationPoint",
    "CostModel",
    "J1_G0_SOURCE",
    "fit_cost_models",
    "read_g0_calibration_points",
]
