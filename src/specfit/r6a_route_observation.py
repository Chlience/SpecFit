
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .r6a_experiment import (
    R6A_EXPERTS_PER_TOKEN,
    R6A_MOE_LAYER_COUNT,
    _digest,
    _int,
    _ints,
    _map,
    _seq,
    _sha,
)


def compute_r6a_route_digest(
    *,
    pool_fingerprint: str,
    verification_position_to_raw_id: Sequence[int],
    rows: Sequence[Mapping[str, object]],
) -> str:

    canonical = [
        {
            "verification_position": _int(
                "route.position", row.get("verification_position")
            ),
            "raw_id": _int("route.raw_id", row.get("raw_id")),
            "layer_id": _int("route.layer_id", row.get("layer_id")),
            "expert_ids": list(
                _ints("route.expert_ids", row.get("expert_ids"))
            ),
        }
        for row in rows
    ]
    canonical.sort(
        key=lambda row: (row["verification_position"], row["layer_id"])
    )
    return _digest(
        {
            "pool_fingerprint": _sha(
                "pool_fingerprint", pool_fingerprint
            ),
            "verification_position_to_raw_id": list(
                verification_position_to_raw_id
            ),
            "rows": canonical,
        }
    )


@dataclass(frozen=True, slots=True)
class RouteObservation:

    position_to_raw: tuple[int, ...]
    digest: str
    expert_ids_by_raw_layer: tuple[tuple[int, int, tuple[int, ...]], ...]

    def unique_expert_keys(
        self,
        raw_ids: Sequence[int] | None = None,
    ) -> frozenset[tuple[int, int]]:

        allowed = None if raw_ids is None else {0, *raw_ids}
        return frozenset(
            (layer_id, expert_id)
            for raw_id, layer_id, experts in self.expert_ids_by_raw_layer
            if allowed is None or raw_id in allowed
            for expert_id in experts
        )

    @classmethod
    def read(
        cls,
        value: object,
        *,
        name: str,
        position_to_raw: tuple[int, ...],
        pool_fingerprint: str,
        num_experts: int,
    ) -> RouteObservation:
        route = _map(name, value)
        observed_pool = _sha(
            f"{name}.pool_fingerprint", route.get("pool_fingerprint")
        )
        if observed_pool != pool_fingerprint:
            raise ValueError(f"{name} pool fingerprint inconsistent")
        rows = tuple(
            _map(f"{name}.rows[{index}]", item)
            for index, item in enumerate(
                _seq(f"{name}.rows", route.get("rows"))
            )
        )
        expected_count = len(position_to_raw) * R6A_MOE_LAYER_COUNT
        if (
            route.get("coverage_rows") != expected_count
            or len(rows) != expected_count
        ):
            raise ValueError(f"{name} route coverage is incomplete")
        seen: set[tuple[int, int]] = set()
        normalized = []
        for row in rows:
            position = _int(
                "route.position", row.get("verification_position")
            )
            layer = _int("route.layer", row.get("layer_id"))
            raw_id = _int("route.raw_id", row.get("raw_id"))
            experts = _ints("route.expert_ids", row.get("expert_ids"))
            invalid = (
                position >= len(position_to_raw)
                or layer >= R6A_MOE_LAYER_COUNT
                or raw_id != position_to_raw[position]
                or len(experts) != R6A_EXPERTS_PER_TOKEN
                or len(set(experts)) != R6A_EXPERTS_PER_TOKEN
                or any(expert >= num_experts for expert in experts)
            )
            if invalid:
                raise ValueError(
                    f"{name} route position/raw/layer/Top8 mismatch"
                )
            seen.add((position, layer))
            normalized.append((raw_id, layer, experts))
        if len(seen) != expected_count:
            raise ValueError(f"{name} route position x layer has duplicates or is missing")
        digest = _sha(f"{name}.digest", route.get("digest"))
        expected_digest = compute_r6a_route_digest(
            pool_fingerprint=pool_fingerprint,
            verification_position_to_raw_id=position_to_raw,
            rows=rows,
        )
        if digest != expected_digest:
            raise ValueError(f"{name} digest inconsistent with routes")
        return cls(
            position_to_raw=position_to_raw,
            digest=digest,
            expert_ids_by_raw_layer=tuple(normalized),
        )


def validate_acquisition_counters(
    counters: Mapping[str, int],
    routes: RouteObservation,
    *,
    name: str,
    expert_bytes: int,
) -> None:

    acquisitions = len(routes.unique_expert_keys())
    classified = (
        counters["h2d_submissions"]
        + counters["resident_hits"]
        + counters["inflight_joins"]
    )
    if (
        classified != acquisitions
        or counters["evictions"] > counters["h2d_submissions"]
        or counters["unique_expert_acquisitions"] != acquisitions
        or counters["expert_bytes"] != expert_bytes
        or counters["logical_h2d_bytes"]
        != counters["h2d_submissions"] * expert_bytes
    ):
        raise ValueError(f"{name} cache counters do not reconcile with own-route acquisitions")


__all__ = [
    "RouteObservation",
    "compute_r6a_route_digest",
    "validate_acquisition_counters",
]
