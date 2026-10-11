from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
import math
from pathlib import Path
import random
import re

from .experiment import canonical_json_sha256
from .f0_proxy_screen import F0_M1_SCREEN_SOURCE
from .j1_calibration import CostModel
from .j1_screen import select_joint_candidate
from .m1_capture_sources import file_descriptor, read_strict_json_file
from .m1_capture_validation import validate_m1_capture_data, M1CaptureProjection
from .m1_capture_validation import M1_CAPTURE_CONFIG
from .m1_screen import _atlas_experts, _ratio_greedy_ids, CandidateScore, M1_QUESTION_IDS
from .m1_screen_artifact import _validate_screen_from_validated_capture
from .r6a_experiment import PoolRecord
from .r6a_route_observation import RouteObservation
from .r6a_selection_contract import enumerate_ancestor_closed_subsets

LEGACY_PROTOCOL = "f0-direct-v2"
PROTOCOL = "f0-direct-v3"
CANDIDATE_COUNT = 3991
POLICIES = ("p7", "f0", "eco7", "hbm7")
COEFFICIENTS = {"fixed_ns": 197496347, "preloaded_ns_per_expert": 186618,
                "h2d_ns_per_expert": 200000}
CAPTURE_SOURCE = {"basename": "m1-full-atlas-v2-e3bf7a2.json", "size_bytes": 111124016,
                  "sha256": "4b45ae37540724bce3840a4d669218c86ff033e85b3de1794f4452a99ac4386b"}
PACKED_SOURCE = {"basename": "manifest.json", "size_bytes": 17444,
                 "sha256": "60708332beaf9d5d1ebebdb91bfa3c28bf3d309c666ad89f5ac07ba4617c88fa"}
CONFIG = {**M1_CAPTURE_CONFIG, "commit_audit": "independent_per_job_and_mode",
          "packed_expert_backing": "registered_mmap", "reduction_mode": "canonical",
          "cache_modes": ["cold", "natural"], "repetitions": 3,
          "warmups_per_job_mode": 1, "audits_per_job_mode": 1,
          "selector_samples": 6}
_EXPERT_BITS = tuple(1 << expert for expert in range(128))


def _atlas_bitmaps(pool: PoolRecord, atlas: RouteObservation) -> tuple[int, ...]:
    if ((0, *(n.raw_id for n in pool.nodes)) != tuple(range(19))
            or atlas.position_to_raw != tuple(range(19))
            or len(atlas.expert_ids_by_raw_layer) != 19 * 48):
        raise ValueError("F0 full atlas must fully cover raw ID 0..18 x 48 layers")
    observed = [0] * 19
    bitmaps = [0] * 19
    for raw, layer, experts in atlas.expert_ids_by_raw_layer:
        if (type(raw) is not int or not 0 <= raw < 19
                or type(layer) is not int or not 0 <= layer < 48 or len(experts) != 8):
            raise ValueError("F0 atlas raw/layer/Top-8 invalid")
        layer_bit = 1 << layer
        if observed[raw] & layer_bit:
            raise ValueError("F0 atlas raw x layer duplicated")
        observed[raw] |= layer_bit
        mask = 0
        for expert in experts:
            if type(expert) is not int or not 0 <= expert < 128:
                raise ValueError("F0 atlas expert ID invalid")
            mask |= _EXPERT_BITS[expert]
        if mask.bit_count() != 8:
            raise ValueError("F0 atlas expert ID duplicated")
        bitmaps[raw] |= mask << (128 * layer)
    return tuple(bitmaps)


def _bitset_scores(pool: PoolRecord, atlas: RouteObservation,
                   subsets: tuple[tuple[int, ...], ...]) -> tuple[CandidateScore, ...]:
    bitmaps = _atlas_bitmaps(pool, atlas)
    probabilities = {n.raw_id: math.exp(n.cumulative_log_probability) for n in pool.nodes}
    rows = []
    for selected in subsets:
        footprint = bitmaps[0]
        for raw in selected:
            footprint |= bitmaps[raw]
        count = footprint.bit_count()
        utility = sum((probabilities[raw] for raw in selected), start=0.0)
        rows.append(CandidateScore(selected, utility, count, count))
    return tuple(rows)

@dataclass(frozen=True)
class DirectSources:
    capture: dict
    projection: M1CaptureProjection
    screen: dict


def load_sources(*, capture_path: Path, screen_path: Path, workload_path: Path,
                 token_lock_path: Path) -> DirectSources:
    documents = []
    for path, expected in ((capture_path, CAPTURE_SOURCE), (screen_path, F0_M1_SCREEN_SOURCE)):
        document, raw = read_strict_json_file(path, "D0 source")
        if file_descriptor(path, raw) != expected:
            raise ValueError(f"D0 fixed source is inconsistent: {path}")
        documents.append(document)
    capture, screen = documents
    commit = capture["environment"]["git_commit"]
    source = (CAPTURE_SOURCE["basename"], CAPTURE_SOURCE["size_bytes"], CAPTURE_SOURCE["sha256"])
    projection = validate_m1_capture_data(capture, workload_path=workload_path,
        token_lock_path=token_lock_path, expected_git_commit=commit, source=source)
    _validate_screen_from_validated_capture(screen, projection,
        expected_git_commit=screen["environment"]["git_commit"])
    return DirectSources(capture, projection, screen)


def select_ids(pool: PoolRecord, atlas: RouteObservation, policy: str) -> tuple[int, ...]:
    if policy == "p7":
        return tuple(sorted(node.raw_id for node in sorted(pool.nodes,
            key=lambda n: (-n.cumulative_log_probability, n.depth, n.generation_ordinal))[:7]))
    if policy == "eco7":
        return _ratio_greedy_ids(pool, probabilities={n.raw_id: math.exp(n.cumulative_log_probability)
            for n in pool.nodes}, experts_by_raw=_atlas_experts(pool, atlas))
    if policy not in ("f0", "hbm7"):
        raise ValueError("Unknown D0 policy")
    subsets = enumerate_ancestor_closed_subsets(pool)
    rows = _bitset_scores(pool, atlas, subsets)
    slope = COEFFICIENTS["preloaded_ns_per_expert"]
    if policy == "f0":
        slope += COEFFICIENTS["h2d_ns_per_expert"]
    return select_joint_candidate(rows, CostModel(COEFFICIENTS["fixed_ns"], slope, 0)).selected_raw_ids


def expert_union(atlas: RouteObservation, selected: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    selected_set = {0, *selected}
    return tuple(tuple(sorted({e for raw, layer, ids in atlas.expert_ids_by_raw_layer
        if raw in selected_set and layer == lid for e in ids})) for lid in range(48))


def pilot_state_ids(states: list[dict]) -> list[str]:
    grouped = {qid: sorted((s for s in states if s["question_id"] == qid),
                          key=lambda s: s["cycle_index"]) for qid in M1_QUESTION_IDS}
    if len(states) != 32 or any(len(rows) != 4 or len({r["cycle_index"] for r in rows}) != 4
                                for rows in grouped.values()):
        raise ValueError("D0 requires 8 prompts x 4 distinct sampled cycles")
    return [grouped[qid][1]["state_id"] for qid in M1_QUESTION_IDS]


def unique_jobs(states: list[dict], pilot_ids: list[str]) -> list[dict]:
    jobs = []
    for state in states:
        if state["state_id"] not in pilot_ids:
            continue
        grouped = {}
        for policy in ("p7", "f0"):
            ids = tuple(state["decisions"][policy]["selected_raw_ids"])
            grouped.setdefault(ids, []).append(policy)
        for ids, policies in grouped.items():
            jobs.append({"job_id": f"{state['state_id']}-{'-'.join(map(str, ids))}",
                         "state_id": state["state_id"], "selected_raw_ids": list(ids),
                         "policies": policies})
    return jobs


def build_schedules(jobs: list[dict]) -> dict[str, list[dict]]:
    rng = random.Random(480)
    queues = ([], [])
    for j, job in enumerate(jobs):
        for rep in range(3):
            queues[(j + rep) % 2].append((job["job_id"], rep))
    for queue in queues:
        rng.shuffle(queue)
    paired = []
    for i in range(3 * len(jobs)):
        order = i % 2
        job_id, rep = queues[order].pop()
        modes = ("cold", "natural") if order == 0 else ("natural", "cold")
        for mode in modes:
            paired.append({"job_id": job_id, "mode": mode,
                           "repetition": rep, "pair_index": i})
    return {"paired": paired}


def build_manifest(sources: DirectSources, *, git_commit: str, protocol: str = PROTOCOL) -> dict:
    if protocol not in (LEGACY_PROTOCOL, PROTOCOL):
        raise ValueError("Unknown F0 direct protocol")
    if re.fullmatch(r"[0-9a-f]{40}", git_commit or "") is None:
        raise ValueError("D0 requires a full Git commit")
    records = {s["state_id"]: s for s in sources.capture["states"]}
    screens = {s["state_id"]: s for s in sources.screen["states"]}
    states = []
    count = 0
    for state in sources.projection.states:
        raw = records[state.state_id]
        rows = {tuple(r["selected_raw_ids"]): r for r in screens[state.state_id]["candidate_sets"]}
        count += len(rows)
        decisions = {}
        for policy in POLICIES:
            ids = select_ids(state.pool, state.atlas, policy)
            row = rows[ids]
            union = expert_union(state.atlas, ids)
            if sum(map(len, union)) != row["F_atlas"]:
                raise ValueError("D0 footprint does not match M1 reconstruction")
            decisions[policy] = {"selected_raw_ids": list(ids), "expected_committed_tokens": 1 + row["draft_utility"],
                "F_atlas": row["F_atlas"], "experts_by_layer": [list(x) for x in union],
                "max_layer_union": max(map(len, union))}
        states.append({"state_id": state.state_id, "question_id": state.question_id,
            "cycle_index": state.cycle_index, "pool_fingerprint": state.pool.fingerprint,
            "route_digest": raw["full_atlas_routes"]["digest"],
            "reference_state_sha256": canonical_json_sha256(raw),
            "pre_logical_state_sha256": raw["pre_logical_state_sha256"],
            "decisions": decisions,
            "same_selection": {f"{a}/{b}": decisions[a]["selected_raw_ids"] == decisions[b]["selected_raw_ids"]
                               for a, b in combinations(POLICIES, 2)}})
    if count != CANDIDATE_COUNT:
        raise ValueError("D0 did not reconstruct 3,991 fixed-7 candidates")
    pilot_ids = pilot_state_ids(states)
    jobs = unique_jobs(states, pilot_ids)
    capacity = [f"{s['state_id']}/{p}" for s in states if s["state_id"] in pilot_ids
                for p in ("p7", "f0") if s["decisions"][p]["max_layer_union"] > 48]
    differences = sum(not s["same_selection"]["p7/f0"] for s in states)
    status = "ready" if differences else "stop_no_decision"
    return {"artifact_type": "f0_direct_manifest", "protocol": protocol, "schema_version": 1,
        "git_commit": git_commit, "config": CONFIG,
        "sources": {"capture": CAPTURE_SOURCE, "screen": F0_M1_SCREEN_SOURCE, "packed": PACKED_SOURCE},
        "coefficients": dict(COEFFICIENTS), "seed": 480, "state_count": len(states),
        "candidate_count": count, "pilot_rule": "second_sampled_cycle_sorted_by_cycle_index_per_prompt",
        "pilot_state_ids": pilot_ids, "states": states, "jobs": jobs, "schedule": build_schedules(jobs),
        "gate": {"status": status, "decision_difference_count": differences, "proxy_capacity_diagnostics": capacity},
        "measurement_scope": "offline_full_atlas_route_signal_local_cycle_including_selector_cost"}


def validate_manifest(manifest: dict, sources: DirectSources, *, expected_git_commit: str) -> None:
    expected = build_manifest(sources, git_commit=expected_git_commit, protocol=manifest.get("protocol"))
    if canonical_json_sha256(manifest) != canonical_json_sha256(expected):
        raise ValueError("D0 manifest cannot be fully reconstructed from the fixed source")


def write_json_exclusive(path: Path, value: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return file_descriptor(path, path.read_bytes())
