#!/usr/bin/env python3
from __future__ import annotations
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import re
import subprocess
import torch
from transformers import AutoTokenizer
from specfit.experiment import canonical_json_sha256
from specfit.gsm8k import load_gsm8k_specbench, load_gsm8k_specbench_lock
from specfit.gsm8k_depth5_experiment import max_new_tokens_for_input
from specfit.m1_capture import (
    ScheduledM1Policy,
    project_probe_state,
    project_reference_rollout,
)
from specfit.m1_capture_validation import (
    M1_CAPTURE_CONFIG, M1_CAPTURE_SCHEMA_VERSION, M1_CAPTURE_TYPE,
    M1_EOS_TOKEN_IDS,
    M1_LIMITATIONS,
    M1_QUESTION_IDS,
    M1_SAMPLING_CONTRACT,
    output_token_diagnostic,
    validate_m1_capture_data,
)
from specfit.r6a_identity import R6A_DATASET_IDENTITY, R6A_MODEL_IDENTITIES
from specfit.r6a_tb2_contract import R6A_TB2_IMAGE_ID
from specfit.runtime import (
    Eagle3ChainDrafter,
    Eagle3TreeDrafter,
    ExpertExecutionPolicy,
    ExplicitRawIdTreeSelector,
    GreedyChainTraceRecorder,
    RecordingCacheTransitionObserver,
    ReductionMode,
    ShallowTreeSpeculator,
    apply_numeric_mode,
    load_offloaded_qwen3,
    target_greedy_generate,
    warmup_eagle3_tree_drafter,
)
from specfit.runtime.eagle3_chain import _normalized_eos_token_ids
from specfit.trace import WorkloadManifestTrace
_TARGET = "/models/qwen3/Qwen3-30B-A3B-Instruct-2507"
_DRAFTER = "/models/qwen3/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3"
_PACKED = ("/packed-cache/Qwen3-30B-A3B-Instruct-2507/"
           "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe/packed-v1")
_ROOT = Path(__file__).resolve().parents[1]
_TOKEN_LOCK = _ROOT / "workloads/m1_gsm8k_prompt_tokens.lock.json"
_MODELS_LOCK = _ROOT / "docker/models.lock.json"
def _args():
    lock = load_gsm8k_specbench_lock()
    parser = ArgumentParser(description="Collect Qwen3 M1 full-atlas artifact")
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--artifact-output", type=Path, required=True)
    default_dataset = Path("/data/workloads/gsm8k") / lock.revision / lock.filename
    parser.add_argument("--dataset", type=Path, default=default_dataset)
    return parser.parse_args()
def _descriptor(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {"basename": path.name, "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}
def _source_preflight(args) -> dict[str, object]:
    def git(*command: str) -> str:
        return subprocess.run(
            ("git", *command), cwd=_ROOT, check=True, text=True,
            capture_output=True).stdout.strip()
    if args.git_commit != git("rev-parse", "HEAD") or git("status", "--porcelain"):
        raise RuntimeError("M1 requires --git-commit to equal the clean repository HEAD")
    lock = json.loads(_MODELS_LOCK.read_text(encoding="utf-8"))
    for name, fixed_path in (("target", _TARGET), ("drafter", _DRAFTER)):
        item = lock[name]
        directory = Path("/models") / item["directory"]
        files = tuple(path for path in directory.iterdir()
                      if path.is_file() and not path.is_symlink())
        identity = {key: item[key] for key in ("repo_id", "revision")}
        observed = (directory, len(files), sum(path.stat().st_size for path in files))
        if identity != R6A_MODEL_IDENTITIES[name] or observed != (
                Path(fixed_path), item["files"], item["bytes"]):
            raise RuntimeError(f"M1 {name} local directory does not match models.lock")
    return {**R6A_MODEL_IDENTITIES, "lock_descriptor": _descriptor(_MODELS_LOCK)}
def _require_m1_eos_ids(model) -> None:
    actual = frozenset(_normalized_eos_token_ids(model))
    if actual != M1_EOS_TOKEN_IDS:
        raise RuntimeError(f"M1 target EOS IDs are inconsistent: {sorted(actual)}")
def _manifest(args, *, run_id: str, prepared: dict[str, object]):
    target, drafter = R6A_MODEL_IDENTITIES["target"], R6A_MODEL_IDENTITIES["drafter"]
    return WorkloadManifestTrace(
        run_id=run_id, request_id=f"gsm8k-q{prepared['question_id']}",
        git_commit=args.git_commit, image_id=args.image_id,
        target_model=target["repo_id"], target_revision=target["revision"],
        drafter_model=drafter["repo_id"], drafter_revision=drafter["revision"],
        tokenizer_revision=target["revision"], device="cuda:0", dtype="bfloat16",
        reduction_mode=ReductionMode.CANONICAL.value,
        drafter_mode="eagle3-top2-depth5-m1-full-atlas", slots_per_layer=48,
        num_moe_layers=48, num_experts_per_layer=128, experts_per_token=8,
        draft_length=5, max_new_tokens=prepared["max_new_tokens"], seed=0,
        raw_prompt=prepared["raw_prompt"], rendered_prompt=prepared["rendered"],
        input_ids=prepared["input_ids"], attention_mask=prepared["attention_mask"],
    )
def _begin_run(loaded, observer, *, run_id: str) -> None:
    loaded.set_expert_execution_policy(ExpertExecutionPolicy.EXPERT_ID_ASCENDING)
    loaded.cold_reset_expert_caches()
    snapshots = loaded.cache_snapshots()
    stats = loaded.cache_stats()
    if any(layer.pending_transfers or any(slot.expert_id is not None
           or slot.active_leases or slot.pending_use_guards for slot in layer.slots)
           for layer in snapshots) or any(getattr(stats, name) for name in (
               "h2d_submissions", "resident_hits", "inflight_joins", "evictions",
               "pending_use_guards")):
        raise RuntimeError("M1 cold reset did not leave expert cache empty")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    observer.reset(run_id=run_id)
def _prefill_gate(loaded, device: torch.device) -> None:
    torch.cuda.synchronize(device)
    if any(layer.pending_transfers or any(
        slot.active_leases or slot.pending_use_guards
        or (slot.expert_id is not None and not slot.ready)
        for slot in layer.slots) for layer in loaded.cache_snapshots()):
        raise RuntimeError("M1 cache is not a quiescent ready state after prefill")
    loaded.set_expert_execution_policy(ExpertExecutionPolicy.RESIDENT_FIRST)
def _tensors(prepared: dict[str, object], device: torch.device):
    return (
        torch.tensor([prepared["input_ids"]], dtype=torch.long, device=device),
        torch.tensor([prepared["attention_mask"]], dtype=torch.long, device=device),
    )
def _run_target(loaded, observer, prepared, device, run_id: str):
    _begin_run(loaded, observer, run_id=run_id)
    input_ids, mask = _tensors(prepared, device)
    with torch.inference_mode():
        result = target_greedy_generate(
            loaded.model, input_ids, attention_mask=mask,
            max_new_tokens=prepared["max_new_tokens"],
            on_prefill_complete=lambda: _prefill_gate(loaded, device),
        )
    torch.cuda.synchronize(device)
    return result
def _run_spec(loaded, drafter, observer, prepared, device, args,
              *, run_id: str, probe_cycle: int | None):
    _begin_run(loaded, observer, run_id=run_id)
    input_ids, mask = _tensors(prepared, device)
    policy = ScheduledM1Policy(probe_cycle=probe_cycle)
    recorder = GreedyChainTraceRecorder(
        _manifest(args, run_id=run_id, prepared=prepared), target=loaded.model,
        cache_snapshot_provider=loaded.cache_snapshots, transition_log=observer,
    )
    speculator = ShallowTreeSpeculator(loaded.model, Eagle3TreeDrafter(drafter))
    with recorder, torch.inference_mode():
        result = speculator.generate(
            input_ids, attention_mask=mask,
            max_new_tokens=prepared["max_new_tokens"], top_k=2, max_depth=5,
            draft_node_budget=18, trace=recorder,
            on_prefill_complete=lambda: _prefill_gate(loaded, device),
            stop_on_eos=True, audit_commits=True,
            selector=ExplicitRawIdTreeSelector(policy), collect_cycle_metrics=False,
            max_cycles=None if probe_cycle is None else probe_cycle + 1,
        )
    torch.cuda.synchronize(device)
    trace = recorder.finalize()
    expected = result.cycles if probe_cycle is None else probe_cycle + 1
    if (result.cycles != expected or len(policy.records) != expected
            or len(result.commit_audits) != expected
            or not all(item.passed for item in result.commit_audits)):
        raise RuntimeError("M1 selector, trace, and generation cycle counts do not close")
    output = tuple(int(token) for token in result.tokens[0].cpu())
    return output, trace, tuple(policy.records)
def _prepare(tokenizer, dataset: Path, token_lock: Path):
    examples = {item.question_id: item for item in load_gsm8k_specbench(dataset)}
    lock = json.loads(token_lock.read_text(encoding="utf-8"))
    locked = {item["question_id"]: tuple(item["input_ids"])
              for item in lock["prompts"]}
    prepared = []
    for question_id in M1_QUESTION_IDS:
        example = examples[question_id]
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": example.prompt}], tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(rendered, return_tensors="pt")
        input_ids = tuple(int(token) for token in encoded.input_ids[0])
        if input_ids != locked.get(question_id):
            raise RuntimeError(f"q{question_id} token identity does not match the authoritative lock")
        prepared.append({
            "question_id": question_id, "raw_prompt": example.prompt,
            "rendered": rendered, "input_ids": input_ids,
            "attention_mask": tuple(int(x) for x in encoded.attention_mask[0]),
            "max_new_tokens": max_new_tokens_for_input(len(input_ids)),
        })
    return prepared

def main() -> None:
    args = _args()
    output, dataset, token_lock = args.artifact_output, args.dataset, _TOKEN_LOCK
    if output.exists():
        raise FileExistsError(f"M1 artifact already exists: {output}")
    if args.image_id != R6A_TB2_IMAGE_ID or re.fullmatch(
        r"[0-9a-f]{40}", args.git_commit or "") is None:
        raise ValueError("M1 requires a fixed NGC image ID and a full lowercase Git commit")
    models = _source_preflight(args)
    device = torch.device("cuda:0")
    if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(device):
        raise RuntimeError("M1 requires the process to expose only one NVIDIA A100")
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(_TARGET, local_files_only=True)
    prepared = _prepare(tokenizer, dataset, token_lock)
    observer = RecordingCacheTransitionObserver(run_id=f"{args.run_prefix}-load")
    loaded = numeric = None
    try:
        loaded = load_offloaded_qwen3(
            _TARGET, device=device, slots_per_layer=48,
            packed_checkpoint_root=_PACKED, packed_expert_backing="registered_mmap",
            verify_packed_hashes=True,
            reduction_mode=ReductionMode.CANONICAL,
            expert_execution_policy=ExpertExecutionPolicy.EXPERT_ID_ASCENDING,
            cache_observer=observer,
        )
        _require_m1_eos_ids(loaded.model)
        numeric = apply_numeric_mode("math_bf16_full_reduction")
        drafter = Eagle3ChainDrafter.from_pretrained(
            _DRAFTER, target_root=_TARGET, device=device, use_torch_compile=False,
            expected_target_repo_id=R6A_MODEL_IDENTITIES["target"]["repo_id"],
        )
        warmup_eagle3_tree_drafter(
            drafter, Eagle3TreeDrafter(drafter), max_sequence_length=506,
            max_depth=5, top_k=2, draft_node_budget=18,
            sequence_anchors=(64, 192, 320, 506),
        )
        targets, references, diagnostics, states = [], [], [], []
        for item in prepared:
            qid = item["question_id"]
            target_result = _run_target(
                loaded, observer, item, device, f"{args.run_prefix}-q{qid}-target")
            target = tuple(int(token) for token in target_result.tokens[0].cpu())
            ref_output, trace, selections = _run_spec(
                loaded, drafter, observer, item, device, args,
                run_id=f"{args.run_prefix}-q{qid}-reference", probe_cycle=None)
            reference = project_reference_rollout(
                question_id=qid, input_token_ids=item["input_ids"],
                trace=trace, selections=selections)
            targets.append({"question_id": qid,
                "max_new_tokens": item["max_new_tokens"],
                "output_token_ids": list(target),
                "output_token_count": len(target),
                "output_token_ids_sha256": canonical_json_sha256(target),
                "stopped_on_eos": target_result.stopped_on_eos,
                "stop_token_id": target_result.stop_token_id})
            references.append(reference)
            diagnostics.append({"question_id": qid,
                **output_token_diagnostic(target, ref_output)})
            for cycle in reference["sampled_cycle_indices"]:
                _, probe_trace, probe_selections = _run_spec(
                    loaded, drafter, observer, item, device, args,
                    run_id=f"{args.run_prefix}-q{qid}-probe-{cycle}", probe_cycle=cycle)
                states.append(project_probe_state(
                    question_id=qid, probe_cycle=cycle,
                    input_token_ids=item["input_ids"], trace=probe_trace,
                    selections=probe_selections,
                    reference_cycle=reference["cycles"][cycle]))
        dataset_identity = {**R6A_DATASET_IDENTITY,
            "workload_descriptor": _descriptor(dataset),
            "token_lock_descriptor": _descriptor(token_lock),
            "prompts": [{"question_id": item["question_id"],
                "input_token_ids": list(item["input_ids"]),
                "input_token_ids_sha256": canonical_json_sha256(item["input_ids"])}
                for item in prepared]}
        artifact = {"schema_version": M1_CAPTURE_SCHEMA_VERSION,
            "artifact_type": M1_CAPTURE_TYPE,
            "status": "complete", "environment": {
                "git_commit": args.git_commit, "image_id": args.image_id,
                "server": "a100-host", "device": "cuda:0",
                "numeric_settings": dict(numeric.actual_settings)},
            "config": dict(M1_CAPTURE_CONFIG), "dataset": dataset_identity,
            "models": models,
            "sampling_contract": dict(M1_SAMPLING_CONTRACT),
            "target_only": targets, "reference_rollouts": references,
            "output_diagnostics": diagnostics,
            "states": states, "limitations": dict(M1_LIMITATIONS)}
        validate_m1_capture_data(
            artifact, workload_path=dataset, token_lock_path=token_lock,
            expected_git_commit=args.git_commit)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(artifact, handle, ensure_ascii=False, indent=2,
                      sort_keys=True, allow_nan=False)
            handle.write("\n")
        print(json.dumps({"artifact": str(output), "states": len(states)}, indent=2))
    finally:
        if numeric is not None:
            numeric.restore()
        if loaded is not None:
            loaded.close()

if __name__ == "__main__":
    main()
