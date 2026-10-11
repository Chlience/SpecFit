#!/usr/bin/env python3
"""Audit and replay whole GSM8K streams with expert caches retained across questions."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import gc
import gzip
import json
from pathlib import Path
import random
import resource
import time
from types import SimpleNamespace

from specfit.experiment import canonical_json_sha256
from specfit.gsm8k import load_gsm8k_specbench
from specfit.serial_replay import (COUNTERS, METHODS, PROTOCOL, TOTAL_TOKENS,
    choose, file_identity, numeric_answer, pool_view, validate_capture, validate_stream)
from specfit.selection_methods import SelectionCostModel
from specfit.r6a_tb2_contract import R6A_TB2_IMAGE_ID


def write_json(path, value):
    data = (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    with path.open("xb") as handle:
        if path.suffix == ".gz":
            with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0, filename="") as compressed:
                compressed.write(data)
        else:
            handle.write(data)
    return file_identity(path)


def read_json(path, descriptor=None):
    if descriptor is not None and file_identity(path) != descriptor:
        raise ValueError(f"artifact bytes changed: {path}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def cache_boundary(loaded):
    """Read a quiescent boundary without reaping completed cache events."""
    import torch
    torch.cuda.synchronize()
    snapshots = loaded.cache_snapshots()
    layers = []
    for lid, snapshot in enumerate(snapshots):
        if snapshot.pending_transfers or any(s.active_leases or s.pending_use_guards for s in snapshot.slots):
            raise RuntimeError("request boundary has pending expert work")
        layers.append(dict(layer=lid, version=snapshot.host_state_version,
            lru=snapshot.lru_sequence, submission=snapshot.submission_sequence,
            slots=[dict(slot=s.slot_id, expert=None if s.expert_id is None else s.expert_id.expert_id,
                generation=s.generation, last_used=s.last_used_sequence, ready=s.ready)
                for s in snapshot.slots], stats=asdict(snapshot.stats)))
    return dict(sha256=canonical_json_sha256(layers), layers=layers,
                occupied=sum(s.expert_id is not None for x in snapshots for s in x.slots),
                counters={k: sum(int(getattr(x.stats, k)) for x in snapshots) for k in COUNTERS})


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "validate"))
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--calibration", type=Path,
                        default=Path("docs/results/selection-refinement-1886852.json"))
    parser.add_argument("--git-commit")
    parser.add_argument("--image-id")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def freeze(args):
    from benchmark_qwen3_selection import head
    if head() != args.git_commit or args.image_id != R6A_TB2_IMAGE_ID:
        raise ValueError("freeze requires the clean source commit")
    examples = load_gsm8k_specbench(args.dataset)
    lock = read_json(Path(__file__).resolve().parents[1] / "workloads/gsm8k_specbench.lock.json")["dataset"]
    actual = file_identity(args.dataset)
    if any(actual[k] != lock[k] for k in ("size_bytes", "sha256")):
        raise ValueError("dataset bytes differ from the project lock")
    calibration = read_json(args.calibration)
    model = calibration["stages"]["calibration"]["models"]["by_mode"]["natural"]["legacy"]
    SelectionCostModel.from_dict(model)
    questions = [404, 439] if args.smoke else [e.question_id for e in examples]
    schedule = [(method, repeat) for repeat in range(2) for method in METHODS]
    random.Random(480).shuffle(schedule)
    frozen = dict(protocol=PROTOCOL, created_ns=time.time_ns(), git_commit=args.git_commit,
        image_id=args.image_id, smoke=args.smoke, total_tokens=TOTAL_TOKENS,
        question_ids=questions, methods=list(METHODS), repetitions=2, seed=480,
        stream_schedule=[dict(method=m, repetition=r) for m, r in schedule],
        dataset=file_identity(args.dataset), calibration=file_identity(args.calibration),
        calibration_source_commit=calibration["source_commit"], cost_model=model,
        execution="ready_demand", slots_per_layer=48, max_inflight=2,
        route_information="offline_full_pool_atlas_from_separate_audited_generation",
        cache_policy="reset_once_per_stream_preserve_across_questions_and_cycles",
        max_gpu_reserved_bytes=38 * 1024**3, max_rss_bytes=128 * 1024**3,
        max_wall_seconds=24 * 3600, stop_on_eos=True,
        prompt_policy="unchanged_single_user_question_with_locked_target_chat_template",
        answer_policy="explicit_numeric_final_answer_no_silent_last_number_fallback",
        timing_scope="measured_request_and_decode_exclude_offline_atlas_and_audit_preparation",
        cross_query_equivalence="reported_against_AR_not_assumed_or_used_for_selection")
    args.directory.mkdir(parents=True, exist_ok=False)
    write_json(args.directory / "manifest.json", frozen)
    print(json.dumps(frozen, ensure_ascii=False), flush=True)


def validate(directory):
    frozen = read_json(directory / "manifest.json")
    index = read_json(directory / "result.json")
    if (index["status"] != "complete" or index["manifest_sha256"] != canonical_json_sha256(frozen)
            or index["git_commit"] != frozen["git_commit"]):
        raise ValueError("run is incomplete or has changed identity")
    model = SelectionCostModel.from_dict(frozen["cost_model"])
    references = {}
    for descriptor in index["references"]:
        row = read_json(directory / descriptor["basename"], descriptor)
        key = row["method"], row["question_id"]
        if key in references:
            raise ValueError("duplicate reference")
        if row["method"] == "expert_fast_dynamic":
            validate_capture(row, model)
        if not all(x["passed"] for x in row["commit_audits"]):
            raise ValueError("reference commit audit failed")
        references[key] = row
    if set(references) != {(m, q) for m in METHODS for q in frozen["question_ids"]}:
        raise ValueError("reference coverage mismatch")
    streams = {}
    for descriptor in index["questions"]:
        row = read_json(directory / descriptor["basename"], descriptor)
        reference = references[row["method"], row["question_id"]]
        if (row["output_token_ids"] != reference["output_token_ids"]
                or row["query_proofs"] != reference["query_proofs"]
                or row["reference_sha256"] != canonical_json_sha256(reference)):
            raise ValueError("measured generation differs from its exact-query reference")
        for cycle in row["cycles"]:
            pool = pool_view(cycle["pool"])
            source = (reference["cycles"][cycle["iteration"]]["atlas"]
                      if row["method"] == "expert_fast_dynamic" else None)
            if list(choose(pool, row["method"], model, source)) != cycle["selected_raw_ids"]:
                raise ValueError("measured selector changed its frozen rule")
        streams.setdefault((row["method"], row["repetition"]), []).append(row)
    expected_streams = {(m, r) for m in METHODS for r in range(frozen["repetitions"])}
    if set(streams) != expected_streams:
        raise ValueError("stream coverage mismatch")
    summaries = []
    for (method, repeat), rows in streams.items():
        summary = validate_stream(rows, question_ids=frozen["question_ids"],
                                  method=method, repetition=repeat)
        summaries.append(dict(method=method, repetition=repeat, **summary))
    print(json.dumps(dict(status="pass", streams=len(streams),
        measured_questions=len(index["questions"]), reference_questions=len(references),
        summaries=summaries), ensure_ascii=False), flush=True)
    return summaries


def run(args, *, experiment=None):
    import torch
    from transformers import AutoTokenizer
    import benchmark_qwen3_m1_full_atlas as m1
    from specfit.r6a_runner import serialize_candidate_pool
    from specfit.runtime import (Eagle3ChainDrafter, Eagle3TreeDrafter,
        ExplicitRawIdTreeSelector, ExpertExecutionPolicy, ReductionMode,
        ShallowTreeSpeculator, apply_numeric_mode, load_offloaded_qwen3,
        target_greedy_generate, warmup_eagle3_tree_drafter, candidate_pool_fingerprint)
    from benchmark_qwen3_selection import head
    frozen = read_json(args.directory / "manifest.json")
    if (head() != frozen["git_commit"] or args.git_commit != frozen["git_commit"]
            or args.image_id != frozen["image_id"] or args.image_id != R6A_TB2_IMAGE_ID
            or file_identity(args.dataset) != frozen["dataset"]):
        raise ValueError("run input or source differs from frozen manifest")
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != "NVIDIA A100-PCIE-40GB":
        raise RuntimeError("serial replay requires one A100 PCIe 40GB")
    source = m1._source_preflight(args)
    if experiment and source != frozen["reference_models"]:
        raise ValueError("models differ from the audited historical references")
    examples = {e.question_id: e for e in load_gsm8k_specbench(args.dataset)}
    tokenizer = AutoTokenizer.from_pretrained(m1._TARGET, local_files_only=True)
    prepared = {}
    for qid in frozen["question_ids"]:
        prompt = tokenizer.apply_chat_template([dict(role="user", content=examples[qid].prompt)],
                                              tokenize=False, add_generation_prompt=True)
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        if not 0 < ids.shape[1] < TOTAL_TOKENS:
            raise ValueError(f"question {qid} does not fit the fixed token budget")
        prepared[qid] = dict(prompt=prompt, ids=ids)
    model = SelectionCostModel.from_dict(frozen["cost_model"])
    index = dict(protocol=frozen["protocol"] if experiment else PROTOCOL, status="running", git_commit=frozen["git_commit"],
        manifest_sha256=canonical_json_sha256(frozen), started_ns=time.time_ns(),
        environment=dict(image_id=args.image_id, models=source,
                         gpu_name=torch.cuda.get_device_name(0)), references=[], questions=[])
    write_json(args.directory / "started.json", index)
    loaded = numeric = None
    began = time.perf_counter_ns()
    gc_enabled = gc.isenabled()

    def event(**value):
        print(json.dumps(dict(time_ns=time.time_ns(), **value), ensure_ascii=False), flush=True)

    def budget():
        if (time.perf_counter_ns() - began > frozen["max_wall_seconds"] * 10**9
                or torch.cuda.max_memory_reserved() > frozen["max_gpu_reserved_bytes"]
                or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > frozen["max_rss_bytes"]):
            raise RuntimeError("serial replay exceeded a frozen resource budget")

    def tensor_digest(tensor):
        return sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

    class Execution:
        def __init__(self, variant="baseline"):
            self.decode = False
            self.probing = False
            self.proofs = []
            self.routes = {}
            self.prefetch_stats = []
            self.runtime = experiment.execution(loaded, variant) if experiment else None

        def close(self):
            if self.runtime is not None:
                self.runtime.close()

        def hook(self, lid):
            def capture(module, inputs, output):
                self.routes[lid] = output[2].detach()
            return capture

        @contextmanager
        def observe(self):
            original = loaded.model.forward
            def forward(*pos, **kw):
                self.routes = {}
                if self.decode:
                    context = (self.runtime.forward() if self.runtime is not None else
                               loaded.prefetch_experts({}, lookahead_layers=0, max_inflight=2))
                    with context as scheduler:
                        output = original(*pos, **kw)
                    self.prefetch_stats.append(scheduler.stats())
                else:
                    output = original(*pos, **kw)
                if not self.probing:
                    self.proofs.append((output.logits.detach(), tuple(self.routes[i] for i in range(48))))
                return output
            with ExitStack() as stack:
                stack.callback(self.close)
                loaded.model.forward = forward
                stack.callback(setattr, loaded.model, "forward", original)
                for lid, layer in enumerate(loaded.model.model.layers):
                    handle = layer.mlp.gate.register_forward_hook(self.hook(lid))
                    stack.callback(handle.remove)
                yield

        def materialize(self):
            return [dict(logits_sha256=tensor_digest(logits), logits_shape=list(logits.shape),
                         routes_sha256=canonical_json_sha256([r.cpu().tolist() for r in routes]))
                    for logits, routes in self.proofs]

        def scheduler_summary(self):
            rows = self.prefetch_stats
            return dict(forwards=len(rows), complete=all(r["complete"] for r in rows),
                peak_inflight=max((r["peak_inflight"] for r in rows), default=0),
                layers={k: sum(layer[k] for row in rows for layer in row["layers"])
                        for k in (rows[0]["layers"][0] if rows else {}) if k != "layer_id"})

    class Selector:
        def __init__(self, method, execution, reference, variant):
            self.method, self.execution, self.reference = method, execution, reference
            self.variant = variant
            self.boundary = None
            self.rows = []

        def at_boundary(self, checkpoint):
            self.boundary = checkpoint
            budget()

        def select_raw_ids(self, pool, *, draft_node_budget, committed_offset=0):
            fingerprint = candidate_pool_fingerprint(pool)
            view = SimpleNamespace(nodes=pool.candidates, fingerprint=fingerprint,
                                   seed_token_id=pool.seed_token_id)
            atlas = None
            if self.method == "expert_fast_dynamic":
                if self.reference is None:
                    state = self.boundary.state
                    tree = pool.to_tree(tuple(n.raw_id for n in pool.candidates))
                    verify_ids = tree.verification_token_ids(state.pending_seed)
                    cache = deepcopy(state.cache)
                    self.execution.probing = True
                    try:
                        output = loaded.model(input_ids=verify_ids,
                            attention_mask=tree.verification_attention_mask(base_kv_length=cache.get_seq_length(), device=verify_ids.device),
                            position_ids=tree.verification_position_ids(base_kv_length=cache.get_seq_length(), device=verify_ids.device),
                            past_key_values=cache, use_cache=True, logits_to_keep=0)
                        torch.cuda.synchronize()
                        atlas = [self.execution.routes[i].reshape(-1, 8).cpu().tolist() for i in range(48)]
                        del output, cache
                    finally:
                        self.execution.probing = False
                else:
                    source_row = self.reference["cycles"][len(self.rows)]
                    if (source_row["pool"]["fingerprint"] != fingerprint
                            or source_row["committed_offset"] != committed_offset):
                        raise RuntimeError("measured prefix or candidate pool differs from atlas preparation")
                    atlas = source_row["atlas"]
            choose_variant = getattr(experiment, "choose_variant", None)
            if choose_variant is not None:
                selected = choose_variant(view, self.method, model, atlas, variant=self.variant)
            else:
                selected = (experiment.choose(view, self.method, model, atlas) if experiment else
                            choose(view, self.method, model, atlas))
            self.rows.append(dict(pool_object=pool, iteration=len(self.rows),
                committed_offset=committed_offset, selected_raw_ids=list(selected), atlas=atlas))
            return selected

        def begin_cycle(self, **kwargs):
            pass

        def end_cycle(self, *, verifier_tokens, committed_token_ids, **kwargs):
            self.rows[-1]["committed_tokens"] = committed_token_ids.flatten().cpu().tolist()

        def abort_cycle(self):
            pass

        def finish_generation(self, **kwargs):
            pass

        def materialize(self, *, include_atlas):
            result = []
            for row in self.rows:
                value = {k: v for k, v in row.items() if k not in ("pool_object", "atlas")}
                value["pool"] = serialize_candidate_pool(row["pool_object"])
                if include_atlas and self.method == "expert_fast_dynamic":
                    value["atlas"] = row["atlas"]
                result.append(value)
            return result

    def question(method, qid, *, reference, repetition, ordinal, variant="baseline"):
        gc.enable()
        gc.collect()
        gc.disable()
        budget()
        work = prepared[qid]
        ids = work["ids"].to("cuda:0")
        attention = torch.ones_like(ids)
        before = cache_boundary(loaded)
        loaded.set_expert_execution_policy(ExpertExecutionPolicy.EXPERT_ID_ASCENDING)
        execution = Execution(variant)
        selector = Selector(method, execution, reference, variant)
        prefill_end = None
        def prefill_done():
            nonlocal prefill_end
            torch.cuda.synchronize()
            prefill_end = time.perf_counter_ns()
            loaded.set_expert_execution_policy(ExpertExecutionPolicy.RESIDENT_FIRST)
            execution.decode = True
        with torch.inference_mode(), execution.observe():
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            if method == "target_only":
                result = target_greedy_generate(loaded.model, ids, attention_mask=attention,
                    max_new_tokens=TOTAL_TOKENS - ids.shape[1], on_prefill_complete=prefill_done,
                    stop_on_eos=True)
            else:
                tree_drafter = (experiment.tree_drafter(drafter, variant) if experiment else
                                Eagle3TreeDrafter(drafter))
                result = ShallowTreeSpeculator(loaded.model, tree_drafter)._generate(
                    ids, attention_mask=attention, max_new_tokens=TOTAL_TOKENS - ids.shape[1],
                    top_k=2, max_depth=5, draft_node_budget=18, stop_on_eos=True,
                    audit_commits=reference is None, collect_cycle_metrics=True,
                    selector=ExplicitRawIdTreeSelector(selector), trace=selector,
                    on_prefill_complete=prefill_done, _on_boundary=selector.at_boundary)
            # Experimental persistent execution pays setup in its first decode
            # forward and teardown here, inside both request and decode timing.
            execution.close()
            torch.cuda.synchronize()
            ended = time.perf_counter_ns()
        proofs = execution.materialize()
        tokens = result.tokens.flatten().cpu().tolist()
        output = tokenizer.decode(tokens, skip_special_tokens=True)
        cycles = selector.materialize(include_atlas=reference is None)
        audits = [dict(**asdict(a), passed=a.passed) for a in result.commit_audits]
        reference_match = reference is None or (tokens == reference["output_token_ids"]
                                                and proofs == reference["query_proofs"])
        commits_passed = all(a["passed"] for a in audits)
        after = cache_boundary(loaded)
        truth = numeric_answer(examples[qid].reference[0], reference=True)
        prediction = numeric_answer(output)
        row = dict(protocol=PROTOCOL, status="complete", method=method, question_id=qid,
            repetition=repetition, ordinal=ordinal, input_tokens=ids.shape[1],
            input_ids_sha256=canonical_json_sha256(ids.flatten().cpu().tolist()),
            rendered_prompt=work["prompt"], output_token_ids=tokens, output_tokens=len(tokens),
            output_text=output, stop_reason="eos" if result.stopped_on_eos else "length",
            stop_token_id=result.stop_token_id, request_ns=ended-started,
            prefill_ns=prefill_end-started, decode_ns=ended-prefill_end,
            pre_cache_sha256=before["sha256"], post_cache_sha256=after["sha256"],
            pre_cache_occupied=before["occupied"], post_cache_occupied=after["occupied"],
            pre_cache=before["layers"], post_cache=after["layers"],
            cache_delta={k: after["counters"][k]-before["counters"][k] for k in COUNTERS},
            scheduler=execution.scheduler_summary(),
            cycles=cycles, cycle_metrics=[asdict(m) for m in result.tree_cycle_metrics],
            query_proofs=proofs, commit_audits=audits, all_commits_passed=commits_passed,
            reference_match=reference_match, reference_sha256=canonical_json_sha256(reference) if reference else None,
            reference_answer=truth, predicted_answer=prediction,
            answer_correct=truth is not None and prediction == truth,
            max_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
            max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        if experiment:
            row.update(variant=variant, experiment_protocol=frozen["protocol"],
                       execution_counts=dict(execution.runtime.counts))
        budget()
        if not reference_match or not commits_passed:
            row["status"] = "failed"
            descriptor = write_json(args.directory / f"failed-{method}-r{repetition}-q{qid}.json.gz", row)
            index["failed_question"] = descriptor
            raise RuntimeError(f"{method} q{qid} failed same-query replay or commit audit")
        return row

    try:
        loaded = load_offloaded_qwen3(m1._TARGET, device=torch.device("cuda:0"), slots_per_layer=48,
            packed_checkpoint_root=m1._PACKED, packed_expert_backing="registered_mmap", verify_packed_hashes=True,
            reduction_mode=ReductionMode.CANONICAL, expert_execution_policy=ExpertExecutionPolicy.EXPERT_ID_ASCENDING,
            h2d_prefetch=False)
        numeric = apply_numeric_mode("math_bf16_full_reduction")
        m1._require_m1_eos_ids(loaded.model)
        index["environment"]["load_timings"] = asdict(loaded.load_timings)
        index["environment"]["numeric_settings"] = dict(numeric.actual_settings)
        drafter = Eagle3ChainDrafter.from_pretrained(m1._DRAFTER, target_root=m1._TARGET,
            device=torch.device("cuda:0"), use_torch_compile=False,
            expected_target_repo_id=source["target"]["repo_id"])
        warmup_eagle3_tree_drafter(drafter, Eagle3TreeDrafter(drafter), max_sequence_length=506,
            max_depth=5, top_k=2, draft_node_budget=18, sequence_anchors=(64, 192, 320, 506))
        index["prepare_ns"] = time.perf_counter_ns() - began
        event(phase="models_ready")
        references = experiment.references(args.directory, frozen, index) if experiment else {}
        for method in (() if experiment else METHODS):
            loaded.cold_reset_expert_caches()
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            previous = None
            for ordinal, qid in enumerate(frozen["question_ids"]):
                row = question(method, qid, reference=None, repetition=-1, ordinal=ordinal)
                descriptor = write_json(args.directory / f"reference-{method}-q{qid}.json.gz", row)
                index["references"].append(descriptor)
                if previous is not None and row["pre_cache_sha256"] != previous:
                    raise RuntimeError("reference stream cache continuity failed")
                previous = row["post_cache_sha256"]
                if method == "expert_fast_dynamic":
                    validate_capture(row, model)
                references[method, qid] = descriptor
                event(phase="reference_complete", method=method, question_id=qid,
                      output_tokens=row["output_tokens"], stop_reason=row["stop_reason"])
        write_json(args.directory / "references.json", dict(files=index["references"], completed_ns=time.time_ns()))
        for job in frozen["stream_schedule"]:
            method, repeat = job["method"], job["repetition"]
            loaded.cold_reset_expert_caches()
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            stream = []
            for ordinal, qid in enumerate(frozen["question_ids"]):
                descriptor = references[method, qid]
                reference = read_json(args.directory / descriptor["basename"], descriptor)
                variant = job.get("variant", "baseline")
                row = question(method, qid, reference=reference, repetition=repeat, ordinal=ordinal, variant=variant)
                tag = variant+"-" if experiment else ""
                descriptor = write_json(args.directory / f"measure-{tag}{method}-r{repeat}-q{qid}.json.gz", row)
                index["questions"].append(descriptor)
                stream.append(row)
                event(phase="measurement_complete", method=method, variant=variant, repetition=repeat,
                    question_id=qid, output_tokens=row["output_tokens"],
                    decode_tokens_per_second=(row["output_tokens"]-1)*1e9/row["decode_ns"],
                    stop_reason=row["stop_reason"])
            validate_stream(stream, question_ids=frozen["question_ids"], method=method, repetition=repeat)
        index.update(status="complete", finished_ns=time.time_ns())
        write_json(args.directory / "result.json", index)
        if experiment:
            experiment.validate(args.directory)
        else:
            validate(args.directory)
    except BaseException as error:
        index.update(status="failed", error=repr(error), finished_ns=time.time_ns())
        write_json(args.directory / "failure.json", index)
        raise
    finally:
        if numeric is not None:
            numeric.restore()
        if loaded is not None:
            loaded.close()
        if gc_enabled:
            gc.enable()


def main():
    args = arguments()
    if args.command == "freeze":
        freeze(args)
    elif args.command == "validate":
        validate(args.directory)
    else:
        run(args)


if __name__ == "__main__":
    main()
