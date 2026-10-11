#!/usr/bin/env python3
"""Prepare pinned HumanEval, AIME25 and Spec-Bench summarization inputs for later cross-dataset evaluation."""
from argparse import ArgumentParser
import gzip
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.request import urlopen

LOCK = Path(__file__).resolve().parents[1]/"workloads/extended_benchmarks.lock.json"


def normalize(name, data, spec):
    if len(data) != spec["bytes"] or sha256(data).hexdigest() != spec["sha256"]:
        raise ValueError("dataset bytes differ from the pinned source")
    text = gzip.decompress(data).decode() if name == "humaneval" else data.decode()
    rows = [json.loads(line) for line in text.splitlines()]
    if len(rows) != spec["count"]:
        raise ValueError("dataset count differs from the pinned source")
    result = []
    for index, row in enumerate(rows):
        if name == "humaneval":
            source_id = row["task_id"]
            if source_id != f"HumanEval/{index}" or any(
                not isinstance(row[key], str) or not row[key]
                for key in ("prompt", "entry_point", "test", "canonical_solution")
            ):
                raise ValueError("HumanEval identity or required content differs")
            prompt = ("Complete the following Python function. Return only the complete Python code, "
                      "without Markdown fences.\n\n"+row["prompt"])
            evaluation = dict(kind="python_unit_tests", entry_point=row["entry_point"],
                              test=row["test"], canonical_solution=row["canonical_solution"],
                              original_prompt=row["prompt"])
        elif name == "aime25":
            source_id = row["id"]
            if (source_id != str(index) or not isinstance(row["problem"], str) or not row["problem"]
                    or type(row["answer"]) not in (str, int)
                    or not re.fullmatch(r"[0-9]{1,3}", str(row["answer"]))):
                raise ValueError("AIME25 identity, problem or integer answer differs")
            prompt = row["problem"]+"\n\nGive your final answer as an integer in \\boxed{...}."
            evaluation = dict(kind="integer_exact_match", answer=str(int(row["answer"])))
        elif name == "summarization":
            source_id = str(row["question_id"])
            if (type(row["question_id"]) is not int or row["question_id"] != spec["source_question_ids"][0]+index
                    or row["category"] != "summarization" or not isinstance(row["prompt"], str)
                    or not row["prompt"].startswith("Summarize: ") or not isinstance(row["reference"], list)
                    or len(row["reference"]) != 1 or not isinstance(row["reference"][0], str) or not row["reference"][0]):
                raise ValueError("summarization identity, prompt or reference differs")
            prompt = row["prompt"]  # Spec-Bench wording, verbatim; the reference never enters the prompt.
            evaluation = dict(kind="reference_summary_not_scored", reference=row["reference"][0])
        else:
            raise ValueError("unknown evaluation dataset")
        result.append(dict(dataset=name, question_id=index, source_id=source_id,
                           prompt=prompt, evaluation=evaluation))
    return result


def preserve_write(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"refusing to overwrite different data: {path}")
        return
    with path.open("xb") as target:
        target.write(data)


def prepare(name, output_dir, *, source_directory=None):
    lock = json.loads(LOCK.read_text())
    spec = lock["datasets"][name]
    destination = Path(output_dir)/name/spec["revision"]
    existing = destination/spec["filename"]
    source = Path(source_directory)/spec["filename"] if source_directory else existing
    if source.exists():
        data = source.read_bytes()
    else:
        if source_directory:
            raise FileNotFoundError(source)
        with urlopen(spec["url"], timeout=60) as response:
            data = response.read()
    rows = normalize(name, data, spec)
    encoded = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True)+"\n" for row in rows).encode()
    receipt = dict(protocol=lock["protocol"], dataset=name, source=spec,
                   prepared_file="requests.jsonl", prepared_sha256=sha256(encoded).hexdigest(),
                   prepared_bytes=len(encoded), examples=len(rows),
                   status="prepared_not_measured", generation_total_tokens=2048 if name == "summarization" else 512,
                   tokenized_prompt_validation="pending")
    destination.mkdir(parents=True, exist_ok=True)
    preserve_write(existing, data)
    preserve_write(destination/"requests.jsonl", encoded)
    preserve_write(destination/"provenance.json", (json.dumps(receipt, indent=2)+"\n").encode())
    return destination/"requests.jsonl", receipt


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("humaneval", "aime25", "summarization", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path)
    args = parser.parse_args()
    names = tuple(json.loads(LOCK.read_text())["datasets"]) if args.dataset == "all" else (args.dataset,)
    for name in names:
        path, receipt = prepare(name, args.output_dir, source_directory=args.source_directory)
        print(json.dumps(dict(path=str(path), **receipt)))
