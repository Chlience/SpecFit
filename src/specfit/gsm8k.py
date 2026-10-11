
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_GSM8K_LOCK_PATH = (
    Path(__file__).resolve().parents[2] / "workloads/gsm8k_specbench.lock.json"
)


class GSM8KWorkloadError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GSM8KSpecBenchLock:

    repo_id: str
    revision: str
    filename: str
    category: str
    expected_count: int
    question_id_start: int
    question_id_end: int

    @property
    def expected_question_ids(self) -> tuple[int, ...]:

        return tuple(range(self.question_id_start, self.question_id_end + 1))


@dataclass(frozen=True, slots=True)
class GSM8KExample:

    question_id: int
    category: str
    reference: tuple[str, ...]
    prompt: str


def load_gsm8k_specbench_lock(
    path: str | Path = DEFAULT_GSM8K_LOCK_PATH,
) -> GSM8KSpecBenchLock:

    lock_path = Path(path)
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        dataset = payload["dataset"]
        return GSM8KSpecBenchLock(
            repo_id=_required_str(dataset, "repo_id"),
            revision=_required_str(dataset, "revision"),
            filename=_required_str(dataset, "filename"),
            category=_required_str(dataset, "category"),
            expected_count=_required_int(dataset, "expected_count"),
            question_id_start=_required_int(dataset, "question_id_start"),
            question_id_end=_required_int(dataset, "question_id_end"),
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise GSM8KWorkloadError(
            f"failed to read GSM8K workload lock file {lock_path}: {error}"
        ) from error


def load_gsm8k_specbench(
    path: str | Path,
    *,
    lock_path: str | Path = DEFAULT_GSM8K_LOCK_PATH,
) -> tuple[GSM8KExample, ...]:

    workload_path = Path(path)
    lock = load_gsm8k_specbench_lock(lock_path)
    try:
        lines = workload_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise GSM8KWorkloadError(
            f"failed to read GSM8K workload {workload_path}: {error}"
        ) from error

    examples: list[GSM8KExample] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise GSM8KWorkloadError(
                f"{workload_path}:{line_number} contains a blank line"
            )
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise GSM8KWorkloadError(
                f"{workload_path}:{line_number} is not valid JSON: {error}"
            ) from error
        examples.append(
            _parse_example(record, workload_path, line_number)
        )

    if len(examples) != lock.expected_count:
        raise GSM8KWorkloadError(
            f"{workload_path} should contain {lock.expected_count} samples, "
            f"actual: {len(examples)} entries"
        )

    question_ids = tuple(example.question_id for example in examples)
    if question_ids != lock.expected_question_ids:
        raise GSM8KWorkloadError(
            f"{workload_path} question_id must cover in order "
            f"{lock.question_id_start}..{lock.question_id_end}"
        )

    unexpected_categories = sorted(
        {example.category for example in examples if example.category != lock.category}
    )
    if unexpected_categories:
        raise GSM8KWorkloadError(
            f"{workload_path} contains non-{lock.category} category: "
            f"{unexpected_categories}"
        )

    return tuple(examples)


def _parse_example(
    record: Any,
    workload_path: Path,
    line_number: int,
) -> GSM8KExample:
    if not isinstance(record, dict):
        raise GSM8KWorkloadError(
            f"{workload_path}:{line_number} must be a JSON object"
        )

    try:
        reference = record["reference"]
        if not isinstance(reference, list) or not reference:
            raise TypeError("reference must be a non-empty array of strings")
        if not all(isinstance(item, str) and item for item in reference):
            raise TypeError("reference must be a non-empty array of strings")
        return GSM8KExample(
            question_id=_required_int(record, "question_id"),
            category=_required_str(record, "category"),
            reference=tuple(reference),
            prompt=_required_str(record, "prompt"),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise GSM8KWorkloadError(
            f"{workload_path}:{line_number} field invalid: {error}"
        ) from error


def _required_str(record: dict[str, Any], key: str) -> str:
    value = record[key]
    if not isinstance(value, str) or not value:
        raise TypeError(f"{key} must be a non-empty string")
    return value


def _required_int(record: dict[str, Any], key: str) -> int:
    value = record[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{key} must be an integer")
    return value
