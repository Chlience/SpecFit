from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from specfit.gsm8k import (
    DEFAULT_GSM8K_LOCK_PATH,
    GSM8KWorkloadError,
    load_gsm8k_specbench,
    load_gsm8k_specbench_lock,
)


class GSM8KWorkloadTest(unittest.TestCase):
    def test_project_lock_matches_redhat_specbench_workload(self) -> None:
        lock = load_gsm8k_specbench_lock()

        self.assertEqual(lock.repo_id, "RedHatAI/speculator_benchmarks")
        self.assertEqual(
            lock.revision,
            "e383d21cf8c005b0fcba4b8d27a2800a075fed0d",
        )
        self.assertEqual(lock.filename, "math_reasoning.jsonl")
        self.assertEqual(lock.category, "math_reasoning")
        self.assertEqual(lock.expected_count, 80)
        self.assertEqual(lock.expected_question_ids, tuple(range(401, 481)))
        self.assertTrue(DEFAULT_GSM8K_LOCK_PATH.is_file())

    def test_loads_all_80_examples_in_stable_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workload_path = Path(temporary_directory) / "math_reasoning.jsonl"
            self._write_fixture(workload_path)

            examples = load_gsm8k_specbench(workload_path)

        self.assertEqual(len(examples), 80)
        self.assertEqual(examples[0].question_id, 401)
        self.assertEqual(examples[-1].question_id, 480)
        self.assertEqual(examples[0].category, "math_reasoning")
        self.assertEqual(examples[0].reference, ("reference 401",))
        self.assertEqual(examples[0].prompt, "question 401")

    def test_rejects_missing_example_instead_of_silently_truncating(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workload_path = Path(temporary_directory) / "math_reasoning.jsonl"
            self._write_fixture(workload_path, question_ids=range(401, 480))

            with self.assertRaises(
                GSM8KWorkloadError,
            ):
                load_gsm8k_specbench(workload_path)

    def test_rejects_reordered_or_replaced_question_id(self) -> None:
        question_ids = list(range(401, 481))
        question_ids[-1] = 401
        with tempfile.TemporaryDirectory() as temporary_directory:
            workload_path = Path(temporary_directory) / "math_reasoning.jsonl"
            self._write_fixture(workload_path, question_ids=question_ids)

            with self.assertRaises(
                GSM8KWorkloadError,
            ):
                load_gsm8k_specbench(workload_path)

    def test_rejects_invalid_required_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workload_path = Path(temporary_directory) / "math_reasoning.jsonl"
            records = self._records()
            records[0]["reference"] = []
            self._write_records(workload_path, records)

            with self.assertRaises(
                GSM8KWorkloadError,
            ):
                load_gsm8k_specbench(workload_path)

    @classmethod
    def _write_fixture(
        cls,
        path: Path,
        *,
        question_ids: range | list[int] = range(401, 481),
    ) -> None:
        cls._write_records(path, cls._records(question_ids))

    @staticmethod
    def _records(
        question_ids: range | list[int] = range(401, 481),
    ) -> list[dict[str, object]]:
        return [
            {
                "question_id": question_id,
                "category": "math_reasoning",
                "reference": [f"reference {question_id}"],
                "prompt": f"question {question_id}",
            }
            for question_id in question_ids
        ]

    @staticmethod
    def _write_records(path: Path, records: list[dict[str, object]]) -> None:
        path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
