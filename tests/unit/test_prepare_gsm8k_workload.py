from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from specfit.gsm8k import GSM8KWorkloadError, load_gsm8k_specbench_lock


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/prepare_gsm8k_workload.py"


def _load_prepare_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_gsm8k_workload",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load prepare script: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_workload(path: Path, question_ids=range(401, 481)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                {
                    "question_id": question_id,
                    "category": "math_reasoning",
                    "reference": [f"reference {question_id}"],
                    "prompt": f"question {question_id}",
                }
            )
            + "\n"
            for question_id in question_ids
        ),
        encoding="utf-8",
    )


class PrepareGSM8KWorkloadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.prepare_module = _load_prepare_module()
        self.lock = load_gsm8k_specbench_lock()
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temporary_directory.name) / "gsm8k"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_default_output_directory_is_revision_scoped(self) -> None:
        self.assertEqual(
            self.prepare_module.default_output_dir(self.lock),
            Path("/data/workloads/gsm8k") / self.lock.revision,
        )

    def test_downloads_locked_dataset_and_validates_all_examples(self) -> None:
        def download_fixture(**kwargs: object) -> str:
            workload_path = Path(str(kwargs["local_dir"])) / self.lock.filename
            _write_workload(workload_path)
            return str(workload_path)

        downloader = mock.Mock(side_effect=download_fixture)
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            exit_code = self.prepare_module.main(
                ["--output-dir", str(self.output_dir)],
                download_file=downloader,
            )

        self.assertEqual(exit_code, 0)
        downloader.assert_called_once_with(
            repo_id=self.lock.repo_id,
            repo_type="dataset",
            revision=self.lock.revision,
            filename=self.lock.filename,
            local_dir=str(self.output_dir),
        )
        self.assertIn("status=downloaded", output.getvalue())
        self.assertIn("examples=80", output.getvalue())

    def test_reuses_existing_valid_file_without_downloading(self) -> None:
        workload_path = self.output_dir / self.lock.filename
        _write_workload(workload_path)
        downloader = mock.Mock()
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            exit_code = self.prepare_module.main(
                ["--output-dir", str(self.output_dir)],
                download_file=downloader,
            )

        self.assertEqual(exit_code, 0)
        downloader.assert_not_called()
        self.assertIn("status=reused", output.getvalue())
        self.assertIn(str(workload_path), output.getvalue())

    def test_keeps_existing_invalid_file_and_does_not_download(self) -> None:
        workload_path = self.output_dir / self.lock.filename
        workload_path.parent.mkdir(parents=True)
        original = b"local invalid data\n"
        workload_path.write_bytes(original)
        downloader = mock.Mock()

        with self.assertRaises(GSM8KWorkloadError):
            self.prepare_module.main(
                ["--output-dir", str(self.output_dir)],
                download_file=downloader,
            )

        downloader.assert_not_called()
        self.assertEqual(workload_path.read_bytes(), original)

    def test_rejects_incomplete_download(self) -> None:
        def download_incomplete(**kwargs: object) -> str:
            workload_path = Path(str(kwargs["local_dir"])) / self.lock.filename
            _write_workload(workload_path, range(401, 480))
            return str(workload_path)

        downloader = mock.Mock(side_effect=download_incomplete)

        with self.assertRaises(
            GSM8KWorkloadError,
        ):
            self.prepare_module.main(
                ["--output-dir", str(self.output_dir)],
                download_file=downloader,
            )

        downloader.assert_called_once()


if __name__ == "__main__":
    unittest.main()
