#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from specfit.gsm8k import (
    GSM8KSpecBenchLock,
    load_gsm8k_specbench,
    load_gsm8k_specbench_lock,
)


DownloadFile = Callable[..., str]


def default_output_dir(lock: GSM8KSpecBenchLock) -> Path:

    return Path("/data/workloads/gsm8k") / lock.revision


def _parse_args(
    argv: Sequence[str] | None,
    *,
    lock: GSM8KSpecBenchLock,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and verify the fixed 80-question GSM8K Spec-Bench set",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(lock),
        help="Data file directory, isolated by locked revision by default",
    )
    return parser.parse_args(argv)


def _hf_hub_download(**kwargs: Any) -> str:

    from huggingface_hub import hf_hub_download

    return hf_hub_download(**kwargs)


def prepare_gsm8k_workload(
    output_dir: str | Path,
    *,
    lock: GSM8KSpecBenchLock,
    download_file: DownloadFile | None = None,
) -> tuple[Path, bool]:

    destination_dir = Path(output_dir).expanduser()
    workload_path = destination_dir / lock.filename

    if workload_path.exists():
        load_gsm8k_specbench(workload_path)
        return workload_path, True

    destination_dir.mkdir(parents=True, exist_ok=True)
    downloader = download_file or _hf_hub_download
    downloader(
        repo_id=lock.repo_id,
        repo_type="dataset",
        revision=lock.revision,
        filename=lock.filename,
        local_dir=str(destination_dir),
    )
    load_gsm8k_specbench(workload_path)
    return workload_path, False


def main(
    argv: Sequence[str] | None = None,
    *,
    download_file: DownloadFile | None = None,
) -> int:
    lock = load_gsm8k_specbench_lock()
    args = _parse_args(argv, lock=lock)
    workload_path, reused = prepare_gsm8k_workload(
        args.output_dir,
        lock=lock,
        download_file=download_file,
    )
    print(
        f"status={'reused' if reused else 'downloaded'} "
        f"examples={lock.expected_count} path={workload_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
