#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


PACKAGES = [
    "torch",
    "torchvision",
    "torch-tensorrt",
    "transformer-engine",
    "flash-attn",
    "triton",
    "numpy",
    "numba",
    "llvmlite",
    "transformers",
    "huggingface-hub",
    "speculators",
    "accelerate",
]


def run(*command: str) -> str:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return (completed.stdout or completed.stderr).strip()


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            if name == "triton":
                import triton

                versions[name] = triton.__version__
            else:
                versions[name] = "MISSING"
    return versions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-commit", default="uncommitted")
    parser.add_argument("--image-id", default="unknown")
    args = parser.parse_args()

    requirements_lock = Path("/opt/specfit/requirements.overlay.lock")
    manifest = {
        "captured_at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": package_versions(),
        "source_commit": args.source_commit,
        "derived_image_id": args.image_id,
        "requirements_overlay_sha256": hashlib.sha256(
            requirements_lock.read_bytes()
        ).hexdigest(),
        "gpu": run(
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,pci.bus_id,pstate",
            "--format=csv,noheader",
        ),
        "gpu_topology": run("nvidia-smi", "topo", "-m"),
        "numa": run("numactl", "--hardware"),
        "base_image": json.loads(
            Path("/opt/specfit/base-image.lock.json").read_text(encoding="utf-8")
        ),
    }
    rendered = json.dumps(manifest, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(args.output)
    else:
        print(rendered)


if __name__ == "__main__":
    main()
