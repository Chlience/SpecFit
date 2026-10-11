#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from specfit.adapters.qwen3_packed_checkpoint import (
    pack_qwen3_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Hugging Face Qwen3-MoE checkpoint into a per-layer contiguous expert layout"
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Source Hugging Face checkpoint directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New derived checkpoint directory; this path must not already exist",
    )
    parser.add_argument(
        "--source-repo-id",
        help="Hugging Face source repo identifier written to the manifest",
    )
    parser.add_argument(
        "--source-revision",
        help="Fixed source model revision written to manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    build_start = time.perf_counter()
    packed = pack_qwen3_checkpoint(
        args.model,
        args.output,
        source_repo_id=args.source_repo_id,
        source_revision=args.source_revision,
    )
    summary = {
        "root": str(packed.root),
        "manifest": str(packed.root / "manifest.json"),
        "num_hidden_layers": packed.census.num_hidden_layers,
        "num_experts": packed.census.num_experts,
        "packed_expert_elements": packed.packed_expert_elements,
        "routed_expert_bytes": packed.census.routed_expert_bytes,
        "nonexpert_bytes": packed.census.nonexpert_bytes,
        "build_seconds": time.perf_counter() - build_start,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
