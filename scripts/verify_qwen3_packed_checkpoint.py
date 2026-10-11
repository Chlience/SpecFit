#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from specfit.adapters.qwen3_checkpoint import CheckpointLayout
from specfit.adapters.qwen3_packed_checkpoint import PackedQwen3Checkpoint


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Source Hugging Face checkpoint directory",
    )
    parser.add_argument(
        "--packed-cache",
        type=Path,
        required=True,
        help="Derived checkpoint directory requiring full verification",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON result path",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    verify_start = time.perf_counter()
    source_layout = CheckpointLayout.from_pretrained(args.model)
    packed = PackedQwen3Checkpoint.open(
        args.packed_cache,
        source_layout=source_layout,
        verify_hashes=True,
    )
    result = {
        "schema_version": 1,
        "source_model": str(args.model.resolve()),
        "packed_checkpoint": str(packed.root),
        "source": packed.manifest["source"],
        "file_hashes_verified": True,
        "num_hidden_layers": packed.census.num_hidden_layers,
        "num_experts": packed.census.num_experts,
        "routed_expert_bytes": packed.census.routed_expert_bytes,
        "nonexpert_bytes": packed.census.nonexpert_bytes,
        "verify_seconds": time.perf_counter() - verify_start,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
