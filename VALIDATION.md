# Release validation

The initial public release was prepared on 2026-10-11. The experimental source,
tests, Docker environment, dataset locks, configurations, and result files were
extracted from the anonymized supplementary material. The code tree was moved to
the repository root without changing its Python package or entry-point paths.
Of the 197 files under `src/`, `scripts/`, `tests/`, `docker/`, `workloads/`,
`docs/`, `configs/`, and `results/`, 195 were preserved byte for byte; only
trailing empty lines were removed from `docker/base-image.lock.json` and
`scripts/docker/build-image.sh`. All Python code, experiment configurations, and
result files are unchanged. Release changes add documentation, a license, a CPU
test dependency list, and ignore rules.

## Checks performed

| Check | Result and coverage |
| --- | --- |
| `PYTHONPATH=src python -m pytest tests/unit -q -p no:cacheprovider` | 239 tests and 356 subtests passed; no skipped tests. Runtime: 104.95 seconds. PyTorch emitted 14 `torch.jit.script_method` deprecation warnings. |
| Theory replay with `python3 results/theory/theory_checks.py` | Parsed output exactly equals `results/theory/theory-checks.json`. Reproduces the recorded subtree and adaptive stopping rule for all 4,407 GSM8K cycles and all 4,389 summarization cycles. |
| Selection timing summaries | Recomputed count, median, p95, mean, total selection time, and share of decode time from `selector_wall_ns` in `results/selection-cost/selection-time.json`; all values agree within relative/absolute tolerance `1e-12`. |
| CLI import and argument parsing | `--help` succeeded for all 20 top-level measurement, preparation, controller, and analysis entry points. `specfit_main_recovery.py` is a shared helper. |
| Python and JSON syntax | All 142 Python files compiled; all 48 JSON documents, including the compressed candidate-pool JSON, parsed successfully. |
| Internal imports | Static inspection found no missing absolute `specfit.*` module imports. |
| Staged whitespace | `git diff --cached --check` passed after removing trailing empty lines from three imported files; no behavior changes. |
| Shell syntax | `bash -n` passed for `scripts/docker/build-image.sh` and `scripts/docker/run-a100.sh`. |
| Local documentation links | Relative Markdown targets in the release documentation resolve. |
| Release content scan | Text and decompressed candidate-pool content showed no matches for the scanned token/private-key/embedded-credential patterns or original private repository and user paths. This is a pattern-based check. |

The unit test environment used Python 3.12.13, PyTorch 2.10.0+cpu, pytest 9.0.2,
NumPy 2.1.0, safetensors 0.7.0, Transformers 5.6.0, huggingface-hub 1.5.0,
Accelerate 1.14.0, and Speculators 0.5.0. Direct test dependencies are listed in
`requirements-test.txt`; this CPU environment is separate from the historical
NGC measurement image pinned by the Docker files.

## Verification limits

The digest-pinned measurement image was not rebuilt, GPU tests were not run,
and the full GPU experiments were not repeated during release preparation. The
model downloads and hardware-specific setup were not recreated. CPU test success
does not establish equivalence to the NGC GPU environment or reproduce the
reported GPU throughput.

The release omits the full per-request measurement and reference traces, so the
aggregate throughput results cannot be independently regenerated from these
files alone. Theory replay and selection timing summaries use the records that
are included. Other analyses require newly measured traces.

Historical commit IDs, host paths, GPU UUIDs, and related identifiers are
anonymized. Configure local hardware and image identities, commit the changes,
and freeze new manifests as described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## File integrity

`SHA256SUMS` records the SHA-256 of every released file except the checksum file
itself. After cloning, verify it from the repository root:

```bash
sha256sum -c SHA256SUMS
```
