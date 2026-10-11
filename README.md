<div align="center">

<h1>SpecFit</h1>
<p><strong>Expert-aware speculative decoding for offloaded MoE models</strong></p>
<p>Choose verification subtrees by predicted throughput, accounting for expert transfers and GPU cache state.</p>

<p>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-0F766E?style=flat-square" alt="License: Apache 2.0"></a>
  <a href="#cpu-unit-tests"><img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python: 3.12"></a>
  <a href="requirements-test.txt"><img src="https://img.shields.io/badge/PyTorch-2.10%20CPU-EE4C2C?style=flat-square&amp;logo=pytorch&amp;logoColor=white" alt="CPU tests: PyTorch 2.10"></a>
  <a href="docker/README.md"><img src="https://img.shields.io/badge/Docker-NGC%2025.11-2496ED?style=flat-square&amp;logo=docker&amp;logoColor=white" alt="GPU environment: NGC 25.11"></a>
  <a href="VALIDATION.md"><img src="https://img.shields.io/badge/Release%20tests-239%20passed-0F766E?style=flat-square" alt="Initial release tests: 239 passed"></a>
</p>

<p>
  <a href="#quick-start">Quick start</a> ·
  <a href="REPRODUCIBILITY.md">Reproduction guide</a> ·
  <a href="configs/">Configurations</a> ·
  <a href="results/">Results</a> ·
  <a href="VALIDATION.md">Validation</a>
</p>

</div>

## Overview

SpecFit selects verification subtrees for speculative decoding of expert-offloaded mixture-of-experts models. It estimates the committed tokens and verification cost of candidate subtrees and selects the subtree with the highest predicted throughput. Its adaptive drafting rule can stop before the maximum drafting depth.

| Selection objective | Cost model | Drafting rule |
| --- | --- | --- |
| Predicted committed tokens per unit verification time | Expert transfers and GPU expert cache state | Stop when the selected subtree needs no node from the newest level |

The experiments use **Qwen3-30B-A3B-Instruct-2507** with an **EAGLE-3** drafter on an **NVIDIA A100 PCIe 40GB** and an **RTX 4090 24GB**. The repository includes the implementation, frozen configurations, and analysis outputs.

> [!IMPORTANT]
> Reported SpecFit measurements use routes collected in separate reference runs; route acquisition is excluded from timing. Route accuracy experiments use simulated predictions, and linear predictor cost is an analytic estimate. These results do not establish online end-to-end performance with a deployed route predictor.

## Quick start

| Goal | Start here | Requirements |
| --- | --- | --- |
| Explore the experiments | [Frozen configurations](configs/) and [analysis results](results/) | Browse the released files |
| Recompute the theory results | [CPU-only replay](#recompute-the-theory-results) | Python standard library; no GPU |
| Run the unit tests | [CPU test environment](#cpu-unit-tests) | Python 3.12 and the test dependencies |
| Measure decoding on a GPU | [GPU experiments](#run-gpu-experiments) | Locked models, measurement image, and configured hardware |

### Recompute the theory results

```bash
git clone https://github.com/chlience/SpecFit.git
cd SpecFit
python3 results/theory/theory_checks.py > /tmp/specfit-theory-checks.json
python3 - <<'PY'
import json
from pathlib import Path
recorded = json.loads(Path("results/theory/theory-checks.json").read_text())
replayed = json.loads(Path("/tmp/specfit-theory-checks.json").read_text())
assert recorded == replayed
print("Recomputed theory results match the released results.")
PY
```

**Initial release validation (2026-10-11):** 239 CPU tests and 356 subtests passed; theory replay matched all **8,796 recorded cycles**. The badges summarize this release and its environments. Full check results and coverage are recorded in [VALIDATION.md](VALIDATION.md).

The released selection timing file also includes the per-cycle measurements. Full per-request traces are omitted (about 0.6 GB including reference runs). Recomputing the remaining aggregate results requires freshly generated traces.

### CPU unit tests

Run from the repository root with Python 3.12. This environment is for CPU unit tests; GPU measurements use the separately pinned [Docker environment](docker/README.md).

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-test.txt
PYTHONPATH=src .venv/bin/python -m pytest tests/unit -q
```

### Run GPU experiments

1. Build the measurement image with `bash scripts/docker/build-image.sh`.
2. Download the models at the revisions in `docker/models.lock.json` and prepare
   the locked workloads.
3. Pack and verify the target checkpoint.
4. Configure the local image identity, GPU UUIDs, host paths, and CPU/NUMA affinity
   using the [new-machine setup guide](REPRODUCIBILITY.md#preparing-a-new-machine).
5. Commit the setup changes, then freeze, run, validate, and analyze a new experiment.

The measurement entry point is `scripts/run_specfit_main_comparison.py` with `freeze`, `run`, and `validate` commands. Host controllers launch these stages; analysis scripts validate completed runs. Always freeze from a clean, committed checkout. Historical manifests retain anonymized revision IDs and hardware identities and require a new freeze on a new machine.

GPU tests additionally require the measurement environment, locked models, and packed checkpoint. The initial release checks did not rebuild the measurement image, run GPU tests, or repeat the full GPU experiments.

## Repository map

<details>
<summary><strong>Source, experiment artifacts, and environment locks</strong></summary>

| Directory | Contents |
| --- | --- |
| [`src/specfit/`](src/specfit/) | Expert offloading, GPU expert cache, EAGLE-3 drafting and verification, exact subtree selection, and experiment validation |
| [`scripts/`](scripts/) | Workload preparation, checkpoint packing, measurement controllers, and result analysis |
| [`docker/`](docker/) | Measurement Dockerfile and pinned environment and model revisions |
| [`workloads/`](workloads/) | Dataset revisions, file hashes, and prompt specifications |
| [`configs/`](configs/) | Historical frozen manifests for the reported runs |
| [`results/`](results/) | Analysis outputs, selection timings, and compressed candidate pools |
| [`docs/results/`](docs/results/) | Calibration and prompt audits read by the experiment scripts |
| [`tests/`](tests/) | CPU unit tests and hardware-dependent GPU tests |

</details>

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the implementation map, experiment configurations, result files, and hardware setup. The source layout is preserved from the experimental code; the Python package is `specfit`.

## License

SpecFit's original code, documentation, and experimental artifacts are released under [Apache-2.0](LICENSE). Dependencies, downloaded datasets, model weights, and container images retain their own licenses and terms; see [THIRD_PARTY.md](THIRD_PARTY.md).