# SpecFit

SpecFit selects verification subtrees for speculative decoding of expert-offloaded
mixture-of-experts models. It estimates the committed tokens and verification cost
of each candidate subtree, accounting for expert transfers and the GPU expert
cache, and selects the subtree with the highest predicted throughput. Its adaptive
drafting rule can stop before reaching the maximum drafting depth.

This repository contains the experimental implementation, frozen configurations,
and analysis outputs for Qwen3-30B-A3B-Instruct-2507 with an EAGLE-3 drafter on an
NVIDIA A100 PCIe 40GB and an RTX 4090 24GB.

**Measurement scope:** the reported SpecFit measurements use routes collected in
separate reference runs. Route acquisition is excluded from timing. The route
accuracy experiments use simulated predictions, and the linear predictor cost is
an analytic estimate. These results measure the selection and decoding protocol;
they do not establish online end-to-end performance with a deployed route predictor.

## Repository contents

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

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the implementation map, experiment
configurations, result files, and hardware setup. The source layout is preserved
from the experimental code; the Python package is `specfit`.

## Check the released results without a GPU

The theory replay uses only the Python standard library:

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

The released selection timing file also includes the per-cycle measurements.
Full per-request traces are omitted (about 0.6 GB including reference runs).
The GPU analysis scripts require freshly generated traces to recompute the
remaining aggregate results.

## CPU unit tests

Use Python 3.12. The CPU environment below is for unit tests; GPU measurements use
the separately pinned [Docker environment](docker/README.md).

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-test.txt
PYTHONPATH=src .venv/bin/python -m pytest tests/unit -q
```

GPU tests additionally require the measurement environment, the locked models,
and the packed checkpoint. See [VALIDATION.md](VALIDATION.md) for the checks run
when preparing this release and their coverage.

## Run GPU experiments

Build the measurement image with `bash scripts/docker/build-image.sh`, download
the models at the revisions in `docker/models.lock.json`, prepare the locked
workloads, and pack and verify the target checkpoint. Configure the local image
identity, GPU UUIDs, host paths, and CPU/NUMA affinity as described in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md#preparing-a-new-machine).

The measurement entry point is `scripts/run_specfit_main_comparison.py` with
`freeze`, `run`, and `validate` commands. The host controllers launch these stages
and the analysis scripts validate completed runs. Always freeze a new manifest
from a clean, committed checkout. The historical manifests retain anonymized
revision IDs and hardware identities and cannot be replayed directly on a new
machine.

## License and third-party materials

SpecFit's original code, documentation, and experimental artifacts are released
under [Apache-2.0](LICENSE). Dependencies, externally downloaded datasets, model
weights, and container images retain their own licenses and terms; see
[THIRD_PARTY.md](THIRD_PARTY.md).
