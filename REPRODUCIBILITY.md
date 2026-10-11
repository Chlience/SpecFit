# Reproducing the SpecFit experiments

This repository contains the experimental source code, frozen run configurations,
measurement scripts, and analysis outputs. The source and experimental artifacts
were extracted from the anonymized supplementary material; paper section numbers
below identify the original experiment groups.

```
src/        Python package
scripts/    workload preparation, measurement, and analysis entry points
tests/      CPU unit tests and hardware-dependent GPU tests
docker/     pinned measurement environment and model revisions
workloads/  pinned dataset identities
configs/    historical frozen run manifests
results/    analysis outputs, grouped by experiment
docs/results/  calibration and prompt audits used by the entry points
```

The historical experimental artifacts retain anonymized user names, host names,
host paths, GPU UUIDs, and source revision IDs from the supplementary material.
Revision IDs that appear in the code and the result files are
consistent pseudonyms: equal ids refer to the same source revision, but they
do not correspond to any public repository. Host paths appear as
`/path/to/data`, `/path/to/model_zoo` and `/path/to/packed-cache`, the A100
machine as `a100-host` and the RTX 4090 machine as `rtx4090-host`.

## Code

`src/specfit/` is the implementation. The files most relevant to the paper:

| Component | Files |
|---|---|
| Expert-offloaded Qwen3 MoE target with the GPU expert cache | `runtime/qwen3_offloaded_model.py`, `runtime/qwen3_offloaded_experts.py`, `runtime/gpu_expert_cache.py`, `runtime/registered_mmap.py` |
| EAGLE-3 drafting and tree verification | `runtime/eagle3_tree.py`, `runtime/eagle3_chain.py`, `runtime/eagle3_attention.py`, `runtime/eagle3_selection.py` |
| SpecFit objective, cost model and exact subtree search | `selection_methods.py`, `selection_refinement.py`, `specfit_exact.py`, `serial_replay.py` |
| Comparison protocol (AR, EAGLE-3, SpecFit; with and without expert cache; adaptive drafting depth; slot capacity; simulated route accuracy) | `main_comparison.py`, `main_comparison_runtime.py` |
| Acceptance and timing statistics | `acceptance_metrics.py` |

`scripts/` holds the entry points. The ones used for the paper:

| Step | Script |
|---|---|
| Download and verify the pinned workloads | `prepare_gsm8k_workload.py`, `prepare_evaluation_workloads.py` (CNN/DailyMail from Spec-Bench) |
| Build the packed expert checkpoint used for offloading | `pack_qwen3_offload_checkpoint.py`, `verify_qwen3_packed_checkpoint.py` |
| Cost-model calibration (natural and cold cache fits, Appendix A) | `benchmark_qwen3_selection_refinement.py` |
| Freeze a run, run one worker, validate a run | `run_specfit_main_comparison.py freeze / run / validate` |
| Host controllers that launch the Docker workers | `control_specfit_main_comparison.py` (GSM8K, AR and EAGLE-3 with cache), `control_specfit_slot_capacity.py` (GSM8K, AR and EAGLE-3 without cache), `control_specfit_summarization.py --adaptive` (SpecFit runs, summarization runs, slot sweep, route-accuracy stages), `control_specfit_device.py` (RTX 4090 calibration and runs) |
| Analysis of sealed runs | `analyze_specfit_main_comparison.py`, `analyze_specfit_slot_capacity.py`, `analyze_specfit_summarization.py`, `analyze_specfit_route_accuracy.py`, `analyze_specfit_device.py` |
| Docker launcher | `docker/run-a100.sh` |

`docs/results/` contains the inputs that the scripts read: the A100
calibration (`selection-refinement-*.json`; SpecFit with the expert cache uses
`by_mode.natural.legacy`, SpecFit without it uses `by_mode.cold.legacy`), the
prompt-length audits of the workloads, and the GSM8K run that the slot
analyzer uses as its 48-slot reference.

The tree contains only the entry points above and the modules they import.
Some shared modules still carry options for configurations that the paper does
not report (for example fixed drafting depth or a revised cost model); the frozen configurations in
`configs/` select the settings used in the paper.

### Environment

- One NVIDIA A100 PCIe 40GB (main results) or one RTX 4090 24GB (Section 4.5), with the experts held in pinned host memory.
- Docker image built from `docker/Dockerfile` (NGC PyTorch base pinned in `docker/base-image.lock.json`, extra packages in `docker/requirements.overlay.lock`).
- Target model `Qwen/Qwen3-30B-A3B-Instruct-2507` and drafter `RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3`, at the revisions pinned in `docker/models.lock.json`.
- Workloads pinned in `workloads/*.lock.json` (GSM8K from Spec-Bench; CNN/DailyMail summarization from `RedHatAI/speculator_benchmarks`).

The containers mount the model root at `/models`, the packed expert checkpoint
at `/packed-cache` and a data directory at `/data` (see `scripts/docker/run-a100.sh`,
configured with `SPECFIT_MODEL_ROOT_HOST`, `SPECFIT_PACKED_CACHE_ROOT_HOST`, `SPECFIT_DATA_ROOT` and `SPECFIT_GPU`).
The controllers check the GPU model and UUID before they start; set the UUID
constants in `src/specfit/main_comparison.py` and the controllers to the local GPU.

### Preparing a new machine

The measurement scripts preserve the experiment's machine-specific checks.
Configure the following before running on a new host. Changing these settings
creates a new reproduction run; leave the historical `configs/` and `results/`
files unchanged.

| Setting | Files to configure |
| --- | --- |
| Actual Docker image ID from `docker image inspect specfit:ngc25.11-selection --format '{{.Id}}'` | `R6A_TB2_IMAGE_ID` in `src/specfit/r6a_tb2_contract.py`; `IMAGE` in `scripts/control_specfit_main_comparison.py` |
| A100 UUIDs from `nvidia-smi --query-gpu=index,name,uuid --format=csv` | `GPU_UUIDS` in `src/specfit/main_comparison.py`; `UUIDS` in `scripts/control_specfit_main_comparison.py` |
| RTX 4090 UUID and device index | `DEVICES` in `src/specfit/main_comparison.py` and `scripts/control_specfit_device.py`; `GPU` in the device controller |
| Host data, model, and packed-checkpoint paths | `DATA`, `PACKED`, and model mount in `scripts/control_specfit_main_comparison.py`; `DATA` and mounts in the device controller |
| Prepared workload directory | `PREP` in `scripts/control_specfit_main_comparison.py`, `scripts/control_specfit_slot_capacity.py`, and `scripts/control_specfit_summarization.py` |
| CPU/NUMA affinity matching the local topology | `docker()` in the main controller; freeze affinity in the summarization controller; `WORKER_CPUS`, `FREEZE_CPUS`, and `NODE` in the device controller; CPU and memory node selection in `scripts/docker/run-a100.sh` |

The `SPECFIT_*` environment variables documented in `docker/README.md` configure
`scripts/docker/run-a100.sh`. The host controllers construct their own Docker
commands and need the settings above; those environment variables do not replace
their constants. A100 measurement manifests also enforce GPU and host-memory
budgets. Use hardware compatible with the selected experiment profile.

Download both models using `huggingface_hub.snapshot_download` with the `repo_id`,
`revision`, and destination `directory` from `docker/models.lock.json` under the
chosen model root. Model files must be available locally before the measurement
containers start.

Inside the measurement image, prepare inputs and a new packed checkpoint:

```bash
python scripts/prepare_gsm8k_workload.py
python scripts/prepare_evaluation_workloads.py --dataset summarization \
    --output-dir /data/artifacts/evaluation-workloads/20260925/prepared
python scripts/pack_qwen3_offload_checkpoint.py \
    --model /models/qwen3/Qwen3-30B-A3B-Instruct-2507 \
    --output /packed-cache/Qwen3-30B-A3B-Instruct-2507/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe/packed-v1 \
    --source-repo-id Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --source-revision 0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe
python scripts/verify_qwen3_packed_checkpoint.py \
    --model /models/qwen3/Qwen3-30B-A3B-Instruct-2507 \
    --packed-cache /packed-cache/Qwen3-30B-A3B-Instruct-2507/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe/packed-v1
```

The packer requires a destination that does not exist. Launch its container with
`SPECFIT_PACKED_CACHE_MODE=rw`; return to the default read-only mount for
measurements. Align each controller's `PREP` with the preparation output. The
historical main comparison also supports HumanEval and AIME25 and requires
`--dataset all` at its own configured preparation path.

Run `python scripts/docker/validate_models.py` in the measurement image to check
the target and drafter configuration; add `--gpu` for the synthetic drafting
smoke test. Commit the machine-specific settings, ensure `git status --porcelain`
is empty, and freeze a new run with that commit. Generated data and traces belong
in the mounted data directory outside the checkout.

Unit tests (CPU only, inside the measurement image):

```
PYTHONPATH=src python -m pytest tests/unit -q
```

The measurement entry points record the source revision and require a clean,
committed checkout of this repository. Commit machine-specific changes before
freezing a new run. Runs must be frozen again, since the revision IDs in
`configs/` are pseudonyms and identify the historical measurements.
For a separate CPU test environment, see the root [README.md](README.md).

### Running a comparison

A run is first frozen into a manifest (methods, questions, expert slots,
cost-model coefficients, drafting configuration), then measured, then
analyzed. Every measured request is checked against an untimed reference
run of the same configuration, which also provides the routes that SpecFit
reads. After preparing the machine and freezing configuration identities, a fresh
adaptive summarization run on 40 articles uses:

```
python3 scripts/control_specfit_summarization.py --adaptive --stages formal \
    --directory /path/to/data/artifacts/summ-reproduction \
    --source "$PWD" --git-commit "$(git rev-parse HEAD)" --gpu 0
python3 scripts/analyze_specfit_summarization.py \
    --directory /path/to/data/artifacts/summ-reproduction/formal-gpu0 \
    --output /tmp/specfit-summarization.json
```

The frozen manifests in `configs/` give the exact settings of every run.

## Frozen run configurations (`configs/`)

| File | Paper use |
|---|---|
| `summ-main.json` | CNN/DailyMail: AR without cache, EAGLE-3 and SpecFit with cache (48 slots) |
| `summ-complete.json` | CNN/DailyMail: AR with cache, EAGLE-3 without cache |
| `summ-specfit-nocache.json` | CNN/DailyMail: SpecFit without cache (cold-cache fit) |
| `summ-slots24.json`, `summ-slots12.json` | CNN/DailyMail slot sweep: EAGLE-3 and SpecFit at 24 and 12 slots |
| `gsm8k-main-ar-eagle-cache.json` | GSM8K: AR and EAGLE-3 with cache |
| `gsm8k-specfit-cache.json` | GSM8K: SpecFit with cache |
| `gsm8k-ar-eagle-nocache.json` | GSM8K: AR and EAGLE-3 without cache |
| `gsm8k-specfit-nocache.json` | GSM8K: SpecFit without cache (cold-cache fit) |
| `rtx4090-*.json` | RTX 4090: calibration and the four comparison runs |
| `route-accuracy-{100,90,80,70,60}.json` | Section 4.6: simulated route predictors |

## Results (`results/`)

All speedups are ratios of total committed tokens over total decode time,
relative to AR without the expert cache on the same questions.

| Paper section | Files |
|---|---|
| 4.2 Summarization (results figure and counter table) | `summarization/main-run.json`, `summarization/complement-run.json`, `summarization/specfit-no-cache.json`, `summarization/prompt-budget.json` |
| 4.3 GSM8K (results figure and counter table) | `gsm8k/specfit-cache.json` (SpecFit with cache, joined with the AR-without-cache and EAGLE-3-with-cache streams), `gsm8k/main-run-ar-eagle-cache.json` (AR and EAGLE-3 with cache; the paper uses worker `0`), `gsm8k/ar-eagle-no-cache.json` (capacity `0`), `gsm8k/specfit-no-cache.json` |
| 4.4 Sensitivity to offload pressure | `summarization/slots24.json`, `summarization/slots12.json`; 48 slots from `summarization/main-run.json`, 0 slots from `summarization/complement-run.json` and `summarization/specfit-no-cache.json` |
| 4.5 Another GPU | `rtx4090/device-4090-all.json` |
| 4.6 Route prediction and selection cost | `route-accuracy/route-accuracy.json`; `selection-cost/selection-time.json` (per-cycle selection time of SpecFit in the GSM8K and summarization runs with the expert cache, written by `selection-cost/selection_time.py`) |
| Appendix A calibration | `cost-model/a100-calibration.json`, `cost-model/a100-calibration.csv`, `cost-model/rtx4090-cost-model.json`, `cost-model/rtx4090-calibration.json` |
| Appendix B selection theory | `theory/theory-checks.json`; `python3 theory/theory_checks.py` recomputes it from `theory/adaptive-pools.json.gz`, the candidate pools that `theory/extract_adaptive.py` extracted from the same runs |

Some result files also contain streams that the paper does not report, such
as SpecFit without the adaptive drafting depth and additional slot counts on
GSM8K. They are records of the same protocol and are left unchanged.

The per-request measurement records (token ids, cycle timings, cache states)
are not included because of their size (about 0.6 GB with the reference
runs). The files above are the verified outputs of the analysis scripts on
those records; the candidate pools and the per-cycle selection times are
included so that Appendix B and the selection cost in Section 4.6 can be
recomputed. The cost of the linear route predictor in Section 4.6 is an
analytic estimate from its size, not a measurement.
