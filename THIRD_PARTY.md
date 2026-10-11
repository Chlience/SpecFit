# Third-party sources

The repository's Apache-2.0 license applies to SpecFit's original work. The
dependencies and external resources below retain their upstream licenses and
terms. The lock files record the revisions used in the experiments. Model
weights, dataset downloads, and container images are obtained separately.

## Runtime dependencies

| Resource | Source and license reference | Version record |
| --- | --- | --- |
| PyTorch | [PyTorch license](https://github.com/pytorch/pytorch/blob/v2.10.0/LICENSE), including its third-party notices | `docker/base-packages.lock.json` |
| Transformers | [Apache-2.0 license](https://github.com/huggingface/transformers/blob/v5.6.0/LICENSE) | `docker/requirements.overlay.lock` |
| Speculators | [Upstream repository and Apache-2.0 license](https://github.com/vllm-project/speculators) | `docker/requirements.overlay.lock` |
| NVIDIA NGC PyTorch container | [NGC container page and terms](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) | `docker/base-image.lock.json` |

Additional packages are listed in the Docker lock files and keep their upstream
licenses. SpecFit integrates with these libraries through imports; this table
does not relicense them.

## Models and workloads

| Resource | Upstream source | Revision record |
| --- | --- | --- |
| Qwen3 target | [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | `docker/models.lock.json` |
| EAGLE-3 drafter | [RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3](https://huggingface.co/RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3) | `docker/models.lock.json` |
| Spec-Bench GSM8K and CNN/DailyMail workloads | [RedHatAI/speculator_benchmarks](https://huggingface.co/datasets/RedHatAI/speculator_benchmarks) | `workloads/gsm8k_specbench.lock.json`, `workloads/extended_benchmarks.lock.json` |
| HumanEval, supported by the shared experiment scripts | [openai/human-eval](https://github.com/openai/human-eval) | `workloads/extended_benchmarks.lock.json` |
| AIME25, supported by the shared experiment scripts | [math-ai/aime25](https://huggingface.co/datasets/math-ai/aime25) | `workloads/extended_benchmarks.lock.json` |

Consult the upstream license and dataset/model cards at the locked revisions
before redistributing downloaded resources or packed derivatives. The result
files and candidate pools here are experimental artifacts; the original model
weights and dataset downloads are not included.
