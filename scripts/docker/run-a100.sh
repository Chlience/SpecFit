#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

image_name="${SPECFIT_IMAGE:-specfit:ngc25.11-selection}"
gpu_index="${SPECFIT_GPU:-0}"
data_root="${SPECFIT_DATA_ROOT:-/path/to/data}"
model_root_host="${SPECFIT_MODEL_ROOT_HOST:-/path/to/model_zoo}"
packed_cache_root_host="${SPECFIT_PACKED_CACHE_ROOT_HOST:-/path/to/packed-cache}"
packed_cache_mode="${SPECFIT_PACKED_CACHE_MODE:-ro}"

case "${packed_cache_mode}" in
  ro|rw) ;;
  *)
    echo "SPECFIT_PACKED_CACHE_MODE must be ro or rw" >&2
    exit 2
    ;;
esac

if [[ ! -d "${model_root_host}" ]]; then
  echo "model_zoo root does not exist: ${model_root_host}" >&2
  echo "SPECFIT_MODEL_ROOT_HOST must point to the directory above qwen3/" >&2
  exit 2
fi
if [[ ! -d "${model_root_host}/qwen3" ]]; then
  echo "qwen3 model family directory does not exist: ${model_root_host}/qwen3" >&2
  echo "SPECFIT_MODEL_ROOT_HOST must point to the model_zoo root above qwen3/" >&2
  exit 2
fi

if [[ ! -d "${packed_cache_root_host}" ]]; then
  echo "packed cache root does not exist: ${packed_cache_root_host}" >&2
  echo "create the shared directory before running this script" >&2
  exit 2
fi

case "${gpu_index}" in
  0)
    cpu_set="12-23"
    memory_node="1"
    ;;
  1)
    cpu_set="24-35"
    memory_node="2"
    ;;
  *)
    echo "SPECFIT_GPU must be 0 or 1" >&2
    exit 2
    ;;
esac

mkdir -p \
  "${data_root}/home" \
  "${data_root}/hf" \
  "${data_root}/cache/triton" \
  "${data_root}/cache/torch_extensions" \
  "${data_root}/artifacts" \
  "${data_root}/traces"

tty_args=()
if [[ -t 0 && -t 1 ]]; then
  tty_args=(-it)
fi

allocator_args=()
if [[ -n "${PYTORCH_ALLOC_CONF:-}" ]]; then
  allocator_args=(
    --env "PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF}"
  )
fi

exec docker run --rm "${tty_args[@]}" \
  "${allocator_args[@]}" \
  --gpus "device=${gpu_index}" \
  --cpuset-cpus "${cpu_set}" \
  --cpuset-mems "${memory_node}" \
  --shm-size 8g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --env HOME=/data/home \
  --env HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}" \
  --env PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple/}" \
  --env SPECFIT_PACKED_CACHE_ROOT=/packed-cache \
  --env PYTHONPATH=/workspace/specfit/src \
  --volume "${data_root}:/data" \
  --volume "${model_root_host}:/models:ro" \
  --volume "${packed_cache_root_host}:/packed-cache:${packed_cache_mode}" \
  --volume "${repo_root}:/workspace/specfit" \
  --workdir /workspace/specfit \
  "${image_name}" "$@"
