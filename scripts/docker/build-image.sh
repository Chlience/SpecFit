#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

image_name="${SPECFIT_IMAGE:-specfit:ngc25.11-selection}"
base_image="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:25.11-py3@sha256:417cbf33f87b5378849df37983552cd1f8bc8b62fe1ceabe004de816a55dff21}"
pip_index_url="${PIP_INDEX_URL:-https://pypi.org/simple/}"

DOCKER_BUILDKIT=1 docker build \
  --platform linux/amd64 \
  --pull=false \
  --progress=plain \
  --build-arg "BASE_IMAGE=${base_image}" \
  --build-arg "PIP_INDEX_URL=${pip_index_url}" \
  --build-arg "APP_UID=$(id -u)" \
  --build-arg "APP_GID=$(id -g)" \
  --file "${repo_root}/docker/Dockerfile" \
  --tag "${image_name}" \
  "${repo_root}"

docker image inspect "${image_name}" \
  --format 'image={{index .RepoTags 0}} id={{.Id}} size={{.Size}} user={{.Config.User}}'
