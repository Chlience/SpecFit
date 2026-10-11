# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3@sha256:417cbf33f87b5378849df37983552cd1f8bc8b62fe1ceabe004de816a55dff21
FROM ${BASE_IMAGE}

ARG PIP_INDEX_URL=https://pypi.org/simple/
ARG HF_ENDPOINT=https://huggingface.co
ARG APP_UID=1000
ARG APP_GID=1000

LABEL org.opencontainers.image.title="specfit research environment" \
      org.opencontainers.image.base.name="nvcr.io/nvidia/pytorch:25.11-py3" \
      org.opencontainers.image.base.digest="sha256:417cbf33f87b5378849df37983552cd1f8bc8b62fe1ceabe004de816a55dff21"

ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    HF_ENDPOINT=${HF_ENDPOINT} \
    HF_HOME=/data/hf \
    XDG_CACHE_HOME=/data/cache \
    TRITON_CACHE_DIR=/data/cache/triton \
    TORCH_EXTENSIONS_DIR=/data/cache/torch_extensions \
    SPECFIT_MODEL_ROOT=/models \
    SPECFIT_ARTIFACT_ROOT=/data/artifacts \
    SPECFIT_TRACE_ROOT=/data/traces \
    PYTHONPATH=/workspace/specfit/src \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_CUDA_ARCH_LIST=8.0 \
    CUDA_MODULE_LOADING=LAZY

COPY docker/requirements.overlay.lock /opt/specfit/requirements.overlay.lock

RUN PIP_ROOT_USER_ACTION=ignore python -m pip install \
      --disable-pip-version-check \
      --no-cache-dir \
      --no-deps \
      --only-binary=:all: \
      --require-hashes \
      --index-url "${PIP_INDEX_URL}" \
      -r /opt/specfit/requirements.overlay.lock

COPY docker/base-image.lock.json /opt/specfit/base-image.lock.json
COPY docker/base-packages.lock.json /opt/specfit/base-packages.lock.json
COPY docker/models.lock.json /opt/specfit/models.lock.json
COPY scripts/docker/validate_environment.py /opt/specfit/validate_environment.py
COPY scripts/docker/validate_models.py /opt/specfit/validate_models.py

RUN python /opt/specfit/validate_environment.py --cpu

RUN groupadd --gid "${APP_GID}" specfit \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" \
      --create-home --home-dir /home/specfit --shell /bin/bash specfit \
    && mkdir -p /workspace/specfit /data \
    && chown -R "${APP_UID}:${APP_GID}" /workspace/specfit /data

WORKDIR /workspace/specfit
USER specfit
CMD ["/bin/bash"]
