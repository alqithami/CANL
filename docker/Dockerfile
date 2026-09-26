# Reproducible image for the CAENL revision campaign (RunPod-compatible).
#
#   docker build -f docker/Dockerfile -t caenl-revision:5.2 .
#   docker inspect --format='{{index .RepoDigests 0}}' caenl-revision:5.2     # record with the results
#
# Base: the official PyTorch CUDA runtime image.  Pin it by digest before building the campaign
# image (replace the tag below with @sha256:... and keep the value in docs/RUNBOOK.md); the
# digest of the *built* image is written to /etc/caenl_image_digest at container start by
# docker/entrypoint.sh so that bootstrap_runpod.sh and every environment record carry it.
# CUDA 12.4 is the IBM Cloud-supported runtime for L40S instances with the
# documented minimum R550 driver.  PyTorch 2.6.0 publishes official cu124
# wheels/images and is also usable on the A100/H100 RunPod profiles.
ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
LABEL org.opencontainers.image.title="caenl-revision" \
      org.opencontainers.image.description="CAENL Franklin Open revision pipeline (FRAOPE-D-25-02381)" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    CAENL_DATA_ROOT=/workspace/data CAENL_RESULTS_ROOT=/workspace/results CAENL_CACHE_ROOT=/workspace/cache \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false

RUN apt-get update -y && apt-get install -y --no-install-recommends \
        p7zip-full default-jre-headless libsndfile1 ffmpeg git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/caenl
COPY pyproject.toml requirements.txt ./
COPY requirements/ requirements/
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY tests/ tests/
COPY docs/ docs/
COPY README.md ./

RUN python -m pip install --upgrade "pip>=24.0" wheel \
    && pip install -e . \
    && for g in core robustness language diffusion audio; do pip install -r requirements/$g.txt; done \
    && python -c "from caenl.validate import AUTOATTACK_PIN, autoattack_provenance as p; q=p(); assert q['commit']==AUTOATTACK_PIN, q" \
    && python -c "import pycocoevalcap.meteor.meteor, pycocoevalcap.spice.spice, yt_dlp" \
    && pip freeze --exclude-editable > /opt/caenl/requirements.image.lock \
    && echo "${BASE_IMAGE}" > /etc/caenl_base_image

COPY docker/entrypoint.sh /usr/local/bin/caenl-entrypoint
RUN chmod +x /usr/local/bin/caenl-entrypoint
ENTRYPOINT ["/usr/local/bin/caenl-entrypoint"]
CMD ["bash"]
