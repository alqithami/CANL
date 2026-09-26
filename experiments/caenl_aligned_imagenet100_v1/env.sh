#!/usr/bin/env bash
BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$HOME/caenl-v5.4.3/caenl_revision_pipeline_v5.4.3_c4_confirmatory/.venv/bin/python"
ROOT=/mnt/caenl/active/results/caenl-imagenet100-aligned-confirmation-v1
LOG=/mnt/caenl/persistent/logs/caenl-imagenet100-aligned-confirmation-v1.log
SESSION=caenl-aligned-imagenet100-v1
export BUNDLE PYTHON ROOT LOG SESSION
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Do NOT source the old environment scripts: no /dev/shm redirect, no pip install.
