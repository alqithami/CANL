#!/usr/bin/env bash
# Source this file before running CÆNL on the IBM Cloud gx3-24x120x1l40s server.
set -euo pipefail

: "${CAENL_MOUNT:=/mnt/caenl}"
: "${CAENL_REQUIRE_MOUNT:=0}"
if [[ "$CAENL_REQUIRE_MOUNT" == "1" ]] && ! findmnt -rn "$CAENL_MOUNT" >/dev/null 2>&1; then
  printf '[ibm-env] ERROR: %s is not a mounted filesystem. Refusing to use the boot disk.\n' "$CAENL_MOUNT" >&2
  printf '[ibm-env] Attach and mount the IBM Block Storage volume, then retry.\n' >&2
  return 1 2>/dev/null || exit 1
fi

export CAENL_MOUNT
export CAENL_DATA_ROOT="${CAENL_DATA_ROOT:-$CAENL_MOUNT/data}"
export CAENL_RESULTS_ROOT="${CAENL_RESULTS_ROOT:-$CAENL_MOUNT/results}"
export CAENL_CACHE_ROOT="${CAENL_CACHE_ROOT:-$CAENL_MOUNT/cache}"
export CAENL_WORK_ROOT="${CAENL_WORK_ROOT:-$CAENL_MOUNT/work}"

export HF_HOME="${HF_HOME:-$CAENL_CACHE_ROOT/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TORCH_HOME="${TORCH_HOME:-$CAENL_CACHE_ROOT/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CAENL_CACHE_ROOT/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CAENL_CACHE_ROOT/pip}"
export TMPDIR="${TMPDIR:-$CAENL_WORK_ROOT/tmp}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p \
  "$CAENL_DATA_ROOT" "$CAENL_RESULTS_ROOT" "$CAENL_CACHE_ROOT" \
  "$CAENL_WORK_ROOT" "$TMPDIR" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR"

printf '[ibm-env] mount=%s\n' "$CAENL_MOUNT"
printf '[ibm-env] data=%s\n' "$CAENL_DATA_ROOT"
printf '[ibm-env] cache=%s\n' "$CAENL_CACHE_ROOT"
printf '[ibm-env] results=%s\n' "$CAENL_RESULTS_ROOT"
