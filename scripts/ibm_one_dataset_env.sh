#!/usr/bin/env bash
# Storage-aware environment for one-dataset-at-a-time execution on the IBM L40S VM.
# Prefer the mounted 2 TB volume at /mnt/caenl. Fall back to /dev/shm only when
# that volume is unavailable (e.g., local smoke/development environments).

set -u

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CAENL_PROJECT_DIR="$PROJECT_DIR"

# A user-created override always wins.
[ -f "$HOME/.caenl-storage-env" ] && source "$HOME/.caenl-storage-env"

# Auto-detect the mounted IBM data volume when no explicit roots were supplied.
if [ -z "${CAENL_ACTIVE_ROOT:-}" ] || [ -z "${CAENL_PERSISTENT_ROOT:-}" ]; then
  if command -v findmnt >/dev/null 2>&1 \
      && findmnt -rn /mnt/caenl >/dev/null 2>&1 \
      && [ -w /mnt/caenl ]; then
    export CAENL_ACTIVE_ROOT="${CAENL_ACTIVE_ROOT:-/mnt/caenl/active}"
    export CAENL_PERSISTENT_ROOT="${CAENL_PERSISTENT_ROOT:-/mnt/caenl/persistent}"
  else
    export CAENL_ACTIVE_ROOT="${CAENL_ACTIVE_ROOT:-/dev/shm/caenl-active}"
    export CAENL_PERSISTENT_ROOT="${CAENL_PERSISTENT_ROOT:-$HOME/caenl-persistent}"
  fi
fi

# Derived roots always follow the selected active root.  This deliberately overrides
# stale values inherited from an older /dev/shm session.
export CAENL_DATA_ROOT="$CAENL_ACTIVE_ROOT/data"
export CAENL_CACHE_ROOT="$CAENL_ACTIVE_ROOT/cache"
export CAENL_RESULTS_ROOT="$CAENL_ACTIVE_ROOT/results"
export CAENL_WORK_ROOT="$CAENL_ACTIVE_ROOT/work"

# Keep downloaded model weights and package caches in persistent storage.
export HF_HOME="${HF_HOME:-$CAENL_PERSISTENT_ROOT/model-cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$CAENL_PERSISTENT_ROOT/model-cache/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CAENL_PERSISTENT_ROOT/model-cache/pip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CAENL_PERSISTENT_ROOT/model-cache/xdg}"

# Avoid CPU oversubscription on the 24-vCPU host.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p \
  "$CAENL_DATA_ROOT" \
  "$CAENL_CACHE_ROOT" \
  "$CAENL_RESULTS_ROOT" \
  "$CAENL_WORK_ROOT" \
  "$CAENL_PERSISTENT_ROOT/archives" \
  "$CAENL_PERSISTENT_ROOT/logs" \
  "$CAENL_PERSISTENT_ROOT/model-cache"

[ -f "$HOME/.caenl-media-env" ] && source "$HOME/.caenl-media-env"
[ -f "$HOME/.caenl-java8-env" ] && source "$HOME/.caenl-java8-env"

export PATH="$PROJECT_DIR/.venv/bin:$PATH"

printf '[one-dataset-env] active=%s\n' "$CAENL_ACTIVE_ROOT"
printf '[one-dataset-env] persistent=%s\n' "$CAENL_PERSISTENT_ROOT"
printf '[one-dataset-env] data=%s\n' "$CAENL_DATA_ROOT"
printf '[one-dataset-env] cache=%s\n' "$CAENL_CACHE_ROOT"
printf '[one-dataset-env] results=%s\n' "$CAENL_RESULTS_ROOT"
