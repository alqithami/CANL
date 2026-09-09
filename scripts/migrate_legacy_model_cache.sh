#!/usr/bin/env bash
# Preserve reusable model weights from the completed v5.3 /dev/shm smoke before cleanup.
set -Eeuo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/scripts/ibm_one_dataset_env.sh" >/dev/null

mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME"

copy_tree() {
  local src="$1" dst="$2"
  [ -d "$src" ] || return 0
  echo "Migrating reusable cache: $src -> $dst"
  mkdir -p "$dst"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --ignore-existing "$src/" "$dst/"
  else
    cp -an "$src/." "$dst/"
  fi
}

copy_tree "/dev/shm/cache/huggingface/hub" "$HUGGINGFACE_HUB_CACHE"
copy_tree "/dev/shm/data/hf" "$HUGGINGFACE_HUB_CACHE"
copy_tree "/dev/shm/cache/torch" "$TORCH_HOME"
copy_tree "/dev/shm/data/torch" "$TORCH_HOME"

printf '\nPersistent model-cache size:\n'
du -sh "$CAENL_PERSISTENT_ROOT/model-cache" 2>/dev/null || true
printf '\nRetained model repositories:\n'
find "$HUGGINGFACE_HUB_CACHE" -maxdepth 1 -type d -name 'models--*' -printf '%f\n' 2>/dev/null | sort || true

echo "MODEL CACHE MIGRATION PASS"
