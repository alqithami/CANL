#!/usr/bin/env bash
# Safe cleanup for CÆNL. Destructive actions require --yes and are restricted to known roots.
set -Eeuo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/scripts/ibm_one_dataset_env.sh" >/dev/null

ACTION="${1:-status}"
TARGET="${2:-}"
CONFIRM="${3:-}"
if [ "$TARGET" = "--yes" ] && [ -z "$CONFIRM" ]; then
  CONFIRM="--yes"
  TARGET=""
fi

has_active_processes() {
  pgrep -af 'caenl (run|preflight)|language_c4|diffusion_ddpm|vision_al|audio_captioning|audio_retrieval' >/dev/null 2>&1
}

require_yes() {
  if [ "$CONFIRM" != "--yes" ]; then
    echo "DRY RUN ONLY. Repeat with --yes as the final argument to delete." >&2
    exit 2
  fi
  if has_active_processes; then
    echo "ERROR: a CÆNL process appears to be active; cleanup refused." >&2
    pgrep -af 'caenl (run|preflight)|language_c4|diffusion_ddpm|vision_al|audio_captioning|audio_retrieval' || true
    exit 1
  fi
}

safe_under() {
  local path="$1" root="$2" resolved rroot
  resolved="$(readlink -m "$path")"
  rroot="$(readlink -m "$root")"
  [ "$resolved" != "$rroot" ] || {
    echo "ERROR: refusing to delete an approved root itself: $resolved" >&2
    exit 1
  }
  [[ "$resolved" == "$rroot"/* ]] || {
    echo "ERROR: refusing path outside approved root: $resolved (root $rroot)" >&2
    exit 1
  }
}

require_default_active_root() {
  local resolved
  resolved="$(readlink -m "$CAENL_ACTIVE_ROOT")"
  case "$resolved" in
    /dev/shm/caenl-active|/mnt/caenl/active) ;;
    *)
      echo "ERROR: active-root cleanup is permitted only for /dev/shm/caenl-active or /mnt/caenl/active; got $resolved" >&2
      exit 1
      ;;
  esac
}

latest_archive_for() {
  local campaign="$1"
  find "$CAENL_PERSISTENT_ROOT/archives" -maxdepth 1 -type f -name "${campaign}-*.tar.gz" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-
}

verify_archive_for() {
  local campaign="$1" archive checksum
  archive="$(latest_archive_for "$campaign")"
  [ -n "$archive" ] || { echo "ERROR: no exported archive found for $campaign" >&2; exit 1; }
  checksum="${archive}.sha256"
  [ -f "$checksum" ] || { echo "ERROR: checksum missing: $checksum" >&2; exit 1; }
  (cd "$(dirname "$archive")" && sha256sum -c "$(basename "$checksum")") >/dev/null
  tar -tzf "$archive" >/dev/null
  echo "$archive"
}

case "$ACTION" in
  status)
    echo "=== ACTIVE ROOT ==="
    du -sh "$CAENL_ACTIVE_ROOT" 2>/dev/null || true
    find "$CAENL_RESULTS_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort || true
    echo
    echo "=== PERSISTENT ROOT ==="
    du -sh "$CAENL_PERSISTENT_ROOT" 2>/dev/null || true
    find "$CAENL_PERSISTENT_ROOT/archives" -maxdepth 1 -type f -name '*.tar.gz' -printf '%TY-%Tm-%Td %TH:%TM %10s %f\n' 2>/dev/null | sort || true
    echo
    echo "=== LEGACY/TRANSIENT CÆNL PATHS ==="
    for p in /dev/shm/results/smoke-gpu /dev/shm/results/ibm-l40s-smoke-core /dev/shm/data /dev/shm/cache /dev/shm/work /dev/shm/audiocaps-direct-test; do
      [ -e "$p" ] && du -sh "$p" || true
    done
    find "$HOME" -maxdepth 1 -type f \( -name 'caenl_revision_pipeline_v5.*.zip' -o -name 'caenl-smoke-share-*.tar.gz' \) -printf '%s %p\n' 2>/dev/null | sort -nr || true
    echo
    df -h "$CAENL_ACTIVE_ROOT" "$CAENL_PERSISTENT_ROOT"
    ;;

  delete-campaign)
    [ -n "$TARGET" ] || { echo "Usage: $0 delete-campaign <campaign-id> [--yes]" >&2; exit 2; }
    [[ "$TARGET" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ERROR: invalid campaign id" >&2; exit 1; }
    dir="$CAENL_RESULTS_ROOT/$TARGET"
    safe_under "$dir" "$CAENL_RESULTS_ROOT"
    archive="$(verify_archive_for "$TARGET")"
    echo "Verified archive: $archive"
    echo "Would delete active campaign: $dir"
    require_yes
    rm -rf --one-file-system "$dir"
    echo "DELETED: $dir"
    ;;

  prune-checkpoints)
    [ -n "$TARGET" ] || { echo "Usage: $0 prune-checkpoints <campaign-id> [--yes]" >&2; exit 2; }
    [[ "$TARGET" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ERROR: invalid campaign id" >&2; exit 1; }
    dir="$CAENL_RESULTS_ROOT/$TARGET"
    safe_under "$dir" "$CAENL_RESULTS_ROOT"
    archive="$(verify_archive_for "$TARGET")"
    echo "Verified archive: $archive"
    echo "Checkpoint directories that would be deleted:"
    find "$dir" -type d -name checkpoints -prune -print 2>/dev/null || true
    require_yes
    find "$dir" -type d -name checkpoints -prune -exec rm -rf --one-file-system {} +
    echo "CHECKPOINTS PRUNED: $dir"
    ;;

  delete-data)
    [ -n "$TARGET" ] || { echo "Usage: $0 delete-data <c4|cifar10|cifar100|imagenet100|audiocaps|clotho|diffusion> [--yes]" >&2; exit 2; }
    case "$TARGET" in
      c4) paths=("$CAENL_CACHE_ROOT/language" "$CAENL_DATA_ROOT/c4") ;;
      cifar10) paths=("$CAENL_CACHE_ROOT/vision/cifar10-r32" "$CAENL_DATA_ROOT/cifar10") ;;
      cifar100) paths=("$CAENL_CACHE_ROOT/vision/cifar100-r32" "$CAENL_DATA_ROOT/cifar100") ;;
      imagenet100) paths=("$CAENL_CACHE_ROOT/vision/imagenet100-r144" "$CAENL_DATA_ROOT/imagenet100") ;;
      audiocaps) paths=("$CAENL_DATA_ROOT/audiocaps" "$CAENL_CACHE_ROOT/audio/audiocaps") ;;
      clotho) paths=("$CAENL_DATA_ROOT/clotho" "$CAENL_CACHE_ROOT/audio/clotho") ;;
      diffusion) paths=("$CAENL_CACHE_ROOT/diffusion" "$CAENL_DATA_ROOT/diffusion") ;;
      *) echo "ERROR: unknown dataset allowlist key: $TARGET" >&2; exit 1 ;;
    esac
    echo "Paths that would be deleted:"
    for p in "${paths[@]}"; do safe_under "$p" "$CAENL_ACTIVE_ROOT"; [ -e "$p" ] && du -sh "$p" || true; done
    require_yes
    for p in "${paths[@]}"; do rm -rf --one-file-system "$p"; done
    echo "DATA DELETED: $TARGET"
    ;;

  clean-legacy-smoke)
    echo "Legacy volatile trees that would be deleted:"
    for p in /dev/shm/results/smoke-gpu /dev/shm/results/ibm-l40s-smoke-core /dev/shm/data /dev/shm/cache /dev/shm/work; do
      [ -e "$p" ] && du -sh "$p" || true
    done
    archive="$(find "$HOME" -maxdepth 1 -type f -name 'caenl-smoke-share-*.tar.gz' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
    [ -n "$archive" ] || { echo "ERROR: no caenl-smoke-share archive in $HOME; refusing cleanup" >&2; exit 1; }
    checksum="${archive}.sha256"
    [ -f "$checksum" ] || { echo "ERROR: smoke checksum missing: $checksum" >&2; exit 1; }
    (cd "$HOME" && sha256sum -c "$(basename "$checksum")") >/dev/null
    require_yes
    rm -rf --one-file-system /dev/shm/results/smoke-gpu /dev/shm/results/ibm-l40s-smoke-core /dev/shm/data /dev/shm/cache /dev/shm/work
    echo "LEGACY SMOKE DATA DELETED; archive retained at $archive"
    ;;

  reset-active)
    require_default_active_root
    echo "Would delete the v5.4 active root only: $CAENL_ACTIVE_ROOT"
    [ -e "$CAENL_ACTIVE_ROOT" ] && du -sh "$CAENL_ACTIVE_ROOT" || true
    require_yes
    rm -rf --one-file-system "$CAENL_ACTIVE_ROOT"
    echo "ACTIVE ROOT RESET: $CAENL_ACTIVE_ROOT"
    ;;

  clean-transient)
    paths=(
      "/dev/shm/audiocaps-direct-test"
      "$HOME/audiocaps-direct-test.log"
      "$HOME/audiocaps-ytdlp-debug.log"
      "$HOME/caenl-real-preflight.log"
      "$HOME/caenl-real-preflight-retry.log"
      "$HOME/caenl-spice-self-test.log"
    )
    echo "Transient diagnostic paths that would be deleted:"
    for p in "${paths[@]}"; do [ -e "$p" ] && ls -ld "$p" || true; done
    if [ -f "$HOME/.caenl-private/youtube-cookies.txt" ]; then
      echo "Sensitive server-side browser-cookie copy would also be deleted."
    fi
    require_yes
    for p in "${paths[@]}"; do rm -rf --one-file-system "$p"; done
    rm -f "$HOME/.caenl-private/youtube-cookies.txt"
    echo "TRANSIENT DIAGNOSTICS AND SERVER COOKIE COPY DELETED"
    ;;

  delete-old-zips)
    echo "Old uploaded CÆNL ZIPs in HOME that would be removed (v5.4 excluded):"
    find "$HOME" -maxdepth 1 -type f -name 'caenl_revision_pipeline_v5.[123]*.zip' -print 2>/dev/null || true
    require_yes
    find "$HOME" -maxdepth 1 -type f -name 'caenl_revision_pipeline_v5.[123]*.zip' -delete 2>/dev/null || true
    echo "OLD ZIP FILES DELETED. The v5.3 source directory is retained because it owns the tested virtualenv."
    ;;

  *)
    cat >&2 <<EOF
Usage:
  $0 status
  $0 clean-legacy-smoke [unused] --yes
  $0 delete-campaign <campaign-id> --yes
  $0 prune-checkpoints <campaign-id> --yes
  $0 delete-data <dataset-key> --yes
  $0 reset-active [unused] --yes
  $0 clean-transient [unused] --yes
  $0 delete-old-zips --yes
EOF
    exit 2
    ;;
esac
