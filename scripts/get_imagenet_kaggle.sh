#!/usr/bin/env bash
# Fetch ImageNet-1K (ILSVRC2012) from Kaggle's "ImageNet Object Localization Challenge" onto the pod
# and lay it out so that `caenl preflight --download` can build the 256 px cache.
#
#   1. Create a Kaggle account, open https://www.kaggle.com/c/imagenet-object-localization-challenge
#      and click "Join competition" (accept the rules; the data is for non-commercial research only).
#   2. Create an API token (Kaggle -> Settings -> API -> Create new token) and either put kaggle.json
#      in ~/.kaggle/ or export KAGGLE_USERNAME and KAGGLE_KEY.
#   3. bash scripts/get_imagenet_kaggle.sh [--remove-zip]
#
# Needs ~170 GB for the zip plus ~155 GB for the extracted JPEGs (the 256 px cache adds 262 GB later):
# use a volume of >= 700 GB, or --remove-zip to drop the archive after extraction (>= 500 GB).
# Resulting layout (recognised by caenl.vision.data.resolve_imagenet_layout):
#   $CAENL_DATA_ROOT/imagenet/ILSVRC/Data/CLS-LOC/{train,val}   + $CAENL_DATA_ROOT/imagenet/LOC_val_solution.csv
set -euo pipefail
: "${CAENL_DATA_ROOT:=/workspace/data}"
DEST="${CAENL_IMAGENET_DIR:-$CAENL_DATA_ROOT/imagenet}"
ZIP="$DEST/imagenet-object-localization-challenge.zip"
REMOVE_ZIP=0
[[ "${1:-}" == "--remove-zip" ]] && REMOVE_ZIP=1
log() { echo "[imagenet] $*"; }
die() { echo "[imagenet] ERROR: $*" >&2; exit 1; }

mkdir -p "$DEST"
if [[ -d "$DEST/ILSVRC/Data/CLS-LOC/train" && -f "$DEST/LOC_val_solution.csv" ]]; then
  n_cls=$(find "$DEST/ILSVRC/Data/CLS-LOC/train" -mindepth 1 -maxdepth 1 -type d | wc -l)
  if [[ "$n_cls" -eq 1000 ]]; then
    log "ImageNet already present under $DEST (1000 train classes); nothing to do"
    exit 0
  fi
fi

command -v kaggle >/dev/null 2>&1 || { log "installing the kaggle CLI"; pip install "kaggle>=1.6" ; }
if [[ ! -f "$HOME/.kaggle/kaggle.json" && -z "${KAGGLE_USERNAME:-}" ]]; then
  die "no Kaggle credentials: put kaggle.json in ~/.kaggle/ or export KAGGLE_USERNAME/KAGGLE_KEY"
fi
[[ -f "$HOME/.kaggle/kaggle.json" ]] && chmod 600 "$HOME/.kaggle/kaggle.json"

MIN_ZIP_BYTES=150000000000   # the archive is ~168 GB; anything smaller is an interrupted download
free_gb=$(df -BG --output=avail "$DEST" | tail -1 | tr -dc '0-9')

# an interrupted download leaves a truncated archive behind: remove it so that it is fetched again
if [[ -f "$ZIP" ]]; then
  size=$(stat -c %s "$ZIP")
  if [[ "$size" -le "$MIN_ZIP_BYTES" ]]; then
    log "existing archive is only $size bytes (incomplete download); removing it and downloading again"
    rm -f "$ZIP"
  fi
fi

if [[ ! -f "$ZIP" ]]; then
  need_gb=340   # ~170 GB archive + ~155 GB extracted JPEGs
  [[ "$free_gb" -ge "$need_gb" ]] || die "only ${free_gb} GB free under $DEST; need >= ${need_gb} GB for the archive + extracted JPEGs (plus 262 GB for the cache later)"
  log "downloading imagenet-object-localization-challenge.zip (~168 GB) to $DEST — join the competition on kaggle.com first"
  ( cd "$DEST" && kaggle competitions download -c imagenet-object-localization-challenge ) \
    || die "download failed (403 = rules not accepted on the competition page; 401 = bad credentials)"
  size=$(stat -c %s "$ZIP")
  [[ "$size" -gt "$MIN_ZIP_BYTES" ]] || die "archive is only $size bytes after the download; rerun the script to fetch it again"
else
  need_gb=160   # the archive is already here: only the extracted JPEGs still need room
  [[ "$free_gb" -ge "$need_gb" ]] || die "only ${free_gb} GB free under $DEST; need >= ${need_gb} GB for the extracted JPEGs (plus 262 GB for the cache later)"
  log "archive already present ($(( $(stat -c %s "$ZIP") / 1000000000 )) GB); skipping the download"
fi
# a truncated archive has no central directory: listing it (fast, no CRC pass) fails, so the zip is discarded
if command -v 7z >/dev/null 2>&1; then
  7z l "$ZIP" > /dev/null 2>&1 || { log "archive is unreadable (truncated download?); removing it — rerun the script to download it again"; rm -f "$ZIP"; exit 1; }
elif command -v unzip >/dev/null 2>&1; then
  unzip -l "$ZIP" > /dev/null 2>&1 || { log "archive is unreadable (truncated download?); removing it — rerun the script to download it again"; rm -f "$ZIP"; exit 1; }
else
  die "neither 7z nor unzip is installed (apt-get install -y p7zip-full)"
fi

log "extracting (this takes about an hour; existing files are kept, so the step is resumable)"
if command -v 7z >/dev/null 2>&1; then
  7z x -y -aos -o"$DEST" "$ZIP" > /dev/null
else
  unzip -q -n "$ZIP" -d "$DEST"
fi

[[ -f "$DEST/LOC_val_solution.csv" ]] || die "LOC_val_solution.csv missing after extraction"
n_cls=$(find "$DEST/ILSVRC/Data/CLS-LOC/train" -mindepth 1 -maxdepth 1 -type d | wc -l)
n_val=$(find "$DEST/ILSVRC/Data/CLS-LOC/val" -maxdepth 1 -name '*.JPEG' | wc -l)
[[ "$n_cls" -eq 1000 ]] || die "expected 1000 train class folders, found $n_cls"
[[ "$n_val" -eq 50000 ]] || die "expected 50000 validation images, found $n_val"
n_train=$(find "$DEST/ILSVRC/Data/CLS-LOC/train" -name '*.JPEG' | wc -l)
log "train images: $n_train (expected 1281167), val images: $n_val, classes: $n_cls"
[[ "$n_train" -eq 1281167 ]] || die "train image count mismatch"

if [[ "$REMOVE_ZIP" == "1" ]]; then
  rm -f "$ZIP" && log "archive removed"
fi
cat <<EOF
[imagenet] done. ImageNet-1K is under $DEST (Kaggle layout).
[imagenet] next: caenl preflight --plan configs/plans/franklin_complete_8gpu.yaml --download   # builds the 256 px cache (~262 GB, 1-2 h)
EOF
