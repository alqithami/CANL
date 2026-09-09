#!/usr/bin/env bash
# Export analysis/provenance artefacts from one campaign while omitting large checkpoints/media.
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/scripts/ibm_one_dataset_env.sh" >/dev/null
source "$PROJECT_DIR/.venv/bin/activate"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <campaign-id> [--with-checkpoints]" >&2
  exit 2
fi

CAMPAIGN_ID="$1"
WITH_CHECKPOINTS=0
[ "${2:-}" = "--with-checkpoints" ] && WITH_CHECKPOINTS=1
ROOT="$CAENL_RESULTS_ROOT/$CAMPAIGN_ID"
[ -d "$ROOT" ] || { echo "ERROR: campaign not found: $ROOT" >&2; exit 1; }

CAENL="$PROJECT_DIR/.venv/bin/caenl"
"$CAENL" report --root "$ROOT" --strict

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BASE="${CAMPAIGN_ID}-${STAMP}"
STAGE="$CAENL_PERSISTENT_ROOT/archives/.${BASE}.staging"
ARCHIVE="$CAENL_PERSISTENT_ROOT/archives/${BASE}.tar.gz"
CHECKSUM="${ARCHIVE}.sha256"

rm -rf "$STAGE"
mkdir -p "$STAGE/campaign" "$STAGE/context"

if [ "$WITH_CHECKPOINTS" -eq 1 ]; then
  cp -a "$ROOT/." "$STAGE/campaign/"
else
  (
    cd "$ROOT"
    find . -type f \
      ! -path '*/checkpoints/*' \
      ! -name '*.pt' ! -name '*.pth' ! -name '*.ckpt' ! -name '*.safetensors' \
      ! -name '*.wav' ! -name '*.mp3' ! -name '*.mp4' ! -name '*.webm' \
      -print0 |
    while IFS= read -r -d '' file; do
      dest="$STAGE/campaign/${file#./}"
      mkdir -p "$(dirname "$dest")"
      cp -p "$file" "$dest"
    done
  )
fi

cp -p "$PROJECT_DIR/pyproject.toml" "$STAGE/context/"
cp -p "$PROJECT_DIR/configs/protocol/revision_protocol.yaml" "$STAGE/context/"
[ -f "$ROOT/PLAN.yaml" ] && cp -p "$ROOT/PLAN.yaml" "$STAGE/context/resolved-PLAN.yaml"
[ -f "$ROOT/PLAN.fingerprint" ] && cp -p "$ROOT/PLAN.fingerprint" "$STAGE/context/resolved-PLAN.fingerprint"
[ -f "$ROOT/PLAN.sha256" ] && cp -p "$ROOT/PLAN.sha256" "$STAGE/context/resolved-PLAN.sha256"
[ -f "$ROOT/frozen_protocol.yaml" ] && cp -p "$ROOT/frozen_protocol.yaml" "$STAGE/context/"
[ -f "$ROOT/frozen_protocol.sha256" ] && cp -p "$ROOT/frozen_protocol.sha256" "$STAGE/context/"
python --version > "$STAGE/context/python-version.txt" 2>&1
python -m pip freeze > "$STAGE/context/pip-freeze.txt" 2>&1
nvidia-smi > "$STAGE/context/nvidia-smi.txt" 2>&1 || true
uname -a > "$STAGE/context/uname.txt" 2>&1
cat /etc/os-release > "$STAGE/context/os-release.txt" 2>&1
find "$ROOT" -type f -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS\t%p\n' | sort -nr > "$STAGE/context/full-result-inventory.tsv"

(
  cd "$STAGE"
  find . -type f ! -name 'SHA256SUMS' -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)

tar -C "$(dirname "$STAGE")" -czf "$ARCHIVE" "$(basename "$STAGE")"
(
  cd "$(dirname "$ARCHIVE")"
  sha256sum "$(basename "$ARCHIVE")" > "$(basename "$CHECKSUM")"
)
tar -tzf "$ARCHIVE" >/dev/null
(
  cd "$(dirname "$ARCHIVE")"
  sha256sum -c "$(basename "$CHECKSUM")"
) >/dev/null

rm -rf "$STAGE"
printf '%s\n' "$ARCHIVE" > "$CAENL_PERSISTENT_ROOT/archives/LATEST_${CAMPAIGN_ID}.txt"

echo "EXPORT PASS"
echo "Archive:  $ARCHIVE"
echo "Checksum: $CHECKSUM"
ls -lh "$ARCHIVE" "$CHECKSUM"
