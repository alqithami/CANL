#!/usr/bin/env bash
# Pack everything the manuscript and its reproducibility statement need from a campaign root into ONE archive,
# so that the pod / VM can be deleted afterwards.
#
#   bash scripts/export_results.sh <campaign_root> [<archive path>] [--with-logits] [--with-checkpoints]
#
# Included: PLAN.yaml + PLAN.sha256, frozen_protocol.yaml(.sha256), campaign_manifest.json, STATUS.md,
#   PUBLICATION_CHECK.json, every STAGE_MANIFEST.json / RESUME_CHECK.json, aggregate/*.csv, manuscript/
#   (tables, figures, results_summary.md), and per job: job.json, status.json, summary.json, events.jsonl,
#   metrics.jsonl, system.csv, environment.json, log.txt, autoattack_log.txt and artifacts/ (selection
#   indices, controller trajectories, acquisition scores, AutoAttack subsets / robust masks / adversarial
#   predictions, APGD masks).  A few GB for the complete campaign.
# Excluded by default: checkpoints/ (hundreds of GB; --with-checkpoints keeps them) and the per-round test
#   logits (artifacts/test_logits_round*.npy, ~1 GB per ImageNet job; --with-logits keeps them).  Data and
#   memmap caches are never part of a campaign root and are re-creatable from the data sources.
# The archive is compressed with zstd when available (tar.zst), else gzip (tar.gz); its SHA-256 is printed
# and written next to it so that the copy on your machine / Object Storage can be verified.
set -euo pipefail
ROOT="${1:-}"
[[ -n "$ROOT" && -d "$ROOT" ]] || { echo "usage: $0 <campaign_root> [<archive path>] [--with-logits] [--with-checkpoints]" >&2; exit 2; }
shift
OUT=""
WITH_LOGITS=0
WITH_CKPT=0
for a in "$@"; do
  case "$a" in
    --with-logits) WITH_LOGITS=1 ;;
    --with-checkpoints) WITH_CKPT=1 ;;
    *) OUT="$a" ;;
  esac
done
ROOT="$(cd "$ROOT" && pwd)"
NAME="$(basename "$ROOT")"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -n "$OUT" ]]; then
  case "$OUT" in
    *.tar.zst) command -v zstd >/dev/null 2>&1 || { echo "[export] ERROR: $OUT asks for zstd but zstd is not installed (apt-get install -y zstd, or use a .tar.gz name)" >&2; exit 1; }; COMP=(--zstd) ;;
    *.tar.gz|*.tgz) COMP=(--gzip) ;;
    *) echo "[export] ERROR: archive path must end in .tar.zst or .tar.gz" >&2; exit 2 ;;
  esac
else
  if command -v zstd >/dev/null 2>&1; then
    EXT="tar.zst"; COMP=(--zstd)
  else
    EXT="tar.gz"; COMP=(--gzip)
  fi
  OUT="$(dirname "$ROOT")/${NAME}-export-${STAMP}.${EXT}"
fi
log() { echo "[export] $*"; }

if [[ -f "$ROOT/PUBLICATION_CHECK.json" ]]; then
  if python3 - "$ROOT/PUBLICATION_CHECK.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if d.get("ok") else 1)
PY
  then log "PUBLICATION_CHECK.json: ok = true (publication-ready campaign)"
  else log "WARNING: PUBLICATION_CHECK.json says ok = false — exporting a campaign that is NOT publication-ready"
  fi
else
  log "WARNING: no PUBLICATION_CHECK.json under $ROOT (run 'caenl report --root $ROOT --strict' first)"
fi

EXCL=(--exclude='*/checkpoints' --exclude='*/checkpoints/*' --exclude='shared/initial' --exclude='shared/initial/*' --exclude='.runner.lock')
[[ "$WITH_CKPT" == "1" ]] && EXCL=(--exclude='.runner.lock')
[[ "$WITH_LOGITS" == "1" ]] || EXCL+=(--exclude='test_logits_round*.npy')

n_jobs=$(find "$ROOT/stages" -name summary.json 2>/dev/null | wc -l)
log "campaign: $ROOT  (jobs with a summary: $n_jobs)"
log "writing $OUT"
tar "${COMP[@]}" -cf "$OUT" -C "$(dirname "$ROOT")" "${EXCL[@]}" "$NAME"
size=$(du -h "$OUT" | cut -f1)
sha=$(sha256sum "$OUT" | cut -d' ' -f1)
echo "$sha  $(basename "$OUT")" > "$OUT.sha256"
n_files=$(tar -tf "$OUT" | wc -l)
log "done: $OUT ($size, $n_files entries), sha256 $sha (also in $OUT.sha256)"
cat <<EOF
[export] next:
[export]   copy it off the machine, e.g.  scp $OUT you@your-host:~/      (or rclone / ibmcloud cos upload / runpodctl send)
[export]   verify the copy:               sha256sum -c $(basename "$OUT").sha256
[export]   keep the archive with the code ($(basename "$(cd "$(dirname "$0")/.." && pwd)")) — together they reproduce every number in the manuscript.
EOF
