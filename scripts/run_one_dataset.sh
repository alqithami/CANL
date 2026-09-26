#!/usr/bin/env bash
# Preflight, run, strict-report and archive one plan. Completed jobs resume automatically.
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/scripts/ibm_one_dataset_env.sh"
source "$PROJECT_DIR/.venv/bin/activate"

# Enforce the one-dataset-at-a-time contract across all tmux/SSH sessions.
command -v flock >/dev/null 2>&1 || { echo "ERROR: flock is required (util-linux)." >&2; exit 1; }
LOCK_FILE="$CAENL_PERSISTENT_ROOT/.one-dataset.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "ERROR: another one-dataset campaign already holds $LOCK_FILE" >&2
  echo "Inspect with: pgrep -af 'run_one_dataset|caenl run'" >&2
  exit 1
fi

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <plan-name-or-path> [--cleanup]" >&2
  exit 2
fi

PLAN_ARG="$1"
CLEANUP=0
[ "${2:-}" = "--cleanup" ] && CLEANUP=1

if [ -f "$PLAN_ARG" ]; then
  PLAN="$(readlink -f "$PLAN_ARG")"
elif [ -f "$PROJECT_DIR/configs/plans/$PLAN_ARG" ]; then
  PLAN="$PROJECT_DIR/configs/plans/$PLAN_ARG"
elif [ -f "$PROJECT_DIR/configs/plans/${PLAN_ARG}.yaml" ]; then
  PLAN="$PROJECT_DIR/configs/plans/${PLAN_ARG}.yaml"
else
  echo "ERROR: plan not found: $PLAN_ARG" >&2
  exit 1
fi

CAMPAIGN_ID="$(python - "$PLAN" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding='utf-8') as f:
    print(yaml.safe_load(f)['campaign_id'])
PY
)"
ROLE="$(python - "$PLAN" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding='utf-8') as f:
    p=yaml.safe_load(f)
print(p.get('campaign_role', 'confirmatory'))
PY
)"

# The audited C4 pilot-v1 exposed a configuration-key defect and an unsuitable
# rank target.  Its original pilot/confirmatory plans remain in the package only
# for provenance and must never be relaunched.
case "$CAMPAIGN_ID" in
  caenl-c4-pilot-v1|caenl-c4-confirmatory-v1|caenl-c4-calibration-v2)
    echo "ERROR: obsolete C4 plan blocked: $CAMPAIGN_ID" >&2
    echo "Use configs/plans/one_c4_confirmatory_v2.yaml." >&2
    exit 1
    ;;
esac
CAENL="$PROJECT_DIR/.venv/bin/caenl"
LOG="$CAENL_PERSISTENT_ROOT/logs/${CAMPAIGN_ID}.log"
ROOT="$CAENL_RESULTS_ROOT/$CAMPAIGN_ID"

# A mounted writable IBM volume must be used for every pilot/confirmatory campaign.
# This prevents stale /dev/shm variables from silently redirecting expensive runs.
if [ "$ROLE" != "smoke" ] && [ "$ROLE" != "engineering" ] \
   && command -v findmnt >/dev/null 2>&1 \
   && findmnt -rn /mnt/caenl >/dev/null 2>&1 \
   && [ -w /mnt/caenl ] \
   && [[ "$CAENL_ACTIVE_ROOT" != /mnt/caenl/* ]]; then
  echo "ERROR: mounted /mnt/caenl is available, but CAENL_ACTIVE_ROOT=$CAENL_ACTIVE_ROOT" >&2
  echo "Run: source \"$HOME/.caenl-storage-env\" and restart the launcher." >&2
  exit 1
fi

trap 'rc=$?; echo "ONE-DATASET RUN FAILED: campaign=$CAMPAIGN_ID exit=$rc time=$(date -Is)" | tee -a "$LOG" >&2; exit $rc' ERR

# ImageNet-1K is intentionally blocked on this 100GB VM: its decoded cache alone is ~252GB.
if grep -Eq 'dataset:[[:space:]]*\[?imagenet1k|name:[[:space:]]*imagenet1k' "$PLAN"; then
  FREE_GB="$(df -Pk "$CAENL_ACTIVE_ROOT" | awk 'NR==2{printf "%.0f", $4/1024/1024}')"
  if [ "$FREE_GB" -lt 320 ]; then
    echo "ERROR: ImageNet-1K plan requires >=320 GB free; current active root has ${FREE_GB} GB." >&2
    echo "Run ImageNet-100/C4/diffusion/AudioCaps now; attach external storage before ImageNet-1K." >&2
    exit 1
  fi
fi

{
  echo "============================================================"
  echo "CÆNL one-dataset run"
  echo "Campaign: $CAMPAIGN_ID"
  echo "Role:     $ROLE"
  echo "Plan:     $PLAN"
  echo "Started:  $(date -Is)"
  echo "============================================================"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
  df -h "$CAENL_ACTIVE_ROOT" "$CAENL_PERSISTENT_ROOT"

  "$CAENL" preflight --plan "$PLAN" --download
  "$CAENL" run --plan "$PLAN" --parallel 1 --gpus 0 --fail-fast
  "$CAENL" report --root "$ROOT" --strict
  "$PROJECT_DIR/scripts/export_campaign_lean.sh" "$CAMPAIGN_ID"

  echo "Completed: $(date -Is)"
} 2>&1 | tee -a "$LOG"

if [ "$CLEANUP" -eq 1 ]; then
  "$PROJECT_DIR/scripts/cleanup_caenl.sh" delete-campaign "$CAMPAIGN_ID" --yes
fi

echo "ONE-DATASET RUN PASS: $CAMPAIGN_ID"
echo "Log: $LOG"
