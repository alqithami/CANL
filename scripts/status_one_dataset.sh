#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/scripts/ibm_one_dataset_env.sh" >/dev/null
source "$PROJECT_DIR/.venv/bin/activate"

PLAN_ARG="${1:?Usage: $0 <plan-name-or-path>}"
if [ -f "$PLAN_ARG" ]; then PLAN="$PLAN_ARG";
elif [ -f "$PROJECT_DIR/configs/plans/$PLAN_ARG" ]; then PLAN="$PROJECT_DIR/configs/plans/$PLAN_ARG";
else PLAN="$PROJECT_DIR/configs/plans/${PLAN_ARG}.yaml"; fi
CAMPAIGN_ID="$(python - "$PLAN" <<'PY'
import sys,yaml
print(yaml.safe_load(open(sys.argv[1]))['campaign_id'])
PY
)"
ROOT="$CAENL_RESULTS_ROOT/$CAMPAIGN_ID"

echo "Campaign: $CAMPAIGN_ID"
echo "Root:     $ROOT"
"$PROJECT_DIR/.venv/bin/caenl" status --root "$ROOT" --details 2>&1 || true
echo
echo "GPU:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv
echo
echo "Processes:"
pgrep -af 'caenl|python' || true
echo
echo "Space:"
df -h "$CAENL_ACTIVE_ROOT" "$CAENL_PERSISTENT_ROOT"
