#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/env.sh"
[ "$(uname -s)" = Linux ] || { echo 'Run on IBM, not on the Mac.'; exit 1; }
[ -x "$PYTHON" ] || { echo "Existing Python environment is missing: $PYTHON"; exit 1; }
mountpoint -q /mnt/caenl || { echo 'Persistent /mnt/caenl is not mounted; no work started.'; exit 1; }
mkdir -p /mnt/caenl/persistent/{logs,locks}
exec 9>/mnt/caenl/persistent/locks/caenl-aligned-imagenet100-gpu0.lock
if ! flock -n 9; then echo 'This campaign is already active. No second run started.'; exit 2; fi
exec > >(tee -a "$LOG") 2>&1
trap 'rc=$?; echo "ALIGNED_AL_EXIT_CODE=$rc at $(date -Is)"; printf "%s\n" "$rc" > /mnt/caenl/persistent/logs/caenl-imagenet100-aligned-confirmation-v1.exit' EXIT
GPU_PROCS="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)"
if [ -n "$GPU_PROCS" ]; then printf 'Another GPU process is active. Nothing was killed:\n%s\n' "$GPU_PROCS"; exit 2; fi
cd "$BUNDLE"
echo "Starting/resuming frozen aligned active learning at $(date -Is)"
"$PYTHON" -u run_campaign.py --self-test
"$PYTHON" -u run_campaign.py
