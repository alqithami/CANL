#!/usr/bin/env bash
set -Eeuo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${CAENL_PROJECT:-$HOME/caenl-v5.4.3/caenl_revision_pipeline_v5.4.3_c4_confirmatory}"
PYTHON="$PROJECT/.venv/bin/python"
SESSION="caenl-c4-holdout-v1"
LOG="/mnt/caenl/persistent/logs/caenl-c4-holdout-v1.log"

if [ ! -x "$PYTHON" ]; then
  echo "ERROR: tested CAENL Python not found: $PYTHON" >&2
  exit 1
fi
if [ ! -d /mnt/caenl/active/results/caenl-c4-confirmatory-v2 ]; then
  echo "ERROR: source confirmatory campaign not found on /mnt/caenl" >&2
  exit 1
fi
mkdir -p /mnt/caenl/persistent/logs
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: tmux session already exists: $SESSION" >&2
  echo "Attach with: tmux attach -t $SESSION" >&2
  exit 1
fi

CMD="cd '$PACKAGE_DIR'; export PYTHONPATH='$PROJECT/src'; '$PYTHON' '$PACKAGE_DIR/run_c4_holdout_eval.py' 2>&1 | tee -a '$LOG'; rc=\${PIPESTATUS[0]}; echo; echo HOLDOUT_EXIT_CODE=\$rc | tee -a '$LOG'; exec bash -i"
tmux new-session -d -s "$SESSION" "$CMD"

echo "STARTED: $SESSION"
echo "Attach:  tmux attach -t $SESSION"
echo "Log:     tail -F $LOG"
echo "Status:  bash $PACKAGE_DIR/status_c4_holdout.sh"
