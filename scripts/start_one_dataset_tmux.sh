#!/usr/bin/env bash
# Start one dataset in a detached tmux session and print exact monitoring commands.
set -Eeuo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/scripts/ibm_one_dataset_env.sh" >/dev/null

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <plan-name-or-path> [--cleanup]" >&2
  exit 2
fi
PLAN_ARG="$1"
CLEAN_ARG="${2:-}"
SAFE="$(printf '%s' "$(basename "$PLAN_ARG" .yaml)" | tr -c 'A-Za-z0-9_-' '-')"
SESSION="caenl-${SAFE}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: tmux session already exists: $SESSION" >&2
  echo "Attach with: tmux attach -t $SESSION" >&2
  exit 1
fi

CMD="cd '$PROJECT_DIR'; bash scripts/run_one_dataset.sh '$PLAN_ARG' '$CLEAN_ARG'; rc=\$?; echo; echo RUN_EXIT_CODE=\$rc; exec bash -i"
tmux new-session -d -s "$SESSION" "$CMD"

echo "STARTED: $SESSION"
echo "Attach:  tmux attach -t $SESSION"
echo "Detach:  Ctrl-b, then d"
echo "Status:  bash scripts/status_one_dataset.sh $PLAN_ARG"
