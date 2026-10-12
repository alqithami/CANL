#!/usr/bin/env bash
set -u
SESSION="caenl-c4-holdout-v1"
OUTPUT="/mnt/caenl/active/results/caenl-c4-confirmatory-v2-holdout-v1"
LOG="/mnt/caenl/persistent/logs/caenl-c4-holdout-v1.log"

echo "=== TMUX ==="
tmux list-sessions 2>&1 | grep -F "$SESSION" || true
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv
echo
echo "=== RESULT COUNT ==="
find "$OUTPUT/per_checkpoint" -type f -name '*.json' 2>/dev/null | wc -l
echo
echo "=== STATUS ==="
cat "$OUTPUT/STATUS.json" 2>/dev/null || true
echo
echo "=== LOG TAIL ==="
tail -n 40 "$LOG" 2>/dev/null || true
