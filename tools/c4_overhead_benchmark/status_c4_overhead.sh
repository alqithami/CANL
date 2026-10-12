#!/usr/bin/env bash
set -u
SESSION="caenl-c4-overhead-v2"
OUT="/mnt/caenl/active/results/caenl-c4-overhead-v2"
LOG="/mnt/caenl/persistent/logs/caenl-c4-overhead-v2.log"

echo "=== TMUX ==="
tmux list-sessions 2>&1 | grep -F "$SESSION" || true
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv
echo
echo "=== COMPLETED TIMED RUNS ==="
if [ -f "$OUT/raw_runs.csv" ]; then
  awk 'END{print NR-1}' "$OUT/raw_runs.csv"
else
  echo 0
fi
echo
echo "=== MARKERS ==="
ls -l "$OUT"/RUNNING "$OUT"/COMPLETE "$OUT"/FAILED 2>/dev/null || true
echo
echo "=== LOG TAIL ==="
tail -n 50 "$LOG" 2>/dev/null || true
