#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="$HOME/caenl-v5.4.3/caenl_revision_pipeline_v5.4.3_c4_confirmatory"
PYTHON="$PROJECT/.venv/bin/python"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="caenl-c4-overhead-v2"
LOG="/mnt/caenl/persistent/logs/caenl-c4-overhead-v2.log"
OUT="/mnt/caenl/active/results/caenl-c4-overhead-v2"

test -x "$PYTHON" || {
  echo "ERROR: tested v5.4.3 Python was not found: $PYTHON"
  exit 1
}
findmnt /mnt/caenl >/dev/null || {
  echo "ERROR: /mnt/caenl is not mounted"
  exit 1
}
test -w /mnt/caenl || {
  echo "ERROR: /mnt/caenl is not writable"
  exit 1
}

mkdir -p /mnt/caenl/persistent/logs /mnt/caenl/active/results

if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
   | grep -Eq '^[0-9]+'; then
  echo "ERROR: another GPU compute process is active. Do not overlap benchmarks."
  nvidia-smi
  exit 1
fi

tmux has-session -t "$SESSION" 2>/dev/null && {
  echo "ERROR: tmux session already exists: $SESSION"
  exit 1
}

# Preserve any prior completed benchmark rather than silently overwriting it.
if [ -d "$OUT" ]; then
  mv "$OUT" "${OUT}.previous.$(date -u +%Y%m%dT%H%M%SZ)"
fi

tmux new-session -d -s "$SESSION" \
  "cd '$HERE'; \
   '$PYTHON' '$HERE/run_c4_overhead.py' \
     --output-root '$OUT' \
     --persistent-root /mnt/caenl/persistent \
     --repeats 5 \
     --warmup-steps 25 \
     --measure-steps 250 \
     --monitor-cadence 25 \
     2>&1 | tee -a '$LOG'; \
   rc=\${PIPESTATUS[0]}; \
   echo; echo OVERHEAD_EXIT_CODE=\$rc | tee -a '$LOG'; \
   exec bash -i"

echo "STARTED: $SESSION"
echo "Attach:  tmux attach -t $SESSION"
echo "Status:  bash '$HERE/status_c4_overhead.sh'"
echo "Log:     tail -F '$LOG'"
