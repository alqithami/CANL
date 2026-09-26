#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/env.sh"
[ "$(uname -s)" = Linux ] || { echo 'Run this in the IBM terminal.'; exit 1; }
[ -x "$PYTHON" ] || { echo "Existing Python environment not found: $PYTHON"; exit 1; }
if [ -s "$ROOT/COMPLETE.json" ]; then
    echo 'ALREADY COMPLETE; no additional experiment started.'
    cat "$ROOT/COMPLETE.json"
    exit 0
fi
if [ -s "$ROOT/RUNNING.json" ] && "$PYTHON" - "$ROOT/RUNNING.json" <<'PY'
import os,json,sys
try:
    p=int(json.load(open(sys.argv[1]))['pid']);os.kill(p,0)
    assert b'run_campaign.py' in open(f'/proc/{p}/cmdline','rb').read()
except (OSError,ValueError,KeyError,AssertionError):sys.exit(1)
PY
then echo 'RUNNING: no duplicate process started.'; exit 0; fi
CMD="bash '$BUNDLE/run.sh'; rc=\$?; echo ALIGNED_AL_LAUNCH_EXIT_CODE=\$rc; exec bash --noprofile --norc -i"
if tmux has-session -t "$SESSION" 2>/dev/null; then
    WINDOW="$(tmux new-window -d -P -F '#{window_id}' -t "$SESSION" -n resume "$CMD")"
    tmux select-window -t "$WINDOW"
else
    tmux new-session -d -s "$SESSION" "$CMD"
fi
printf 'STARTED: %s\nStatus: bash "%s/status.sh"\nLog: %s\n' "$SESSION" "$BUNDLE" "$LOG"
