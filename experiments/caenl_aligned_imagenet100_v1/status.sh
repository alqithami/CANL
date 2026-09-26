#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
echo "SERVER: $(hostname) $(date -Is)"
"$PYTHON" - "$ROOT" <<'PY'
from pathlib import Path
import os,json,sys
root=Path(sys.argv[1])
if (root/'COMPLETE.json').exists():
    print('ALIGNED IMAGENET100 COMPLETE');print((root/'COMPLETE.json').read_text())
elif (root/'FAILED.json').exists():
    print('RUN STOPPED WITH AN ERROR — do not launch a duplicate.');print((root/'FAILED.json').read_text())
elif (root/'RUNNING.json').exists():
    rec=json.loads((root/'RUNNING.json').read_text());alive=False
    try:
        os.kill(int(rec['pid']),0);alive=b'run_campaign.py' in Path(f'/proc/{rec["pid"]}/cmdline').read_bytes()
    except (OSError,KeyError,ValueError):pass
    print('RUNNING' if alive else 'NO LIVE RUNNER: last saved status below');print(json.dumps(rec,indent=2))
else:print('Runner not started or still in preliminary self-tests. See log below.')
print('Completed initializations (target 6):',len(list(root.glob('seed*/shared_initial/DONE.json'))))
print('Completed method runs (target 54):',len([p for p in root.glob('seed*/*/DONE.json') if p.parent.name!='shared_initial']))
print('Completed AL training phases (target 270):',len(list(root.glob('seed*/*/rounds/round*/DONE.json'))))
PY
printf '\nGPU:\n'
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv || true
printf '\nLast log lines:\n'
tail -n 25 "$LOG" 2>/dev/null || true
