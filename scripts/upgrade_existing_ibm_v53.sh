#!/usr/bin/env bash
# Reuse the already-tested v5.3 virtualenv and media/Java fixes; install v5.4 editable code.
set -Eeuo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLD_PROJECT="${CAENL_V53_PROJECT:-$HOME/caenl-src/caenl_revision_pipeline_v5.3_ibm_rhel_l40s}"

if [ ! -d "$PROJECT_DIR/.venv" ]; then
  if [ ! -x "$OLD_PROJECT/.venv/bin/python" ]; then
    echo "ERROR: tested v5.3 virtualenv not found at $OLD_PROJECT/.venv" >&2
    exit 1
  fi
  ln -s "$OLD_PROJECT/.venv" "$PROJECT_DIR/.venv"
fi

source "$PROJECT_DIR/.venv/bin/activate"
python -m pip install -e "$PROJECT_DIR" --no-deps

source "$PROJECT_DIR/scripts/ibm_one_dataset_env.sh" >/dev/null

python - <<'PY'
import caenl, torch
print("CÆNL:", caenl.__version__)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
if caenl.__version__ != "5.4.3":
    raise SystemExit("ERROR: v5.4.3 was not installed")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is unavailable")
PY

PYTHONPATH="$PROJECT_DIR/src" pytest -q \
  "$PROJECT_DIR/tests/test_core.py" \
  "$PROJECT_DIR/tests/test_storage_aware.py" \
  "$PROJECT_DIR/tests/test_plan_and_report.py" \
  "$PROJECT_DIR/tests/test_ibm_l40s.py" \
  "$PROJECT_DIR/tests/test_vision.py"

echo "V5.4.3 UPGRADE PASS"
