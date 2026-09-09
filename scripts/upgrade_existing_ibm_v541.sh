#!/usr/bin/env bash
# Reuse the tested IBM virtual environment and install the v5.4.3 confirmatory code.
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V541_PROJECT="${CAENL_V541_PROJECT:-$HOME/caenl-v5.4.1/caenl_revision_pipeline_v5.4.1_ibm_volume}"
V53_PROJECT="${CAENL_V53_PROJECT:-$HOME/caenl-src/caenl_revision_pipeline_v5.3_ibm_rhel_l40s}"

if [ ! -e "$PROJECT_DIR/.venv" ]; then
  if [ -x "$V541_PROJECT/.venv/bin/python" ]; then
    ln -s "$V541_PROJECT/.venv" "$PROJECT_DIR/.venv"
  elif [ -x "$V53_PROJECT/.venv/bin/python" ]; then
    ln -s "$V53_PROJECT/.venv" "$PROJECT_DIR/.venv"
  else
    echo "ERROR: no tested IBM CÆNL virtual environment was found." >&2
    echo "Checked: $V541_PROJECT/.venv" >&2
    echo "Checked: $V53_PROJECT/.venv" >&2
    exit 1
  fi
fi

source "$PROJECT_DIR/.venv/bin/activate"
python -m pip install -e "$PROJECT_DIR" --no-deps

source "$PROJECT_DIR/scripts/ibm_one_dataset_env.sh" >/dev/null

python - <<'PY'
import caenl
import torch

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
  "$PROJECT_DIR/tests/test_vision.py" \
  "$PROJECT_DIR/tests/test_integration_full_macc.py" \
  "$PROJECT_DIR/tests/test_c4_calibration_v542.py"

echo "V5.4.3 UPGRADE PASS"
