#!/usr/bin/env bash
# Bootstrap a fresh RunPod (or any CUDA Linux) pod for the CAENL revision campaign.
#
#   bash scripts/bootstrap_runpod.sh              # install into ./.venv, run the unit tests
#   source .venv/bin/activate
#
# Every step is fatal on error (no "|| true"): a pod that is not completely provisioned must not
# start a paid campaign.  Task groups are installed separately from pinned requirement files so
# that a failure names the group.  After the GPU smoke succeeds, run
#   bash scripts/bootstrap_runpod.sh --lock
# to freeze the exact environment into requirements.lock (and record the container image digest).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${CAENL_DATA_ROOT:=/workspace/data}"
: "${CAENL_RESULTS_ROOT:=/workspace/results}"
: "${CAENL_CACHE_ROOT:=/workspace/cache}"
: "${CAENL_JAVA_METRICS:=1}"          # 0 => skip the Java/pycocoevalcap requirement (then set audio.captioning.java_metrics: false)
: "${CAENL_SKIP_TESTS:=0}"
export CAENL_DATA_ROOT CAENL_RESULTS_ROOT CAENL_CACHE_ROOT
mkdir -p "$CAENL_DATA_ROOT" "$CAENL_RESULTS_ROOT" "$CAENL_CACHE_ROOT"

log() { echo "[bootstrap] $*"; }
die() { echo "[bootstrap] ERROR: $*" >&2; exit 1; }

record_environment() {
  local out="$CAENL_RESULTS_ROOT/environment_$(date -u +%Y%m%d%H%M%S).txt"
  {
    echo "date: $(date -u +%FT%TZ)"
    echo "hostname: $(hostname)"
    echo "python: $(python --version 2>&1)"
    python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'cudnn:', torch.backends.cudnn.version(), 'gpus:', torch.cuda.device_count())"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "nvidia-smi: unavailable"
    echo "image: ${RUNPOD_POD_IMAGE:-${CAENL_IMAGE:-unknown}}"
    echo "image_digest: $(cat /etc/caenl_image_digest 2>/dev/null || echo unknown)"
    echo "java: $(java -version 2>&1 | head -1 || echo none)"
    echo "yt-dlp: $(yt-dlp --version 2>/dev/null || echo none)"
    echo "ffmpeg: $(ffmpeg -version 2>/dev/null | head -1 || echo none)"
    echo "autoattack: $(python -c "from caenl.validate import autoattack_provenance as p; print(p())")"
    echo "--- pip freeze ---"
    pip freeze
  } > "$out"
  log "environment recorded in $out"
}

if [[ "${1:-}" == "--lock" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip freeze --exclude-editable > requirements.lock
  {
    echo "# frozen $(date -u +%FT%TZ) on ${RUNPOD_POD_IMAGE:-${CAENL_IMAGE:-unknown image}}"
    echo "# image digest: $(cat /etc/caenl_image_digest 2>/dev/null || echo unknown)"
    echo "# torch: $(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
    cat requirements.lock
  } > requirements.lock.tmp && mv requirements.lock.tmp requirements.lock
  record_environment
  log "requirements.lock written ($(wc -l < requirements.lock) lines). Commit it with the results."
  exit 0
fi

log "system packages (p7zip for Clotho archives, Java for METEOR/SPICE, libsndfile, ffmpeg for AudioCaps)"
if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y --no-install-recommends p7zip-full default-jre-headless libsndfile1 ffmpeg git curl ca-certificates python3-venv
else
  log "apt-get not found: make sure 7z, java, libsndfile and ffmpeg are installed"
fi

PY=${PYTHON:-python3}
if [[ ! -d .venv ]]; then
  $PY -m venv .venv --system-site-packages
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade "pip>=24.0" wheel

log "PyTorch (CUDA build, >= 2.3)"
torch_ok() { python -c "import torch, sys; v=tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]); sys.exit(0 if (torch.cuda.is_available() and v >= (2, 3)) else 1)" 2>/dev/null; }
if ! torch_ok; then
  log "no suitable CUDA torch found in the image: installing"
  if [[ -n "${CAENL_TORCH_INDEX:-}" ]]; then
    pip install --upgrade torch torchvision --index-url "$CAENL_TORCH_INDEX"
  else
    pip install --upgrade torch torchvision
  fi
fi
torch_ok || die "no CUDA-enabled torch>=2.3 (set CAENL_TORCH_INDEX to the PyTorch wheel index matching this driver, e.g. https://download.pytorch.org/whl/cu124)"

log "pipeline package"
pip install -e .
for group in core robustness language diffusion audio; do
  log "dependency group: $group"
  pip install -r "requirements/$group.txt" || die "installing requirements/$group.txt failed"
done

log "verifying the pinned AutoAttack commit"
python - <<'PY' || exit 1
from caenl.validate import AUTOATTACK_PIN, autoattack_provenance
p = autoattack_provenance()
assert p["installed"], "autoattack did not install"
assert p["commit"] == AUTOATTACK_PIN, f"autoattack commit {p['commit']} != pinned {AUTOATTACK_PIN}"
print("[bootstrap] autoattack", p["commit"])
PY

if [[ "$CAENL_JAVA_METRICS" == "1" ]]; then
  log "checking Java + pycocoevalcap (METEOR/SPICE are required by the protocol)"
  command -v java >/dev/null 2>&1 || die "java not found but Java metrics are enabled (set CAENL_JAVA_METRICS=0 to opt out and disable audio.captioning.java_metrics)"
  python -c "import pycocoevalcap, pycocoevalcap.meteor.meteor, pycocoevalcap.spice.spice" || die "pycocoevalcap not importable"
fi
command -v yt-dlp >/dev/null 2>&1 || log "WARNING: yt-dlp not on PATH (AudioCaps audio cannot be downloaded from this pod)"
command -v ffmpeg >/dev/null 2>&1 || log "WARNING: ffmpeg not on PATH (AudioCaps audio cannot be converted)"

record_environment

if [[ "$CAENL_SKIP_TESTS" != "1" ]]; then
  log "unit tests"
  python -m pytest -q tests --timeout=1800 || die "unit tests failed"
fi

cat <<EOF

[bootstrap] done. Next steps (safe order):
  source .venv/bin/activate
  export CAENL_DATA_ROOT=$CAENL_DATA_ROOT CAENL_RESULTS_ROOT=$CAENL_RESULTS_ROOT CAENL_CACHE_ROOT=$CAENL_CACHE_ROOT
  bash scripts/get_imagenet_kaggle.sh --remove-zip                                      # ImageNet-1K (Kaggle token + competition rules)
  caenl preflight --plan configs/plans/franklin_complete_8gpu.yaml --download           # must print READY (exit 0)
  caenl run --plan configs/plans/smoke_gpu.yaml --parallel 1 --gpus 0                  # real GPU smoke (~1.5-2 h)
  caenl report --root \$CAENL_RESULTS_ROOT/smoke-gpu --strict                          # must exit 0
  bash scripts/bootstrap_runpod.sh --lock                                               # freeze requirements.lock
  nohup caenl run --plan configs/plans/franklin_complete_8gpu.yaml --gpus 0,1,2,3,4,5,6,7 \\
        > \$CAENL_RESULTS_ROOT/complete.log 2>&1 &                                     # (no --parallel: per-stage values apply)
  caenl status --root \$CAENL_RESULTS_ROOT/franklin-complete --details
  # budget alternative (8x A100, ~3.3 days, ~US\$1,000): use configs/plans/franklin_budget_8gpu.yaml in the two commands above
  bash scripts/export_results.sh \$CAENL_RESULTS_ROOT/<campaign>                        # before the pod / VM is deleted
EOF
