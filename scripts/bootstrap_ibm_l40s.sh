#!/usr/bin/env bash
# Provision the CÆNL environment on an IBM Cloud L40S VSI.
# Supported host families: RHEL 9-compatible (dnf) and Ubuntu/Debian (apt-get).
# The NVIDIA driver must already be visible through nvidia-smi.
# Any required installation or validation failure is fatal.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CAENL_REQUIRE_MOUNT=1
bash scripts/ibm_verify_host.sh
# shellcheck disable=SC1091
source scripts/ibm_env.sh

log() { printf '[ibm-bootstrap] %s\n' "$*"; }
warn() { printf '[ibm-bootstrap] WARN: %s\n' "$*" >&2; }
die() { printf '[ibm-bootstrap] ERROR: %s\n' "$*" >&2; exit 1; }

install_apt() {
  log 'detected Ubuntu/Debian; installing required system packages'
  sudo apt-get update -y
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev build-essential git curl wget ca-certificates \
    unzip zip jq tmux htop nvme-cli xfsprogs p7zip-full default-jre-headless \
    libsndfile1 ffmpeg rsync aria2
  PYTHON_BIN="${PYTHON:-python3}"
}

install_dnf() {
  log 'detected RHEL/Fedora-compatible host; installing required system packages'

  # Refresh metadata first. IBM RHEL images normally use RHUI repositories.
  sudo dnf -y makecache

  # RHEL 9 provides Python 3.11 in AppStream. Keep the dependency set limited to
  # packages needed by the pipeline itself; optional download utilities are
  # attempted separately below because their repository availability varies.
  sudo dnf -y install \
    python3.11 python3.11-devel python3.11-pip \
    gcc gcc-c++ make git curl wget ca-certificates \
    unzip zip jq tmux nvme-cli xfsprogs \
    java-17-openjdk-headless libsndfile rsync

  # Helpful but not required for the core smoke. Missing optional packages are
  # reported explicitly and task preflight remains responsible for enforcing
  # them when a particular dataset workflow needs them.
  for pkg in htop aria2 p7zip p7zip-plugins ffmpeg-free ffmpeg; do
    if sudo dnf -y install "$pkg" >/dev/null 2>&1; then
      log "installed optional package: $pkg"
    fi
  done

  PYTHON_BIN="${PYTHON:-python3.11}"
}

if command -v apt-get >/dev/null 2>&1; then
  install_apt
elif command -v dnf >/dev/null 2>&1; then
  install_dnf
else
  die 'Unsupported host: neither apt-get nor dnf was found.'
fi

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "$PYTHON_BIN was not installed"

pyver="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
"$PYTHON_BIN" - <<'PY' || die 'Python 3.11 or 3.12 is required'
import sys
assert (3, 11) <= sys.version_info[:2] <= (3, 12), sys.version
PY
log "using Python $pyver from $(command -v "$PYTHON_BIN")"

if [[ ! -d .venv ]]; then
  log 'creating isolated virtual environment (.venv)'
  "$PYTHON_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade 'pip>=25.0' wheel setuptools

log 'installing official PyTorch 2.6.0 CUDA 12.4 wheels'
pip install --upgrade \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

python - <<'PY' || die 'PyTorch/CUDA verification failed'
import torch
assert torch.__version__.split('+')[0] == '2.6.0', torch.__version__
assert torch.version.cuda and torch.version.cuda.startswith('12.4'), torch.version.cuda
assert torch.cuda.is_available(), 'CUDA is not available to PyTorch'
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
name = torch.cuda.get_device_name(0)
assert 'L40S' in name.upper(), name
p = torch.cuda.get_device_properties(0)
assert p.total_memory >= 44 * 1024**3, p.total_memory
x = torch.randn(2048, 2048, device='cuda', dtype=torch.float16)
y = x @ x
assert torch.isfinite(y).all()
print({
    'torch': torch.__version__,
    'torch_cuda': torch.version.cuda,
    'gpu': name,
    'memory_gib': round(p.total_memory / 1024**3, 2),
})
PY

log 'installing CÆNL and pinned dependency groups'
pip install -e . --no-deps
for group in core robustness language diffusion audio; do
  log "requirements/$group.txt"
  pip install -r "requirements/$group.txt"
done

log 'verifying imports and pinned AutoAttack provenance'
python - <<'PY'
from caenl.validate import AUTOATTACK_PIN, autoattack_provenance
mods = [
    'numpy', 'scipy', 'pandas', 'yaml', 'matplotlib', 'torch', 'torchvision',
    'autoattack', 'transformers', 'datasets', 'diffusers', 'torch_fidelity',
    'accelerate', 'soundfile', 'pycocoevalcap',
]
for mod in mods:
    __import__(mod)
p = autoattack_provenance()
assert p['installed'], p
assert p['commit'] == AUTOATTACK_PIN, (p, AUTOATTACK_PIN)
print('AutoAttack:', p)
PY

log 'running unit tests; no publication run starts when tests fail'
python -m compileall -q src tests
python -m pytest -q tests --timeout=1800

stamp="$CAENL_RESULTS_ROOT/environment/ibm-l40s-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$stamp"
{
  echo "date=$(date -u +%FT%TZ)"
  echo "hostname=$(hostname)"
  echo "os=$(source /etc/os-release; echo "$PRETTY_NAME")"
  echo "python=$(python --version 2>&1)"
  python - <<'PY'
import torch
print('torch=' + torch.__version__)
print('torch_cuda=' + str(torch.version.cuda))
print('cudnn=' + str(torch.backends.cudnn.version()))
print('gpu=' + torch.cuda.get_device_name(0))
PY
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
  echo '--- pip freeze ---'
  pip freeze
} > "$stamp/environment.txt"
cp pyproject.toml requirements.txt "$stamp/"
cp -r requirements "$stamp/requirements"

log "bootstrap complete; environment record: $stamp/environment.txt"
cat <<'NEXT'

Next safe command (synthetic, real-GPU smoke; no licensed data required):

  source scripts/ibm_env.sh
  source .venv/bin/activate
  caenl preflight --plan configs/plans/ibm_l40s_smoke_core.yaml
  bash scripts/run_ibm_l40s_smoke.sh

Do not start a publication campaign until the smoke report passes strict validation.
NEXT
