#!/usr/bin/env bash
# Idempotent RHEL 9 media/caption-metric runtime for AudioCaps.
set -Eeuo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -x "$PROJECT_DIR/.venv/bin/python" ] || { echo "ERROR: project virtualenv missing" >&2; exit 1; }
source "$PROJECT_DIR/.venv/bin/activate"

sudo dnf -y install curl tar bzip2 java-1.8.0-openjdk-headless

mkdir -p "$HOME/.local" "$HOME/.cache/micromamba" "$HOME/.caenl-tools"
if [ ! -x "$HOME/.local/bin/micromamba" ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "$HOME/.local" bin/micromamba
fi
export MAMBA_ROOT_PREFIX="$HOME/.cache/micromamba"
FFMPEG_PREFIX="$HOME/.caenl-tools/ffmpeg"
if [ ! -x "$FFMPEG_PREFIX/bin/ffmpeg" ] || ! "$FFMPEG_PREFIX/bin/ffmpeg" -version >/dev/null 2>&1; then
  rm -rf "$FFMPEG_PREFIX"
  "$HOME/.local/bin/micromamba" create -y -p "$FFMPEG_PREFIX" -c conda-forge "ffmpeg=7.1.1"
fi

export DENO_INSTALL="$HOME/.deno"
if [ ! -x "$DENO_INSTALL/bin/deno" ]; then
  curl -fsSL https://deno.land/install.sh | sh
fi
python -m pip install --upgrade "yt-dlp[default]==2026.8.19"

JAVA8_BIN="$(rpm -ql java-1.8.0-openjdk-headless | grep -E '/(jre/)?bin/java$' | head -n1)"
[ -x "$JAVA8_BIN" ] || { echo "ERROR: Java 8 executable not found" >&2; exit 1; }
JAVA8_HOME="$(dirname "$(dirname "$JAVA8_BIN")")"

cat > "$HOME/.caenl-media-env" <<EOF
export MAMBA_ROOT_PREFIX="$HOME/.cache/micromamba"
export FFMPEG_PREFIX="$FFMPEG_PREFIX"
export CAENL_FFMPEG_DIR="$FFMPEG_PREFIX/bin"
export DENO_INSTALL="$HOME/.deno"
export PATH="$FFMPEG_PREFIX/bin:$HOME/.deno/bin:$HOME/.local/bin:\$PATH"
export CAENL_YTDLP_ARGS="--ffmpeg-location $FFMPEG_PREFIX/bin --js-runtimes deno:$HOME/.deno/bin/deno"
EOF
chmod 600 "$HOME/.caenl-media-env"

cat > "$HOME/.caenl-java8-env" <<EOF
export CAENL_JAVA8_BIN="$JAVA8_BIN"
export JAVA_HOME="$JAVA8_HOME"
export PATH="$(dirname "$JAVA8_BIN"):\$PATH"
EOF
chmod 600 "$HOME/.caenl-java8-env"

source "$HOME/.caenl-media-env"
source "$HOME/.caenl-java8-env"

SPICE_FILE="$(python - <<'PY'
import pycocoevalcap.spice.spice as spice
print(spice.__file__)
PY
)"
python - "$SPICE_FILE" <<'PY'
from pathlib import Path
import shutil, sys
p=Path(sys.argv[1]); s=p.read_text(encoding='utf-8')
old="spice_cmd = ['java', '-jar', '-Xmx8G', SPICE_JAR,"
new="spice_cmd = ['java', '-Xmx8G', '-jar', SPICE_JAR,"
if old in s:
    backup=p.with_suffix(p.suffix+'.caenl-backup')
    if not backup.exists(): shutil.copy2(p, backup)
    p.write_text(s.replace(old,new,1), encoding='utf-8')
elif new not in s:
    raise SystemExit('ERROR: unrecognised SPICE command; no patch applied')
PY

ffmpeg -hide_banner -version | head -n1
ffprobe -hide_banner -version | head -n1
deno --version | head -n1
java -version 2>&1 | head -n2
python -m py_compile "$SPICE_FILE"
echo "MEDIA RUNTIME PASS"
