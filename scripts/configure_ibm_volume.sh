#!/usr/bin/env bash
# Configure the already-mounted IBM block volume for CÆNL. This script never
# formats a device; it only verifies /mnt/caenl and writes user environment settings.
set -Eeuo pipefail

MOUNT="${1:-/mnt/caenl}"

if ! command -v findmnt >/dev/null 2>&1; then
  echo "ERROR: findmnt is unavailable" >&2
  exit 1
fi
if ! findmnt -rn "$MOUNT" >/dev/null 2>&1; then
  echo "ERROR: $MOUNT is not a mounted filesystem" >&2
  exit 1
fi
if [ ! -w "$MOUNT" ]; then
  echo "ERROR: $MOUNT is not writable by $(id -un)" >&2
  exit 1
fi

mkdir -p "$MOUNT/active" "$MOUNT/persistent"
chmod 2775 "$MOUNT/active" "$MOUNT/persistent"

cat > "$HOME/.caenl-storage-env" <<EOF2
export CAENL_ACTIVE_ROOT="$MOUNT/active"
export CAENL_PERSISTENT_ROOT="$MOUNT/persistent"
export CAENL_DATA_ROOT="$MOUNT/active/data"
export CAENL_CACHE_ROOT="$MOUNT/active/cache"
export CAENL_RESULTS_ROOT="$MOUNT/active/results"
export CAENL_WORK_ROOT="$MOUNT/active/work"
EOF2
chmod 600 "$HOME/.caenl-storage-env"

for shellrc in "$HOME/.bashrc" "$HOME/.bash_profile"; do
  touch "$shellrc"
  grep -qF '.caenl-storage-env' "$shellrc" || \
    printf '\n[ -f "$HOME/.caenl-storage-env" ] && source "$HOME/.caenl-storage-env"\n' >> "$shellrc"
done

source "$HOME/.caenl-storage-env"

touch "$CAENL_ACTIVE_ROOT/.write-test"
rm -f "$CAENL_ACTIVE_ROOT/.write-test"

echo "IBM VOLUME CONFIGURATION PASS"
echo "Active root:     $CAENL_ACTIVE_ROOT"
echo "Persistent root: $CAENL_PERSISTENT_ROOT"
findmnt "$MOUNT"
df -hT "$MOUNT"
