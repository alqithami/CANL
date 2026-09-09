#!/usr/bin/env bash
# Safely mount an IBM Block Storage data volume.
set -euo pipefail

FORMAT=0
if [[ "${1:-}" == "--format" ]]; then FORMAT=1; shift; fi
DEVICE="${1:-}"
MOUNT="${2:-/mnt/caenl}"

usage() {
  cat >&2 <<USAGE
Usage:
  sudo $0 DEVICE [MOUNTPOINT]
  sudo $0 --format DEVICE [MOUNTPOINT]
USAGE
  exit 2
}
[[ -n "$DEVICE" ]] || usage
[[ $EUID -eq 0 ]] || { echo 'Run this script with sudo.' >&2; exit 2; }
[[ -b "$DEVICE" ]] || { echo "$DEVICE is not a block device" >&2; exit 2; }

root_src="$(findmnt -n -o SOURCE /)"
root_parent="$(lsblk -ndo PKNAME "$root_src" 2>/dev/null || true)"
root_disk="$root_src"; [[ -n "$root_parent" ]] && root_disk="/dev/$root_parent"
[[ "$(readlink -f "$DEVICE")" != "$(readlink -f "$root_src")" ]] || { echo 'Refusing to touch the root filesystem device.' >&2; exit 3; }
[[ "$(readlink -f "$DEVICE")" != "$(readlink -f "$root_disk")" ]] || { echo 'Refusing to touch the root disk.' >&2; exit 3; }

if findmnt -rn -S "$DEVICE" >/dev/null 2>&1; then
  echo "$DEVICE is already mounted: $(findmnt -rn -S "$DEVICE")" >&2; exit 3
fi
children="$(lsblk -nrpo NAME,TYPE "$DEVICE" | awk -v d="$(readlink -f "$DEVICE")" '$1 != d {print}')"
[[ -z "$children" ]] || { echo "$DEVICE has child partitions; mount the intended partition explicitly instead:\n$children" >&2; exit 3; }

fstype="$(blkid -s TYPE -o value "$DEVICE" 2>/dev/null || true)"
if [[ -z "$fstype" ]]; then
  if (( ! FORMAT )); then
    echo "$DEVICE has no recognized filesystem. Verify it is the new empty IBM data volume, then rerun with --format." >&2
    exit 4
  fi
  command -v mkfs.xfs >/dev/null 2>&1 || { echo 'mkfs.xfs is unavailable.' >&2; exit 4; }
  echo "About to DESTROY all data on $DEVICE and create an XFS filesystem."
  lsblk -o NAME,MODEL,SIZE,TYPE,FSTYPE,MOUNTPOINTS "$DEVICE"
  read -r -p "Type FORMAT-${DEVICE##*/} to continue: " answer
  [[ "$answer" == "FORMAT-${DEVICE##*/}" ]] || { echo 'Confirmation did not match; aborting.' >&2; exit 4; }
  wipefs -a "$DEVICE"
  mkfs.xfs -f -L CAENL_DATA "$DEVICE"
  fstype=xfs
fi

mkdir -p "$MOUNT"
uuid="$(blkid -s UUID -o value "$DEVICE")"
[[ -n "$uuid" ]] || { echo "Could not read UUID from $DEVICE" >&2; exit 5; }
opts="defaults,nofail,noatime"
entry="UUID=$uuid $MOUNT $fstype $opts 0 2"
if ! grep -qE "^[[:space:]]*UUID=$uuid[[:space:]]" /etc/fstab; then
  cp -a /etc/fstab "/etc/fstab.caenl.$(date -u +%Y%m%d%H%M%S).bak"
  printf '%s\n' "$entry" >> /etc/fstab
fi
mount "$MOUNT"
findmnt "$MOUNT"
mkdir -p "$MOUNT"/{data,cache,results,work,backups}
owner="${SUDO_USER:-root}"
chown -R "$owner":"$(id -gn "$owner")" "$MOUNT"/{data,cache,results,work,backups}
chmod 2775 "$MOUNT" "$MOUNT"/{data,cache,results,work,backups}

echo "Mounted $DEVICE at $MOUNT and persisted it in /etc/fstab."
df -hT "$MOUNT"
