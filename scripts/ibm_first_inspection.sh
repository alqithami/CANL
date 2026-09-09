#!/usr/bin/env bash
# First command to run after SSHing into the IBM Cloud L40S VSI. Read-only.
set -euo pipefail
printf '=== identity ===\n'
id
hostnamectl 2>/dev/null || true
printf '\n=== operating system ===\n'
cat /etc/os-release
printf '\n=== GPU ===\n'
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version,pci.bus_id --format=csv
else
  echo 'nvidia-smi: NOT FOUND'
fi
printf '\n=== CPU and RAM ===\n'
nproc
free -h
printf '\n=== storage ===\n'
lsblk -e7 -o NAME,PATH,MODEL,SERIAL,SIZE,TYPE,FSTYPE,FSVER,MOUNTPOINTS,UUID
printf '\n=== filesystems ===\n'
df -hT
printf '\n=== network ===\n'
ip -br address
