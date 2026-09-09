#!/usr/bin/env bash
# Read-only host validation for IBM Cloud profile gx3-24x120x1l40s.
set -euo pipefail

MIN_DRIVER="${CAENL_MIN_NVIDIA_DRIVER:-550}"
MIN_GPU_MIB="${CAENL_MIN_GPU_MIB:-45000}"
MIN_RAM_GIB="${CAENL_MIN_RAM_GIB:-108}"
MIN_CPUS="${CAENL_MIN_CPUS:-20}"
MOUNT="${CAENL_MOUNT:-/mnt/caenl}"
REQUIRE_MOUNT="${CAENL_REQUIRE_MOUNT:-0}"

fail=0
ok()   { printf '[host-check] OK   %s\n' "$*"; }
warn() { printf '[host-check] WARN %s\n' "$*" >&2; }
bad()  { printf '[host-check] FAIL %s\n' "$*" >&2; fail=1; }

printf '[host-check] date: %s\n' "$(date -u +%FT%TZ)"
printf '[host-check] host: %s\n' "$(hostname)"
printf '[host-check] os:   %s\n' "$(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-unknown}")"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  bad 'nvidia-smi is missing.'
else
  mapfile -t gpu_rows < <(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits)
  if [[ ${#gpu_rows[@]} -ne 1 ]]; then
    bad "expected exactly one GPU, found ${#gpu_rows[@]}"
  else
    IFS=',' read -r gpu_name gpu_mem gpu_driver <<<"${gpu_rows[0]}"
    gpu_name="$(xargs <<<"$gpu_name")"; gpu_mem="$(xargs <<<"$gpu_mem")"; gpu_driver="$(xargs <<<"$gpu_driver")"
    [[ "$gpu_name" =~ [Ll]40[Ss] ]] && ok "GPU is $gpu_name" || bad "expected NVIDIA L40S, found $gpu_name"
    (( ${gpu_mem%.*} >= MIN_GPU_MIB )) && ok "GPU memory ${gpu_mem} MiB" || bad "GPU memory ${gpu_mem} MiB is below ${MIN_GPU_MIB} MiB"
    driver_major="${gpu_driver%%.*}"
    (( driver_major >= MIN_DRIVER )) && ok "NVIDIA driver $gpu_driver" || bad "NVIDIA driver $gpu_driver is below $MIN_DRIVER"
  fi
fi

cpus="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc)"
(( cpus >= MIN_CPUS )) && ok "$cpus logical CPUs" || bad "$cpus logical CPUs; expected at least $MIN_CPUS"
ram_kib="$(awk '/MemTotal:/ {print $2}' /proc/meminfo)"; ram_gib=$(( ram_kib / 1024 / 1024 ))
(( ram_gib >= MIN_RAM_GIB )) && ok "${ram_gib} GiB host RAM" || bad "${ram_gib} GiB host RAM; expected at least ${MIN_RAM_GIB} GiB"

if [[ -d "$MOUNT" ]] && findmnt -rn "$MOUNT" >/dev/null 2>&1; then
  ok "$MOUNT is mounted"
  [[ -w "$MOUNT" ]] && ok "$MOUNT is writable" || bad "$MOUNT is not writable"
elif [[ "$REQUIRE_MOUNT" == "1" ]]; then
  bad "$MOUNT is not mounted"
else
  warn "$MOUNT is not mounted yet"
fi

printf '\n[host-check] block devices:\n'; lsblk -o NAME,MODEL,SIZE,TYPE,FSTYPE,MOUNTPOINTS,UUID
printf '\n[host-check] filesystem capacity:\n'; df -hT
(( fail )) && exit 1
ok 'host hardware checks passed'
