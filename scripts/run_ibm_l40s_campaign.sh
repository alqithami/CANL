#!/usr/bin/env bash
# Sequential publication campaign for one IBM L40S. Each phase must pass before the next begins.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CAENL_REQUIRE_MOUNT=1
bash scripts/ibm_verify_host.sh >/dev/null

phases=(
  ibm_l40s_imagenet_r50
  ibm_l40s_imagenet_vit
  ibm_l40s_c4
  ibm_l40s_diffusion
  ibm_l40s_audiocaps
  ibm_l40s_clotho
  ibm_l40s_supporting
)

start="${1:-}"
skip=0
[[ -z "$start" ]] && skip=1
for phase in "${phases[@]}"; do
  if (( ! skip )); then
    [[ "$phase" == "$start" ]] && skip=1 || continue
  fi
  printf '\n[campaign] ===== %s =====\n' "$phase"
  bash scripts/run_ibm_l40s_phase.sh "$phase"
done
printf '\n[campaign] Every IBM L40S phase passed. Run scripts/merge_ibm_reports.sh for the consolidated report.\n'
