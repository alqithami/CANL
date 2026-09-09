#!/usr/bin/env bash
# Merge independently validated IBM L40S campaign roots into the primary ImageNet/ResNet report.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CAENL_REQUIRE_MOUNT=1
bash scripts/ibm_verify_host.sh >/dev/null
source scripts/ibm_env.sh
source .venv/bin/activate

base="$CAENL_RESULTS_ROOT/ibm-l40s-imagenet-r50"
extras=(
  "$CAENL_RESULTS_ROOT/ibm-l40s-imagenet-vit"
  "$CAENL_RESULTS_ROOT/ibm-l40s-c4"
  "$CAENL_RESULTS_ROOT/ibm-l40s-diffusion"
  "$CAENL_RESULTS_ROOT/ibm-l40s-audiocaps"
  "$CAENL_RESULTS_ROOT/ibm-l40s-clotho"
  "$CAENL_RESULTS_ROOT/ibm-l40s-supporting"
)
for p in "$base" "${extras[@]}"; do
  [[ -f "$p/PLAN.yaml" ]] || { echo "Missing completed campaign: $p" >&2; exit 2; }
done
caenl report --root "$base" --extra-roots "${extras[@]}" --strict
printf '[merge] consolidated report: %s/manuscript\n' "$base"
