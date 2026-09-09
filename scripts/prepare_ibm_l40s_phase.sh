#!/usr/bin/env bash
# Explicitly prepare/download and preflight one IBM L40S production phase.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CAENL_REQUIRE_MOUNT=1
bash scripts/ibm_verify_host.sh >/dev/null
source scripts/ibm_env.sh
source .venv/bin/activate

name="${1:-}"
[[ -n "$name" ]] || { echo "Usage: $0 PLAN_BASENAME (for example ibm_l40s_c4)" >&2; exit 2; }
plan="configs/plans/${name%.yaml}.yaml"
[[ -f "$plan" ]] || { echo "Plan not found: $plan" >&2; exit 2; }

printf '[ibm-prepare] preparing and validating %s\n' "$plan"
caenl preflight --plan "$plan" --download
printf '[ibm-prepare] READY: %s\n' "$name"
