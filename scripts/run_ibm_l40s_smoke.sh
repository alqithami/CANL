#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CAENL_REQUIRE_MOUNT=1
bash scripts/ibm_verify_host.sh >/dev/null
source scripts/ibm_env.sh
source .venv/bin/activate
PLAN=configs/plans/ibm_l40s_smoke_core.yaml
caenl preflight --plan "$PLAN"
caenl run --plan "$PLAN" --parallel 1 --gpus 0 --fail-fast
caenl report --root "$CAENL_RESULTS_ROOT/ibm-l40s-smoke-core" --strict
printf '[ibm-smoke] PASS: %s\n' "$CAENL_RESULTS_ROOT/ibm-l40s-smoke-core"
