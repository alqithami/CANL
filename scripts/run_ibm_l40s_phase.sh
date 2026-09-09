#!/usr/bin/env bash
# Run one named production phase with preflight, fail-fast execution, and a strict publication report. Resume is automatic.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CAENL_REQUIRE_MOUNT=1
bash scripts/ibm_verify_host.sh >/dev/null
source scripts/ibm_env.sh
source .venv/bin/activate

name="${1:-}"
[[ -n "$name" ]] || { echo "Usage: $0 PLAN_BASENAME (for example ibm_l40s_imagenet_r50)" >&2; exit 2; }
plan="configs/plans/${name%.yaml}.yaml"
[[ -f "$plan" ]] || { echo "Plan not found: $plan" >&2; exit 2; }

campaign="$(python - "$plan" <<'PY'
import sys,yaml
print(yaml.safe_load(open(sys.argv[1]))['campaign_id'])
PY
)"

caenl preflight --plan "$plan"
caenl run --plan "$plan" --parallel 1 --gpus 0 --fail-fast
caenl report --root "$CAENL_RESULTS_ROOT/$campaign" --strict
printf '[ibm-phase] PASS: %s (%s)\n' "$name" "$CAENL_RESULTS_ROOT/$campaign"
