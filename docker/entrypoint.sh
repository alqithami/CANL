#!/usr/bin/env bash
# Records the running image's digest (when the runtime exposes it) and execs the command.
set -euo pipefail
if [[ -n "${CAENL_IMAGE_DIGEST:-}" ]]; then
  echo "$CAENL_IMAGE_DIGEST" > /etc/caenl_image_digest
elif [[ -n "${RUNPOD_POD_IMAGE:-}" ]]; then
  echo "$RUNPOD_POD_IMAGE" > /etc/caenl_image_digest
fi
mkdir -p "${CAENL_DATA_ROOT:-/workspace/data}" "${CAENL_RESULTS_ROOT:-/workspace/results}" "${CAENL_CACHE_ROOT:-/workspace/cache}"
exec "$@"
