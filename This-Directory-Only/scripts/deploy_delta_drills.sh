#!/usr/bin/env bash
set -euo pipefail

CANONICAL_SCRIPT="/home/stellar-thread/Applications/Delta-Drills-Local/This-Directory-Only/scripts/deploy_delta_drills.sh"

if [ ! -f "$CANONICAL_SCRIPT" ]; then
  echo "deploy_delta_drills: canonical script not found at $CANONICAL_SCRIPT" >&2
  exit 1
fi

exec "$CANONICAL_SCRIPT" "$@"
