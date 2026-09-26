#!/usr/bin/env bash
set -euo pipefail

# Download a prespecified snapshot into a new directory and retain all raw sources.
# Usage: bash scripts/setup_openproteinset_roda.sh NEW_DATA_ROOT --source-lock SOURCE_LOCK.json
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python}" "$SCRIPT_DIR/prepare_openproteinset_roda.py" setup "$@"
