#!/usr/bin/env sh
set -eu

# Copy RODA inputs into new data/alignments directories; retain sources on success or failure.
# Usage: sh scripts/flatten_roda.sh RODA_DIR OUTPUT_DIR
SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)"
exec "${PYTHON:-python}" "$SCRIPT_DIR/prepare_openproteinset_roda.py" flatten "$@"
