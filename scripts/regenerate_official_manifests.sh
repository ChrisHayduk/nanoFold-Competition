#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/regenerate_official_manifests.sh --help

The published manifests and fingerprints are frozen. CATH-aware sampling is a
new dataset policy and cannot reproduce or automatically replace that contract.
This wrapper refuses regeneration before modifying any file.

Generate and independently verify a candidate in a fresh private directory:

  python scripts/stage_dataset_candidate.py \
    --stage-dir .nanofold_private/cath-v2-candidate-01 \
    --chain-data-cache data/openproteinset/pdb_data/data_caches/chain_data_cache.json \
    --metadata-sources-dir data/metadata_sources \
    --metadata-source-lock data/metadata_sources/structure_metadata_sources.lock.json

Metadata sources must already be downloaded and pinned; hidden generation needs
NANOFOLD_HIDDEN_SPLIT_SALT. Use a new staging directory for each candidate.
The stage command records success only after independent manifest verification.
Review its quality report and audit processed features before a dataset release.
Do not sync these candidate hashes into the published track configuration.

To rebuild and audit assets for the frozen published manifests, use:
  bash scripts/full_official_data_refresh.sh --skip-manifest-regen
EOF
}

if [[ "$#" -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
  usage
  exit 0
fi
usage >&2
exit 1
