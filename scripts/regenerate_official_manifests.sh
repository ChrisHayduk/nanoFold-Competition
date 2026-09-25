#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/regenerate_official_manifests.sh --help

The published manifests and fingerprints are frozen. CATH-aware sampling is a
new dataset policy and cannot reproduce or automatically replace that contract.
This wrapper refuses regeneration before modifying any file.

Build a full-universe group map with scripts/build_homology_groups.py, then
generate and independently verify a candidate in a fresh private directory:

  python scripts/stage_dataset_candidate.py \
    --sampling-policy cath-v3 \
    --stage-dir .nanofold_private/cath-v3-candidate-01 \
    --chain-data-cache data/openproteinset/pdb_data/data_caches/chain_data_cache.json \
    --metadata-sources-dir data/metadata_sources \
    --metadata-source-lock data/metadata_sources/structure_metadata_sources.lock.json \
    --cluster-tsv .nanofold_private/original-clusters.tsv \
    --split-group-tsv .nanofold_private/cath-v3-groups-01/split-groups.tsv \
    --split-group-receipt .nanofold_private/cath-v3-groups-01/split-groups.receipt.json

Metadata sources must already be downloaded and pinned; hidden generation needs
NANOFOLD_HIDDEN_SPLIT_SALT. Use a new staging directory for each candidate.
The stage command defaults to cath-v3 and requires both group map and receipt.
Explicit --sampling-policy cath-v2 is available without the group-map options.
Success requires independent manifest replay, full-group coverage for cath-v3,
and both complete cache-target sequence audits. Review the quality report, then
audit processed tensors and residual MSA homology before a dataset release.
See docs/DATA_SAMPLING.md for the complete graph-building and staging workflow.
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
