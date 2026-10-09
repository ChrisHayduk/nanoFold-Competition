#!/usr/bin/env bash
set -euo pipefail

# Default to 16 workers (matching the SLURM allocation CPUs)
WORKERS="${1:-16}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

MANIFEST="data/manifests/train.txt"
SHARD_DIR="data/manifests/.train_shards_${WORKERS}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Error: Manifest $MANIFEST not found."
  exit 1
fi

TOTAL_CHAINS=$(wc -l < "$MANIFEST")
echo "=== Starting Parallel Preprocessing with $WORKERS workers ==="
echo "Total training chains: $TOTAL_CHAINS"

rm -rf "$SHARD_DIR"
mkdir -p "$SHARD_DIR" logs

# Split manifest evenly across WORKERS chunks
split -n "l/$WORKERS" -d "$MANIFEST" "$SHARD_DIR/shard_"

echo "Sharded into $WORKERS parts under $SHARD_DIR"
echo "Launching parallel preprocessing across $WORKERS CPU cores..."

START_TIME=$(date +%s)

find "$SHARD_DIR" -type f -name "shard_*" | sort | xargs -n 1 -P "$WORKERS" -I {} \
  python scripts/preprocess.py \
    --raw-root data/openproteinset \
    --mmcif-root data/openproteinset/pdb_data/mmcif_files \
    --processed-features-dir data/processed_features \
    --processed-labels-dir data/processed_labels \
    --msa-name uniref90_hits.a3m \
    --disable-templates \
    --skip-existing \
    --manifest {}

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Clean up shards
rm -rf "$SHARD_DIR"

TOTAL_FEAT=$(ls -1 data/processed_features/*.npz 2>/dev/null | wc -l)
TOTAL_LBL=$(ls -1 data/processed_labels/*.npz 2>/dev/null | wc -l)

echo "=== Parallel Preprocessing Complete ==="
echo "Time taken: $((DURATION / 60)) minutes $((DURATION % 60)) seconds"
echo "Total processed features in data/processed_features/: $TOTAL_FEAT"
echo "Total processed labels in data/processed_labels/: $TOTAL_LBL"
