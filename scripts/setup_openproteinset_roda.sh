#!/usr/bin/env bash
set -euo pipefail

# Set up OpenProteinSet/OpenFold training data from RODA using the canonical OpenFold flow.
#
# Usage:
#   bash scripts/setup_openproteinset_roda.sh [data_root]
#
# Example:
#   bash scripts/setup_openproteinset_roda.sh data/openproteinset

if [[ $# -gt 1 ]]; then
  echo "Usage: bash scripts/setup_openproteinset_roda.sh [data_root]"
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${1:-data/openproteinset}"

FLATTEN_SCRIPT="$SCRIPT_DIR/flatten_roda.sh"
EXPAND_SCRIPT="$SCRIPT_DIR/expand_alignment_duplicates.py"

if [[ ! -f "$FLATTEN_SCRIPT" ]]; then
  echo "Missing required script: $FLATTEN_SCRIPT"
  exit 1
fi

if [[ ! -f "$EXPAND_SCRIPT" ]]; then
  echo "Missing required script: $EXPAND_SCRIPT"
  exit 1
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI not found. Install awscli first."
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "unzip not found. Install unzip first."
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python not found. Activate your environment first."
  exit 1
fi

if ! python -c "import tqdm" >/dev/null 2>&1; then
  echo "python package 'tqdm' not found. Run: pip install -r requirements.txt"
  exit 1
fi

ALIGN_RODA_DIR="$DATA_ROOT/alignment_data/alignment_dir_roda"
ALIGN_DIR="$DATA_ROOT/alignment_data/alignments"
PDB_DIR="$DATA_ROOT/pdb_data"
DATA_CACHES_DIR="$PDB_DIR/data_caches"

mkdir -p "$ALIGN_RODA_DIR" "$PDB_DIR" "$DATA_CACHES_DIR"

echo "[1/6] Downloading alignment directories from RODA..."
aws s3 cp s3://openfold/pdb/ "$ALIGN_RODA_DIR/" --recursive --no-sign-request

echo "[2/6] Downloading mmCIF archive + duplicate chains list..."
aws s3 cp s3://openfold/pdb_mmcif.zip "$PDB_DIR/" --no-sign-request
aws s3 cp s3://openfold/duplicate_pdb_chains.txt "$PDB_DIR/" --no-sign-request

echo "[3/6] Unzipping mmCIF files..."
unzip -o "$PDB_DIR/pdb_mmcif.zip" -d "$PDB_DIR"

echo "[4/6] Flattening RODA alignment structure..."
bash "$FLATTEN_SCRIPT" "$ALIGN_RODA_DIR" "$DATA_ROOT/alignment_data"

echo "Removing intermediate unflattened alignment directory..."
rm -rf "$ALIGN_RODA_DIR"

echo "[5/6] Expanding duplicate chains (symlink-based)..."
python "$EXPAND_SCRIPT" \
  "$ALIGN_DIR" \
  "$PDB_DIR/duplicate_pdb_chains.txt"

echo "[6/6] Downloading OpenFold data caches (mmcif_cache + chain_data_cache)..."
aws s3 cp s3://openfold/data_caches/ "$DATA_CACHES_DIR/" --recursive --no-sign-request

echo ""
echo "OpenProteinSet setup complete."
echo "Data root: $DATA_ROOT"
echo ""
echo "Next steps for this benchmark:"
echo "1) Build fixed manifests from chain_data_cache.json (one-time):"
echo "   python scripts/build_manifests.py --chain-data-cache $PDB_DIR/data_caches/chain_data_cache.json --out-dir data/manifests --seed 0"
echo ""
echo "2) Preprocess train split:"
echo "   python scripts/preprocess.py --alignments-root $ALIGN_DIR --mmcif-root $PDB_DIR/mmcif_files --manifest data/manifests/train.txt --processed-features-dir data/processed_features --processed-labels-dir data/processed_labels"
echo ""
echo "3) Preprocess val split:"
echo "   python scripts/preprocess.py --alignments-root $ALIGN_DIR --mmcif-root $PDB_DIR/mmcif_files --manifest data/manifests/val.txt --processed-features-dir data/processed_features --processed-labels-dir data/processed_labels"
