#!/usr/bin/env bash
set -euo pipefail

# End-to-end setup for the competition's subset workflow.
# This avoids downloading the full OpenProteinSet mirror.
#
# What this script does:
#  1) downloads OpenFold data caches (chain_data_cache + mmcif_cache)
#  2) builds fixed train/val manifests
#  3) downloads per-chain MSA + template hits for manifest chains only
#  4) downloads + unzips pdb_mmcif.zip (targets + template structures)
#  5) preprocesses train/val into data/processed/*.npz
#
# Usage:
#   bash scripts/setup_competition_data.sh
#   bash scripts/setup_competition_data.sh --train-size 1000 --val-size 100
#   bash scripts/setup_competition_data.sh --disable-templates

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_competition_data.sh [options]

Options:
  --data-root <path>          Root for downloaded OpenProteinSet files (default: data/openproteinset)
  --manifests-dir <path>      Output dir for train/val manifests (default: data/manifests)
  --processed-dir <path>      Output dir for preprocessed .npz files (default: data/processed)
  --train-size <int>          Number of training chains (default: 10000)
  --val-size <int>            Number of validation chains (default: 500)
  --seed <int>                Manifest sampling seed (default: 0)
  --min-len <int>             Minimum chain length filter (default: 40)
  --max-len <int>             Maximum chain length filter (default: 256)
  --max-resolution <float>    Maximum resolution filter (default: 3.0)
  --msa-name <filename>       MSA filename to download/use (default: uniref90_hits.a3m)
  --template-hhr-name <name>  Template hits filename (default: pdb70_hits.hhr)
  --download-retries <int>    Retries per failed aws chain download (default: 2)
  --download-retry-delay-seconds <float>
                              Base delay for retries in seconds (default: 2.0)
  --disable-templates         Skip template-hit download and template preprocessing
  --skip-preprocess           Do not run preprocess.py
  --dry-run                   Print commands without executing
  -h, --help                  Show this message
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MAKE_MANIFEST_SCRIPT="$SCRIPT_DIR/make_manifest.py"
PREPARE_DATA_SCRIPT="$SCRIPT_DIR/prepare_data.py"
PREPROCESS_SCRIPT="$SCRIPT_DIR/preprocess.py"

DATA_ROOT="data/openproteinset"
MANIFESTS_DIR="data/manifests"
PROCESSED_DIR="data/processed"
TRAIN_SIZE=10000
VAL_SIZE=500
SEED=0
MIN_LEN=40
MAX_LEN=256
MAX_RESOLUTION=3.0
MSA_NAME="uniref90_hits.a3m"
TEMPLATE_HHR_NAME="pdb70_hits.hhr"
DOWNLOAD_RETRIES=2
DOWNLOAD_RETRY_DELAY_SECONDS=2.0
USE_TEMPLATES=1
SKIP_PREPROCESS=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --manifests-dir)
      MANIFESTS_DIR="$2"
      shift 2
      ;;
    --processed-dir)
      PROCESSED_DIR="$2"
      shift 2
      ;;
    --train-size)
      TRAIN_SIZE="$2"
      shift 2
      ;;
    --val-size)
      VAL_SIZE="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --min-len)
      MIN_LEN="$2"
      shift 2
      ;;
    --max-len)
      MAX_LEN="$2"
      shift 2
      ;;
    --max-resolution)
      MAX_RESOLUTION="$2"
      shift 2
      ;;
    --msa-name)
      MSA_NAME="$2"
      shift 2
      ;;
    --template-hhr-name)
      TEMPLATE_HHR_NAME="$2"
      shift 2
      ;;
    --download-retries)
      DOWNLOAD_RETRIES="$2"
      shift 2
      ;;
    --download-retry-delay-seconds)
      DOWNLOAD_RETRY_DELAY_SECONDS="$2"
      shift 2
      ;;
    --disable-templates)
      USE_TEMPLATES=0
      shift 1
      ;;
    --skip-preprocess)
      SKIP_PREPROCESS=1
      shift 1
      ;;
    --dry-run)
      DRY_RUN=1
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

run_cmd() {
  echo "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

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

if [[ "$DRY_RUN" -eq 0 ]] && ! python -c "import tqdm" >/dev/null 2>&1; then
  echo "python package 'tqdm' not found. Run: pip install -r requirements.txt"
  exit 1
fi

PDB_DIR="$DATA_ROOT/pdb_data"
DATA_CACHES_DIR="$PDB_DIR/data_caches"
MMCIF_ROOT="$PDB_DIR/mmcif_files"

mkdir -p "$DATA_ROOT" "$PDB_DIR" "$DATA_CACHES_DIR" "$MANIFESTS_DIR" "$PROCESSED_DIR"

echo "[1/6] Downloading OpenFold data caches from RODA..."
run_cmd aws s3 cp s3://openfold/data_caches/ "$DATA_CACHES_DIR/" --recursive --no-sign-request
run_cmd aws s3 cp s3://openfold/duplicate_pdb_chains.txt "$PDB_DIR/" --no-sign-request

CHAIN_DATA_CACHE_PATH="$DATA_CACHES_DIR/chain_data_cache.json"
if [[ ! -f "$CHAIN_DATA_CACHE_PATH" && -f "$PDB_DIR/chain_data_cache.json" ]]; then
  CHAIN_DATA_CACHE_PATH="$PDB_DIR/chain_data_cache.json"
fi

if [[ "$DRY_RUN" -eq 0 ]] && [[ ! -f "$CHAIN_DATA_CACHE_PATH" ]]; then
  echo "Could not find chain_data_cache.json in either:"
  echo "  - $DATA_CACHES_DIR"
  echo "  - $PDB_DIR"
  exit 1
fi

echo "[2/6] Building fixed manifests..."
run_cmd python "$MAKE_MANIFEST_SCRIPT" \
  --chain-data-cache "$CHAIN_DATA_CACHE_PATH" \
  --out-dir "$MANIFESTS_DIR" \
  --train-size "$TRAIN_SIZE" \
  --val-size "$VAL_SIZE" \
  --min-len "$MIN_LEN" \
  --max-len "$MAX_LEN" \
  --max-resolution "$MAX_RESOLUTION" \
  --seed "$SEED"

ALL_MANIFEST="$MANIFESTS_DIR/all.txt"
echo "[3/6] Building union manifest at $ALL_MANIFEST..."
if [[ "$DRY_RUN" -eq 0 ]]; then
  cat "$MANIFESTS_DIR/train.txt" "$MANIFESTS_DIR/val.txt" | awk 'NF {print $0}' | sort -u > "$ALL_MANIFEST"
else
  echo "+ cat $MANIFESTS_DIR/train.txt $MANIFESTS_DIR/val.txt | awk 'NF {print \$0}' | sort -u > $ALL_MANIFEST"
fi

echo "[4/6] Downloading per-chain MSA + template hits for manifest chains..."
PREPARE_CMD=(
  python "$PREPARE_DATA_SCRIPT"
  --data-root "$DATA_ROOT"
  --manifest "$ALL_MANIFEST"
  --duplicate-chains-file "$PDB_DIR/duplicate_pdb_chains.txt"
  --msa-name "$MSA_NAME"
  --download-retries "$DOWNLOAD_RETRIES"
  --download-retry-delay-seconds "$DOWNLOAD_RETRY_DELAY_SECONDS"
)
if [[ "$USE_TEMPLATES" -eq 1 ]]; then
  PREPARE_CMD+=(--template-hits-name "$TEMPLATE_HHR_NAME")
else
  PREPARE_CMD+=(--no-template-hits)
fi
if [[ "$DRY_RUN" -eq 1 ]]; then
  PREPARE_CMD+=(--dry-run)
fi
run_cmd "${PREPARE_CMD[@]}"

echo "[5/6] Downloading + unpacking mmCIF archive (targets + templates)..."
run_cmd aws s3 cp s3://openfold/pdb_mmcif.zip "$PDB_DIR/" --no-sign-request
run_cmd unzip -o "$PDB_DIR/pdb_mmcif.zip" -d "$PDB_DIR"

if [[ "$SKIP_PREPROCESS" -eq 1 ]]; then
  echo "[6/6] Skipping preprocess (--skip-preprocess set)."
else
  echo "[6/6] Preprocessing train/val to .npz..."
  PREPROCESS_COMMON=(
    python "$PREPROCESS_SCRIPT"
    --raw-root "$DATA_ROOT"
    --mmcif-root "$MMCIF_ROOT"
    --processed-dir "$PROCESSED_DIR"
    --msa-name "$MSA_NAME"
  )
  if [[ "$USE_TEMPLATES" -eq 1 ]]; then
    PREPROCESS_COMMON+=(--template-hhr-name "$TEMPLATE_HHR_NAME")
  else
    PREPROCESS_COMMON+=(--disable-templates)
  fi

  run_cmd "${PREPROCESS_COMMON[@]}" --manifest "$MANIFESTS_DIR/train.txt"
  run_cmd "${PREPROCESS_COMMON[@]}" --manifest "$MANIFESTS_DIR/val.txt"
fi

echo ""
echo "Subset setup complete."
echo "Data root: $DATA_ROOT"
echo "Manifests: $MANIFESTS_DIR"
echo "Processed: $PROCESSED_DIR"
echo ""
echo "Next:"
echo "  python train.py --config submissions/seed_esmfold/config.yaml"
echo "  python train.py --config submissions/seed_openfold/config.yaml"
