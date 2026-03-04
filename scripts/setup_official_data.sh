#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_official_data.sh [options]

Options:
  --data-root <path>          Root for downloaded OpenProteinSet files (default: data/openproteinset)
  --manifests-dir <path>      Target manifests dir to use (default: data/manifests)
  --processed-dir <path>      Output dir for preprocessed .npz files (default: data/processed)
  --msa-name <filename>       MSA filename to download/use (default: uniref90_hits.a3m)
  --template-hhr-name <name>  Template hits filename (default: pdb70_hits.hhr)
  --download-retries <int>    Retries per failed aws chain download (default: 2)
  --download-retry-delay-seconds <float>
                              Base delay for retries in seconds (default: 2.0)
  --disable-templates         Skip template-hit download and template preprocessing
  --skip-preprocess           Do not run preprocess.py
  --force                     Allow overwriting manifest files when copying to --manifests-dir
  --dry-run                   Print commands without executing
  -h, --help                  Show this message
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

PREPARE_DATA_SCRIPT="$SCRIPT_DIR/prepare_data.py"
PREPROCESS_SCRIPT="$SCRIPT_DIR/preprocess.py"

DATA_ROOT="data/openproteinset"
MANIFESTS_DIR="data/manifests"
PROCESSED_DIR="data/processed"
MSA_NAME="uniref90_hits.a3m"
TEMPLATE_HHR_NAME="pdb70_hits.hhr"
DOWNLOAD_RETRIES=2
DOWNLOAD_RETRY_DELAY_SECONDS=2.0
USE_TEMPLATES=1
SKIP_PREPROCESS=0
FORCE=0
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
    --force)
      FORCE=1
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
SOURCE_MANIFESTS_DIR="$REPO_ROOT/data/manifests"
SOURCE_TRAIN="$SOURCE_MANIFESTS_DIR/train.txt"
SOURCE_VAL="$SOURCE_MANIFESTS_DIR/val.txt"
SOURCE_ALL="$SOURCE_MANIFESTS_DIR/all.txt"
SOURCE_MANIFESTS_DIR_ABS="$(python -c 'import pathlib; print(pathlib.Path("'"$SOURCE_MANIFESTS_DIR"'").resolve())')"
MANIFESTS_DIR_ABS="$(python -c 'import pathlib; print(pathlib.Path("'"$MANIFESTS_DIR"'").resolve())')"

if [[ ! -f "$SOURCE_TRAIN" || ! -f "$SOURCE_VAL" ]]; then
  echo "Missing committed official manifests under $SOURCE_MANIFESTS_DIR."
  exit 1
fi

mkdir -p "$DATA_ROOT" "$PDB_DIR" "$DATA_CACHES_DIR" "$MANIFESTS_DIR" "$PROCESSED_DIR"

TARGET_TRAIN="$MANIFESTS_DIR/train.txt"
TARGET_VAL="$MANIFESTS_DIR/val.txt"
TARGET_ALL="$MANIFESTS_DIR/all.txt"

if [[ "$MANIFESTS_DIR_ABS" != "$SOURCE_MANIFESTS_DIR_ABS" ]]; then
  if [[ "$FORCE" -ne 1 && ( -e "$TARGET_TRAIN" || -e "$TARGET_VAL" ) ]]; then
    echo "Refusing to overwrite existing manifests in $MANIFESTS_DIR (pass --force to override)."
    exit 1
  fi
  run_cmd cp "$SOURCE_TRAIN" "$TARGET_TRAIN"
  run_cmd cp "$SOURCE_VAL" "$TARGET_VAL"
  if [[ -f "$SOURCE_ALL" ]]; then
    run_cmd cp "$SOURCE_ALL" "$TARGET_ALL"
  fi
fi

if [[ ! -f "$TARGET_ALL" ]]; then
  if [[ "$DRY_RUN" -eq 0 ]]; then
    cat "$TARGET_TRAIN" "$TARGET_VAL" | awk 'NF {print $0}' | sort -u > "$TARGET_ALL"
  else
    echo "+ cat $TARGET_TRAIN $TARGET_VAL | awk 'NF {print \$0}' | sort -u > $TARGET_ALL"
  fi
fi

echo "[1/5] Downloading OpenFold cache metadata from RODA..."
run_cmd aws s3 cp s3://openfold/data_caches/ "$DATA_CACHES_DIR/" --recursive --no-sign-request
run_cmd aws s3 cp s3://openfold/duplicate_pdb_chains.txt "$PDB_DIR/" --no-sign-request

echo "[2/5] Downloading per-chain MSA + template hits for official manifests..."
PREPARE_CMD=(
  python "$PREPARE_DATA_SCRIPT"
  --data-root "$DATA_ROOT"
  --manifest "$TARGET_ALL"
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

echo "[3/5] Downloading + unpacking mmCIF archive..."
run_cmd aws s3 cp s3://openfold/pdb_mmcif.zip "$PDB_DIR/" --no-sign-request
run_cmd unzip -o "$PDB_DIR/pdb_mmcif.zip" -d "$PDB_DIR"

if [[ "$SKIP_PREPROCESS" -eq 1 ]]; then
  echo "[4/5] Skipping preprocess (--skip-preprocess set)."
else
  echo "[4/5] Preprocessing official train/val manifests..."
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
  run_cmd "${PREPROCESS_COMMON[@]}" --manifest "$TARGET_TRAIN"
  run_cmd "${PREPROCESS_COMMON[@]}" --manifest "$TARGET_VAL"
fi

echo "[5/5] Done."
echo "Official manifest setup complete."
echo "Data root: $DATA_ROOT"
echo "Manifests dir: $MANIFESTS_DIR"
echo "Processed dir: $PROCESSED_DIR"
