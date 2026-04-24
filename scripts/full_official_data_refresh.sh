#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/full_official_data_refresh.sh [options]

Single maintainer end-to-end flow for official data refresh:
  0) bootstrap chain_data_cache.json if missing
  1) regenerate official train/val/all manifests from locked chain cache inputs
  2) sync official manifest hashes/counts across track + lock + docs
  3) download OpenFold assets and preprocess split NPZs (features + labels)
  4) rebuild official dataset fingerprint

Options:
  --track-id <id>                     Track id metadata for fingerprint (default: limited_large)
  --data-root <path>                  Download root (default: data/openproteinset)
  --manifests-dir <path>              Manifest directory (default: data/manifests)
  --processed-features-dir <path>     Feature NPZ output dir (default: data/processed_features)
  --processed-labels-dir <path>       Label NPZ output dir (default: data/processed_labels)
  --chain-data-cache <path>           chain_data_cache.json path
                                      (default: data/openproteinset/pdb_data/data_caches/chain_data_cache.json)
  --lock-file <path>                  Official manifest lock file
                                      (default: leaderboard/official_manifest_source.lock.json)
  --track-file <path>                 Track policy YAML to update/check
                                      (default: tracks/limited_large.yaml)
  --fingerprint-out <path>            Fingerprint output path
                                      (default: leaderboard/official_dataset_fingerprint.json)
  --rewrite-lock                      Rewrite lock metadata after manifest regeneration
  --skip-manifest-regen               Skip manifest regeneration step
  --skip-setup                        Skip download+preprocess step
  --skip-fingerprint                  Skip fingerprint rebuild step
  --enable-templates                  Pass through to setup_official_data.sh
  --disable-templates                 Pass through to setup_official_data.sh (default)
  --download-retries <int>            Pass through to setup_official_data.sh (default: 2)
  --download-retry-delay-seconds <f>  Pass through to setup_official_data.sh (default: 2.0)
  --dry-run                           Print commands without executing
  -h, --help                          Show this message
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

TRACK_ID="limited_large"
DATA_ROOT="data/openproteinset"
MANIFESTS_DIR="data/manifests"
PROCESSED_FEATURES_DIR="data/processed_features"
PROCESSED_LABELS_DIR="data/processed_labels"
CHAIN_DATA_CACHE="data/openproteinset/pdb_data/data_caches/chain_data_cache.json"
LOCK_FILE="leaderboard/official_manifest_source.lock.json"
TRACK_FILE="tracks/limited_large.yaml"
FINGERPRINT_OUT="leaderboard/official_dataset_fingerprint.json"
DOWNLOAD_RETRIES=2
DOWNLOAD_RETRY_DELAY_SECONDS=2.0
USE_TEMPLATES=0
REWRITE_LOCK=0
SKIP_MANIFEST_REGEN=0
SKIP_SETUP=0
SKIP_FINGERPRINT=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --track-id)
      TRACK_ID="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --manifests-dir)
      MANIFESTS_DIR="$2"
      shift 2
      ;;
    --processed-features-dir)
      PROCESSED_FEATURES_DIR="$2"
      shift 2
      ;;
    --processed-labels-dir)
      PROCESSED_LABELS_DIR="$2"
      shift 2
      ;;
    --chain-data-cache)
      CHAIN_DATA_CACHE="$2"
      shift 2
      ;;
    --lock-file)
      LOCK_FILE="$2"
      shift 2
      ;;
    --track-file)
      TRACK_FILE="$2"
      shift 2
      ;;
    --fingerprint-out)
      FINGERPRINT_OUT="$2"
      shift 2
      ;;
    --rewrite-lock)
      REWRITE_LOCK=1
      shift 1
      ;;
    --skip-manifest-regen)
      SKIP_MANIFEST_REGEN=1
      shift 1
      ;;
    --skip-setup)
      SKIP_SETUP=1
      shift 1
      ;;
    --skip-fingerprint)
      SKIP_FINGERPRINT=1
      shift 1
      ;;
    --disable-templates)
      USE_TEMPLATES=0
      shift 1
      ;;
    --enable-templates)
      USE_TEMPLATES=1
      shift 1
      ;;
    --download-retries)
      DOWNLOAD_RETRIES="$2"
      shift 2
      ;;
    --download-retry-delay-seconds)
      DOWNLOAD_RETRY_DELAY_SECONDS="$2"
      shift 2
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

ensure_chain_data_cache() {
  if [[ "$SKIP_MANIFEST_REGEN" -eq 1 ]]; then
    return
  fi
  if [[ -f "$CHAIN_DATA_CACHE" ]]; then
    return
  fi

  local cache_dir
  cache_dir="$(dirname "$CHAIN_DATA_CACHE")"
  mkdir -p "$cache_dir"
  echo "chain_data_cache.json not found at $CHAIN_DATA_CACHE; bootstrapping from RODA."

  if [[ "$DRY_RUN" -eq 0 ]] && ! command -v aws >/dev/null 2>&1; then
    echo "aws CLI not found. Install awscli first."
    exit 1
  fi

  run_cmd aws s3 cp s3://openfold/data_caches/chain_data_cache.json "$CHAIN_DATA_CACHE" --no-sign-request
}

if ! command -v python >/dev/null 2>&1; then
  echo "python not found. Activate your environment first."
  exit 1
fi

cd "$REPO_ROOT"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY-RUN mode enabled: commands will be printed but not executed."
fi

ensure_chain_data_cache

echo "[1/4] Manifest regeneration + hash sync"
if [[ "$SKIP_MANIFEST_REGEN" -eq 0 ]]; then
  REGEN_CMD=(
    bash "$SCRIPT_DIR/regenerate_official_manifests.sh"
    --chain-data-cache "$CHAIN_DATA_CACHE"
    --out-dir "$MANIFESTS_DIR"
    --lock-file "$LOCK_FILE"
    --sync-hashes
  )
  if [[ "$REWRITE_LOCK" -eq 1 ]]; then
    REGEN_CMD+=(--rewrite-lock)
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    REGEN_CMD+=(--dry-run)
  fi
  run_cmd "${REGEN_CMD[@]}"
else
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "+ python scripts/sync_official_manifest_hashes.py --check ..."
  else
    python "$SCRIPT_DIR/sync_official_manifest_hashes.py" \
      --manifests-dir "$MANIFESTS_DIR" \
      --track-file "$TRACK_FILE" \
      --lock-file "$LOCK_FILE" \
      --readme "$REPO_ROOT/README.md" \
      --competition-doc "$REPO_ROOT/COMPETITION.md" \
      --check
  fi
fi

echo "[2/4] Download + preprocess split NPZ data"
if [[ "$SKIP_SETUP" -eq 0 ]]; then
  SETUP_CMD=(
    bash "$SCRIPT_DIR/setup_official_data.sh"
    --data-root "$DATA_ROOT"
    --manifests-dir "$MANIFESTS_DIR"
    --processed-features-dir "$PROCESSED_FEATURES_DIR"
    --processed-labels-dir "$PROCESSED_LABELS_DIR"
    --download-retries "$DOWNLOAD_RETRIES"
    --download-retry-delay-seconds "$DOWNLOAD_RETRY_DELAY_SECONDS"
  )
  if [[ "$USE_TEMPLATES" -eq 0 ]]; then
    SETUP_CMD+=(--disable-templates)
  else
    SETUP_CMD+=(--enable-templates)
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    SETUP_CMD+=(--dry-run)
  fi
  run_cmd "${SETUP_CMD[@]}"
else
  echo "Skipping setup_official_data.sh (--skip-setup)."
fi

echo "[3/4] Build official fingerprint"
if [[ "$SKIP_FINGERPRINT" -eq 0 ]]; then
  FP_CMD=(
    python "$SCRIPT_DIR/build_fingerprint.py"
    --processed-features-dir "$PROCESSED_FEATURES_DIR"
    --processed-labels-dir "$PROCESSED_LABELS_DIR"
    --train-manifest "$MANIFESTS_DIR/train.txt"
    --val-manifest "$MANIFESTS_DIR/val.txt"
    --track "$TRACK_ID"
    --source-lock "$LOCK_FILE"
    --output "$FINGERPRINT_OUT"
  )
  run_cmd "${FP_CMD[@]}"
else
  echo "Skipping fingerprint build (--skip-fingerprint)."
fi

echo "[4/4] Final hash consistency check"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "+ python scripts/sync_official_manifest_hashes.py --check ..."
else
  python "$SCRIPT_DIR/sync_official_manifest_hashes.py" \
    --manifests-dir "$MANIFESTS_DIR" \
    --track-file "$TRACK_FILE" \
    --lock-file "$LOCK_FILE" \
    --readme "$REPO_ROOT/README.md" \
    --competition-doc "$REPO_ROOT/COMPETITION.md" \
    --check
fi

echo ""
echo "Official data refresh flow complete."
echo "Data root: $DATA_ROOT"
echo "Manifests: $MANIFESTS_DIR"
echo "Processed features: $PROCESSED_FEATURES_DIR"
echo "Processed labels: $PROCESSED_LABELS_DIR"
echo "Fingerprint: $FINGERPRINT_OUT"
