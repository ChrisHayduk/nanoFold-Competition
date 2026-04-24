#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/full_official_data_refresh.sh [options]

Single maintainer end-to-end flow for official data refresh:
  0) bootstrap chain_data_cache.json if missing
  1) build required structure metadata for split stratification
  2) regenerate official train/val/hidden_val/all manifests from locked inputs
  3) sync official manifest hashes/counts across track + lock + docs
  4) download OpenFold assets and preprocess public split NPZs (features + labels)
  5) rebuild official dataset fingerprint

Options:
  --track-id <id>                     Track id metadata for fingerprint (default: limited_large)
  --data-root <path>                  Download root (default: data/openproteinset)
  --manifests-dir <path>              Manifest directory (default: data/manifests)
  --processed-features-dir <path>     Feature NPZ output dir (default: data/processed_features)
  --processed-labels-dir <path>       Label NPZ output dir (default: data/processed_labels)
  --hidden-features-dir <path>        Hidden feature NPZ output dir (default: data/hidden_processed_features)
  --hidden-labels-dir <path>          Hidden label NPZ output dir (default: data/hidden_processed_labels)
  --chain-data-cache <path>           chain_data_cache.json path
                                      (default: data/openproteinset/pdb_data/data_caches/chain_data_cache.json)
  --structure-metadata <path>         Required structure metadata JSON for split generation
                                      (default: data/manifests/structure_metadata.json)
  --metadata-sources-dir <path>       Downloaded structure metadata source directory
                                      (default: data/metadata_sources)
  --metadata-source-lock <path>       Structure metadata source lock JSON
                                      (default: data/metadata_sources/structure_metadata_sources.lock.json)
  --data-source-lock <path>           Raw official data source lock JSON
                                      (default: leaderboard/official_data_source.lock.json)
  --structure-candidates <path>       Candidate manifest used to fetch mmCIFs for metadata
                                      (default: data/manifests/structure_candidates.txt)
  --lock-file <path>                  Official manifest lock file
                                      (default: leaderboard/official_manifest_source.lock.json)
  --track-file <path>                 Track policy YAML to update/check
                                      (default: tracks/limited_large.yaml)
  --fingerprint-out <path>            Fingerprint output path
                                      (default: leaderboard/official_dataset_fingerprint.json)
  --hidden-fingerprint-out <path>     Hidden fingerprint output path
                                      (default: leaderboard/official_hidden_fingerprint.json)
  --hidden-lock-file <path>           Hidden asset lock path
                                      (default: leaderboard/official_hidden_assets.lock.json)
  --msa-names <csv>                   Comma-separated MSA filenames to download/preprocess
  --rewrite-lock                      Rewrite lock metadata after manifest regeneration
  --skip-manifest-regen               Skip manifest regeneration step
  --skip-setup                        Skip download+preprocess step
  --skip-fingerprint                  Skip fingerprint rebuild step
  --skip-hidden                       Skip hidden split download/preprocess/fingerprint/pinning
  --enable-templates                  Pass through to setup_official_data.sh
  --disable-templates                 Pass through to setup_official_data.sh (default)
  --mmcif-mode <mode>                 Pass through to setup_official_data.sh (default: subset)
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
HIDDEN_FEATURES_DIR="data/hidden_processed_features"
HIDDEN_LABELS_DIR="data/hidden_processed_labels"
CHAIN_DATA_CACHE="data/openproteinset/pdb_data/data_caches/chain_data_cache.json"
STRUCTURE_METADATA="data/manifests/structure_metadata.json"
METADATA_SOURCES_DIR="data/metadata_sources"
METADATA_SOURCE_LOCK="data/metadata_sources/structure_metadata_sources.lock.json"
DATA_SOURCE_LOCK="leaderboard/official_data_source.lock.json"
STRUCTURE_CANDIDATES="data/manifests/structure_candidates.txt"
LOCK_FILE="leaderboard/official_manifest_source.lock.json"
TRACK_FILE="tracks/limited_large.yaml"
FINGERPRINT_OUT="leaderboard/official_dataset_fingerprint.json"
HIDDEN_FINGERPRINT_OUT="leaderboard/official_hidden_fingerprint.json"
HIDDEN_LOCK_FILE="leaderboard/official_hidden_assets.lock.json"
DOWNLOAD_RETRIES=2
DOWNLOAD_RETRY_DELAY_SECONDS=2.0
MMCIF_MODE="subset"
MSA_NAMES=""
USE_TEMPLATES=0
REWRITE_LOCK=0
SKIP_MANIFEST_REGEN=0
SKIP_SETUP=0
SKIP_FINGERPRINT=0
SKIP_HIDDEN=0
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
    --hidden-features-dir)
      HIDDEN_FEATURES_DIR="$2"
      shift 2
      ;;
    --hidden-labels-dir)
      HIDDEN_LABELS_DIR="$2"
      shift 2
      ;;
    --chain-data-cache)
      CHAIN_DATA_CACHE="$2"
      shift 2
      ;;
    --structure-metadata)
      STRUCTURE_METADATA="$2"
      shift 2
      ;;
    --metadata-sources-dir)
      METADATA_SOURCES_DIR="$2"
      shift 2
      ;;
    --metadata-source-lock)
      METADATA_SOURCE_LOCK="$2"
      shift 2
      ;;
    --data-source-lock)
      DATA_SOURCE_LOCK="$2"
      shift 2
      ;;
    --structure-candidates)
      STRUCTURE_CANDIDATES="$2"
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
    --hidden-fingerprint-out)
      HIDDEN_FINGERPRINT_OUT="$2"
      shift 2
      ;;
    --hidden-lock-file)
      HIDDEN_LOCK_FILE="$2"
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
    --skip-hidden)
      SKIP_HIDDEN=1
      shift 1
      ;;
    --disable-templates)
      USE_TEMPLATES=0
      shift 1
      ;;
    --mmcif-mode)
      MMCIF_MODE="$2"
      shift 2
      ;;
    --msa-names)
      MSA_NAMES="$2"
      shift 2
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

echo "[1/5] Build required structure metadata"
MMCIF_ROOT="$DATA_ROOT/pdb_data/mmcif_files"
if [[ "$SKIP_MANIFEST_REGEN" -eq 0 ]]; then
  METADATA_SOURCE_CMD=(
    python "$SCRIPT_DIR/download_structure_metadata_sources.py"
    --chain-data-cache "$CHAIN_DATA_CACHE"
    --out-dir "$METADATA_SOURCES_DIR"
    --source-lock "$METADATA_SOURCE_LOCK"
    --download-retries "$DOWNLOAD_RETRIES"
    --download-retry-delay-seconds "$DOWNLOAD_RETRY_DELAY_SECONDS"
  )
  if [[ "$DRY_RUN" -eq 1 ]]; then
    METADATA_SOURCE_CMD+=(--dry-run)
  fi
  run_cmd "${METADATA_SOURCE_CMD[@]}"

  STRUCTURE_META_CMD=(
    python "$SCRIPT_DIR/build_structure_metadata.py"
    --chain-data-cache "$CHAIN_DATA_CACHE"
    --mmcif-root "$MMCIF_ROOT"
    --metadata-out "$STRUCTURE_METADATA"
    --metadata-sources-dir "$METADATA_SOURCES_DIR"
    --metadata-source-lock "$METADATA_SOURCE_LOCK"
    --candidate-manifest-out "$STRUCTURE_CANDIDATES"
  )
  run_cmd "${STRUCTURE_META_CMD[@]}"

  MMCIF_CANDIDATE_CMD=(
    python "$SCRIPT_DIR/prepare_data.py"
    --data-root "$DATA_ROOT"
    --manifest "$STRUCTURE_CANDIDATES"
    --only-mmcif-subset
    --download-retries "$DOWNLOAD_RETRIES"
    --download-retry-delay-seconds "$DOWNLOAD_RETRY_DELAY_SECONDS"
    --strict-downloads
  )
  if [[ "$DRY_RUN" -eq 1 ]]; then
    MMCIF_CANDIDATE_CMD+=(--dry-run)
  fi
  run_cmd "${MMCIF_CANDIDATE_CMD[@]}"
  run_cmd "${STRUCTURE_META_CMD[@]}"
else
  echo "Skipping structure metadata rebuild (--skip-manifest-regen)."
fi

echo "[2/5] Manifest regeneration + hash sync"
if [[ "$SKIP_MANIFEST_REGEN" -eq 0 ]]; then
  REGEN_CMD=(
    bash "$SCRIPT_DIR/regenerate_official_manifests.sh"
    --chain-data-cache "$CHAIN_DATA_CACHE"
    --out-dir "$MANIFESTS_DIR"
    --lock-file "$LOCK_FILE"
    --structure-metadata "$STRUCTURE_METADATA"
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

echo "[3/5] Download + preprocess public split NPZ data"
if [[ "$SKIP_SETUP" -eq 0 ]]; then
  SETUP_CMD=(
    bash "$SCRIPT_DIR/setup_official_data.sh"
    --data-root "$DATA_ROOT"
    --manifests-dir "$MANIFESTS_DIR"
    --processed-features-dir "$PROCESSED_FEATURES_DIR"
    --processed-labels-dir "$PROCESSED_LABELS_DIR"
    --mmcif-mode "$MMCIF_MODE"
    --download-retries "$DOWNLOAD_RETRIES"
    --download-retry-delay-seconds "$DOWNLOAD_RETRY_DELAY_SECONDS"
  )
  if [[ -n "$MSA_NAMES" ]]; then
    SETUP_CMD+=(--msa-names "$MSA_NAMES")
  fi
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

echo "[3b/5] Download + preprocess hidden split NPZ data"
if [[ "$SKIP_HIDDEN" -eq 0 && "$SKIP_SETUP" -eq 0 ]]; then
  HIDDEN_MANIFEST="$MANIFESTS_DIR/hidden_val.txt"
  HIDDEN_PREPARE_CMD=(
    python "$SCRIPT_DIR/prepare_data.py"
    --data-root "$DATA_ROOT"
    --manifest "$HIDDEN_MANIFEST"
    --duplicate-chains-file "$DATA_ROOT/pdb_data/duplicate_pdb_chains.txt"
    --download-retries "$DOWNLOAD_RETRIES"
    --download-retry-delay-seconds "$DOWNLOAD_RETRY_DELAY_SECONDS"
    --strict-downloads
  )
  if [[ -n "$MSA_NAMES" ]]; then
    HIDDEN_PREPARE_CMD+=(--msa-names "$MSA_NAMES")
  fi
  if [[ "$USE_TEMPLATES" -eq 1 ]]; then
    HIDDEN_PREPARE_CMD+=(--template-hits-name "pdb70_hits.hhr")
  else
    HIDDEN_PREPARE_CMD+=(--no-template-hits)
  fi
  if [[ "$MMCIF_MODE" == "subset" ]]; then
    HIDDEN_PREPARE_CMD+=(--download-mmcif-subset)
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    HIDDEN_PREPARE_CMD+=(--dry-run)
  fi
  run_cmd "${HIDDEN_PREPARE_CMD[@]}"

  HIDDEN_PREPROCESS_CMD=(
    python "$SCRIPT_DIR/preprocess.py"
    --raw-root "$DATA_ROOT"
    --mmcif-root "$MMCIF_ROOT"
    --processed-features-dir "$HIDDEN_FEATURES_DIR"
    --processed-labels-dir "$HIDDEN_LABELS_DIR"
    --manifest "$HIDDEN_MANIFEST"
    --strict
  )
  if [[ -n "$MSA_NAMES" ]]; then
    HIDDEN_PREPROCESS_CMD+=(--msa-names "$MSA_NAMES")
  fi
  if [[ "$USE_TEMPLATES" -eq 1 ]]; then
    HIDDEN_PREPROCESS_CMD+=(--template-hhr-name "pdb70_hits.hhr")
  else
    HIDDEN_PREPROCESS_CMD+=(--disable-templates)
  fi
  run_cmd "${HIDDEN_PREPROCESS_CMD[@]}"
else
  echo "Skipping hidden split data build (--skip-hidden or --skip-setup)."
fi

echo "[3c/5] Build raw source lock"
SOURCE_LOCK_CMD=(
  python "$SCRIPT_DIR/build_data_source_lock.py"
  --data-root "$DATA_ROOT"
  --manifests-dir "$MANIFESTS_DIR"
  --chain-data-cache "$CHAIN_DATA_CACHE"
  --structure-metadata "$STRUCTURE_METADATA"
  --metadata-source-lock "$METADATA_SOURCE_LOCK"
  --manifest-lock "$LOCK_FILE"
  --output "$DATA_SOURCE_LOCK"
)
if [[ -n "$MSA_NAMES" ]]; then
  SOURCE_LOCK_CMD+=(--msa-names "$MSA_NAMES")
fi
if [[ "$USE_TEMPLATES" -eq 1 ]]; then
  SOURCE_LOCK_CMD+=(--enable-templates)
fi
if [[ "$SKIP_HIDDEN" -eq 0 ]]; then
  SOURCE_LOCK_CMD+=(--include-hidden)
fi
if [[ "$SKIP_SETUP" -eq 0 ]]; then
  SOURCE_LOCK_CMD+=(--require-complete)
fi
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "+ ${SOURCE_LOCK_CMD[*]}"
else
  run_cmd "${SOURCE_LOCK_CMD[@]}"
fi

echo "[4/5] Build official fingerprint"
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
  if [[ "$SKIP_HIDDEN" -eq 0 ]]; then
    HIDDEN_FP_CMD=(
      python "$SCRIPT_DIR/build_fingerprint.py"
      --processed-features-dir "$HIDDEN_FEATURES_DIR"
      --processed-labels-dir "$HIDDEN_LABELS_DIR"
      --manifest "hidden_val=$MANIFESTS_DIR/hidden_val.txt"
      --track "$TRACK_ID"
      --source-lock "$LOCK_FILE"
      --output "$HIDDEN_FINGERPRINT_OUT"
    )
    run_cmd "${HIDDEN_FP_CMD[@]}"
    PIN_HIDDEN_CMD=(
      python "$SCRIPT_DIR/pin_hidden_assets.py"
      --hidden-manifest "$MANIFESTS_DIR/hidden_val.txt"
      --hidden-features-dir "$HIDDEN_FEATURES_DIR"
      --hidden-labels-dir "$HIDDEN_LABELS_DIR"
      --hidden-fingerprint "$HIDDEN_FINGERPRINT_OUT"
      --track-file "$TRACK_FILE"
      --lock-file "$HIDDEN_LOCK_FILE"
    )
    run_cmd "${PIN_HIDDEN_CMD[@]}"
  fi
else
  echo "Skipping fingerprint build (--skip-fingerprint)."
fi

echo "[5/5] Final hash consistency check"
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
echo "Structure metadata: $STRUCTURE_METADATA"
echo "Metadata sources: $METADATA_SOURCES_DIR"
echo "Data source lock: $DATA_SOURCE_LOCK"
