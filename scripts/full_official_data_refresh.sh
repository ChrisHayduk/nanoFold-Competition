#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/full_official_data_refresh.sh [options]

Maintainer asset rebuild for frozen official manifests (requires --skip-manifest-regen).
New cath-v2 datasets use stage_dataset_candidate.py with fresh private paths; see
regenerate_official_manifests.sh --help. Automatic manifest replacement is refused.

Asset rebuild flow:
  1) verify the frozen official manifest hashes/counts
  2) download OpenFold assets for public + hidden splits
  3) build a heldout-target MSA row filter for the manifest trio
  4) preprocess public + hidden split NPZs (features + labels) with that filter
  5) require complete features and zero cross-split target/target and target/MSA hits
  6) stage candidate fingerprints/locks and verify published manifest hashes

Rebuilt assets are candidates for review. Published fingerprints and existing
sealed hidden reference files are never replaced by this command.

Options:
  --track-id <id>                     Track id metadata for fingerprint (default: limited)
  --data-root <path>                  Download root (default: data/openproteinset)
  --manifests-dir <path>              Manifest directory (default: data/manifests)
  --private-root <path>               Maintainer-only hidden asset root (default: .nanofold_private)
  --hidden-manifests-dir <path>       Hidden manifest directory (default: <private-root>/manifests)
  --processed-features-dir <path>     Feature NPZ output dir (default: data/processed_features)
  --processed-labels-dir <path>       Label NPZ output dir (default: data/processed_labels)
  --hidden-features-dir <path>        Hidden feature NPZ output dir (default: <private-root>/hidden_processed_features)
  --hidden-labels-dir <path>          Hidden label NPZ output dir (default: <private-root>/hidden_processed_labels)
  --chain-data-cache <path>           chain_data_cache.json path
                                      (default: data/openproteinset/pdb_data/data_caches/chain_data_cache.json)
  --structure-metadata <path>         Required structure metadata JSON for split generation
                                      (default: data/manifests/structure_metadata.json)
  --metadata-sources-dir <path>       Downloaded structure metadata source directory
                                      (default: data/metadata_sources)
  --metadata-source-lock <path>       Structure metadata source lock JSON
                                      (default: data/metadata_sources/structure_metadata_sources.lock.json)
  --feature-exclusion-list <path>     Chain IDs excluded from official splitting because required feature assets are unavailable
                                      (default: data/manifests/openfold_required_feature_exclusions.txt)
  --processability-exclusion-list <path>
                                      Chain IDs excluded from official splitting because atom14 labels fail the processability gate
                                      (default: data/manifests/official_processability_exclusions.txt)
  --data-source-lock <path>           Raw official data source lock JSON
                                      (default: <private-root>/asset_rebuild/leaderboard/official_data_source.lock.json)
  --structure-candidates <path>       Accepted chain universe written by structure metadata build
                                      (default: data/manifests/structure_candidates.txt)
  --lock-file <path>                  Official manifest lock file
                                      (default: leaderboard/official_manifest_source.lock.json)
  --track-file <path>                 Frozen track policy YAML to check
                                      (default: tracks/limited.yaml)
  --fingerprint-out <path>            Candidate fingerprint output path
                                      (default: <private-root>/asset_rebuild/leaderboard/official_dataset_fingerprint.json)
                                      Other public track fingerprints are staged beside this file.
  --hidden-fingerprint-out <path>     Hidden fingerprint output path
                                      (default: <private-root>/asset_rebuild/leaderboard/official_hidden_fingerprint.json)
  --hidden-lock-file <path>           Candidate hidden asset lock path
                                      (default: <private-root>/asset_rebuild/leaderboard/private_hidden_assets.lock.json)
  --private-manifest-lock <path>      Hidden manifest source lock path
                                      (default: <private-root>/leaderboard/private_hidden_manifest_source.lock.json)
  --msa-names <csv>                   Comma-separated MSA filenames to download/preprocess
  --msa-row-filter <path>             Existing JSON of held-out-homolog MSA row hashes for public/hidden preprocessing
  --msa-row-filter-out <path>         Output path when auto-building the held-out MSA row filter
                                      (default: <private-root>/asset_rebuild/msa_row_filters/official_heldout_target_homology_filter.json)
  --skip-msa-row-filter-build         Do not auto-build an MSA row filter when --msa-row-filter is omitted.
                                      This is for exploratory/debug runs only, not official trio refreshes.
  --skip-manifest-regen               Required: keep the frozen published manifests
  --skip-setup                        Skip download+preprocess step
  --skip-fingerprint                  Skip fingerprint rebuild step
  --skip-hidden                       Skip hidden split download/preprocess/fingerprint/pinning
  --resume-preprocess                 Reuse readable feature+label NPZs and preprocess only missing or invalid chains
  --enable-templates                  Pass through to setup_official_data.sh
  --disable-templates                 Pass through to setup_official_data.sh (default)
  --mmcif-mode <mode>                 Pass through to setup_official_data.sh (default: subset)
  --download-retries <int>            Pass through to setup_official_data.sh (default: 2)
  --download-retry-delay-seconds <f>  Pass through to setup_official_data.sh (default: 2.0)
  --download-workers <int>            Concurrent per-chain and mmCIF downloads (default: 32)
  --dry-run                           Print commands without executing
  -h, --help                          Show this message
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

TRACK_ID="limited"
DATA_ROOT="data/openproteinset"
MANIFESTS_DIR="data/manifests"
PRIVATE_ROOT=".nanofold_private"
HIDDEN_MANIFESTS_DIR=""
PROCESSED_FEATURES_DIR="data/processed_features"
PROCESSED_LABELS_DIR="data/processed_labels"
HIDDEN_FEATURES_DIR=""
HIDDEN_LABELS_DIR=""
CHAIN_DATA_CACHE="data/openproteinset/pdb_data/data_caches/chain_data_cache.json"
STRUCTURE_METADATA="data/manifests/structure_metadata.json"
METADATA_SOURCES_DIR="data/metadata_sources"
METADATA_SOURCE_LOCK="data/metadata_sources/structure_metadata_sources.lock.json"
FEATURE_EXCLUSION_LIST="data/manifests/openfold_required_feature_exclusions.txt"
PROCESSABILITY_EXCLUSION_LIST="data/manifests/official_processability_exclusions.txt"
DATA_SOURCE_LOCK=""
STRUCTURE_CANDIDATES="data/manifests/structure_candidates.txt"
LOCK_FILE="leaderboard/official_manifest_source.lock.json"
TRACK_FILE="tracks/limited.yaml"
FINGERPRINT_OUT=""
HIDDEN_FINGERPRINT_OUT=""
HIDDEN_LOCK_FILE=""
PRIVATE_MANIFEST_LOCK=""
DOWNLOAD_RETRIES=2
DOWNLOAD_RETRY_DELAY_SECONDS=2.0
DOWNLOAD_WORKERS=32
MMCIF_MODE="subset"
MSA_NAMES=""
MSA_ROW_FILTER=""
MSA_ROW_FILTER_OUT=""
USER_MSA_ROW_FILTER=0
USE_TEMPLATES=0
SKIP_MANIFEST_REGEN=0
SKIP_SETUP=0
SKIP_FINGERPRINT=0
SKIP_HIDDEN=0
SKIP_MSA_ROW_FILTER_BUILD=0
RESUME_PREPROCESS=0
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
    --private-root)
      PRIVATE_ROOT="$2"
      shift 2
      ;;
    --hidden-manifests-dir|--hidden-out-dir)
      HIDDEN_MANIFESTS_DIR="$2"
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
    --feature-exclusion-list)
      FEATURE_EXCLUSION_LIST="$2"
      shift 2
      ;;
    --processability-exclusion-list)
      PROCESSABILITY_EXCLUSION_LIST="$2"
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
    --private-manifest-lock)
      PRIVATE_MANIFEST_LOCK="$2"
      shift 2
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
    --resume-preprocess)
      RESUME_PREPROCESS=1
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
    --msa-row-filter)
      MSA_ROW_FILTER="$2"
      USER_MSA_ROW_FILTER=1
      shift 2
      ;;
    --msa-row-filter-out)
      MSA_ROW_FILTER_OUT="$2"
      shift 2
      ;;
    --skip-msa-row-filter-build)
      SKIP_MSA_ROW_FILTER_BUILD=1
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
    --download-workers)
      DOWNLOAD_WORKERS="$2"
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

if [[ "$SKIP_MANIFEST_REGEN" -eq 0 ]]; then
  echo "Official manifests are frozen; automatic regeneration is disabled." >&2
  echo "Use --skip-manifest-regen to rebuild/audit published assets, or run" >&2
  echo "bash scripts/regenerate_official_manifests.sh --help for cath-v2 staging commands." >&2
  exit 1
fi

if [[ -z "$HIDDEN_MANIFESTS_DIR" ]]; then
  HIDDEN_MANIFESTS_DIR="$PRIVATE_ROOT/manifests"
fi
if [[ -z "$HIDDEN_FEATURES_DIR" ]]; then
  HIDDEN_FEATURES_DIR="$PRIVATE_ROOT/hidden_processed_features"
fi
if [[ -z "$HIDDEN_LABELS_DIR" ]]; then
  HIDDEN_LABELS_DIR="$PRIVATE_ROOT/hidden_processed_labels"
fi
if [[ -z "$DATA_SOURCE_LOCK" ]]; then
  DATA_SOURCE_LOCK="$PRIVATE_ROOT/asset_rebuild/leaderboard/official_data_source.lock.json"
fi
if [[ -z "$FINGERPRINT_OUT" ]]; then
  FINGERPRINT_OUT="$PRIVATE_ROOT/asset_rebuild/leaderboard/official_dataset_fingerprint.json"
fi
if [[ -z "$HIDDEN_FINGERPRINT_OUT" ]]; then
  HIDDEN_FINGERPRINT_OUT="$PRIVATE_ROOT/asset_rebuild/leaderboard/official_hidden_fingerprint.json"
fi
if [[ -z "$HIDDEN_LOCK_FILE" ]]; then
  HIDDEN_LOCK_FILE="$PRIVATE_ROOT/asset_rebuild/leaderboard/private_hidden_assets.lock.json"
fi
if [[ -z "$PRIVATE_MANIFEST_LOCK" ]]; then
  PRIVATE_MANIFEST_LOCK="$PRIVATE_ROOT/leaderboard/private_hidden_manifest_source.lock.json"
fi
if [[ -z "$MSA_ROW_FILTER_OUT" ]]; then
  MSA_ROW_FILTER_OUT="$PRIVATE_ROOT/asset_rebuild/msa_row_filters/official_heldout_target_homology_filter.json"
fi
if [[ "$SKIP_SETUP" -eq 0 && "$SKIP_MSA_ROW_FILTER_BUILD" -eq 0 && "$USER_MSA_ROW_FILTER" -eq 0 ]]; then
  MSA_ROW_FILTER="$MSA_ROW_FILTER_OUT"
fi

run_cmd() {
  echo "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

if ! command -v python >/dev/null 2>&1; then
  echo "python not found. Activate your environment first."
  exit 1
fi

cd "$REPO_ROOT"

# Resolve aliases before any download or output write. Hash consistency checks
# alone cannot detect replacing the published fingerprint with new asset bytes.
CANDIDATE_OUTPUTS=("$DATA_SOURCE_LOCK")
if [[ "$SKIP_FINGERPRINT" -eq 0 ]]; then
  CANDIDATE_OUTPUTS+=("$FINGERPRINT_OUT")
  if [[ "$TRACK_ID" == "limited" ]]; then
    CANDIDATE_FINGERPRINT_DIR="$(dirname -- "$FINGERPRINT_OUT")"
    RESEARCH_FINGERPRINT_OUT="$CANDIDATE_FINGERPRINT_DIR/research_large_dataset_fingerprint.json"
    UNLIMITED_FINGERPRINT_OUT="$CANDIDATE_FINGERPRINT_DIR/unlimited_dataset_fingerprint.json"
    CANDIDATE_OUTPUTS+=("$RESEARCH_FINGERPRINT_OUT" "$UNLIMITED_FINGERPRINT_OUT")
  fi
  if [[ "$SKIP_HIDDEN" -eq 0 ]]; then
    CANDIDATE_OUTPUTS+=("$HIDDEN_FINGERPRINT_OUT" "$HIDDEN_LOCK_FILE")
  fi
fi
if [[ "$SKIP_SETUP" -eq 0 && "$USER_MSA_ROW_FILTER" -eq 0 && "$SKIP_MSA_ROW_FILTER_BUILD" -eq 0 ]]; then
  CANDIDATE_OUTPUTS+=("$MSA_ROW_FILTER_OUT")
fi
python - "$REPO_ROOT" "$PRIVATE_ROOT" "$MANIFESTS_DIR" "$HIDDEN_MANIFESTS_DIR" \
  "$LOCK_FILE" "$TRACK_FILE" "$PRIVATE_MANIFEST_LOCK" "$STRUCTURE_METADATA" \
  "$METADATA_SOURCE_LOCK" "$FEATURE_EXCLUSION_LIST" "$PROCESSABILITY_EXCLUSION_LIST" \
  "${CANDIDATE_OUTPUTS[@]}" <<'PYTHON'
import sys
from pathlib import Path

repo, private, manifests, hidden_manifests = (Path(p).resolve() for p in sys.argv[1:5])
protected = [Path(p).resolve() for p in sys.argv[5:12]]
protected.extend(manifests / name for name in ("train.txt", "val.txt", "all.txt"))
protected.append(hidden_manifests / "hidden_val.txt")
protected.append(private / "msa_row_filters/official_heldout_target_homology_filter.json")
protected.extend(private / "leaderboard" / name for name in (
    "official_data_source.lock.json", "official_hidden_fingerprint.json", "private_hidden_assets.lock.json",
))
protected.extend(repo / "leaderboard" / name for name in (
    "official_dataset_fingerprint.json", "research_large_dataset_fingerprint.json",
    "unlimited_dataset_fingerprint.json",
))
protected_roots = [repo / "leaderboard", repo / "tracks", repo / "data/manifests"]
protected.extend(path for root in protected_roots for path in root.rglob("*") if path.is_file())
outputs = [Path(p).resolve() for p in sys.argv[12:]]

def aliases(left, right):
    return left == right or (left.exists() and right.exists() and left.samefile(right))

for index, output in enumerate(outputs):
    if any(output.is_relative_to(root.resolve()) for root in protected_roots) or any(
        aliases(output, reference) for reference in protected
    ):
        raise SystemExit("Refusing to overwrite a frozen dataset contract; use a candidate output path.")
    if any(aliases(output, previous) for previous in outputs[:index]):
        raise SystemExit("Candidate output paths must be distinct.")
PYTHON

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY-RUN mode enabled: commands will be printed but not executed."
fi

echo "[1/5] Verify frozen official manifest hashes"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "+ python scripts/sync_official_manifest_hashes.py --check ..."
else
  python "$SCRIPT_DIR/sync_official_manifest_hashes.py" \
    --manifests-dir "$MANIFESTS_DIR" \
    --hidden-manifest "$HIDDEN_MANIFESTS_DIR/hidden_val.txt" \
    --track-file "$TRACK_FILE" \
    --track-file "$REPO_ROOT/tracks/research_large.yaml" \
    --track-file "$REPO_ROOT/tracks/unlimited.yaml" \
    --lock-file "$LOCK_FILE" \
    --readme "$REPO_ROOT/README.md" \
    --competition-doc "$REPO_ROOT/docs/COMPETITION.md" \
    --check
fi

echo "[3/5] Download public split raw assets"
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
    --download-workers "$DOWNLOAD_WORKERS"
    --skip-preprocess
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
  if ! run_cmd "${SETUP_CMD[@]}"; then
    echo "Data refresh stopped. Inspect the preprocessing error audit in the corresponding feature root."
    echo "Manifests and exclusion lists will not be changed in response to preprocessing failures."
    exit 1
  fi
else
  echo "Skipping setup_official_data.sh (--skip-setup)."
fi

echo "[3b/5] Download hidden split raw assets"
if [[ "$SKIP_HIDDEN" -eq 0 && "$SKIP_SETUP" -eq 0 ]]; then
  HIDDEN_MANIFEST="$HIDDEN_MANIFESTS_DIR/hidden_val.txt"
  HIDDEN_PREPARE_CMD=(
    python "$SCRIPT_DIR/prepare_data.py"
    --data-root "$DATA_ROOT"
    --manifest "$HIDDEN_MANIFEST"
    --duplicate-chains-file "$DATA_ROOT/pdb_data/duplicate_pdb_chains.txt"
    --download-retries "$DOWNLOAD_RETRIES"
    --download-retry-delay-seconds "$DOWNLOAD_RETRY_DELAY_SECONDS"
    --download-workers "$DOWNLOAD_WORKERS"
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
else
  echo "Skipping hidden split raw asset download (--skip-hidden or --skip-setup)."
fi

echo "[3c/5] Build held-out-target MSA row filter"
if [[ "$SKIP_SETUP" -eq 0 && "$USER_MSA_ROW_FILTER" -eq 0 && "$SKIP_MSA_ROW_FILTER_BUILD" -eq 0 ]]; then
  MSA_FILTER_CMD=(
    python "$SCRIPT_DIR/build_msa_row_filter.py"
    --source-manifest "$MANIFESTS_DIR/train.txt"
    --source-manifest "$MANIFESTS_DIR/val.txt"
    --heldout-manifest "$MANIFESTS_DIR/val.txt"
    --chain-data-cache "$CHAIN_DATA_CACHE"
    --raw-root "$DATA_ROOT"
    --threads "$DOWNLOAD_WORKERS"
    --output "$MSA_ROW_FILTER"
  )
  if [[ "$SKIP_HIDDEN" -eq 0 ]]; then
    MSA_FILTER_CMD+=(
      --source-manifest "$HIDDEN_MANIFESTS_DIR/hidden_val.txt"
      --heldout-manifest "$HIDDEN_MANIFESTS_DIR/hidden_val.txt"
    )
  fi
  if [[ -n "$MSA_NAMES" ]]; then
    MSA_FILTER_CMD+=(--msa-names "$MSA_NAMES")
  fi
  run_cmd "${MSA_FILTER_CMD[@]}"
elif [[ -n "$MSA_ROW_FILTER" ]]; then
  echo "Using existing MSA row filter: $MSA_ROW_FILTER"
else
  echo "MSA row-filter build skipped and no --msa-row-filter was supplied; outputs will not be held-out sanitized."
fi

echo "[3d/5] Preprocess public split NPZ data"
if [[ "$SKIP_SETUP" -eq 0 ]]; then
  PUBLIC_PREPROCESS_COMMON=(
    python "$SCRIPT_DIR/preprocess.py"
    --raw-root "$DATA_ROOT"
    --mmcif-root "$DATA_ROOT/pdb_data/mmcif_files"
    --processed-features-dir "$PROCESSED_FEATURES_DIR"
    --processed-labels-dir "$PROCESSED_LABELS_DIR"
  )
  if [[ -n "$MSA_NAMES" ]]; then
    PUBLIC_PREPROCESS_COMMON+=(--msa-names "$MSA_NAMES")
  fi
  if [[ -n "$MSA_ROW_FILTER" ]]; then
    PUBLIC_PREPROCESS_COMMON+=(--msa-row-filter "$MSA_ROW_FILTER")
  fi
  if [[ "$USE_TEMPLATES" -eq 1 ]]; then
    PUBLIC_PREPROCESS_COMMON+=(--template-hhr-name "pdb70_hits.hhr")
  else
    PUBLIC_PREPROCESS_COMMON+=(--disable-templates)
  fi
  if [[ "$RESUME_PREPROCESS" -eq 1 ]]; then
    PUBLIC_PREPROCESS_COMMON+=(--skip-existing)
  fi
  if ! run_cmd "${PUBLIC_PREPROCESS_COMMON[@]}" --manifest "$MANIFESTS_DIR/train.txt"; then
    echo "Data refresh stopped. Inspect the preprocessing error audit in the corresponding feature root."
    echo "Manifests and exclusion lists will not be changed in response to preprocessing failures."
    exit 1
  fi
  if ! run_cmd "${PUBLIC_PREPROCESS_COMMON[@]}" --manifest "$MANIFESTS_DIR/val.txt"; then
    echo "Data refresh stopped. Inspect the preprocessing error audit in the corresponding feature root."
    echo "Manifests and exclusion lists will not be changed in response to preprocessing failures."
    exit 1
  fi
else
  echo "Skipping public split preprocessing (--skip-setup)."
fi

echo "[3e/5] Preprocess hidden split NPZ data"
if [[ "$SKIP_HIDDEN" -eq 0 && "$SKIP_SETUP" -eq 0 ]]; then
  HIDDEN_PREPROCESS_CMD=(
    python "$SCRIPT_DIR/preprocess.py"
    --raw-root "$DATA_ROOT"
    --mmcif-root "$DATA_ROOT/pdb_data/mmcif_files"
    --processed-features-dir "$HIDDEN_FEATURES_DIR"
    --processed-labels-dir "$HIDDEN_LABELS_DIR"
    --manifest "$HIDDEN_MANIFESTS_DIR/hidden_val.txt"
  )
  if [[ -n "$MSA_NAMES" ]]; then
    HIDDEN_PREPROCESS_CMD+=(--msa-names "$MSA_NAMES")
  fi
  if [[ -n "$MSA_ROW_FILTER" ]]; then
    HIDDEN_PREPROCESS_CMD+=(--msa-row-filter "$MSA_ROW_FILTER")
  fi
  if [[ "$USE_TEMPLATES" -eq 1 ]]; then
    HIDDEN_PREPROCESS_CMD+=(--template-hhr-name "pdb70_hits.hhr")
  else
    HIDDEN_PREPROCESS_CMD+=(--disable-templates)
  fi
  if [[ "$RESUME_PREPROCESS" -eq 1 ]]; then
    HIDDEN_PREPROCESS_CMD+=(--skip-existing)
  fi
  if ! run_cmd "${HIDDEN_PREPROCESS_CMD[@]}"; then
    echo "Data refresh stopped. Inspect the preprocessing error audit in the corresponding feature root."
    echo "Manifests and exclusion lists will not be changed in response to preprocessing failures."
    exit 1
  fi
else
  echo "Skipping hidden split preprocessing (--skip-hidden or --skip-setup)."
fi

echo "[3f/5] Enforce processed MSA and target separation audits"
PUBLIC_AUDIT_CMD=(
  python "$SCRIPT_DIR/audit_msa_split.py"
  --train-manifest "$MANIFESTS_DIR/train.txt"
  --val-manifest "$MANIFESTS_DIR/val.txt"
  --features-dir "$PROCESSED_FEATURES_DIR"
  --feature-exclusion-list "$FEATURE_EXCLUSION_LIST"
  --out-dir "$PRIVATE_ROOT/audits/public_msa_split"
  --strict --run-mmseqs --skip-plot
  --threads "$DOWNLOAD_WORKERS"
)
run_cmd "${PUBLIC_AUDIT_CMD[@]}"
if [[ "$SKIP_HIDDEN" -eq 0 ]]; then
  HIDDEN_AUDIT_CMD=(
    python "$SCRIPT_DIR/audit_msa_split.py"
    --train-manifest "$MANIFESTS_DIR/all.txt"
    --val-manifest "$HIDDEN_MANIFESTS_DIR/hidden_val.txt"
    --features-dir "$PROCESSED_FEATURES_DIR"
    --val-features-dir "$HIDDEN_FEATURES_DIR"
    --feature-exclusion-list "$FEATURE_EXCLUSION_LIST"
    --out-dir "$PRIVATE_ROOT/audits/hidden_msa_split"
    --strict --run-mmseqs --skip-plot
    --threads "$DOWNLOAD_WORKERS"
  )
  run_cmd "${HIDDEN_AUDIT_CMD[@]}"
fi

echo "[3f/5] Sync processed NPZ data to official manifests"
if [[ "$SKIP_SETUP" -eq 0 ]]; then
  SYNC_PUBLIC_CMD=(
    python "$SCRIPT_DIR/sync_processed_npz_files.py"
    --features-dir "$PROCESSED_FEATURES_DIR"
    --labels-dir "$PROCESSED_LABELS_DIR"
    --manifest "$MANIFESTS_DIR/train.txt"
    --manifest "$MANIFESTS_DIR/val.txt"
    --remove-errors
  )
  run_cmd "${SYNC_PUBLIC_CMD[@]}"
  if [[ "$SKIP_HIDDEN" -eq 0 ]]; then
    SYNC_HIDDEN_CMD=(
      python "$SCRIPT_DIR/sync_processed_npz_files.py"
      --features-dir "$HIDDEN_FEATURES_DIR"
      --labels-dir "$HIDDEN_LABELS_DIR"
      --manifest "$HIDDEN_MANIFESTS_DIR/hidden_val.txt"
      --remove-errors
    )
    run_cmd "${SYNC_HIDDEN_CMD[@]}"
  fi
else
  echo "Skipping processed NPZ sync (--skip-setup)."
fi

echo "[3g/5] Build raw source lock"
SOURCE_LOCK_CMD=(
  python "$SCRIPT_DIR/build_data_source_lock.py"
  --data-root "$DATA_ROOT"
  --manifests-dir "$MANIFESTS_DIR"
  --chain-data-cache "$CHAIN_DATA_CACHE"
  --structure-metadata "$STRUCTURE_METADATA"
  --metadata-source-lock "$METADATA_SOURCE_LOCK"
  --manifest-lock "$LOCK_FILE"
  --hidden-manifest "$HIDDEN_MANIFESTS_DIR/hidden_val.txt"
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

echo "[4/5] Stage candidate dataset fingerprints"
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
  if [[ "$TRACK_ID" == "limited" ]]; then
    RESEARCH_FP_CMD=(
      python "$SCRIPT_DIR/build_fingerprint.py"
      --processed-features-dir "$PROCESSED_FEATURES_DIR"
      --processed-labels-dir "$PROCESSED_LABELS_DIR"
      --train-manifest "$MANIFESTS_DIR/train.txt"
      --val-manifest "$MANIFESTS_DIR/val.txt"
      --track "research_large"
      --source-lock "$LOCK_FILE"
      --output "$RESEARCH_FINGERPRINT_OUT"
    )
    run_cmd "${RESEARCH_FP_CMD[@]}"
    UNLIMITED_FP_CMD=(
      python "$SCRIPT_DIR/build_fingerprint.py"
      --processed-features-dir "$PROCESSED_FEATURES_DIR"
      --processed-labels-dir "$PROCESSED_LABELS_DIR"
      --train-manifest "$MANIFESTS_DIR/train.txt"
      --val-manifest "$MANIFESTS_DIR/val.txt"
      --track "unlimited"
      --source-lock "$LOCK_FILE"
      --output "$UNLIMITED_FINGERPRINT_OUT"
    )
    run_cmd "${UNLIMITED_FP_CMD[@]}"
  fi
  if [[ "$SKIP_HIDDEN" -eq 0 ]]; then
    HIDDEN_FP_CMD=(
      python "$SCRIPT_DIR/build_fingerprint.py"
      --processed-features-dir "$HIDDEN_FEATURES_DIR"
      --processed-labels-dir "$HIDDEN_LABELS_DIR"
      --manifest "hidden_val=$HIDDEN_MANIFESTS_DIR/hidden_val.txt"
      --track "$TRACK_ID"
      --source-lock "$LOCK_FILE"
      --output "$HIDDEN_FINGERPRINT_OUT"
    )
    run_cmd "${HIDDEN_FP_CMD[@]}"
    PIN_HIDDEN_CMD=(
      python "$SCRIPT_DIR/pin_hidden_assets.py"
      --hidden-manifest "$HIDDEN_MANIFESTS_DIR/hidden_val.txt"
      --hidden-features-dir "$HIDDEN_FEATURES_DIR"
      --hidden-labels-dir "$HIDDEN_LABELS_DIR"
      --hidden-fingerprint "$HIDDEN_FINGERPRINT_OUT"
      --track-id "$TRACK_ID"
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
    --hidden-manifest "$HIDDEN_MANIFESTS_DIR/hidden_val.txt" \
    --track-file "$TRACK_FILE" \
    --track-file "$REPO_ROOT/tracks/research_large.yaml" \
    --track-file "$REPO_ROOT/tracks/unlimited.yaml" \
    --lock-file "$LOCK_FILE" \
    --readme "$REPO_ROOT/README.md" \
    --competition-doc "$REPO_ROOT/docs/COMPETITION.md" \
    --check
fi

echo ""
echo "Asset rebuild complete. Candidate fingerprints and locks require review before a versioned release."
echo "Data root: $DATA_ROOT"
echo "Manifests: $MANIFESTS_DIR"
echo "Processed features: $PROCESSED_FEATURES_DIR"
echo "Processed labels: $PROCESSED_LABELS_DIR"
echo "Fingerprint: $FINGERPRINT_OUT"
if [[ -n "$MSA_ROW_FILTER" ]]; then
  echo "MSA row filter: $MSA_ROW_FILTER"
fi
echo "Structure metadata: $STRUCTURE_METADATA"
echo "Metadata sources: $METADATA_SOURCES_DIR"
echo "Data source lock: $DATA_SOURCE_LOCK"
echo "Private hidden workspace: $PRIVATE_ROOT"
