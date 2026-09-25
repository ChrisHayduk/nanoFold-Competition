#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/full_official_data_refresh.sh [options]

Maintainer asset rebuild for frozen official manifests (requires --skip-manifest-regen).
New candidates use stage_dataset_candidate.py (default cath-v3) with fresh private
paths and both --split-group-tsv and --split-group-receipt. See
regenerate_official_manifests.sh --help. Automatic manifest replacement is refused.

Asset rebuild flow:
  1) verify the frozen official manifest hashes/counts
  2) download OpenFold assets for public + hidden splits
  3) build directed train-to-heldout and public-validation-to-hidden MSA filters
  4) preprocess public NPZs with split-specific filters and hidden NPZs unfiltered
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
                                      (default: <data-root>/pdb_data/data_caches/chain_data_cache.json)
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
  --train-msa-row-filter <path>       Existing exhaustive train-to-public/hidden-validation filter
  --val-msa-row-filter <path>         Existing exhaustive public-validation-to-hidden filter
                                      Supply both filters together, or both are built from raw MSAs.
  --msa-filter-policy-out <path>      Split-filter policy output (default: <private-root>/asset_rebuild/msa_row_filters/public_directed_policy.json)
                                      Auto-built train_filter.json and val_filter.json are written beside it.
  --msa-filter-checkpoint-dir <path>  Bounded exhaustive MSA search checkpoint root (default: beside policy, checkpoints/)
  --skip-msa-row-filter-build         Require the two existing filters when preprocessing.
  --skip-manifest-regen               Required: keep the frozen published manifests
  --skip-fingerprint                  Skip fingerprint rebuild step
  --skip-hidden                       Skip hidden assets; hidden target manifest/cache remain required for public filters
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
CHAIN_DATA_CACHE=""
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
MSA_MIN_SEQ_ID=0.30
MSA_COVERAGE=0.80
TRAIN_MSA_ROW_FILTER=""
VAL_MSA_ROW_FILTER=""
MSA_FILTER_POLICY_OUT=""
MSA_FILTER_CHECKPOINT_DIR=""
USER_MSA_ROW_FILTERS=0
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
      echo "--skip-setup cannot establish directed MSA provenance; use --resume-preprocess for validated reuse." >&2
      exit 1
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
    --train-msa-row-filter)
      TRAIN_MSA_ROW_FILTER="$2"
      shift 2
      ;;
    --val-msa-row-filter)
      VAL_MSA_ROW_FILTER="$2"
      shift 2
      ;;
    --msa-filter-policy-out)
      MSA_FILTER_POLICY_OUT="$2"
      shift 2
      ;;
    --msa-filter-checkpoint-dir)
      MSA_FILTER_CHECKPOINT_DIR="$2"
      shift 2
      ;;
    --msa-row-filter|--msa-row-filter-out)
      echo "Global MSA filters are not supported here; supply --train-msa-row-filter and --val-msa-row-filter together." >&2
      exit 1
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
  echo "bash scripts/regenerate_official_manifests.sh --help for versioned candidate staging commands." >&2
  exit 1
fi

if [[ -z "$HIDDEN_MANIFESTS_DIR" ]]; then
  HIDDEN_MANIFESTS_DIR="$PRIVATE_ROOT/manifests"
fi
if [[ -z "$CHAIN_DATA_CACHE" ]]; then
  CHAIN_DATA_CACHE="$DATA_ROOT/pdb_data/data_caches/chain_data_cache.json"
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
if [[ -n "$TRAIN_MSA_ROW_FILTER" || -n "$VAL_MSA_ROW_FILTER" ]]; then
  if [[ -z "$TRAIN_MSA_ROW_FILTER" || -z "$VAL_MSA_ROW_FILTER" ]]; then
    echo "Supply both --train-msa-row-filter and --val-msa-row-filter." >&2
    exit 1
  fi
  USER_MSA_ROW_FILTERS=1
fi
if [[ "$SKIP_SETUP" -eq 0 && "$SKIP_MSA_ROW_FILTER_BUILD" -eq 1 && "$USER_MSA_ROW_FILTERS" -eq 0 ]]; then
  echo "Preprocessing requires both directed filters; --skip-msa-row-filter-build requires existing filters." >&2
  exit 1
fi
if [[ -z "$MSA_FILTER_POLICY_OUT" ]]; then
  MSA_FILTER_POLICY_OUT="$PRIVATE_ROOT/asset_rebuild/msa_row_filters/public_directed_policy.json"
fi
MSA_FILTER_DIR="$(dirname -- "$MSA_FILTER_POLICY_OUT")"
if [[ -z "$MSA_FILTER_CHECKPOINT_DIR" ]]; then
  MSA_FILTER_CHECKPOINT_DIR="$MSA_FILTER_DIR/checkpoints"
fi
if [[ "$USER_MSA_ROW_FILTERS" -eq 0 ]]; then
  TRAIN_MSA_ROW_FILTER="$MSA_FILTER_DIR/train_filter.json"
  VAL_MSA_ROW_FILTER="$MSA_FILTER_DIR/val_filter.json"
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

ASSET_DIRS=("$DATA_ROOT" "$PROCESSED_FEATURES_DIR" "$PROCESSED_LABELS_DIR")
if [[ "$SKIP_HIDDEN" -eq 0 ]]; then
  ASSET_DIRS+=("$HIDDEN_FEATURES_DIR" "$HIDDEN_LABELS_DIR")
fi
python - "$REPO_ROOT" "$MANIFESTS_DIR" "$HIDDEN_MANIFESTS_DIR" "${ASSET_DIRS[@]}" <<'PYTHON'
import sys
from pathlib import Path

repo, manifests, hidden_manifests, *assets = [Path(value).resolve() for value in sys.argv[1:]]
protected = [manifests, hidden_manifests, *(repo / name for name in ("leaderboard", "tracks", "data/manifests"))]

def overlaps(left, right):
    return left.is_relative_to(right) or right.is_relative_to(left)

for index, asset in enumerate(assets):
    if any(overlaps(asset, other.resolve()) for other in [*protected, *assets[:index]]):
        raise SystemExit("Raw, public and hidden asset directories must be separate from manifests, contracts and each other.")
    if asset.exists() and not asset.is_dir():
        raise SystemExit("Asset output paths must be directories.")
for hidden in assets[3:]:
    if hidden.is_relative_to(repo) and not hidden.is_relative_to(repo / ".nanofold_private"):
        raise SystemExit("In-repository hidden assets must stay under .nanofold_private.")
PYTHON

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
if [[ "$SKIP_SETUP" -eq 0 ]]; then
  CANDIDATE_OUTPUTS+=("$MSA_FILTER_POLICY_OUT")
  if [[ "$USER_MSA_ROW_FILTERS" -eq 0 ]]; then
    CANDIDATE_OUTPUTS+=("$TRAIN_MSA_ROW_FILTER" "$VAL_MSA_ROW_FILTER")
  fi
fi
PROTECTED_TRAIN_MSA_ROW_FILTER=""
PROTECTED_VAL_MSA_ROW_FILTER=""
if [[ "$USER_MSA_ROW_FILTERS" -eq 1 ]]; then
  PROTECTED_TRAIN_MSA_ROW_FILTER="$TRAIN_MSA_ROW_FILTER"
  PROTECTED_VAL_MSA_ROW_FILTER="$VAL_MSA_ROW_FILTER"
fi
python - "$REPO_ROOT" "$PRIVATE_ROOT" "$MANIFESTS_DIR" "$HIDDEN_MANIFESTS_DIR" \
  "$LOCK_FILE" "$TRACK_FILE" "$PRIVATE_MANIFEST_LOCK" "$STRUCTURE_METADATA" \
  "$METADATA_SOURCE_LOCK" "$FEATURE_EXCLUSION_LIST" "$PROCESSABILITY_EXCLUSION_LIST" \
  "$CHAIN_DATA_CACHE" "$PROTECTED_TRAIN_MSA_ROW_FILTER" "$PROTECTED_VAL_MSA_ROW_FILTER" \
  "${CANDIDATE_OUTPUTS[@]}" <<'PYTHON'
import sys
from pathlib import Path

repo, private, manifests, hidden_manifests = (Path(p).resolve() for p in sys.argv[1:5])
protected = [Path(p).resolve() for p in sys.argv[5:15] if p]
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
outputs = [Path(p).resolve() for p in sys.argv[15:]]

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

# Search checkpoints and generated policy/filter files must stay outside asset
# roots, and checkpoint jobs must never write through a protected input path.
python - "$MSA_FILTER_CHECKPOINT_DIR" \
  "$MANIFESTS_DIR" "$HIDDEN_MANIFESTS_DIR" "$REPO_ROOT/leaderboard" "$REPO_ROOT/tracks" \
  "$TRAIN_MSA_ROW_FILTER" "$VAL_MSA_ROW_FILTER" "$CHAIN_DATA_CACHE" \
  "${#CANDIDATE_OUTPUTS[@]}" "${CANDIDATE_OUTPUTS[@]}" "${ASSET_DIRS[@]}" <<'PYTHON'
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1]).resolve()
protected = [Path(p).resolve() for p in sys.argv[2:9]]
count = int(sys.argv[9])
outputs = [Path(p).resolve() for p in sys.argv[10:10 + count]]
assets = [Path(p).resolve() for p in sys.argv[10 + count:]]

def overlaps(left, right):
    return left.is_relative_to(right) or right.is_relative_to(left)

if any(overlaps(output, asset) for output in outputs for asset in assets):
    raise SystemExit("Candidate output files must be separate from raw and processed asset roots.")
if any(overlaps(checkpoint, path) for path in [*protected, *outputs, *assets]):
    raise SystemExit("MSA filter checkpoints must be separate from inputs, outputs and asset roots.")
if checkpoint.exists() and not checkpoint.is_dir():
    raise SystemExit("MSA filter checkpoint path must be a directory.")
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
    --chain-data-cache "$CHAIN_DATA_CACHE"
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

echo "[3c/5] Build directed MSA row filters and public split policy"
if [[ "$SKIP_SETUP" -eq 0 ]]; then
  if [[ "$USER_MSA_ROW_FILTERS" -eq 0 ]]; then
    MSA_FILTER_COMMON=(
      python "$SCRIPT_DIR/build_msa_row_filter.py"
      --chain-data-cache "$CHAIN_DATA_CACHE"
      --raw-root "$DATA_ROOT"
      --threads "$DOWNLOAD_WORKERS"
      --exhaustive
      --min-seq-id "$MSA_MIN_SEQ_ID" --coverage "$MSA_COVERAGE"
    )
    if [[ -n "$MSA_NAMES" ]]; then
      MSA_FILTER_COMMON+=(--msa-names "$MSA_NAMES")
    fi
    run_cmd "${MSA_FILTER_COMMON[@]}" \
      --source-manifest "$MANIFESTS_DIR/train.txt" \
      --heldout-manifest "$MANIFESTS_DIR/val.txt" \
      --heldout-manifest "$HIDDEN_MANIFESTS_DIR/hidden_val.txt" \
      --checkpoint-dir "$MSA_FILTER_CHECKPOINT_DIR/train" \
      --output "$TRAIN_MSA_ROW_FILTER"
    run_cmd "${MSA_FILTER_COMMON[@]}" \
      --source-manifest "$MANIFESTS_DIR/val.txt" \
      --heldout-manifest "$HIDDEN_MANIFESTS_DIR/hidden_val.txt" \
      --checkpoint-dir "$MSA_FILTER_CHECKPOINT_DIR/val" \
      --output "$VAL_MSA_ROW_FILTER"
  fi
  run_cmd python "$SCRIPT_DIR/build_msa_filter_policy.py" \
    --train-manifest "$MANIFESTS_DIR/train.txt" \
    --val-manifest "$MANIFESTS_DIR/val.txt" \
    --hidden-manifest "$HIDDEN_MANIFESTS_DIR/hidden_val.txt" \
    --train-msa-row-filter "$TRAIN_MSA_ROW_FILTER" \
    --val-msa-row-filter "$VAL_MSA_ROW_FILTER" \
    --output "$MSA_FILTER_POLICY_OUT"
else
  echo "Skipping MSA filter construction (--skip-setup)."
fi

echo "[3d/5] Preprocess public split NPZ data"
if [[ "$SKIP_SETUP" -eq 0 ]]; then
  PUBLIC_PREPROCESS_COMMON=(
    python "$SCRIPT_DIR/preprocess.py"
    --raw-root "$DATA_ROOT"
    --chain-data-cache "$CHAIN_DATA_CACHE"
    --mmcif-root "$DATA_ROOT/pdb_data/mmcif_files"
    --processed-features-dir "$PROCESSED_FEATURES_DIR"
    --processed-labels-dir "$PROCESSED_LABELS_DIR"
  )
  if [[ -n "$MSA_NAMES" ]]; then
    PUBLIC_PREPROCESS_COMMON+=(--msa-names "$MSA_NAMES")
  fi
  PUBLIC_PREPROCESS_COMMON+=(--msa-filter-policy "$MSA_FILTER_POLICY_OUT")
  if [[ "$USE_TEMPLATES" -eq 1 ]]; then
    PUBLIC_PREPROCESS_COMMON+=(--template-hhr-name "pdb70_hits.hhr")
  else
    PUBLIC_PREPROCESS_COMMON+=(--disable-templates)
  fi
  if [[ "$RESUME_PREPROCESS" -eq 1 ]]; then
    PUBLIC_PREPROCESS_COMMON+=(--skip-existing)
  fi
  if ! run_cmd "${PUBLIC_PREPROCESS_COMMON[@]}" --manifest "$MANIFESTS_DIR/all.txt"; then
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
    --chain-data-cache "$CHAIN_DATA_CACHE"
    --mmcif-root "$DATA_ROOT/pdb_data/mmcif_files"
    --processed-features-dir "$HIDDEN_FEATURES_DIR"
    --processed-labels-dir "$HIDDEN_LABELS_DIR"
    --manifest "$HIDDEN_MANIFESTS_DIR/hidden_val.txt"
  )
  if [[ -n "$MSA_NAMES" ]]; then
    HIDDEN_PREPROCESS_CMD+=(--msa-names "$MSA_NAMES")
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
  --strict --run-mmseqs --exhaustive --skip-plot
  --min-seq-id "$MSA_MIN_SEQ_ID" --coverage "$MSA_COVERAGE"
  --checkpoint-dir "$MSA_FILTER_CHECKPOINT_DIR/residual_train_val"
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
    --strict --run-mmseqs --exhaustive --skip-plot
    --min-seq-id "$MSA_MIN_SEQ_ID" --coverage "$MSA_COVERAGE"
    --checkpoint-dir "$MSA_FILTER_CHECKPOINT_DIR/residual_public_hidden"
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
if [[ "$SKIP_SETUP" -eq 0 ]]; then
  echo "Directed public MSA filter policy: $MSA_FILTER_POLICY_OUT"
fi
echo "Structure metadata: $STRUCTURE_METADATA"
echo "Metadata sources: $METADATA_SOURCES_DIR"
echo "Data source lock: $DATA_SOURCE_LOCK"
echo "Private hidden workspace: $PRIVATE_ROOT"
