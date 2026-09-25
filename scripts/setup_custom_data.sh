#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash scripts/setup_custom_data.sh [options]

Download and preprocess the public train/val assets of an already-staged custom
candidate. Generate and verify manifests first with stage_dataset_candidate.py.
This command verifies the supplied source lock and never regenerates manifests,
rewrites all.txt, or updates a published dataset contract. It downloads only the
manifest mmCIF subset and preprocesses without templates.

Required options:
  --manifests-dir <path>          Existing staged public train.txt, val.txt and all.txt
  --manifest-lock-file <path>     Existing candidate public source lock (read-only)
  --data-root <path>              Custom raw asset directory
  --processed-features-dir <path> Custom feature NPZ directory
  --processed-labels-dir <path>   Custom label NPZ directory

Optional:
  --msa-name <filename>          MSA filename (default: uniref90_hits.a3m)
  --msa-row-filter <path>        Existing private held-out-target MSA filter
  --download-retries <int>       AWS download retries (default: 2)
  --download-retry-delay-seconds <float>
                                 Retry delay (default: 2.0)
  --disable-templates            Explicitly select the default no-template policy
  --resume-preprocess            Reuse pairs with matching schema-3 provenance
  --skip-preprocess              Download assets only
  --dry-run                     Validate inputs and print commands without writes
  -h, --help                    Show this message

All asset paths must be distinct custom directories, separate from the manifests
and published asset paths. Strict homology audits and fingerprints remain required
before a candidate can be adopted as a dataset release.
HELP
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT=""
MANIFESTS_DIR=""
MANIFEST_LOCK_FILE=""
PROCESSED_FEATURES_DIR=""
PROCESSED_LABELS_DIR=""
MSA_NAME="uniref90_hits.a3m"
MSA_ROW_FILTER=""
DOWNLOAD_RETRIES=2
DOWNLOAD_RETRY_DELAY_SECONDS=2.0
SKIP_PREPROCESS=0
RESUME_PREPROCESS=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="${2:?Missing --data-root value}"; shift 2 ;;
    --manifests-dir) MANIFESTS_DIR="${2:?Missing --manifests-dir value}"; shift 2 ;;
    --manifest-lock-file) MANIFEST_LOCK_FILE="${2:?Missing --manifest-lock-file value}"; shift 2 ;;
    --processed-features-dir) PROCESSED_FEATURES_DIR="${2:?Missing --processed-features-dir value}"; shift 2 ;;
    --processed-labels-dir) PROCESSED_LABELS_DIR="${2:?Missing --processed-labels-dir value}"; shift 2 ;;
    --msa-name) MSA_NAME="${2:?Missing --msa-name value}"; shift 2 ;;
    --msa-row-filter) MSA_ROW_FILTER="${2:?Missing --msa-row-filter value}"; shift 2 ;;
    --download-retries) DOWNLOAD_RETRIES="${2:?Missing --download-retries value}"; shift 2 ;;
    --download-retry-delay-seconds) DOWNLOAD_RETRY_DELAY_SECONDS="${2:?Missing retry delay}"; shift 2 ;;
    --disable-templates) shift ;;
    --resume-preprocess) RESUME_PREPROCESS=1; shift ;;
    --skip-preprocess) SKIP_PREPROCESS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --train-size|--val-size|--seed|--min-len|--max-len|--max-resolution|--expected-chain-cache-sha256)
      echo "Manifest generation flags are not accepted. Use scripts/stage_dataset_candidate.py first." >&2
      exit 1 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$DATA_ROOT" || -z "$MANIFESTS_DIR" || -z "$MANIFEST_LOCK_FILE" || -z "$PROCESSED_FEATURES_DIR" || -z "$PROCESSED_LABELS_DIR" ]]; then
  echo "Explicit custom data, manifest, source-lock, feature and label paths are required; see --help." >&2
  exit 1
fi
if ! command -v python >/dev/null 2>&1; then
  echo "python not found. Activate your environment first." >&2
  exit 1
fi

# All validation is read-only and precedes even output directory creation.
python - "$REPO_ROOT" "$MANIFESTS_DIR" "$MANIFEST_LOCK_FILE" "$MSA_ROW_FILTER" \
  "$DATA_ROOT" "$PROCESSED_FEATURES_DIR" "$PROCESSED_LABELS_DIR" <<'PYTHON'
import hashlib
import json
import sys
from pathlib import Path

repo, manifests, lock_path = (Path(p).resolve() for p in sys.argv[1:4])
filter_path = Path(sys.argv[4]).resolve() if sys.argv[4] else None
outputs = [Path(p).resolve() for p in sys.argv[5:8]]
protected = [repo / name for name in (
    "data/manifests", "data/openproteinset", "data/processed_features", "data/processed_labels",
    "data/hidden_processed_features", "data/hidden_processed_labels", ".nanofold_private/hidden_processed_features",
    ".nanofold_private/hidden_processed_labels", "leaderboard", "tracks",
)]

def overlaps(left, right):
    return left.is_relative_to(right) or right.is_relative_to(left)

if manifests.is_relative_to(repo / "data/manifests"):
    raise SystemExit("Use staged custom manifests; official assets use setup_official_data.sh.")
for index, output in enumerate(outputs):
    if any(overlaps(output, path.resolve()) for path in [*protected, manifests, *outputs[:index]]):
        raise SystemExit("Custom asset paths must be separate from published paths, manifests and each other.")
    if lock_path.is_relative_to(output) or (filter_path and filter_path.is_relative_to(output)):
        raise SystemExit("Custom asset output paths must not contain source inputs.")
    if output.exists() and not output.is_dir():
        raise SystemExit("Custom asset output paths must be directories.")
if not lock_path.is_file() or (filter_path and not filter_path.is_file()):
    raise SystemExit("The source lock and optional MSA filter must already exist.")
lock = json.loads(lock_path.read_text())
records = lock.get("outputs", {})
if not isinstance(records, dict):
    raise SystemExit("Candidate source lock outputs must be an object.")
ids = {}
for split, count_key in (("train", "train_count"), ("val", "val_count"), ("all", "unique_count")):
    path = manifests / f"{split}.txt"
    if not path.is_file():
        raise SystemExit(f"Missing staged {split} manifest; all.txt is never generated here.")
    if path.resolve().is_relative_to(repo / "data/manifests"):
        raise SystemExit("Staged manifests must not alias published manifests.")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if records.get(f"{split}_manifest_sha256") != digest:
        raise SystemExit(f"Candidate {split} manifest does not match its source lock.")
    ids[split] = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    if not ids[split] or len(set(ids[split])) != len(ids[split]) or records.get(count_key) != len(ids[split]):
        raise SystemExit(f"Candidate {split} manifest has invalid counts or duplicate IDs.")
if set(ids["train"]) & set(ids["val"]) or set(ids["all"]) != set(ids["train"]) | set(ids["val"]):
    raise SystemExit("Candidate manifests must be disjoint with an exact train/val union.")
print("Verified staged public manifests against their source lock.")
PYTHON

if [[ "$DRY_RUN" -eq 0 ]]; then
  if ! command -v aws >/dev/null 2>&1; then
    echo "aws CLI not found. Install awscli first." >&2
    exit 1
  fi
  if ! python -c "import tqdm" >/dev/null 2>&1; then
    echo "python package 'tqdm' not found. Run: pip install -r requirements.txt" >&2
    exit 1
  fi
fi

run_cmd() {
  echo "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

PDB_DIR="$DATA_ROOT/pdb_data"
run_cmd mkdir -p "$PDB_DIR" "$PROCESSED_FEATURES_DIR" "$PROCESSED_LABELS_DIR"
run_cmd aws s3 cp s3://openfold/duplicate_pdb_chains.txt "$PDB_DIR/" --only-show-errors --no-sign-request
PREPARE_CMD=(
  python "$SCRIPT_DIR/prepare_data.py"
  --data-root "$DATA_ROOT" --manifest "$MANIFESTS_DIR/all.txt"
  --duplicate-chains-file "$PDB_DIR/duplicate_pdb_chains.txt"
  --msa-name "$MSA_NAME" --no-template-hits --download-mmcif-subset --strict-downloads
  --download-retries "$DOWNLOAD_RETRIES" --download-retry-delay-seconds "$DOWNLOAD_RETRY_DELAY_SECONDS"
)
run_cmd "${PREPARE_CMD[@]}"
if [[ "$SKIP_PREPROCESS" -eq 0 ]]; then
  PREPROCESS_COMMON=(
    python "$SCRIPT_DIR/preprocess.py" --raw-root "$DATA_ROOT" --mmcif-root "$PDB_DIR/mmcif_files"
    --processed-features-dir "$PROCESSED_FEATURES_DIR" --processed-labels-dir "$PROCESSED_LABELS_DIR"
    --msa-name "$MSA_NAME" --disable-templates
  )
  if [[ -n "$MSA_ROW_FILTER" ]]; then
    PREPROCESS_COMMON+=(--msa-row-filter "$MSA_ROW_FILTER")
  fi
  if [[ "$RESUME_PREPROCESS" -eq 1 ]]; then
    PREPROCESS_COMMON+=(--skip-existing)
  fi
  run_cmd "${PREPROCESS_COMMON[@]}" --manifest "$MANIFESTS_DIR/train.txt"
  run_cmd "${PREPROCESS_COMMON[@]}" --manifest "$MANIFESTS_DIR/val.txt"
fi

echo "Custom public asset setup complete. Manifests and source lock are unchanged."
echo "Run strict split/homology audits and fingerprint the candidate before any release."
