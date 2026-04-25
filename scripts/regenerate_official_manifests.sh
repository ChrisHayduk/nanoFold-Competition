#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/regenerate_official_manifests.sh [options]

Regenerates official manifests using locked args, locked chain cache SHA256,
and required structure metadata from:
  leaderboard/official_manifest_source.lock.json

Then verifies generated manifest SHA256 digests match:
  tracks/limited_large.yaml

Options:
  --chain-data-cache <path>   Path to chain_data_cache.json
                              (default: data/openproteinset/pdb_data/data_caches/chain_data_cache.json)
  --out-dir <path>            Manifest output directory (default: data/manifests)
  --private-root <path>        Maintainer-only hidden asset root (default: .nanofold_private)
  --hidden-out-dir <path>      Hidden manifest output directory
                              (default: <private-root>/manifests)
  --lock-file <path>          Lock metadata path (default: leaderboard/official_manifest_source.lock.json)
  --private-lock-file <path>  Maintainer-only hidden manifest lock output
                              (default: <private-root>/leaderboard/private_hidden_manifest_source.lock.json)
  --structure-metadata <path> Structure metadata JSON override
  --rewrite-lock              Rewrite lock metadata at --lock-file after successful generation
  --sync-hashes               Sync manifest SHA256/count references across track + docs + lock
  --dry-run                   Print commands without executing
  -h, --help                  Show this message
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

CHAIN_DATA_CACHE="data/openproteinset/pdb_data/data_caches/chain_data_cache.json"
OUT_DIR="data/manifests"
PRIVATE_ROOT=".nanofold_private"
HIDDEN_OUT_DIR=""
LOCK_FILE="leaderboard/official_manifest_source.lock.json"
PRIVATE_LOCK_FILE=""
STRUCTURE_METADATA_OVERRIDE=""
REWRITE_LOCK=0
SYNC_HASHES=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --chain-data-cache)
      CHAIN_DATA_CACHE="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --private-root)
      PRIVATE_ROOT="$2"
      shift 2
      ;;
    --hidden-out-dir|--hidden-manifests-dir)
      HIDDEN_OUT_DIR="$2"
      shift 2
      ;;
    --lock-file)
      LOCK_FILE="$2"
      shift 2
      ;;
    --private-lock-file)
      PRIVATE_LOCK_FILE="$2"
      shift 2
      ;;
    --structure-metadata)
      STRUCTURE_METADATA_OVERRIDE="$2"
      shift 2
      ;;
    --rewrite-lock)
      REWRITE_LOCK=1
      shift 1
      ;;
    --sync-hashes)
      SYNC_HASHES=1
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

if [[ -z "$HIDDEN_OUT_DIR" ]]; then
  HIDDEN_OUT_DIR="$PRIVATE_ROOT/manifests"
fi
if [[ -z "$PRIVATE_LOCK_FILE" ]]; then
  PRIVATE_LOCK_FILE="$PRIVATE_ROOT/leaderboard/private_hidden_manifest_source.lock.json"
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

if [[ ! -f "$LOCK_FILE" ]]; then
  echo "Lock file not found: $LOCK_FILE"
  exit 1
fi

eval "$(
  python - "$LOCK_FILE" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

lock = json.loads(Path(sys.argv[1]).read_text())
args = lock.get("args", {})
required = ("train_size", "val_size", "hidden_val_size", "min_len", "max_len", "max_resolution", "seed", "structure_metadata")
for key in required:
    if key not in args:
        raise SystemExit(f"Lock file missing args.{key}")
cache_sha = str(lock.get("chain_data_cache_sha256", "")).strip().lower()
if len(cache_sha) != 64 or any(ch not in "0123456789abcdef" for ch in cache_sha):
    raise SystemExit("Lock file has invalid chain_data_cache_sha256")
min_seq_id = float(args.get("min_seq_id", 0.30))
coverage = float(args.get("coverage", 0.80))
mmseqs_bin = str(args.get("mmseqs_bin", "mmseqs")).strip() or "mmseqs"
hidden_val_size = int(args.get("hidden_val_size", 0))
cluster_tsv = str(args.get("cluster_tsv") or "").strip()
structure_metadata = str(args.get("structure_metadata") or "").strip()
if hidden_val_size <= 0:
    raise SystemExit("Lock file args.hidden_val_size must be positive")
if not structure_metadata:
    raise SystemExit("Lock file missing args.structure_metadata")
print(f"TRAIN_SIZE={int(args['train_size'])}")
print(f"VAL_SIZE={int(args['val_size'])}")
print(f"HIDDEN_VAL_SIZE={hidden_val_size}")
print(f"MIN_LEN={int(args['min_len'])}")
print(f"MAX_LEN={int(args['max_len'])}")
print(f"MAX_RESOLUTION={float(args['max_resolution'])}")
print(f"SEED={int(args['seed'])}")
print(f"MIN_SEQ_ID={min_seq_id}")
print(f"COVERAGE={coverage}")
print(f"MMSEQS_BIN={mmseqs_bin}")
print(f"CLUSTER_TSV={cluster_tsv}")
print(f"STRUCTURE_METADATA={structure_metadata}")
print(f"EXPECTED_CACHE_SHA={cache_sha}")
PY
)"

if [[ -n "$STRUCTURE_METADATA_OVERRIDE" ]]; then
  STRUCTURE_METADATA="$STRUCTURE_METADATA_OVERRIDE"
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
  if [[ -z "${NANOFOLD_HIDDEN_SPLIT_SALT:-}" ]]; then
    echo "NANOFOLD_HIDDEN_SPLIT_SALT is required for official hidden manifest generation."
    exit 1
  fi
  if [[ "${#NANOFOLD_HIDDEN_SPLIT_SALT}" -lt 32 ]]; then
    echo "NANOFOLD_HIDDEN_SPLIT_SALT must be at least 32 characters."
    exit 1
  fi
fi

BUILD_CMD=(
  python "$SCRIPT_DIR/build_manifests.py"
  --chain-data-cache "$CHAIN_DATA_CACHE"
  --out-dir "$OUT_DIR"
  --hidden-out-dir "$HIDDEN_OUT_DIR"
  --train-size "$TRAIN_SIZE"
  --val-size "$VAL_SIZE"
  --hidden-val-size "$HIDDEN_VAL_SIZE"
  --min-len "$MIN_LEN"
  --max-len "$MAX_LEN"
  --max-resolution "$MAX_RESOLUTION"
  --seed "$SEED"
  --min-seq-id "$MIN_SEQ_ID"
  --coverage "$COVERAGE"
  --mmseqs-bin "$MMSEQS_BIN"
  --structure-metadata "$STRUCTURE_METADATA"
  --expected-chain-cache-sha256 "$EXPECTED_CACHE_SHA"
)
if [[ -n "$CLUSTER_TSV" ]]; then
  BUILD_CMD+=(--cluster-tsv "$CLUSTER_TSV")
fi
if [[ "$REWRITE_LOCK" -eq 1 ]]; then
  BUILD_CMD+=(--lock-file "$LOCK_FILE")
  BUILD_CMD+=(--private-lock-file "$PRIVATE_LOCK_FILE")
fi

run_cmd "${BUILD_CMD[@]}"

if [[ "$SYNC_HASHES" -eq 1 ]]; then
  run_cmd python "$SCRIPT_DIR/sync_official_manifest_hashes.py" \
    --manifests-dir "$OUT_DIR" \
    --hidden-manifest "$HIDDEN_OUT_DIR/hidden_val.txt" \
    --track-file "$REPO_ROOT/tracks/limited_large.yaml" \
    --lock-file "$LOCK_FILE" \
    --readme "$REPO_ROOT/README.md" \
    --competition-doc "$REPO_ROOT/docs/COMPETITION.md"
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
  python - "$REPO_ROOT" "$OUT_DIR" <<'PY'
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from nanofold.competition_policy import load_track_spec


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


out_dir = Path(sys.argv[2]).resolve()
track = load_track_spec("limited_large")

checks = [
    ("train_manifest", out_dir / "train.txt", track.train_manifest_sha256),
    ("val_manifest", out_dir / "val.txt", track.val_manifest_sha256),
    ("all_manifest", out_dir / "all.txt", track.all_manifest_sha256),
]
for label, path, expected in checks:
    if expected is None:
        continue
    if not path.exists():
        raise SystemExit(f"Missing expected manifest file: {path}")
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(
            f"{label} hash mismatch:\n"
            f"  path:     {path}\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}"
        )
print("Official manifest regeneration matched pinned hashes in track policy.")
PY
fi
