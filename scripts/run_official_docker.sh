#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-nanofold-official-runner}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PRIVATE_ROOT_HOST="${NANOFOLD_PRIVATE_ROOT:-$REPO_ROOT/.nanofold_private}"

resolve_host_path() {
  python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$1"
}

PRIVATE_ROOT_HOST="$(resolve_host_path "$PRIVATE_ROOT_HOST")"
NANOFOLD_HIDDEN_MANIFEST="$(resolve_host_path "${NANOFOLD_HIDDEN_MANIFEST:-$PRIVATE_ROOT_HOST/manifests/hidden_val.txt}")"
NANOFOLD_HIDDEN_FEATURES_DIR="$(resolve_host_path "${NANOFOLD_HIDDEN_FEATURES_DIR:-$PRIVATE_ROOT_HOST/hidden_processed_features}")"
NANOFOLD_HIDDEN_LABELS_DIR="$(resolve_host_path "${NANOFOLD_HIDDEN_LABELS_DIR:-$PRIVATE_ROOT_HOST/hidden_processed_labels}")"
NANOFOLD_HIDDEN_FINGERPRINT="$(resolve_host_path "${NANOFOLD_HIDDEN_FINGERPRINT:-$PRIVATE_ROOT_HOST/leaderboard/official_hidden_fingerprint.json}")"
NANOFOLD_HIDDEN_LOCK_FILE="$(resolve_host_path "${NANOFOLD_HIDDEN_LOCK_FILE:-$PRIVATE_ROOT_HOST/leaderboard/private_hidden_assets.lock.json}")"
export NANOFOLD_HIDDEN_MANIFEST
export NANOFOLD_HIDDEN_FEATURES_DIR
export NANOFOLD_HIDDEN_LABELS_DIR
export NANOFOLD_HIDDEN_FINGERPRINT
export NANOFOLD_HIDDEN_LOCK_FILE

PRIVATE_MASK_DIR="$(mktemp -d)"
trap 'rm -rf "$PRIVATE_MASK_DIR"' EXIT

echo "+ docker build -f Dockerfile.official -t $IMAGE_NAME $REPO_ROOT"
docker build -f "$REPO_ROOT/Dockerfile.official" -t "$IMAGE_NAME" "$REPO_ROOT"

DOCKER_ARGS=(
  --rm
  --network=none
  --cap-drop=ALL
  -e NANOFOLD_OFFICIAL_SEALED_RUNTIME=1
  -v "$REPO_ROOT:/workspace"
  -v "$PRIVATE_MASK_DIR:/workspace/.nanofold_private:ro"
  -w /workspace
)

has_flag() {
  local needle="$1"
  shift
  for token in "$@"; do
    if [[ "$token" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

add_hidden_mount() {
  local env_key="$1"
  local container_path="$2"
  local -n args_ref="$3"
  local value="${!env_key:-}"
  if [[ -z "$value" ]]; then
    return 0
  fi
  if [[ ! -e "$value" ]]; then
    echo "ERROR: $env_key points to a missing path: $value"
    exit 1
  fi
  args_ref+=(-e "$env_key=$container_path")
  args_ref+=(-v "$value:$container_path:ro")
}

run_stage() {
  local -n stage_args_ref="$1"
  shift
  echo "+ docker run ${stage_args_ref[*]} $IMAGE_NAME $*"
  docker run "${stage_args_ref[@]}" "$IMAGE_NAME" "$@"
}

PREDICT_ARGS=("${DOCKER_ARGS[@]}")
if ! has_flag --disable-hidden "$@"; then
  add_hidden_mount NANOFOLD_HIDDEN_MANIFEST /workspace/.nanofold_hidden_manifest.txt PREDICT_ARGS
  add_hidden_mount NANOFOLD_HIDDEN_FEATURES_DIR /workspace/.nanofold_hidden_features PREDICT_ARGS
  add_hidden_mount NANOFOLD_HIDDEN_FINGERPRINT /workspace/.nanofold_hidden_fingerprint.json PREDICT_ARGS
fi

run_stage PREDICT_ARGS "$@" --skip-hidden-scoring

if has_flag --disable-hidden "$@"; then
  exit 0
fi

SCORE_ARGS=("${DOCKER_ARGS[@]}")
add_hidden_mount NANOFOLD_HIDDEN_MANIFEST /workspace/.nanofold_hidden_manifest.txt SCORE_ARGS
add_hidden_mount NANOFOLD_HIDDEN_FEATURES_DIR /workspace/.nanofold_hidden_features SCORE_ARGS
add_hidden_mount NANOFOLD_HIDDEN_LABELS_DIR /workspace/.nanofold_hidden_labels SCORE_ARGS
add_hidden_mount NANOFOLD_HIDDEN_FINGERPRINT /workspace/.nanofold_hidden_fingerprint.json SCORE_ARGS
add_hidden_mount NANOFOLD_HIDDEN_LOCK_FILE /workspace/.nanofold_hidden_assets.lock.json SCORE_ARGS

run_stage SCORE_ARGS "$@" --score-hidden-only --skip-train
