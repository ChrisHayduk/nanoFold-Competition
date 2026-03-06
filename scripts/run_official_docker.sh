#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-nanofold-official-runner}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

echo "+ docker build -f Dockerfile.official -t $IMAGE_NAME $REPO_ROOT"
docker build -f "$REPO_ROOT/Dockerfile.official" -t "$IMAGE_NAME" "$REPO_ROOT"

DOCKER_ARGS=(
  --rm
  --network=none
  --cap-drop=ALL
  -e NANOFOLD_OFFICIAL_SEALED_RUNTIME=1
  -v "$REPO_ROOT:/workspace"
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
  local -n args_ref="$2"
  local value="${!env_key:-}"
  if [[ -z "$value" ]]; then
    return 0
  fi
  if [[ ! -e "$value" ]]; then
    echo "ERROR: $env_key points to a missing path: $value"
    exit 1
  fi
  args_ref+=(-e "$env_key=$value")
  case "$value" in
    "$REPO_ROOT"/*) ;;
    *)
      args_ref+=(-v "$value:$value:ro")
      ;;
  esac
}

run_stage() {
  local -n stage_args_ref="$1"
  shift
  echo "+ docker run ${stage_args_ref[*]} $IMAGE_NAME $*"
  docker run "${stage_args_ref[@]}" "$IMAGE_NAME" "$@"
}

PREDICT_ARGS=("${DOCKER_ARGS[@]}")
add_hidden_mount NANOFOLD_HIDDEN_MANIFEST PREDICT_ARGS
add_hidden_mount NANOFOLD_HIDDEN_FEATURES_DIR PREDICT_ARGS
add_hidden_mount NANOFOLD_HIDDEN_FINGERPRINT PREDICT_ARGS

run_stage PREDICT_ARGS "$@" --skip-hidden-scoring

if has_flag --disable-hidden "$@"; then
  exit 0
fi

SCORE_ARGS=("${DOCKER_ARGS[@]}")
add_hidden_mount NANOFOLD_HIDDEN_MANIFEST SCORE_ARGS
add_hidden_mount NANOFOLD_HIDDEN_FEATURES_DIR SCORE_ARGS
add_hidden_mount NANOFOLD_HIDDEN_LABELS_DIR SCORE_ARGS
add_hidden_mount NANOFOLD_HIDDEN_FINGERPRINT SCORE_ARGS

run_stage SCORE_ARGS "$@" --score-hidden-only --skip-train
