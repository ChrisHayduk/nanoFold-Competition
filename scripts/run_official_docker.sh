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
  -v "$REPO_ROOT:/workspace"
  -w /workspace
)

for ENV_KEY in \
  NANOFOLD_HIDDEN_MANIFEST \
  NANOFOLD_HIDDEN_FEATURES_DIR \
  NANOFOLD_HIDDEN_LABELS_DIR \
  NANOFOLD_HIDDEN_FINGERPRINT
do
  VALUE="${!ENV_KEY:-}"
  if [[ -z "$VALUE" ]]; then
    continue
  fi
  if [[ ! -e "$VALUE" ]]; then
    echo "ERROR: $ENV_KEY points to a missing path: $VALUE"
    exit 1
  fi
  # Pass through env var for scripts/run_official.py path resolution.
  DOCKER_ARGS+=(-e "$ENV_KEY=$VALUE")

  # Mount hidden assets read-only unless already under the repo mount.
  case "$VALUE" in
    "$REPO_ROOT"/*) ;;
    *)
      DOCKER_ARGS+=(-v "$VALUE:$VALUE:ro")
      ;;
  esac
done

echo "+ docker run ${DOCKER_ARGS[*]} $IMAGE_NAME $*"
docker run "${DOCKER_ARGS[@]}" "$IMAGE_NAME" "$@"
