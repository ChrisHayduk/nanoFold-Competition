#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-nanofold-official-runner}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

echo "+ docker build -f Dockerfile.official -t $IMAGE_NAME $REPO_ROOT"
docker build -f "$REPO_ROOT/Dockerfile.official" -t "$IMAGE_NAME" "$REPO_ROOT"

echo "+ docker run --rm --network=none -v $REPO_ROOT:/workspace -w /workspace $IMAGE_NAME $*"
docker run --rm --network=none -v "$REPO_ROOT:/workspace" -w /workspace "$IMAGE_NAME" "$@"
