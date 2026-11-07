#!/usr/bin/env bash
set -euo pipefail

: "${PROTENIX_ROOT:?set PROTENIX_ROOT to the pinned Protenix checkout}"
: "${PROTENIX_ROOT_DIR:?set PROTENIX_ROOT_DIR to Protenix chemical data}"
: "${NANOFOLD_DATA_ROOT:?set NANOFOLD_DATA_ROOT to the verified tensor root}"
: "${NANOFOLD_RUN_ROOT:?set NANOFOLD_RUN_ROOT to a persistent-volume run directory}"
: "${NANOFOLD_VERIFY_REPORT:?set NANOFOLD_VERIFY_REPORT to verify_dataset.py JSON}"

EXPECTED_COMMIT="c3bfc365b3e1341a11935eddfe7bfdc308092147"
ACTUAL_COMMIT="$(git -C "${PROTENIX_ROOT}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "Protenix commit mismatch: ${ACTUAL_COMMIT}" >&2
  exit 2
fi

GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
if [[ "${GPU_COUNT}" != "8" ]]; then
  echo "Expected exactly 8 visible GPUs; found ${GPU_COUNT}" >&2
  exit 2
fi

test -f "${NANOFOLD_DATA_ROOT}/manifests/train.txt"
test -d "${NANOFOLD_DATA_ROOT}/processed_features"
test -d "${NANOFOLD_DATA_ROOT}/processed_labels"
test -f "${PROTENIX_ROOT_DIR}/common/components.cif.rdkit_mol.pkl"
test -x /workspace/envs/protenix/bin/python
mkdir -p "${NANOFOLD_RUN_ROOT}/production"

export PYTHONPATH="${PROTENIX_ROOT}:${PYTHONPATH:-}"
export PATH="/workspace/envs/protenix/bin:${PATH}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TRIANGLE_ATTENTION="${TRIANGLE_ATTENTION:-cuequivariance}"
export TRIANGLE_MULTIPLICATIVE="${TRIANGLE_MULTIPLICATIVE:-cuequivariance}"

exec /workspace/envs/protenix/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 \
  "$(dirname "$0")/train.py" \
  --protenix-root "${PROTENIX_ROOT}" \
  --data-root "${NANOFOLD_DATA_ROOT}" \
  --dataset-verification-report "${NANOFOLD_VERIFY_REPORT}" \
  --run-dir "${NANOFOLD_RUN_ROOT}/production" \
  --max-steps 30000 \
  --checkpoint-interval 250 \
  --log-interval 10 \
  --crop-size 256 \
  --msa-depth 192 \
  --checkpoint-budget-gib 100 \
  --checkpoint-reserve-gib 20 \
  --resume auto
