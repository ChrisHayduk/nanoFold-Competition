#!/usr/bin/env bash
set -euo pipefail

: "${PROTENIX_ROOT:?set PROTENIX_ROOT to the pinned Protenix checkout}"
: "${PROTENIX_ROOT_DIR:?set PROTENIX_ROOT_DIR to Protenix chemical data}"
: "${NANOFOLD_DATA_ROOT:?set NANOFOLD_DATA_ROOT to the verified tensor root}"
: "${NANOFOLD_RUN_ROOT:?set NANOFOLD_RUN_ROOT to a persistent-volume run directory}"
: "${NANOFOLD_VERIFY_REPORT:?set NANOFOLD_VERIFY_REPORT to verify_dataset.py JSON}"

EXPECTED_COMMIT="c3bfc365b3e1341a11935eddfe7bfdc308092147"
test "$(git -C "${PROTENIX_ROOT}" rev-parse HEAD)" = "${EXPECTED_COMMIT}"
test "$(nvidia-smi -L | wc -l | tr -d ' ')" = "8"
test -f "${PROTENIX_ROOT_DIR}/common/components.cif.rdkit_mol.pkl"
test -x /workspace/envs/protenix/bin/python
mkdir -p "${NANOFOLD_RUN_ROOT}/overfit-8chain"

export PYTHONPATH="${PROTENIX_ROOT}:${PYTHONPATH:-}"
export PATH="/workspace/envs/protenix/bin:${PATH}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TRIANGLE_ATTENTION="${TRIANGLE_ATTENTION:-cuequivariance}"
export TRIANGLE_MULTIPLICATIVE="${TRIANGLE_MULTIPLICATIVE:-cuequivariance}"

# This diagnostic intentionally schedules only the first eight chains after the
# complete 10,000-chain tensor tree has been verified. Its checkpoints are
# provenance-marked claim-ineligible. A fixed-seed checkpoint-0/final evaluator
# must still establish the overfit pass criterion.
exec /workspace/envs/protenix/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 \
  "$(dirname "$0")/train.py" \
  --protenix-root "${PROTENIX_ROOT}" \
  --data-root "${NANOFOLD_DATA_ROOT}" \
  --dataset-verification-report "${NANOFOLD_VERIFY_REPORT}" \
  --run-dir "${NANOFOLD_RUN_ROOT}/overfit-8chain" \
  --max-steps "${NANOFOLD_OVERFIT_STEPS:-200}" \
  --checkpoint-interval 25 \
  --log-interval 1 \
  --crop-size 64 \
  --msa-depth 32 \
  --train-chain-limit 8 \
  --save-initial-checkpoint \
  --checkpoint-budget-gib 100 \
  --checkpoint-reserve-gib 20 \
  --resume auto
