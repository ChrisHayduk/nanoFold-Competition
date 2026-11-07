#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash run_af3.sh {tiny|medium|large} {train|prefix250|continue|evaluate}" >&2
  exit 2
fi
profile="$1"
phase="$2"
case "$profile" in tiny|medium|large) ;; *) echo "Unknown profile: $profile" >&2; exit 2 ;; esac
case "$phase" in train|prefix250|continue|evaluate) ;; *) echo "Unknown phase: $phase" >&2; exit 2 ;; esac
if [[ "$phase" == prefix250 && "$profile" != large ]]; then
  echo "prefix250 is the recorded large-run launch phase" >&2
  exit 2
fi

root="$(cd "$(dirname "$0")" && pwd)"
runtime="$root/runtime/$profile/protenix-af3-style"
python_bin="${AF3_PYTHON:-python}"
: "${PROTENIX_ROOT:?Set PROTENIX_ROOT to the pinned Protenix checkout}"
: "${PROTENIX_ROOT_DIR:?Set PROTENIX_ROOT_DIR to the verified CCD data root}"
: "${NANOFOLD_DATA_ROOT:?Set NANOFOLD_DATA_ROOT to the public NPZ input tree}"
: "${NANOFOLD_VERIFY_REPORT:?Set NANOFOLD_VERIFY_REPORT to the dataset report}"
: "${NANOFOLD_RUN_DIR:?Set NANOFOLD_RUN_DIR to a local run directory}"

export PYTHONPATH="$PROTENIX_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TRIANGLE_ATTENTION=cuequivariance TRIANGLE_MULTIPLICATIVE=cuequivariance
export LAYERNORM_TYPE=fast_layernorm
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
command=("$python_bin" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8)
if [[ "$phase" == evaluate ]]; then
  : "${NANOFOLD_SCORER_ROOT:?Set NANOFOLD_SCORER_ROOT to the pinned NanoFold scorer checkout}"
  eval_config=eval-final-world8.json
  if [[ "$profile" == large ]]; then eval_config=eval-final.json; fi
  command+=("$runtime/evaluate.py" --config "$runtime/configs/$eval_config"
    --checkpoint "$NANOFOLD_RUN_DIR/checkpoints/completed-step-0030000.pt"
    --training-provenance "$NANOFOLD_RUN_DIR/provenance.json"
    --protenix-root "$PROTENIX_ROOT" --data-root "$NANOFOLD_DATA_ROOT"
    --dataset-verification "$NANOFOLD_VERIFY_REPORT" --nanofold-source-root "$NANOFOLD_SCORER_ROOT"
    --output-dir "$NANOFOLD_RUN_DIR/evaluation-completed-step-0030000")
else
  resume=never
  if [[ "$phase" == continue ]]; then resume=required; fi
  command+=("$runtime/train.py" --protenix-root "$PROTENIX_ROOT"
    --data-root "$NANOFOLD_DATA_ROOT" --dataset-verification-report "$NANOFOLD_VERIFY_REPORT"
    --run-dir "$NANOFOLD_RUN_DIR" --max-steps 30000 --seed 20260725 --crop-size 256
    --msa-depth 192 --checkpoint-interval 250 --log-interval 10 --resume "$resume")
  if [[ "$phase" == prefix250 ]]; then command+=(--smoke-steps 250); fi
fi

if [[ "${AF3_PRINT_COMMAND:-0}" == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
else
  "$python_bin" - "$root/ORIGINAL-SOURCE-BINDINGS.json" "$NANOFOLD_VERIFY_REPORT" <<'PYTHON'
import json
import sys
from pathlib import Path

expected = json.loads(Path(sys.argv[1]).read_text())["public_data_original_snapshot"]
actual = json.loads(Path(sys.argv[2]).read_text())
inventory = {key: expected[key] for key in ("file_count", "total_bytes", "tree_sha256")}
required = ("train_count", "validation_count", "manifest_sha256")
matches_report = all(actual.get(key) == expected[key] for key in required)
if actual.get("inventory") != inventory or not matches_report:
    raise SystemExit("Dataset report differs from the recorded AF3 public tensor snapshot")
PYTHON
  exec "${command[@]}"
fi
