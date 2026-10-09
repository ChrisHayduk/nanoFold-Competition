#!/bin/bash
#SBATCH --job-name=nanofold_v1_eval
#SBATCH --partition=dgxnp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# Ensure SLURM binaries and environment are loaded
export PATH=/opt/slurm-20.11.7/bin:$PATH
cd /nlsasfs/home/oncolytic/saruna/ppi_tools/.experiment
source .venv/bin/activate

mkdir -p logs
mkdir -p runs/nanofold_v1/_forbid_labels
mkdir -p runs/nanofold_v1/public_predictions

echo "=== [SLURM] Starting NanoFold v1 Sealed Validation Inference ==="
echo "Start time: $(date)"

python predict.py \
  --config submissions/nanofold_v1/config.yaml \
  --ckpt runs/nanofold_v1/checkpoints/ckpt_last.pt \
  --split val \
  --track limited \
  --official \
  --forbid-labels-dir runs/nanofold_v1/_forbid_labels \
  --pred-out-dir runs/nanofold_v1/public_predictions \
  --save runs/nanofold_v1/predict_val.json

echo "=== [SLURM] Starting CASP15 FoldScore Scoring ==="
python score.py \
  --prediction-summary runs/nanofold_v1/predict_val.json \
  --labels-dir data/processed_labels \
  --save runs/nanofold_v1/eval_val.json \
  --per-chain-out runs/nanofold_v1/per_chain_scores_val.jsonl

echo "=== [SLURM] Evaluation Finished ==="
echo "Results written to: runs/nanofold_v1/eval_val.json"
echo "End time: $(date)"
