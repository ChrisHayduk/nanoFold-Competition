#!/bin/bash
#SBATCH --job-name=nanofold_v1_train
#SBATCH --partition=dgxnp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Ensure SLURM binaries and environment are loaded
export PATH=/opt/slurm-20.11.7/bin:$PATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /nlsasfs/home/oncolytic/saruna/ppi_tools/.experiment
source .venv/bin/activate

mkdir -p logs

echo "=== [SLURM] Starting NanoFold v1 Official Training ==="
echo "Node: $(hostname)"
echo "GPU assigned:"
nvidia-smi --query-gpu=name,index,memory.total --format=csv
echo "Start time: $(date)"

python train.py \
  --config submissions/nanofold_v1/config.yaml \
  --track limited \
  --official \
  --reset-run

echo "=== [SLURM] NanoFold v1 Training Finished ==="
echo "End time: $(date)"
