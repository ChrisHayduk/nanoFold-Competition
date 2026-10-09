#!/bin/bash
#SBATCH --job-name=nanofold_data
#SBATCH --partition=dgxnp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/data_setup_%j.out
#SBATCH --error=logs/data_setup_%j.err

# Ensure SLURM binaries and environment are loaded
export PATH=/opt/slurm-20.11.7/bin:$PATH
cd /nlsasfs/home/oncolytic/saruna/ppi_tools/.experiment
source .venv/bin/activate

mkdir -p logs

echo "=== [SLURM] Starting nanoFold Data Download & Preprocessing ==="
echo "Node: $(hostname)"
echo "Start time: $(date)"

bash scripts/setup_official_data.sh \
  --data-root data/openproteinset \
  --processed-features-dir data/processed_features \
  --processed-labels-dir data/processed_labels \
  --download-workers 16 \
  --mmcif-mode subset \
  --disable-templates

echo "=== [SLURM] Data Download & Preprocessing Complete ==="
echo "End time: $(date)"
