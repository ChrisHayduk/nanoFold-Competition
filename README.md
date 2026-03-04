# nanoFold Data Efficiency Competition

A minimalist, reproducible benchmark for **protein folding algorithms in the fixed-data regime**.

This repo is inspired by:
- `slowrun` (fixed data, unlimited compute) and `modded-nanogpt` (fixed compute) style competitions, but
- adapted to **structure prediction**, where submissions must output **3D coordinates**.

The intent is to make it easy to:
1) download & preprocess a *fixed* dataset (sequence + MSA + templates + labels),
2) run a **standardized** training budget,
3) score predictions with a **single shared evaluator**,
4) maintain a transparent leaderboard.

## Task

Given an input protein (sequence + its MSA + optional templates), predict a 3D structure.

**Primary metric:** mean lDDT-Cα on the validation set (higher is better).

## Official Limited Track (Large-v3)

Official fixed constants:
- `seed = 0`
- `data.crop_size = 256`
- `data.msa_depth = 192`
- `effective_batch_size = data.batch_size * train.grad_accum_steps = 2`
- `train.max_steps = 10000`
- `data.val_crop_mode = center`
- `data.val_msa_sample_mode = top`

Derived data budget:
- `B_res = max_steps * effective_batch_size * crop_size = 5,120,000`

Canonical maintainer config:
- `configs/official_limited_large_v3.yaml`

## Quickstart

```bash
# 1) create environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) one-command subset setup (recommended for participants)
# - downloads data caches
# - builds train/val manifests
# - downloads per-chain uniref90 MSA + pdb70 template hits
# - downloads + unpacks pdb_mmcif.zip
# - preprocesses into data/processed/*.npz
bash scripts/setup_competition_data.sh \
  --data-root data/openproteinset \
  --train-size 10000 \
  --val-size 500 \
  --download-retries 5 \
  --download-retry-delay-seconds 3 \
  --seed 0
# for a quick local smoke run, start with:
# bash scripts/setup_competition_data.sh --train-size 1000 --val-size 100

# 3) train baseline
python train.py --config configs/baseline.yaml

# 4) evaluate
python eval.py --config configs/baseline.yaml --ckpt runs/baseline/checkpoints/ckpt_last.pt

# 5) official-mode baseline run (strict policy + dataset fingerprint)
python scripts/build_dataset_fingerprint.py \
  --config configs/official_limited_large_v3.yaml \
  --output leaderboard/official_dataset_fingerprint.json
# Note: this command fails if any manifest chain is missing from data/processed.
python train.py --config configs/official_limited_large_v3.yaml --official
python eval.py --config configs/official_limited_large_v3.yaml \
  --ckpt runs/official_limited_large_v3_baseline/checkpoints/ckpt_last.pt --official
```

## Data Flow (Detailed)

Use this if you want explicit control of setup/resume behavior.

1. Run full subset setup (safe to rerun if interrupted):
```bash
bash scripts/setup_competition_data.sh \
  --data-root data/openproteinset \
  --train-size 10000 \
  --val-size 500 \
  --seed 0 \
  --download-retries 5 \
  --download-retry-delay-seconds 3
```

2. If setup fails at step `[6/6]` (preprocessing), rerun only preprocessing:
```bash
python scripts/preprocess.py \
  --raw-root data/openproteinset \
  --mmcif-root data/openproteinset/pdb_data/mmcif_files \
  --processed-dir data/processed \
  --msa-name uniref90_hits.a3m \
  --template-hhr-name pdb70_hits.hhr \
  --manifest data/manifests/train.txt

python scripts/preprocess.py \
  --raw-root data/openproteinset \
  --mmcif-root data/openproteinset/pdb_data/mmcif_files \
  --processed-dir data/processed \
  --msa-name uniref90_hits.a3m \
  --template-hhr-name pdb70_hits.hhr \
  --manifest data/manifests/val.txt
```

3. If you regenerate manifests, redownload only those manifest chains:
```bash
cat data/manifests/train.txt data/manifests/val.txt | sort -u > data/manifests/all.txt
python scripts/prepare_data.py \
  --data-root data/openproteinset \
  --manifest data/manifests/all.txt \
  --duplicate-chains-file data/openproteinset/pdb_data/duplicate_pdb_chains.txt \
  --msa-name uniref90_hits.a3m \
  --template-hits-name pdb70_hits.hhr \
  --download-retries 5 \
  --download-retry-delay-seconds 3
```

4. Optional quick health checks:
```bash
ls data/processed/*.npz | wc -l
ls data/processed/*.error.txt | wc -l
```

## Seed Runs

```bash
# Seed ESMFold-style model
python train.py --config submissions/seed_esmfold/config.yaml
python eval.py --config submissions/seed_esmfold/config.yaml \
  --ckpt runs/seed_esmfold_v1/checkpoints/ckpt_last.pt

# Seed OpenFold-style model (uses MSA + template features)
python train.py --config submissions/seed_openfold/config.yaml
python eval.py --config submissions/seed_openfold/config.yaml \
  --ckpt runs/seed_openfold_v1/checkpoints/ckpt_last.pt
```

## OpenProteinSet Setup Notes

- Recommended path for participants: `scripts/setup_competition_data.sh`
  - avoids the full multi-terabyte mirror
  - samples train/val manifests with protein-disjoint splits (no PDB appears in both)
  - downloads only manifest-selected chains
  - includes `uniref90_hits.a3m` and `pdb70_hits.hhr` (template hits)
  - automatically resolves duplicate-chain IDs using `duplicate_pdb_chains.txt`
  - downloads mmCIF structures from `pdb_mmcif.zip` for targets/templates
- Full canonical mirror path is still available via `scripts/setup_openproteinset_roda.sh` (maintainer/HPC use).
- Optional OpenFold steps like alignment DB shards and MMSeqs cluster-file generation are only required for upstream `train_openfold.py`; they are not required for this benchmark's `train.py`.

## Submitting

Open a PR that adds:
- a new folder under `submissions/<your_name>/`
- a `config.yaml`
- a `submission.py` entrypoint implementing:
  - `build_model(cfg)`
  - `build_optimizer(cfg, model)`
  - `run_batch(model, batch, cfg, training)` returning:
    - always `pred_ca`
    - `loss` when `training=True`
    - no dependence on `ca_coords` / `ca_mask` when `training=False` (official eval strips labels)
- any additional code files your method needs (`model.py`, `loss.py`, etc.)
- a short `notes.md` describing what changed and why

In `config.yaml`, point the runner at your entrypoint:

```yaml
submission:
  path: submission.py
```

Before opening the PR, run:

```bash
python scripts/validate_submission.py --submission submissions/<your_name>
```

Maintainers run the standardized training/eval and update the leaderboard.

## Leaderboard

<!-- LEADERBOARD_START -->
| # | Score (lDDT-Cα) | Track | Date | Commit | Description |
|---:|---:|---|---|---|---|
| 1 | 0.0000 | limited | 2026-03-03 | `seedesm` | Seed submission: ESMFold-inspired trunk (pending official benchmark run) |
| 2 | 0.0000 | limited | 2026-03-03 | `seedope` | Seed submission: OpenFold-style Evoformer + template stack (pending official benchmark run) |
<!-- LEADERBOARD_END -->
