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

## Quickstart

```bash
# 1) create environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) canonical OpenProteinSet setup from RODA (requires awscli + unzip)
bash scripts/setup_openproteinset_roda.sh data/openproteinset

# 3) create fixed manifests from chain_data_cache.json
python scripts/make_manifest.py \
  --chain-data-cache data/openproteinset/pdb_data/data_caches/chain_data_cache.json \
  --out-dir data/manifests \
  --seed 0

# 4) preprocess train/val into compact .npz files
python scripts/preprocess.py \
  --alignments-root data/openproteinset/alignment_data/alignments \
  --mmcif-root data/openproteinset/pdb_data/mmcif_files \
  --manifest data/manifests/train.txt \
  --processed-dir data/processed
python scripts/preprocess.py \
  --alignments-root data/openproteinset/alignment_data/alignments \
  --mmcif-root data/openproteinset/pdb_data/mmcif_files \
  --manifest data/manifests/val.txt \
  --processed-dir data/processed

# 5) train baseline
python train.py --config configs/baseline.yaml

# 6) evaluate
python eval.py --config configs/baseline.yaml --ckpt runs/baseline/checkpoints/ckpt_last.pt
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

- The setup script follows the official OpenFold RODA flow:
  - downloads `s3://openfold/pdb/` alignments + `pdb_mmcif.zip`
  - runs local `scripts/flatten_roda.sh`
  - expands duplicates with local `scripts/expand_alignment_duplicates.py`
  - downloads `data_caches/` (including `chain_data_cache.json`)
- Optional OpenFold steps like alignment DB shards and MMSeqs cluster-file generation are only required for training with upstream `train_openfold.py`; they are not required for this benchmark's `train.py`.
- For this benchmark, preprocessing currently expects the flattened **alignment directory** format (`alignment_data/alignments`), not alignment DB shards.
- `scripts/prepare_data.py` is still available as a minimal downloader, but the canonical path is `scripts/setup_openproteinset_roda.sh`.
- `scripts/setup_openproteinset_roda.sh` also checks for `python` + `tqdm` because duplicate expansion runs `scripts/expand_alignment_duplicates.py`.

## Submitting

Open a PR that adds:
- a new folder under `submissions/<your_name>/`
- a `config.yaml`
- a `submission.py` entrypoint implementing:
  - `build_model(cfg)`
  - `build_optimizer(cfg, model)`
  - `run_batch(model, batch, cfg, training)` returning at least `pred_ca` and (when training) `loss`
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
