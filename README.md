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

## Tracks

- **Limited compute (default):** fixed number of optimization steps + fixed sample budget (recommended for fairness).
- **Unlimited compute (optional):** no compute cap, fixed data (research toy track).

(Tracks are just conventions; you can keep only one if you prefer.)

## Quickstart

```bash
# 1) create environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) download / prepare data (set --manifest and data paths for your setup)
python scripts/prepare_data.py --help

# 3) preprocess into a compact .npz format
# (includes optional template feature extraction from pdb70_hits.hhr by default)
python scripts/preprocess.py --help

# 4) train baseline
python train.py --config configs/baseline.yaml

# 5) evaluate
python eval.py --config configs/baseline.yaml --ckpt runs/baseline/checkpoints/ckpt_last.pt
```

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
