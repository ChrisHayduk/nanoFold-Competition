# Competition rules & evaluation protocol

This document defines the official benchmark contract used for leaderboard runs.

## 1) Allowed data

Competitors may only use data produced by `scripts/prepare_data.py` + `scripts/preprocess.py`.

Allowed:
- subsampling / reweighting / curriculum over the provided data
- feature engineering derived from provided data
- architecture, optimizer, scheduler, and loss changes

Not allowed:
- external sequences, structures, weights, or pretrained models
- external MSA/template searches
- PDB structures outside the provided dataset

## 2) Required submission interface

Entrypoint (`submissions/<name>/submission.py`) must define:
- `build_model(cfg)`
- `build_optimizer(cfg, model)`
- `run_batch(model, batch, cfg, training)`

`run_batch` requirements:
- always return `pred_ca` with shape `(B, L, 3)` and floating dtype
- when `training=True`, must also return scalar tensor `loss`
- when `training=False`, official runner passes an **unlabeled batch** (no `ca_coords` / `ca_mask`)

## 3) Fixed dataset splits

Official manifests are fixed:
- `data/manifests/train.txt`
- `data/manifests/val.txt`

Submissions must keep these paths unchanged.

## 4) Official limited track (Large-v3)

Official constants:
- `seed = 0`
- `data.crop_size = 256`
- `data.msa_depth = 192`
- `effective_batch_size = data.batch_size * train.grad_accum_steps = 2`
- `train.max_steps = 10000`
- `data.val_crop_mode = "center"`
- `data.val_msa_sample_mode = "top"`

Derived fixed data budget:
- `B_res = max_steps * effective_batch_size * crop_size = 5,120,000`

## 5) Dataset integrity requirements (official runs)

Official runs require:
- canonical fingerprint JSON (default path: `leaderboard/official_dataset_fingerprint.json`)
- zero missing manifest chains (`allow_missing=False`)

Fingerprint tooling:
- build/update: `python scripts/build_dataset_fingerprint.py --config <config> --output leaderboard/official_dataset_fingerprint.json`
- official train/eval verify this fingerprint when `--official` is enabled

## 6) Scoring

Primary score:
- mean lDDT-Cα on validation set

Tie-breaker (optional):
- lower wall-clock time

## 7) Maintainer runbook for official evaluation

1. Checkout PR commit.
2. Validate submission:
   `python scripts/validate_submission.py --submission submissions/<name>`
3. Train (official mode):
   `python train.py --config submissions/<name>/config.yaml --official`
4. Evaluate (official mode):
   `python eval.py --config submissions/<name>/config.yaml --ckpt <ckpt> --official`
5. Save `runs/<name>/metrics.json`.
6. Update `leaderboard/leaderboard.json` and run `python scripts/render_leaderboard.py`.
