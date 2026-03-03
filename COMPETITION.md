# Competition rules & evaluation protocol

This document is meant to be “runbook-level” concrete, so that runs are reproducible and comparable.

## 1) Allowed data

Competitors **may only use the dataset produced by `scripts/prepare_data.py`**.

They may:
- subsample it
- reweight it
- derive new features from it (e.g., amino-acid counts, MSA profiles, predicted secondary structure)
- change architecture / optimizer / losses
- implement custom training/eval logic inside their submission module

They may not:
- download or use additional sequences, structures, weights, or pretrained models
- run MSA/template search against external databases
- use PDB structures outside the provided dataset

**Enforcement suggestion:** maintainer runs training/eval with network disabled (e.g., Docker `--network none`)
and only mounts the dataset directory read-only.

## 2) Required output interface

A submission must output **Cartesian coordinates** for each residue in the crop.

Minimum requirement:
- Cα coordinates: `(L, 3)` in Ångström units

You may additionally output backbone or all-atom coordinates, but evaluation uses Cα only.

Submission entrypoint requirements (`submissions/<name>/submission.py`):
- `build_model(cfg)`
- `build_optimizer(cfg, model)`
- `run_batch(model, batch, cfg, training)` returning:
  - `pred_ca`: `(B, L, 3)` float tensor
  - `loss`: scalar tensor when `training=True`

## 3) Dataset splits

The dataset is defined by manifests:
- `manifests/train.txt`
- `manifests/val.txt`

These are part of the benchmark. Do not change them in a submission PR.

## 4) Fixed budget (limited track)

A run is defined by:
- `max_steps` (optimizer steps)
- `effective_batch_size` (including gradient accumulation)
- `crop_size`
- RNG seed

The maintainer should record:
- hardware (GPU model, #GPUs)
- wall-clock time
- commit hash

## 5) Scoring

Primary score:
- mean lDDT-Cα on the validation set

Tie-breakers (optional):
- lower wall-clock time
- fewer parameters
- lower violation metrics (if you add them)

## 6) How maintainers should evaluate a PR

1) Checkout the PR commit.
2) Run `python train.py --config submissions/<name>/config.yaml` (or similar).
3) Run `python eval.py --config ... --ckpt ...`.
4) Save `runs/<name>/metrics.json`.
5) Update `leaderboard/leaderboard.json` and run `python scripts/render_leaderboard.py`.
