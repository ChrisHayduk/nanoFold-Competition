# Competition Rules and Official Protocol

This file defines the enforceable contract for official leaderboard runs.

## 1) Allowed and Disallowed Data

Allowed inputs are limited to data produced by this repo's preprocessing pipeline:
- `scripts/prepare_data.py`
- `scripts/preprocess.py`

Allowed:
- architecture, optimizer, scheduler, and loss changes
- sampling/reweighting/curriculum over provided benchmark data
- feature engineering derived from provided benchmark data

Disallowed:
- external sequences, structures, pretrained weights, or checkpoints
- external MSA/template searches
- network downloads during official execution

## 2) Submission Interface Contract

Submission entrypoint (`submissions/<name>/submission.py`) must implement:
- `build_model(cfg)`
- `build_optimizer(cfg, model)`
- `run_batch(model, batch, cfg, training)`

`run_batch` requirements:
- always returns `pred_ca` with shape `(B, L, 3)` and floating dtype
- returns scalar `loss` when `training=True`
- supports unlabeled inference when `training=False`

Validation and runtime checks are enforced by:
- `scripts/validate_submission.py`
- `nanofold/submission_runtime.py`

Full interface details are documented in [API.md](API.md).

## 3) Official Track Definition

Official track id: `limited_large_v3`

Source of truth:
- `tracks/limited_large_v3.yaml`

Required manifest paths:
- `data/manifests/train.txt`
- `data/manifests/val.txt`

Fixed constants:
- `seed = 0`
- `data.crop_size = 256`
- `data.msa_depth = 192`
- `effective_batch_size = 2`
- `train.max_steps = 10,000`
- `data.val_crop_mode = center`
- `data.val_msa_sample_mode = top`

Split sizes:
- train: `10,000` chains
- val: `1,000` chains

Derived budgets:
- sample budget `B_sample = 20,000`
- residue budget `B_res = 5,120,000`

## 4) Dataset Integrity Requirements (Official)

Official runs require:
- dataset fingerprint match against `leaderboard/official_dataset_fingerprint.json`
- strict no-missing preprocessed chains
- NPZ schema validation (required keys, dtypes, and shape invariants)

Canonical fingerprint command:

```bash
python scripts/build_fingerprint.py \
  --config configs/limited_large_v3_official_baseline.yaml \
  --output leaderboard/official_dataset_fingerprint.json
```

## 5) Scoring

Primary metric:
- mean lDDT-Ca on validation set

Implementation:
- cutoff `15.0A`
- thresholds `[0.5, 1.0, 2.0, 4.0]`
- mean over valid residues with at least one neighbor in cutoff

Sanity property:
- perfect prediction (`pred_ca == true_ca` with valid mask) scores `1.0`

## 6) Official Maintainer Workflow

Canonical runner:

```bash
python scripts/run_official.py \
  --submission submissions/<name> \
  --track limited_large_v3 \
  --update-leaderboard
```

No-network Docker runner:

```bash
bash scripts/run_official_docker.sh \
  --submission submissions/<name> \
  --track limited_large_v3 \
  --update-leaderboard
```

## 7) Data Setup Paths

Official data setup (uses committed manifests, no regeneration):

```bash
bash scripts/setup_official_data.sh
```

Custom/research setup (regenerates manifests):

```bash
bash scripts/setup_custom_data.sh
```
