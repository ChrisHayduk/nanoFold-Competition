# nanoFold Data Efficiency Competition

Reproducible benchmark for protein structure prediction under fixed data and fixed training budget.

## Overview

This repo standardizes:
- data splits and preprocessing
- track-specific training/eval policy
- shared lDDT-Ca scoring
- official run and leaderboard workflow

Submissions plug into a fixed runtime interface:
- `build_model(cfg)`
- `build_optimizer(cfg, model)`
- `run_batch(model, batch, cfg, training)`

API details: [API.md](API.md)

## Official Track

Official track id: `limited_large_v3` (`tracks/limited_large_v3.yaml`)

Fixed criteria:
- train split: `10,000` chains
- val split: `1,000` chains
- `seed = 0`
- `data.crop_size = 256`
- `data.msa_depth = 192`
- `effective_batch_size = data.batch_size * train.grad_accum_steps = 2`
- `train.max_steps = 10,000`
- `data.val_crop_mode = center`
- `data.val_msa_sample_mode = top`

Derived budgets:
- sample budget: `B_sample = 20,000`
- residue budget: `B_res = 5,120,000`

Official runs also enforce:
- fixed manifest paths (`data/manifests/train.txt`, `data/manifests/val.txt`)
- fingerprint match (`leaderboard/official_dataset_fingerprint.json`)
- strict no-missing preprocessed chains

## Quickstart (Official Path)

```bash
# 1) environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) setup data using committed official manifests
bash scripts/setup_official_data.sh \
  --data-root data/openproteinset \
  --processed-dir data/processed

# 3) build or refresh fingerprint
python scripts/build_fingerprint.py \
  --config configs/limited_large_v3_official_baseline.yaml \
  --output leaderboard/official_dataset_fingerprint.json \
  --allow-missing

# 4) train/eval under official enforcement
python train.py \
  --config configs/limited_large_v3_official_baseline.yaml \
  --track limited_large_v3 \
  --official

python eval.py \
  --config configs/limited_large_v3_official_baseline.yaml \
  --ckpt runs/official_limited_large_v3_baseline/checkpoints/ckpt_last.pt \
  --track limited_large_v3 \
  --official \
  --save runs/official_limited_large_v3_baseline/eval_val.json \
  --per-chain-out runs/official_limited_large_v3_baseline/per_chain_scores.jsonl
```

If preprocessing is incomplete, official mode fails early with explicit setup guidance.

## Custom/Research Data Path

Use this path only when you intentionally want custom manifests/splits:

```bash
bash scripts/setup_custom_data.sh \
  --data-root data/openproteinset \
  --train-size 10000 \
  --val-size 1000 \
  --seed 0
```

This is not the canonical official setup path.

## Official Runner

Single canonical maintainer entrypoint:

```bash
python scripts/run_official.py \
  --submission submissions/<name> \
  --track limited_large_v3 \
  --update-leaderboard
```

Dockerized no-network variant:

```bash
bash scripts/run_official_docker.sh \
  --submission submissions/<name> \
  --track limited_large_v3 \
  --update-leaderboard
```

## Submission Validation

Before opening a PR:

```bash
python scripts/validate_submission.py \
  --submission submissions/<your_name> \
  --track limited_large_v3
```

## Reference Configs

- `configs/limited_large_v3_official_baseline.yaml`: canonical official baseline config.
- `configs/unlimited_research_baseline.yaml`: non-official research baseline config.

## Repo Map

- `tracks/`: track policy definitions
- `configs/`: runnable configs (official + baseline)
- `scripts/setup_official_data.sh`: official data setup
- `scripts/setup_custom_data.sh`: custom/research setup
- `scripts/build_manifests.py`: maintainer manifest generation
- `scripts/build_fingerprint.py`: canonical fingerprint builder
- `scripts/run_official.py`: official validate/train/eval/result pipeline
- `leaderboard/`: leaderboard and fingerprint artifacts

## Leaderboard

<!-- LEADERBOARD_START -->
| # | Score (lDDT-Cα) | Track | Date | Commit | Description |
|---:|---:|---|---|---|---|
| 1 | 0.0000 | limited | 2026-03-03 | `seedesm` | Seed submission: ESMFold-inspired trunk (pending official benchmark run) |
| 2 | 0.0000 | limited | 2026-03-03 | `seedope` | Seed submission: OpenFold-style Evoformer + template stack (pending official benchmark run) |
<!-- LEADERBOARD_END -->
