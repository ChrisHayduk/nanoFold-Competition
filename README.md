# nanoFold Data Efficiency Competition

Reproducible benchmark for protein structure prediction under fixed data and fixed training budget.

## What This Repo Enforces

- Fixed manifests and pinned manifest hashes for official runs.
- Fixed track-level budget/policy (`tracks/*.yaml`).
- Fixed runtime contract for submissions:
  - `build_model(cfg)`
  - `build_optimizer(cfg, model)`
  - `run_batch(model, batch, cfg, training)`
- Common lDDT-Ca metric implementation and official runner artifact format.

Primary docs:
- [COMPETITION.md](COMPETITION.md): enforceable rules and official protocol
- [API.md](API.md): submission/runtime API contract

## Official Track (`limited_large_v3`)

Source of truth: `tracks/limited_large_v3.yaml`

Official constants:
- train chains: `10,000`
- val chains: `1,000`
- `seed = 0`
- `data.crop_size = 256`
- `data.msa_depth = 192`
- effective batch size: `data.batch_size * train.grad_accum_steps = 2`
- `train.max_steps = 10,000`
- `data.val_crop_mode = center`
- `data.val_msa_sample_mode = top`

Derived budgets:
- sample budget: `B_sample = max_steps * effective_batch_size = 20,000`
- residue budget: `B_res = max_steps * effective_batch_size * crop_size = 5,120,000`

## Data Format (Breaking Cutover)

Official preprocessing now writes split artifacts per chain:
- features: `data/processed_features/<chain_id>.npz`
- labels: `data/processed_labels/<chain_id>.npz`

Required feature keys:
- `aatype`, `msa`, `deletions`, `template_aatype`, `template_ca_coords`, `template_ca_mask`

Required label keys:
- `ca_coords`, `ca_mask`

Config schema uses:
- `data.processed_features_dir`
- `data.processed_labels_dir`

`data.processed_dir` is no longer the canonical field.

## Official Policy Semantics

In `--official` mode, the runner applies **override + validate**:
- immutable track constants are forced into config at startup
- then policy validation and manifest hash checks run
- model parameter cap is enforced from track policy (`model.max_params`)

This is implemented in:
- `nanofold/competition_policy.py`
- `train.py`
- `eval.py`
- `scripts/validate_submission.py`

## Quickstart (Official Public Data)

```bash
# 1) environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) setup official data using committed manifests
bash scripts/setup_official_data.sh \
  --data-root data/openproteinset \
  --processed-features-dir data/processed_features \
  --processed-labels-dir data/processed_labels

# 3) build official fingerprint (split features+labels)
python scripts/build_fingerprint.py \
  --config configs/limited_large_v3_official_baseline.yaml \
  --track limited_large_v3 \
  --source-lock leaderboard/official_manifest_source.lock.json \
  --output leaderboard/official_dataset_fingerprint.json

# 4) official train/eval
python train.py \
  --config configs/limited_large_v3_official_baseline.yaml \
  --track limited_large_v3 \
  --official

mkdir -p runs/official_limited_large_v3_baseline/_forbid_labels
python eval.py \
  --config configs/limited_large_v3_official_baseline.yaml \
  --ckpt runs/official_limited_large_v3_baseline/checkpoints/ckpt_last.pt \
  --split val \
  --track limited_large_v3 \
  --official \
  --forbid-labels-dir runs/official_limited_large_v3_baseline/_forbid_labels \
  --score-labels-dir data/processed_labels \
  --save runs/official_limited_large_v3_baseline/eval_val.json \
  --per-chain-out runs/official_limited_large_v3_baseline/per_chain_scores_val.jsonl
```

## Single End-to-End Official Data Flow (Maintainers)

Run this one command to execute the full pipeline:
- regenerate official manifests from locked chain cache inputs
- sync manifest hashes/counts across track + docs + lock
- download required OpenFold assets
- preprocess split NPZs (`processed_features` + `processed_labels`)
- rebuild official fingerprint

```bash
bash scripts/full_official_data_refresh.sh --rewrite-lock
```

Dry-run preview:

```bash
bash scripts/full_official_data_refresh.sh --rewrite-lock --dry-run
```

## Official Hidden Pipeline (Maintainers)

Leaderboard ranking metric is:
- `final_hidden_lddt_ca` (hidden split, final checkpoint)

Secondary metrics:
- `lddt_auc_hidden`
- `lddt_at_steps` (default checkpoints: `1000,2000,5000,10000`)

Required maintainer env vars for hidden mode:
- `NANOFOLD_HIDDEN_MANIFEST`
- `NANOFOLD_HIDDEN_FEATURES_DIR`
- `NANOFOLD_HIDDEN_LABELS_DIR`
- `NANOFOLD_HIDDEN_FINGERPRINT`

Canonical run:

```bash
python scripts/run_official.py \
  --submission submissions/<name> \
  --track limited_large_v3 \
  --update-leaderboard
```

No-network containerized run:

```bash
bash scripts/run_official_docker.sh \
  --submission submissions/<name> \
  --track limited_large_v3 \
  --update-leaderboard
```

Hidden lock metadata (hashes only, no secret paths) is stored in:
- `leaderboard/official_hidden_assets.lock.json`

## Manifest Reproducibility

Check committed official manifest hashes:

```bash
shasum -a 256 data/manifests/train.txt data/manifests/val.txt data/manifests/all.txt
```

Expected `limited_large_v3` values:
- `train.txt`: `c3288fe5f855b602921734ea0113a858b09c8acfb28e53468940a5657abe2682`
- `val.txt`: `7c279df96fad21e04909bc331466d96118256d3b0f69bccf1c2cc86d957d1f67`
- `all.txt`: `2b2a298d078b7a398a5f3379769bdbfb33b27fe39239a5f052e07d24800df778`

Maintainer-only manifest regeneration:

```bash
bash scripts/regenerate_official_manifests.sh --rewrite-lock
```

Sync hashes/counts across all pinned references (track + lock + docs):

```bash
python scripts/sync_official_manifest_hashes.py
```

## Submitter Self-Check

```bash
python scripts/validate_submission.py \
  --submission submissions/<your_name> \
  --track limited_large_v3 \
  --strict

if git diff --name-only origin/main...HEAD | grep -Eq '^data/manifests/(train|val)\.txt$'; then
  echo "ERROR: PR edits protected manifests (train/val). Ask maintainer for explicit approval label."
  exit 1
fi

echo "Self-check passed."
```

CI enforces the same PR guardrail:
- edits to `data/manifests/train.txt` or `data/manifests/val.txt` fail unless label `manifest-change-approved` is present.

## Repo Map

- `tracks/`: track policy definitions
- `configs/`: official/research config profiles
- `scripts/setup_official_data.sh`: official participant setup
- `scripts/setup_custom_data.sh`: research/custom manifest setup
- `scripts/build_manifests.py`: maintainer manifest generation
- `scripts/sync_official_manifest_hashes.py`: sync official manifest hashes across track/lock/docs
- `scripts/build_fingerprint.py`: split dataset fingerprint generator
- `scripts/run_official.py`: canonical official validate/train/eval/result runner
- `scripts/run_official_docker.sh`: no-network official container runner
- `nanofold/submission_runtime.py`: runtime API enforcement
- `leaderboard/`: leaderboard and official lock/fingerprint artifacts

## Leaderboard

<!-- LEADERBOARD_START -->
| # | Rank Score | Hidden Final | Hidden AUC | Public Val | Track | Date | Commit | Description |
|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 0.0000 | n/a | n/a | 0.0000 | limited | 2026-03-03 | `seedesm` | Seed submission: ESMFold-inspired trunk (pending official benchmark run) |
| 2 | 0.0000 | n/a | n/a | 0.0000 | limited | 2026-03-03 | `seedope` | Seed submission: OpenFold-style Evoformer + template stack (pending official benchmark run) |
<!-- LEADERBOARD_END -->
