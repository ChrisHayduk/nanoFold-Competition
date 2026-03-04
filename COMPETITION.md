# Competition Rules and Official Protocol

This document defines the enforceable contract for official leaderboard runs.

## 1) Track Source of Truth

Track policy is defined in `tracks/*.yaml`.

Official leaderboard track:
- `limited_large_v3`

Official runs must use:
- `--track limited_large_v3`
- `--official`

In official mode the runtime uses **override + validate**:
- immutable constants from track policy are applied to config first
- policy validation then checks budget/paths/hashes
- model parameter cap from track (`model.max_params`) is enforced

## 2) Allowed and Disallowed Data

Allowed:
- benchmark data produced by this repo’s preprocessing path
- architecture/optimizer/loss changes
- curriculum/sampling over provided benchmark data

Disallowed:
- external sequences, structures, pretrained weights, checkpoints
- external MSA/template retrieval
- network downloads during official execution

Validation and runtime guardrails:
- `scripts/validate_submission.py` blocks large artifacts and forbidden weight extensions
- suspicious network/download imports are flagged
- official Docker runner uses `--network=none`

## 3) Split Data Contract (Breaking Format)

Official preprocessing outputs split artifacts per chain:
- `processed_features/<chain_id>.npz`
- `processed_labels/<chain_id>.npz`

Config fields:
- `data.processed_features_dir`
- `data.processed_labels_dir`

Official eval path is features-only for submission runtime:
- supervision keys are stripped inside runtime when `training=False`
- hidden labels are used only in maintainer scoring stage, outside submission runtime

## 4) Submission Interface Contract

Submission entrypoint (`submissions/<name>/submission.py`) must implement:
- `build_model(cfg)`
- `build_optimizer(cfg, model)`
- `run_batch(model, batch, cfg, training)`
- optional: `build_scheduler(cfg, optimizer)`

`run_batch` requirements:
- always return `pred_ca` shaped `(B, L, 3)` with floating dtype and finite values
- return scalar finite `loss` when `training=True`
- training `loss` must require gradients
- support unlabeled inference path (`training=False`)

Enforced by:
- `nanofold/submission_runtime.py`
- `scripts/validate_submission.py`

## 5) Official Dataset and Manifests

Official split definition:
- train: `10,000` chains
- val: `1,000` chains

Protected official manifests:
- `data/manifests/train.txt`
- `data/manifests/val.txt`
- `data/manifests/all.txt`

Manifest SHA256 digests (track pinned):
- train: `c3288fe5f855b602921734ea0113a858b09c8acfb28e53468940a5657abe2682`
- val: `7c279df96fad21e04909bc331466d96118256d3b0f69bccf1c2cc86d957d1f67`
- all: `2b2a298d078b7a398a5f3379769bdbfb33b27fe39239a5f052e07d24800df778`

Official setup path (participants):
- `scripts/setup_official_data.sh`

Maintainer manifest generation path:
- `scripts/build_manifests.py`
- `scripts/regenerate_official_manifests.sh`
- `scripts/sync_official_manifest_hashes.py`
- `scripts/full_official_data_refresh.sh`
- lock metadata: `leaderboard/official_manifest_source.lock.json`

Single end-to-end maintainer flow:

```bash
bash scripts/full_official_data_refresh.sh --rewrite-lock
```

## 6) Fingerprint and Integrity Requirements

Fingerprint tooling:
- canonical builder: `scripts/build_fingerprint.py`
- verifier: `nanofold/dataset_integrity.py`

Fingerprint includes:
- manifest chain counts/hashes
- chain-id digest
- feature file hash aggregate
- label file hash aggregate
- schema version and optional track/lock metadata

Official mode requirements:
- pinned manifest SHA checks
- fingerprint verification
- strict no-missing chain files
- NPZ schema validation for required keys/dtypes/shapes

## 7) Budgets and Determinism

Official budget definitions:
- sample budget: `B_sample = max_steps * effective_batch_size`
- residue budget: `B_res = max_steps * effective_batch_size * crop_size`

Official constants (`limited_large_v3`):
- `seed = 0`
- `crop_size = 256`
- `msa_depth = 192`
- `effective_batch_size = 2`
- `max_steps = 10,000`
- deterministic val settings (`center`, `top`)

Runtime reproducibility:
- deterministic seeding support
- deterministic DataLoader generator + worker seeding
- checkpoint stores/restores RNG state for resume path

## 8) Scoring and Ranking

Metric:
- lDDT-Ca (`cutoff=15.0A`, thresholds `[0.5,1.0,2.0,4.0]`)

Leaderboard ranking metric:
- `final_hidden_lddt_ca`

Secondary metrics:
- `lddt_auc_hidden`
- `lddt_at_steps` (`1000`, `2000`, `5000`, `10000` by default)
- public val score is retained for diagnostics only

Canonical result artifact:
- `runs/<run_name>/result.json`
- schema versioned (`schema_version`)

## 9) Official Hidden Pipeline

Canonical maintainer runner:

```bash
python scripts/run_official.py --submission submissions/<name> --track limited_large_v3 --update-leaderboard
```

Hidden assets are resolved via env (or explicit CLI overrides):
- `NANOFOLD_HIDDEN_MANIFEST`
- `NANOFOLD_HIDDEN_FEATURES_DIR`
- `NANOFOLD_HIDDEN_LABELS_DIR`
- `NANOFOLD_HIDDEN_FINGERPRINT`

Hidden lock metadata (safe to commit):
- `leaderboard/official_hidden_assets.lock.json`

Containerized no-network execution:

```bash
bash scripts/run_official_docker.sh --submission submissions/<name> --track limited_large_v3 --update-leaderboard
```

## 10) CI and PR Guardrails

CI enforces:
- `ruff`, `mypy`, `pytest`
- protected manifest PR guard
- JSON schema checks for leaderboard/result artifacts
- hidden path hardcode guard in track files
- synthetic smoke run for official train/eval path

Protected manifest PR rule:
- if `data/manifests/train.txt` or `data/manifests/val.txt` changes,
- PR fails unless label `manifest-change-approved` is set by maintainers

## 11) Submitter Self-Check

```bash
python scripts/validate_submission.py --submission submissions/<your_name> --track limited_large_v3 --strict
if git diff --name-only origin/main...HEAD | grep -Eq '^data/manifests/(train|val)\.txt$'; then
  echo "ERROR: PR edits protected manifests (train/val)."
  exit 1
fi
echo "Self-check passed."
```
