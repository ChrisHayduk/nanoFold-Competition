# Competition Rules and Official Protocol

nanoFold is a protein-folding slowrun: a fixed-data, fixed-budget benchmark for learning useful structure under biological data scarcity.

The point of the competition is not to approximate production AlphaFold systems with more retrieval or bigger pretrained priors. It is to ask a sharper research question: when the training set is fixed and comparatively small, which architectures, losses, curricula, and biological inductive biases learn the most?

Official rules are strict because the benchmark is only meaningful if the scarce-data premise is real. External structures, pretrained weights, network retrieval, and template lookup can all blur the line between learning from the official train set and importing knowledge from outside it.

## Design Principles

- Reward data efficiency, not just final checkpoint quality.
- Make biological priors compete under the same sample budget.
- Keep hidden labels sealed from submission code.
- Make dataset and preprocessing changes auditable through fingerprints.
- Prefer boring, enforceable rules over ambiguous leaderboard judgment calls.

The rest of this document is the enforceable contract for official leaderboard runs.

## 1) Track Source of Truth

Track policy is defined in `tracks/*.yaml`.

Official leaderboard track:
- `limited_large`

Official runs must use:
- `--track limited_large`
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
- train-from-scratch biological priors implemented directly in submission code

Disallowed:
- external sequences, structures, pretrained weights, checkpoints
- external MSA/template retrieval
- network downloads during official execution

Official template policy:
- template feature tensors are part of the schema, but official preprocessing uses `T=0`
- a future template-enabled track must add release-date and homology leakage filtering before templates can affect ranking

Rationale: if templates are enabled without a stronger policy, the contest risks rewarding target-template lookup instead of architectures that learn from limited data.

Validation and runtime guardrails:
- `scripts/validate_submission.py` blocks large artifacts and forbidden weight extensions
- suspicious network/download imports are flagged
- official Docker runner uses `--network=none`

## 3) Split Data Contract

Official preprocessing outputs split artifacts per chain:
- `processed_features/<chain_id>.npz`
- `processed_labels/<chain_id>.npz`

Required feature NPZ keys (per chain):
- `aatype (L,) int32`
- `msa (N,L) int32`
- `deletions (N,L) int32`
- `template_aatype (T,L) int32`
- `template_ca_coords (T,L,3) float32`
- `template_ca_mask (T,L) bool`

Required label NPZ keys (per chain):
- `ca_coords (L,3) float32`
- `ca_mask (L,) bool`
- `atom14_positions (L,14,3) float32` — full AF2 atom14 layout
- `atom14_mask (L,14) bool` — 1 where coordinate was present in mmCIF

Optional label NPZ keys:
- `residue_index (L,) int32` — contiguous 0..L-1 (AF2 supplement 1.2.9)
- `resolution () float32` — Å; 0.0 when unknown

Config fields:
- `data.processed_features_dir`
- `data.processed_labels_dir`

Official eval path is features-only for submission runtime:
- supervision keys are stripped inside runtime when `training=False`
- hidden labels are used only in maintainer scoring stage, outside submission runtime
- official hidden prediction rejects configs that set `data.processed_labels_dir`

Fingerprint contract:
- `leaderboard/official_dataset_fingerprint.json` pins manifest metadata, split file hashes, source-lock hash, and `preprocess_config_sha256`
- `preprocess_config_sha256` is hashed from `<processed_features_dir>/preprocess_meta.json`, so preprocessing flag changes invalidate the fingerprint
- hidden prediction may use `features_only` comparison so labels stay absent from the prediction runtime

## 4) Submission Interface Contract

Submission entrypoint (`submissions/<name>/submission.py`) must implement:
- `build_model(cfg)`
- `build_optimizer(cfg, model)`
- `run_batch(model, batch, cfg, training)`
- optional: `build_scheduler(cfg, optimizer)`

`run_batch` requirements:
- return `pred_atom14` shaped `(B, L, 14, 3)`
- runtime artifacts derive the Cα diagnostic view from atom14 slot 1
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
- hidden val: `1,000` chains

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
- track, source-lock, and preprocessing metadata

Official mode requirements:
- pinned manifest SHA checks
- fingerprint verification
- strict no-missing chain files
- NPZ schema validation for required keys/dtypes/shapes

## 7) Budgets and Determinism

Official budget definitions:
- sample budget: `B_sample = max_steps * effective_batch_size`
- residue budget: `B_res = max_steps * effective_batch_size * crop_size`

Official constants (`limited_large`):
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

Primary metric:
- `FoldScore = 0.55*lDDT-Ca + 0.30*lDDT-backbone-atom14 + 0.15*lDDT-all-atom14`
- all components use `cutoff=15.0A`, thresholds `[0.5,1.0,2.0,4.0]`, label masks, and equal chain weighting

Leaderboard ranking metric:
- `foldscore_auc_hidden`, trapezoidal AUC over cumulative samples on `[0, B_sample]`

Why AUC: the competition is a slowrun, so learning speed matters. A method that learns robust geometry earlier in the sample budget should be rewarded, even when final checkpoint scores are close.

Secondary metrics:
- `final_hidden_foldscore`
- `foldscore_at_steps` (`0`, `1000`, `2000`, `5000`, `last` by default)
- `foldscore_at_samples`
- `final_hidden_lddt_ca` and `lddt_at_*` diagnostics
- public val score is retained for diagnostics only

Canonical result artifact:
- `runs/<run_name>/result.json`

## 9) Official Hidden Pipeline

Canonical maintainer runner:

```bash
python scripts/run_official.py --submission submissions/<name> --track limited_large --update-leaderboard
```

Hidden assets are resolved via env (or explicit CLI overrides):
- `NANOFOLD_HIDDEN_MANIFEST`
- `NANOFOLD_HIDDEN_FEATURES_DIR`
- `NANOFOLD_HIDDEN_LABELS_DIR`
- `NANOFOLD_HIDDEN_FINGERPRINT`

Hidden lock metadata (safe to commit):
- `leaderboard/official_hidden_assets.lock.json`
- populate/update it with `python scripts/pin_hidden_assets.py ...`

Hidden official runs are split into:
- prediction stage: hidden features mounted, labels absent
- scoring stage: saved predictions + hidden labels, no submission hooks

Official orchestration entrypoints:
- `predict.py` for prediction only
- `score.py` for label-only scoring
- `scripts/run_official.py` for orchestration

Hidden leaderboard runs must execute in a sealed runtime. The supported maintainer path is:

Containerized no-network execution:

```bash
bash scripts/run_official_docker.sh --submission submissions/<name> --track limited_large --update-leaderboard
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
python scripts/validate_submission.py --submission submissions/<your_name> --track limited_large --strict
if git diff --name-only origin/main...HEAD | grep -Eq '^data/manifests/(train|val)\.txt$'; then
  echo "ERROR: PR edits protected manifests (train/val)."
  exit 1
fi
echo "Self-check passed."
```
