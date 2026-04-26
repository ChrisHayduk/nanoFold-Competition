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

Track policy is defined in `tracks/*.yaml`. The public track set is:

| Track | Purpose | Data contract | Training budget | Rank metric | Submission selector |
|---|---|---|---:|---|---|
| `limited` | primary competition track for accessible data-efficiency work | official train/public val/hidden val only | `20,000` samples (`10,000` steps x effective batch `2`) | `foldscore_auc_hidden` | `--track limited` |
| `research_large` | larger fixed-data track for methods that need more optimization to show their shape | same official data and hidden evaluation as `limited` | `100,000` samples (`50,000` steps x effective batch `2`) | `foldscore_auc_hidden` | `--track research_large` |
| `unlimited` | open-ended fixed-data track for best final structure quality under sealed hidden evaluation | same official data and hidden evaluation as `limited` | unrestricted | `final_hidden_foldscore` | `--track unlimited` |

All three tracks use atom14 FoldScore, hidden labels remain sealed, templates are disabled, and public validation is diagnostic only. `limited` and `research_large` are sample-budget slowruns, so they rank by hidden area under the learning curve. `unlimited` is ranked separately by final hidden FoldScore because there is no common sample axis.

Official runs must use:
- `--track <track_id>`
- `--official`

Participants submit code/configuration PRs for a chosen track. Maintainers run the sealed hidden pipeline and update leaderboard artifacts after acceptance; participant PRs must not edit `leaderboard/leaderboard.json` or the rendered leaderboard table.

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
- template-enabled tracks require release-date and homology leakage filtering before templates can affect ranking

Rationale: if templates are enabled without a stronger policy, the contest risks rewarding target-template lookup instead of architectures that learn from limited data.

Validation and runtime guardrails:
- `scripts/validate_submission.py` blocks large artifacts and forbidden weight extensions
- suspicious network/download imports are flagged
- official Docker runner uses `--network=none`

## 3) Split Data Contract

Official preprocessing outputs split artifacts per chain:
- `processed_features/<encoded_chain_id>.npz`
- `processed_labels/<encoded_chain_id>.npz`

`<encoded_chain_id>` is nanoFold's filesystem-safe chain stem, so case-sensitive
PDB chain IDs stay distinct on every supported filesystem.

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

Split construction policy:
- candidate chains are filtered by length, resolution, monomer status, and strict standard-amino-acid sequence content
- sequence clusters are built with MMseqs2 or a locked TSV produced with the same MMseqs2 identity/coverage settings
- train, public val, and hidden val must be sequence-cluster-disjoint at the configured identity/coverage threshold
- different chains from the same PDB entry cannot cross split boundaries
- structure metadata is required; split allocation is stratified by secondary-structure class, broad domain architecture, length bin, and resolution bin
- metadata sources include pinned CATH, SCOPe, ECOD, and RCSB-style structural classifications when those records cover a chain
- required OpenFold feature-asset availability is enforced before split generation
- source hashes, filter counts, clustering method, grouping policy, stratification fields, split distributions, and split-quality metrics are recorded in locks
- `all.txt` is the public union of train + public val only; hidden chain IDs stay in maintainer-only hidden manifests

Protected official manifests:
- `data/manifests/train.txt`
- `data/manifests/val.txt`
- `data/manifests/all.txt`

Manifest SHA256 digests (track pinned):
- train: `d36d1f77ba43b7c4509a6e9dfd3f9414e1ce60f8364b24e0086c1734ba6aef6d`
- val: `d4a0265bcd0a021e116c0c889f21e86bc24006460bcc42dec2f9a80b70c8812b`
- all: `0d1b21a3536cd0c602be301993fcbacd8ecc5710a459c239f9808d164d0ee85d`

Official setup path (participants):
- `scripts/setup_official_data.sh`

Participant data setup:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt awscli
bash scripts/setup_official_data.sh \
  --data-root data/openproteinset \
  --processed-features-dir data/processed_features \
  --processed-labels-dir data/processed_labels \
  --mmcif-mode subset \
  --disable-templates
```

This command verifies the committed public manifest hashes, downloads the required OpenFold chain cache/MSAs and manifest mmCIF subset, then preprocesses train and public validation features plus labels. Rerun it with `--resume-preprocess` after an interrupted preprocessing run.

The setup script expects `aws`, `unzip`, and `python` on `PATH`; it keeps hidden validation assets out of the public setup path.

Maintainer manifest generation path:
- `scripts/build_manifests.py`
- `scripts/regenerate_official_manifests.sh`
- `scripts/build_hidden_manifest.py`
- `scripts/sync_official_manifest_hashes.py`
- `scripts/full_official_data_refresh.sh`
- lock metadata: `leaderboard/official_manifest_source.lock.json`

The committed public manifests are the participant-facing data contract. Hidden validation is generated privately against that fixed public split:

```bash
mkdir -p .nanofold_private/secrets
python -c "import pathlib,secrets; pathlib.Path('.nanofold_private/secrets/hidden_split_salt.txt').write_text(secrets.token_urlsafe(48) + '\n')"
chmod 600 .nanofold_private/secrets/hidden_split_salt.txt

python scripts/build_hidden_manifest.py \
  --hidden-split-salt-file .nanofold_private/secrets/hidden_split_salt.txt

python scripts/verify_hidden_manifest.py
```

This maintainer flow requires MMseqs2 plus the public setup dependencies. The salt must be at least 32 characters and must never be committed. The hidden builder excludes every sequence/PDB component that touches train or public validation, then stratifies hidden validation against the public train+validation distribution.

Private hidden preprocessing failures belong only in `.nanofold_private/manifests/hidden_processability_exclusions.txt`. After adding any failed chain IDs there, rerun `scripts/build_hidden_manifest.py` and `scripts/verify_hidden_manifest.py`.

Commit-safe public outputs:
- `data/manifests/train.txt`
- `data/manifests/val.txt`
- `data/manifests/all.txt`
- `leaderboard/official_dataset_fingerprint.json`
- `leaderboard/official_manifest_source.lock.json`

Maintainer-only outputs live under the ignored `.nanofold_private/` workspace:
- `.nanofold_private/manifests/hidden_val.txt`
- `.nanofold_private/manifests/split_quality_report.json`
- `.nanofold_private/manifests/hidden_processability_exclusions.txt`
- `.nanofold_private/hidden_processed_features/`
- `.nanofold_private/hidden_processed_labels/`
- `.nanofold_private/leaderboard/official_hidden_fingerprint.json`
- `.nanofold_private/leaderboard/private_hidden_assets.lock.json`
- `.nanofold_private/leaderboard/private_hidden_manifest_source.lock.json`
- `.nanofold_private/leaderboard/official_data_source.lock.json`

Full public-data rebuilds use `bash scripts/full_official_data_refresh.sh --rewrite-lock` and are reserved for deliberate changes to the official public data contract.

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

Track budget constants:

| Track | Seed | Crop size | MSA depth | Effective batch | Max steps | Sample budget | Residue budget | Parameter cap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `limited` | `0` | `256` | `192` | `2` | `10,000` | `20,000` | `5,120,000` | `50,000,000` |
| `research_large` | `0` | `256` | `192` | `2` | `50,000` | `100,000` | `25,600,000` | `50,000,000` |
| `unlimited` | submitter-defined | submitter-defined | submitter-defined | submitter-defined | submitter-defined | unrestricted | unrestricted | unrestricted |

All tracks use deterministic public validation settings (`center`, `top`) when the track defines them.

Runtime reproducibility:
- deterministic seeding support
- deterministic DataLoader generator + worker seeding
- checkpoint stores/restores RNG state for resume path

## 8) Scoring and Ranking

Primary metric:
- `FoldScore = 0.55*lDDT-Ca + 0.30*lDDT-backbone-atom14 + 0.15*lDDT-all-atom14`
- all components use `cutoff=15.0A`, thresholds `[0.5,1.0,2.0,4.0]`, label masks, and equal chain weighting

Leaderboard ranking metric:
- `limited`: `foldscore_auc_hidden`, trapezoidal AUC over cumulative samples on `[0, B_sample]`
- `research_large`: `foldscore_auc_hidden`, trapezoidal AUC over cumulative samples on `[0, B_sample]`
- `unlimited`: `final_hidden_foldscore`

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
python scripts/run_official.py --submission submissions/<name> --track <track_id> --update-leaderboard
```

Hidden assets are resolved via env (or explicit CLI overrides):
- `NANOFOLD_HIDDEN_MANIFEST`
- `NANOFOLD_HIDDEN_FEATURES_DIR`
- `NANOFOLD_HIDDEN_LABELS_DIR`
- `NANOFOLD_HIDDEN_FINGERPRINT`
- `NANOFOLD_HIDDEN_LOCK_FILE`

Hidden assets default to `.nanofold_private/`; env vars and CLI flags are only needed when assets live elsewhere. Hidden lock metadata is maintainer-local and ignored by git. Populate/update it with `python scripts/pin_hidden_assets.py ...`.

Hidden manifests, hidden NPZ directories, hidden fingerprints, hidden source locks, and private split salt metadata are maintainer-local artifacts. Public files contain the public dataset contract and public split counts only.

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
bash scripts/run_official_docker.sh --submission submissions/<name> --track <track_id> --update-leaderboard
```

The Docker build copies only source-oriented files into the image. Generated datasets, hidden assets, checkpoints, local environments, and run outputs are excluded from the build context and are supplied through runtime mounts.

## 10) CI and PR Guardrails

CI enforces:
- `ruff`, `mypy`, `pytest`
- protected manifest PR guard
- JSON schema checks for leaderboard/result artifacts
- hidden path hardcode guard in track files and public lock metadata
- synthetic smoke run for official train/eval path

Protected manifest PR rule:
- if `data/manifests/train.txt`, `data/manifests/val.txt`, or `data/manifests/all.txt` changes,
- PR fails unless label `manifest-change-approved` is set by maintainers

## 11) Submitter Self-Check

```bash
python scripts/validate_submission.py --submission submissions/<your_name> --track <track_id> --strict
if git diff --name-only origin/main...HEAD | grep -Eq '^data/manifests/(train|val|all)\.txt$'; then
  echo "ERROR: PR edits protected manifests (train/val)."
  exit 1
fi
echo "Self-check passed."
```

Use the same `<track_id>` in the submission config, local training command, validation command, and pull-request description. Submit separate configs or separate submission directories for the same method on multiple tracks.
