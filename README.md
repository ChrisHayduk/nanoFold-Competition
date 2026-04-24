# nanoFold: A Protein Folding Slowrun

nanoFold is a data-efficiency competition for protein structure prediction. It is inspired by the nanoGPT slowrun: everyone trains under a fixed budget, and the leaderboard rewards models that learn more structure from the same amount of data.

The core bet is simple: biological data is expensive. Text and image models often improve by consuming more data, but protein structure data is far more constrained, far harder to generate, and far more sensitive to leakage. If we want better biological foundation models, we need architectures and training methods that make stronger use of the data we already have.

This repo turns that idea into a benchmark. Participants get the same official train set, the same sample budget, and the same hidden evaluation path. Progress should come from better biological priors, better inductive biases, better objectives, better curricula, and better optimization under scarcity.

## What Counts As Progress

nanoFold is not meant to be a pretrained-weight contest, a web-retrieval contest, or a race to find near-duplicate templates. The official track is deliberately strict:

- fixed official data and manifests
- no external structures, pretrained weights, external MSA retrieval, or network access
- no template features in the official track
- hidden ranking by area under the learning curve, not just final checkpoint score
- atom14-aware scoring so methods are rewarded for more than Cα traces

The intended question is: **given the same limited training set, who learns useful protein geometry fastest and best?**

## How The Slowrun Works

Each official run trains for the same sample budget. The hidden leaderboard evaluates multiple checkpoints and ranks by `foldscore_auc_hidden`: trapezoidal area under hidden FoldScore versus cumulative samples seen.

That makes early learning matter. A model that gets useful structure after 2,000 samples should beat a model that only wakes up at the end, even if their final scores are close. This is the pressure that should surface architectures with better biological priors.

FoldScore combines:

```text
0.55*lDDT-Ca + 0.30*lDDT-backbone-atom14 + 0.15*lDDT-all-atom14
```

The final hidden FoldScore is the tie-breaker. Public validation exists for debugging, not ranking.

## What This Repo Provides

- official track policy in `tracks/limited_large.yaml`
- minAlphaFold2-derived preprocessing for A3M/mmCIF alignment, atom14 labels, residue constants, and template plumbing
- sealed prediction/scoring entrypoints that keep hidden labels away from submission code
- a strict submission API with `build_model`, `build_optimizer`, and `run_batch`
- dataset fingerprints and manifest checks so official data changes are visible
- a pinned minAlphaFold2 reference submission plus a template submission that pass the official atom14 contract

Primary docs:
- [COMPETITION.md](COMPETITION.md): enforceable rules and official protocol
- [API.md](API.md): submission/runtime API contract

## Official Track At A Glance

Source of truth: `tracks/limited_large.yaml`

| Item | Value |
|---|---:|
| Train chains | `10,000` |
| Public val chains | `1,000` |
| Hidden val chains | `1,000` |
| Seed | `0` |
| Crop size | `256` |
| MSA depth | `192` |
| Effective batch size | `2` |
| Max steps | `10,000` |
| Sample budget | `20,000` |
| Residue budget | `5,120,000` |
| Rank metric | `foldscore_auc_hidden` |
| Tie-breaker | `final_hidden_foldscore` |

## Data Format

Official preprocessing writes split artifacts per chain:
- features: `data/processed_features/<chain_id>.npz`
- labels: `data/processed_labels/<chain_id>.npz`

Required feature keys:
- `aatype`, `msa`, `deletions`, `template_aatype`, `template_ca_coords`, `template_ca_mask`

Required label keys:
- `ca_coords`, `ca_mask`
- `atom14_positions` — `(L, 14, 3)` float32, full atom14 layout
- `atom14_mask` — `(L, 14)` bool, True where coordinate was present in mmCIF

Additional label metadata:
- `residue_index` — `(L,)` int32, contiguous 0..L-1
- `resolution` — `()` float32, Å (0.0 if unknown)

Preprocessing run metadata is captured in `<processed_features_dir>/preprocess_meta.json`. Its SHA256 is folded into the dataset fingerprint, so changes to preprocessing flags, projection thresholds, dependency metadata, or source revision are visible to the verifier.

Official scoring requires atom14 labels, and submissions must return `pred_atom14` shaped `(B, L, 14, 3)`. The runtime derives the Cα view from atom14 slot 1 for diagnostics and baseline losses.

The official track disables templates by preprocessing with `T=0`; template tensors remain in the API so a future template-enabled track can add leakage filters explicitly.

Config schema uses:
- `data.processed_features_dir`
- `data.processed_labels_dir`

## Official Policy

In `--official` mode, the runner applies override + validate:
- immutable track constants are forced into config at startup
- then policy validation and manifest hash checks run
- model parameter cap is enforced from track policy (`model.max_params`)

This is implemented in:
- `nanofold/competition_policy.py`
- `train.py`
- `eval.py`
- `scripts/validate_submission.py`

## Quickstart

```bash
# 1) environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
git submodule update --init --recursive

# 2) setup official data using committed manifests
bash scripts/setup_official_data.sh \
  --data-root data/openproteinset \
  --processed-features-dir data/processed_features \
  --processed-labels-dir data/processed_labels

# 3) build official fingerprint (split features+labels)
python scripts/build_fingerprint.py \
  --config configs/official_baseline.yaml \
  --track limited_large \
  --source-lock leaderboard/official_manifest_source.lock.json \
  --output leaderboard/official_dataset_fingerprint.json

# 4) official train + public validation scoring
python train.py \
  --config configs/official_baseline.yaml \
  --track limited_large \
  --official

mkdir -p runs/official_limited_large_baseline/_forbid_labels
python predict.py \
  --config configs/official_baseline.yaml \
  --ckpt runs/official_limited_large_baseline/checkpoints/ckpt_last.pt \
  --split val \
  --track limited_large \
  --official \
  --forbid-labels-dir runs/official_limited_large_baseline/_forbid_labels \
  --pred-out-dir runs/official_limited_large_baseline/public_predictions \
  --save runs/official_limited_large_baseline/predict_val.json

python score.py \
  --prediction-summary runs/official_limited_large_baseline/predict_val.json \
  --labels-dir data/processed_labels \
  --save runs/official_limited_large_baseline/eval_val.json \
  --per-chain-out runs/official_limited_large_baseline/per_chain_scores_val.jsonl
```

## Maintainer Data Refresh

Maintainers can refresh the official public assets with one command. This path:
- regenerates official manifests from locked chain cache inputs
- syncs manifest hashes/counts across track + docs + lock
- downloads required OpenFold assets
- preprocesses split NPZs (`processed_features` + `processed_labels`)
- rebuilds the official fingerprint

```bash
bash scripts/full_official_data_refresh.sh --rewrite-lock
```

Dry-run preview:

```bash
bash scripts/full_official_data_refresh.sh --rewrite-lock --dry-run
```

## Hidden Leaderboard Runs

Hidden leaderboard runs are maintainer-only. The prediction stage sees submission code plus hidden features. The scoring stage sees saved predictions plus hidden labels. Submission code is never imported during hidden scoring.

Required maintainer env vars for hidden mode:
- `NANOFOLD_HIDDEN_MANIFEST`
- `NANOFOLD_HIDDEN_FEATURES_DIR`
- `NANOFOLD_HIDDEN_LABELS_DIR`
- `NANOFOLD_HIDDEN_FINGERPRINT`

Canonical runner entrypoint:

```bash
python scripts/run_official.py \
  --submission submissions/<name> \
  --track limited_large \
  --update-leaderboard
```

Hidden leaderboard runs require a sealed no-network runtime. The supported path is:

```bash
bash scripts/run_official_docker.sh \
  --submission submissions/<name> \
  --track limited_large \
  --update-leaderboard
```

Hidden lock metadata (hashes only, no secret paths) is stored in:
- `leaderboard/official_hidden_assets.lock.json`
- populate/update it with `python scripts/pin_hidden_assets.py ...`

## Manifest Reproducibility

Check committed official manifest hashes:

```bash
shasum -a 256 data/manifests/train.txt data/manifests/val.txt data/manifests/all.txt
```

Expected `limited_large` values:
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
  --track limited_large \
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
- `third_party/minAlphaFold2`: pinned upstream minAlphaFold2 implementation used by `submissions/minalphafold2`
- `leaderboard/`: leaderboard and official lock/fingerprint artifacts

## Leaderboard

<!-- LEADERBOARD_START -->
| # | FoldScore AUC | Hidden FoldScore | Public FoldScore | Track | Date | Commit | Description |
|---:|---:|---:|---:|---|---|---|---|
<!-- LEADERBOARD_END -->
