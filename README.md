![nanoFold](assets/nanofold.png)

# nanoFold: A Protein Folding Slowrun

nanoFold is a data-efficiency competition for protein structure prediction. It is inspired by the nanoGPT slowrun: everyone trains under a fixed budget, and the leaderboard rewards models that learn more structure from the same amount of data.

The core bet is simple: biological data is expensive. Text and image models often improve by consuming more data, but protein structure data is far more constrained, far harder to generate, and far more sensitive to leakage. If we want better biological foundation models, we need architectures and training methods that make stronger use of the data we already have.

This repo turns that idea into a benchmark. Participants get the same official train set, the same sample budget, and the same hidden evaluation path. Progress should come from better biological priors, better inductive biases, better objectives, better curricula, and better optimization under scarcity.

## Documentation

Start here:
- [Competition rules and official protocol](docs/COMPETITION.md)
- [Submission API contract](docs/API.md)
- [Data sources, splits, preprocessing, and tensor formats](docs/DATA.md)

Useful deep links:
- [What data sources are used](docs/DATA.md#1-data-sources)
- [How to download the official public data](docs/DATA.md#2-how-data-is-downloaded)
- [How train/val/hidden splits are generated](docs/DATA.md#3-how-dataset-splits-are-determined)
- [How raw data becomes model input](docs/DATA.md#4-how-input-data-is-prepped)
- [Example model input batch](docs/DATA.md#5-example-model-input-sample)
- [Required prediction output format](docs/DATA.md#6-output-data-format)
- [Example model output](docs/DATA.md#7-example-model-output-sample)

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

## Split Curation

Official splits treat proteins as grouped biological examples, not IID rows in a text file:

- candidates are filtered by length, resolution, monomer status, and strict sequence content
- candidates with non-standard amino-acid tokens are excluded from the official split
- MMseqs2, or a locked TSV produced with the same MMseqs2 settings, defines sequence-homology groups
- train, public val, and hidden val are cluster-disjoint and PDB-entry-disjoint
- representatives are chosen by structure quality before length, so duplicate groups do not over-contribute low-quality chains
- structure metadata is required before splitting; the official allocation is stratified by secondary-structure class, broad domain architecture, length bin, and resolution bin
- metadata sources include pinned CATH, SCOPe, ECOD, and RCSB-style structural classifications when those records cover a chain
- required OpenFold feature-asset availability is enforced before split generation
- the lock records source hashes, filtering counts, clustering mode, grouping policy, stratification fields, split quality metrics, and per-split alpha/beta/mixed distributions

The metadata builder projects broad secondary-structure fractions from domain architecture classes so every eligible chain has a deterministic split signal before any label mmCIFs are downloaded. All metadata signals are used only for split balancing and audit reports, never for scoring.

## Data Format

Official preprocessing writes split artifacts per chain:
- features: `data/processed_features/<encoded_chain_id>.npz`
- labels: `data/processed_labels/<encoded_chain_id>.npz`

`<encoded_chain_id>` is nanoFold's filesystem-safe chain stem. It preserves
case-sensitive PDB chain IDs on every supported filesystem.

Required feature keys:
- `aatype`, `msa`, `deletions`
- `residue_index` — `(L,)` int32, contiguous 0..L-1 and available during sealed inference
- `between_segment_residues` — `(L,)` int32, zero for single-chain examples
- `template_aatype`, `template_ca_coords`, `template_ca_mask`

Required label keys:
- `ca_coords`, `ca_mask`
- `atom14_positions` — `(L, 14, 3)` float32, full atom14 layout
- `atom14_mask` — `(L, 14)` bool, True where coordinate was present in mmCIF

Additional label metadata:
- `residue_index` — `(L,)` int32, duplicated for convenience when present
- `resolution` — `()` float32, Å (0.0 if unknown)

Preprocessing run metadata is captured in `<processed_features_dir>/preprocess_meta.json`. Its SHA256 is folded into the dataset fingerprint, so changes to preprocessing flags, projection thresholds, dependency metadata, or source revision are visible to the verifier.

Official scoring requires atom14 labels, and submissions must return `pred_atom14` shaped `(B, L, 14, 3)`. The runtime derives the Cα view from atom14 slot 1 for diagnostics and baseline losses.

The official track disables templates by preprocessing with `T=0`; template-enabled tracks require explicit leakage filters.

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
pip install -r requirements.txt awscli
git submodule update --init --recursive

# 2) download and preprocess the official public data
bash scripts/setup_official_data.sh \
  --data-root data/openproteinset \
  --processed-features-dir data/processed_features \
  --processed-labels-dir data/processed_labels \
  --mmcif-mode subset \
  --disable-templates

# 3) official train + public validation scoring
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

`setup_official_data.sh` is the standard public data path. It verifies the committed manifest hashes, downloads the required OpenFold chain cache/MSAs plus the manifest mmCIF subset, and writes:
- `data/processed_features/<encoded_chain_id>.npz`
- `data/processed_labels/<encoded_chain_id>.npz`

If preprocessing is interrupted, rerun the same command with `--resume-preprocess`. The public setup path does not download hidden validation labels or hidden manifests.

System requirements for data setup:
- `aws` CLI on `PATH` for unsigned OpenFold S3 downloads
- `unzip` on `PATH`
- enough local disk for raw OpenFold assets plus processed NPZs; expect tens of GB for the official public split

## Maintainer Data Refresh

Maintainers can refresh the official data assets with one command. This path:
- downloads and pins structural metadata sources
- builds required structure metadata from chain cache, pinned structural-classification files, and required feature-asset coverage
- regenerates official manifests from locked chain cache inputs plus a private hidden split salt
- syncs manifest hashes/counts across track + docs + lock
- downloads required OpenFold assets and manifest mmCIFs with strict missing-file checks
- preprocesses public and hidden split NPZs
- rebuilds public and hidden fingerprints
- writes public metadata plus all hidden artifacts under the ignored `.nanofold_private/` workspace

```bash
export NANOFOLD_HIDDEN_SPLIT_SALT="<maintainer-private-random-string>"
bash scripts/full_official_data_refresh.sh --rewrite-lock
```

This flow requires MMseqs2 on `PATH`. `NANOFOLD_HIDDEN_SPLIT_SALT` must be at least 32 characters and must never be committed. It regenerates split metadata, public manifests, public NPZs, the public dataset fingerprint, and maintainer-only hidden assets from the locked official inputs.

Public outputs land in stable, commit-safe paths:
- `data/manifests/train.txt`
- `data/manifests/val.txt`
- `data/manifests/all.txt`
- `leaderboard/official_dataset_fingerprint.json`
- `leaderboard/official_manifest_source.lock.json`

Maintainer-only outputs land under `.nanofold_private/`:
- `.nanofold_private/manifests/hidden_val.txt`
- `.nanofold_private/manifests/split_quality_report.json`
- `.nanofold_private/hidden_processed_features/`
- `.nanofold_private/hidden_processed_labels/`
- `.nanofold_private/leaderboard/official_hidden_fingerprint.json`
- `.nanofold_private/leaderboard/private_hidden_assets.lock.json`
- `.nanofold_private/leaderboard/private_hidden_manifest_source.lock.json`
- `.nanofold_private/leaderboard/official_data_source.lock.json`

The metadata builder also writes `data/manifests/structure_candidates.txt` as an ignored local audit artifact. Commit only public manifests, the public dataset fingerprint, and sanitized public lock metadata.

Dry-run preview:

```bash
bash scripts/full_official_data_refresh.sh --rewrite-lock --dry-run
```

## Hidden Leaderboard Runs

Hidden leaderboard runs are maintainer-only. The prediction stage sees submission code plus hidden features. The scoring stage sees saved predictions plus hidden labels. Submission code is never imported during hidden scoring.

Hidden mode uses `.nanofold_private/` by default. Override these only when hidden assets live elsewhere:
- `NANOFOLD_HIDDEN_MANIFEST`
- `NANOFOLD_HIDDEN_FEATURES_DIR`
- `NANOFOLD_HIDDEN_LABELS_DIR`
- `NANOFOLD_HIDDEN_FINGERPRINT`
- `NANOFOLD_HIDDEN_LOCK_FILE`

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

Hidden lock metadata is maintainer-local and ignored by git. Populate/update it with `python scripts/pin_hidden_assets.py ...`.

## Manifest Reproducibility

Check committed official manifest hashes:

```bash
shasum -a 256 data/manifests/train.txt data/manifests/val.txt data/manifests/all.txt
```

Expected `limited_large` values:
- `train.txt`: `d36d1f77ba43b7c4509a6e9dfd3f9414e1ce60f8364b24e0086c1734ba6aef6d`
- `val.txt`: `d4a0265bcd0a021e116c0c889f21e86bc24006460bcc42dec2f9a80b70c8812b`
- `all.txt`: `0d1b21a3536cd0c602be301993fcbacd8ecc5710a459c239f9808d164d0ee85d`

Maintainer-only manifest regeneration:

```bash
bash scripts/regenerate_official_manifests.sh --rewrite-lock
```

Sync public hashes/counts across pinned references (track + lock + docs):

```bash
python scripts/sync_official_manifest_hashes.py
```

## Submitter Self-Check

```bash
python scripts/validate_submission.py \
  --submission submissions/<your_name> \
  --track limited_large \
  --strict

if git diff --name-only origin/main...HEAD | grep -Eq '^data/manifests/(train|val|all)\.txt$'; then
  echo "ERROR: PR edits protected manifests. Ask maintainer for explicit approval label."
  exit 1
fi

echo "Self-check passed."
```

CI enforces the same PR guardrail:
- edits to `data/manifests/train.txt`, `data/manifests/val.txt`, or `data/manifests/all.txt` fail unless label `manifest-change-approved` is present.

## Repo Map

- `docs/`: data guide, API contract, and official competition protocol
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
