![nanoFold](assets/nanofold.png)

# nanoFold: A Protein Folding Slowrun

nanoFold is a data-efficiency competition for protein structure prediction. It is inspired by the nanoGPT slowrun: everyone trains under a fixed budget, and the leaderboard rewards models that learn more structure from the same amount of data.

The core bet is simple: biological data is expensive. Text and image models often improve by consuming more data, but protein structure data is far more constrained, far harder to generate, and far more sensitive to leakage. If we want better biological foundation models, we need architectures and training methods that make stronger use of the data we already have.

This repo turns that idea into a benchmark. Participants get the same official train set, the same sample budget, and the same hidden evaluation path. Progress should come from better biological priors, better inductive biases, better objectives, better curricula, and better optimization under scarcity.

## Start Here

New to nanoFold? Follow the [Quickstart](docs/QUICKSTART.md) to set up the environment, download and preprocess the public official data, and start your first training run.

## Documentation

Start here:
- [Quickstart: setup, data prep, and first training run](docs/QUICKSTART.md)
- [Competition rules and official protocol](docs/COMPETITION.md)
- [Submission API contract](docs/API.md)
- [Data sources, splits, preprocessing, and tensor formats](docs/DATA.md)

Useful deep links:
- [Download and preprocess public data](docs/QUICKSTART.md#2-download-and-preprocess-public-data)
- [Start a training run](docs/QUICKSTART.md#3-start-training)
- [Run training on a Modal GPU](docs/QUICKSTART.md#optional-run-on-a-modal-gpu)
- [Submit to the leaderboard](docs/QUICKSTART.md#6-submit-to-the-leaderboard)
- [Allowed and disallowed data](docs/COMPETITION.md#2-allowed-and-disallowed-data)
- [Scoring and ranking](docs/COMPETITION.md#8-scoring-and-ranking)
- [Batch contract by mode](docs/API.md#batch-contract-by-mode)
- [`run_batch` output contract](docs/API.md#run_batch-output-contract)
- [Dataset split methodology](docs/DATA.md#3-how-dataset-splits-are-determined)
- [Example model input sample](docs/DATA.md#5-example-model-input-sample)

Training can also run across CUDA GPUs using `torchrun`; the runner preserves
the global sample budget. See [training sampling and multiple GPUs](docs/API.md#training-sampling-and-multiple-gpus)
for rank and accumulation requirements. The [data guide](docs/DATA.md#2-how-data-is-downloaded)
covers download timeouts and reports for incomplete preprocessing.

## What Counts As Progress

nanoFold is not meant to be a pretrained-weight contest, a web-retrieval contest, or a race to find near-duplicate templates. The official tracks are deliberately strict:

- fixed official data and manifests
- no external structures, pretrained weights, external MSA retrieval, or network access
- no template features in the official tracks
- hidden ranking by area under the learning curve for fixed-budget tracks
- atom14-aware scoring so methods are rewarded for more than Cα traces

The intended question is: **given the same limited training set, who learns useful protein geometry fastest and best?**

## How The Slowrun Works

The `limited` and `research_large` tracks train for fixed sample budgets. Their hidden leaderboards evaluate multiple checkpoints and rank by `foldscore_auc_hidden`: trapezoidal area under hidden FoldScore versus cumulative samples seen.

That makes early learning matter. A model that gets useful structure after 2,000 samples should beat a model that only wakes up at the end, even if their final scores are close. This is the pressure that should surface architectures with better biological priors.

FoldScore is a CASP15-inspired raw score over the CASP metrics that can be
computed reproducibly from the official atom14 output contract and official
residue identities:

```text
FoldScore =
  0.25*GDT_HA-Ca
+ 0.09375*(lDDT-all-atom14 + CADaa-atom14 + SG-atom14 + SC-atom14)
+ 0.125*(MolProbity-clash-atom14 + BB-atom14 + DipDiff-atom14)
```

`GDT_HA-Ca` measures global C-alpha fold accuracy with threshold-specific GDT
superpositions. The remaining terms measure local all-atom agreement, contact
preservation, SphereGrinder local motifs, chi1/chi2 side-chain geometry,
atom-name-aware heavy-atom clashes, phi/psi/omega backbone geometry, and
three-residue C-alpha/O distance windows. The weights are the CASP15 formula
weights renormalized over the structure-derived components supported by
nanoFold. `ASE` requires submitted confidence estimates and `reLLG_lddt`
requires a crystallographic molecular-replacement scoring path, so they are not
part of the official rank score.

The final hidden FoldScore is the tie-breaker for fixed-budget tracks and the primary rank metric for `unlimited`. Public validation exists for debugging, not ranking.

## What This Repo Provides

- official track policy in `tracks/limited.yaml`
- minAlphaFold2-derived preprocessing for A3M/mmCIF alignment, atom14 labels, residue constants, and template plumbing
- sealed prediction/scoring entrypoints that keep hidden labels away from submission code
- a strict submission API with `build_model`, `build_optimizer`, and `run_batch`
- dataset fingerprints and manifest checks so official data changes are visible
- an optional Modal GPU runner with local-disk data staging for public-data training
- a pinned minAlphaFold2 tiny reference submission plus a template submission that pass the official atom14 contract

## Tracks At A Glance

Source of truth: `tracks/*.yaml`. All tracks use the same official public train set, public validation set, sealed hidden validation set, CASP15-inspired FoldScore, and no-template policy. They differ in how much training budget is allowed and what scientific question the leaderboard is meant to answer.

| Track | Purpose | Fixed training budget | Rank metric | Use this when | Submit with |
|---|---|---:|---|---|---|
| `limited` | Main accessible slowrun leaderboard | `240,000` samples (`30,000` steps x effective batch `8`) | `foldscore_auc_hidden` | you want the primary competition result under a small, reproducible budget | `--track limited` |
| `research_large` | Larger fixed-data research leaderboard | `960,000` samples (`120,000` steps x effective batch `8`) | `foldscore_auc_hidden` | you want to study whether an approach still wins with more optimization while using the same data | `--track research_large` |
| `unlimited` | Open-ended fixed-data research leaderboard | unrestricted steps and model size with effective batch `8` | `final_hidden_foldscore` | you want the best final hidden structure quality while keeping the hidden set sealed and the public data contract fixed | `--track unlimited` |

For all three tracks, set `track: <track_id>` in `submissions/<name>/config.yaml`, validate with `python scripts/validate_submission.py --submission submissions/<name> --track <track_id> --strict`, and open a submission PR naming the intended track. Submit separate configs or separate submission directories when one method targets multiple tracks. Maintainers create accepted leaderboard entries after sealed hidden evaluation; participant PRs should not edit leaderboard artifacts.

Leaderboard entries include a linked `Name` column pointing to the accepted
submission directory and a `Team` column for identity. Use a lab, company,
project team, or individual researcher name consistently across submissions. If
`--team` is omitted during a PR-triggered GitHub Actions run, nanoFold falls
back to the PR author's GitHub username. Manual maintainer automation can set
`NANOFOLD_PR_AUTHOR=<github-username>` for the same fallback.

The `limited` constants are:

| Item | Value |
|---|---:|
| Train chains | `10,000` |
| Public val chains | `1,000` |
| Hidden val chains | `1,000` |
| Seed | `0` |
| Crop size | `256` |
| MSA depth | `192` |
| Effective batch size | `8` |
| Max steps | `30,000` |
| Sample budget | `240,000` |
| Residue budget | `61,440,000` |
| Parameter cap | unrestricted |
| Tie-breaker | `final_hidden_foldscore` |

## Split Curation

The committed manifests, source locks, and fingerprints define the published
benchmark and remain frozen. The examples are individual protein chains;
biological monomer status was not established by the original cache metadata.
The split groups assigned sequence clusters and PDB entries. Cluster
disjointness does not prove that every cross-split pair is below the clustering
identity threshold, or that CATH homologous superfamilies are disjoint.

The `cath-v2` sampler is available for staging a separately versioned dataset.
It requires schema-2 structure metadata and an explicit
`--sampling-policy cath-v2`:

- CATH Class, Architecture, Topology, and Homologous superfamily identifiers are
  retained per domain, including multiple domains on a chain. Missing annotation
  is explicit; broad alpha/beta labels do not substitute for CATH Architecture.
- Allocation targets the eligible disjoint-unit distribution hierarchically
  across each chain's set of CATH Architecture IDs, then length and resolution
  bins. Measured secondary structure is reported separately when available.
- Assigned sequence clusters, exact duplicate sequences, and shared PDB entries
  form indivisible groups. One representative is retained per connected group.
- Public validation and training are allocated before hidden validation, so the
  public selection is independent of the private hidden salt.
- Source hashes, annotation coverage, structural distributions, and held-out
  CATH familiarity are recorded for review. Structural balancing does not itself
  create a family-disjoint generalization test.

Candidate staging requires complete cache-target sequence audits across both
split boundaries before issuing a verified receipt, even with precomputed
clusters. Asset setup then requires acquired MSA queries to match the pinned
cache targets exactly and records that binding in preprocessing provenance.
Processed-tensor and residual MSA audits remain separate release gates.

The published benchmark's broad structural metadata and existing results must
not be relabeled as `cath-v2` results. See [dataset sampling](docs/DATA_SAMPLING.md)
for the staging workflow, validation gates, and remaining scientific limits.

Training resume preserves the sampler epoch/cursor, loader generators and per-rank RNG, and checks the actual sampling weights and dataset bytes. Checkpoint replacement is atomic, and invalid RNG or AMP state fails before changing run history. Checkpoints lacking supported sampling state require a fresh training run. See the [training sampling contract](docs/API.md#training-sampling-and-multiple-gpus).

Unweighted distributed epochs drop incomplete tails instead of repeating
chains. Resume also validates exposure counters against the step and global
batch size. Split verification replays CATH-v2 selection from the locked inputs,
including exact integer quota ties and private exclusions.

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

Preprocessing schema 3 records its configuration and implementation digests in
`<processed_features_dir>/preprocess_meta.json`; the stable data-producing
configuration digest is included in the dataset fingerprint. Each feature/label
pair also records the configuration and raw-input digests. Resume checks the raw
source bytes before reusing a pair. Changed or missing run provenance requires
fresh feature and label directories; existing outputs are not relabeled with
new metadata. See [data preprocessing](docs/DATA.md) for the full contract.

Official scoring requires atom14 labels, and submissions must return `pred_atom14` shaped `(B, L, 14, 3)`. The runtime derives the Cα view from atom14 slot 1 for diagnostics.

Preprocessing maps observed coordinates through the deposited polymer sequence
and residue indices, preserving unresolved positions in repeated sequences.
Feature archives containing supervision are rejected, and feature/label output
directories must be separate. Collated batches include `msa_mask` to distinguish
real MSA gaps from padding; maintained submissions use it for MSA statistics.
Scoring rejects truncated or nonfinite predictions and binds official scoring
inputs to the maintainer's evaluation contract.

Maintainer preprocessing uses a private held-out-target MSA row filter built
from the train/public-validation/hidden-validation source MSAs against both
held-out target sets. It removes detected non-query homolog rows before the MSA
depth cap while retaining each query. The filter digest is recorded in the
features and preprocessing metadata. Filters derived from hidden targets remain
private.

`scripts/audit_msa_split.py` reports availability, MSA-depth balance, target
chain/PDB/cluster overlap, exact sequence overlap, and optional MMseqs homology
and raw A3M identifier overlap. Add `--strict --run-mmseqs` to fail on incomplete
features, target overlap, or residual held-out-target hits. A cluster audit
requires a complete supplied cluster TSV; without it, cluster separation is
reported as unchecked. Zero MMseqs hits applies to the configured heuristic
search over non-query rows of at least 20 residues, not all possible homology.
The default depth CSV and examples omit identifiers; identifier flags are
explicit opt-ins for public or sealed private reports.

The fixed-manifest asset workflow uses
`scripts/full_official_data_refresh.sh --skip-manifest-regen` and runs strict
public/hidden MSA audits before fingerprinting. Preprocessing failures stop the
workflow without changing exclusion lists or resampling targets. A processed
feature filter can support local audits of unbound research assets when raw
A3Ms are unavailable. Schema-3 assets require preprocessing again from raw
inputs; postprocessing cannot preserve their paired provenance or recover rows
lost at the original depth cap. The compact summary from
`scripts/write_msa_row_filter_source_lock.py` checks the final filter digest and
filtering counts; it is not a residual-homology audit. Detailed commands and
source-lock handling are in [docs/DATA.md](docs/DATA.md).

The official tracks disable templates by preprocessing with `T=0`; template-enabled tracks require explicit leakage filters.

Config schema uses:
- `data.processed_features_dir`
- `data.processed_labels_dir`

## Official Policy

In `--official` mode, the runner applies override + validate:
- immutable track constants are forced into config at startup
- then policy validation and manifest hash checks run
- trainable parameter count is reported; no public track enforces a hard parameter cap

This is implemented in:
- `nanofold/competition_policy.py`
- `train.py`
- `eval.py`
- `scripts/validate_submission.py`

## Quickstart

The fastest participant path is in [docs/QUICKSTART.md](docs/QUICKSTART.md). It covers:

- environment setup
- official public data download and preprocessing
- first minAlphaFold2 tiny reference training run
- public validation prediction and scoring
- creating and validating a new submission
- submitting a leaderboard pull request

## Maintainer Hidden Assets

The published public manifests and sealed hidden assets belong to one fixed
evaluation contract. Keep hidden manifests, salts, feature/label directories,
row filters, fingerprints, and private locks in the ignored
`.nanofold_private/` workspace. Public releases contain only the reviewed public
manifests, public fingerprints, and public-safe provenance summaries.

New `cath-v2` candidates require complete schema-2 metadata and both source locks.
`scripts/stage_dataset_candidate.py` builds and verifies the manifests in a fresh
private directory and records completion only after independent verification.
`scripts/regenerate_official_manifests.sh` refuses automatic regeneration;
its `--help` points to the staged workflow. The full refresh wrapper also refuses
manifest regeneration and requires `--skip-manifest-regen` for asset maintenance.
Rebuilt fingerprints and asset locks default to
`.nanofold_private/asset_rebuild/leaderboard/`; protected release paths and their
aliases are rejected before work starts.
Do not update published track hashes to make a candidate dataset pass the frozen
contract checks.

Follow [dataset sampling](docs/DATA_SAMPLING.md) to construct and review a new
candidate, and [data handling](docs/DATA.md) for private MSA filtering, feature
audits, fingerprints, and hidden-asset locks. Preprocessing schema 3 requires
fresh output directories when existing outputs lack matching provenance.
Hidden preprocessing failures remain private and do not trigger automatic
changes to public exclusions or manifests.

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
  --track <track_id> \
  --team "<team or individual name>" \
  --update-leaderboard
```

Hidden leaderboard runs require a sealed no-network runtime. The supported path is:

```bash
bash scripts/run_official_docker.sh \
  --submission submissions/<name> \
  --track <track_id> \
  --team "<team or individual name>" \
  --update-leaderboard
```

Invoke the wrapper from a trusted checkout; `NANOFOLD_TRUSTED_ROOT` selects its
source explicitly when needed. It builds only selected tracked source files,
pins the resulting image ID, and uses separate containers for public training,
hidden prediction and scoring. Hidden features are unavailable during training;
hidden labels are mounted only for scoring. Scoring reads immutable predictions
and writes separate fresh result files. The host repository is never mounted.
Use `NANOFOLD_PUBLIC_FEATURES_DIR`, `NANOFOLD_PUBLIC_LABELS_DIR` and
`NANOFOLD_RUNS_ROOT` to relocate host assets and outputs. Config data paths remain
`data/processed_features` and `data/processed_labels` inside the container.
Processed directories may contain only the selected manifest's NPZs and the
documented preprocessing sidecars; extra files, directories and links fail
preflight. The packaged vendor tree contains code and required configuration,
excluding example structures and other research artifacts.

Maintainers can run the same two-stage hidden evaluation on Modal when the checkpoint already lives in the `nanofold-runs` Modal volume:

```bash
modal run scripts/modal_official.py \
  --upload-public-data \
  --upload-hidden-assets \
  --upload-only
```

```bash
modal run scripts/modal_official.py \
  --submission submissions/<name> \
  --config submissions/<name>/config.yaml \
  --track <track_id> \
  --team "<team or individual name>" \
  --update-leaderboard
```

For long Modal runs, launch the prediction and scoring stages as detached calls, then download the durable result artifact and update the local leaderboard:

```bash
modal run --detach scripts/modal_official.py \
  --submission submissions/<name> \
  --config submissions/<name>/config.yaml \
  --track <track_id> \
  --team "<team or individual name>" \
  --skip-score \
  --background-predict
```

```bash
modal run --detach scripts/modal_official.py \
  --submission submissions/<name> \
  --config submissions/<name>/config.yaml \
  --track <track_id> \
  --team "<team or individual name>" \
  --skip-predict \
  --background-score
```

```bash
modal volume get nanofold-runs <run_name>/modal_official_result.json runs/<run_name>/modal_official_result.json
python scripts/add_leaderboard_entry.py \
  --result runs/<run_name>/modal_official_result.json \
  --leaderboard leaderboard/leaderboard.json \
  --readme README.md \
  --description "<leaderboard description>" \
  --team "<team or individual name>"
```

Use the upload command the first time a Modal environment is prepared, or whenever public/hidden assets change. Hidden Modal execution uses separate no-network prediction and scoring functions; hidden labels are not mounted during prediction. Both Modal stages request the configured GPU because atom14 structural scoring is pairwise-distance heavy.

Hidden lock metadata is maintainer-local and ignored by git. Populate/update it with `python scripts/pin_hidden_assets.py ...`.

## Manifest Reproducibility

Check committed official manifest hashes:

```bash
shasum -a 256 data/manifests/train.txt data/manifests/val.txt data/manifests/all.txt
```

Expected public manifest values for all tracks:
- `train.txt`: `d36d1f77ba43b7c4509a6e9dfd3f9414e1ce60f8364b24e0086c1734ba6aef6d`
- `val.txt`: `d4a0265bcd0a021e116c0c889f21e86bc24006460bcc42dec2f9a80b70c8812b`
- `all.txt`: `0d1b21a3536cd0c602be301993fcbacd8ecc5710a459c239f9808d164d0ee85d`

Check that published hashes/counts agree across the tracks, lock, and docs:

```bash
python scripts/sync_official_manifest_hashes.py --check
```

Manifest generation for a new sampling policy is a separate staged dataset
release. Follow [docs/DATA_SAMPLING.md](docs/DATA_SAMPLING.md); the official
regeneration wrapper refuses to overwrite the published contract.

## Submitter Self-Check

```bash
python scripts/validate_submission.py \
  --submission submissions/<your_name> \
  --track <track_id> \
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
- `scripts/setup_custom_data.sh`: public asset setup from verified staged custom manifests
- `scripts/build_manifests.py`: maintainer manifest generation
- `scripts/sync_official_manifest_hashes.py`: sync official manifest hashes across track/lock/docs
- `scripts/build_fingerprint.py`: split dataset fingerprint generator
- `scripts/run_official.py`: canonical official validate/train/eval/result runner
- `scripts/run_official_docker.sh`: no-network official container runner
- `scripts/modal_official.py`: maintainer-only Modal hidden evaluation runner
- `nanofold/submission_runtime.py`: runtime API enforcement
- `third_party/minAlphaFold2`: pinned upstream minAlphaFold2 implementation used by `submissions/minalphafold2`
- `leaderboard/`: leaderboard and official lock/fingerprint artifacts

## Leaderboard

<!-- LEADERBOARD_START -->
### `limited`
| # | Name | Team | Rank Score | Hidden FoldScore | Public FoldScore | Date | Commit | Description |
|---:|---|---|---:|---:|---:|---|---|---|
| 1 | [minalphafold2_full](submissions/minalphafold2_full) | nanoFold Maintainers | 0.2634 | 0.3423 | 0.3426 | 2026-05-01 | `85f8c9c` | minAlphaFold2 full profile limited-track benchmark |
<!-- LEADERBOARD_END -->
