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
| `limited` | primary competition track for accessible data-efficiency work | official train/public val/hidden val only | `240,000` samples (`30,000` steps x effective batch `8`) | `foldscore_auc_hidden` | `--track limited` |
| `research_large` | larger fixed-data track for methods that need more optimization to show their shape | same official data and hidden evaluation as `limited` | `960,000` samples (`120,000` steps x effective batch `8`) | `foldscore_auc_hidden` | `--track research_large` |
| `unlimited` | open-ended fixed-data track for best final structure quality under sealed hidden evaluation | same official data and hidden evaluation as `limited` | unrestricted steps with effective batch `8` | `final_hidden_foldscore` | `--track unlimited` |

All three tracks use CASP15-inspired FoldScore, hidden labels remain sealed, templates are disabled, and public validation is diagnostic only. `limited` and `research_large` are sample-budget slowruns, so they rank by hidden area under the learning curve. `unlimited` is ranked separately by final hidden FoldScore because there is no common sample axis.

Official runs must use:
- `--track <track_id>`
- `--official`

Participants submit code/configuration PRs for a chosen track and provide the lab, company, project team, or individual researcher name that should appear on the leaderboard. Each accepted entry renders a linked submission `Name` plus that leaderboard identity. If the maintainer does not pass an explicit team name during PR-triggered GitHub Actions evaluation, nanoFold falls back to the PR author's GitHub username. Manual maintainer automation can set `NANOFOLD_PR_AUTHOR=<github-username>` for the same fallback. Maintainers run the sealed hidden pipeline and update leaderboard artifacts after acceptance; participant PRs must not edit `leaderboard/leaderboard.json` or the rendered leaderboard table.

In official mode the runtime uses **override + validate**:
- immutable constants from track policy are applied to config first
- policy validation then checks budget/paths/hashes
- trainable parameter count is reported; public tracks do not enforce a hard parameter cap

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

The committed manifests, fingerprints, and source locks are frozen. They define
single-chain prediction examples; the original chain cache did not establish
biological monomer status. Assigned sequence clusters and PDB entries define
split-separation groups. This is not a guarantee that every cross-split pair
falls below the MMseqs identity threshold or that homologous families are absent
across splits. `all.txt` contains train plus public validation only.

Qualified candidate generation explicitly selects `--sampling-policy cath-v4`,
with schema-2 structure metadata and fresh private outputs. Staging defaults
to `cath-v3` when no policy is supplied. CATH Class/Architecture/Topology/Homologous-superfamily
identifiers remain attached to every annotated domain, with missing annotation
explicit. Hierarchical balance targets the validated representatives' CATH
Architecture-label sets, then length and resolution. Measured secondary structure
is a separate annotation. `cath-v2` and `cath-v3` remain explicitly selectable
policies with their own replay contracts.

Original selection units join assigned sequence clusters, exact duplicate
sequences and PDB entries transitively. CATH v4 retains the first member in each
unit's deterministic quality ranking that passes raw-query and strict
historical-structure validation. Every earlier rank must definitively fail
before a later member is selected; exhaustive failure is required to remove a
unit from the selectable pool. Missing files, download errors and pending checks
cannot prove failure. Strict projection thresholds stay fixed; partial-label
relaxation and current-source substitution are not allowed.

The complete original homology graph and all connecting members remain intact.
Only a full group with every original unit exhausted may be excluded, and the
exhausted-group set must exactly match an explicitly approved catalog. The
builder requires original `--cluster-tsv`, `--split-group-tsv`,
`--split-group-receipt`, `--representative-eligibility-receipt` and
`--approved-group-exclusions`; source locks bind them for independent replay.

For the 10,000/1,000/1,000 candidate contract:

1. Jointly select 2,000 retained full groups while preserving training capacity.
   Public-seed allocation fixes 1,000 public-validation and 1,000 hidden groups,
   with one selected target per evaluation group.
2. Select one training anchor from every other retained group, then fill the
   remaining slots from validated original-unit representatives. No unit is
   selected twice and no full group crosses a split boundary.
3. Use the private salt only to choose members and order inside fixed hidden
   groups. Anchor quotas do not promise exact quotas for the members' strata;
   actual hidden distributions must pass quality checks. Hidden-only generation
   replays the same plan; new failures cannot trigger silent resampling.

The approved candidate excludes six exhausted groups from the original 8,856,
retaining 8,850. Training therefore requires 6,850 anchors plus 3,150 additional
validated representatives. Original and validated unit counts/digests, retained
group counts, maximum training-group share, annotation coverage and CATH
familiarity are reported separately. Capacity and large groups can skew the
training/evaluation mixture; recalculate concentration from the validated pool.
Default training weights selected targets equally, so a group with `m` targets
receives `m` exposures in a complete traversal before batch/tail drops.
Inverse-group weighting is a separate ablation. Coverage and structural balance
do not demonstrate better model performance or family-disjoint generalization.

Complete retained-group coverage has a privacy limit: anyone who knows the
original universe, grouping and approved exclusions can infer hidden groups as
the retained complement of public groups. Singleton hidden-group membership is then inferable. Keeping hidden IDs,
salts, labels, and artifacts private does not guarantee membership secrecy.

Candidate generation does not change the committed benchmark or establish that
processed assets, homology audits and release gates have passed. Existing
results retain their original dataset identity. See [dataset sampling](DATA_SAMPLING.md)
for candidate generation, source validation, quality gates, and release review.

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

This command verifies the committed public manifest hashes, downloads the
required OpenFold chain cache/MSAs and manifest mmCIF subset, then preprocesses
train and public validation features plus labels. Use `--resume-preprocess` only
for outputs with matching schema-3 preprocessing provenance. Resume verifies
each pair's raw-input digest before reuse. Changed or missing run provenance
requires fresh feature and label directories. New preprocessing bytes must still
match the published fingerprint to qualify for the frozen benchmark.

Setup verifies the chain cache against the manifest source lock and requires
every acquired MSA query to match its selected cache sequence exactly before
coordinate projection. Cache bytes and target identity are bound into each
processed pair's provenance; changed cache bytes require fresh outputs.
Coordinate labels require a unique observed-residue mapping across all optimal
deposited-polymer alignments; unresolved positions may have equivalent placements.
Maintainer rebuilds can retrieve pinned historical structures with
`scripts/fetch_openproteinset_mmcif.py`, which retries transient transport failures
within fixed bounds and rejects integrity failures. Successful retrieval establishes source
integrity only; complete target, structure, split and tensor gates still apply.

The setup script expects `aws`, `unzip`, and `python` on `PATH`; it keeps hidden validation assets out of the public setup path.

Maintainer asset rebuilding against the frozen manifests requires
`scripts/full_official_data_refresh.sh --skip-manifest-regen`. The workflow
builds two private filters: training source MSAs against public-plus-hidden
validation targets, and public-validation source MSAs against hidden targets.
Public `all.txt` is preprocessed once with explicit split assignments so each
chain receives its intended filter through `preprocess.py --msa-filter-policy`.
The policy binds both filters and all three split manifests, fixes 30% identity
and 80% bidirectional coverage, and requires both complete verified search
checkpoints. Hidden preprocessing omits both `--msa-filter-policy` and
`--msa-row-filter`.
Detected non-query homologs are removed before the depth cap; query row 0 is
always preserved. Ordinary MSA normalization and depth limits remain in place.
Policy and effective-filter provenance bind the resulting tensors. Keep all
artifacts derived from hidden targets private.

The workflow runs `scripts/audit_msa_split.py --strict --run-mmseqs --exhaustive` on public
training versus public validation and on all public targets versus hidden
validation before fingerprinting. These are directed boundaries; there is no
reverse or within-split homology exclusion. Strict mode rejects missing features,
duplicate targets, chain/PDB overlap, exact target-sequence leakage, and detected
residual MMseqs hits. Cluster checks require a complete `--cluster-tsv`; absent
cluster assignments are not interpreted as proof of separation. Target searches
align all source/held-out pairs with prefiltering and masking disabled and a
permissive E-value limit. Both filter construction and residual audits use
exhaustive checkpointed searches over non-query MSA rows of at least 20 residues.
Standalone tools use heuristic MSA search unless `--exhaustive` is supplied.
Exhaustive mode requires enough `--max-seqs`
capacity for the whole source population, or each complete source batch when
checkpointed batching is enabled. The filter builder's `--inventory-only`
reports unique sequence counts and pair workload without searching or producing
a usable filter. `--checkpoint-dir` with exhaustive mode adds bounded jobs and
verified resume; the final result requires the complete query/source job grid,
not a subset of finished shards. Resource monitoring is sampled, not an OS hard
limit. See [the search contract](DATA_SAMPLING.md#bounded-resumable-exhaustive-msa-searches).
Reports record the search mode. A completed search does not prove absence under
every possible alignment or of all biological homology.
Exact held-out sequence matches are also removed independently of MMseqs,
including low-complexity sequences that its masking can suppress.

Standalone audits remain descriptive unless `--strict` is supplied. They also
report MSA-depth distributions, exact background-row overlap, and optional raw
A3M identifier overlap. Default output omits chain IDs from the depth CSV and
omits examples. `--include-depth-identifiers`, `--include-examples`, and
`--include-identifiers-in-mmseqs` explicitly enable these details for public or
sealed private reports. Use `--val-features-dir` for a separate hidden feature
root.

Automatic regeneration is disabled in both official wrappers.
`scripts/regenerate_official_manifests.sh --help` gives staging guidance; new
candidates use `scripts/stage_dataset_candidate.py` to build complete metadata,
generate both source locks, and independently replay fresh private outputs.
Explicit `--sampling-policy cath-v4` staging requires original clustering, audited
full-group assignments, source-validated representative evidence and the approved
whole-group exclusion catalog. Both cache-target sequence audits must pass
before a verified receipt is issued, including when clusters are precomputed.
This early gate does not replace processed-tensor and MSA audits. Direct builders
require an explicit `--sampling-policy cath-v4`, `cath-v3` or `cath-v2`.
The hidden builder requires a matching versioned public source lock; cath-v4
verification checks fixed group ownership and complete retained coverage as well as
the selected representatives. Follow [DATA_SAMPLING.md](DATA_SAMPLING.md) for the
complete workflow.
Do not sync candidate hashes into the published track configuration. Asset
rebuilds write candidate fingerprints and raw/hidden asset locks under
`.nanofold_private/asset_rebuild/leaderboard/` by default and reject protected
release paths and aliases before execution.

Keep hidden manifests, salts, labels, features, row filters, full quality reports,
and hidden fingerprints/locks in the ignored `.nanofold_private/` workspace.
Preprocessing failures stop asset rebuilding without modifying exclusions or
resampling any split. Hidden failure details must not enter public exclusion
lists. The detailed feature, source-lock, and private-asset procedures live in
[DATA.md](DATA.md).

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

Preprocessing schema 3 records data-producing configuration and implementation
digests, and binds each feature/label pair to its raw input bytes. A populated
output root must retain matching run provenance. Resume cannot overwrite that
provenance to describe outputs made under another configuration.

Filtering existing processed MSAs with `scripts/filter_processed_msa_features.py`
is limited to research assets without paired schema-3 provenance, writes a new
root, and binds reuse to source bytes. Schema-3 pairs require fresh preprocessing.
Filtering cannot recover rows discarded by an earlier depth cap. Filter raw MSA
rows during preprocessing and audit the final features. If a residual MMseqs
search finds hits, the extension/filtering tools can remove those rows before a
new audit; the resulting zero-hit claim remains specific to that search.

`scripts/write_msa_row_filter_source_lock.py` records filter digests, thresholds,
normalized search settings, public manifest hashes, counts, and aggregate
filtering results. By default it redacts unrecognized/private paths and private
manifest hashes, validates final filter-digest correspondence and row counts,
and omits per-chain records and excluded sequence hashes. Directed preprocessing
uses one summary per split, with `--msa-filter-policy`, `--split train` or
`--split val`, and the corresponding filter and split audit. It summarizes
filtering provenance; it does not independently establish zero residual homology.

Official mode requirements:
- pinned manifest SHA checks
- fingerprint verification
- strict no-missing chain files
- NPZ schema validation for required keys/dtypes/shapes

## 7) Budgets and Determinism

Official budget definitions:
- sample budget: `B_sample = max_steps * effective_batch_size`
- residue budget: `B_res = max_steps * effective_batch_size * crop_size`

Multi-GPU training preserves the configured global effective batch and sample
budget by dividing local gradient accumulation across ranks. Rank count must
divide the configured accumulation count exactly; otherwise training fails.
Per-step sample and residue counters sum all ranks. Weighted sampling is an
explicitly enabled training study and must be identified as such; it still
uses only the fixed training manifest and the same sample budget. See the
[training API](API.md#training-sampling-and-multiple-gpus) for configuration.

Unweighted distributed epochs drop shuffled tails that cannot fill ranks and
local batches, instead of repeating chains. The checkpoint contract records the
omitted population. MSA row/residue padding has an explicit `msa_mask`; models
must respect it when calculating profiles or deletion statistics. Public
validation diagnostics weight targets equally across unequal batches, with
undefined RMSDs excluded individually.

Track budget constants:

| Track | Seed | Crop size | MSA depth | Effective batch | Max steps | Sample budget | Residue budget | Parameter cap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `limited` | `0` | `256` | `192` | `8` | `30,000` | `240,000` | `61,440,000` | unrestricted |
| `research_large` | `0` | `256` | `192` | `8` | `120,000` | `960,000` | `245,760,000` | unrestricted |
| `unlimited` | submitter-defined | submitter-defined | submitter-defined | `8` | submitter-defined | unrestricted | unrestricted | unrestricted |

All tracks use deterministic public validation settings (`center`, `top`) when the track defines them.

Runtime reproducibility:

- Deterministic seeding and distinct worker random streams across ranks.
- Versioned checkpoint sampling state records epoch/cursor, loader generators, every rank's RNG and a canonical chain/weight digest.
- Resume replays the consumed loader prefix before restoring model RNG, preserving sampling across interruptions under the same loader configuration.
- Changed data bytes, sampling weights, worker/batch settings, world size or device type fail resume checks. Older checkpoints missing sampling state require a fresh training run and remain usable for prediction/scoring.
- Atomic checkpoint replacement preserves the previous checkpoint on write failure. Every rank receives save failures; resume validates all rank RNG states and required AMP scaler/scheduler state before changing history.
- Exposure counters must be exact nonnegative integers consistent with the step,
  effective batch, crop bounds and counter aliases before resume changes history.

See the [training sampling contract](API.md#training-sampling-and-multiple-gpus) for replay costs and limits. These checks do not establish bitwise CUDA reproducibility across environments.

## 8) Scoring and Ranking

Primary metric:
- `FoldScore = 0.25*GDT_HA-Ca + 0.09375*(lDDT-all-atom14 + CADaa-atom14 + SG-atom14 + SC-atom14) + 0.125*(MolProbity-clash-atom14 + BB-atom14 + DipDiff-atom14)`
- `GDT_HA-Ca` uses C-alpha coordinates from atom14 slot `1`, threshold-specific GDT superpositions, and high-accuracy GDT cutoffs `[0.5, 1.0, 2.0, 4.0]`
- `lDDT-all-atom14` uses true-structure neighborhood cutoff `15.0A`, lDDT thresholds `[0.5, 1.0, 2.0, 4.0]`, atom14 label masks, no intra-residue atom pairs, and equal chain weighting
- `CADaa-atom14` scores all-resolved-atom contact preservation
- `SG-atom14` uses the CASP SphereGrinder radius/cutoff setup: target-centered `6A` atom spheres, local superposition, and the mean of residue fractions below `2A` and `4A` local RMSD
- `SC-atom14` scores chi1/chi2 side-chain dihedral agreement using residue identity, symmetry-aware terminal groups, heavier chi1 weighting, and target-side burial weighting
- `MolProbity-clash-atom14` scores atom-name-aware heavy-atom van der Waals overlaps above `0.4A` per 1000 atoms
- `BB-atom14` scores phi, psi, and omega backbone dihedral agreement with equal angle-class weighting
- `DipDiff-atom14` scores local three-residue C-alpha/O distance-window agreement
- The weights are the CASP15 formula weights renormalized over the structure-derived components supported by the official atom14 output contract
- `ASE` requires submitted confidence estimates and `reLLG_lddt` requires crystallographic molecular-replacement scoring, so they are not official rank components
- Metric references: [CASP15 assessors formula](https://predictioncenter.org/casp15/zscores_final.cgi?formula=assessors) and [Simpkin et al., 2023](https://pubmed.ncbi.nlm.nih.gov/37746927/)

Leaderboard ranking metric:
- `limited`: `foldscore_auc_hidden`, trapezoidal AUC over cumulative samples on `[0, B_sample]`
- `research_large`: `foldscore_auc_hidden`, trapezoidal AUC over cumulative samples on `[0, B_sample]`
- `unlimited`: `final_hidden_foldscore`

Why AUC: the competition is a slowrun, so learning speed matters. A method that learns robust geometry earlier in the sample budget should be rewarded, even when final checkpoint scores are close.

Secondary metrics:
- `final_hidden_foldscore`
- `foldscore_at_steps` (`0`, `1000`, `2000`, `5000`, `last` by default)
- `foldscore_at_samples`
- `final_hidden_<component>`, `<component>_auc_hidden`, `<component>_at_steps`, and `<component>_at_samples` for each FoldScore component
- `gdt_ts_ca`, `lddt_ca`, and `lddt_backbone_atom14` diagnostics
- public val score is retained for diagnostics only

Official scoring requires exact prediction lengths and finite coordinates and
metrics; malformed targets fail the run rather than disappearing from an
average. The maintainer fixes the manifest, split, prediction root, crop,
milestones and exposure counts independently of submission-writable prediction
metadata. Official AUC tracks use the documented checkpoint schedule; choosing
different checkpoints changes the evaluation and is rejected.

Canonical result artifact:
- `runs/<run_name>/result.json`

Rendered leaderboard entries include `name` and `submission_path` fields. The
README renderer uses them for the linked `Name` column so readers can inspect
the accepted submission implementation directly.

## 9) Official Hidden Pipeline

Canonical maintainer runner:

```bash
python scripts/run_official.py --submission submissions/<name> --track <track_id> --team "<team or individual name>" --update-leaderboard
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

The Docker wrapper builds a selected set of tracked source files from a trusted
checkout and pins the image ID across three separate containers: public training
and validation, hidden prediction, and fresh scoring. It never mounts the host
repository. Scoring receives read-only run inputs with separate output-file
mounts; only that stage can update the leaderboard. Submission configs stay
inside the submission directory. Host asset relocation uses the wrapper's
documented environment variables; stage, Python and scoring-path overrides are
rejected. Direct local Python execution assumes trusted maintainer code and is
not an isolation boundary.

Mounted processed roots are limited to manifest-selected NPZs and known feature
sidecars (`preprocess_meta.json`, `msa_row_filter_audit.json`, plus
`msa_row_filter_audit.train.json` and `msa_row_filter_audit.val.json` for public
features only). Extra files, subdirectories and links fail before execution. The image includes its required
public source lock and excludes vendor example structures and research assets.

Official orchestration entrypoints:
- `predict.py` for prediction only
- `score.py` for sealed scoring with feature-side residue identities plus labels
- `scripts/run_official.py` for orchestration

Hidden leaderboard runs must execute in a sealed runtime. The supported maintainer path is:

Containerized no-network execution:

```bash
bash scripts/run_official_docker.sh --submission submissions/<name> --track <track_id> --team "<team or individual name>" --update-leaderboard
```

The Docker build copies only source-oriented files into the image. Generated datasets, hidden assets, checkpoints, local environments, and run outputs are excluded from the build context and are supplied through runtime mounts.

Modal no-network execution for checkpoints already stored in the `nanofold-runs` Modal volume:

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

For long Modal evaluations, maintainers can detach prediction and scoring separately and then update the local leaderboard from the result stored in the `nanofold-runs` volume:

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

Run the upload command the first time a Modal maintainer environment is prepared, or whenever public/hidden assets change. The Modal runner uses separate prediction and scoring functions. The prediction function mounts public data, hidden features, and hidden fingerprints. The scoring function mounts saved predictions, hidden labels, hidden features, hidden fingerprints, and the private hidden lock. Both stages request the configured GPU; scoring uses it for atom14 structural metric calculations.

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
