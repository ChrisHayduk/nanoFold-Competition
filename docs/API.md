# nanoFold Submission API

Public API contract for competition submissions.

This API is intentionally small. nanoFold is meant to compare ideas under data scarcity, so the runtime gives every submission the same processed features, the same training/evaluation hooks, and the same sealed inference path. The interesting work should live in the model, objective, optimizer, and curriculum, not in bespoke data access.

Submissions return atom14 predictions because the leaderboard should reward protein geometry beyond a Cα trace. The runtime derives Cα coordinates from atom14 slot 1 whenever a diagnostic Cα view is needed.

## Required Entrypoint

Each submission provides `submission.py` with:

1. `build_model(cfg) -> torch.nn.Module`
2. `build_optimizer(cfg, model) -> optimizer-like object`
3. `run_batch(model, batch, cfg, training) -> dict`

Optional:

1. `build_scheduler(cfg, optimizer) -> scheduler-like object | None`

## Config Wiring

`config.yaml` must set exactly one of:

```yaml
submission:
  path: submission.py
```

or

```yaml
submission:
  module: package.module.path
```

Competition submissions should use `submission.path` under `submissions/<name>/`.

## Data Config Schema

- `data.processed_features_dir`
- `data.processed_labels_dir`
- `data.train_manifest`
- `data.val_manifest`
- `data.crop_size`
- `data.msa_depth`
- `data.batch_size`

### Training sampling and multiple GPUs

`train.py` supports `torchrun` with one CUDA device per rank and the NCCL
backend. `data.batch_size` is per rank; the configured effective batch remains
`data.batch_size * train.grad_accum_steps` across all ranks. The runner divides
accumulation by `WORLD_SIZE` and rejects combinations that cannot preserve that
batch exactly. Only rank zero writes checkpoints and runs public validation.
`NANOFOLD_DDP_TIMEOUT_SECONDS`, when set, must be a positive integer.

Training normally shuffles the manifest each epoch, giving equal weight to selected targets. A group containing `m` selected targets contributes `m` exposures in a complete traversal before incomplete-batch and distributed-tail drops. Inverse-group weighting is a separate sampling ablation; complete group coverage alone does not equalize group exposure. For an explicitly labeled
weighted-sampling study, set `data.train_sample_weights` to a JSON mapping from
chain ID to weight, or a text file with `chain_id weight` or `chain_id,weight`
rows, and set `data.allow_train_sample_weights: true`. Every training chain must
have a finite, nonnegative weight, and the total must be positive. Sampling is
with replacement; zero-weight chains are not drawn. The same seed and epoch
produce the same global weighted draw before partitioning it across ranks.
Weights do not add data or change the configured sample budget.

Unweighted distributed sampling never pads an epoch with duplicate chains. It
drops the shuffled remainder that cannot be partitioned across ranks; the
loader also drops incomplete local batches. The sampling contract records this
policy, the population size, and the number of samples available per epoch.
Chains omitted at an epoch boundary can reappear in later shuffled epochs.

Training checkpoints include a versioned `sampling_state`: the manifest-aligned sampling-weight digest, loader/sampler configuration, epoch and consumed-batch cursor, epoch-start generator state, and RNG state for every rank. Resume reconstructs the ordered loader, replays consumed batches, then restores that rank's model RNG. Worker seeds differ across ranks while the distributed sampling plan uses a shared seed. Replaying the current epoch prefix adds startup I/O but does not count as new training exposure.

Resume requires the same effective sampling population and weights, dataset contents, world size, device type, worker count, batch/accumulation settings and crop/MSA configuration. Changing weights or feature/label bytes at the same file path is detected. Verified runs bind their checked fingerprint; other runs stream-hash train and validation files once at startup. Checkpoints without supported sampling state cannot resume training safely and produce an explicit error; they remain usable for prediction/scoring. `--allow-resume-mismatch` does not bypass sampling-state checks. Deterministic sample/augmentation replay does not promise bitwise numerical agreement across different hardware or library versions.

Checkpoint writes use staged, flushed files followed by atomic replacement. A failed write preserves the previous checkpoint, and rank-zero save failures propagate to every rank. Resume validates RNG states for all ranks and requires valid AMP scaler state when AMP is enabled, plus the configured scheduler's saved state, before changing run history.

Exposure counters must be nonnegative integers consistent with the checkpoint
step and effective batch size. Samples must equal `step * effective_batch_size`;
nonpadding and cropped residue counts must lie within the corresponding sample
and crop bounds. Conflicting counter aliases fail validation before run history
changes.

### Candidate manifest policy

Candidate generation is a maintainer interface; participant data configuration and published manifest pins remain unchanged. Qualified regeneration explicitly selects `scripts/stage_dataset_candidate.py --sampling-policy cath-v4`; the staging default remains `cath-v3`. It requires original `--cluster-tsv`, unchanged `--split-group-tsv` and `--split-group-receipt`, plus `--representative-eligibility-receipt` and `--approved-group-exclusions`. `scripts/build_representative_eligibility.py` binds ranked raw-query and strict historical-structure validation before selection. It accepts only the first passing member after a complete failed rank prefix, or an exhaustively failed original unit. A whole group may be excluded only when all its units fail and the exact exhausted-group set matches the approved catalog.

CATH v4 covers every retained group and confines each group to one split, with at most one validated representative per original unit. Public and hidden validation each use one target per group; training covers the remaining groups before filling extra target slots. For the approved six-group exclusion catalog, 8,850 of the original 8,856 groups remain, requiring 6,850 training anchors and 3,150 extras for 10,000/1,000/1,000 targets. The graph is preserved, and strict label thresholds are not relaxed. `cath-v3` still requires full original-group coverage; `cath-v2` retains its one-representative-per-original-unit contract. These policies keep distinct replay semantics.

Group ownership uses the public seed. The hidden salt affects member choice and order only inside fixed hidden groups; anchor-based quotas need not equal the chosen members' strata. Actual distributions must pass quality checks. Full retained coverage makes hidden groups inferable from the known retained universe and public complement; private files and the sealed label boundary do not imply membership secrecy.

Both source locks bind graph, original and validated representative populations, eligibility evidence, approved exclusions and source inputs. Verification replays membership/order and checks retained coverage independently. Completion also requires both complete cache-target audits with matching inputs, counts, thresholds and artifact hashes. `candidate_status.json` records `target_sequences_audited: true` only after those gates; processed tensors, MSA audits, fingerprints and adoption remain separate. See [dataset sampling](DATA_SAMPLING.md) for executable commands and group-concentration diagnostics.

Maintainer exhaustive MSA tools accept `--checkpoint-dir` with `--exhaustive`; the auditor also requires `--run-mmseqs`. `--source-batch-size`, `--query-batch-size`, `--search-disk-reserve-gib` and `--search-max-rss-gib` control bounded jobs and sampled resource checks. Repeating an identical command reuses only validated complete jobs. A final filter/audit requires all jobs; inventory or partial results are not consumable filters. These tools do not change submission tensor fields. See [bounded exhaustive searches](DATA_SAMPLING.md#bounded-resumable-exhaustive-msa-searches).

## Batch Contract by Mode

Training batches include labels because submissions need supervision to learn. Inference batches do not. The runtime strips supervision keys itself when `training=False`, which protects hidden evaluation even if a caller accidentally passes a labeled batch.

### Training mode (`training=True`)

Guaranteed keys:
- `chain_id: list[str]`
- `aatype: (B, L) int`
- `msa: (B, N, L) int`
- `msa_mask: (B, N, L) bool` — true for biological MSA cells, including gaps;
  false for batch padding in either the row or residue dimension
- `deletions: (B, N, L) int`
- `template_aatype: (B, T, L) int`
- `template_ca_coords: (B, T, L, 3) float`
- `template_ca_mask: (B, T, L) bool`
- `residue_mask: (B, L) bool`
- `ca_coords: (B, L, 3) float`
- `ca_mask: (B, L) bool`
- `atom14_positions: (B, L, 14, 3) float` — full atom14 layout (AF2 supplement 1.2.1)
- `atom14_mask: (B, L, 14) bool` — True where coordinate was present in mmCIF

Optional metadata keys:
- `residue_index: (B, L) int` — contiguous 0..L-1 per chain (AF2 supplement 1.2.9)
- `between_segment_residues: (B, L) int` — zero for these single-chain targets
- `resolution: (B,) float` — Å, 0.0 if unknown

Typical use:

```python
ca_from_atom14 = batch["atom14_positions"][..., 1, :]  # slot 1 == Cα
backbone_valid = batch["atom14_mask"][..., :4].all(dim=-1)
```

Processed NPZ validation checks token ranges, nonnegative deletions, query/MSA agreement, finite coordinates, and matching feature/label lengths and residue indices. CA coordinates and masks must agree with atom14 slot 1. Query/structure sequence mismatches have all atom labels masked. Schema-3 preprocessing adds matching `preprocess_config_sha256` and `source_inputs_sha256` provenance to each pair; these are storage/audit metadata, not model inputs. Fingerprinting rejects mixed configurations and stored chain IDs that disagree with the manifest. Schema 3 requires the chain ID in both files.

Maintainer setup wrappers bind preprocessing to the source-locked chain cache.
Direct `scripts/preprocess.py` calls enable this with `--chain-data-cache <path>`.
The acquired ungapped query must exactly equal the selected canonical cache
sequence before coordinate projection. Missing or inconsistent cache records
fail, and the cache SHA256 and selected chain/sequence mapping enter stored
configuration and source provenance. Resume also checks processed `aatype`
against the expected sequence; changed cache bytes or a newly enabled cache
binding require fresh output directories. These checks add no model inputs. Maintainers can run `scripts/audit_target_identity.py` with complete manifests and expected counts to verify raw and/or processed target identity independently; it requires actual assets and leaves labels, geometry and homology to their separate gates.

Coordinate projection requires all optimal deposited-polymer alignments to agree
on every observed residue's query position; ambiguous observed mappings fail.

`scripts/fetch_openproteinset_mmcif.py` requires a manifest, fresh private output
root, archive byte size, exact quoted ETag and explicit version ID. Its
`retrieval-receipt.json` binds the inputs and retrieved members and reports
`retrieved_unvalidated_structures`; it is not a biological identity or dataset
acceptance receipt. Pass its `pdb_data/mmcif_files` directory to `--mmcif-root`.
Transient transport failures receive at most four attempts per range with
1, 2 and 4 second backoffs. Integrity failures abort immediately, and the private
receipt records retry attempts.

Maintainers can run `scripts/audit_label_geometry.py --chain-data-cache CACHE --split NAME MANIFEST FEATURES LABELS COUNT --out-dir PRIVATE_AUDIT`, repeating `--split` for train, public validation and hidden validation. It requires complete selected manifests, exact cache/feature/label identities, valid feature and label schemas, and atom14 masks supported by each actual residue type. Nonexistent observed atoms fail. The receipt binds source and asset hashes and reports masked CA and peptide C–N distances, skipping explicit segment breaks, nonconsecutive residue indices and missing atoms. Distance outliers are diagnostics only; they do not reject experimental structures. An optional verified `--split-group-tsv` adds group summaries. Keep all reports private.

Feature archives must not contain supervision arrays. A features-only loader
rejects such archives before passing data to submission code. Manifests must be
nonempty and contain unique chain IDs; fingerprinted splits must be disjoint.
Requiring labels also requires an explicit label directory. Models must use
`msa_mask` when aggregating MSA profiles or deletion features so another chain's
MSA depth cannot change the current chain's features.

### Inference mode (`training=False`)

Guaranteed keys:
- all non-supervision keys above
- runtime forwards only the inference allowlist: `chain_id`, `aatype`, `msa`, `msa_mask`, `deletions`, `template_*`, `residue_index`, `between_segment_residues`, `residue_mask`

Not present:
- `ca_coords`, `ca_mask`
- `atom14_positions`, `atom14_mask`, `resolution`

Runtime strips supervision keys internally in inference mode even if caller accidentally passes them.

## `run_batch` Output Contract

- `pred_atom14`: tensor `(B, L, 14, 3)` floating dtype, finite values
- the runtime derives `pred_ca` from `pred_atom14[:, :, 1, :]` after validating the atom14 tensor

When `training=True`:
- `loss`: scalar finite tensor
- `loss.requires_grad` must be `True`

Runtime rejects:
- non-dict outputs
- missing `pred_atom14`
- wrong `pred_atom14` shape/dtype
- NaN/Inf in predictions or `loss`
- non-scalar `loss`
- non-differentiable `loss` in training mode

## Track and Policy Enforcement

Use `--track` to select policy (`tracks/*.yaml`).

In `--official` mode runtime enforces:
- override+validate on immutable track constants
- fixed manifest path policy
- pinned manifest SHA checks (when present in track)
- dataset fingerprint verification
- trainable parameter count reporting; `model.max_params` is only enforced if a custom track sets it
- official prediction sanitizes `data.processed_labels_dir` before submission hooks are constructed

This keeps the benchmark focused on fixed-data learning. Public validation can be used for debugging, but hidden ranking is sealed and scoring never imports submission hooks.

## Budget Definitions

- `effective_batch_size = data.batch_size * train.grad_accum_steps`
- `B_sample = train.max_steps * effective_batch_size`
- `B_res = train.max_steps * effective_batch_size * data.crop_size`

During `train.py`, the `cfg` passed to `run_batch` also includes
`cfg["_runtime"]` with `step`, `cumulative_samples_seen`, `max_steps`, and
`sample_budget`. Submissions can use this for schedule changes such as
stage-specific losses or learning-rate handoffs while staying inside the
official budget.

## FoldScore Metric

Official ranking uses equal chain weighting and a CASP15-inspired raw score
over the CASP metrics that can be computed reproducibly from `pred_atom14` and
the official residue identities.

Each prediction must have the exact expected residue count and finite atom14
coordinates. Scoring rejects truncated or extra residues, invalid residue
identities, malformed labels, and nonfinite metrics; it does not silently score
a shared prefix or omit failed targets. Official scoring binds the manifest,
split, crop, prediction root, checkpoint schedule, and exposure counts to
maintainer-supplied inputs rather than trusting a prediction summary.

The composite is:

```text
FoldScore =
  0.25*GDT_HA-Ca
+ 0.09375*(lDDT-all-atom14 + CADaa-atom14 + SG-atom14 + SC-atom14)
+ 0.125*(MolProbity-clash-atom14 + BB-atom14 + DipDiff-atom14)
```

Hidden leaderboard ranking is track-specific. `limited` and `research_large` use `foldscore_auc_hidden`, trapezoidal AUC over cumulative samples from `0` to `B_sample`, with `final_hidden_foldscore` as the tie-breaker. `unlimited` fixes effective batch size at `8` but uses `final_hidden_foldscore` because it has no shared max-step or sample budget.

Leaderboard JSON entries include `name` and `submission_path` fields. The
rendered README leaderboard hyperlinks `name` to `submission_path` for the
accepted submission implementation.

Scoring reads feature-side residue identities to interpret atom14 slots for
side-chain torsions and atom-name-aware clashes. `ASE` is not included because
it requires submitted confidence estimates. `reLLG_lddt` is not included
because it requires crystallographic molecular-replacement scoring.

| Output key | CASP15 role | nanoFold computation |
|---|---|---|
| `gdt_ha_ca` | global fold accuracy | C-alpha GDT_HA with threshold-specific GDT superpositions |
| `lddt_atom14` | local all-atom agreement | lDDT over resolved atom14 coordinates, excluding intra-residue atom pairs |
| `cad_atom14` | contact-area agreement | all-resolved-atom contact preservation |
| `sg_atom14` | local atomic environment agreement | target-centered `6A` atom spheres, local superposition, `2A`/`4A` RMSD cutoffs |
| `sc_atom14` | side-chain geometry | chi1/chi2 agreement with symmetry, chi weighting, and burial weighting |
| `molprobity_clash_atom14` | model stereochemical plausibility | atom-name-aware heavy-atom van der Waals overlaps above `0.4A` |
| `bb_atom14` | backbone geometry | phi/psi/omega dihedral agreement with equal angle-class weighting |
| `dipdiff_atom14` | neighboring-residue geometry | three-residue C-alpha/O local distance-window agreement |

`GDT_TS-Ca`, `lDDT-Ca`, and backbone atom14 lDDT are also reported as diagnostic
metrics.

Implementation uses:
- GDT_HA thresholds `[0.5, 1.0, 2.0, 4.0]`
- lDDT true-structure neighborhood cutoff `15.0A`
- lDDT thresholds `[0.5, 1.0, 2.0, 4.0]`
- atom/residue mask logic over valid coordinates and residue identities
- the CASP15 formula weights renormalized over supported structure-derived components

Sanity vector:
- if predictions match labels exactly, at least three C-alpha atoms plus valid inter-residue atom14 pairs are present, and the predicted structure has no heavy-atom clashes under the clash component, score is `1.0`.
