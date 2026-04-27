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

## Batch Contract by Mode

Training batches include labels because submissions need supervision to learn. Inference batches do not. The runtime strips supervision keys itself when `training=False`, which protects hidden evaluation even if a caller accidentally passes a labeled batch.

### Training mode (`training=True`)

Guaranteed keys:
- `chain_id: list[str]`
- `aatype: (B, L) int`
- `msa: (B, N, L) int`
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
- `resolution: (B,) float` — Å, 0.0 if unknown

Typical use:

```python
ca_from_atom14 = batch["atom14_positions"][..., 1, :]  # slot 1 == Cα
backbone_valid = batch["atom14_mask"][..., :4].all(dim=-1)
```

### Inference mode (`training=False`)

Guaranteed keys:
- all non-supervision keys above
- runtime forwards only the inference allowlist: `chain_id`, `aatype`, `msa`, `deletions`, `template_*`, `residue_mask`

Not present:
- `ca_coords`, `ca_mask`
- `atom14_positions`, `atom14_mask`, `residue_index`, `resolution`

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
- model parameter cap (`model.max_params`) if set
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
the official residue identities:

```text
FoldScore =
  0.25*GDT_HA-Ca
+ 0.09375*(lDDT-all-atom14 + CADaa-atom14 + SG-atom14 + SC-atom14)
+ 0.125*(MolProbity-clash-atom14 + BB-atom14 + DipDiff-atom14)
```

Hidden leaderboard ranking is track-specific. `limited` and `research_large` use `foldscore_auc_hidden`, trapezoidal AUC over cumulative samples from `0` to `B_sample`, with `final_hidden_foldscore` as the tie-breaker. `unlimited` uses `final_hidden_foldscore` because it has no shared sample budget.

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
