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

This keeps the benchmark focused on fixed-data learning. Public validation can be used for debugging, but hidden ranking is sealed and label-only scoring never imports submission hooks.

## Budget Definitions

- `effective_batch_size = data.batch_size * train.grad_accum_steps`
- `B_sample = train.max_steps * effective_batch_size`
- `B_res = train.max_steps * effective_batch_size * data.crop_size`

## FoldScore Metric

Official ranking uses equal chain weighting and:

```text
FoldScore = 0.55*lDDT-Ca + 0.30*lDDT-backbone-atom14 + 0.15*lDDT-all-atom14
```

Hidden leaderboard ranking is `foldscore_auc_hidden`, trapezoidal AUC over cumulative samples from `0` to `B_sample`. The tie-breaker is `final_hidden_foldscore`.

`lDDT-Ca` remains reported as a diagnostic metric.

Implementation uses:
- true-structure neighborhood cutoff `15.0A`
- thresholds `[0.5, 1.0, 2.0, 4.0]`
- atom/residue mask logic over valid pairs

Sanity vector:
- if predictions match labels exactly and at least two points are valid, score is `1.0`.
