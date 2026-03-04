# nanoFold Submission API

Public API contract for competition submissions.

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

Current schema:

- `data.processed_features_dir`
- `data.processed_labels_dir`
- `data.train_manifest`
- `data.val_manifest`
- `data.crop_size`
- `data.msa_depth`
- `data.batch_size`

Legacy `data.processed_dir` is not the official schema.

## Batch Contract by Mode

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

### Inference mode (`training=False`)

Guaranteed keys:
- all non-supervision keys above

Not present:
- `ca_coords`
- `ca_mask`

Runtime strips supervision keys internally in inference mode even if caller accidentally passes them.

## `run_batch` Output Contract

Always required:
- `pred_ca`: tensor `(B, L, 3)` floating dtype, finite values

When `training=True`:
- `loss`: scalar finite tensor
- `loss.requires_grad` must be `True`

Runtime rejects:
- non-dict outputs
- missing `pred_ca`
- wrong `pred_ca` shape/dtype
- NaN/Inf in `pred_ca` or `loss`
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

## Budget Definitions

- `effective_batch_size = data.batch_size * train.grad_accum_steps`
- `B_sample = train.max_steps * effective_batch_size`
- `B_res = train.max_steps * effective_batch_size * data.crop_size`

## lDDT-Ca Metric

Implementation uses:
- true-structure neighborhood cutoff `15.0A`
- thresholds `[0.5, 1.0, 2.0, 4.0]`
- residue mask logic over valid residue pairs

Sanity vector:
- if `pred_ca == true_ca` and at least two residues are valid, score is `1.0`.
