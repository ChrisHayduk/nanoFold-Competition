# nanoFold Submission API

This document is the public API contract for competition submissions.

## Required Entrypoint

Each submission must provide `submission.py` with:

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

Competition submissions should use `submission.path` within the submission folder.

## `run_batch` Output Contract

Required:
- always return `pred_ca` with shape `(B, L, 3)` and floating dtype

Training (`training=True`):
- must also return scalar tensor `loss`

Inference (`training=False`):
- batch may be unlabeled (`ca_coords` and `ca_mask` removed)
- output must not depend on supervision-only keys

Runtime rejects:
- non-dict outputs
- missing `pred_ca`
- wrong shape/dtype
- NaN/Inf in `pred_ca` or `loss`
- non-scalar `loss`

## Batch Keys

Typical batch includes:
- `chain_id: list[str]`
- `aatype: (B, L)`
- `msa: (B, N, L)`
- `deletions: (B, N, L)`
- `template_aatype: (B, T, L)`
- `template_ca_coords: (B, T, L, 3)`
- `template_ca_mask: (B, T, L)`
- `residue_mask: (B, L)`
- training-only supervision keys: `ca_coords`, `ca_mask`

## Track Enforcement

Use `--track` to select policy (`tracks/*.yaml`). In official mode (`--official`) the runner enforces:
- fixed config constants
- fixed manifest paths
- dataset fingerprint
- strict no-missing requirement

## Budget Definitions

- sample budget: `B_sample = max_steps * effective_batch_size`
- residue budget: `B_res = max_steps * effective_batch_size * crop_size`

where `effective_batch_size = data.batch_size * train.grad_accum_steps`.

## Scoring (`lDDT-Ca`)

The implementation uses:
- pairwise C-alpha distances
- neighborhood cutoff `15.0A` in true structure
- thresholds `[0.5, 1.0, 2.0, 4.0]`
- mean over thresholds, then mean over residues with at least one valid neighbor

Deterministic toy vector:
- if `pred_ca == true_ca` and at least two residues are unmasked, score is exactly `1.0`.
