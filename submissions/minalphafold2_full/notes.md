## Submission Summary

This submission benchmarks the full paper-spec minAlphaFold2 monomer profile on
the `limited` track budget.

- upstream repository: `https://github.com/ChrisHayduk/minAlphaFold2`
- pinned commit: `f2e6c237fe4b98f2ca94ebdcc5ecc06dc04852c3`
- vendored path: `third_party/minAlphaFold2`
- model profile: `third_party/minAlphaFold2/configs/alphafold2.toml`

`submission.py` reuses the nanoFold minAlphaFold2 adapter from
`submissions/minalphafold2/submission.py`. The adapter converts nanoFold's
official tensors into minAlphaFold2 features, derives AlphaFold2-style
supervision, trains with `minalphafold.losses.AlphaFoldLoss`, and returns
`pred_atom14 = output["atom14_coords"]`.

The limited-track parameter cap is `100,000,000` trainable parameters. This
full-profile run has about `94.2M` trainable parameters and keeps the standard
limited-track budget: `10,000` optimizer steps, effective batch size `2`,
crop size `256`, and MSA depth `192`.

The train/fine-tune handoff scales AlphaFold2's initial/fine-tune sample ratio
into the official budget. Fine-tuning starts at step `8696`, leaving `1304`
optimizer updates for the fine-tune window. The loss linearly blends from the
initial loss to the fine-tune loss for the first `1000` fine-tune steps so the
handoff is smooth instead of a hard objective switch.

## Method Rationale

This run measures whether the original AlphaFold2-sized minAlphaFold2
architecture can make better use of the scarce official public data than the
tiny reference. It keeps the public data contract fixed and changes only model
capacity, recycling count, and the smoother fine-tune loss transition.

## Competition compliance checklist

- [x] Used only the provided benchmark data (no external data, weights, or template/MSA searches).
- [x] Kept dataset manifests fixed (`data/manifests/train.txt` and `data/manifests/val.txt`).
- [x] Model outputs atom14 coordinates per residue (`(L, 14, 3)` in Angstrom).
- [x] `run_batch(..., training=False)` does not depend on supervision labels (`ca_coords`, `ca_mask`).

## Required run metadata (limited track)

- max_steps: 10000
- effective_batch_size: 2
- sample_budget: 20000
- residue_budget: 5120000
- crop_size: 256
- seed: 0
- trainable_parameters: about 94.2M
- hardware: Modal H100 requested
- wall_clock_time: pending Modal benchmark completion
- commit: pending merge commit hash

## How to run

```bash
python scripts/validate_submission.py --submission submissions/minalphafold2_full --track limited --strict
python train.py --config submissions/minalphafold2_full/config.yaml --track limited --official
```

Modal setup and run:

```bash
modal run scripts/modal_train.py --upload-data --skip-train
NANOFOLD_MODAL_GPU=H100 modal run scripts/modal_train.py \
  --config submissions/minalphafold2_full/config.yaml \
  --track limited \
  --reset-run
```
