## What changed?

This submission uses the upstream `ChrisHayduk/minAlphaFold2` implementation as
the actual model code.

- upstream repository: `https://github.com/ChrisHayduk/minAlphaFold2`
- pinned commit: `f2e6c237fe4b98f2ca94ebdcc5ecc06dc04852c3`
- vendored path: `third_party/minAlphaFold2`

`submission.py` is intentionally just an adapter. It converts nanoFold's
official batch tensors into the feature tensors expected by
`minalphafold.model.AlphaFold2`, calls that model, and returns
`pred_atom14 = output["atom14_coords"]`.

## Why should it help?

minAlphaFold2 is a direct, readable AlphaFold2-style implementation with an
Evoformer trunk, recycling, invariant point attention, and structure-module
atom14 coordinate generation. It gives nanoFold a real biological-prior
reference submission instead of a placeholder architecture named after larger
systems.

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
- hardware: pending maintainer benchmark run
- wall_clock_time: pending maintainer benchmark run
- commit: pending merge commit hash

## How to run

```bash
python scripts/validate_submission.py --submission submissions/minalphafold2 --track limited_large --strict
python train.py --config submissions/minalphafold2/config.yaml --track limited_large --official
```
