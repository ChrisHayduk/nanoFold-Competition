## What changed?

Template submission scaffold for participants.

Suggested edits:
- briefly describe architectural/training differences from baseline
- list any added files under your submission directory
- state any key hyperparameter choices

## Why should it help?

Summarize the expected gains and intuition behind your method.

## Competition compliance checklist

- [x] Used only the provided benchmark data (no external data, weights, or template/MSA searches).
- [x] Kept dataset manifests fixed (`data/manifests/train.txt` and `data/manifests/val.txt`).
- [x] Model outputs C-alpha coordinates per residue (`(L, 3)` in Angstrom).
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
python train.py --config submissions/<your_name>/config.yaml --track limited_large_v3
python eval.py --config submissions/<your_name>/config.yaml --ckpt runs/your_name_run1/checkpoints/ckpt_last.pt --track limited_large_v3
```
