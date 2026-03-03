## What changed?

Describe what you changed relative to baseline.
List any files added under your submission folder (for example `submission.py`, custom modules, utilities).

## Why should it help?

Give intuition and (ideally) references / ablations.

## Competition compliance checklist

- [ ] Used only the provided benchmark data (no external data, weights, or template/MSA searches).
- [ ] Kept dataset manifests fixed (`data/manifests/train.txt` and `data/manifests/val.txt`).
- [ ] Model outputs C-alpha coordinates per residue (`(L, 3)` in Angstrom).

## Required run metadata (limited track)

- max_steps:
- effective_batch_size:
- crop_size:
- seed:
- hardware:
- wall_clock_time:
- commit:

## How to run

```bash
python train.py --config submissions/<your_name>/config.yaml
python eval.py --config submissions/<your_name>/config.yaml --ckpt runs/your_name_run1/checkpoints/ckpt_last.pt
```
