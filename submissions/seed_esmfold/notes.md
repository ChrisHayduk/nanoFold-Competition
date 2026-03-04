## What changed?

This is a seed submission that introduces an ESMFold-inspired train-from-scratch trunk under
`submissions/seed_esmfold/`.

Added files:
- `submission.py` (submission entrypoint)
- `esmfold_seed_model.py` (model implementation)
- `THIRD_PARTY_LICENSES.md` (attribution/license for reused public code ideas)

Architecture summary:
- Single-sequence encoder only (amino-acid tokens + positional embedding); no MSA features are used.
- Pair state initialized with ESMFold-style relative positional embedding plus sequence-conditioned pair init.
- Repeated seq/pair trunk blocks with pair-biased sequence attention and outer-product pair updates.
- ESMFold-style auxiliary heads/losses:
  - pair distogram cross-entropy
  - per-residue confidence (pLDDT-style) cross-entropy
- C-alpha head outputs `(B, L, 3)` coordinates.

## Why should it help?

Compared with a plain sequence-only baseline, this trunk introduces explicit pair-state modeling,
pair-conditioned attention, and ESMFold-like auxiliary supervision while staying trainable from scratch
under the benchmark's fixed budget and no-pretraining constraint.

## Competition compliance checklist

- [x] Used only the provided benchmark data (no external data, weights, or template/MSA searches).
- [x] Kept dataset manifests fixed (`data/manifests/train.txt` and `data/manifests/val.txt`).
- [x] Model outputs C-alpha coordinates per residue (`(L, 3)` in Angstrom).

## Required run metadata (limited track)

- max_steps: 10000
- effective_batch_size: 2
- residue_budget: 5120000
- crop_size: 256
- seed: 0
- hardware: pending maintainer benchmark run
- wall_clock_time: pending maintainer benchmark run
- commit: pending merge commit hash

## How to run

```bash
python train.py --config submissions/seed_esmfold/config.yaml
python eval.py --config submissions/seed_esmfold/config.yaml --ckpt runs/seed_esmfold_v1/checkpoints/ckpt_last.pt
```
