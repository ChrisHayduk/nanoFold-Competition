## What changed?

This adds a second seed submission with an OpenFold/AlphaFold2-style architecture under
`submissions/seed_openfold/`.

Added files:
- `submission.py` (entrypoint and loss composition)
- `openfold_seed_model.py` (Evoformer/structure-style model)
- `THIRD_PARTY_LICENSES.md` (OpenFold attribution/license)

Model summary:
- Uses **MSA tokens + deletions** as primary input.
- Uses **template features** (`template_aatype`, `template_ca_coords`, `template_ca_mask`) when present.
- Evoformer-style stack:
  - MSA row attention with pair bias
  - MSA column attention
  - MSA transition
  - outer-product mean to pair representation
  - triangle multiplicative updates
  - pair axial attentions + transition
- Structure-style stack over single representation to predict C-alpha coordinates.
- Auxiliary heads/losses:
  - distogram cross-entropy
  - pLDDT-style confidence cross-entropy

## Why should it help?

OpenFold-style MSA/pair co-evolution processing is a stronger inductive bias than a plain sequence-only
transformer. Template pair features can inject structural priors when good template hits are available.

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
# preprocessing should include templates (default behavior in scripts/preprocess.py)
python train.py --config submissions/seed_openfold/config.yaml
python eval.py --config submissions/seed_openfold/config.yaml --ckpt runs/seed_openfold_v1/checkpoints/ckpt_last.pt
```
