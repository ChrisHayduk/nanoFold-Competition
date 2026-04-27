## Submission Summary

This submission uses the upstream `ChrisHayduk/minAlphaFold2` implementation as
the actual model code.

- upstream repository: `https://github.com/ChrisHayduk/minAlphaFold2`
- pinned commit: `f2e6c237fe4b98f2ca94ebdcc5ecc06dc04852c3`
- vendored path: `third_party/minAlphaFold2`

`submission.py` is intentionally just an adapter. It converts nanoFold's
official batch tensors into the feature tensors expected by
`minalphafold.model.AlphaFold2`, derives the upstream AlphaFold2-style
supervision tensors, trains with `minalphafold.losses.AlphaFoldLoss`, and
returns `pred_atom14 = output["atom14_coords"]`.

The model architecture is loaded directly from
`third_party/minAlphaFold2/configs/tiny.toml`.

The training protocol scales AlphaFold2's initial/fine-tune sample ratio into
the official budget. With `max_steps=10000` and effective batch size `2`,
fine-tuning starts at step `8696`, leaving `1304` optimizer updates for the
fine-tune loss. Fine-tune auxiliary loss weights ramp linearly for the next
500 steps. The learning-rate warmup and one-shot decay are scaled from the
same AlphaFold2 protocol proportions. The reference config saves
checkpoints and runs public validation every 1,000 steps, matching the
checkpoint cadence used for official hidden AUC evaluation.
Training runs in full precision to match the upstream minAlphaFold2 trainer's
numerics.

Because the limited-track reference uses a single recycling cycle, the
recycling LayerNorm parameters are frozen at their initial values. The model
still uses minAlphaFold2's forward path, but the config does not allocate
training budget to learning a previous-cycle embedding that is never observed.

## Method Rationale

minAlphaFold2 is a direct, readable AlphaFold2-style implementation with an
Evoformer trunk, recycling, invariant point attention, and structure-module
atom14 coordinate generation. It gives nanoFold a biological-prior reference
submission with a compact AlphaFold2-style training stack: masked-MSA,
distogram, backbone and all-atom FAPE, torsion, and pLDDT objectives.

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
python scripts/validate_submission.py --submission submissions/minalphafold2 --track limited --strict
python train.py --config submissions/minalphafold2/config.yaml --track limited --official
```
