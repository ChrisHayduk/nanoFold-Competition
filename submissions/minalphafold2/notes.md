## Submission Summary

This submission uses the upstream `ChrisHayduk/minAlphaFold2` implementation as
the actual model code.

- upstream repository: `https://github.com/ChrisHayduk/minAlphaFold2`
- pinned commit: `c382fe70b628a445e8e17b2b9e680ca93951ffd9`
- vendored path: `third_party/minAlphaFold2`

`submission.py` is intentionally just an adapter. It converts nanoFold's
official batch tensors into the feature tensors expected by
`minalphafold.model.AlphaFold2`, derives the upstream AlphaFold2-style
supervision tensors, trains with `minalphafold.losses.AlphaFoldLoss`, and
returns `pred_atom14 = output["atom14_coords"]`.
Padded residues and nonexistent atom slots are zeroed in the returned coordinates.
Padded residues do not contribute to experimentally-resolved or geometry losses;
peptide/torsion supervision respects residue-index gaps and explicit segment breaks.

The model architecture is loaded directly from
`third_party/minAlphaFold2/configs/tiny.toml`.

The training protocol scales AlphaFold2's initial/fine-tune sample ratio into
the official budget. With `max_steps=30000` and effective batch size `8`,
fine-tuning starts at step `26087`, leaving `3913` optimizer updates for the
fine-tune window. The loss linearly blends from the initial loss to the
fine-tune loss for the next 500 steps. The learning-rate warmup and one-shot
decay are scaled from the
same AlphaFold2 protocol proportions. The reference config saves
checkpoints and runs public validation every 1,000 steps, matching the
checkpoint cadence used for official hidden AUC evaluation.
Training runs in full precision to match the upstream minAlphaFold2 trainer's
numerics.

Because the limited-track reference uses a single recycling cycle, the
recycling LayerNorm parameters are frozen at their initial values. The model
still uses minAlphaFold2's forward path, but the config does not allocate
training budget to learning a previous-cycle embedding that is never observed.

## Recipe controls

The adapter validates its configuration before model construction. `experiment.fape`
accepts `backbone_clamp_mode` values `fully_clamped` (default), `fully_unclamped`,
`batchwise`, `samplewise`, `soft_mix`, and `finetune_unclamp`. The last mode uses
fully clamped backbone FAPE until the fine-tuning phase. Side-chain FAPE stays
clamped in every mode. `clamped_probability` must lie in `[0, 1]`; its default is
0.9. Batchwise sampling shares one decision across all microbatches and ranks of
an optimizer update. Evaluation uses the deterministic expected mixture.

`experiment.recycle.train_mode` is `sample_uniform` by default or `fixed`.
Training samples from 1 through `model.n_cycles` in the first mode and uses exactly
that many cycles in the second. Evaluation always uses `model.n_cycles`.
If supplied, `min_train_cycles` must be 1, `max_train_cycles` must equal
`model.n_cycles`, and `eval_mode` must be `fixed_max`. The adapter explicitly sets
the model's training/evaluation mode.

`experiment.loss_schedule.mode: initial_only` disables fine-tuning losses,
including at the final checkpoint. The default `finetune` linearly blends the
complete initial and fine-tuning objectives once across the configured ramp.
Learning-rate scheduling is independent: an initial-only recipe must explicitly
specify its intended `finetune_start_step` and `finetune_lr_scale`.

`data.msa_depth` caps the raw rows delivered by the loader. `model.msa_depth` and
`model.extra_msa_depth` cap centers and remaining extra rows after block deletion.
The reference raw/center caps are both 192, so it supplies no extra rows even
when the extra-row cap is 64. Research configurations requesting up to 192 centers
and 64 extras need a raw cap of at least 256; shallow alignments and training block
deletion can still yield fewer rows. The adapter reports `msa_cluster_rows`,
`msa_extra_rows` and `recycle_cycles` from the actual forward pass. Official track
caps remain enforced by the runner.

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

- max_steps: 30000
- effective_batch_size: 8
- sample_budget: 240000
- residue_budget: 61440000
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
