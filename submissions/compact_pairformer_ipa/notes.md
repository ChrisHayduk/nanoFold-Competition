## Submission Summary

`compact_pairformer_ipa` is a train-from-scratch, atom14 protein-folding model for the official
`limited` track. It compresses the official MSA with a short Evoformer stack, discards the deep MSA
state after the query row, and continues with a compact pair/single trunk before the proven
minAlphaFold2 IPA structure module.

The model has:

- two MSA Evoformer blocks over the official MSA features
- eight Pairformer-style blocks with triangle geometry updates
- QK-normalized, pair-biased single attention
- ReLU-squared single transitions and zero-initialized residual outputs
- a four-layer IPA structure module with stereochemical atom14 construction
- Muon updates for hidden trunk matrices and Adam updates for embeddings, gates, heads, norms, and
  structure parameters
- BF16 trunk execution on CUDA with the AlphaFold loss evaluated from FP32 tensors

The Muon implementation follows the public Muon method by Keller Jordan and collaborators. The model
uses no pretrained parameters from Muon, OpenFold3, Protenix, AlphaFold, or any other external model.

## Method Rationale

The architecture keeps the official MSA signal but spends most model depth on pair and single
representations. This follows the data-efficient direction of AlphaFold3-style systems without
bringing their external distillation data, templates, pretrained weights, or full diffusion sampler
into the benchmark.

The hidden rank metric is area under the fixed-sample learning curve. The loss therefore blends from
the initial AlphaFold objective into the geometry-aware fine-tuning objective between steps 1,000 and
5,000. The learning rate drops by half when that ramp completes. This makes backbone, side-chain,
and stereochemical supervision active for the high-weight middle checkpoint, then uses lower-rate
updates to refine geometry for the remainder of the run.

## Competition Compliance Checklist

- [x] Used only the provided benchmark data; no external structures or labels are loaded.
- [x] Uses no external structures, pretrained weights, model embeddings, template lookup, or network
  access.
- [x] Kept dataset manifests fixed at `data/manifests/train.txt` and `data/manifests/val.txt`.
- [x] Ignores template tensors and does not run template search.
- [x] Model outputs atom14 coordinates as `pred_atom14` with shape `(B, L, 14, 3)` in Angstroms.
- [x] Inference does not read supervision labels.

## Required Run Metadata (`limited`)

- max_steps: 30000
- effective_batch_size: 8
- sample_budget: 240000
- residue_budget: 61440000
- crop_size: 256
- MSA depth: 192
- seed: 0
- trainable_parameters: 6024402
- hardware: NVIDIA A100-PCIE-40GB 40 GB (CUDA 12.8, PyTorch 2.7.1; full-shape smoke)
- wall_clock_time: 11 seconds for four full-shape smoke train steps; full 30000-step run pending
- commit: f7aab255f3af417e5498fb925588e869fd6b3628 (validated model/config)

## Validation Evidence

The committed source was exercised on an A100 with the official crop size and MSA depth
(`B=1, L=256, N=192`). A BF16 forward, backward, gradient clip, and Muon/Adam optimizer step used
2.60 GiB peak allocated CUDA memory and 2.96 GiB peak reserved memory. All gradient tensors were
finite, and the step completed in 1.82 seconds after model construction.

The official effective batch is implemented as `batch_size=4` with two gradient-accumulation
microbatches. A full-shape `B=4` microbatch completed in 4.62 seconds at 0.865 samples/second and
used 11.02 GiB peak allocated / 12.42 GiB peak reserved CUDA memory. `B=8` was only 3.7% faster in
throughput while reserving 24.74 GiB, so the submitted configuration keeps the lower-memory option.

An official-mode four-step synthetic smoke run wrote checkpoints at steps 0, 2, and 4. Resuming
from step 2 reproduced the step-3 and step-4 losses exactly. Label-free multi-checkpoint prediction
and the separate FoldScore process both completed. The synthetic score is intentionally not reported
as a benchmark result; the full public-data learning curve and sealed hidden score remain pending.

The training loop keeps the expensive FoldScore component evaluation disabled. It records standard
validation loss, lDDT, and RMSD diagnostics every 5,000 steps while retaining checkpoints every
1,000 steps for the required learning-curve points. The separate sealed prediction/scoring stages
compute the complete ranking metric without duplicating that work during optimization.

## Throughput profile

The training loader can use `data.bucket_by_length: true` with `data.length_bucket_size: 16`. It
keeps the fixed-seed shuffle but groups nearby chain lengths before forming each batch, reducing
padding in the pairwise trunk. On the Kolmogorov RTX 4090 pilot with the official B=4, L<=256,
MSA=192 shape and deterministic CUDA settings, eight one-microbatch updates took 11.79 seconds
with ordinary shuffled batches and 7.92 seconds with length buckets (33% faster). The model and
official data contract were unchanged. Checkpoint RNG-state preservation is disabled explicitly for
this zero-dropout configuration; the generic model default remains enabled.

A deterministic optimizer pilot used 128 official training chains and 32 official validation chains
with the same effective batch size, crop size, MSA depth, model, and geometry-loss ramp shape as the
submitted run. The table reports the complete FoldScore computed in a separate label-aware scoring
process; it is a small public-data pilot, not a hidden-set or leaderboard result.

| Optimizer | FoldScore at step 20 | FoldScore at step 50 |
| --- | ---: | ---: |
| Muon, learning rate 0.02 | 0.208203 | 0.225949 |
| Muon, learning rate 0.01 | **0.211507** | **0.266542** |
| Adam, learning rate 0.001 | 0.195205 | 0.222400 |

Muon at 0.01 was selected because its step-50 improvement over Muon at 0.02 was supported by CADaa
(0.2623 versus 0.1319), SphereGrinder (0.1524 versus 0.0412), and DipDiff (0.4498 versus 0.3555),
while its clash and backbone-geometry components did not regress. Adam produced a higher step-50
lDDT but a lower complete FoldScore and substantially larger loss and gradient-norm excursions.

A second deterministic pilot used 1,024 training chains and 128 validation chains. Its schedule was
scaled by training-set epochs: pilot steps 100 and 500 correspond to official steps 1,000 and 5,000.
Two same-host runs were tensor-identical through step 500 (877 model tensors, maximum absolute
difference 0). The common step-500 checkpoint scored 0.323871. Over the next 100 updates, keeping
Muon at 0.01 reached 0.338234, while dropping it to 0.005 reached **0.343251**. The cooldown improved
GDT_HA, atom14 lDDT, CADaa, backbone geometry, and DipDiff; only SphereGrinder and clash preservation
declined slightly. A paired chain-level bootstrap placed the aggregate FoldScore difference at 0.0050
with a 95% confidence interval of -0.0012 to 0.0115, so this is optimizer-selection evidence rather
than a guarantee of hidden-set improvement. DipDiff and backbone geometry had positive confidence
intervals. The submitted schedule therefore applies the measured 0.5 decay at step 5,000.

## How to run

```bash
python scripts/validate_submission.py \
  --submission submissions/compact_pairformer_ipa \
  --track limited \
  --strict

python train.py \
  --config submissions/compact_pairformer_ipa/config.yaml \
  --track limited \
  --official
```
