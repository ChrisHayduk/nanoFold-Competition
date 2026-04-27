## Submission Summary

Template submission scaffold for participants.

Suggested edits:
- briefly describe the architecture and training method
- list any added files under your submission directory
- state any key hyperparameter choices

## Method Rationale

Summarize the expected gains and intuition behind your method.

## Competition compliance checklist

- [x] Used only the provided benchmark data (no external data, weights, or template/MSA searches).
- [x] Kept dataset manifests fixed (`data/manifests/train.txt` and `data/manifests/val.txt`).
- [x] Model outputs atom14 coordinates per residue (`(L, 14, 3)` in Angstrom).
- [x] `run_batch(..., training=False)` does not depend on supervision labels (`ca_coords`, `ca_mask`).

## Required run metadata (`limited`)

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
python train.py --config submissions/<your_name>/config.yaml --track limited --official
python predict.py \
  --config submissions/<your_name>/config.yaml \
  --ckpt runs/<run_name>/checkpoints/ckpt_last.pt \
  --split val \
  --track limited \
  --official \
  --forbid-labels-dir runs/<run_name>/_forbid_labels \
  --pred-out-dir runs/<run_name>/public_predictions \
  --save runs/<run_name>/predict_val.json
python score.py \
  --prediction-summary runs/<run_name>/predict_val.json \
  --labels-dir data/processed_labels \
  --save runs/<run_name>/eval_val.json \
  --per-chain-out runs/<run_name>/per_chain_scores_val.jsonl
```
