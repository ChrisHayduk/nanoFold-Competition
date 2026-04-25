# Quickstart

This is the shortest path from a fresh clone to training a nanoFold model on the official public dataset.

The data setup step is the slow part: it downloads OpenFold assets for the public train and validation manifests, downloads the required mmCIF files, and preprocesses them into nanoFold NPZ files. After that, training is a normal `train.py` command.

## 1) Create The Environment

Prerequisites:
- Python 3.11 or newer
- `git`
- `aws` CLI for unsigned OpenFold S3 downloads
- `unzip`
- enough disk for raw OpenFold assets plus processed features and labels

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt awscli
git submodule update --init --recursive
```

## 2) Download And Preprocess Public Data

```bash
bash scripts/setup_official_data.sh \
  --data-root data/openproteinset \
  --processed-features-dir data/processed_features \
  --processed-labels-dir data/processed_labels \
  --mmcif-mode subset \
  --disable-templates
```

This writes:

- `data/processed_features/<encoded_chain_id>.npz`
- `data/processed_labels/<encoded_chain_id>.npz`

The command also verifies the committed public manifest hashes before downloading. Hidden validation data is never downloaded by this public setup path.

If preprocessing is interrupted, rerun the same command with:

```bash
--resume-preprocess
```

## 3) Start Training

Train the lightweight baseline:

```bash
python train.py \
  --config configs/official_baseline.yaml \
  --track limited_large \
  --official
```

Outputs land in:

```text
runs/official_limited_large_baseline/
```

To train the minAlphaFold2 reference submission instead:

```bash
python train.py \
  --config submissions/minalphafold2/config.yaml \
  --track limited_large \
  --official
```

## 4) Score Public Validation

After training has written a checkpoint, generate public validation predictions:

```bash
mkdir -p runs/official_limited_large_baseline/_forbid_labels

python predict.py \
  --config configs/official_baseline.yaml \
  --ckpt runs/official_limited_large_baseline/checkpoints/ckpt_last.pt \
  --split val \
  --track limited_large \
  --official \
  --forbid-labels-dir runs/official_limited_large_baseline/_forbid_labels \
  --pred-out-dir runs/official_limited_large_baseline/public_predictions \
  --save runs/official_limited_large_baseline/predict_val.json
```

Then score those predictions:

```bash
python score.py \
  --prediction-summary runs/official_limited_large_baseline/predict_val.json \
  --labels-dir data/processed_labels \
  --save runs/official_limited_large_baseline/eval_val.json \
  --per-chain-out runs/official_limited_large_baseline/per_chain_scores_val.jsonl
```

Public validation is for debugging and iteration. The leaderboard ranking uses the sealed hidden validation path.

## 5) Make Your Own Submission

Start from the template:

```bash
cp -R submissions/template submissions/your_name
```

Edit:

- `submissions/your_name/submission.py`
- `submissions/your_name/config.yaml`
- `submissions/your_name/notes.md`

Validate the submission contract:

```bash
python scripts/validate_submission.py \
  --submission submissions/your_name \
  --track limited_large
```

Your `run_batch` implementation must return:

```python
{"pred_atom14": pred_atom14}
```

where `pred_atom14` has shape `(B, L, 14, 3)`.

## 6) Submit To The Leaderboard

Hidden leaderboard scoring is maintainer-run so hidden features, labels, manifests, salts, and lock files never leave the sealed evaluation path.

Before opening a leaderboard PR, run:

```bash
python scripts/validate_submission.py \
  --submission submissions/your_name \
  --track limited_large \
  --strict
```

Commit only the submission files needed to reproduce your method, usually:

- `submissions/your_name/submission.py`
- `submissions/your_name/config.yaml`
- `submissions/your_name/notes.md`
- any small helper files imported by `submission.py`

Do not commit data, processed NPZs, hidden files, checkpoints, run directories, local logs, or modified official manifests.

Open a pull request with:

- title: `Submission: your_name`
- a short method summary
- the public validation score path or summary from step 4
- confirmation that `scripts/validate_submission.py --strict` passed

Maintainers will run the sealed official leaderboard command:

```bash
bash scripts/run_official_docker.sh \
  --submission submissions/your_name \
  --track limited_large \
  --update-leaderboard
```

That hidden run produces the official `foldscore_auc_hidden` rank score and updates the leaderboard artifacts if accepted.

## Where To Go Next

- [Competition rules](COMPETITION.md)
- [Submission API](API.md)
- [Data sources, splits, preprocessing, and tensor formats](DATA.md)
