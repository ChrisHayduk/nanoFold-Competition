# Quickstart

This is the shortest path from a fresh clone to training a nanoFold model on the official public dataset.

The data setup step is the slow part: it downloads OpenFold assets for the public train and validation manifests, downloads the required mmCIF files, and preprocesses them into nanoFold NPZ files. After that, training is a normal `train.py` command.

## Choose A Track

Use `limited` for your first run. The other tracks use the same public data setup and the same hidden evaluation path, but answer different research questions.

| Track | Purpose | Budget | Ranking | Local command pattern | PR label/description |
|---|---|---:|---|---|---|
| `limited` | main accessible slowrun leaderboard | `20,000` samples | hidden FoldScore AUC | `--track limited --official` | `Track: limited` |
| `research_large` | larger fixed-data slowrun for deeper optimization studies | `100,000` samples | hidden FoldScore AUC | `--track research_large --official` | `Track: research_large` |
| `unlimited` | open-ended fixed-data run for best final hidden quality | unrestricted | final hidden FoldScore | `--track unlimited --official` | `Track: unlimited` |

Set the same track in `submissions/<name>/config.yaml`, every `--track` command, and the pull request description. If you want to enter multiple tracks, keep separate configs or separate submission directories so each run is reproducible.

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

Train the minAlphaFold2 tiny reference submission:

```bash
python train.py \
  --config submissions/minalphafold2/config.yaml \
  --track limited \
  --official
```

Outputs land in:

```text
runs/minalphafold2_reference/
```

Official training verifies the public dataset fingerprint before the progress bar starts. On a CPU-only laptop, the full reference run can take a few hours; by default it writes checkpoints and runs public validation every 500 steps.

On macOS with Python 3.13 or newer, nanoFold automatically uses `data.num_workers=0` for DataLoader stability. That message is informational.

Validation prints `val_lddt_ca`, `val_loss`, `val_rmsd_ca`, and `val_rmsd_atom14` directly in the terminal so you can see both competition-aligned quality and coordinate error during the run.

The reference submission loads its model architecture directly from:

```text
third_party/minAlphaFold2/configs/tiny.toml
```

## 4) Score Public Validation

After training has written a checkpoint, generate public validation predictions:

```bash
mkdir -p runs/minalphafold2_reference/_forbid_labels

python predict.py \
  --config submissions/minalphafold2/config.yaml \
  --ckpt runs/minalphafold2_reference/checkpoints/ckpt_last.pt \
  --split val \
  --track limited \
  --official \
  --forbid-labels-dir runs/minalphafold2_reference/_forbid_labels \
  --pred-out-dir runs/minalphafold2_reference/public_predictions \
  --save runs/minalphafold2_reference/predict_val.json
```

Then score those predictions:

```bash
python score.py \
  --prediction-summary runs/minalphafold2_reference/predict_val.json \
  --labels-dir data/processed_labels \
  --save runs/minalphafold2_reference/eval_val.json \
  --per-chain-out runs/minalphafold2_reference/per_chain_scores_val.jsonl
```

Public validation is for debugging and iteration. The leaderboard ranking uses the sealed hidden validation path.

## 5) Make Your Own Submission

Start from the template:

```bash
git switch -c submission/your_name
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
  --track limited
```

Your `run_batch` implementation must return:

```python
{"pred_atom14": pred_atom14}
```

where `pred_atom14` has shape `(B, L, 14, 3)`.

Train your submission with:

```bash
python train.py \
  --config submissions/your_name/config.yaml \
  --track <track_id> \
  --official
```

Then score it with the same public validation commands from step 4, replacing the config path, track, and run directory with your submission's `run_name`.

## 6) Submit To The Leaderboard

Hidden leaderboard scoring is maintainer-run so hidden features, labels, manifests, salts, and lock files never leave the sealed evaluation path.

Submission mechanics are the same for all tracks:

| Track | Validate before PR | Maintainer hidden command |
|---|---|---|
| `limited` | `python scripts/validate_submission.py --submission submissions/your_name --track limited --strict` | `bash scripts/run_official_docker.sh --submission submissions/your_name --track limited --update-leaderboard` |
| `research_large` | `python scripts/validate_submission.py --submission submissions/your_name --track research_large --strict` | `bash scripts/run_official_docker.sh --submission submissions/your_name --track research_large --update-leaderboard` |
| `unlimited` | `python scripts/validate_submission.py --submission submissions/your_name --track unlimited --strict` | `bash scripts/run_official_docker.sh --submission submissions/your_name --track unlimited --update-leaderboard` |

Before opening a leaderboard PR, run:

```bash
python scripts/validate_submission.py \
  --submission submissions/your_name \
  --track <track_id> \
  --strict
```

Commit only the submission files needed to reproduce your method, usually:

- `submissions/your_name/submission.py`
- `submissions/your_name/config.yaml`
- `submissions/your_name/notes.md`
- any small helper files imported by `submission.py`

Do not commit data, processed NPZs, hidden files, checkpoints, run directories, local logs, or modified official manifests.

```bash
git add submissions/your_name
git commit -m "Add your_name nanoFold submission"
git push -u origin submission/your_name
```

Open a pull request with:

- title: `Submission: your_name`
- target track, for example `Track: limited`
- a short method summary
- the public validation score path or summary from step 4
- confirmation that `scripts/validate_submission.py --strict` passed

Maintainers will run the sealed official leaderboard command:

```bash
bash scripts/run_official_docker.sh \
  --submission submissions/your_name \
  --track <track_id> \
  --update-leaderboard
```

That hidden run produces the official rank score for the selected track and updates the leaderboard artifacts if accepted.

## Where To Go Next

- [Competition rules](COMPETITION.md)
- [Submission API](API.md)
- [Data sources, splits, preprocessing, and tensor formats](DATA.md)
