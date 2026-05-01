# Quickstart

This is the shortest path from a fresh clone to training a nanoFold model on the official public dataset.

The data setup step is the slow part: it downloads OpenFold assets for the public train and validation manifests, downloads the required mmCIF files, and preprocesses them into nanoFold NPZ files. After that, training is a normal `train.py` command.

## Choose A Track

Use `limited` for your first run. The other tracks use the same public data setup and the same hidden evaluation path, but answer different research questions.

| Track | Purpose | Budget | Ranking | Local command pattern | PR label/description |
|---|---|---:|---|---|---|
| `limited` | main accessible slowrun leaderboard | `20,000` samples | hidden CASP15-inspired FoldScore AUC | `--track limited --official` | `Track: limited` |
| `research_large` | larger fixed-data slowrun for deeper optimization studies | `100,000` samples | hidden CASP15-inspired FoldScore AUC | `--track research_large --official` | `Track: research_large` |
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

If you prefer conda, keep the environment local to the repo:

```bash
conda create -p .conda/nanofold python=3.12 pip -y
conda activate ./.conda/nanofold
python -m pip install -r requirements.txt awscli
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

Official training verifies the public dataset fingerprint before the progress bar starts. On a CPU-only laptop, the full reference run can take a few hours; by default it writes checkpoints and runs public validation every 1,000 steps.

Training also prints compact status lines during the run with training loss, learning rate, gradient norm, sample-budget progress, residue-budget progress, throughput, elapsed time, and ETA. The same per-step records are written to `runs/minalphafold2_reference/train_metrics.jsonl`.

To restart a run from step 0 and clear stale metrics/checkpoints for that run name, add:

```bash
--reset-run
```

On macOS with Python 3.13 or newer, nanoFold automatically uses `data.num_workers=0` for DataLoader stability. That message is informational.

Validation prints `val_lddt_ca`, `val_loss`, `val_rmsd_ca`, and `val_rmsd_atom14` directly in the terminal so you can see both competition-aligned quality and coordinate error during the run.

The reference submission loads its model architecture directly from:

```text
third_party/minAlphaFold2/configs/tiny.toml
```

### Optional: Run On A Modal GPU

If local training is too slow, run the same official training command on a Modal GPU. This uploads the public processed data to read-only Modal Volumes, stages it onto the remote container's local disk before fingerprint verification and training, and writes checkpoints and metrics to a separate `nanofold-runs` volume.

Install and authenticate Modal locally:

```bash
pip install modal
modal setup
```

Upload the public processed data after step 2 has completed. The default upload format is a pair of tar archives so Modal startup and training avoid thousands of small network-volume reads.

```bash
modal run scripts/modal_train.py --upload-data --skip-train
```

Launch the limited-track minAlphaFold2 reference run:

```bash
NANOFOLD_MODAL_GPU=A10G modal run scripts/modal_train.py \
  --config submissions/minalphafold2/config.yaml \
  --track limited \
  --reset-run
```

The default GPU is `A10G`. Set `NANOFOLD_MODAL_GPU=L40S` or another Modal-supported GPU type when you want a larger or faster worker.

Remote runs auto-resume from `runs/<run_name>/checkpoints/ckpt_last.pt` in the `nanofold-runs` volume. To continue a run, repeat the same command without `--reset-run`:

```bash
NANOFOLD_MODAL_GPU=A10G modal run scripts/modal_train.py \
  --config submissions/minalphafold2/config.yaml \
  --track limited
```

Fetch a checkpoint back from Modal when you want to run local public prediction/scoring:

```bash
mkdir -p runs/minalphafold2_reference/checkpoints
modal volume get nanofold-runs \
  minalphafold2_reference/checkpoints/ckpt_last.pt \
  runs/minalphafold2_reference/checkpoints/ckpt_last.pt
```

Use separate run names or separate Modal run volumes for concurrent experiments. Do not upload hidden validation assets in participant workflows; this path only uses public train and public validation data.

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
  --features-dir data/processed_features \
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

Hidden leaderboard scoring and leaderboard updates are maintainer-run so hidden features, labels, manifests, salts, lock files, and official result artifacts never leave the sealed evaluation path. Participants submit a validated, reproducible PR; maintainers create the accepted leaderboard entry after the sealed run.

Submission mechanics are the same for all tracks:

| Track | Validate before PR | Maintainer hidden command |
|---|---|---|
| `limited` | `python scripts/validate_submission.py --submission submissions/your_name --track limited --strict` | `bash scripts/run_official_docker.sh --submission submissions/your_name --track limited --update-leaderboard` |
| `research_large` | `python scripts/validate_submission.py --submission submissions/your_name --track research_large --strict` | `bash scripts/run_official_docker.sh --submission submissions/your_name --track research_large --update-leaderboard` |
| `unlimited` | `python scripts/validate_submission.py --submission submissions/your_name --track unlimited --strict` | `bash scripts/run_official_docker.sh --submission submissions/your_name --track unlimited --update-leaderboard` |

Maintainers can add `--team "<team or individual name>"` to the hidden command
so the accepted entry renders under the right leaderboard identity. The rendered
leaderboard also links the entry name to the accepted `submissions/<name>`
directory. If omitted during a PR-triggered GitHub Actions run, nanoFold falls
back to the PR author's GitHub username. Manual maintainer automation can set
`NANOFOLD_PR_AUTHOR=<github-username>` for the same fallback.

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

Do not edit `leaderboard/leaderboard.json` or the rendered README leaderboard table in a submission PR. Those files are updated only by the maintainer-run hidden evaluation.

```bash
git add submissions/your_name
git commit -m "Add your_name nanoFold submission"
git push -u origin submission/your_name
```

Open a pull request with:

- title: `Submission: your_name`
- team or individual researcher name for leaderboard display
- target track, for example `Track: limited`
- a short method summary
- the public validation score path or summary from step 4
- confirmation that `scripts/validate_submission.py --strict` passed

Maintainers will run the sealed official leaderboard command:

```bash
bash scripts/run_official_docker.sh \
  --submission submissions/your_name \
  --track <track_id> \
  --team "<team or individual name>" \
  --update-leaderboard
```

When the checkpoint is already in Modal's `nanofold-runs` volume, maintainers can run the sealed hidden evaluation there instead:

```bash
modal run scripts/modal_official.py \
  --upload-public-data \
  --upload-hidden-assets \
  --upload-only
```

```bash
modal run scripts/modal_official.py \
  --submission submissions/your_name \
  --config submissions/your_name/config.yaml \
  --track <track_id> \
  --team "<team or individual name>" \
  --update-leaderboard
```

For long Modal evaluations, maintainers can detach the two stages and update the leaderboard from the result artifact written to the `nanofold-runs` volume:

```bash
modal run --detach scripts/modal_official.py \
  --submission submissions/your_name \
  --config submissions/your_name/config.yaml \
  --track <track_id> \
  --team "<team or individual name>" \
  --skip-score \
  --background-predict
```

```bash
modal run --detach scripts/modal_official.py \
  --submission submissions/your_name \
  --config submissions/your_name/config.yaml \
  --track <track_id> \
  --team "<team or individual name>" \
  --skip-predict \
  --background-score
```

```bash
modal volume get nanofold-runs <run_name>/modal_official_result.json runs/<run_name>/modal_official_result.json
python scripts/add_leaderboard_entry.py \
  --result runs/<run_name>/modal_official_result.json \
  --leaderboard leaderboard/leaderboard.json \
  --readme README.md \
  --description "<leaderboard description>" \
  --team "<team or individual name>"
```

That hidden run produces the official rank score for the selected track and updates the leaderboard artifacts if accepted.

## Where To Go Next

- [Competition rules](COMPETITION.md)
- [Submission API](API.md)
- [Data sources, splits, preprocessing, and tensor formats](DATA.md)
