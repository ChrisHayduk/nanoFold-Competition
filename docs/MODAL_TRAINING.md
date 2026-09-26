# Modal training and resumable experiments

`scripts/modal_train.py` preserves the official track defaults when no experiment bundle is selected. Training mounts public feature and label volumes read-only, blocks worker network access, verifies the dataset fingerprint, and writes outputs to a separate run volume. Image packaging includes repository code, model profiles, and the vendored stereochemical table; local checkpoints and arbitrary submission-directory files are excluded.

## Prepare a candidate experiment locally

A candidate experiment can use public train/validation assets before an official dataset release. Its config must use a repository module such as `submission.module: submissions.minalphafold2.submission`, a unique safe `run_name`, a nonnegative integer seed, and repository-relative manifest and model-profile paths. Explicit config inputs may live outside the checkout. Model code and referenced data metadata must live inside it.

Prepare one bundle per track and fingerprint after the experiment code and profiles are frozen:

```bash
python scripts/prepare_modal_training_bundle.py \
  --config CONFIG.yaml \
  --fingerprint PUBLIC_FINGERPRINT.json \
  --output artifacts/modal-training-bundle
```

Repeat `--config` to include multiple seeds or variants. The command performs local validation only. It copies normalized configs, public manifests, the public fingerprint, and the exact public source-lock bytes at their original repository-relative paths. It records hashes of executable sources and profiles and checks complete, disjoint public split membership. It copies no processed tensors or hidden assets. Subsequent source changes invalidate the bundle and require preparing a fresh one.

Candidate mode requires three distinct experiment volume names, separate from the default official volumes. Install and authenticate the Modal client before an explicitly authorized upload or launch:

```bash
export NANOFOLD_MODAL_TRAINING_BUNDLE=artifacts/modal-training-bundle
export NANOFOLD_MODAL_FEATURES_VOLUME=experiment-public-features
export NANOFOLD_MODAL_LABELS_VOLUME=experiment-public-labels
export NANOFOLD_MODAL_RUNS_VOLUME=experiment-runs
export NANOFOLD_MODAL_GPU=A10G

modal run scripts/modal_train.py \
  --run-name RUN_NAME --track limited \
  --features-dir PUBLIC_FEATURES --labels-dir PUBLIC_LABELS \
  --upload-data --skip-train

modal run scripts/modal_train.py \
  --run-name RUN_NAME --track limited --background
```

An upload checks the full public fingerprint before any files are sent. It selects only the declared public NPZ files and feature preprocessing metadata; unrelated audit files are not uploaded, and unexpected NPZ files or links cause failure. Use fresh data volumes for each distinct dataset and keep the upload format consistent. Every worker checks the entire mounted volume before staging: archive mode permits only its declared archive, and file mode permits only the selected public files. Extracted contents receive the same check. The worker uses a fresh local staging directory and verifies the fingerprint again before training.

Candidate mode preserves the config seed and uses `--verify-fingerprint --deterministic` without applying official policy overrides. It is a private experiment, not an official leaderboard run. Official runs continue to use `--config`, `--track`, and `--official` internally.

## Pause and resume

The worker requests a graceful optimizer-boundary pause after at most 22 hours by default, with additional allowance for scheduled validation and checkpoint writing below the 24-hour platform limit. `--max-runtime-seconds` can shorten a segment without changing the config, total optimizer-step budget, or learning-rate schedule. A paused trainer returns status 75; the launcher records `paused` and does not launch another paid invocation automatically. Repeat the same launch command to continue.

The supervisor commits the run volume whenever a closed, atomically replaced checkpoint appears, checks every 15 seconds, and commits again on exit. A storage-commit failure stops training. The final per-segment result is written to `runs/.modal_execution/<run_name>.result.json`; `completed` means the trainer reached its configured final step. A platform hard kill can still lose work since the last successful commit. A segment that is too short to finish an optimizer step, scheduled validation, and checkpoint write must be enlarged.

Before the first training segment, the launcher records `runs/.modal_execution/<run_name>.json`. Resume requires matching config and fingerprint bytes, bundle identity, all copied source/profile/resource hashes, Python and installed package versions, CUDA/cuDNN and GPU identity, and volume names. It also uses the trainer's checkpoint sampling/RNG checks. Changed code, runtime, or data requires a new run name. Existing runs without this execution record cannot be adopted silently. `--reset-run` resets only a run with the same execution identity.

A per-run Modal Dict claim rejects concurrent writers, and a committed active marker in the run volume prevents an interrupted run from reopening when the Dict entry expires. The launcher releases both only after a clean pause or completion and successful final commits. This uses Modal's [conditional Dict insertion](https://modal.com/docs/sdk/py/latest/Dict#put); it does not rely on volume writes acting as distributed locks.

After any interrupted or failed call, an operator must verify that the recorded Modal call and all child processes have ended, preserve its diagnostics and last checkpoint, and remove both the run's `.active.json` marker and its `<runs-volume>/<run_name>` entry from the `nanofold-training-writer-claims-v1` Dict before retrying. There is no automatic timeout-based takeover or force-unlock launch flag. Use separate run names for concurrent seeds and variants. Keep the final config budget fixed across segments and do not repurpose a data volume while a run is active.

The image pins its scientific package versions and sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before PyTorch initialization. TF32 is explicitly disabled through `NVIDIA_TF32_OVERRIDE=0`; the image and subprocess environment use the same pinned precision policy, which is recorded in the execution identity. These settings follow the [PyTorch deterministic-operation requirements](https://docs.pytorch.org/docs/2.14/generated/torch.use_deterministic_algorithms.html) and [CUDA environment controls](https://docs.pytorch.org/docs/2.14/cuda_environment_variables.html). Local unit and process tests cannot establish GPU capacity, successful image construction, deterministic CUDA operator support, or checkpoint visibility after a real Modal worker restart. Validate those with an explicitly authorized small GPU smoke run before a full campaign.
