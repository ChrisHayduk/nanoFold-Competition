# NanoFold RFdiffusion-style reproductions

These are NanoFold sequence- and MSA-conditioned reconstructions inspired by
[RFdiffusion](https://github.com/RosettaCommons/RFdiffusion). The models start
from random initialization. Training requires no upstream RFdiffusion
checkout or weights. The upstream [BSD 3-Clause license](https://github.com/RosettaCommons/RFdiffusion/blob/main/LICENSE)
is listed for attribution.

| Profile | Parameters | Recorded training configuration | Training layout |
| --- | ---: | --- | --- |
| Tiny | 88,784 | [Exact JSON](small/architecture-scaling/rfdiffusion-style/configs/tiny-no-checkpoint.json) | Two GPU ranks, four accumulation microsteps |
| Medium | 3,096,860 | [Exact JSON](small/architecture-scaling/rfdiffusion-style/configs/medium-no-checkpoint.json) | Two GPU ranks, four accumulation microsteps |
| Large | 93,342,970 | [Exact JSON](large/architecture-reconstructions/rfdiffusion-style/configs/full.json) | Eight GPU ranks, one example per rank |

Each profile uses seed 0, 30,000 optimizer updates, an effective batch of eight,
and 240,000 sampled examples. The published JSON files are the original trainer
inputs. Their model, data, and training sections match the corresponding
terminal run configs. [Recorded configs](recorded-configs/) show the
checkpoint-bound runtime fields after removing absolute provenance paths.
Use the exact trainer inputs above for fresh launches.
[Provenance](PROVENANCE.json) records the source and config
hashes, including the distinct original distributed runtime for historical
Large. [SHA256SUMS](SHA256SUMS) covers the release files.

## Data and environment

Use Linux with CUDA-capable GPUs, a CUDA-enabled PyTorch build with NCCL, NumPy,
and Python 3.12. The recorded Tiny/Medium environment used PyTorch
2.8.0+cu128 and NumPy 2.1.2 on H100 GPUs. Historical Large used eight H100s.
Its archived evaluation runtime reports Python 3.12.3 and PyTorch 2.9.1+cu128.
See the package's [public benchmark quickstart](https://github.com/ChrisHayduk/nanoFold-Competition/blob/main/docs/QUICKSTART.md)
and [NanoFold Public data card](https://huggingface.co/datasets/ChrisHayduk/nanofold-public)
for setup and public data access.

Set `NANOFOLD_DATA` to a preprocessed tree with `manifests/train.txt`,
`manifests/val.txt`, `processed_features/`, and `processed_labels/`.
The recorded 10,000/1,000-chain experiment uses the exact tensor snapshot bound
by [data-lock.json](data-lock.json). A parquet export or a newer preprocessing
run can produce different NPZ bytes. Run the verifier before training to confirm
the input snapshot. The lock contains only the counts and digests consumed by
the original verifier, with the historical storage paths and sample metadata
removed.

```sh
python run_rf.py verify-data --data-root "$NANOFOLD_DATA"
```

## Train and evaluate

Run these commands from this directory. The launcher's `--dry-run` prints the
resolved `torch.distributed.run` command for review. Use a fresh output directory
for each profile.

```sh
python run_rf.py train tiny --data-root "$NANOFOLD_DATA" --run-dir runs/rf-tiny
python run_rf.py train medium --data-root "$NANOFOLD_DATA" --run-dir runs/rf-medium
python run_rf.py train large --data-root "$NANOFOLD_DATA" --run-dir runs/rf-large
```

Each trainer writes checkpoints and SHA-256 sidecars beneath its run directory.
Evaluation reads the public-validation split and saves five 50-step diffusion
samples per chain. Set `NANOFOLD_SCORER` to the root of the NanoFold checkout
that contains this bundle. The launcher checks that its two scorer files
match the recorded hashes. The scorer is also available at NanoFold commit
`7ce9638a4fd762f9ec438d11510a654cfecd2410`. Substitute the checkpoint path
emitted by the trainer:

```sh
python run_rf.py evaluate tiny --data-root "$NANOFOLD_DATA" \
  --checkpoint runs/rf-tiny/checkpoints/step-0030000.pt \
  --scorer-root "$NANOFOLD_SCORER" --output-dir evaluations/rf-tiny
```

For Medium or Large, change the profile, checkpoint and output directory.
The historical Tiny/Medium evaluation spread four logical shards across
separate workers. The launcher runs the same underlying evaluator on four
local ranks. Large uses its recorded eight-rank evaluator. The package records
the original Tiny/Medium sharding source digests for traceability. The
portable evaluator does not run the separately audited coordinate replay or
recreate the original distributed worker orchestration.

This release contains the scientific code, exact training config files, and
the input identity needed to check the experiments. The public data release
is linked above. Full checkpoint bodies and original coordinate outputs remain
in the experiment archive. The release checks cover configuration and source
hashes, launch argument construction, model parameter counts, and the
full recorded input tree. [Validation](VALIDATION.json) records their scope.
