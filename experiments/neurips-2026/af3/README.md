# AF3-style NanoFold training and evaluation

The three profiles train randomly initialized Protenix-based reconstructions on NanoFold's public train/validation split. The terminal comparison uses one training seed, eight ranks, 30,000 optimizer updates and 240,000 sampled exposures per profile.

| Profile | Parameters | Recorded hardware | Exact resolved training configuration |
| --- | ---: | --- | --- |
| Tiny | 98,651 | 8 × A40 | [tiny-train.resolved.json](configs/tiny-train.resolved.json) |
| Medium | 2,773,243 | 8 × H100 | [medium-train.resolved.json](configs/medium-train.resolved.json) |
| Large | 368,088,191 | 8 × H100 | [large-train.resolved.json](configs/large-train.resolved.json) |

The executed training code is under `runtime/{tiny,medium,large}`. It includes the trainer, dataset loader and verifier, adapter, DDP and RNG handling, checkpoint manager, configuration builder, and evaluator. The builders resolve the pinned Protenix defaults and apply the runtime's exact overrides. The resolved JSONs are full configuration records with canonical SHA256 values that match the terminal run provenance. The runtime's descriptive `config.yaml` is also a provenance input. [Original source and configuration bindings](ORIGINAL-SOURCE-BINDINGS.json).

The original training code hashes several ancillary scripts when it creates provenance. The runtime directories therefore retain the byte-exact `launch_8xh100.sh`, `launch_overfit_8xh100.sh`, `pilot_8xh100.sh`, and prediction and CCD exporters. Use the separate portable launcher below for the terminal experiment. The original resolved JSONs retain the path-valued settings included in their recorded hashes. On a different filesystem, set the paths in the portable launcher and use the same chemistry root for training and evaluation of the resulting checkpoint.

## Runtime setup

Use Linux x86-64, Python 3.12, an NVIDIA driver compatible with CUDA 12.6, CUDA build tools, a C++ compiler, and eight visible GPUs. Install the pinned PyTorch wheel first and then the other runtime packages. The code bundle supplies the NanoFold-specific executable modules. The Protenix and NanoFold Git checkouts, Python dependencies, CCD files, and public tensors are external inputs.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install pip==25.2 setuptools==80.9.0 wheel==0.45.1
python -m pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.1
python -m pip install -r environment/af3-minimal-requirements.lock.txt
python -m pip check

git clone https://github.com/bytedance/Protenix.git protenix
git -C protenix checkout --detach c3bfc365b3e1341a11935eddfe7bfdc308092147
git clone https://github.com/ChrisHayduk/nanoFold-Competition.git nanofold-scorer
git -C nanofold-scorer checkout --detach 7ce9638a4fd762f9ec438d11510a654cfecd2410

export PROTENIX_ROOT="$PWD/protenix"
export PROTENIX_ROOT_DIR="$PWD/chemistry"
export NANOFOLD_SCORER_ROOT="$PWD/nanofold-scorer"
export PYTHONPATH="$PROTENIX_ROOT"
python environment/download_af3_ccd.py --protenix-root "$PROTENIX_ROOT" \
  --output-root "$PROTENIX_ROOT_DIR" \
  --exporter runtime/large/protenix-af3-style/export_ccd_reference.py \
  --reference-output "$PWD/ccd-reference.json"
```

The [Protenix source pin](https://github.com/bytedance/Protenix/tree/c3bfc365b3e1341a11935eddfe7bfdc308092147) uses the [Apache 2.0 license](licenses/Protenix-LICENSE.txt). The CCD helper verifies the downloaded archive, selected members, and derived reference digest. The evaluator verifies the [NanoFold scorer checkout](https://github.com/ChrisHayduk/nanoFold-Competition/tree/7ce9638a4fd762f9ec438d11510a654cfecd2410) and its two metric source hashes. [Scorer copies](scorer/nanofold/) record the exact executed bytes for inspection. Use the pinned Git checkout for evaluation.

## Public input data

The loader expects `manifests/train.txt`, `manifests/val.txt`, `processed_features/`, and `processed_labels/` under one data root. The [NanoFold public data setup](https://github.com/ChrisHayduk/nanoFold-Competition/blob/main/docs/DATA.md) describes downloading the public OpenProteinSet and RCSB inputs and preprocessing the public manifests. Run its `scripts/setup_official_data.sh` command with `--mmcif-mode subset --disable-templates` in the NanoFold checkout, then verify the resulting tensor tree. The published [Hugging Face release](https://huggingface.co/datasets/ChrisHayduk/nanofold-public) offers the same public splits in Parquet and documents its separately sanitized feature fingerprint.

```bash
export NANOFOLD_DATA_ROOT="$NANOFOLD_SCORER_ROOT/data"
export NANOFOLD_VERIFY_REPORT="$PWD/public-dataset-verification.json"
python runtime/tiny/common/verify_dataset.py --data-root "$NANOFOLD_DATA_ROOT" \
  --output "$NANOFOLD_VERIFY_REPORT"
```

An exact input match has 10,000 training chains and 1,000 validation chains. Its 22,002-file inventory totals 1,128,037,744 bytes and has tree SHA256 `98c141bafcb51c2a8532ec3eba115ff170e9a9da9c54e4f18e5e7c83b4ef1e00`. The local recovery of the original public snapshot matched this tree. A fresh download and preprocessing run may produce different tensor bytes. The portable launcher checks the report against the recorded tree, and the native training and evaluation verifiers rehash the selected files. Keep the same generated report for training and evaluation. See the public data and manifest bindings in the source receipt.

## Train and evaluate

Set a fresh run directory for each profile. The launcher takes the active Python interpreter from `AF3_PYTHON` or the environment. It calls the original train and evaluation entry points with eight ranks, crop 256, MSA depth 192, and seed 20260725. Set `AF3_PRINT_COMMAND=1` to inspect its command without launching.

```bash
export NANOFOLD_RUN_DIR="$PWD/runs/tiny"
bash run_af3.sh tiny train
bash run_af3.sh tiny evaluate

export NANOFOLD_RUN_DIR="$PWD/runs/medium"
bash run_af3.sh medium train
bash run_af3.sh medium evaluate

export NANOFOLD_RUN_DIR="$PWD/runs/large"
bash run_af3.sh large prefix250
bash run_af3.sh large continue
bash run_af3.sh large evaluate
```

The large run started at random initialization, committed its initial 250-update prefix, then resumed its model, optimizer, scheduler and rank RNG state for the remaining updates. The original publisher and machine-specific admission controller are outside the portable path. Terminal evaluation uses all 1,000 public validation chains, four recycles, one model seed and diffusion draw, and 200 diffusion steps. The independent replay scripts are in `runtime/small` and `runtime/large`; use their `--help` with the pinned scorer checkout.

The release preflight checks source and configuration hashes, the original local public tensor inventory, and local command/import wiring. A fresh GPU training and evaluation run is still needed to exercise the portable setup end to end.
