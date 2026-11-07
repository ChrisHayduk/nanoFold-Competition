# NanoFold NeurIPS 2026 training recipes

These research runs train AF3-style and RFdiffusion-style models from scratch on
NanoFold. The family directories contain the model and training code, the six
recorded training configurations, and instructions for training and scoring on
public validation.

| Family | Tiny parameters | Medium parameters | Large parameters | Training seed | Validation draws per chain |
|---|---:|---:|---:|---:|---:|
| [AF3-style Protenix reconstruction](af3/README.md) | 98,651 | 2,773,243 | 368,088,191 | 20260725 | 1 |
| [RFdiffusion-style reconstruction](rfdiffusion/README.md) | 88,784 | 3,096,860 | 93,342,970 | 0 | 5 |

Each run uses 30,000 optimizer updates and 240,000 sampled training examples
with an effective batch size of eight. The families use different objectives
and inference budgets.

Prepare the 10,000 public training chains and 1,000 public validation chains
using the [NanoFold public data setup](../../docs/QUICKSTART.md#2-download-and-preprocess-public-data).
The [dataset card](https://huggingface.co/datasets/ChrisHayduk/nanofold-public)
describes the public split and tensor schema. Use each family guide to verify
the inputs, set up the software environment and choose a training profile.

For the runs reported in the paper, the processed input tree digest is
`98c141bafcb51c2a8532ec3eba115ff170e9a9da9c54e4f18e5e7c83b4ef1e00`.
It covers both public manifests, 11,000 feature NPZs and 11,000 label NPZs.
The recorded tensors match the feature and label digests in the
[public dataset lock](../../leaderboard/official_dataset_fingerprint.json).
Follow the public preprocessing instructions and run the family verifier
before training. The Hugging Face export has a separate fingerprint because
it applies MSA row filtering.

The model-specific launchers run the paper experiments on public data. The
[competition guide](../../docs/COMPETITION.md) describes the separate submission
API and leaderboard workflow.
