"""Publish the public nanoFold train/validation data as a Hugging Face dataset.

The uploaded dataset is HF-native: each processed NPZ field is expanded into a
typed dataset column rather than stored as an opaque binary blob. Only public
train/validation manifests are read.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

# Allow running as `python scripts/upload_hf_public_dataset.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofold.chain_paths import chain_npz_path
from nanofold.data import read_manifest
from nanofold.utils import sha256_file

DEFAULT_TRAIN_MANIFEST = Path("data/manifests/train.txt")
DEFAULT_VAL_MANIFEST = Path("data/manifests/val.txt")
DEFAULT_ALL_MANIFEST = Path("data/manifests/all.txt")
DEFAULT_FEATURES_DIR = Path("data/processed_features")
DEFAULT_LABELS_DIR = Path("data/processed_labels")
DEFAULT_FINGERPRINT = Path("leaderboard/official_dataset_fingerprint.json")
DEFAULT_MANIFEST_LOCK = Path("leaderboard/official_manifest_source.lock.json")
DEFAULT_EVAL_YAML = Path("eval.yaml")

OPTIONAL_DEPENDENCY_MESSAGE = (
    "Install the optional Hugging Face upload dependencies first:\n"
    "  python -m pip install -U datasets huggingface_hub hf_xet pyarrow"
)

FEATURE_DESCRIPTIONS: tuple[tuple[str, str], ...] = (
    ("chain_id", "NanoFold chain identifier in <pdb_id>_<chain_id> form."),
    ("pdb_id", "Lowercase four-character PDB entry ID parsed from chain_id."),
    ("pdb_chain_id", "Author/asym chain suffix parsed from chain_id."),
    ("split", "Dataset split: train or validation."),
    ("length", "Number of residues after official preprocessing and projection."),
    ("msa_depth", "Number of MSA rows retained in the processed feature file."),
    ("template_count", "Number of template hits encoded in the feature tensors; official public data uses T=0."),
    ("aatype", "Target amino-acid IDs with AF2 ordering ARNDCQEGHILKMFPSTWYV plus unknown=20."),
    ("msa", "Tokenized A3M MSA with shape (N, L); 0-19 are residues, 20 unknown, 21 gap, 22 mask."),
    ("deletions", "A3M insertion/deletion counts aligned to msa with shape (N, L)."),
    ("residue_index", "Contiguous residue indices available at inference time."),
    ("between_segment_residues", "Segment-boundary flags; zero for the official single-chain data."),
    ("projection_seq_identity", "Sequence identity between feature query and projected coordinate sequence."),
    ("projection_alignment_coverage", "Coordinate projection coverage after sequence alignment."),
    ("projection_aligned_fraction", "Fraction of residues aligned during coordinate projection."),
    ("projection_valid_ca_count", "Number of residues with valid projected C-alpha coordinates."),
    ("template_aatype", "Template amino-acid IDs with shape (T, L); empty in the official public release."),
    ("template_ca_coords", "Template C-alpha coordinates with shape (T, L, 3); empty in the official public release."),
    ("template_ca_mask", "Template C-alpha validity mask with shape (T, L); empty in the official public release."),
    ("ca_coords", "Projected C-alpha label coordinates in Angstroms with shape (L, 3)."),
    ("ca_mask", "Boolean mask for residues with valid C-alpha labels."),
    ("atom14_positions", "Atom14 label coordinates in Angstroms with shape (L, 14, 3)."),
    ("atom14_mask", "Boolean mask for real/resolved atom14 slots with shape (L, 14)."),
    ("resolution", "Experimental structure resolution in Angstroms, or 0.0 if unknown in source metadata."),
    ("feature_sha256", "SHA256 of the source processed feature NPZ for this chain."),
    ("label_sha256", "SHA256 of the source processed label NPZ for this chain."),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and optionally upload the public nanoFold train/validation data "
            "as a columnar Hugging Face DatasetDict."
        )
    )
    parser.add_argument("--repo-id", default="", help="Target HF dataset repo, e.g. org/nanofold-public.")
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VAL_MANIFEST)
    parser.add_argument("--all-manifest", type=Path, default=DEFAULT_ALL_MANIFEST)
    parser.add_argument("--processed-features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--processed-labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--fingerprint", type=Path, default=DEFAULT_FINGERPRINT)
    parser.add_argument("--manifest-lock", type=Path, default=DEFAULT_MANIFEST_LOCK)
    parser.add_argument(
        "--eval-yaml",
        type=Path,
        default=DEFAULT_EVAL_YAML,
        help="Optional Hugging Face benchmark eval.yaml to upload at the dataset repo root when present.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional datasets cache directory. Useful in sandboxes where ~/.cache is not writable.",
    )
    parser.add_argument("--readme-out", type=Path, default=None, help="Write the generated HF dataset README here.")
    parser.add_argument("--save-to-disk", type=Path, default=None, help="Save the HF DatasetDict locally.")
    parser.add_argument(
        "--max-examples-per-split",
        type=int,
        default=0,
        help="Debug limit for local smoke checks. Leave at 0 for the full public dataset.",
    )
    parser.add_argument("--max-shard-size", default="500MB", help="Passed to DatasetDict.push_to_hub.")
    parser.add_argument("--num-proc", type=int, default=None, help="Optional parallelism for push_to_hub.")
    parser.add_argument("--commit-message", default="Upload public nanoFold dataset")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and render the README without HF upload.")
    parser.add_argument("--skip-push", action="store_true", help="Build/save locally but do not push to the Hub.")
    parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create the HF dataset repo as private when it does not already exist.",
    )
    return parser.parse_args()


def _require_hf_modules() -> tuple[Any, Any]:
    try:
        hf_datasets = importlib.import_module("datasets")
        huggingface_hub = importlib.import_module("huggingface_hub")
    except ImportError as exc:  # pragma: no cover - depends on optional local env
        raise SystemExit(OPTIONAL_DEPENDENCY_MESSAGE) from exc
    return hf_datasets, huggingface_hub


def _scalar(value: np.ndarray, *, dtype: type[int] | type[float] | type[str]) -> int | float | str:
    item = np.asarray(value).item()
    return dtype(item)


def _chain_parts(chain_id: str) -> tuple[str, str]:
    pdb_id, sep, chain_suffix = chain_id.partition("_")
    if not sep or not pdb_id or not chain_suffix:
        raise ValueError(f"Expected chain id in <pdb_id>_<chain_id> form, got {chain_id!r}.")
    return pdb_id.lower(), chain_suffix


def _optional_array(data: Mapping[str, np.ndarray], key: str, shape: tuple[int, ...], dtype: str) -> np.ndarray:
    if key in data:
        return np.asarray(data[key])
    return np.zeros(shape, dtype=dtype)


def _optional_float(data: Mapping[str, np.ndarray], key: str) -> float:
    if key not in data:
        return float("nan")
    return float(np.asarray(data[key]).item())


def _optional_int(data: Mapping[str, np.ndarray], key: str) -> int:
    if key not in data:
        return -1
    return int(np.asarray(data[key]).item())


def _check_npz_chain_id(data: Mapping[str, np.ndarray], expected_chain_id: str, path: Path) -> None:
    if "chain_id" not in data:
        return
    observed = str(np.asarray(data["chain_id"]).item())
    if observed != expected_chain_id:
        raise ValueError(f"{path} contains chain_id={observed!r}, expected {expected_chain_id!r}.")


def _public_row(
    *,
    chain_id: str,
    split: str,
    processed_features_dir: Path,
    processed_labels_dir: Path,
) -> dict[str, Any]:
    feature_path = chain_npz_path(processed_features_dir, chain_id)
    label_path = chain_npz_path(processed_labels_dir, chain_id)
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing public feature NPZ for {chain_id}: {feature_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing public label NPZ for {chain_id}: {label_path}")

    with np.load(feature_path) as feature_npz, np.load(label_path) as label_npz:
        features = {key: np.asarray(feature_npz[key]) for key in feature_npz.files}
        labels = {key: np.asarray(label_npz[key]) for key in label_npz.files}

    _check_npz_chain_id(features, chain_id, feature_path)
    _check_npz_chain_id(labels, chain_id, label_path)

    aatype = np.asarray(features["aatype"], dtype=np.int32)
    msa = np.asarray(features["msa"], dtype=np.int32)
    deletions = np.asarray(features["deletions"], dtype=np.int32)
    length = int(aatype.shape[0])
    pdb_id, pdb_chain_id = _chain_parts(chain_id)

    label_residue_index = labels.get("residue_index")
    if label_residue_index is not None and not np.array_equal(features["residue_index"], label_residue_index):
        raise ValueError(f"{chain_id} feature and label residue_index arrays differ.")

    return {
        "chain_id": chain_id,
        "pdb_id": pdb_id,
        "pdb_chain_id": pdb_chain_id,
        "split": split,
        "length": length,
        "msa_depth": int(msa.shape[0]),
        "template_count": int(_optional_array(features, "template_aatype", (0, length), "int32").shape[0]),
        "aatype": aatype,
        "msa": msa,
        "deletions": deletions,
        "residue_index": np.asarray(features["residue_index"], dtype=np.int32),
        "between_segment_residues": np.asarray(features["between_segment_residues"], dtype=np.int32),
        "projection_seq_identity": _optional_float(features, "projection_seq_identity"),
        "projection_alignment_coverage": _optional_float(features, "projection_alignment_coverage"),
        "projection_aligned_fraction": _optional_float(features, "projection_aligned_fraction"),
        "projection_valid_ca_count": _optional_int(features, "projection_valid_ca_count"),
        "template_aatype": _optional_array(features, "template_aatype", (0, length), "int32"),
        "template_ca_coords": _optional_array(features, "template_ca_coords", (0, length, 3), "float32"),
        "template_ca_mask": _optional_array(features, "template_ca_mask", (0, length), "bool"),
        "ca_coords": np.asarray(labels["ca_coords"], dtype=np.float32),
        "ca_mask": np.asarray(labels["ca_mask"], dtype=bool),
        "atom14_positions": np.asarray(labels["atom14_positions"], dtype=np.float32),
        "atom14_mask": np.asarray(labels["atom14_mask"], dtype=bool),
        "resolution": _optional_float(labels, "resolution"),
        "feature_sha256": sha256_file(feature_path),
        "label_sha256": sha256_file(label_path),
    }


def generate_rows(
    manifest_path: str,
    split: str,
    processed_features_dir: str,
    processed_labels_dir: str,
    max_examples: int = 0,
) -> Iterator[dict[str, Any]]:
    chain_ids = read_manifest(manifest_path)
    if max_examples > 0:
        chain_ids = chain_ids[:max_examples]
    for chain_id in chain_ids:
        yield _public_row(
            chain_id=chain_id,
            split=split,
            processed_features_dir=Path(processed_features_dir),
            processed_labels_dir=Path(processed_labels_dir),
        )


def build_hf_features(hf_datasets: Any) -> Any:
    return hf_datasets.Features(
        {
            "chain_id": hf_datasets.Value("string"),
            "pdb_id": hf_datasets.Value("string"),
            "pdb_chain_id": hf_datasets.Value("string"),
            "split": hf_datasets.Value("string"),
            "length": hf_datasets.Value("int32"),
            "msa_depth": hf_datasets.Value("int32"),
            "template_count": hf_datasets.Value("int32"),
            "aatype": hf_datasets.List(hf_datasets.Value("int32")),
            "msa": hf_datasets.List(hf_datasets.List(hf_datasets.Value("int32"))),
            "deletions": hf_datasets.List(hf_datasets.List(hf_datasets.Value("int32"))),
            "residue_index": hf_datasets.List(hf_datasets.Value("int32")),
            "between_segment_residues": hf_datasets.List(hf_datasets.Value("int32")),
            "projection_seq_identity": hf_datasets.Value("float32"),
            "projection_alignment_coverage": hf_datasets.Value("float32"),
            "projection_aligned_fraction": hf_datasets.Value("float32"),
            "projection_valid_ca_count": hf_datasets.Value("int32"),
            "template_aatype": hf_datasets.List(hf_datasets.List(hf_datasets.Value("int32"))),
            "template_ca_coords": hf_datasets.List(
                hf_datasets.List(hf_datasets.List(hf_datasets.Value("float32")))
            ),
            "template_ca_mask": hf_datasets.List(hf_datasets.List(hf_datasets.Value("bool"))),
            "ca_coords": hf_datasets.Array2D(shape=(None, 3), dtype="float32"),
            "ca_mask": hf_datasets.List(hf_datasets.Value("bool")),
            "atom14_positions": hf_datasets.Array3D(shape=(None, 14, 3), dtype="float32"),
            "atom14_mask": hf_datasets.Array2D(shape=(None, 14), dtype="bool"),
            "resolution": hf_datasets.Value("float32"),
            "feature_sha256": hf_datasets.Value("string"),
            "label_sha256": hf_datasets.Value("string"),
        }
    )


def build_dataset_dict(args: argparse.Namespace, hf_datasets: Any) -> Any:
    cache_dir = str(args.cache_dir) if args.cache_dir else None
    features = build_hf_features(hf_datasets)
    common_kwargs = {
        "processed_features_dir": str(args.processed_features_dir),
        "processed_labels_dir": str(args.processed_labels_dir),
        "max_examples": int(args.max_examples_per_split),
    }
    return hf_datasets.DatasetDict(
        {
            "train": hf_datasets.Dataset.from_generator(
                generate_rows,
                gen_kwargs={"manifest_path": str(args.train_manifest), "split": "train", **common_kwargs},
                features=features,
                cache_dir=cache_dir,
            ),
            "validation": hf_datasets.Dataset.from_generator(
                generate_rows,
                gen_kwargs={"manifest_path": str(args.val_manifest), "split": "validation", **common_kwargs},
                features=features,
                cache_dir=cache_dir,
            ),
        }
    )


def _load_fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Fingerprint must be a JSON object: {path}")
    return raw


def _summary(args: argparse.Namespace) -> dict[str, Any]:
    train_ids = read_manifest(args.train_manifest)
    val_ids = read_manifest(args.val_manifest)
    fingerprint = _load_fingerprint(args.fingerprint)
    return {
        "train_count": len(train_ids),
        "validation_count": len(val_ids),
        "total_count": len(set(train_ids + val_ids)),
        "train_manifest_sha256": sha256_file(args.train_manifest),
        "validation_manifest_sha256": sha256_file(args.val_manifest),
        "dataset_fingerprint": fingerprint,
    }


def render_dataset_card(summary: Mapping[str, Any]) -> str:
    fingerprint = summary.get("dataset_fingerprint", {})
    feature_files_sha256 = ""
    label_files_sha256 = ""
    if isinstance(fingerprint, Mapping):
        feature_files_sha256 = str(fingerprint.get("feature_files_sha256", ""))
        label_files_sha256 = str(fingerprint.get("label_files_sha256", ""))

    feature_rows = "\n".join(f"| `{name}` | {description} |" for name, description in FEATURE_DESCRIPTIONS)

    return f"""---
pretty_name: NanoFold Public
license: cc-by-4.0
task_categories:
- feature-extraction
tags:
- protein-folding
- protein-structure
- structural-biology
- openproteinset
- openfold
- nanofold
size_categories:
- 10K<n<100K
---

# NanoFold Public

NanoFold Public is the public train/validation portion of the nanoFold protein-folding benchmark. It packages a compact, fixed, auditable subset of OpenProteinSet/OpenFold-derived protein structure training data for fast iteration on data-efficient folding models.

The dataset has `{summary["train_count"]}` train chains and `{summary["validation_count"]}` public validation chains. Each row is one protein chain. The original processed `.npz` tensors are unrolled into Hugging Face Dataset columns so users can load the data with `datasets.load_dataset` and work directly with typed arrays/lists.

## Why This Dataset Exists

OpenProteinSet was created to provide a large, openly available AlphaFold-style corpus of MSAs, structural template hits, and structure-related assets. NanoFold uses that ecosystem as a reproducible foundation, then deliberately samples a much smaller fixed benchmark so researchers can study protein folding under data scarcity and compute constraints.

This matters because many protein-folding ideas are hard to evaluate when every experiment requires massive data pipelines, large models, and long training runs. NanoFold is designed for smaller models, ablations, objectives, curricula, and optimization experiments where iteration speed is part of the scientific value.

## Source And Sampling

The public NanoFold data was built from the OpenFold/OpenProteinSet PDB-chain data foundation plus RCSB mmCIF coordinate files. OpenProteinSet provides precomputed MSA assets and OpenFold-compatible source metadata; RCSB mmCIF files provide the experimental atom coordinates used to build C-alpha and atom14 labels.

The candidate pool was filtered to keep the benchmark small, clean, and learnable:

- single-chain monomer examples
- standard amino-acid sequences
- chain length between 40 and 256 residues
- resolution at or below 3.0 Angstrom when known
- required OpenProteinSet/OpenFold MSA assets available
- processable coordinate projection into NanoFold's atom14 label schema

The split was then sampled with leakage controls and structural stratification. Chains were grouped to keep PDB entries and coarse sequence clusters disjoint across splits. The public train/validation allocation was balanced across broad structural and quality metadata, including secondary-structure class, domain-architecture class, length bin, and resolution bin. These fields are derived from OpenFold/OpenProteinSet chain metadata together with structural classification sources used by NanoFold's manifest builder.

The goal is not to mirror the full PDB distribution perfectly. The goal is a representative, fixed, tractable slice of protein fold space that rewards models that learn useful geometry from limited biological data.

## Splits

| Split | Chains |
|---|---:|
| `train` | {summary["train_count"]} |
| `validation` | {summary["validation_count"]} |

## Column Schema

| Column | Meaning |
|---|---|
{feature_rows}

Shape notation:

- `L` is the chain length.
- `N` is the retained MSA depth.
- `T` is the number of templates. Official public NanoFold features use `T=0`; template columns are present for schema consistency.
- Atom14 slot order follows the AlphaFold-style atom14 convention used by NanoFold: slots 0-3 are `N`, `CA`, `C`, `O`, followed by residue-specific side-chain atoms.

## Loading

```python
from datasets import load_dataset

ds = load_dataset("YOUR_ORG/nanofold-public")
train = ds["train"]
example = train[0]

print(example["chain_id"], example["length"], example["msa_depth"])
print(len(example["aatype"]))
print(len(example["msa"]), len(example["msa"][0]))
```

For PyTorch:

```python
torch_ds = ds.with_format("torch")
example = torch_ds["train"][0]

atom14 = example["atom14_positions"]  # (L, 14, 3)
atom14_mask = example["atom14_mask"]  # (L, 14)
```

`msa`, `deletions`, and template columns are stored as nested list features because their first two dimensions are dynamic across chains. With `with_format("torch")`, non-empty per-row nested lists are converted to tensors by Hugging Face Datasets.

## Integrity

The public manifest SHA256 hashes are:

- train manifest: `{summary["train_manifest_sha256"]}`
- validation manifest: `{summary["validation_manifest_sha256"]}`

The processed public tensor fingerprints are:

- feature files fingerprint: `{feature_files_sha256}`
- label files fingerprint: `{label_files_sha256}`

The dataset repository also includes the public NanoFold manifest/fingerprint metadata files used to audit this release.

## Intended Use

NanoFold Public is intended for:

- training and evaluating smaller protein-folding models
- testing data-efficient architectures and objectives
- prototyping AlphaFold-style geometry learning without full-scale data requirements
- teaching and reproducible benchmarking around protein-structure prediction

It is not intended to replace full-scale OpenProteinSet/OpenFold training data. It is a deliberately constrained benchmark slice.

## Data Policy

This public dataset includes only NanoFold train and public validation examples. It does not include hidden validation chains, hidden labels, private salts, private manifests, model checkpoints, or external template lookup results.

Official NanoFold competition runs should not add external structures, pretrained weights, external MSA retrieval, template lookup, network access, or hidden-label exposure unless a track explicitly allows those resources.

## Upstream References

- OpenProteinSet: Training data for structural biology at scale, arXiv:2308.05326.
- OpenFold: a trainable open-source reproduction of AlphaFold-style protein-structure prediction.
- RCSB Protein Data Bank mmCIF coordinate archive.

OpenProteinSet is distributed under CC BY 4.0. RCSB PDB archive data files are available under CC0 1.0; users are encouraged to attribute original PDB structure depositors where possible.
"""


def _validate_public_inputs(args: argparse.Namespace) -> None:
    required = (
        args.train_manifest,
        args.val_manifest,
        args.processed_features_dir,
        args.processed_labels_dir,
        args.fingerprint,
        args.manifest_lock,
    )
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing required public dataset input(s): " + ", ".join(missing))

    forbidden_parts = {
        ".nanofold_private",
        "hidden_processed_features",
        "hidden_processed_labels",
        "hidden_val.txt",
    }
    for path in required:
        as_posix = Path(path).as_posix()
        if any(part in as_posix for part in forbidden_parts):
            raise ValueError(f"Refusing to use hidden/private path for public HF upload: {path}")


def _upload_auxiliary_files(args: argparse.Namespace, huggingface_hub: Any, readme_text: str) -> None:
    api = huggingface_hub.HfApi()
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="dataset",
        path_or_fileobj=readme_text.encode("utf-8"),
        path_in_repo="README.md",
        commit_message="Add NanoFold public dataset card",
    )
    auxiliary_files = (
        (args.eval_yaml, "eval.yaml"),
        (args.train_manifest, "manifests/train.txt"),
        (args.val_manifest, "manifests/val.txt"),
        (args.all_manifest, "manifests/all.txt"),
        (args.fingerprint, "metadata/official_dataset_fingerprint.json"),
        (args.manifest_lock, "metadata/official_manifest_source.lock.json"),
    )
    for local_path, path_in_repo in auxiliary_files:
        if not local_path.exists():
            continue
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="dataset",
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            commit_message=f"Add {path_in_repo}",
        )


def _print_summary(summary: Mapping[str, Any]) -> None:
    print("NanoFold public HF dataset summary")
    print(f"- train chains: {summary['train_count']}")
    print(f"- validation chains: {summary['validation_count']}")
    print(f"- total public chains: {summary['total_count']}")
    print(f"- train manifest sha256: {summary['train_manifest_sha256']}")
    print(f"- validation manifest sha256: {summary['validation_manifest_sha256']}")


def main() -> None:
    args = parse_args()
    _validate_public_inputs(args)
    summary = _summary(args)
    readme_text = render_dataset_card(summary)

    if args.readme_out:
        args.readme_out.parent.mkdir(parents=True, exist_ok=True)
        args.readme_out.write_text(readme_text)
        print(f"Wrote dataset README to {args.readme_out.resolve()}")

    if args.dry_run:
        _print_summary(summary)
        for split_name, manifest in (("train", args.train_manifest), ("validation", args.val_manifest)):
            first = next(
                generate_rows(
                    manifest_path=str(manifest),
                    split=split_name,
                    processed_features_dir=str(args.processed_features_dir),
                    processed_labels_dir=str(args.processed_labels_dir),
                    max_examples=1,
                )
            )
            print(
                f"- {split_name} first row: {first['chain_id']} "
                f"L={first['length']} msa_depth={first['msa_depth']} templates={first['template_count']}"
            )
        return

    if not args.repo_id and args.save_to_disk is None:
        raise SystemExit("Provide --repo-id for upload, --save-to-disk for local export, or --dry-run.")

    hf_datasets, huggingface_hub = _require_hf_modules()
    dataset = build_dataset_dict(args, hf_datasets)

    if args.save_to_disk is not None:
        args.save_to_disk.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(args.save_to_disk))
        (args.save_to_disk / "README.md").write_text(readme_text)
        print(f"Saved HF DatasetDict to {args.save_to_disk.resolve()}")

    if args.skip_push:
        _print_summary(summary)
        return

    dataset.push_to_hub(
        args.repo_id,
        private=bool(args.private),
        max_shard_size=args.max_shard_size,
        num_proc=args.num_proc,
        commit_message=args.commit_message,
    )
    _upload_auxiliary_files(args, huggingface_hub, readme_text)
    print(f"Uploaded NanoFold public dataset to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
