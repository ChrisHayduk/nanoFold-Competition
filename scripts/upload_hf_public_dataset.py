"""Publish the public nanoFold train/validation data as a Hugging Face dataset.

The uploaded dataset is HF-native: each processed NPZ field is expanded into a
typed dataset column rather than stored as an opaque binary blob. Only public
train/validation manifests are read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import re
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Allow running as `python scripts/upload_hf_public_dataset.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofold.chain_paths import chain_npz_path
from nanofold.data import read_manifest
from nanofold.dataset_integrity import FINGERPRINT_COMPARISON_KEYS, verify_dataset_against_fingerprint
from nanofold.utils import sha256_file

DEFAULT_TRAIN_MANIFEST = Path("data/manifests/train.txt")
DEFAULT_VAL_MANIFEST = Path("data/manifests/val.txt")
DEFAULT_ALL_MANIFEST = Path("data/manifests/all.txt")
DEFAULT_FEATURES_DIR = Path("data/processed_features")
DEFAULT_LABELS_DIR = Path("data/processed_labels")
DEFAULT_FINGERPRINT = Path("leaderboard/official_dataset_fingerprint.json")
DEFAULT_MANIFEST_LOCK = Path("leaderboard/official_manifest_source.lock.json")
DEFAULT_EVAL_YAML = Path("eval.yaml")


@dataclass(frozen=True)
class PublicExport:
    summary: dict[str, Any]
    input_digest: str
    file_hashes: dict[str, tuple[str, str]]
    auxiliary_files: dict[str, bytes]


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
    ("msa_row_filter_sha256", "SHA256 of the recorded MSA row-filter pass applied to this feature NPZ, or empty if no filter was applied."),
    ("msa_rows_before_filter", "Number of MSA rows before the recorded row-filter pass, or -1 if unavailable."),
    ("msa_rows_removed_by_filter", "Number of non-query MSA rows removed in the recorded row-filter pass, or -1 if unavailable."),
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
        "--msa-row-filter-source-lock",
        type=Path,
        default=None,
        help="Reviewed public MSA filtering summary under leaderboard/; private/staged metadata is rejected.",
    )
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
        help="Local-only debug limit; requires --dry-run or --skip-push. Uploads must contain both complete public splits.",
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


def _optional_str(data: Mapping[str, np.ndarray], key: str) -> str:
    if key not in data:
        return ""
    return str(np.asarray(data[key]).item())


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
    expected_file_hashes: tuple[str, str] | None = None,
) -> dict[str, Any]:
    feature_path = chain_npz_path(processed_features_dir, chain_id)
    label_path = chain_npz_path(processed_labels_dir, chain_id)
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing public feature NPZ for {chain_id}: {feature_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing public label NPZ for {chain_id}: {label_path}")

    feature_bytes = feature_path.read_bytes()
    label_bytes = label_path.read_bytes()
    file_hashes = (hashlib.sha256(feature_bytes).hexdigest(), hashlib.sha256(label_bytes).hexdigest())
    if expected_file_hashes is not None and file_hashes != expected_file_hashes:
        raise ValueError(f"Public tensors changed after validation for {chain_id}")
    with np.load(io.BytesIO(feature_bytes)) as feature_npz, np.load(io.BytesIO(label_bytes)) as label_npz:
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
        "msa_row_filter_sha256": _optional_str(features, "msa_row_filter_sha256"),
        "msa_rows_before_filter": _optional_int(features, "msa_rows_before_filter"),
        "msa_rows_removed_by_filter": _optional_int(features, "msa_rows_removed_by_filter"),
        "template_count": int(_optional_array(features, "template_aatype", (0, length), "int32").shape[0]),
        "aatype": aatype,
        "msa": msa,
        "deletions": deletions,
        "residue_index": np.asarray(features["residue_index"], dtype=np.int32),
        "between_segment_residues": np.asarray(features["between_segment_residues"], dtype=np.int32),
        "projection_seq_identity": _optional_float(labels, "projection_seq_identity"),
        "projection_alignment_coverage": _optional_float(labels, "projection_alignment_coverage"),
        "projection_aligned_fraction": _optional_float(labels, "projection_aligned_fraction"),
        "projection_valid_ca_count": _optional_int(labels, "projection_valid_ca_count"),
        "template_aatype": _optional_array(features, "template_aatype", (0, length), "int32"),
        "template_ca_coords": _optional_array(features, "template_ca_coords", (0, length, 3), "float32"),
        "template_ca_mask": _optional_array(features, "template_ca_mask", (0, length), "bool"),
        "ca_coords": np.asarray(labels["ca_coords"], dtype=np.float32),
        "ca_mask": np.asarray(labels["ca_mask"], dtype=bool),
        "atom14_positions": np.asarray(labels["atom14_positions"], dtype=np.float32),
        "atom14_mask": np.asarray(labels["atom14_mask"], dtype=bool),
        "resolution": _optional_float(labels, "resolution"),
        "feature_sha256": file_hashes[0],
        "label_sha256": file_hashes[1],
    }


def generate_rows(
    manifest_path: str,
    split: str,
    processed_features_dir: str,
    processed_labels_dir: str,
    max_examples: int = 0,
    input_digest: str = "",
    expected_file_hashes: Mapping[str, tuple[str, str]] | None = None,
) -> Iterator[dict[str, Any]]:
    # input_digest binds the Hugging Face generator cache to verified input bytes.
    chain_ids = read_manifest(manifest_path)
    if max_examples > 0:
        chain_ids = chain_ids[:max_examples]
    for chain_id in chain_ids:
        yield _public_row(
            chain_id=chain_id,
            split=split,
            processed_features_dir=Path(processed_features_dir),
            processed_labels_dir=Path(processed_labels_dir),
            expected_file_hashes=expected_file_hashes[chain_id] if expected_file_hashes is not None else None,
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
            "msa_row_filter_sha256": hf_datasets.Value("string"),
            "msa_rows_before_filter": hf_datasets.Value("int32"),
            "msa_rows_removed_by_filter": hf_datasets.Value("int32"),
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


def build_dataset_dict(args: argparse.Namespace, hf_datasets: Any, validated: PublicExport) -> Any:
    cache_dir = str(args.cache_dir) if args.cache_dir else None
    features = build_hf_features(hf_datasets)
    common_kwargs = {
        "processed_features_dir": str(args.processed_features_dir),
        "processed_labels_dir": str(args.processed_labels_dir),
        "max_examples": int(args.max_examples_per_split),
        "input_digest": validated.input_digest,
        "expected_file_hashes": validated.file_hashes,
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


def _load_msa_row_filter_source_lock(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"MSA row filter source lock must be a JSON object: {path}")
    aggregate = raw.get("aggregate_processed_filtering")
    row_filter = raw.get("msa_row_filter")
    if not isinstance(aggregate, Mapping) or not isinstance(row_filter, Mapping):
        raise ValueError(f"MSA row filter source lock is missing expected sections: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "filter_sha256": str(row_filter.get("sha256", "")),
        "excluded_sequence_count": row_filter.get("excluded_sequence_count"),
        "cumulative_msa_rows_removed": aggregate.get("cumulative_msa_rows_removed"),
        "final_total_msa_rows_after_filter": aggregate.get("final_total_msa_rows_after_filter"),
        "pass_count": aggregate.get("pass_count"),
    }


def _summary(args: argparse.Namespace, fingerprint: dict[str, Any]) -> dict[str, Any]:
    train_ids = read_manifest(args.train_manifest)
    val_ids = read_manifest(args.val_manifest)
    limit = int(args.max_examples_per_split)
    if limit > 0:
        train_ids, val_ids = train_ids[:limit], val_ids[:limit]
    return {
        "local_subset": limit > 0,
        "train_count": len(train_ids),
        "validation_count": len(val_ids),
        "total_count": len(set(train_ids + val_ids)),
        "train_manifest_sha256": sha256_file(args.train_manifest),
        "validation_manifest_sha256": sha256_file(args.val_manifest),
        "dataset_fingerprint": fingerprint,
        "msa_row_filter_source_lock": _load_msa_row_filter_source_lock(args.msa_row_filter_source_lock),
    }


def render_dataset_card(summary: Mapping[str, Any]) -> str:
    subset_notice = "**Local debug subset: not a complete benchmark release and not uploadable.**\n\n" if summary.get("local_subset") else ""
    fingerprint = summary.get("dataset_fingerprint", {})
    feature_files_sha256 = ""
    label_files_sha256 = ""
    if isinstance(fingerprint, Mapping):
        feature_files_sha256 = str(fingerprint.get("feature_files_sha256", ""))
        label_files_sha256 = str(fingerprint.get("label_files_sha256", ""))

    msa_filter = summary.get("msa_row_filter_source_lock", {})
    msa_filter_section = ""
    if isinstance(msa_filter, Mapping) and msa_filter:
        msa_filter_section = f"""
## MSA Row Filtering

The reviewed MSA filtering summary records the filter and aggregate row-removal counts shown below. Query rows are preserved. This provenance summary does not establish zero residual homology or improved model performance. Per-row metadata describes the recorded feature-filtering pass; consult the accompanying summary for its actual search thresholds and scope.

- MSA row filter SHA256: `{msa_filter.get("filter_sha256", "")}`
- public-safe source lock SHA256: `{msa_filter.get("sha256", "")}`
- excluded MSA-row sequence hashes: `{msa_filter.get("excluded_sequence_count", "")}`
- cumulative MSA rows removed across filtering passes: `{msa_filter.get("cumulative_msa_rows_removed", "")}`
"""

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

{subset_notice}NanoFold Public is the public train/validation portion of the nanoFold protein-folding benchmark. It packages a compact, fixed, auditable subset of OpenProteinSet/OpenFold-derived protein structure training data for fast iteration on data-efficient folding models.

The dataset has `{summary["train_count"]}` train chains and `{summary["validation_count"]}` public validation chains. Each row is one protein chain. The original processed `.npz` tensors are unrolled into Hugging Face Dataset columns so users can load the data with `datasets.load_dataset` and work directly with typed arrays/lists.

## Why This Dataset Exists

OpenProteinSet was created to provide a large, openly available AlphaFold-style corpus of MSAs, structural template hits, and structure-related assets. NanoFold uses that ecosystem as a reproducible foundation, then deliberately samples a much smaller fixed benchmark so researchers can study protein folding under data scarcity and compute constraints.

This matters because many protein-folding ideas are hard to evaluate when every experiment requires massive data pipelines, large models, and long training runs. NanoFold is designed for smaller models, ablations, objectives, curricula, and optimization experiments where iteration speed is part of the scientific value.

## Source And Sampling

The public NanoFold data was built from the OpenFold/OpenProteinSet PDB-chain data foundation plus RCSB mmCIF coordinate files. OpenProteinSet provides precomputed MSA assets and OpenFold-compatible source metadata; RCSB mmCIF files provide the experimental atom coordinates used to build C-alpha and atom14 labels.

The recorded candidate filters include:

- individual protein chains selected using the source cache oligomeric-count field; this does not establish biological monomer status
- standard amino-acid sequences
- chain length between 40 and 256 residues
- resolution at or below 3.0 Angstrom when known
- required OpenProteinSet/OpenFold MSA assets available
- processable coordinate projection into NanoFold's atom14 label schema

The reviewed manifest source lock describes the split policy and structural stratification actually used for this release. PDB-entry and sequence-cluster disjointness are grouping controls, not proof of absence of all evolutionary relationships. Where the lock records broad domain classes and projected secondary-structure classes, those secondary fractions are fixed class-based proxies rather than measured secondary structure or independent axes of structural diversity. A CATH topology policy applies only when the reviewed public source lock records it. These controls do not establish representativeness of the full PDB or model performance.
{msa_filter_section}

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

ds = load_dataset("ChrisHayduk/nanofold-public")
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

The processed public tensor fingerprints for the Hugging Face release are:

- feature files fingerprint: `{feature_files_sha256}`
- label files fingerprint: `{label_files_sha256}`

Before export, the complete feature and label bytes are checked against the supplied public fingerprint, and the manifests and source lock are checked against the reviewed public contract. The dataset repository includes those validated public metadata files. For a local debug subset, these fingerprints describe the complete input dataset, not the subset.

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


def _reviewed_metadata_path(path: Path) -> None:
    """Publication metadata must be explicitly placed in the public release area."""
    resolved = path.resolve(strict=True)
    public_root = REPO_ROOT.resolve() / "leaderboard"
    if not resolved.is_relative_to(public_root):
        raise ValueError(f"Publication metadata must use a reviewed public leaderboard/ path: {path}")
    _reject_private_path(resolved)


def _reject_private_path(path: Path) -> None:
    for candidate in (path, path.resolve()):
        if any(part.lower().startswith((".nanofold_private", "results_private", "private_", "hidden_")) for part in candidate.parts):
            raise ValueError(f"Refusing hidden/private input for public HF export: {path}")


def _json_object(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Expected public JSON object: {path}")
    return raw


def _check_public_metadata(value: Any, *, location: str = "metadata") -> None:
    """Reject unredacted local paths and identifying data in public summary JSON."""
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key).lower()
            if any(token in key_text for token in ("salt", "secret", "token", "password")):
                raise ValueError(f"Private metadata field: {location}.{key}")
            if "hidden" in key_text and key_text != "hidden_val_size":
                raise ValueError(f"Hidden metadata field: {location}.{key}")
            if key_text in {"sequences", "sequence", "chain_ids", "per_chain", "excluded_sequence_sha256"}:
                raise ValueError(f"Identifying metadata field: {location}.{key}")
            _check_public_metadata(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for child in value:
            _check_public_metadata(child, location=location)
    elif isinstance(value, str):
        if value == "<private>":
            return
        normalized = value.replace("\\", "/")
        if (
            normalized.startswith(("/", "~/"))
            or re.match(r"^[A-Za-z]:/", normalized)
            or ".." in normalized.split("/")
            or any(marker in normalized.lower() for marker in (".nanofold_private", "results_private", "hidden_val.txt"))
            or ("/" in normalized and any(part.lower() in {"private", "hidden", "secrets"} for part in normalized.split("/")))
            or re.search(r"(?<![A-Za-z0-9])[0-9][A-Za-z0-9]{3}_[A-Za-z0-9]+", normalized)
            or re.fullmatch(r"[ARNDCQEGHILKMFPSTWYVX]{20,}", normalized)
        ):
            raise ValueError(f"Unsafe public metadata value at {location}")
        path_field = location.rsplit(".", 1)[-1] in {
            "path", "raw_root", "processed_features_dirs", "out_features_dir",
            "source_processed_features_dirs", "source_lock_path",
        }
        if path_field and normalized and not normalized.startswith(("data/", "leaderboard/")):
            raise ValueError(f"Unreviewed public metadata path at {location}")


def _validate_filter_summary(path: Path) -> dict[str, Any]:
    _reviewed_metadata_path(path)
    raw = _json_object(path)
    allowed_sections = {
        "schema_version", "artifact_type", "public_safe", "validation", "safety", "msa_row_filter",
        "processed_feature_filter_passes", "aggregate_processed_filtering",
    }
    if set(raw) != allowed_sections or raw.get("public_safe") is not True:
        raise ValueError("MSA filter metadata must be a reviewed public-safe source-lock summary")
    if raw.get("schema_version") != 1 or raw.get("artifact_type") != "msa_row_filter_source_lock":
        raise ValueError("Unsupported public MSA filter source-lock schema")
    safety = raw.get("safety")
    if safety != {
        "raw_sequences_included": False, "excluded_sequence_sha256_included": False,
        "per_chain_processed_audit_included": False, "private_paths_redacted": True,
    }:
        raise ValueError("MSA filter source lock has unsafe disclosure settings")
    _check_public_metadata(raw)
    # Require every unstructured field to come from the compact source-lock schema.
    allowed_fields = allowed_sections | {
        "final_audit_filter_hash_checked", "residual_homology_audited", "scope",
        "raw_sequences_included", "excluded_sequence_sha256_included", "per_chain_processed_audit_included",
        "private_paths_redacted", "path", "sha256", "filter_type", "threshold", "min_seq_id", "coverage",
        "cov_mode", "seq_id_mode", "alignment_mode", "excluded_sequence_count", "mmseqs", "binary",
        "version", "command", "max_seqs", "hit_count", "source", "mode", "manifests", "role", "chain_count",
        "raw_root", "processed_features_dirs", "msa_names", "stats", "chains", "missing_features",
        "missing_msa_files", "rows", "non_query_rows", "unique_non_query_rows", "heldout",
        "unique_target_sequence_count", "chain_data_cache", "operation", "out_features_dir",
        "source_processed_features_dirs", "filtered_chain_count", "missing_chain_count", "msa_row_filter_sha256",
        "total_msa_rows_before_filter", "total_msa_rows_after_filter", "total_msa_rows_removed", "per_chain_included",
        "pass_count", "cumulative_msa_rows_removed", "final_total_msa_rows_after_filter",
        "final_filtered_chain_count", "final_msa_row_filter_sha256",
    }

    def check_fields(value: Any) -> None:
        if isinstance(value, dict):
            if set(value) - allowed_fields:
                raise ValueError("Unrecognized field in public MSA filter summary")
            if value.get("role") == "hidden" and (value.get("path") != "<private>" or value.get("sha256") != "<private>"):
                raise ValueError("Hidden manifest provenance must be redacted in public filter summaries")
            for child in value.values():
                check_fields(child)
        elif isinstance(value, list):
            for child in value:
                check_fields(child)

    check_fields(raw)
    return raw


def _validate_public_inputs(args: argparse.Namespace) -> PublicExport:
    limit = int(args.max_examples_per_split)
    if limit < 0:
        raise ValueError("--max-examples-per-split must be nonnegative")
    if limit and not (args.dry_run or args.skip_push):
        raise ValueError("Debug subsets are local-only; use --dry-run or --skip-push")

    required = (
        args.train_manifest, args.val_manifest, args.all_manifest, args.processed_features_dir,
        args.processed_labels_dir, args.fingerprint, args.manifest_lock,
    )
    for path in required:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing required public dataset input: {path}")
        _reject_private_path(Path(path))
    _reviewed_metadata_path(args.fingerprint)
    _reviewed_metadata_path(args.manifest_lock)

    # The checked-in public contract is the release boundary. Private candidate splits
    # must be reviewed and promoted there before this publisher can expose them.
    public_paths = (
        (args.train_manifest, DEFAULT_TRAIN_MANIFEST), (args.val_manifest, DEFAULT_VAL_MANIFEST),
        (args.all_manifest, DEFAULT_ALL_MANIFEST), (args.manifest_lock, DEFAULT_MANIFEST_LOCK),
    )
    for supplied, canonical in public_paths:
        if Path(supplied).read_bytes() != (REPO_ROOT / canonical).read_bytes():
            raise ValueError(f"Input does not match the reviewed public contract: {canonical}")

    train_ids, val_ids, all_ids = (read_manifest(path) for path in (
        args.train_manifest, args.val_manifest, args.all_manifest,
    ))
    for ids in (train_ids, val_ids, all_ids):
        if len(ids) != len(set(ids)):
            raise ValueError("Public manifests must not contain duplicate chain IDs")
    if set(train_ids) & set(val_ids) or set(all_ids) != set(train_ids) | set(val_ids):
        raise ValueError("Public train/validation must be disjoint and all.txt must be their exact union")
    for chain_id in all_ids:
        _chain_parts(chain_id)

    lock = _json_object(args.manifest_lock)
    _check_public_metadata(lock)
    outputs = lock.get("outputs", {})
    if not isinstance(outputs, dict) or not isinstance(lock.get("args"), dict):
        raise ValueError("Public manifest source lock requires outputs and args objects")
    for name, path, count in (
        ("train", args.train_manifest, len(train_ids)), ("val", args.val_manifest, len(val_ids)),
        ("all", args.all_manifest, len(all_ids)),
    ):
        count_key = "unique_count" if name == "all" else f"{name}_count"
        if outputs.get(f"{name}_manifest_sha256") != sha256_file(path) or outputs.get(count_key) != count:
            raise ValueError(f"Public manifest source lock does not match {name} manifest hash/count")
        if name != "all" and lock.get("args", {}).get(f"{name}_size") != count:
            raise ValueError(f"Public manifest source lock args do not match {name} count")

    expected = _json_object(args.fingerprint)
    allowed_fingerprint_keys = set(FINGERPRINT_COMPARISON_KEYS) | {
        "manifests", "missing_feature_chain_ids", "missing_label_chain_ids", "source_lock_path",
        "train_manifest_chain_count", "val_manifest_chain_count", "train_manifest_sha256", "val_manifest_sha256",
    }
    if set(expected) - allowed_fingerprint_keys:
        raise ValueError("Unrecognized fields in public dataset fingerprint")
    if not set(FINGERPRINT_COMPARISON_KEYS).issubset(expected):
        raise ValueError("Public dataset fingerprint is missing required comparison fields")
    _check_public_metadata(expected)
    fingerprint = verify_dataset_against_fingerprint(
        processed_features_dir=args.processed_features_dir, processed_labels_dir=args.processed_labels_dir,
        train_manifest=args.train_manifest, val_manifest=args.val_manifest, expected_fingerprint_path=args.fingerprint,
        require_no_missing=True, require_labels=True, validate_schema=True,
        source_lock_path=args.manifest_lock, track_id=expected.get("track_id"),
    )
    file_hashes: dict[str, tuple[str, str]] = {}
    feature_hasher, label_hasher = hashlib.sha256(), hashlib.sha256()
    for chain_id in sorted(all_ids):
        paths = (chain_npz_path(args.processed_features_dir, chain_id), chain_npz_path(args.processed_labels_dir, chain_id))
        for path in paths:
            _reject_private_path(path)
        hashes = (sha256_file(paths[0]), sha256_file(paths[1]))
        file_hashes[chain_id] = hashes
        with np.load(paths[0]) as features, np.load(paths[1]) as labels:
            if any("chain_id" not in data.files for data in (features, labels)):
                raise ValueError("Public exported tensors must record their chain identity")
            if features["template_aatype"].shape[0] != 0:
                raise ValueError("Official public exports require templates disabled")
        for hasher, digest in zip((feature_hasher, label_hasher), hashes, strict=True):
            hasher.update(f"{chain_id}\t{digest}\n".encode())
    if (feature_hasher.hexdigest(), label_hasher.hexdigest()) != (
        fingerprint["feature_files_sha256"], fingerprint["label_files_sha256"],
    ):
        raise ValueError("Public tensors changed during export validation")

    auxiliary_paths = {
        "manifests/train.txt": args.train_manifest, "manifests/val.txt": args.val_manifest,
        "manifests/all.txt": args.all_manifest, "metadata/official_dataset_fingerprint.json": args.fingerprint,
        "metadata/official_manifest_source.lock.json": args.manifest_lock,
    }
    if args.eval_yaml is not None and args.eval_yaml.exists():
        _reject_private_path(args.eval_yaml)
        canonical_eval = REPO_ROOT / DEFAULT_EVAL_YAML
        if args.eval_yaml.read_bytes() != canonical_eval.read_bytes():
            raise ValueError("eval.yaml must match the reviewed public benchmark definition")
        config = yaml.safe_load(args.eval_yaml.read_text())
        if not isinstance(config, dict) or config.get("tasks") != [{"id": "nanofold", "config": "default", "split": "validation"}]:
            raise ValueError("eval.yaml must target only the public validation split")
        _check_public_metadata(config)
        auxiliary_paths["eval.yaml"] = args.eval_yaml
    if args.msa_row_filter_source_lock is not None:
        raw_filter = _validate_filter_summary(args.msa_row_filter_source_lock)
        recorded_filter = raw_filter.get("msa_row_filter", {}).get("sha256")
        if not isinstance(recorded_filter, str) or not re.fullmatch(r"[0-9a-f]{64}", recorded_filter):
            raise ValueError("Invalid filter digest in public MSA source lock")
        for chain_id in all_ids:
            with np.load(chain_npz_path(args.processed_features_dir, chain_id)) as data:
                if _optional_str(data, "msa_row_filter_sha256") != recorded_filter:
                    raise ValueError("MSA source lock does not match the exported feature filtering pass")
        auxiliary_paths["metadata/msa_row_filter_source_lock_public_safe.json"] = args.msa_row_filter_source_lock
    auxiliary_files = {name: path.read_bytes() for name, path in auxiliary_paths.items()}
    input_digest = hashlib.sha256(json.dumps({
        "fingerprint": fingerprint,
        "auxiliary": {name: hashlib.sha256(data).hexdigest() for name, data in auxiliary_files.items()},
    }, sort_keys=True).encode()).hexdigest()
    return PublicExport(_summary(args, fingerprint), input_digest, file_hashes, auxiliary_files)


def _upload_auxiliary_files(
    args: argparse.Namespace, huggingface_hub: Any, readme_text: str, validated: PublicExport,
) -> None:
    api = huggingface_hub.HfApi()
    files = {"README.md": readme_text.encode("utf-8"), **validated.auxiliary_files}
    for path_in_repo, payload in files.items():
        api.upload_file(
            repo_id=args.repo_id, repo_type="dataset", path_or_fileobj=payload,
            path_in_repo=path_in_repo, commit_message=f"Add {path_in_repo}",
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
    validated = _validate_public_inputs(args)
    summary = validated.summary
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

    if not args.repo_id and not args.skip_push:
        raise SystemExit("Provide --repo-id for upload, or --skip-push for local export.")
    if not args.repo_id and args.save_to_disk is None:
        raise SystemExit("Provide --repo-id for upload, --save-to-disk for local export, or --dry-run.")

    hf_datasets, huggingface_hub = _require_hf_modules()
    dataset = build_dataset_dict(args, hf_datasets, validated)
    for split in ("train", "validation"):
        if len(dataset[split]) != summary[f"{split}_count"]:
            raise ValueError(f"Exported {split} count does not match validated public inputs")

    if args.save_to_disk is not None:
        args.save_to_disk.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(args.save_to_disk))
        (args.save_to_disk / "README.md").write_text(readme_text)
        print(f"Saved HF DatasetDict to {args.save_to_disk.resolve()}")

    if args.skip_push:
        _print_summary(summary)
        return

    if _validate_public_inputs(args).input_digest != validated.input_digest:
        raise ValueError("Public inputs changed before publication")
    dataset.push_to_hub(
        args.repo_id,
        private=bool(args.private),
        max_shard_size=args.max_shard_size,
        num_proc=args.num_proc,
        commit_message=args.commit_message,
    )
    _upload_auxiliary_files(args, huggingface_hub, readme_text, validated)
    print(f"Uploaded NanoFold public dataset to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
