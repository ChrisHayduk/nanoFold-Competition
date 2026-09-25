from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from nanofold.chain_paths import chain_npz_path
from nanofold.dataset_integrity import build_dataset_fingerprint
from nanofold.utils import sha256_file
from scripts import upload_hf_public_dataset as exporter
from scripts.upload_hf_public_dataset import generate_rows, render_dataset_card
from scripts.write_msa_row_filter_source_lock import build_source_lock


def _write_npz_pair(
    features_dir: Path,
    labels_dir: Path,
    chain_id: str,
    *,
    length: int = 4,
    msa_depth: int = 2,
    filter_sha256: str = "",
    rows_before_filter: int = -1,
    rows_removed_by_filter: int = -1,
) -> None:
    feature_path = chain_npz_path(features_dir, chain_id)
    label_path = chain_npz_path(labels_dir, chain_id)
    optional_filter_fields = {}
    if filter_sha256:
        optional_filter_fields = {
            "msa_row_filter_sha256": np.asarray(filter_sha256),
            "msa_rows_before_filter": np.asarray(rows_before_filter, dtype=np.int32),
            "msa_rows_removed_by_filter": np.asarray(rows_removed_by_filter, dtype=np.int32),
        }
    np.savez_compressed(
        feature_path,
        chain_id=np.asarray(chain_id),
        aatype=np.arange(length, dtype=np.int32),
        msa=np.arange(msa_depth * length, dtype=np.int32).reshape(msa_depth, length),
        deletions=np.zeros((msa_depth, length), dtype=np.int32),
        residue_index=np.arange(length, dtype=np.int32),
        between_segment_residues=np.zeros((length,), dtype=np.int32),
        projection_seq_identity=np.asarray(1.0, dtype=np.float32),
        projection_alignment_coverage=np.asarray(1.0, dtype=np.float32),
        projection_aligned_fraction=np.asarray(1.0, dtype=np.float32),
        projection_valid_ca_count=np.asarray(length, dtype=np.int32),
        template_aatype=np.zeros((0, length), dtype=np.int32),
        template_ca_coords=np.zeros((0, length, 3), dtype=np.float32),
        template_ca_mask=np.zeros((0, length), dtype=bool),
        **optional_filter_fields,
    )
    np.savez_compressed(
        label_path,
        chain_id=np.asarray(chain_id),
        ca_coords=np.zeros((length, 3), dtype=np.float32),
        ca_mask=np.ones((length,), dtype=bool),
        atom14_positions=np.zeros((length, 14, 3), dtype=np.float32),
        atom14_mask=np.ones((length, 14), dtype=bool),
        residue_index=np.arange(length, dtype=np.int32),
        resolution=np.asarray(1.5, dtype=np.float32),
    )


def test_generate_rows_unrolls_npz_fields(tmp_path: Path) -> None:
    features_dir = tmp_path / "features"
    labels_dir = tmp_path / "labels"
    features_dir.mkdir()
    labels_dir.mkdir()
    manifest = tmp_path / "train.txt"
    manifest.write_text("1abc_A\n")
    _write_npz_pair(features_dir, labels_dir, "1abc_A")

    row = next(
        generate_rows(
            manifest_path=str(manifest),
            split="train",
            processed_features_dir=str(features_dir),
            processed_labels_dir=str(labels_dir),
        )
    )

    assert row["chain_id"] == "1abc_A"
    assert row["pdb_id"] == "1abc"
    assert row["pdb_chain_id"] == "A"
    assert row["split"] == "train"
    assert row["length"] == 4
    assert row["msa_depth"] == 2
    assert row["msa_row_filter_sha256"] == ""
    assert row["msa_rows_before_filter"] == -1
    assert row["msa_rows_removed_by_filter"] == -1
    assert row["template_count"] == 0
    assert row["msa"].shape == (2, 4)
    assert row["atom14_positions"].shape == (4, 14, 3)
    assert len(row["feature_sha256"]) == 64
    assert len(row["label_sha256"]) == 64


def test_generate_rows_includes_msa_row_filter_metadata(tmp_path: Path) -> None:
    features_dir = tmp_path / "features"
    labels_dir = tmp_path / "labels"
    features_dir.mkdir()
    labels_dir.mkdir()
    manifest = tmp_path / "train.txt"
    manifest.write_text("1abc_A\n")
    filter_sha256 = "f" * 64
    _write_npz_pair(
        features_dir,
        labels_dir,
        "1abc_A",
        msa_depth=3,
        filter_sha256=filter_sha256,
        rows_before_filter=5,
        rows_removed_by_filter=2,
    )

    row = next(
        generate_rows(
            manifest_path=str(manifest),
            split="train",
            processed_features_dir=str(features_dir),
            processed_labels_dir=str(labels_dir),
        )
    )

    assert row["msa_depth"] == 3
    assert row["msa_row_filter_sha256"] == filter_sha256
    assert row["msa_rows_before_filter"] == 5
    assert row["msa_rows_removed_by_filter"] == 2


def test_render_dataset_card_documents_columns_and_sampling() -> None:
    card = render_dataset_card(
        {
            "train_count": 10_000,
            "validation_count": 1_000,
            "total_count": 11_000,
            "train_manifest_sha256": "a" * 64,
            "validation_manifest_sha256": "b" * 64,
            "dataset_fingerprint": {
                "feature_files_sha256": "c" * 64,
                "label_files_sha256": "d" * 64,
            },
            "msa_row_filter_source_lock": {
                "filter_sha256": "e" * 64,
                "sha256": "f" * 64,
                "excluded_sequence_count": 12,
                "cumulative_msa_rows_removed": 34,
            },
        }
    )

    assert "OpenProteinSet" in card
    assert "structural stratification" in card
    assert "`msa`" in card
    assert "`msa_row_filter_sha256`" in card
    assert "MSA Row Filtering" in card
    assert "does not establish zero residual homology or improved model performance" in card
    assert "`atom14_positions`" in card
    assert 'load_dataset("ChrisHayduk/nanofold-public")' in card
    assert "complete feature and label bytes are checked" in card
    assert "does not establish biological monomer status" in card
    assert "fixed class-based proxies" in card
    assert "smaller protein-folding models" in card


def test_eval_yaml_declares_public_foldscore_task() -> None:
    config = yaml.safe_load(Path("eval.yaml").read_text())

    assert config["name"] == "NanoFold Public FoldScore"
    assert config["evaluation_framework"] == "nanofold"
    assert "FoldScore" in config["description"]
    assert config["tasks"] == [
        {
            "id": "nanofold",
            "config": "default",
            "split": "validation",
        }
    ]


class _FakeApi:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def upload_file(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class _FakeHub:
    def __init__(self) -> None:
        self.api = _FakeApi()

    def HfApi(self) -> _FakeApi:
        return self.api


def _write_contract(args: Namespace) -> None:
    ids = {name: exporter.read_manifest(getattr(args, f"{name}_manifest")) for name in ("train", "val", "all")}
    outputs: dict[str, Any] = {f"{name}_manifest_sha256": sha256_file(getattr(args, f"{name}_manifest")) for name in ids}
    outputs.update(train_count=len(ids["train"]), val_count=len(ids["val"]), unique_count=len(ids["all"]))
    args.manifest_lock.write_text(json.dumps({
        "args": {"train_size": len(ids["train"]), "val_size": len(ids["val"])}, "outputs": outputs,
    }))
    _write_fingerprint(args)


def _write_fingerprint(args: Namespace) -> None:
    fingerprint = build_dataset_fingerprint(
        processed_features_dir=args.processed_features_dir, processed_labels_dir=args.processed_labels_dir,
        train_manifest=args.train_manifest, val_manifest=args.val_manifest, require_no_missing=True,
        source_lock_path=args.manifest_lock, track_id="limited",
    )
    fingerprint["source_lock_path"] = "leaderboard/official_manifest_source.lock.json"
    args.fingerprint.write_text(json.dumps(fingerprint))


@pytest.fixture
def public_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Namespace:
    root = tmp_path / "public_repo"
    manifests, leaderboard = root / "data/manifests", root / "leaderboard"
    features, labels = root / "features", root / "labels"
    for path in (manifests, leaderboard, features, labels):
        path.mkdir(parents=True)
    args = Namespace(
        repo_id="example/nanofold", train_manifest=manifests / "train.txt", val_manifest=manifests / "val.txt",
        all_manifest=manifests / "all.txt", manifest_lock=leaderboard / "official_manifest_source.lock.json",
        fingerprint=leaderboard / "official_dataset_fingerprint.json", processed_features_dir=features,
        processed_labels_dir=labels, msa_row_filter_source_lock=None, eval_yaml=root / "eval.yaml",
        max_examples_per_split=0, dry_run=False, skip_push=False, cache_dir=None, readme_out=None,
        save_to_disk=None, private=True, max_shard_size="500MB", num_proc=None, commit_message="Test",
    )
    args.train_manifest.write_text("1abc_A\n2def_B\n")
    args.val_manifest.write_text("3ghi_C\n4jkl_D\n")
    args.all_manifest.write_text("1abc_A\n2def_B\n3ghi_C\n4jkl_D\n")
    args.eval_yaml.write_text(Path("eval.yaml").read_text())
    for chain in exporter.read_manifest(args.all_manifest):
        _write_npz_pair(features, labels, chain)
    _write_contract(args)
    monkeypatch.setattr(exporter, "REPO_ROOT", root)
    return args


def _edit_npz(path: Path, **changes: Any) -> None:
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    arrays.update(changes)
    np.savez_compressed(path, **arrays)


def test_validated_auxiliary_upload_uses_validated_bytes(public_inputs: Namespace) -> None:
    validated = exporter._validate_public_inputs(public_inputs)
    hub = _FakeHub()
    public_inputs.eval_yaml.write_text("mutated after validation")
    exporter._upload_auxiliary_files(public_inputs, hub, "readme", validated)
    uploaded = {call["path_in_repo"]: call["path_or_fileobj"] for call in hub.api.calls}
    assert set(uploaded) == {"README.md", "eval.yaml", "manifests/train.txt", "manifests/val.txt", "manifests/all.txt",
                             "metadata/official_dataset_fingerprint.json", "metadata/official_manifest_source.lock.json"}
    assert uploaded["eval.yaml"] == validated.auxiliary_files["eval.yaml"]
    assert all(isinstance(value, bytes) for value in uploaded.values())


@pytest.mark.parametrize("field", ["train_manifest", "val_manifest", "all_manifest", "manifest_lock"])
def test_copied_nonpublic_contract_is_rejected(public_inputs: Namespace, tmp_path: Path, field: str) -> None:
    replacement = public_inputs.manifest_lock.parent / "unreviewed.json" if field == "manifest_lock" else tmp_path / "alternate.txt"
    replacement.write_text("{}" if field == "manifest_lock" else "9xyz_Z\n")
    setattr(public_inputs, field, replacement)
    with pytest.raises(ValueError, match="reviewed public contract"):
        exporter._validate_public_inputs(public_inputs)


@pytest.mark.parametrize("corruption", ["all_extra", "all_missing", "overlap", "duplicate", "lock_hash", "lock_count"])
def test_inconsistent_public_contract_is_rejected(public_inputs: Namespace, corruption: str) -> None:
    if corruption == "all_extra":
        public_inputs.all_manifest.write_text(public_inputs.all_manifest.read_text() + "9xyz_Z\n")
    elif corruption == "all_missing":
        public_inputs.all_manifest.write_text("1abc_A\n")
    elif corruption == "overlap":
        public_inputs.val_manifest.write_text("1abc_A\n")
    elif corruption == "duplicate":
        public_inputs.train_manifest.write_text("1abc_A\n1abc_A\n")
    else:
        lock = json.loads(public_inputs.manifest_lock.read_text())
        lock["outputs"]["train_manifest_sha256" if corruption == "lock_hash" else "train_count"] = "bad"
        public_inputs.manifest_lock.write_text(json.dumps(lock))
    with pytest.raises(ValueError):
        exporter._validate_public_inputs(public_inputs)


def test_changed_tensor_bytes_cannot_use_frozen_fingerprint(public_inputs: Namespace) -> None:
    _edit_npz(chain_npz_path(public_inputs.processed_features_dir, "1abc_A"), deletions=np.ones((2, 4), dtype=np.int32))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        exporter._validate_public_inputs(public_inputs)


@pytest.mark.parametrize("field", ["fingerprint", "manifest_lock", "msa_row_filter_source_lock", "eval_yaml"])
def test_private_or_symlinked_auxiliary_paths_are_rejected(public_inputs: Namespace, tmp_path: Path, field: str) -> None:
    private_dir = tmp_path / "private_candidate"
    private_dir.mkdir()
    source = getattr(public_inputs, field) or public_inputs.fingerprint
    private_file = private_dir / source.name
    private_file.write_bytes(source.read_bytes())
    link = public_inputs.manifest_lock.parent / "looks_public.json"
    link.symlink_to(private_file)
    setattr(public_inputs, field, link)
    with pytest.raises(ValueError, match="private|reviewed public"):
        exporter._validate_public_inputs(public_inputs)


def test_fingerprint_cannot_carry_extra_private_metadata(public_inputs: Namespace) -> None:
    fingerprint = json.loads(public_inputs.fingerprint.read_text())
    fingerprint["debug_notes"] = {"hidden_chain_ids": ["9xyz_Z"]}
    public_inputs.fingerprint.write_text(json.dumps(fingerprint))
    with pytest.raises(ValueError, match="Unrecognized fields"):
        exporter._validate_public_inputs(public_inputs)


def test_debug_subset_is_local_only_and_card_counts_match(public_inputs: Namespace) -> None:
    public_inputs.max_examples_per_split = 1
    with pytest.raises(ValueError, match="local-only"):
        exporter._validate_public_inputs(public_inputs)
    public_inputs.skip_push = True
    validated = exporter._validate_public_inputs(public_inputs)
    assert validated.summary["train_count"] == validated.summary["validation_count"] == 1
    assert "Local debug subset" in render_dataset_card(validated.summary)


class _FakeDataset(dict[str, list[dict[str, Any]]]):
    def __init__(self, rows: dict[str, list[dict[str, Any]]]) -> None:
        super().__init__(rows)
        self.pushed = False

    def push_to_hub(self, *args: Any, **kwargs: Any) -> None:
        self.pushed = True


class _FakeDatasets:
    def __init__(self) -> None:
        self.kwargs: list[dict[str, Any]] = []
        self.cache: dict[str, list[dict[str, Any]]] = {}
        self.Dataset = self
        self.last_dataset: _FakeDataset | None = None

    def from_generator(self, generator: Any, **kwargs: Any) -> list[dict[str, Any]]:
        self.kwargs.append(kwargs)
        generator_args = kwargs["gen_kwargs"]
        key = json.dumps(generator_args, sort_keys=True)
        if key not in self.cache:
            self.cache[key] = list(generator(**generator_args))
        return self.cache[key]

    def DatasetDict(self, rows: dict[str, list[dict[str, Any]]]) -> _FakeDataset:
        self.last_dataset = _FakeDataset(rows)
        return self.last_dataset


def test_generator_cache_changes_with_tensor_bytes(public_inputs: Namespace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exporter, "build_hf_features", lambda _: None)
    fake = _FakeDatasets()
    before = exporter._validate_public_inputs(public_inputs)
    first = exporter.build_dataset_dict(public_inputs, fake, before)
    _edit_npz(chain_npz_path(public_inputs.processed_features_dir, "1abc_A"), deletions=np.ones((2, 4), dtype=np.int32))
    _write_fingerprint(public_inputs)
    after = exporter._validate_public_inputs(public_inputs)
    second = exporter.build_dataset_dict(public_inputs, fake, after)
    assert before.input_digest != after.input_digest
    assert np.all(first["train"][0]["deletions"] == 0)
    assert np.all(second["train"][0]["deletions"] == 1)


def test_generator_rejects_tensor_change_after_validation(public_inputs: Namespace) -> None:
    validated = exporter._validate_public_inputs(public_inputs)
    _edit_npz(chain_npz_path(public_inputs.processed_features_dir, "1abc_A"), deletions=np.ones((2, 4), dtype=np.int32))
    with pytest.raises(ValueError, match="changed after validation"):
        next(generate_rows(str(public_inputs.train_manifest), "train", str(public_inputs.processed_features_dir),
                           str(public_inputs.processed_labels_dir), expected_file_hashes=validated.file_hashes))


def test_main_stops_before_hf_import_on_invalid_metadata(public_inputs: Namespace, monkeypatch: pytest.MonkeyPatch) -> None:
    public_inputs.all_manifest.write_text("9xyz_Z\n")
    monkeypatch.setattr(exporter, "parse_args", lambda: public_inputs)
    monkeypatch.setattr(exporter, "_require_hf_modules", lambda: pytest.fail("HF must not be loaded for invalid inputs"))
    with pytest.raises(ValueError):
        exporter.main()


def test_main_revalidates_before_any_network_write(public_inputs: Namespace, monkeypatch: pytest.MonkeyPatch) -> None:
    hub, fake = _FakeHub(), _FakeDatasets()
    monkeypatch.setattr(exporter, "parse_args", lambda: public_inputs)
    monkeypatch.setattr(exporter, "build_hf_features", lambda _: None)
    monkeypatch.setattr(exporter, "_require_hf_modules", lambda: (fake, hub))
    original = exporter.build_dataset_dict

    def mutate_after_build(*args: Any) -> Any:
        dataset = original(*args)
        public_inputs.all_manifest.write_text("9xyz_Z\n")
        return dataset

    monkeypatch.setattr(exporter, "build_dataset_dict", mutate_after_build)
    with pytest.raises(ValueError):
        exporter.main()
    assert fake.last_dataset is not None and not fake.last_dataset.pushed
    assert hub.api.calls == []


def _attach_public_filter_summary(args: Namespace, tmp_path: Path) -> Path:
    raw_filter = tmp_path / "row_filter.json"
    raw_filter.write_text(json.dumps({
        "source": {"manifests": [], "stats": {}}, "heldout": {"manifests": []}, "mmseqs": {},
        "chain_data_cache": {}, "threshold": {"min_seq_id": 0.3, "coverage": 0.8, "cov_mode": 0},
        "excluded_sequence_count": 2,
    }))
    digest = sha256_file(raw_filter)
    for chain in exporter.read_manifest(args.all_manifest):
        _edit_npz(chain_npz_path(args.processed_features_dir, chain), msa_row_filter_sha256=np.asarray(digest))
    _write_fingerprint(args)
    path = args.manifest_lock.parent / "msa_summary.json"
    summary = build_source_lock(row_filter_path=raw_filter, processed_filter_audit_paths=[])
    path.write_text(json.dumps(summary))
    args.msa_row_filter_source_lock = path
    return path


def test_safe_filter_summary_is_bound_to_exported_features(public_inputs: Namespace, tmp_path: Path) -> None:
    path = _attach_public_filter_summary(public_inputs, tmp_path)
    validated = exporter._validate_public_inputs(public_inputs)
    assert validated.auxiliary_files["metadata/msa_row_filter_source_lock_public_safe.json"] == path.read_bytes()
    _edit_npz(chain_npz_path(public_inputs.processed_features_dir, "1abc_A"), msa_row_filter_sha256=np.asarray("0" * 64))
    _write_fingerprint(public_inputs)
    with pytest.raises(ValueError, match="does not match the exported feature"):
        exporter._validate_public_inputs(public_inputs)


@pytest.mark.parametrize("corruption", ["private_flag", "arbitrary", "sequence_list", "hidden_manifest", "local_path", "staged_path"])
def test_filter_summary_rejects_unsafe_payload_before_hf(
    public_inputs: Namespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str,
) -> None:
    path = _attach_public_filter_summary(public_inputs, tmp_path)
    summary = json.loads(path.read_text())
    if corruption == "private_flag":
        summary["public_safe"] = False
    elif corruption == "arbitrary":
        summary["msa_row_filter"]["notes"] = "unreviewed payload"
    elif corruption == "sequence_list":
        summary["msa_row_filter"]["excluded_sequence_sha256"] = ["f" * 64]
    elif corruption == "hidden_manifest":
        summary["msa_row_filter"]["heldout"]["manifests"] = [{"role": "hidden", "path": "data/manifests/val.txt", "sha256": "f" * 64}]
    elif corruption == "local_path":
        summary["msa_row_filter"]["mmseqs"]["command"] = ["mmseqs", str((tmp_path / "secrets/input").resolve())]
    else:
        summary["msa_row_filter"]["path"] = "staging/cath_candidate/filter.json"
    path.write_text(json.dumps(summary))
    monkeypatch.setattr(exporter, "parse_args", lambda: public_inputs)
    monkeypatch.setattr(exporter, "_require_hf_modules", lambda: pytest.fail("Unsafe metadata reached HF"))
    with pytest.raises(ValueError):
        exporter.main()


def test_full_validated_export_reaches_only_fake_hf(public_inputs: Namespace, monkeypatch: pytest.MonkeyPatch) -> None:
    hub, fake = _FakeHub(), _FakeDatasets()
    monkeypatch.setattr(exporter, "parse_args", lambda: public_inputs)
    monkeypatch.setattr(exporter, "build_hf_features", lambda _: None)
    monkeypatch.setattr(exporter, "_require_hf_modules", lambda: (fake, hub))
    exporter.main()
    assert fake.last_dataset is not None and fake.last_dataset.pushed
    assert len(fake.last_dataset["train"]) == len(fake.last_dataset["validation"]) == 2
    assert len(hub.api.calls) == 7
