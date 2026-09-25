from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from scripts.verify_hidden_manifest import verify


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value))


def _fixture(tmp_path: Path, *, duplicate_sequence: bool = False) -> argparse.Namespace:
    ids = ["1aaa_A", "2bbb_A", "3ccc_A", "4ddd_A"]
    cache = tmp_path / "chains.json"
    _write_json(cache, {
        chain: {"sequence": ("A" if duplicate_sequence and i == 1 else residue) * 40,
                "seq_length": 40, "resolution": 2.0}
        for i, (chain, residue) in enumerate(zip(ids, "ACDE", strict=True))
    })
    metadata = tmp_path / "metadata.json"
    _write_json(metadata, {
        "schema_version": 2,
        "source": {"chain_data_cache_sha256": _sha(cache), "chain_id_namespace": "author_asym_id"},
        "config": {"min_len": 40, "max_len": 256, "max_resolution": 3.0, "require_monomer": False,
                   "sample_limit": 0, "max_unknown_aa_fraction": 0.0},
        "summary": {"eligible_candidate_count": len(ids), "candidate_count": len(ids)},
        "chains": [{
            "chain_id": chain, "accepted": True, "domain_architecture_class": "alpha",
            "domain_architecture_source": "cath", "metadata_source_count": 1,
            "cath_annotation_status": "annotated", "cath_architecture_ids": ["1.10"],
            "cath_topology_ids": ["1.10.10"], "cath_homologous_superfamily_ids": ["1.10.10.10"],
            "cath_domains": [{"domain_id": chain.replace("_", "") + "01", "class_id": "1",
                              "architecture_id": "1.10", "topology_id": "1.10.10",
                              "homologous_superfamily_id": "1.10.10.10"}],
        } for chain in ids],
    })
    clusters = tmp_path / "clusters.tsv"
    clusters.write_text("".join(f"{chain}\t{chain}\n" for chain in ids))
    clustering = {"cluster_assignments_sha256": _sha(clusters)}
    paths = {}
    for name, members in {"train": ids[:1], "val": ids[1:2], "all": ids[:2], "hidden": ids[2:3]}.items():
        path = tmp_path / f"{name}.txt"
        path.write_text("\n".join(members) + "\n")
        paths[name] = path
    lock = tmp_path / "public.lock.json"
    _write_json(lock, {
        "sampling_policy": "cath-v2", "chain_data_cache_sha256": _sha(cache),
        "structure_metadata": {"sha256": _sha(metadata)}, "clustering": clustering,
        "args": {"hidden_val_size": 1, "min_len": 40, "max_len": 256, "max_resolution": 3.0,
                 "seed": 0, "min_seq_id": .3, "coverage": .8, "mmseqs_bin": "unused",
                 "cluster_tsv": str(clusters), "structure_metadata": str(metadata)},
        "outputs": {"train_manifest_sha256": _sha(paths["train"]), "train_count": 1,
                    "val_manifest_sha256": _sha(paths["val"]), "val_count": 1,
                    "all_manifest_sha256": _sha(paths["all"]), "unique_count": 2},
    })
    private_lock = tmp_path / "private.lock.json"
    exclusions = tmp_path / "private-exclusions.txt"
    exclusions.write_text("")
    _write_json(private_lock, {
        "sampling_policy": "cath-v2", "source_lock_sha256": _sha(lock),
        "chain_data_cache_sha256": _sha(cache), "hidden_val_manifest_sha256": _sha(paths["hidden"]),
        "hidden_val_count": 1, "structure_metadata": {"sha256": _sha(metadata)}, "clustering": clustering,
        "public_contract": {**{f"{name}_manifest_sha256": _sha(paths[name]) for name in ("train", "val", "all")},
                            "train_count": 1, "val_count": 1, "all_count": 2},
        "stratification": {"reference": "public_train_plus_public_val"},
        "filtering": {"private_processability_exclusion_count": 0,
                      "private_processability_exclusions": {"path": str(exclusions), "sha256": _sha(exclusions),
                                                           "canonical_ids_sha256": _sha(exclusions)}},
    })
    return argparse.Namespace(chain_data_cache=cache, lock_file=lock, public_train_manifest=paths["train"],
                              public_val_manifest=paths["val"], public_all_manifest=paths["all"],
                              hidden_manifest=paths["hidden"], private_lock_file=private_lock,
                              structure_metadata=None, cluster_tsv=None, mmseqs_bin=None)


def test_verification_requires_complete_private_evidence(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    report = verify(args)
    assert report["status"] == "ok"
    assert report["overlap"]["exact_sequences"] == 0
    assert report["stratification"]["reference_population"] == "public_train_plus_public_val"
    args.private_lock_file.unlink()
    with pytest.raises(FileNotFoundError, match="private hidden manifest source lock is required"):
        verify(args)


@pytest.mark.parametrize("field", ["source_lock_sha256", "chain_data_cache_sha256", "structure_metadata",
                                   "clustering", "public_contract", "sampling_policy", "stratification"])
def test_private_evidence_cannot_be_dropped(tmp_path: Path, field: str) -> None:
    args = _fixture(tmp_path)
    record = json.loads(args.private_lock_file.read_text())
    record.pop(field)
    _write_json(args.private_lock_file, record)
    with pytest.raises(ValueError, match="Private hidden manifest lock"):
        verify(args)


def test_swapping_public_members_does_not_pass_unchanged_union_hash(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    train = args.public_train_manifest.read_bytes()
    args.public_train_manifest.write_bytes(args.public_val_manifest.read_bytes())
    args.public_val_manifest.write_bytes(train)
    with pytest.raises(ValueError, match="Public train manifest SHA256"):
        verify(args)


def test_exact_duplicates_fail_even_with_distinct_cluster_assignments(tmp_path: Path) -> None:
    args = _fixture(tmp_path, duplicate_sequence=True)
    with pytest.raises(RuntimeError, match="train/val exact sequence overlap"):
        verify(args)


def test_ineligible_hidden_error_does_not_disclose_identity(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    args.hidden_manifest.write_text("9zzz_PRIVATE\n")
    with pytest.raises(ValueError, match="not eligible") as error:
        verify(args)
    assert "9zzz_PRIVATE" not in str(error.value)


@pytest.mark.parametrize("declared_reference", [None, "selected_chains", "public_train_plus_public_val"])
def test_frozen_policy_keeps_its_historical_quality_reference(tmp_path: Path, declared_reference: str | None) -> None:
    args = _fixture(tmp_path)
    public = json.loads(args.lock_file.read_text())
    metadata_path = Path(public["args"]["structure_metadata"])
    metadata = json.loads(metadata_path.read_text())
    metadata["schema_version"] = 1
    for chain in metadata["chains"]:
        chain.update(secondary_structure_class="alpha", secondary_helix_fraction=0.7,
                     secondary_beta_fraction=0.1, secondary_coil_fraction=0.2)
    _write_json(metadata_path, metadata)
    public["sampling_policy"] = "legacy-v1"
    public["structure_metadata"]["sha256"] = _sha(metadata_path)
    _write_json(args.lock_file, public)
    private = json.loads(args.private_lock_file.read_text())
    private.update(sampling_policy="legacy-v1", source_lock_sha256=_sha(args.lock_file))
    private["structure_metadata"]["sha256"] = _sha(metadata_path)
    if declared_reference is None:
        private.pop("stratification")
    else:
        private["stratification"]["reference"] = declared_reference
    _write_json(args.private_lock_file, private)
    result = verify(args)
    assert result["stratification"]["reference_population"] == "selected_chains"
    assert result["stratification"]["fields"][0] == "secondary_structure_class"


@pytest.mark.parametrize("alignment_mode", [None, 3])
def test_clustering_replays_the_policy_identity_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alignment_mode: int | None,
) -> None:
    import subprocess

    from scripts import build_manifests as sampler

    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        Path(command[3] + "_cluster.tsv").write_text("1aaa_A\t1aaa_A\n")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(sampler.shutil, "which", lambda binary: "/fixture/mmseqs")
    monkeypatch.setattr(sampler.subprocess, "run", run)
    assignments, locked_command = sampler._cluster_sequences_mmseqs(
        sequences={"1aaa_A": "A" * 40}, mmseqs_bin="mmseqs", min_seq_id=.3, coverage=.8,
        alignment_mode=alignment_mode,
    )
    assert assignments == {"1aaa_A": "1aaa_A"}
    assert locked_command is not None
    for command in (commands[0], locked_command):
        if alignment_mode is None:
            assert "--alignment-mode" not in command
        else:
            assert command[command.index("--alignment-mode") + 1] == "3"


def _bridge_fixture(tmp_path: Path, *, public_link: str | None = None) -> argparse.Namespace:
    """The unselected B chain shares the hidden PDB and optionally links to train."""
    from scripts.build_manifests import _cluster_assignments_sha256

    args = _fixture(tmp_path)
    cache = json.loads(args.chain_data_cache.read_text())
    cache["3ccc_B"] = cache.pop("4ddd_A")
    if public_link == "sequence":
        cache["3ccc_B"]["sequence"] = cache["1aaa_A"]["sequence"]
    _write_json(args.chain_data_cache, cache)
    public = json.loads(args.lock_file.read_text())
    metadata_path = Path(public["args"]["structure_metadata"])
    metadata = json.loads(metadata_path.read_text())
    metadata["source"]["chain_data_cache_sha256"] = _sha(args.chain_data_cache)
    metadata["chains"][-1]["chain_id"] = "3ccc_B"
    metadata["chains"][-1]["cath_domains"][0]["domain_id"] = "3cccB01"
    _write_json(metadata_path, metadata)
    clusters = {chain: chain for chain in cache}
    if public_link == "cluster":
        clusters["3ccc_B"] = "1aaa_A"
    Path(public["args"]["cluster_tsv"]).write_text("".join(f"{cluster}\t{chain}\n" for chain, cluster in clusters.items()))
    public["chain_data_cache_sha256"] = _sha(args.chain_data_cache)
    public["structure_metadata"]["sha256"] = _sha(metadata_path)
    public["clustering"]["cluster_assignments_sha256"] = _cluster_assignments_sha256(clusters)
    _write_json(args.lock_file, public)
    private = json.loads(args.private_lock_file.read_text())
    private.update(source_lock_sha256=_sha(args.lock_file), chain_data_cache_sha256=_sha(args.chain_data_cache),
                   structure_metadata=public["structure_metadata"], clustering=public["clustering"])
    _write_json(args.private_lock_file, private)
    return args


@pytest.mark.parametrize("public_link", ["cluster", "sequence"])
def test_unselected_bridge_cannot_connect_public_and_hidden(tmp_path: Path, public_link: str) -> None:
    args = _bridge_fixture(tmp_path, public_link=public_link)
    with pytest.raises(RuntimeError, match="transitive cluster/PDB/exact-sequence components") as error:
        verify(args)
    assert "3ccc" not in str(error.value)


def test_single_split_cannot_count_one_connected_unit_twice(tmp_path: Path) -> None:
    from scripts.build_manifests import _cluster_assignments_sha256

    args = _fixture(tmp_path)
    args.public_train_manifest.write_text("1aaa_A\n4ddd_A\n")
    args.public_all_manifest.write_text("1aaa_A\n2bbb_A\n4ddd_A\n")
    public = json.loads(args.lock_file.read_text())
    clusters = {"1aaa_A": "1aaa_A", "2bbb_A": "2bbb_A", "3ccc_A": "3ccc_A", "4ddd_A": "1aaa_A"}
    Path(public["args"]["cluster_tsv"]).write_text("".join(f"{cluster}\t{chain}\n" for chain, cluster in clusters.items()))
    public["clustering"]["cluster_assignments_sha256"] = _cluster_assignments_sha256(clusters)
    public["outputs"].update(train_manifest_sha256=_sha(args.public_train_manifest), train_count=2,
                             all_manifest_sha256=_sha(args.public_all_manifest), unique_count=3)
    _write_json(args.lock_file, public)
    private = json.loads(args.private_lock_file.read_text())
    private.update(source_lock_sha256=_sha(args.lock_file), clustering=public["clustering"])
    private["public_contract"].update(train_manifest_sha256=_sha(args.public_train_manifest), train_count=2,
                                       all_manifest_sha256=_sha(args.public_all_manifest), all_count=3)
    _write_json(args.private_lock_file, private)
    with pytest.raises(RuntimeError, match="multiple representatives of 1 connected components"):
        verify(args)


def _set_exclusions(args: argparse.Namespace, chains: list[str]) -> Path:
    private = json.loads(args.private_lock_file.read_text())
    record = private["filtering"]["private_processability_exclusions"]
    path = Path(record["path"])
    path.write_text("".join(f"{chain}\n" for chain in sorted(chains)))
    record.update(sha256=_sha(path), canonical_ids_sha256=_sha(path))
    private["filtering"]["private_processability_exclusion_count"] = len(chains)
    _write_json(args.private_lock_file, private)
    return path


@pytest.mark.parametrize("excluded", ["3ccc_A", "3ccc_B"])
def test_hidden_components_cannot_touch_private_exclusions(tmp_path: Path, excluded: str) -> None:
    args = _bridge_fixture(tmp_path)
    _set_exclusions(args, [excluded])
    with pytest.raises(RuntimeError, match="privately excluded connected components") as error:
        verify(args)
    assert "3ccc" not in str(error.value)


@pytest.mark.parametrize("change", ["bytes", "missing", "count", "canonical", "record"])
def test_private_exclusion_input_must_match_lock(tmp_path: Path, change: str) -> None:
    args = _fixture(tmp_path)
    path = _set_exclusions(args, ["4ddd_A"])
    private = json.loads(args.private_lock_file.read_text())
    if change == "bytes":
        path.write_text("# changed\n4ddd_A\n")
    elif change == "missing":
        path.unlink()
    elif change == "count":
        private["filtering"]["private_processability_exclusion_count"] = 0
    elif change == "canonical":
        private["filtering"]["private_processability_exclusions"]["canonical_ids_sha256"] = "0" * 64
    else:
        private.pop("filtering")
    _write_json(args.private_lock_file, private)
    with pytest.raises(ValueError, match="[Pp]rivate .*exclusion"):
        verify(args)


def test_private_exclusion_file_can_be_relocated_without_repinning(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    path = _set_exclusions(args, ["4ddd_A"])
    relocated = tmp_path / "relocated.txt"
    path.rename(relocated)
    args.private_processability_exclusion_list = relocated
    assert verify(args)["overlap"]["private_exclusion_components"] == 0


def test_precomputed_cluster_bytes_remain_bound_when_assignments_are_unchanged(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    public = json.loads(args.lock_file.read_text())
    clusters = Path(public["args"]["cluster_tsv"])
    public["clustering"]["cluster_tsv_sha256"] = _sha(clusters)
    _write_json(args.lock_file, public)
    private = json.loads(args.private_lock_file.read_text())
    private["source_lock_sha256"] = _sha(args.lock_file)
    _write_json(args.private_lock_file, private)
    assert verify(args)["status"] == "ok"
    clusters.write_text(clusters.read_text() + "\n")
    with pytest.raises(ValueError, match="TSV.*SHA256"):
        verify(args)
