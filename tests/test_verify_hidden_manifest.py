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
        "args": {"sampling_policy": "cath-v2", "train_size": 1, "val_size": 1,
                 "require_monomer": False, "stratify_fields": ["cath_architecture_ids", "length_bin", "resolution_bin"],
                 "hidden_val_size": 1, "min_len": 40, "max_len": 256, "max_resolution": 3.0,
                 "seed": 37, "min_seq_id": .3, "coverage": .8, "mmseqs_bin": "unused",
                 "cluster_tsv": str(clusters), "structure_metadata": str(metadata)},
        "filtering": {"candidate_count": 4, "disjoint_unit_count": 4, "target_policy": "single_chain",
                      "representative_policy": "best_resolution_then_longest_then_release_date_seeded_tie"},
        "grouping": {"cluster_disjoint": True, "pdb_disjoint": True, "exact_sequence_disjoint": True},
        "outputs": {"train_manifest_sha256": _sha(paths["train"]), "train_count": 1,
                    "val_manifest_sha256": _sha(paths["val"]), "val_count": 1,
                    "all_manifest_sha256": _sha(paths["all"]), "unique_count": 2},
    })
    private_lock = tmp_path / "private.lock.json"
    exclusions = tmp_path / "private-exclusions.txt"
    exclusions.write_text("")
    _write_json(private_lock, {
        "sampling_policy": "cath-v2", "source_lock_sha256": _sha(lock),
        "hidden_split_salt_sha256": hashlib.sha256(b"test-private-salt").hexdigest(),
        "chain_data_cache_sha256": _sha(cache), "hidden_val_manifest_sha256": _sha(paths["hidden"]),
        "hidden_val_count": 1, "structure_metadata": {"sha256": _sha(metadata)}, "clustering": clustering,
        "public_contract": {**{f"{name}_manifest_sha256": _sha(paths[name]) for name in ("train", "val", "all")},
                            "train_count": 1, "val_count": 1, "all_count": 2},
        "stratification": {"reference": "public_train_plus_public_val",
                           "fields": ["cath_architecture_ids", "length_bin", "resolution_bin"]},
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
    public["args"]["sampling_policy"] = "legacy-v1"
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


def _repin_public_lock(args: argparse.Namespace, public: dict) -> None:
    _write_json(args.lock_file, public)
    private = json.loads(args.private_lock_file.read_text())
    private["source_lock_sha256"] = _sha(args.lock_file)
    _write_json(args.private_lock_file, private)


@pytest.mark.parametrize("field", ["train_size", "val_size", "sampling_policy", "require_monomer", "stratify_fields"])
def test_cath_source_contract_cannot_drop_sampling_configuration(tmp_path: Path, field: str) -> None:
    args = _fixture(tmp_path)
    public = json.loads(args.lock_file.read_text())
    del public["args"][field]
    _repin_public_lock(args, public)
    with pytest.raises(ValueError, match=f"missing args.{field}"):
        verify(args)


@pytest.mark.parametrize("field,value", [("train_size", 999), ("val_size", 2), ("seed", 7),
                                         ("stratify_fields", ["length_bin"])])
def test_sampling_contract_edits_cannot_relabel_existing_public_splits(
    tmp_path: Path, field: str, value: object,
) -> None:
    args = _fixture(tmp_path)
    public = json.loads(args.lock_file.read_text())
    public["args"][field] = value
    _repin_public_lock(args, public)
    with pytest.raises(ValueError, match="locked sampling sizes|do not reproduce|stratify_fields"):
        verify(args)


def test_wrong_component_representative_cannot_pass_verification(tmp_path: Path) -> None:
    from scripts.build_manifests import _cluster_assignments_sha256

    args = _fixture(tmp_path)
    cache = json.loads(args.chain_data_cache.read_text())
    cache["4ddd_A"]["resolution"] = 1.0
    _write_json(args.chain_data_cache, cache)
    public = json.loads(args.lock_file.read_text())
    metadata_path = Path(public["args"]["structure_metadata"])
    metadata = json.loads(metadata_path.read_text())
    metadata["source"]["chain_data_cache_sha256"] = _sha(args.chain_data_cache)
    _write_json(metadata_path, metadata)
    clusters = {"1aaa_A": "1aaa_A", "2bbb_A": "2bbb_A", "3ccc_A": "3ccc_A", "4ddd_A": "1aaa_A"}
    Path(public["args"]["cluster_tsv"]).write_text("".join(f"{cluster}\t{chain}\n" for chain, cluster in clusters.items()))
    public["chain_data_cache_sha256"] = _sha(args.chain_data_cache)
    public["structure_metadata"]["sha256"] = _sha(metadata_path)
    public["clustering"]["cluster_assignments_sha256"] = _cluster_assignments_sha256(clusters)
    public["filtering"]["disjoint_unit_count"] = 3
    _repin_public_lock(args, public)
    private = json.loads(args.private_lock_file.read_text())
    private.update(chain_data_cache_sha256=_sha(args.chain_data_cache), structure_metadata=public["structure_metadata"],
                   clustering=public["clustering"])
    _write_json(args.private_lock_file, private)
    with pytest.raises(ValueError, match="representative, and sampling policy"):
        verify(args)


@pytest.mark.parametrize("digest", [None, "", "invalid"])
def test_hidden_selection_requires_and_replays_its_salt_digest(tmp_path: Path, digest: str | None) -> None:
    args = _fixture(tmp_path)
    private = json.loads(args.private_lock_file.read_text())
    private["hidden_split_salt_sha256"] = digest
    _write_json(args.private_lock_file, private)
    with pytest.raises(ValueError, match="salt"):
        verify(args)


def test_repinning_hidden_manifest_hash_does_not_bypass_selection_replay(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    args.hidden_manifest.write_text("4ddd_A\n")
    private = json.loads(args.private_lock_file.read_text())
    private["hidden_val_manifest_sha256"] = _sha(args.hidden_manifest)
    _write_json(args.private_lock_file, private)
    with pytest.raises(ValueError, match="does not reproduce the locked salt"):
        verify(args)


@pytest.mark.parametrize("field", ["chain_data_cache", "lock_file", "private_lock_file", "hidden_manifest",
                                   "public_train_manifest", "public_val_manifest", "public_all_manifest"])
def test_verification_report_cannot_replace_audited_inputs(tmp_path: Path, field: str) -> None:
    args = _fixture(tmp_path)
    path = getattr(args, field)
    before = path.read_bytes()
    args.json_out = path
    with pytest.raises(ValueError, match="must not alias"):
        verify(args)
    assert path.read_bytes() == before


@pytest.mark.parametrize("link_type", ["symlink", "hardlink"])
def test_verification_report_rejects_link_aliases(tmp_path: Path, link_type: str) -> None:
    args = _fixture(tmp_path)
    args.json_out = tmp_path / "report.json"
    if link_type == "symlink":
        args.json_out.symlink_to(args.hidden_manifest)
    else:
        args.json_out.hardlink_to(args.hidden_manifest)
    with pytest.raises(ValueError, match="must not alias"):
        verify(args)


def test_supplied_missing_private_exclusions_fail_before_hidden_generation(tmp_path: Path) -> None:
    from scripts.build_hidden_manifest import _build_hidden_manifest

    args = _fixture(tmp_path)
    args.hidden_val_size = None
    args.seed = None
    args.private_processability_exclusion_list = tmp_path / "typo-exclusions.txt"
    args.hidden_out_dir = tmp_path / "new-hidden"
    args.private_lock_file = tmp_path / "new-private.lock.json"
    with pytest.raises(ValueError, match="existing file"):
        _build_hidden_manifest(args)
    assert not args.hidden_out_dir.exists()
    assert not args.private_lock_file.exists()


@pytest.mark.parametrize("chain_id", ["1AAA_A", "1aaa_A_B", " 1aaa_A", "bad"])
def test_noncanonical_candidate_names_cannot_create_disjoint_aliases(tmp_path: Path, chain_id: str) -> None:
    from scripts.build_manifests import _build_candidates

    with pytest.raises(ValueError, match="Chain IDs"):
        _build_candidates({chain_id: {}}, min_len=40, max_len=256, max_resolution=3,
                          structure_metadata={}, structure_metadata_accepted=set())


def test_private_exclusion_chain_names_are_not_silently_ignored(tmp_path: Path) -> None:
    from scripts.build_hidden_manifest import _read_optional_chain_list

    path = tmp_path / "exclusions.txt"
    path.write_text("1AAA_A\n")
    with pytest.raises(ValueError, match="Chain IDs"):
        _read_optional_chain_list(path)


@pytest.mark.parametrize("field,value", [("seed", 7), ("train_size", 999)])
def test_hidden_builder_cannot_repin_a_different_public_sampling_contract(
    tmp_path: Path, field: str, value: int,
) -> None:
    from scripts.build_hidden_manifest import _build_hidden_manifest

    args = _fixture(tmp_path)
    public = json.loads(args.lock_file.read_text())
    public["args"][field] = value
    _write_json(args.lock_file, public)
    args.hidden_val_size = None
    args.seed = None
    args.private_processability_exclusion_list = None
    args.hidden_out_dir = tmp_path / "new-hidden"
    args.private_lock_file = tmp_path / "new-private.lock.json"
    with pytest.raises(ValueError, match="locked sampling sizes|do not reproduce"):
        _build_hidden_manifest(args)
    assert not args.hidden_out_dir.exists()
    assert not args.private_lock_file.exists()


@pytest.mark.parametrize("digest", [None, "", "INVALID"])
def test_public_cache_digest_is_required_for_cath_v2(tmp_path: Path, digest: str | None) -> None:
    args = _fixture(tmp_path)
    public = json.loads(args.lock_file.read_text())
    public["chain_data_cache_sha256"] = digest
    _repin_public_lock(args, public)
    with pytest.raises(ValueError, match="chain_data_cache_sha256 is missing or invalid"):
        verify(args)
