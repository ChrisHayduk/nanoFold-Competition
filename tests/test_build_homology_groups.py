from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


@dataclass
class Universe:
    cache: Path
    metadata: Path
    clusters: Path
    sequences: dict[str, str]
    cluster_map: dict[str, str]


def _universe(tmp_path: Path, *, ac_only: bool = False) -> Universe:
    if ac_only:
        sequences = {"1aaa_A": "A" * 48, "2bbb_A": "C" * 48, "3ccc_A": "A" * 48}
    else:
        # A--bridge--B must be inseparable even when the bridge is never selected.
        # Exact duplication, a second chain in that PDB, and an old cluster extend it.
        sequences = {
            "1aaa_A": "A" * 40,
            "2bbb_A": "C" * 40,
            "3ccc_A": "D" * 40,
            "4ddd_A": "A" * 40,
            "4ddd_B": "E" * 40,
            "5eee_A": "F" * 40,
            "6fff_A": "G" * 40,
        }
    cache = tmp_path / "chains.json"
    _write_json(cache, {chain: {"sequence": sequence, "seq_length": len(sequence), "resolution": 2.0}
                        for chain, sequence in sequences.items()})
    metadata = tmp_path / "metadata.json"
    _write_json(metadata, {
        "schema_version": 2,
        "source": {"chain_data_cache_sha256": _sha(cache), "chain_id_namespace": "author_asym_id"},
        "config": {"min_len": 40, "max_len": 256, "max_resolution": 3.0, "require_monomer": False,
                   "sample_limit": 0, "max_unknown_aa_fraction": 0.0},
        "summary": {"eligible_candidate_count": len(sequences), "candidate_count": len(sequences)},
        "chains": [{
            "chain_id": chain, "accepted": True, "domain_architecture_class": "alpha",
            "domain_architecture_source": "cath", "metadata_source_count": 1,
            "cath_annotation_status": "annotated", "cath_architecture_ids": ["1.10"],
            "cath_topology_ids": ["1.10.10"], "cath_homologous_superfamily_ids": ["1.10.10.10"],
            "cath_domains": [{"domain_id": chain.replace("_", "") + "01", "class_id": "1",
                              "architecture_id": "1.10", "topology_id": "1.10.10",
                              "homologous_superfamily_id": "1.10.10.10"}],
        } for chain in sequences],
    })
    cluster_map = {chain: chain for chain in sequences}
    if not ac_only:
        cluster_map["5eee_A"] = "4ddd_B"
    clusters = tmp_path / "original-clusters.tsv"
    clusters.write_text("".join(f"{root}\t{chain}\n" for chain, root in cluster_map.items()))
    return Universe(cache, metadata, clusters, sequences, cluster_map)


def _fake_mmseqs(tmp_path: Path) -> Path:
    binary = tmp_path / "fake-mmseqs"
    binary.write_text('''#!/usr/bin/env python3
import pathlib
import sys

args = sys.argv[1:]
if args == ["version"]:
    print("fixture-mmseqs-1")
    raise SystemExit(0)
assert args[0] == "easy-search"
fasta = pathlib.Path(args[1]).read_text().splitlines()
records = {fasta[i + 1]: fasta[i][1:] for i in range(0, len(fasta), 2)}
rows = [f"{header}\\t{header}\\t100\\t{len(sequence)}\\t1\\t1\\t0\\t100"
        for sequence, header in records.items()]
for left, right in [("A" * 40, "D" * 40), ("D" * 40, "C" * 40)]:
    if left in records and right in records:
        rows.append(f"{records[left]}\\t{records[right]}\\t50\\t40\\t1\\t1\\t0.001\\t40")
rows.sort(key=lambda row: row.split("\\t", 1)[0])
pathlib.Path(args[3]).write_text("\\n".join(rows) + "\\n")
print(f"Query database size: {len(records)} type: Aminoacid")
print(f"Target database size: {len(records)} type: Aminoacid")
print(f"{len(records) ** 2} alignments calculated")
print(f"{len(rows)} sequence pairs passed the thresholds")
''')
    binary.chmod(0o700)
    return binary


def _cli(universe: Universe, out: Path, binary: Path | str) -> list[str]:
    return ["--chain-data-cache", str(universe.cache), "--structure-metadata", str(universe.metadata),
            "--cluster-tsv", str(universe.clusters), "--out-dir", str(out), "--mmseqs-bin", str(binary),
            "--threads", "1", "--min-seq-id", "0.3", "--coverage", "0.8"]


def _mapping(path: Path) -> dict[str, str]:
    rows = [line.split("\t") for line in path.read_text().splitlines()]
    assert all(len(row) == 2 for row in rows)
    assert len({member for _, member in rows}) == len(rows)
    return {member: root for root, member in rows}


def _build(tmp_path: Path, *, ac_only: bool = False, binary: Path | str | None = None) -> tuple[Universe, Path]:
    from scripts import build_homology_groups as groups

    universe = _universe(tmp_path, ac_only=ac_only)
    out = tmp_path / "graph"
    assert groups.main(_cli(universe, out, binary or _fake_mmseqs(tmp_path))) == 0
    return universe, out / "split-groups.receipt.json"


def _validate(universe: Universe, receipt_path: Path) -> dict[str, Any]:
    from scripts.build_homology_groups import validate_group_receipt

    receipt = json.loads(receipt_path.read_text())
    return validate_group_receipt(
        receipt_path, split_group_tsv=Path(receipt["split_groups"]["path"]),
        chain_data_cache=universe.cache, structure_metadata=universe.metadata,
        cluster_map=universe.cluster_map, candidate_ids=set(universe.sequences), min_seq_id=0.3, coverage=0.8,
    )


def _artifact(receipt: dict[str, Any], name: str) -> Path:
    return Path(receipt["proof_artifacts"][name]["path"])


def _reseal_artifact(receipt: dict[str, Any], name: str, path: Path) -> None:
    """Keep redundant hashes coherent so mutations exercise the semantic checks."""
    digest = _sha(path)
    receipt["proof_artifacts"][name]["sha256"] = digest
    execution = receipt["execution"]
    if name == "fasta":
        execution["query_fasta"]["sha256"] = execution["target_fasta"]["sha256"] = digest
    elif name == "m8":
        execution["output_m8"]["sha256"] = digest
    else:
        execution[f"{name}_sha256"] = digest


def test_groups_close_all_constraints_through_unselected_bridge(tmp_path: Path) -> None:
    universe, receipt_path = _build(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    mapping = _mapping(Path(receipt["split_groups"]["path"]))
    assert set(mapping) == set(universe.sequences)
    assert len({mapping[chain] for chain in universe.sequences if chain != "6fff_A"}) == 1
    assert mapping["6fff_A"] != mapping["1aaa_A"]
    assert len(set(mapping.values())) == 2
    assert _validate(universe, receipt_path)["original_unit_count"] == 4


def test_generated_search_contract_is_all_pairs_protein_alignment(tmp_path: Path) -> None:
    _, receipt_path = _build(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    argv = json.loads(_artifact(receipt, "command").read_text())["argv"]
    for flag, expected in {
        "--dbtype": "1", "--search-type": "1", "--prefilter-mode": "2", "--alignment-mode": "3",
        "--seq-id-mode": "0", "--cov-mode": "0", "--mask": "0", "--comp-bias-corr": "0",
        "--min-seq-id": "0.3", "-c": "0.8",
        "--format-output": "query,target,pident,alnlen,qcov,tcov,evalue,bits",
    }.items():
        assert argv[argv.index(flag) + 1] == expected
    assert float(argv[argv.index("-e") + 1]) == 1e308
    assert int(argv[argv.index("--max-seqs") + 1]) >= 6


@pytest.mark.parametrize("missing", ["sources", "universe", "proof_artifacts", "counts"])
def test_receipt_requires_complete_evidence(tmp_path: Path, missing: str) -> None:
    universe, receipt_path = _build(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    del receipt[missing]
    _write_json(receipt_path, receipt)
    with pytest.raises(ValueError):
        _validate(universe, receipt_path)


@pytest.mark.parametrize("status", ["running", "error"])
def test_unfinished_graph_cannot_authorize_sampling(tmp_path: Path, status: str) -> None:
    universe, receipt_path = _build(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    receipt["status"] = status
    _write_json(receipt_path, receipt)
    with pytest.raises(ValueError):
        _validate(universe, receipt_path)


@pytest.mark.parametrize("artifact", ["fasta", "m8", "command", "log"])
def test_changed_proof_bytes_are_rejected(tmp_path: Path, artifact: str) -> None:
    universe, receipt_path = _build(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    path = _artifact(receipt, artifact)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        _validate(universe, receipt_path)


@pytest.mark.parametrize(("flag", "value"), [
    ("--dbtype", "2"), ("--prefilter-mode", "0"), ("--alignment-mode", "2"),
    ("--seq-id-mode", "1"), ("--cov-mode", "1"), ("--mask", "1"), ("--comp-bias-corr", "1"),
    ("--max-seqs", "2"), ("-e", "0.001"),
])
def test_resealed_heuristic_or_wrong_alignment_contract_is_rejected(tmp_path: Path, flag: str, value: str) -> None:
    universe, receipt_path = _build(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    path = _artifact(receipt, "command")
    command = json.loads(path.read_text())
    command["argv"][command["argv"].index(flag) + 1] = value
    _write_json(path, command)
    _reseal_artifact(receipt, "command", path)
    _write_json(receipt_path, receipt)
    with pytest.raises(ValueError):
        _validate(universe, receipt_path)


@pytest.mark.parametrize("replacement", ["executable", "query_and_target", "output"])
def test_resealed_command_cannot_name_unrelated_execution_artifacts(tmp_path: Path, replacement: str) -> None:
    universe, receipt_path = _build(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    path = _artifact(receipt, "command")
    command = json.loads(path.read_text())
    unrelated = str(tmp_path / "unrelated-artifact")
    if replacement == "executable":
        command["argv"][0] = unrelated
    elif replacement == "query_and_target":
        command["argv"][2] = command["argv"][3] = unrelated
    else:
        command["argv"][4] = unrelated
    _write_json(path, command)
    _reseal_artifact(receipt, "command", path)
    _write_json(receipt_path, receipt)
    with pytest.raises(ValueError):
        _validate(universe, receipt_path)


@pytest.mark.parametrize("derivation", [None, "unrecognized_producer", "replayed_existing_audit"])
def test_derivation_must_be_known_and_replay_must_have_parent_evidence(tmp_path: Path, derivation: str | None) -> None:
    universe, receipt_path = _build(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    receipt["derivation"] = derivation
    _write_json(receipt_path, receipt)
    with pytest.raises(ValueError):
        _validate(universe, receipt_path)


def test_resealed_map_cannot_separate_a_recorded_homology_edge(tmp_path: Path) -> None:
    from scripts.build_manifests import _cluster_assignments_sha256

    universe, receipt_path = _build(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    path = Path(receipt["split_groups"]["path"])
    mapping = _mapping(path)
    mapping["2bbb_A"] = "2bbb_A"
    path.write_text("".join(f"{root}\t{member}\n" for member, root in sorted(mapping.items())))
    receipt["split_groups"].update(sha256=_sha(path), assignments_sha256=_cluster_assignments_sha256(mapping),
                                   group_count=len(set(mapping.values())))
    _write_json(receipt_path, receipt)
    with pytest.raises(ValueError):
        _validate(universe, receipt_path)


@pytest.mark.parametrize("corruption", ["extra_member", "duplicate_member", "substituted_member"])
def test_split_map_requires_exact_unique_universe(tmp_path: Path, corruption: str) -> None:
    universe, receipt_path = _build(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    path = Path(receipt["split_groups"]["path"])
    rows = path.read_text().splitlines()
    if corruption == "extra_member":
        rows.append("7ggg_A\t7ggg_A")
    elif corruption == "duplicate_member":
        rows.append(rows[0])
    else:
        rows = [row.replace("6fff_A", "7ggg_A") for row in rows]
    path.write_text("\n".join(rows) + "\n")
    receipt["split_groups"]["sha256"] = _sha(path)
    _write_json(receipt_path, receipt)
    with pytest.raises(ValueError):
        _validate(universe, receipt_path)


def test_anonymous_sequence_order_cannot_be_rebound(tmp_path: Path) -> None:
    universe, receipt_path = _build(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    path = _artifact(receipt, "fasta")
    lines = path.read_text().splitlines()
    lines[1], lines[3] = lines[3], lines[1]
    path.write_text("\n".join(lines) + "\n")
    _reseal_artifact(receipt, "fasta", path)
    _write_json(receipt_path, receipt)
    with pytest.raises(ValueError):
        _validate(universe, receipt_path)


def test_receipt_cannot_authorize_a_different_same_size_candidate_universe(tmp_path: Path) -> None:
    from scripts.build_homology_groups import validate_group_receipt

    universe, receipt_path = _build(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    candidate_ids = set(universe.sequences) - {"6fff_A"} | {"7ggg_A"}
    with pytest.raises(ValueError):
        validate_group_receipt(
            receipt_path, split_group_tsv=Path(receipt["split_groups"]["path"]),
            chain_data_cache=universe.cache, structure_metadata=universe.metadata,
            cluster_map=universe.cluster_map, candidate_ids=candidate_ids, min_seq_id=0.3, coverage=0.8,
        )


def test_receipt_cannot_authorize_changed_source_sequences(tmp_path: Path) -> None:
    universe, receipt_path = _build(tmp_path)
    cache = json.loads(universe.cache.read_text())
    cache["6fff_A"]["sequence"] = "H" * 40
    _write_json(universe.cache, cache)
    with pytest.raises(ValueError):
        _validate(universe, receipt_path)


def test_source_relocation_uses_supplied_bytes_and_cluster_assignments(tmp_path: Path) -> None:
    universe, receipt_path = _build(tmp_path)
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    for attribute in ("cache", "metadata"):
        old_path = getattr(universe, attribute)
        new_path = relocated / old_path.name
        shutil.copyfile(old_path, new_path)
        old_path.unlink()
        setattr(universe, attribute, new_path)
    # Callers supply complete original cluster assignments; the old TSV location is not an input.
    universe.clusters.unlink()
    assert _validate(universe, receipt_path)["original_unit_count"] == 4


def test_receipt_is_required_and_metadata_relocation_preserves_only_identical_semantics(tmp_path: Path) -> None:
    from scripts.build_homology_groups import validate_group_receipt

    universe, receipt_path = _build(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    with pytest.raises((FileNotFoundError, ValueError)):
        validate_group_receipt(
            receipt_path.with_name("missing.json"), split_group_tsv=Path(receipt["split_groups"]["path"]),
            chain_data_cache=universe.cache, structure_metadata=universe.metadata,
            cluster_map=universe.cluster_map, candidate_ids=set(universe.sequences), min_seq_id=0.3, coverage=0.8,
        )
    relocated_metadata = tmp_path / "regenerated-metadata.json"
    metadata = json.loads(universe.metadata.read_text())
    metadata["generated_at"] = "2099-01-01T00:00:00Z"
    metadata["source"]["chain_data_cache"] = "/another/location/chains.json"
    _write_json(relocated_metadata, metadata)
    universe.metadata = relocated_metadata
    assert _validate(universe, receipt_path)["original_unit_count"] == 4
    metadata["chains"][0]["accepted"] = False
    _write_json(relocated_metadata, metadata)
    with pytest.raises(ValueError):
        _validate(universe, receipt_path)


def _historical_audit(universe: Universe, receipt_path: Path, out: Path) -> Path:
    """Model the historical audit schema without rewriting its source evidence."""
    receipt = json.loads(receipt_path.read_text())
    out.mkdir()
    filenames = {"fasta": "unique_eligible_sequences.fasta", "m8": "alignments.m8",
                 "command": "command.json", "log": "mmseqs.log"}
    for name, filename in filenames.items():
        shutil.copyfile(_artifact(receipt, name), out / filename)
    command_path = out / "command.json"
    command = json.loads(command_path.read_text())
    command["argv"][2] = command["argv"][3] = str(out / filenames["fasta"])
    command["argv"][4] = str(out / filenames["m8"])
    command["argv"][5] = str(out / "mmseqs-work")
    dbtype_index = command["argv"].index("--dbtype")
    del command["argv"][dbtype_index:dbtype_index + 2]
    _write_json(command_path, command)
    members = [{"header": f"sequence_{index}", "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                "members": sorted(chain for chain, value in universe.sequences.items() if value == sequence)}
               for index, sequence in enumerate(sorted(set(universe.sequences.values())))]
    _write_json(out / "private_sequence_members.json", members)
    inputs = {old: receipt["sources"][new] for old, new in (
        ("chain_data_cache", "chain_data_cache"), ("structure_metadata", "structure_metadata"),
        ("existing_cluster_tsv", "original_clusters"),
    )}
    snapshots = {}
    for old, new in (("runner", "producer"), ("alignment_helper", "shared_parser"),
                     ("sampling_helper", "sampling_helper"), ("target_preflight_runner", "producer")):
        source = (Path(__file__).resolve().parents[1] / "scripts/build_manifests.py"
                  if new == "sampling_helper" else _artifact(receipt, new))
        path = out / f"{old}.py"
        shutil.copyfile(source, path)
        inputs[old] = {"path": str(path), "sha256": _sha(path)}
        snapshots[old] = {"snapshot_path": str(path), "sha256": _sha(path)}
    inputs["mmseqs_binary"] = {"path": command["argv"][0], "sha256": receipt["mmseqs_binary_sha256"]}
    contract = receipt["alignment_contract"].copy()
    del contract["dbtype"]
    legacy = {"schema_version": 1, "status": "complete", "alignment_contract": contract, "inputs": inputs,
              "source_records_sha256": receipt["universe"]["sequence_records_sha256"],
              "unique_sequences": receipt["universe"]["unique_sequence_count"],
              "eligible_chains": len(universe.sequences), "mmseqs_version": receipt["mmseqs_version"],
              "fasta_sha256": _sha(out / filenames["fasta"]), "m8_sha256": _sha(out / filenames["m8"]),
              "command_sha256": _sha(command_path), "mmseqs_log_sha256": _sha(out / filenames["log"]),
              "private_members_sha256": _sha(out / "private_sequence_members.json")}
    summary_path = out / "summary.json"
    _write_json(summary_path, legacy)
    _write_json(out / "bound-source-snapshots.json", {"graph_summary_sha256": _sha(summary_path), "sources": snapshots})
    return summary_path


def test_pinned_historical_audit_can_be_replayed_with_observed_protein_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import build_homology_groups as groups

    universe, receipt_path = _build(tmp_path)
    historical = _historical_audit(universe, receipt_path, tmp_path / "historical")
    historical_sha = _sha(historical)

    def no_subprocess(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Historical audit replay must not execute a new search")

    monkeypatch.setattr(groups.subprocess, "run", no_subprocess)
    monkeypatch.setattr(groups.subprocess, "Popen", no_subprocess)
    out = tmp_path / "replayed"
    args = groups.parse_args([*_cli(universe, out, "__missing_mmseqs__"),
                             "--reuse-audit-dir", str(historical.parent), "--expected-audit-sha256", historical_sha])
    replayed_path = groups.build_groups(args)
    assert _sha(historical) == historical_sha
    replayed = _validate(universe, replayed_path)
    assert replayed["search_evidence"]["protein_type_evidence"] == "observed_aminoacid_databases"
    assert replayed["split_groups"]["group_count"] == 2


@pytest.mark.parametrize("corruption", ["missing_parent", "wrong_pin", "unrelated_parent"])
def test_replayed_receipt_requires_matching_parent_proof_lineage(tmp_path: Path, corruption: str) -> None:
    from scripts import build_homology_groups as groups

    universe, receipt_path = _build(tmp_path)
    out = tmp_path / "replayed"
    replayed_path = groups.build_groups(groups.parse_args([
        *_cli(universe, out, "__missing_mmseqs__"), "--reuse-audit-dir", str(receipt_path.parent),
        "--expected-audit-sha256", _sha(receipt_path),
    ]))
    receipt = json.loads(replayed_path.read_text())
    if corruption == "missing_parent":
        del receipt["proof_artifacts"]["reused_receipt"]
    elif corruption == "wrong_pin":
        receipt["reused_audit_receipt_sha256"] = "0" * 64
    else:
        other_dir = tmp_path / "unrelated-search"
        other_dir.mkdir()
        _, other_receipt = _build(other_dir)
        receipt["proof_artifacts"]["reused_receipt"] = {"path": str(other_receipt), "sha256": _sha(other_receipt)}
        receipt["reused_audit_receipt_sha256"] = _sha(other_receipt)
    _write_json(replayed_path, receipt)
    with pytest.raises(ValueError):
        _validate(universe, replayed_path)


@pytest.mark.parametrize("expected_digest", [None, "0" * 64])
def test_reuse_requires_the_exact_prior_audit_pin(tmp_path: Path, expected_digest: str | None) -> None:
    from scripts import build_homology_groups as groups

    universe, receipt_path = _build(tmp_path)
    argv = [*_cli(universe, tmp_path / "reused", "__missing_mmseqs__"),
            "--reuse-audit-dir", str(receipt_path.parent)]
    if expected_digest is not None:
        argv.extend(["--expected-audit-sha256", expected_digest])
    with pytest.raises(ValueError):
        groups.build_groups(groups.parse_args(argv))


@pytest.mark.parametrize("corruption", ["missing_self_hit", "incomplete_alignment_count"])
def test_incomplete_search_is_rejected_even_with_resealed_artifact(tmp_path: Path, corruption: str) -> None:
    universe, receipt_path = _build(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    artifact = "m8" if corruption == "missing_self_hit" else "log"
    path = _artifact(receipt, artifact)
    lines = path.read_text().splitlines()
    if corruption == "missing_self_hit":
        index = next(i for i, line in enumerate(lines) if line.split("\t")[0] == line.split("\t")[1])
        del lines[index]
    else:
        lines = [line.replace("36 alignments calculated", "35 alignments calculated") for line in lines]
    path.write_text("\n".join(lines) + "\n")
    _reseal_artifact(receipt, artifact, path)
    _write_json(receipt_path, receipt)
    with pytest.raises(ValueError):
        _validate(universe, receipt_path)


def test_real_ac_only_proteins_build_and_reuse_without_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import build_homology_groups as groups

    binary = os.environ.get("MMSEQS_TEST_BINARY") or shutil.which("mmseqs")
    if not binary:
        pytest.skip("Set MMSEQS_TEST_BINARY to run the real protein graph and reuse regression")
    universe, receipt_path = _build(tmp_path, ac_only=True, binary=binary)
    receipt = json.loads(receipt_path.read_text())
    expected = _mapping(Path(receipt["split_groups"]["path"]))
    assert expected["1aaa_A"] == expected["3ccc_A"]
    assert expected["1aaa_A"] != expected["2bbb_A"]

    def no_subprocess(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Reusing a completed graph must not execute MMseqs or another subprocess")

    monkeypatch.setattr(groups.subprocess, "run", no_subprocess)
    monkeypatch.setattr(groups.subprocess, "Popen", no_subprocess)
    out = tmp_path / "reused"
    assert groups.main([*_cli(universe, out, "__missing_mmseqs__"), "--reuse-audit-dir", str(receipt_path.parent),
                        "--expected-audit-sha256", _sha(receipt_path)]) == 0
    reused_receipt = out / "split-groups.receipt.json"
    reused = json.loads(reused_receipt.read_text())
    assert _mapping(Path(reused["split_groups"]["path"])) == expected
    assert _validate(universe, reused_receipt)["original_unit_count"] == 2


def test_real_search_length_screening_matches_float32_coverage_boundary(tmp_path: Path) -> None:
    from scripts import build_homology_groups as groups

    binary = os.environ.get("MMSEQS_TEST_BINARY") or shutil.which("mmseqs")
    if not binary:
        pytest.skip("Set MMSEQS_TEST_BINARY to run the real coverage-boundary regression")
    universe = _universe(tmp_path, ac_only=True)
    universe.sequences = {"1aaa_A": "A" * 55, "2bbb_A": "A" * 100, "3ccc_A": "A" * 55}
    _write_json(universe.cache, {
        chain: {"sequence": sequence, "seq_length": len(sequence), "resolution": 2.0}
        for chain, sequence in universe.sequences.items()
    })
    metadata = json.loads(universe.metadata.read_text())
    metadata["source"]["chain_data_cache_sha256"] = _sha(universe.cache)
    _write_json(universe.metadata, metadata)
    out = tmp_path / "coverage-boundary"
    argv = _cli(universe, out, binary)
    argv[argv.index("--coverage") + 1] = "0.55"
    receipt_path = groups.build_groups(groups.parse_args(argv))
    receipt = json.loads(receipt_path.read_text())
    assert receipt["search_evidence"]["alignments_calculated"] == 4
    assert receipt["counts"]["qualifying_ordered_alignments_including_self"] == 4
    assert receipt["split_groups"]["group_count"] == 1
