"""Build complete, source-bound protein homology split groups before sampling."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from audit_cache_targets import _reject_json_constant, _unique_json_object, sequences_for, sha256, write_json

from nanofold.mmseqs import iter_mmseqs_hits, sequence_records_sha256, validate_search_parameters
from nanofold.structure_metadata import validate_author_chain_id

ARTIFACT_TYPE = "nanofold_full_universe_homology_groups"
FORMAT = "query,target,pident,alnlen,qcov,tcov,evalue,bits"
SCOPE = (
    "Complete eligible-universe closure of returned all-pair protein local alignments, exact sequences, "
    "PDB entries and original assigned clusters. Both directed orientations are submitted; MMseqs performs "
    "its bilateral coverage length screening. This does not prove absence of every possible alignment or "
    "biological homology. Groups constrain split placement; original de-redundant representatives remain "
    "the selection population. Processed feature, MSA, label and tensor audits are separate."
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(), object_pairs_hook=_unique_json_object, parse_constant=_reject_json_constant)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def verify_record(value: Any) -> Path:
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise ValueError("Missing bound artifact record")
    path = Path(value["path"])
    if not path.is_file() or sha256(path) != value.get("sha256"):
        raise ValueError("A bound graph artifact is missing or changed")
    return path


def assignments_sha256(mapping: dict[str, str]) -> str:
    return hashlib.sha256("".join(f"{member}\t{root}\n" for member, root in sorted(mapping.items())).encode()).hexdigest()


def ids_sha256(ids: set[str]) -> str:
    return hashlib.sha256("".join(chain + "\n" for chain in sorted(ids)).encode()).hexdigest()


def metadata_semantic_sha256(path: Path) -> str:
    """Ignore only generation time and explicitly identified source locations."""
    value = read_json(path)
    value.pop("generated_at", None)
    source = value.get("source", {})
    for key in ("chain_data_cache", "metadata_sources_dir"):
        source.pop(key, None)
    for key in ("feature_exclusion_list", "processability_exclusion_list"):
        if isinstance(source.get(key), dict):
            source[key].pop("path", None)
    lock = source.get("metadata_source_lock", {})
    lock.pop("lock_path", None)
    for item in lock.get("sources", {}).values():
        item.pop("path", None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_map(path: Path, ids: set[str], *, canonical: bool = False) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        columns = line.split("\t")
        if len(columns) != 2:
            raise ValueError("Group TSV must contain representative and member columns")
        root, member = columns
        validate_author_chain_id(root)
        validate_author_chain_id(member)
        if member in mapping:
            raise ValueError("Group TSV contains duplicate members")
        mapping[member] = root
    if canonical and set(mapping) != ids:
        raise ValueError("Group TSV does not cover exactly the eligible universe")
    mapping = {chain: root for chain, root in mapping.items() if chain in ids}
    if set(mapping) != ids:
        raise ValueError("Cluster TSV does not cover the complete eligible universe")
    groups: dict[str, list[str]] = defaultdict(list)
    for chain, root in mapping.items():
        groups[root].append(chain)
    for root, members in groups.items():
        if root in mapping and mapping[root] != root:
            raise ValueError("A cluster representative belongs to another cluster")
        if canonical and root != min(members):
            raise ValueError("Group roots must be canonical member chains")
    return mapping


def load_universe(cache_path: Path, metadata_path: Path) -> dict[str, str]:
    # Import only inside construction to keep the receipt validator safe for sampler imports.
    from build_manifests import _build_candidates, _load_structure_metadata, _validate_metadata_identity

    cache = read_json(cache_path)
    metadata = read_json(metadata_path)
    config = metadata["config"]
    policy = argparse.Namespace(**{name: config[name] for name in ("min_len", "max_len", "max_resolution", "require_monomer")})
    _validate_metadata_identity(metadata_path, sha256(cache_path), policy)
    accepted, rows = _load_structure_metadata(metadata_path, require_schema_version=2)
    candidates, _, _ = _build_candidates(
        cache, min_len=policy.min_len, max_len=policy.max_len, max_resolution=policy.max_resolution,
        structure_metadata=rows, structure_metadata_accepted=accepted,
        require_monomer=policy.require_monomer, sampling_policy="cath-v2",
    )
    ids = sorted(candidate.chain_id for candidate in candidates)
    if not ids:
        raise ValueError("Eligible sequence universe is empty")
    sequences = sequences_for(ids, cache, min_len=policy.min_len, max_len=policy.max_len)
    return dict(zip(ids, sequences, strict=True))


def ordered_records(sequences: dict[str, str]) -> list[tuple[str, str]]:
    return [(f"sequence_{index}", sequence) for index, sequence in enumerate(sorted(set(sequences.values())))]


def fasta_text(records: list[tuple[str, str]]) -> str:
    return "".join(f">{header}\n{sequence}\n" for header, sequence in records)


class Components:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))
        self.size = [1] * count

    def find(self, node: int) -> int:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]


def close_graph(
    sequences: dict[str, str], cluster_map: dict[str, str], m8: Path, *, min_seq_id: float, coverage: float,
) -> tuple[dict[str, str], dict[str, int]]:
    records = ordered_records(sequences)
    indices = {sequence: index for index, (_, sequence) in enumerate(records)}
    header_index = {header: index for index, (header, _) in enumerate(records)}
    chain_index = {chain: indices[sequence] for chain, sequence in sequences.items()}
    graph, original = Components(len(records)), Components(len(records))
    self_hits: set[str] = set()
    finished_queries: set[str] = set()
    last_query: str | None = None
    targets: set[str] = set()
    hit_count = 0
    for query, target in iter_mmseqs_hits(
        m8, query_ids=set(header_index), target_ids=set(header_index), min_seq_id=min_seq_id, coverage=coverage,
    ):
        if query != last_query:
            if query in finished_queries:
                raise ValueError("MMseqs output repeats a completed query block")
            if last_query is not None:
                finished_queries.add(last_query)
            targets.clear()
            last_query = query
        if target in targets:
            raise ValueError("MMseqs output contains duplicate query/target alignments")
        targets.add(target)
        hit_count += 1
        graph.union(header_index[query], header_index[target])
        if query == target:
            self_hits.add(query)
    if len(self_hits) != len(records):
        raise ValueError("Complete graph search is missing self alignments")

    def count(components: Components) -> int:
        return len({components.find(index) for index in range(len(records))})

    counts = {"homology_exact": count(graph), "qualifying_ordered_alignments_including_self": hit_count,
              "self_alignment_sequences": len(self_hits)}
    pdbs: dict[str, list[str]] = defaultdict(list)
    clusters: dict[str, list[str]] = defaultdict(list)
    for chain in sequences:
        pdbs[chain.split("_", 1)[0].lower()].append(chain)
        clusters[cluster_map[chain]].append(chain)
    for label, groups in (("homology_exact_pdb", pdbs), ("homology_exact_pdb_existing_clusters", clusters)):
        for members in groups.values():
            for member in members[1:]:
                graph.union(chain_index[members[0]], chain_index[member])
                original.union(chain_index[members[0]], chain_index[member])
        counts[label] = count(graph)
    counts["original_unit_count"] = count(original)
    roots: dict[int, str] = {}
    for chain in sorted(sequences):
        roots.setdefault(graph.find(chain_index[chain]), chain)
    return {chain: roots[graph.find(index)] for chain, index in chain_index.items()}, counts


def search_contract(min_seq_id: float, coverage: float) -> dict[str, Any]:
    return {"min_seq_id": min_seq_id, "coverage": coverage, "cov_mode": 0, "alignment_mode": 3,
            "seq_id_mode": 0, "search_type": 1, "dbtype": 1, "prefilter_mode": 2, "evalue": 1e308,
            "sensitivity": 5.7, "mask": 0, "comp_bias_corr": 0}


def execution_record(binary: dict[str, str], artifacts: dict[str, dict[str, str]]) -> dict[str, Any]:
    return {"binary": binary, "query_fasta": artifacts["fasta"], "target_fasta": artifacts["fasta"],
            "output_m8": artifacts["m8"], **{f"{key}_sha256": artifacts[key]["sha256"]
                                             for key in ("command", "log", "producer", "shared_parser")}}


def verify_execution(receipt: dict[str, Any], artifacts: dict[str, Path], seen: set[str] | None = None) -> None:
    execution = receipt.get("execution")
    if not isinstance(execution, dict) or not all(isinstance(execution.get(key), dict)
                                                for key in ("binary", "query_fasta", "target_fasta", "output_m8")):
        raise ValueError("Graph receipt lacks recorded execution identities")
    command = read_json(artifacts["command"]).get("argv")
    if not isinstance(command, list) or len(command) < 6:
        raise ValueError("Graph receipt command is invalid")
    for key, index, proof_key in (("binary", 0, None), ("query_fasta", 2, "fasta"),
                                  ("target_fasta", 3, "fasta"), ("output_m8", 4, "m8")):
        expected_sha = receipt["mmseqs_binary_sha256"] if proof_key is None else receipt["proof_artifacts"][proof_key]["sha256"]
        if command[index] != execution[key].get("path") or execution[key].get("sha256") != expected_sha:
            raise ValueError("MMseqs command identities differ from the recorded execution")
    for key in ("command", "log", "producer", "shared_parser"):
        if execution.get(f"{key}_sha256") != receipt["proof_artifacts"][key]["sha256"]:
            raise ValueError("Search execution evidence hashes differ")
    derivation = receipt.get("derivation")
    if derivation == "new_mmseqs_search":
        if "reused_audit_receipt_sha256" in receipt or "reused_receipt" in artifacts:
            raise ValueError("Fresh search receipt cannot claim reused ancestry")
        return
    if derivation != "replayed_existing_audit" or "reused_receipt" not in artifacts:
        raise ValueError("Reused graph requires a pinned parent audit receipt")
    pin = receipt.get("reused_audit_receipt_sha256")
    if not isinstance(pin, str) or pin != receipt["proof_artifacts"]["reused_receipt"]["sha256"]:
        raise ValueError("Reused graph parent receipt pin differs")
    visited = set() if seen is None else seen
    if pin in visited:
        raise ValueError("Graph receipt ancestry contains a cycle")
    visited.add(pin)
    parent = read_json(artifacts["reused_receipt"])
    if parent.get("status") != "complete" or parent.get("schema_version") != 1:
        raise ValueError("Reused graph parent is incomplete")
    if parent.get("artifact_type") == ARTIFACT_TYPE:
        if parent.get("execution") != execution or parent.get("alignment_contract") != receipt["alignment_contract"]:
            raise ValueError("Reused graph execution differs from its parent audit")
        if parent.get("universe") != receipt["universe"]:
            raise ValueError("Reused graph parent has a different sequence universe")
        parent_paths = {name: verify_record(item) for name, item in parent["proof_artifacts"].items()}
        for key in ("fasta", "m8", "command", "log", "producer", "shared_parser"):
            if parent["proof_artifacts"][key]["sha256"] != receipt["proof_artifacts"][key]["sha256"]:
                raise ValueError("Reused search evidence differs from its parent audit")
        verify_execution(parent, parent_paths, visited)
    else:
        contract = dict(receipt["alignment_contract"])
        contract.pop("dbtype")
        if parent.get("alignment_contract") != contract or parent.get("source_records_sha256") != receipt["universe"]["sequence_records_sha256"]:
            raise ValueError("Reused historical audit contract or sequence universe differs")
        for key, parent_key in (("fasta", "fasta_sha256"), ("m8", "m8_sha256"), ("command", "command_sha256"), ("log", "mmseqs_log_sha256")):
            if parent.get(parent_key) != receipt["proof_artifacts"][key]["sha256"]:
                raise ValueError("Reused historical search artifacts differ")
        for key, parent_key in (("producer", "runner"), ("shared_parser", "alignment_helper")):
            if parent["inputs"][parent_key]["sha256"] != receipt["proof_artifacts"][key]["sha256"]:
                raise ValueError("Reused historical producer sources differ")
        if parent["inputs"]["mmseqs_binary"] != execution["binary"] or parent.get("mmseqs_version") != receipt["mmseqs_version"]:
            raise ValueError("Reused historical executable provenance differs")


def verify_search_evidence(
    command_path: Path, log_path: Path, records: list[tuple[str, str]], counts: dict[str, int],
    *, min_seq_id: float, coverage: float, allow_observed_protein: bool = False,
) -> dict[str, Any]:
    command = read_json(command_path)["argv"]
    if not isinstance(command, list) or len(command) < 6 or not all(isinstance(arg, str) for arg in command) or command[1] != "easy-search":
        raise ValueError("Invalid MMseqs search command")
    if len(command) < 6 or command[2] != command[3]:
        raise ValueError("Complete graph requires the identical full query and target FASTA")
    options = command[6:]
    if len(options) % 2 or len(set(options[::2])) != len(options[::2]):
        raise ValueError("Ambiguous MMseqs command options")
    flags = dict(zip(options[::2], options[1::2], strict=True))
    expected = {"--min-seq-id": min_seq_id, "-c": coverage, "--cov-mode": 0, "--alignment-mode": 3,
                "--seq-id-mode": 0, "--search-type": 1, "--prefilter-mode": 2, "-e": 1e308,
                "-s": 5.7, "--mask": 0, "--comp-bias-corr": 0}
    if not allow_observed_protein or "--dbtype" in flags:
        expected["--dbtype"] = 1
    if set(flags) != set(expected) | {"--threads", "--max-seqs", "--format-output"}:
        raise ValueError("MMseqs command has missing or unrecognized search options")
    if any(float(flags[key]) != value for key, value in expected.items()) or flags["--format-output"] != FORMAT:
        raise ValueError("MMseqs command disagrees with the declared all-pair alignment contract")
    if int(flags["--threads"]) < 1 or int(flags["--max-seqs"]) < len(records):
        raise ValueError("MMseqs search truncates the complete eligible sequence population")
    log = log_path.read_text()
    for role in ("Query", "Target"):
        matches = re.findall(rf"{role} database size: (\d+) type: (\w+)", log)
        if matches != [(str(len(records)), "Aminoacid")]:
            raise ValueError("MMseqs log does not prove the complete protein database population")
    lengths = Counter(len(sequence) for _, sequence in records)

    def float32(value: float) -> float:
        return struct.unpack("f", struct.pack("f", value))[0]

    # MMseqs canBeCovered compares float32 length ratios, not double products.
    compatible = sum(left_count * right_count for left, left_count in lengths.items()
                     for right, right_count in lengths.items()
                     if min(float32(left / right), float32(right / left)) >= float32(coverage))
    if re.findall(r"^(\d+) alignments calculated$", log, re.MULTILINE) != [str(compatible)]:
        raise ValueError("MMseqs alignment count does not cover all length-compatible ordered pairs")
    if re.findall(r"^(\d+) sequence pairs passed the thresholds", log, re.MULTILINE) != [str(counts["qualifying_ordered_alignments_including_self"])]:
        raise ValueError("MMseqs output row count disagrees with the completed search log")
    return {"threads": int(flags["--threads"]), "max_seqs": int(flags["--max-seqs"]),
            "protein_type_evidence": "explicit_dbtype_1" if "--dbtype" in flags else "observed_aminoacid_databases",
            "requested_ordered_pairs": len(records) ** 2, "alignments_calculated": compatible}


def validate_group_receipt(
    receipt_path: Path, *, split_group_tsv: Path, chain_data_cache: Path, structure_metadata: Path,
    cluster_map: dict[str, str], candidate_ids: set[str], min_seq_id: float, coverage: float,
) -> dict[str, Any]:
    receipt_digest = sha256(Path(receipt_path))
    selection_inputs = {Path(path): sha256(Path(path)) for path in (chain_data_cache, structure_metadata, split_group_tsv)}
    receipt = read_json(Path(receipt_path))
    if receipt.get("schema_version") != 1 or receipt.get("artifact_type") != ARTIFACT_TYPE or receipt.get("status") != "complete":
        raise ValueError("Missing or incomplete homology group receipt")
    required_objects = {
        "sources": {"chain_data_cache", "structure_metadata", "original_clusters"},
        "proof_artifacts": {"fasta", "m8", "command", "log", "producer", "shared_parser"},
        "split_groups": {"path", "sha256", "assignments_sha256", "group_count"},
        "universe": {"chain_count", "chain_ids_sha256", "unique_sequence_count", "sequence_records_sha256"},
        "counts": {"original_unit_count", "self_alignment_sequences", "qualifying_ordered_alignments_including_self"},
    }
    if any(not isinstance(receipt.get(key), dict) or not names <= set(receipt[key])
           for key, names in required_objects.items()):
        raise ValueError("Graph receipt lacks required provenance objects")
    if receipt.get("alignment_contract") != search_contract(min_seq_id, coverage):
        raise ValueError("Graph receipt alignment thresholds or protein contract differ")
    sources = receipt["sources"]
    if sha256(Path(chain_data_cache)) != sources["chain_data_cache"]["sha256"]:
        raise ValueError("Graph receipt chain cache differs from selection")
    if metadata_semantic_sha256(Path(structure_metadata)) != sources["structure_metadata"]["semantic_sha256"]:
        raise ValueError("Graph receipt metadata semantics differ from selection")
    if assignments_sha256(cluster_map) != sources["original_clusters"]["assignments_sha256"]:
        raise ValueError("Graph receipt original cluster assignments differ from selection")
    sequences = load_universe(Path(chain_data_cache), Path(structure_metadata))
    if set(sequences) != set(candidate_ids) or set(cluster_map) != set(candidate_ids):
        raise ValueError("Graph receipt universe differs from selection")
    records = ordered_records(sequences)
    expected_universe = {"chain_count": len(sequences), "chain_ids_sha256": ids_sha256(set(sequences)),
                         "unique_sequence_count": len(records), "sequence_records_sha256": sequence_records_sha256(records)}
    if receipt.get("universe") != expected_universe:
        raise ValueError("Graph receipt does not bind the complete ordered sequence universe")
    artifacts = {name: verify_record(item) for name, item in receipt["proof_artifacts"].items()}
    if not {"fasta", "m8", "command", "log", "producer", "shared_parser"} <= set(artifacts):
        raise ValueError("Graph receipt lacks required search provenance artifacts")
    if artifacts["fasta"].read_text() != fasta_text(records):
        raise ValueError("Graph FASTA order or sequence content differs from the eligible universe")
    verify_execution(receipt, artifacts)
    expected_map, counts = close_graph(sequences, cluster_map, artifacts["m8"], min_seq_id=min_seq_id, coverage=coverage)
    evidence = verify_search_evidence(
        artifacts["command"], artifacts["log"], records, counts, min_seq_id=min_seq_id, coverage=coverage,
        allow_observed_protein=receipt.get("derivation") == "replayed_existing_audit",
    )
    if receipt.get("search_evidence") != evidence or receipt.get("counts") != counts or receipt.get("original_unit_count") != counts["original_unit_count"]:
        raise ValueError("Graph receipt counts disagree with reconstructed search evidence")
    group_record = receipt["split_groups"]
    group_map = load_map(Path(split_group_tsv), set(sequences), canonical=True)
    if sha256(Path(split_group_tsv)) != group_record["sha256"] or group_map != expected_map:
        raise ValueError("Split groups differ from the complete recorded graph closure")
    if group_record.get("assignments_sha256") != assignments_sha256(group_map) or group_record.get("group_count") != len(set(group_map.values())):
        raise ValueError("Group receipt assignment digest or count differs")
    if not isinstance(receipt.get("mmseqs_version"), str) or not receipt["mmseqs_version"] or not re.fullmatch(r"[0-9a-f]{64}", receipt.get("mmseqs_binary_sha256", "")):
        raise ValueError("Graph receipt lacks MMseqs executable provenance")
    for item in receipt["proof_artifacts"].values():
        verify_record(item)
    if sha256(Path(receipt_path)) != receipt_digest or any(sha256(path) != digest for path, digest in selection_inputs.items()):
        raise ValueError("Graph evidence or selection inputs changed during validation")
    return receipt


def import_audit(
    audit: Path, pin: str, out: Path, sequences: dict[str, str], sources: dict[str, Any],
    *, min_seq_id: float, coverage: float,
) -> dict[str, Any]:
    receipt_path = audit / "split-groups.receipt.json"
    if not receipt_path.exists():
        receipt_path = audit / "summary.json"
    if not re.fullmatch(r"[0-9a-f]{64}", pin) or sha256(receipt_path) != pin:
        raise ValueError("Reused audit receipt does not match the explicitly pinned digest")
    previous = read_json(receipt_path)
    if previous.get("status") != "complete":
        raise ValueError("Cannot reuse an incomplete graph audit")
    records = ordered_records(sequences)
    legacy = previous.get("artifact_type") != ARTIFACT_TYPE
    if legacy:
        contract = search_contract(min_seq_id, coverage)
        contract.pop("dbtype")
        if previous.get("schema_version") != 1 or previous.get("alignment_contract") != contract:
            raise ValueError("Existing audit contract differs from the requested graph")
        for current, old in (("chain_data_cache", "chain_data_cache"), ("structure_metadata", "structure_metadata"),
                             ("original_clusters", "existing_cluster_tsv")):
            old_record = previous["inputs"][old]
            if old_record["sha256"] == sources[current]["sha256"]:
                matches = True
            elif current == "structure_metadata":
                path = verify_record(old_record)
                matches = metadata_semantic_sha256(path) == sources[current]["semantic_sha256"]
            else:
                matches = False
            if not matches:
                raise ValueError("Existing audit inputs differ from the requested universe")
        if previous.get("source_records_sha256") != sequence_records_sha256(records) or previous.get("unique_sequences") != len(records) or previous.get("eligible_chains") != len(sequences):
            raise ValueError("Existing audit sequence universe differs")
        paths = {"fasta": audit / "unique_eligible_sequences.fasta", "m8": audit / "alignments.m8",
                 "command": audit / "command.json", "log": audit / "mmseqs.log"}
        for name, key in (("fasta", "fasta_sha256"), ("m8", "m8_sha256"), ("command", "command_sha256"), ("log", "mmseqs_log_sha256")):
            if sha256(paths[name]) != previous.get(key):
                raise ValueError("Existing audit artifact changed")
        members_path = audit / "private_sequence_members.json"
        if sha256(members_path) != previous.get("private_members_sha256"):
            raise ValueError("Existing audit private sequence mapping changed")
        expected_members: dict[str, list[str]] = defaultdict(list)
        for chain, sequence in sorted(sequences.items()):
            expected_members[sequence].append(chain)
        expected = [{"header": header, "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                     "members": expected_members[sequence]} for header, sequence in records]
        if json.loads(members_path.read_text()) != expected:
            raise ValueError("Existing audit sequence-member mapping differs")
        snapshot_index = read_json(audit / "bound-source-snapshots.json")
        if snapshot_index.get("graph_summary_sha256") != pin:
            raise ValueError("Producer source snapshots are not bound to the reused audit")
        for name, old_name in (("producer", "runner"), ("shared_parser", "alignment_helper"), ("sampling_helper", "sampling_helper"),
                               ("target_preflight_runner", "target_preflight_runner")):
            item = snapshot_index["sources"][old_name]
            path = Path(item["snapshot_path"])
            if sha256(path) != previous["inputs"][old_name]["sha256"] or item["sha256"] != previous["inputs"][old_name]["sha256"]:
                raise ValueError("Preserved producer source does not match the original audit")
            paths[name] = path
        version, binary_sha = previous["mmseqs_version"], previous["inputs"]["mmseqs_binary"]["sha256"]
        execution = execution_record(previous["inputs"]["mmseqs_binary"], {name: record(path) for name, path in paths.items()})
    else:
        cluster_map = load_map(Path(sources["original_clusters"]["path"]), set(sequences))
        validate_group_receipt(
            receipt_path, split_group_tsv=verify_record(previous["split_groups"]),
            chain_data_cache=Path(sources["chain_data_cache"]["path"]), structure_metadata=Path(sources["structure_metadata"]["path"]),
            cluster_map=cluster_map, candidate_ids=set(sequences), min_seq_id=min_seq_id, coverage=coverage,
        )
        paths = {name: verify_record(item) for name, item in previous["proof_artifacts"].items()}
        version, binary_sha = previous["mmseqs_version"], previous["mmseqs_binary_sha256"]
        execution = previous["execution"]
    copied = {}
    for name, source in paths.items():
        expected_sha = sha256(source)
        target = out / ("imported-" + name + source.suffix)
        shutil.copyfile(source, target)
        copied[name] = record(target)
        if copied[name]["sha256"] != expected_sha or sha256(source) != expected_sha:
            raise ValueError("Reused audit evidence changed while copying")
    shutil.copyfile(receipt_path, out / "reused-audit.receipt.json")
    copied["reused_receipt"] = record(out / "reused-audit.receipt.json")
    if sha256(receipt_path) != pin or copied["reused_receipt"]["sha256"] != pin:
        raise ValueError("Reused audit receipt changed during import")
    return {"proof_artifacts": copied, "mmseqs_version": version, "mmseqs_binary_sha256": binary_sha,
            "derivation": "replayed_existing_audit", "reused_audit_receipt_sha256": pin, "execution": execution}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for name in ("chain-data-cache", "structure-metadata", "cluster-tsv", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--mmseqs-bin", default="mmseqs")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--min-seq-id", type=float, default=0.3)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--reuse-audit-dir", type=Path)
    parser.add_argument("--expected-audit-sha256", default="", help="Required receipt digest when reusing an existing complete audit.")
    return parser.parse_args(argv)


def build_groups(args: argparse.Namespace) -> Path:
    out = args.out_dir.resolve()
    if args.out_dir.is_symlink() or (out.is_relative_to(REPO_ROOT) and not out.is_relative_to(REPO_ROOT / ".nanofold_private")):
        raise ValueError("Group outputs must remain in a fresh private directory")
    if bool(args.reuse_audit_dir) != bool(args.expected_audit_sha256):
        raise ValueError("Reusing an audit requires an explicit receipt digest")
    paths = {"chain_data_cache": args.chain_data_cache.resolve(), "structure_metadata": args.structure_metadata.resolve(),
             "original_clusters": args.cluster_tsv.resolve()}
    if any(path.is_relative_to(out) for path in paths.values()):
        raise ValueError("Graph outputs must not contain their inputs")
    validate_search_parameters(min_seq_id=args.min_seq_id, coverage=args.coverage, threads=args.threads, max_seqs=1)
    original_umask = os.umask(0o077)
    try:
        out.mkdir(mode=0o700, parents=True, exist_ok=False)
        receipt_path = out / "split-groups.receipt.json"
        receipt: dict[str, Any] = {"schema_version": 1, "artifact_type": ARTIFACT_TYPE, "status": "running", "scope": SCOPE,
                                   "alignment_contract": search_contract(args.min_seq_id, args.coverage)}
        write_json(receipt_path, receipt)
        try:
            implementations = {
                "builder": Path(__file__).resolve(), "current_parser": REPO_ROOT / "nanofold/mmseqs.py",
                "universe_helper": REPO_ROOT / "scripts/build_manifests.py",
                "cache_validator": REPO_ROOT / "scripts/audit_cache_targets.py",
            }
            implementation_records = {}
            for name, path in implementations.items():
                target = out / f"{name}.py"
                shutil.copyfile(path, target)
                implementation_records[name] = record(target)
            sources = {name: record(path) for name, path in paths.items()}
            sources["structure_metadata"]["semantic_sha256"] = metadata_semantic_sha256(paths["structure_metadata"])
            sequences = load_universe(paths["chain_data_cache"], paths["structure_metadata"])
            cluster_map = load_map(paths["original_clusters"], set(sequences))
            sources["original_clusters"]["assignments_sha256"] = assignments_sha256(cluster_map)
            receipt["sources"] = sources
            records = ordered_records(sequences)
            receipt["universe"] = {"chain_count": len(sequences), "chain_ids_sha256": ids_sha256(set(sequences)),
                                   "unique_sequence_count": len(records), "sequence_records_sha256": sequence_records_sha256(records)}
            if args.reuse_audit_dir:
                receipt.update(import_audit(args.reuse_audit_dir.resolve(), args.expected_audit_sha256, out, sequences, sources,
                                            min_seq_id=args.min_seq_id, coverage=args.coverage))
            else:
                binary = shutil.which(args.mmseqs_bin)
                if binary is None:
                    raise ValueError("MMseqs executable is unavailable")
                binary = str(Path(binary).resolve())
                receipt["mmseqs_version"] = subprocess.run([binary, "version"], capture_output=True, text=True, check=True).stdout.strip()
                receipt["mmseqs_binary_sha256"] = sha256(Path(binary))
                fasta, m8, command_path, log_path = [out / name for name in ("sequences.fasta", "alignments.m8", "command.json", "mmseqs.log")]
                fasta.write_text(fasta_text(records))
                with tempfile.TemporaryDirectory(prefix="mmseqs-", dir=out) as temporary:
                    command = [binary, "easy-search", str(fasta), str(fasta), str(m8), temporary,
                               "--min-seq-id", str(args.min_seq_id), "-c", str(args.coverage), "--cov-mode", "0", "--alignment-mode", "3",
                               "--seq-id-mode", "0", "--search-type", "1", "--dbtype", "1", "--prefilter-mode", "2", "-e", "1e308",
                               "-s", "5.7", "--mask", "0", "--comp-bias-corr", "0", "--threads", str(args.threads),
                               "--max-seqs", str(max(1_000_000, len(records))), "--format-output", FORMAT]
                    write_json(command_path, {"argv": command})
                    print(json.dumps({"phase": "mmseqs", "unique_sequences": len(records), "requested_ordered_pairs": len(records) ** 2}), flush=True)
                    with log_path.open("w") as log:
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                if sha256(Path(binary)) != receipt["mmseqs_binary_sha256"]:
                    raise ValueError("MMseqs executable changed during the search")
                artifacts = {"fasta": fasta, "m8": m8, "command": command_path, "log": log_path}
                artifacts.update(producer=Path(implementation_records["builder"]["path"]),
                                 shared_parser=Path(implementation_records["current_parser"]["path"]))
                receipt.update(derivation="new_mmseqs_search", proof_artifacts={name: record(path) for name, path in artifacts.items()})
                receipt["execution"] = execution_record({"path": binary, "sha256": receipt["mmseqs_binary_sha256"]}, receipt["proof_artifacts"])
            receipt["proof_artifacts"].update(implementation_records)
            artifacts = {name: verify_record(item) for name, item in receipt["proof_artifacts"].items()}
            if artifacts["fasta"].read_text() != fasta_text(records):
                raise ValueError("Search FASTA differs from the complete eligible universe")
            verify_execution(receipt, artifacts)
            group_map, counts = close_graph(sequences, cluster_map, artifacts["m8"], min_seq_id=args.min_seq_id, coverage=args.coverage)
            evidence = verify_search_evidence(artifacts["command"], artifacts["log"], records, counts,
                                              min_seq_id=args.min_seq_id, coverage=args.coverage,
                                              allow_observed_protein=bool(args.reuse_audit_dir))
            group_path = out / "split-groups.tsv"
            group_path.write_text("".join(f"{root}\t{chain}\n" for chain, root in sorted(group_map.items())))
            receipt.update(counts=counts, original_unit_count=counts["original_unit_count"], search_evidence=evidence,
                           split_groups={**record(group_path), "assignments_sha256": assignments_sha256(group_map),
                                         "group_count": len(set(group_map.values()))})
            for name, path in paths.items():
                if sha256(path) != sources[name]["sha256"]:
                    raise ValueError("A bound group input changed during construction")
            if any(sha256(path) != implementation_records[name]["sha256"] for name, path in implementations.items()):
                raise ValueError("A bound graph implementation changed during construction")
            for item in receipt["proof_artifacts"].values():
                verify_record(item)
            receipt["status"] = "complete"
            write_json(receipt_path, receipt)
            print(json.dumps({"status": "complete", "eligible_chains": len(sequences), "original_unit_count": counts["original_unit_count"],
                              "split_group_count": len(set(group_map.values()))}), flush=True)
            return receipt_path
        except BaseException as exc:
            receipt.update(status="error", error_type=type(exc).__name__)
            (out / "error.log").write_text(traceback.format_exc())
            write_json(receipt_path, receipt)
            raise
    finally:
        os.umask(original_umask)


def main(argv: list[str] | None = None) -> int:
    try:
        build_groups(parse_args(argv))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
