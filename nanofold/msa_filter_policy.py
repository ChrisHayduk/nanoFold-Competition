"""Bind directed MSA filtering to disjoint manifests and immutable source evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from nanofold.chain_paths import chain_data_dir
from nanofold.mmseqs import (
    _batch_records,
    _batch_validate_job,
    _batch_verify_partitions,
    batched_mmseqs_summary,
    canonical_target_sequence,
    iter_mmseqs_hits,
)
from nanofold.msa_filter_summary import validate_batching_summary

_POLICY_TYPE = "directed-heldout-v1"
_ROLES = ("train", "val")
_SPLITS = (*_ROLES, "hidden")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _digest(value: Any) -> str:
    _require(isinstance(value, str) and _SHA256.fullmatch(value) is not None, "Expected a lowercase SHA-256 digest")
    return value


def _count(value: Any, *, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, "Expected an integer count with a valid lower bound")
    return value


def _object(value: Any, fields: set[str] | None = None) -> dict[str, Any]:
    _require(isinstance(value, dict), "Expected a JSON object")
    if fields is not None:
        _require(set(value) == fields, "Unsupported or missing policy fields")
    return value


def _list(value: Any) -> list[Any]:
    _require(isinstance(value, list), "Expected a JSON list")
    return value


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, "Duplicate JSON object key")
        result[key] = value
    return result


@dataclass
class _Snapshots:
    files: dict[Path, str] = field(default_factory=dict)
    resolved: dict[Path, Path] = field(default_factory=dict)

    def resolve(self, path: Path) -> Path:
        original = path.absolute()
        if original not in self.resolved:
            self.resolved[original] = original.resolve(strict=True)
        return self.resolved[original]

    def add(self, path: Path, digest: str) -> None:
        if path in self.files:
            _require(self.files[path] == digest, "A policy dependency changed while loading")
        self.files[path] = digest

    def capture(self, path: Path, expected: str | None = None) -> Path:
        resolved = self.resolve(path)
        _require(resolved.is_file(), "Policy dependency must be a regular file")
        digest = _sha256(resolved)
        if expected is not None:
            _require(digest == _digest(expected), "Policy dependency SHA-256 mismatch")
        self.add(resolved, digest)
        return resolved

    def read_json(self, path: Path, expected: str | None = None) -> tuple[Path, dict[str, Any]]:
        resolved = self.resolve(path)
        data = resolved.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if expected is not None:
            _require(digest == _digest(expected), "Policy dependency SHA-256 mismatch")
        self.add(resolved, digest)
        return resolved, _object(json.loads(data, object_pairs_hook=_json_pairs))

    def assert_unchanged(self) -> None:
        for path, digest in self.files.items():
            _require(path.is_file() and _sha256(path) == digest, "A bound MSA policy dependency changed")


def _reference(value: Any, base: Path) -> Path:
    _require(isinstance(value, str) and bool(value.strip()), "Dependency path must be a nonempty string")
    path = Path(value)
    return path if path.is_absolute() else base / path


@dataclass(frozen=True)
class MSARowFilter:
    path: Path
    sha256: str
    excluded_sequence_sha256: frozenset[str]


def _scalar_filter(path: Path, snapshots: _Snapshots, expected: str | None = None) -> tuple[MSARowFilter, dict[str, Any]]:
    resolved, payload = snapshots.read_json(path, expected)
    values = _list(payload.get("excluded_sequence_sha256"))
    excluded = frozenset(_digest(str(value).strip().lower()) for value in values)
    return MSARowFilter(resolved, snapshots.files[resolved], excluded), payload


def load_msa_row_filter(path: str | Path) -> MSARowFilter:
    """Load scalar exclusions, hashing exactly the bytes that were parsed.

    Scalar use supports exploratory filters. Production directed policies apply
    the complete provenance and search-contract validation below.
    """
    return _scalar_filter(Path(path), _Snapshots())[0]


def validate_msa_filter_policy_provenance(value: Any) -> dict[str, Any]:
    """Validate the public metadata shape without admitting hidden references."""
    value = _object(value, {"schema_version", "policy_type", "policy_sha256", "splits"})
    _require(type(value["schema_version"]) is int and value["schema_version"] == 1
             and value["policy_type"] == _POLICY_TYPE, "Unsupported MSA filter policy provenance")
    _digest(value["policy_sha256"])
    splits = _object(value["splits"], set(_ROLES))
    for role in _ROLES:
        record = _object(splits[role], {"manifest_sha256", "chain_count", "filter_sha256"})
        _digest(record["manifest_sha256"])
        _digest(record["filter_sha256"])
        _count(record["chain_count"], minimum=1)
    return copy.deepcopy(value)


def _manifest(record: Any, base: Path, snapshots: _Snapshots) -> tuple[dict[str, Any], tuple[str, ...]]:
    record = _object(record, {"path", "sha256", "chain_count"})
    path = snapshots.resolve(_reference(record["path"], base))
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    _require(digest == _digest(record["sha256"]), "Manifest SHA-256 mismatch")
    snapshots.add(path, digest)
    chains = tuple(line.strip() for line in data.decode("utf-8").splitlines()
                   if line.strip() and not line.lstrip().startswith("#"))
    _require(bool(chains) and len(chains) == len(set(chains)), "Manifest must contain nonempty unique chain assignments")
    _require(all(not any(character.isspace() or ord(character) < 32 for character in chain) for chain in chains),
             "Manifest contains an invalid chain identifier")
    _require(_count(record["chain_count"], minimum=1) == len(chains), "Manifest chain count mismatch")
    return {"path": str(path), "sha256": digest, "chain_count": len(chains)}, chains


def _manifest_record(path: Path, snapshots: _Snapshots) -> dict[str, Any]:
    resolved = snapshots.resolve(path)
    data = resolved.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    snapshots.add(resolved, digest)
    count = sum(bool(line.strip()) and not line.lstrip().startswith("#") for line in data.decode("utf-8").splitlines())
    return {"path": str(resolved), "sha256": digest, "chain_count": count}


def _threshold(value: Any) -> dict[str, Any]:
    value = _object(value, {"min_seq_id", "coverage", "cov_mode", "alignment_mode", "seq_id_mode"})
    for name in ("min_seq_id", "coverage"):
        number = value[name]
        _require(type(number) in (float, int) and math.isfinite(number) and 0 < number <= 1,
                 "MSA identity and coverage must be finite values in (0, 1]")
    for name, expected in (("cov_mode", 0), ("alignment_mode", 3), ("seq_id_mode", 0)):
        _require(type(value[name]) is int and value[name] == expected, "Unsupported alignment threshold semantics")
    _require(value["min_seq_id"] == 0.3 and value["coverage"] == 0.8,
             "Directed-heldout-v1 requires identity 0.30 and bidirectional coverage 0.80")
    return value


def _source_assets(source: dict[str, Any], binding: dict[str, Any], chains: tuple[str, ...],
                   base: Path, snapshots: _Snapshots) -> tuple[tuple[str, ...], dict[Path, str]]:
    _require(source.get("mode") == "raw_a3m", "Production directed filters require complete raw A3M sources")
    names = _list(source.get("msa_names"))
    _require(bool(names) and all(isinstance(name, str) and name not in ("", ".", "..")
             and Path(name).name == name and "/" not in name and "\\" not in name for name in names), "Invalid MSA source names")
    _require(len(set(names)) == len(names), "Duplicate MSA source names")
    records = _list(binding.get("source_assets"))
    _require(bool(records), "Filter requires immutable raw source asset records")
    _require(hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest() == _digest(binding.get("source_asset_manifest_sha256")),
             "Raw source asset manifest digest mismatch")
    assets: dict[Path, str] = {}
    for record in records:
        record = _object(record, {"path", "sha256"})
        path = snapshots.capture(_reference(record["path"], base), record["sha256"])
        _require(path not in assets, "Duplicate raw source asset binding")
        assets[path] = record["sha256"]
    root = snapshots.resolve(_reference(source.get("raw_root"), base)) / "roda_pdb"
    expected_paths: set[Path] = set()
    for chain in chains:
        directory = chain_data_dir(root, chain)
        for name in names:
            candidates = [path for path in (directory / name, directory / "a3m" / name) if path.is_file()]
            if not candidates:
                candidates = [path for path in directory.rglob(name) if path.is_file()]
            _require(bool(candidates), "A required source MSA is missing")
            # A production policy must identify one immutable file per name.
            paths = {snapshots.resolve(candidate) for candidate in candidates}
            _require(len(paths) == 1, "Ambiguous raw MSA source paths")
            expected_paths.update(paths)
    _require(set(assets) == expected_paths, "Raw source asset bindings do not exactly cover the source manifest")
    return tuple(names), assets


def _verify_checkpoint(payload: dict[str, Any], base: Path, snapshots: _Snapshots,
                       input_files: dict[Path, str], expected_query_hashes: set[str]) -> None:
    settings, binding = payload["mmseqs"], payload["input_binding"]
    batching = settings["batching"]
    receipt_path, receipt = snapshots.read_json(_reference(batching["receipt_path"], base), batching["receipt_sha256"])
    result = {**receipt, "receipt_path": str(receipt_path), "receipt_sha256": batching["receipt_sha256"]}
    _require(batched_mmseqs_summary(result) == batching, "Filter batching metadata differs from its completed receipt")
    plan_path, plan = snapshots.read_json(_reference(receipt["plan_path"], receipt_path.parent), receipt["plan_sha256"])
    root = receipt_path.parent
    plan_binding = _object(plan.get("binding"))
    _require(plan_binding.get("search_contract") == "mmseqs-local-all-pairs-v1", "Checkpoint is not exhaustive")
    expected_context = hashlib.sha256(json.dumps({
        "files": [{"path": str(path), "sha256": digest} for path, digest in sorted(input_files.items())],
        "source_assets": binding["source_assets"],
    }, sort_keys=True).encode()).hexdigest()
    _require(plan_binding.get("input_binding_sha256") == expected_context, "Checkpoint does not bind the filter's original source dependencies")
    _require(plan_binding["source_input"]["sha256"] == binding["source_fasta_sha256"]
             and plan_binding["query_input"]["sha256"] == binding["heldout_fasta_sha256"], "Checkpoint FASTA bindings differ from the filter")
    binary_record = _object(plan_binding.get("binary"), {"path", "sha256", "version"})
    binary = snapshots.capture(_reference(binary_record["path"], root), binary_record["sha256"])
    _require(binary_record["version"] == settings["version"], "Checkpoint binary version differs from the filter")
    flags = _list(plan_binding.get("flags"))
    _require(len(flags) % 2 == 0 and all(isinstance(value, str) for value in flags), "Invalid checkpoint flags")
    parsed = dict(zip(flags[::2], flags[1::2], strict=True))
    _require(len(parsed) * 2 == len(flags), "Duplicate checkpoint flags")
    _validate_flags(parsed, payload)
    for role, fasta_field in (("source", "source_fasta_sha256"), ("query", "heldout_fasta_sha256")):
        parts = plan[f"{role}_partitions"]
        _require(plan[f"{role}_count"] == batching[f"{role}_count"] and len(parts) == batching[f"{role}_partition_count"]
                 and plan_binding[f"{role}_batch_size"] == batching[f"{role}_batch_size"], "Checkpoint partition totals differ from the filter")
        for part in parts:
            snapshots.capture(_reference(part["path"], root), part["sha256"])
        _batch_verify_partitions(parts, {"count": batching[f"{role}_count"], "sha256": binding[fasta_field]},
                                 batching[f"{role}_batch_size"], root)
    query_hashes = {hashlib.sha256(sequence.encode()).hexdigest()
                    for part in plan["query_partitions"] for _, sequence in _batch_records(Path(part["path"]))}
    _require(query_hashes == expected_query_hashes and len(query_hashes) == batching["query_count"],
             "Checkpoint heldout sequences differ from their bound cache targets")
    query_set_sha = hashlib.sha256("".join(f"{digest}\n" for digest in sorted(query_hashes)).encode()).hexdigest()
    _require(query_set_sha == binding["heldout_sequence_set_sha256"], "Heldout sequence-set binding mismatch")
    source_set_sha = hashlib.sha256()
    exact_hashes: set[str] = set()
    for part in plan["source_partitions"]:
        for header, sequence in _batch_records(Path(part["path"])):
            digest = hashlib.sha256(sequence.encode()).hexdigest()
            _require(len(sequence) >= 20 and header == "seq_" + digest, "Checkpoint source header does not identify its eligible sequence")
            source_set_sha.update(f"{digest}\n".encode())
            if digest in query_hashes:
                exact_hashes.add(digest)
    _require(source_set_sha.hexdigest() == binding["source_sequence_set_sha256"], "Source sequence-set binding mismatch")
    pairs = [(source, query) for source in plan["source_partitions"] for query in plan["query_partitions"]]
    jobs = _list(receipt.get("jobs"))
    _require(len(jobs) == len(pairs) == batching["batch_count"], "Checkpoint does not contain every partition pair")
    hits = 0
    alignments = 0
    excluded_hashes = set(exact_hashes)
    for job, (source, query) in zip(jobs, pairs, strict=True):
        job_path, job_receipt = snapshots.read_json(_reference(job["receipt_path"], root), job["receipt_sha256"])
        _require(job_receipt == {key: value for key, value in job.items() if key not in ("receipt_path", "receipt_sha256")}, "Checkpoint job receipt mismatch")
        for kind in ("m8", "log"):
            snapshots.capture(_reference(job[f"{kind}_path"], root), job[f"{kind}_sha256"])
        _batch_validate_job(job, query, source, plan_sha256=receipt["plan_sha256"], root=root, job_root=job_path.parent,
                            binary=binary, flags=flags, min_seq_id=payload["threshold"]["min_seq_id"], coverage=payload["threshold"]["coverage"])
        query_ids = {header for header, _ in _batch_records(Path(query["path"]))}
        source_ids = {header for header, _ in _batch_records(Path(source["path"]))}
        for _, target in iter_mmseqs_hits(Path(job["m8_path"]), query_ids=query_ids, target_ids=source_ids,
                                          min_seq_id=payload["threshold"]["min_seq_id"], coverage=payload["threshold"]["coverage"]):
            excluded_hashes.add(target.removeprefix("seq_"))
        hits += job["hit_count"]
        alignments += job["alignment_count"]
    _require(hits == receipt["hit_count"] == settings["hit_count"] and alignments == batching["alignments_calculated"], "Checkpoint hit accounting differs from filter metadata")
    _require(excluded_hashes == set(payload["excluded_sequence_sha256"])
             and len(exact_hashes) == payload["exact_match_excluded_sequence_count"],
             "Filter exclusions do not equal verified search hits plus exact sequence matches")
    _require(plan_path.parent == root, "Checkpoint plan must belong to the completed checkpoint")


def _validate_flags(flags: dict[str, str], payload: dict[str, Any]) -> None:
    settings, threshold = payload["mmseqs"], payload["threshold"]
    expected = {"--min-seq-id": str(threshold["min_seq_id"]), "-c": str(threshold["coverage"]),
                "--cov-mode": "0", "--alignment-mode": "3", "--seq-id-mode": "0", "--search-type": "1", "--dbtype": "1",
                "--prefilter-mode": "2", "-e": "1e308", "-s": "5.7", "--mask": "0", "--comp-bias-corr": "0",
                "--max-seqs": str(settings["max_seqs"]), "--max-accept": str(settings["max_seqs"]), "--max-rejected": str(settings["max_seqs"]),
                "--format-output": "query,target,pident,alnlen,qcov,tcov,evalue,bits"}
    _require(set(flags) == set(expected) | {"--threads"}, "Search command has missing or unsupported flags")
    _require(all(flags[key] == value for key, value in expected.items()), "Search command differs from the exhaustive filter contract")
    _require(flags["--threads"].isdecimal() and 1 <= int(flags["--threads"]) <= 2_147_483_647, "Invalid search thread count")


def _production_filter(row_filter: MSARowFilter, payload: dict[str, Any], *, source_manifest: dict[str, Any],
                       heldout_manifests: list[dict[str, Any]], source_chains: tuple[str, ...],
                       snapshots: _Snapshots) -> tuple[dict[str, Any], str, tuple[str, ...], dict[Path, str]]:
    _require(type(payload.get("schema_version")) is int and payload["schema_version"] == 1
             and payload.get("filter_type") == "heldout_target_homology" and payload.get("exhaustive") is True
             and payload.get("search_contract") == "mmseqs-local-all-pairs-v1" and "inventory_type" not in payload,
             "Directed policy requires a completed exhaustive homology filter")
    values = payload["excluded_sequence_sha256"]
    _require(all(isinstance(value, str) and _SHA256.fullmatch(value) for value in values)
             and values == sorted(row_filter.excluded_sequence_sha256), "Production exclusions must be sorted unique lowercase hashes")
    excluded_count = _count(payload.get("excluded_sequence_count"))
    _require(excluded_count == len(values), "Excluded sequence count mismatch")
    exact_count = _count(payload.get("exact_match_excluded_sequence_count"))
    _require(exact_count <= excluded_count, "Exact exclusion count exceeds total exclusions")
    threshold = _threshold(payload.get("threshold"))
    source, heldout, settings, binding = (_object(payload.get(name)) for name in ("source", "heldout", "mmseqs", "input_binding"))
    _require(source.get("complete") is True and type(source.get("minimum_non_query_sequence_length")) is int
             and source["minimum_non_query_sequence_length"] == 20,
             "Directed policy requires complete source rows with the fixed minimum length")
    base = row_filter.path.parent
    input_files: dict[Path, str] = {}
    heldout_chains: list[str] = []
    for container, expected_records in ((source, [source_manifest]), (heldout, heldout_manifests)):
        records = _list(container.get("manifests"))
        _require(len(records) == len(expected_records), "Filter has the wrong directed source or heldout manifests")
        identities = []
        for record in records:
            actual, actual_chains = _manifest(record, base, snapshots)
            if container is heldout:
                heldout_chains.extend(actual_chains)
            identities.append((actual["sha256"], actual["chain_count"]))
            input_files[Path(actual["path"])] = actual["sha256"]
        _require(sorted(identities) == sorted((record["sha256"], record["chain_count"]) for record in expected_records),
                 "Filter manifest identities do not match the directed split assignment")
    stats = _object(source.get("stats"))
    _require(_count(stats.get("chains"), minimum=1) == len(source_chains) and _count(stats.get("missing_msa_files")) == 0,
             "Filter source coverage is incomplete")
    source_count = _count(stats.get("unique_non_query_rows"), minimum=1)
    _require(source_count <= _count(stats.get("non_query_rows")) <= _count(stats.get("rows")), "Invalid source row counts")
    _require(excluded_count <= source_count, "Exclusions exceed collected source sequences")
    query_count = _count(heldout.get("unique_target_sequence_count"), minimum=1)
    _require(query_count <= sum(record["chain_count"] for record in heldout_manifests), "Heldout sequence count exceeds manifest size")
    _require(_count(settings.get("possible_pair_count")) == source_count * query_count, "Search pair count mismatch")
    hit_count = _count(settings.get("hit_count"))
    _require(hit_count <= source_count * query_count and excluded_count <= hit_count + exact_count, "Exclusion counts disagree with search hits")
    maximum = _count(settings.get("max_seqs"), minimum=1)
    _require(maximum <= 2_147_483_647, "Search capacity exceeds the native integer limit")
    params = _object(settings.get("search_parameters"), {"prefilter_mode", "evalue", "sensitivity", "mask", "comp_bias_corr", "search_type", "dbtype", "max_accept", "max_rejected"})
    expected_params = {"prefilter_mode": 2, "evalue": 1e308, "sensitivity": 5.7, "mask": 0,
                       "comp_bias_corr": 0, "search_type": 1, "dbtype": 1, "max_accept": maximum, "max_rejected": maximum}
    _require(params == expected_params and all(type(value) in (int, float) and math.isfinite(value) for value in params.values()),
             "Filter does not use the exhaustive native search settings")
    cache = _object(payload.get("chain_data_cache"), {"path", "sha256"})
    cache_path, cache_data = snapshots.read_json(_reference(cache["path"], base), cache["sha256"])
    input_files[cache_path] = cache["sha256"]
    expected_query_hashes = set()
    for chain in heldout_chains:
        entry = _object(cache_data.get(chain))
        sequence = next((entry[key] for key in ("sequence", "seq", "seqres", "aatype_sequence")
                         if isinstance(entry.get(key), str) and entry[key].strip()), None)
        _require(sequence is not None, "A heldout target sequence is missing from the bound cache")
        assert sequence is not None
        expected_query_hashes.add(hashlib.sha256(canonical_target_sequence(sequence).encode()).hexdigest())
    _require(len(expected_query_hashes) == query_count, "Heldout unique count differs from bound cache sequences")
    for name in ("source_fasta_sha256", "heldout_fasta_sha256", "source_sequence_set_sha256", "heldout_sequence_set_sha256"):
        _digest(binding.get(name))
    names, assets = _source_assets(source, binding, source_chains, base, snapshots)
    validate_batching_summary(payload, public_safe=False)
    _require("batching" in settings, "Production directed filters require retained verified checkpoint evidence")
    _require(maximum >= min(source_count, settings["batching"]["source_batch_size"]), "Batch hit capacity cannot cover its source partition")
    _verify_checkpoint(payload, base, snapshots, input_files, expected_query_hashes)
    return threshold, cache["sha256"], names, assets


@dataclass(frozen=True)
class MSAFilterPolicy:
    path: Path
    sha256: str
    filters: Mapping[str, MSARowFilter]
    assignments: Mapping[str, str]
    manifest_paths: Mapping[str, Path]
    msa_names: Mapping[str, tuple[str, ...]]
    chain_data_cache_sha256: str
    _provenance: dict[str, Any] = field(repr=False)
    _snapshots: _Snapshots = field(repr=False)
    _assets: Mapping[str, dict[Path, str]] = field(repr=False)

    @property
    def provenance(self) -> dict[str, Any]:
        return copy.deepcopy(self._provenance)

    def filter_for(self, chain_id: str) -> MSARowFilter:
        _require(chain_id in self.assignments, "Chain has no public directed-filter assignment")
        return self.filters[self.assignments[chain_id]]

    def assert_unchanged(self) -> None:
        self._snapshots.assert_unchanged()

    def assert_output_roots_safe(self, roots: Sequence[Path]) -> None:
        dependency_inodes = {(path.stat().st_dev, path.stat().st_ino) for path in self._snapshots.files}
        for root in roots:
            resolved = Path(root).resolve()
            _require(not any(path == resolved or path.is_relative_to(resolved) for path in self._snapshots.files),
                     "Preprocessing output root contains an immutable policy dependency")
            if resolved.is_dir():
                for existing in resolved.rglob("*"):
                    _require(not (existing.is_symlink() and existing.is_dir()), "Preprocessing output directories must not contain directory symlinks")
                    if existing.is_file():
                        stat = existing.stat()
                        _require((stat.st_dev, stat.st_ino) not in dependency_inodes,
                                 "An existing output aliases an immutable policy dependency")

    def validate_output_paths(self, paths: Sequence[Path]) -> list[Path]:
        """Resolve output targets and reject aliases of any bound evidence file."""
        resolved = [Path(path).resolve() for path in paths]
        for target in resolved:
            _require(all(target != path and not (target.exists() and target.samefile(path)) for path in self._snapshots.files),
                     "Output would overwrite an immutable policy dependency")
        return resolved

    def assert_msa_names(self, names: Sequence[str]) -> None:
        _require(len(names) == len(set(names)) and all(set(names) == set(self.msa_names[role]) for role in _ROLES),
                 "Preprocessing MSA names differ from the directed filters")

    def assert_msa_inputs(self, chain_id: str, msa_paths: Mapping[str, Path | None]) -> dict[str, Path]:
        self.filter_for(chain_id)
        role = self.assignments[chain_id]
        _require(set(msa_paths) == set(self.msa_names[role]), "Preprocessing MSA names differ from the assigned filter")
        resolved: dict[str, Path] = {}
        for name, path in msa_paths.items():
            _require(path is not None, "Preprocessing MSA is missing from the complete directed source")
            assert path is not None
            actual = Path(path).resolve(strict=True)
            _require(actual in self._assets[role] and _sha256(actual) == self._assets[role][actual],
                     "Preprocessing MSA bytes or path are not bound by the assigned filter")
            resolved[name] = actual
        return resolved


def _load_payload(payload: dict[str, Any], base: Path, snapshots: _Snapshots) -> tuple[dict[str, Any], dict[str, MSARowFilter], dict[str, str], str, dict[str, tuple[str, ...]], dict[str, dict[Path, str]]]:
    _object(payload, {"schema_version", "policy_type", "manifests", "filters"})
    _require(type(payload["schema_version"]) is int and payload["schema_version"] == 1 and payload["policy_type"] == _POLICY_TYPE,
             "Unsupported directed MSA filter policy")
    manifest_values, filter_values = _object(payload["manifests"], set(_SPLITS)), _object(payload["filters"], set(_ROLES))
    manifests, chains = {}, {}
    for role in _SPLITS:
        manifests[role], chains[role] = _manifest(manifest_values[role], base, snapshots)
    _require(all(set(chains[left]).isdisjoint(chains[right]) for index, left in enumerate(_SPLITS) for right in _SPLITS[index + 1:]),
             "Directed policy manifests overlap")
    filters, thresholds, cache_hashes, names, assets = {}, [], [], {}, {}
    for role in _ROLES:
        record = _object(filter_values[role], {"path", "sha256"})
        row_filter, raw = _scalar_filter(_reference(record["path"], base), snapshots, record["sha256"])
        heldout_roles = ("val", "hidden") if role == "train" else ("hidden",)
        threshold, cache_hash, names[role], assets[role] = _production_filter(
            row_filter, raw, source_manifest=manifests[role], heldout_manifests=[manifests[key] for key in heldout_roles],
            source_chains=chains[role], snapshots=snapshots)
        filters[role] = row_filter
        thresholds.append(threshold)
        cache_hashes.append(cache_hash)
    _require(thresholds[0] == thresholds[1] and cache_hashes[0] == cache_hashes[1], "Directed filters must use the same thresholds and chain cache")
    assignments = {chain: role for role in _ROLES for chain in chains[role]}
    return manifests, filters, assignments, cache_hashes[0], names, assets


def load_msa_filter_policy(path: str | Path, expected_chain_ids: Sequence[str] | None = None) -> MSAFilterPolicy:
    snapshots = _Snapshots()
    resolved, payload = snapshots.read_json(Path(path))
    manifests, filters, assignments, cache_hash, names, assets = _load_payload(payload, resolved.parent, snapshots)
    if expected_chain_ids is not None:
        _require(len(expected_chain_ids) == len(set(expected_chain_ids)) and set(expected_chain_ids) == set(assignments),
                 "Preprocessing manifest must exactly cover public policy assignments")
    digest = snapshots.files[resolved]
    provenance = validate_msa_filter_policy_provenance({"schema_version": 1, "policy_type": _POLICY_TYPE, "policy_sha256": digest,
        "splits": {role: {"manifest_sha256": manifests[role]["sha256"], "chain_count": manifests[role]["chain_count"],
                          "filter_sha256": filters[role].sha256} for role in _ROLES}})
    snapshots.assert_unchanged()
    return MSAFilterPolicy(resolved, digest, MappingProxyType(filters), MappingProxyType(assignments),
                          MappingProxyType({role: Path(manifests[role]["path"]) for role in _ROLES}), MappingProxyType(names),
                          cache_hash, provenance, snapshots, MappingProxyType(assets))


def build_msa_filter_policy(*, train_manifest: Path, val_manifest: Path, hidden_manifest: Path,
                            train_filter: Path, val_filter: Path) -> dict[str, Any]:
    snapshots = _Snapshots()
    manifests = {role: _manifest_record(path, snapshots) for role, path in zip(_SPLITS, (train_manifest, val_manifest, hidden_manifest), strict=True)}
    filters = {}
    for role, path in zip(_ROLES, (train_filter, val_filter), strict=True):
        resolved = snapshots.capture(path)
        filters[role] = {"path": str(resolved), "sha256": snapshots.files[resolved]}
    payload = {"schema_version": 1, "policy_type": _POLICY_TYPE, "manifests": manifests, "filters": filters}
    _load_payload(payload, Path.cwd(), snapshots)
    snapshots.assert_unchanged()
    return payload


def validate_msa_filter_policy_output(output: Path, payload: dict[str, Any]) -> Path:
    """Return a resolved output path after protecting every transitive dependency."""
    snapshots = _Snapshots()
    _load_payload(payload, Path.cwd(), snapshots)
    target = output.resolve()
    _require(all(target != path and not (target.exists() and target.samefile(path)) for path in snapshots.files),
             "Policy output would overwrite an immutable source dependency")
    snapshots.assert_unchanged()
    return target
