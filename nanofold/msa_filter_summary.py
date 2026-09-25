"""Validate aggregate provenance for bounded MSA-row searches."""

from __future__ import annotations

from typing import Any

BATCHING_PUBLIC_FIELDS = frozenset({
    "schema_version", "status", "source_batch_size", "query_batch_size", "batch_count",
    "source_partition_count", "query_partition_count", "source_count", "query_count",
    "alignments_calculated", "receipt_sha256",
})


def validate_batching_summary(row_filter: dict[str, Any], *, public_safe: bool) -> None:
    """Validate completed batch aggregates without exposing individual search jobs."""
    mmseqs = row_filter.get("mmseqs", {})
    binding = row_filter.get("input_binding", {})
    if binding is None:
        binding = {}
    if not isinstance(mmseqs, dict) or not isinstance(binding, dict):
        raise ValueError("MSA search settings and input binding must be objects")
    if "batching" not in mmseqs:
        if "batch_receipt_sha256" in binding:
            raise ValueError("MSA batch receipt requires completed batching metadata")
        return
    batching = mmseqs["batching"]
    fields = BATCHING_PUBLIC_FIELDS if public_safe else BATCHING_PUBLIC_FIELDS | {"receipt_path"}
    if not isinstance(batching, dict) or set(batching) != fields:
        raise ValueError("Unsupported MSA batching summary fields")
    if type(batching["schema_version"]) is not int or batching["schema_version"] != 1 or batching["status"] != "complete":
        raise ValueError("MSA batching summary must describe a completed supported search")
    if row_filter.get("search_contract") != "mmseqs-local-all-pairs-v1" or row_filter.get("exhaustive") is not True:
        raise ValueError("MSA batching requires the exhaustive local alignment contract")
    if mmseqs.get("command") != [] or "m8_sha256" in binding:
        raise ValueError("MSA batching cannot contain a single search command or result binding")

    def count(value: Any, name: str, minimum: int = 0) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"MSA batching {name} must be an integer >= {minimum}")
        return value

    for field in sorted(BATCHING_PUBLIC_FIELDS - {"schema_version", "status", "receipt_sha256"}):
        count(batching[field], field, 0 if field == "alignments_calculated" else 1)
    source = row_filter.get("source")
    heldout = row_filter.get("heldout")
    if not isinstance(source, dict) or not isinstance(source.get("stats"), dict) or not isinstance(heldout, dict):
        raise ValueError("MSA batching requires source and heldout sequence counts")
    source_count = count(source["stats"].get("unique_non_query_rows"), "source sequence count", 1)
    query_count = count(heldout.get("unique_target_sequence_count"), "heldout sequence count", 1)
    if batching["source_batch_size"] > 1_000_000 or batching["query_batch_size"] > 256:
        raise ValueError("MSA batching sizes exceed the supported search bounds")
    capacity = count(mmseqs.get("max_seqs"), "search capacity", 1)
    if capacity > 2_147_483_647 or capacity < min(batching["source_batch_size"], source_count):
        raise ValueError("MSA batching search capacity does not cover every source batch within native limits")
    parameters = mmseqs.get("search_parameters")
    if not isinstance(parameters, dict) or any(
        count(parameters.get(field), field, 1) != capacity for field in ("max_accept", "max_rejected")
    ) or ("max_seqs" in parameters and count(parameters["max_seqs"], "max_seqs", 1) != capacity):
        raise ValueError("MSA batching search parameters disagree with the declared exhaustive capacity")
    possible = count(mmseqs.get("possible_pair_count"), "possible-pair count")
    hits = count(mmseqs.get("hit_count"), "hit count")
    if batching["source_count"] != source_count or batching["query_count"] != query_count or possible != source_count * query_count:
        raise ValueError("MSA batching sequence counts disagree with the source and heldout totals")
    for role, total in (("source", source_count), ("query", query_count)):
        size = batching[f"{role}_batch_size"]
        if batching[f"{role}_partition_count"] != (total + size - 1) // size:
            raise ValueError("MSA batching partition counts disagree with the batch sizes")
    if batching["batch_count"] != batching["source_partition_count"] * batching["query_partition_count"]:
        raise ValueError("MSA batching does not cover the complete partition grid")
    if not hits <= batching["alignments_calculated"] <= possible:
        raise ValueError("MSA batching alignment counts disagree with possible and passing pairs")
    digest = batching["receipt_sha256"]
    if public_safe:
        if digest != "<private>":
            raise ValueError("MSA batching receipt hash must be redacted in public summaries")
    else:
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("MSA batching receipt hash must be a SHA256 digest")
        if not isinstance(batching["receipt_path"], str) or not batching["receipt_path"]:
            raise ValueError("MSA batching receipt path must be a nonempty string")
    if binding.get("batch_receipt_sha256") != digest:
        raise ValueError("MSA batching receipt hash disagrees with the input binding")
