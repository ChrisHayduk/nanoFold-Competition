"""Pinned CATH domain boundaries with sequence-verified residue coordinates."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import gemmi

CATH_VERSION = "4.4.0"
CATH_BOUNDARY_SOURCES = {
    "cath_domain_boundaries_seqres": {
        "url": "https://download.cathdb.info/cath/releases/all-releases/v4_4_0/cath-classification-data/cath-domain-boundaries-seqreschopping-v4_4_0.txt",
        "path": "cath-domain-boundaries-seqres.txt",
        "version": CATH_VERSION,
    },
    "cath_domain_sequences": {
        "url": "https://download.cathdb.info/cath/releases/all-releases/v4_4_0/sequence-data/cath-domain-seqs-v4_4_0.fa",
        "path": "cath-domain-seqs.fa",
        "version": CATH_VERSION,
    },
}
_DOMAIN_ID = re.compile(r"[0-9A-Za-z]{4}[0-9A-Za-z][0-9]{2}")
_AUTHOR_SEGMENT = re.compile(r"(-?\d+)(?:\(([A-Za-z0-9])\)|([A-Za-z]))?-(-?\d+)(?:\(([A-Za-z0-9])\)|([A-Za-z]))?")


def load_cath_boundaries(source_dir: Path) -> dict[str, dict[str, Any]]:
    """Join release-matched SEQRES intervals and author-numbered domain FASTA.

    A few rows in the released SEQRES file have reversed endpoints. Preserve
    these as explicit unresolved annotations instead of inventing a correction.
    The caller must verify the source lock before invoking this parser.
    """
    records: dict[str, dict[str, Any]] = {}
    path = source_dir / CATH_BOUNDARY_SOURCES["cath_domain_boundaries_seqres"]["path"]
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 2 or not _DOMAIN_ID.fullmatch(fields[0]):
                raise ValueError(f"Malformed CATH SEQRES row: {path.name}:{line_number}")
            segments = []
            for segment in fields[1].split(","):
                if not re.fullmatch(r"\d+-\d+", segment):
                    raise ValueError(f"Malformed CATH SEQRES segment: {path.name}:{line_number}")
                start, end = map(int, segment.split("-"))
                segments.append({"start": start, "end": end})
            domain_id = fields[0][:4].lower() + fields[0][4:]
            record = {"seqres_segments": segments, "author_segments": [], "sequence": None}
            if domain_id in records and records[domain_id] != record:
                raise ValueError(f"Conflicting CATH SEQRES boundaries: {domain_id}")
            records[domain_id] = record
    if not records:
        raise ValueError("CATH boundary source contains no domain records")

    current: str | None = None
    segments_author: list[dict[str, Any]] = []
    sequence_parts: list[str] = []
    seen: set[str] = set()

    def finish() -> None:
        if current is None:
            return
        sequence = "".join(sequence_parts)
        if not sequence or not re.fullmatch(r"[A-Z]+", sequence):
            raise ValueError(f"Invalid CATH domain FASTA sequence: {current}")
        if current in seen:
            raise ValueError(f"Duplicate CATH domain FASTA record: {current}")
        seen.add(current)
        if current in records:
            records[current].update(author_segments=segments_author, sequence=sequence)

    path = source_dir / CATH_BOUNDARY_SOURCES["cath_domain_sequences"]["path"]
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                finish()
                match = re.fullmatch(r">cath\|4_4_0\|([^/]+)/([^\s]+)", line)
                if not match or not _DOMAIN_ID.fullmatch(match[1]):
                    raise ValueError(f"Malformed or wrong-version CATH FASTA header: {path.name}:{line_number}")
                current = match[1][:4].lower() + match[1][4:]
                segments_author = []
                for segment in match[2].split("_"):
                    endpoint = _AUTHOR_SEGMENT.fullmatch(segment)
                    if not endpoint:
                        raise ValueError(f"Malformed CATH author segment: {current}")
                    segments_author.append({
                        "start_number": int(endpoint[1]),
                        "start_insertion_code": endpoint[2] or endpoint[3] or "",
                        "end_number": int(endpoint[4]),
                        "end_insertion_code": endpoint[5] or endpoint[6] or "",
                    })
                sequence_parts = []
            else:
                if current is None:
                    raise ValueError("CATH domain FASTA sequence precedes its header")
                sequence_parts.append(line)
        finish()
    if not seen:
        raise ValueError("CATH domain FASTA contains no records")
    return records


def _positions(segments: list[dict[str, int]], length: int) -> set[int] | None:
    positions: set[int] = set()
    for segment in segments:
        start, end = segment["start"], segment["end"]
        if not 1 <= start <= end <= length:
            return None
        current = set(range(start, end + 1))
        if positions.intersection(current):
            return None
        positions.update(current)
    return positions or None


def domain_metadata(
    domains: list[dict[str, Any]], sequence: str, boundaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Map release coordinates only when their exact domain sequence agrees.

    ``label_segments`` uses the input polymer sequence's 1-based coordinates;
    author residue numbers are never interpreted as array indices. Annotation
    absence means CATH-unannotated, not absence of a structural domain.
    """
    out: list[dict[str, Any]] = []
    covered: set[int] = set()
    overlap = False
    for domain in domains:
        record = boundaries.get(domain["domain_id"])
        item = {**domain, "domain_length": None, "author_segments": [], "seqres_segments": [],
                "label_segments": [], "boundary_status": "missing_release_boundaries"}
        if record:
            item.update(author_segments=record["author_segments"], seqres_segments=record["seqres_segments"])
            positions = _positions(record["seqres_segments"], len(sequence))
            if not record["sequence"]:
                item["boundary_status"] = "missing_release_domain_sequence"
            elif positions is None:
                item["boundary_status"] = "invalid_or_out_of_range_seqres_segments"
            else:
                projected = "".join(sequence[segment["start"] - 1:segment["end"]]
                                    for segment in record["seqres_segments"])
                if projected != record["sequence"]:
                    item["boundary_status"] = "domain_sequence_mismatch"
                else:
                    item.update(domain_length=len(positions), label_segments=record["seqres_segments"],
                                boundary_status="verified_domain_sequence")
                    overlap |= bool(covered.intersection(positions))
                    covered.update(positions)
            # Release SEQRES chopping can be stale after sequence revisions.
            # A unique exact contiguous domain sequence establishes its current
            # position without assuming an author-number offset or alignment.
            domain_sequence = record["sequence"]
            if (item["boundary_status"] != "verified_domain_sequence" and domain_sequence
                    and len(record["author_segments"]) == 1 and len(record["seqres_segments"]) == 1):
                start = sequence.find(domain_sequence)
                if start >= 0 and sequence.find(domain_sequence, start + 1) < 0:
                    positions = set(range(start + 1, start + 1 + len(domain_sequence)))
                    item.update(domain_length=len(domain_sequence),
                                label_segments=[{"start": start + 1, "end": start + len(domain_sequence)}],
                                boundary_status="verified_unique_domain_sequence")
                    overlap |= bool(covered.intersection(positions))
                    covered.update(positions)
        out.append(item)
    # Conflicting domains cannot support an additive domain exposure estimate.
    if overlap:
        for item in out:
            if item["boundary_status"].startswith("verified_"):
                item.update(domain_length=None, label_segments=[], boundary_status="overlapping_domain_annotations")
    complete = not overlap and all(item["boundary_status"].startswith("verified_") for item in out)
    count = len(covered) if complete else None
    unannotated = len(sequence) - count if count is not None else None
    order = sorted(out, key=lambda item: min(part["start"] for part in item["label_segments"])) if out and complete else []
    return {
        "cath_domains": out,
        "cath_domain_count": len(out),
        "cath_topology_counts": dict(sorted(Counter(item["topology_id"] for item in out).items())),
        "cath_superfamily_counts": dict(sorted(Counter(item["homologous_superfamily_id"] for item in out).items())),
        "cath_boundary_status": "unannotated" if not out else ("verified" if complete else "unresolved"),
        "cath_annotated_residue_count": count,
        "cath_unannotated_residue_count": unannotated,
        "cath_unannotated_fraction": unannotated / len(sequence) if unannotated is not None and sequence else None,
        "cath_domain_order": [item["domain_id"] for item in order] if complete else None,
    }


def author_segments_to_label(
    mmcif_path: Path, author_chain: str, author_segments: list[dict[str, Any]], length: int,
) -> list[dict[str, int]] | None:
    """Map author endpoints using the polymer scheme, including insertion codes.

    This optional cross-check needs an unambiguous complete polymer mapping.
    It returns unknown for missing endpoint mappings, multiple label chains, or
    duplicate numbering. Domain segments remain separate and discontinuous.
    """
    block = gemmi.cif.read_file(str(mmcif_path)).sole_block()
    table = block.find_mmcif_category("_pdbx_poly_seq_scheme.")
    endpoints: dict[tuple[int, str], int] = {}
    labels: set[str] = set()
    positions: set[int] = set()
    if not table:
        return None
    for row in table:
        def value(field: str, row: Any = row) -> str:
            try:
                text = str(row[f"_pdbx_poly_seq_scheme.{field}"])
            except (KeyError, IndexError, RuntimeError):
                return ""
            return "" if text in {".", "?"} else text
        if value("pdb_strand_id") != author_chain:
            continue
        try:
            position = int(value("seq_id"))
            author_number = int(value("pdb_seq_num"))
        except ValueError:
            return None
        key = (author_number, value("pdb_ins_code"))
        label = value("asym_id")
        if not label or key in endpoints or position in positions or not 1 <= position <= length:
            return None
        labels.add(label)
        positions.add(position)
        endpoints[key] = position
    if len(labels) != 1 or positions != set(range(1, length + 1)):
        return None
    out = []
    for segment in author_segments:
        start = endpoints.get((segment["start_number"], segment["start_insertion_code"]))
        end = endpoints.get((segment["end_number"], segment["end_insertion_code"]))
        if start is None or end is None or start > end:
            return None
        out.append({"start": start, "end": end})
    return out if _positions(out, length) is not None else None
