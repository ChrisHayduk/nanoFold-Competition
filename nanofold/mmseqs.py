"""Validate the alignment contract shared by dataset homology searches."""

from __future__ import annotations

import math
from collections.abc import Iterator, Set
from pathlib import Path


def validate_search_parameters(*, min_seq_id: float, coverage: float, threads: int, max_seqs: int) -> None:
    if not 0 < min_seq_id <= 1 or not 0 < coverage <= 1:
        raise ValueError("MMseqs identity and coverage thresholds must be in (0, 1]")
    if threads < 1 or max_seqs < 1:
        raise ValueError("MMseqs threads and max-seqs must be positive")


def iter_mmseqs_hits(
    path: Path,
    *,
    query_ids: Set[str],
    target_ids: Set[str],
    min_seq_id: float,
    coverage: float,
) -> Iterator[tuple[str, str]]:
    """Read known query/target pairs from validated eight-column MMseqs output.

    Searches request pident in percent and qcov/tcov as fractions. A tolerance
    of 0.001 in fractional units accommodates MMseqs' printed-value rounding.
    The caller must use alignment mode 3, sequence identity mode 0, and coverage
    mode 0 so these checks have the same meaning as the search thresholds.
    """
    if not path.is_file():
        raise RuntimeError("MMseqs completed without writing the expected alignment output")
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            columns = line.rstrip("\r\n").split("\t")
            if len(columns) != 8:
                raise ValueError(f"MMseqs alignment row {line_number} must contain eight columns")
            query, target = columns[:2]
            if query not in query_ids or target not in target_ids:
                raise ValueError(f"MMseqs alignment row {line_number} references an unknown sequence")
            try:
                identity = float(columns[2]) / 100
                alignment_length = int(columns[3])
                query_coverage, target_coverage, evalue, bits = map(float, columns[4:])
            except ValueError as exc:
                raise ValueError(f"MMseqs alignment row {line_number} contains invalid numeric fields") from exc
            values = (identity, query_coverage, target_coverage, evalue, bits)
            if (
                not all(math.isfinite(value) for value in values)
                or alignment_length < 1
                or not 0 <= identity <= 1
                or not 0 <= query_coverage <= 1
                or not 0 <= target_coverage <= 1
                or evalue < 0
                or bits < 0
            ):
                raise ValueError(f"MMseqs alignment row {line_number} contains out-of-range numeric fields")
            if identity + 0.001 < min_seq_id or min(query_coverage, target_coverage) + 0.001 < coverage:
                raise ValueError(f"MMseqs alignment row {line_number} does not meet the requested thresholds")
            yield query, target
