"""Validate the alignment contract shared by dataset homology searches."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping, Sequence, Set
from pathlib import Path

import numpy as np

from nanofold.a3m import GAP_ID, RESTYPES, UNK_ID


def validate_msa_arrays(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """Validate the unmasked, query-aligned MSA consumed by homology audits."""
    if "msa" not in arrays or "aatype" not in arrays:
        raise ValueError("Processed MSA auditing requires msa and aatype arrays")
    msa = np.asarray(arrays["msa"])
    query = np.asarray(arrays["aatype"])
    if msa.ndim != 2 or not all(msa.shape):
        raise ValueError("Processed MSA must be a nonempty two-dimensional array")
    if not np.issubdtype(msa.dtype, np.integer) or np.any(msa < 0) or np.any(msa > GAP_ID):
        raise ValueError("Processed MSA must contain integer residue/gap tokens without masks")
    if query.ndim != 1 or not np.issubdtype(query.dtype, np.integer):
        raise ValueError("Processed query must be a one-dimensional integer array")
    if np.any(query < 0) or np.any(query > UNK_ID):
        raise ValueError("Processed query must contain residue tokens without gaps or masks")
    if not np.array_equal(msa[0], query):
        raise ValueError("Processed MSA query row does not match aatype")
    if "deletions" in arrays:
        deletions = np.asarray(arrays["deletions"])
        if deletions.shape != msa.shape or not np.issubdtype(deletions.dtype, np.integer) or np.any(deletions < 0):
            raise ValueError("Processed deletion counts must be nonnegative integers matching the MSA")
    return msa


def load_msa_for_audit(path: Path, *, chain_id: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        msa = validate_msa_arrays(data)
        if "chain_id" in data.files:
            identity = np.asarray(data["chain_id"])
            if identity.shape != () or str(identity.item()) != chain_id:
                raise ValueError("Processed feature chain identity does not match its manifest")
        return msa


def canonical_target_sequence(sequence: str) -> str:
    """Match preprocessing's unknown-residue normalization without accepting FASTA syntax."""
    normalized = sequence.strip().upper()
    if not normalized or any(char not in RESTYPES + "XBZJUO" for char in normalized):
        raise ValueError("Target sequence must be a nonempty ungapped amino-acid sequence")
    return "".join(char if char in RESTYPES else "X" for char in normalized)


def sequence_records_sha256(records: Sequence[tuple[str, str]]) -> str:
    """Bind anonymous record indices to their ordered sequence contents."""
    rows = [(header, hashlib.sha256(sequence.encode()).hexdigest()) for header, sequence in records]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def validate_search_parameters(*, min_seq_id: float, coverage: float, threads: int, max_seqs: int) -> None:
    if not 0 < min_seq_id <= 1 or not 0 < coverage <= 1:
        raise ValueError("MMseqs identity and coverage thresholds must be in (0, 1]")
    # MMseqs parses these CLI counts with atoi, including size_t max-seqs.
    # Exhaustive searches also reuse max-seqs for max-accept/max-rejected.
    if not 1 <= threads <= 2_147_483_647 or not 1 <= max_seqs <= 2_147_483_647:
        raise ValueError("MMseqs threads and max-seqs must be in [1, 2147483647]")


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
