"""A3M multiple-sequence-alignment parsing and amino-acid tokenisation.

A3M is HHsuite's FASTA-like MSA format. Each record's sequence encodes one
row of an alignment with four character classes:

* **Uppercase letters** are aligned match states (columns of the MSA).
* **Lowercase letters** are insertions relative to the query — they carry
  information but do not occupy an alignment column. The count of lowercase
  characters preceding each match state becomes the ``deletion`` feature
  (AF2 supplement 1.2.9, Table 1: ``cluster_has_deletion`` /
  ``cluster_deletion_value``).
* **Dashes** are deletions relative to the query: they occupy an aligned
  column but carry no residue.
* **Dots** are optional gaps aligned to insertions: they are ignored and
  contribute neither an aligned column nor a deletion count.

This file converts raw A3M text into the two arrays the rest of the pipeline
needs: a ``(N_seq, N_res)`` integer MSA and a matching ``(N_seq, N_res)``
deletion-count array.

Alphabet conventions (AF2 supplement 1.9.9):

* ``target_feat`` uses 21 classes — 20 amino acids + unknown — matching
  ``SEQ_ALPHABET_SIZE`` and Table 1 ``aatype``.
* MSA features use 23 classes — 20 amino acids + unknown + gap + mask token
  — matching ``MSA_ALPHABET_SIZE`` and Table 1 ``cluster_msa`` /
  ``extra_msa``.

The one-letter ordering in ``RESTYPES`` matches DeepMind's canonical AF2
alphabet (A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np

# Canonical AF2 one-letter alphabet (20 standard amino acids).
RESTYPES = "ARNDCQEGHILKMFPSTWYV"
RESTYPE_TO_ID = {aa: i for i, aa in enumerate(RESTYPES)}

# Fixed IDs for non-standard tokens.
UNK_ID = 20   # any letter outside RESTYPES
GAP_ID = 21   # alignment gap '-'
MASK_ID = 22  # BERT-style mask token used by any masked-MSA training path

# Alphabet sizes used by feature builders (Table 1).
SEQ_ALPHABET_SIZE = 21  # target_feat: 20 AAs + unknown
MSA_ALPHABET_SIZE = 23  # cluster_msa / extra_msa: + gap + mask


def _validate_record(header: str, sequence: str) -> None:
    if not header.strip():
        raise ValueError("A3M record has an empty header")
    if not sequence:
        raise ValueError(f"A3M record {header!r} has an empty sequence")
    if re.fullmatch(r"[A-Za-z.\-]+", sequence) is None:
        raise ValueError(f"A3M record {header!r} contains invalid sequence characters")


def _aligned_row(sequence: str) -> Tuple[str, np.ndarray]:
    characters: List[str] = []
    deletions: List[int] = []
    insertions = 0
    for char in sequence:
        if char == ".":
            continue
        if char.islower():
            insertions += 1
            continue
        characters.append(char)
        deletions.append(insertions)
        insertions = 0
    return "".join(characters), np.asarray(deletions, dtype=np.int32)


def _iter_a3m_records(path: Path) -> Iterator[Tuple[str, str]]:
    header: str | None = None
    parts: List[str] = []
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    sequence = "".join(parts)
                    _validate_record(header, sequence)
                    yield header, sequence
                header = line[1:]
                parts = []
            else:
                if header is None:
                    raise ValueError(f"A3M sequence content occurs before the first header in {path}")
                parts.append(line)
    if header is None:
        raise ValueError(f"No sequences found in {path}")
    sequence = "".join(parts)
    _validate_record(header, sequence)
    yield header, sequence


def iter_a3m_rows(path: str | Path) -> Iterator[Tuple[str, str, np.ndarray]]:
    """Yield header, aligned sequence, and deletion counts using one-row memory.

    Every row must match the query's aligned width. Consumers must exhaust the
    iterator even when retaining a capped MSA, so malformed trailing rows fail.
    """
    width: int | None = None
    for header, sequence in _iter_a3m_records(Path(path)):
        aligned, deletions = _aligned_row(sequence)
        if width is None:
            if not aligned.replace("-", ""):
                raise ValueError("A3M query must contain at least one aligned residue")
            width = len(aligned)
        elif len(aligned) != width:
            raise ValueError(f"A3M parse produced inconsistent aligned lengths: expected {width}, got {len(aligned)}")
        yield header, aligned, deletions


def aa_to_id(aa: str) -> int:
    """Map a single character (one-letter AA code, '-', or unknown) to an ID."""
    aa = aa.upper()
    if aa == "-":
        return GAP_ID
    return RESTYPE_TO_ID.get(aa, UNK_ID)


def sequence_to_ids(sequence: str) -> np.ndarray:
    """Tokenise an ungapped sequence to an int32 array of AA IDs."""
    return np.fromiter(
        (aa_to_id(aa) for aa in sequence),
        dtype=np.int32,
        count=len(sequence),
    )


def ungap_query_columns(
    msa: np.ndarray,
    deletions: np.ndarray,
    query_aligned: str,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Drop columns where the query has a gap, returning the ungapped target.

    In an A3M alignment the query (first row) can contain dashes when other
    rows inserted residues that couldn't be absorbed as lowercase
    insertions. These gap columns are not part of the original target
    sequence, so we strip them before producing the MSA / deletion arrays
    consumed by the model. Returns ``(msa_without_gap_cols, deletions_without_gap_cols,
    target_sequence)``.
    """
    query_mask = np.asarray([char != "-" for char in query_aligned], dtype=bool)
    if query_mask.shape[0] != msa.shape[1]:
        raise ValueError(
            f"Aligned query length {query_mask.shape[0]} does not match MSA width {msa.shape[1]}."
        )
    target_sequence = "".join(char for char in query_aligned if char != "-")
    return msa[:, query_mask], deletions[:, query_mask], target_sequence


@dataclass
class A3M:
    headers: List[str]
    seqs_raw: List[str]

    def __post_init__(self) -> None:
        if not self.seqs_raw or len(self.headers) != len(self.seqs_raw):
            raise ValueError("A3M requires one header per nonempty sequence record")
        for header, sequence in zip(self.headers, self.seqs_raw, strict=True):
            _validate_record(header, sequence)
        if not any("A" <= char <= "Z" for char in self.seqs_raw[0]):
            raise ValueError("A3M query must contain at least one aligned residue")

    @property
    def n_seqs(self) -> int:
        return len(self.seqs_raw)

    def to_aligned_msa(self) -> Tuple[List[str], np.ndarray]:
        """Convert A3M to an aligned MSA (uppercase + gaps), and a deletion matrix.

        A3M convention:
        - lowercase letters are insertions relative to the query alignment; these are *removed* when producing
          an aligned MSA.
        - dots are gaps aligned to insertions and are ignored without increasing insertion counts.
        - the deletion matrix at position i stores the number of insertions encountered since the previous
          aligned position (AlphaFold's 'deletion_matrix_int').

        Returns:
          aligned_seqs: list of length N, each string length L
          deletions: int32 array of shape (N, L)
        """
        aligned: List[str] = []
        deletions_list: List[np.ndarray] = []

        for s in self.seqs_raw:
            aligned_seq, del_counts = _aligned_row(s)
            aligned.append(aligned_seq)
            deletions_list.append(del_counts)

        # Validate all aligned seqs same length
        Ls = {len(s) for s in aligned}
        if len(Ls) != 1:
            raise ValueError(f"A3M parse produced inconsistent aligned lengths: {sorted(Ls)}")

        deletions = np.asarray(deletions_list, dtype=np.int32)
        return aligned, deletions

    def to_tokens(self, max_seqs: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        aligned, deletions = self.to_aligned_msa()
        if max_seqs is not None:
            aligned = aligned[:max_seqs]
            deletions = deletions[:max_seqs]

        N = len(aligned)
        L = len(aligned[0])
        msa = np.zeros((N, L), dtype=np.int32)
        for i, seq in enumerate(aligned):
            msa[i] = np.fromiter((aa_to_id(ch) for ch in seq), count=L, dtype=np.int32)

        return msa, deletions


def read_a3m(path: str | Path) -> A3M:
    headers: List[str] = []
    seqs: List[str] = []
    for header, sequence in _iter_a3m_records(Path(path)):
        headers.append(header)
        seqs.append(sequence)
    return A3M(headers=headers, seqs_raw=seqs)
