"""A3M multiple-sequence-alignment parsing and amino-acid tokenisation.

A3M is HHsuite's FASTA-like MSA format. Each record's sequence encodes one
row of an alignment with three character classes:

* **Uppercase letters** are aligned match states (columns of the MSA).
* **Lowercase letters** are insertions relative to the query — they carry
  information but do not occupy an alignment column. The count of lowercase
  characters preceding each match state becomes the ``deletion`` feature
  (AF2 supplement 1.2.9, Table 1: ``cluster_has_deletion`` /
  ``cluster_deletion_value``).
* **Dashes** are deletions relative to the query: they occupy an aligned
  column but carry no residue.

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

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

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

    @property
    def n_seqs(self) -> int:
        return len(self.seqs_raw)

    def to_aligned_msa(self) -> Tuple[List[str], np.ndarray]:
        """Convert A3M to an aligned MSA (uppercase + gaps), and a deletion matrix.

        A3M convention:
        - lowercase letters are insertions relative to the query alignment; these are *removed* when producing
          an aligned MSA.
        - the deletion matrix at position i stores the number of insertions encountered since the previous
          aligned position (AlphaFold's 'deletion_matrix_int').

        Returns:
          aligned_seqs: list of length N, each string length L
          deletions: int32 array of shape (N, L)
        """
        aligned: List[str] = []
        deletions_list: List[List[int]] = []

        for s in self.seqs_raw:
            del_counts: List[int] = []
            out_chars: List[str] = []
            cur_del = 0
            for ch in s:
                if ch.islower():
                    cur_del += 1
                    continue
                # uppercase or gap => aligned position
                out_chars.append(ch.upper())
                del_counts.append(cur_del)
                cur_del = 0
            aligned_seq = "".join(out_chars)
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
    path = Path(path)
    headers: List[str] = []
    seqs: List[str] = []

    cur_header: str | None = None
    cur_seq_parts: List[str] = []

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):

            if cur_header is not None:
                headers.append(cur_header)
                seqs.append("".join(cur_seq_parts))
            cur_header = line[1:]
            cur_seq_parts = []
        else:
            cur_seq_parts.append(line)

    if cur_header is not None:
        headers.append(cur_header)
        seqs.append("".join(cur_seq_parts))

    if not seqs:
        raise ValueError(f"No sequences found in {path}")

    return A3M(headers=headers, seqs_raw=seqs)
