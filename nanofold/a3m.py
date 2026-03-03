from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

# 20 standard AAs (order doesn't matter as long as consistent everywhere).
RESTYPES = "ARNDCQEGHILKMFPSTWYV"
RESTYPE_TO_ID = {aa: i for i, aa in enumerate(RESTYPES)}

# Extra tokens for MSA processing
UNK_ID = 20          # unknown / non-standard AA
GAP_ID = 21          # '-' in aligned MSA
MASK_ID = 22         # used if you do masked-MSA training

MSA_ALPHABET_SIZE = 23  # 20 + UNK + GAP + MASK
SEQ_ALPHABET_SIZE = 21  # 20 + UNK


def aa_to_id(aa: str) -> int:
    aa = aa.upper()
    if aa == "-":
        return GAP_ID
    return RESTYPE_TO_ID.get(aa, UNK_ID)


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
