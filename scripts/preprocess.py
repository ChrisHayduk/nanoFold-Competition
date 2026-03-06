from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
try:
    from Bio.Align import PairwiseAligner
except Exception:  # pragma: no cover - fallback path depends on local environment
    PairwiseAligner = None  # type: ignore[assignment]
from tqdm import tqdm

# Allow running as `python scripts/preprocess.py` from repo root (or via absolute script path).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofold.a3m import read_a3m
from nanofold.a3m import RESTYPE_TO_ID, UNK_ID
try:
    from nanofold.mmcif import extract_chain_ca, aatype_from_sequence
except Exception:  # pragma: no cover - depends on optional gemmi runtime
    extract_chain_ca = None  # type: ignore[assignment]

    def aatype_from_sequence(sequence: str) -> np.ndarray:
        return np.fromiter(
            (RESTYPE_TO_ID.get(aa.upper(), UNK_ID) for aa in sequence),
            dtype=np.int32,
            count=len(sequence),
        )

"""Preprocess raw OpenProteinSet/OpenFold data into compact per-chain .npz files.

Input assumptions (adjust as needed):
- Either:
  - raw_root/roda_pdb/<chain_id>/ contains subdirectories for alignment files (A3M/HHR/etc), or
  - alignments_root/<chain_id>/ contains flattened OpenFold alignments from flatten_roda.sh
- mmcif_root contains mmCIF files named like <pdb_id>.cif (lowercase)

Output:
- processed_features_dir/<chain_id>.npz containing:
    aatype: (L,) int32
    msa: (N,L) int32
    deletions: (N,L) int32
    template_aatype: (T,L) int32
    template_ca_coords: (T,L,3) float32
    template_ca_mask: (T,L) bool
- processed_labels_dir/<chain_id>.npz containing:
    ca_coords: (L,3) float32
    ca_mask: (L,) bool
"""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=str, default="data/raw", help="Root from scripts/prepare_data.py")
    ap.add_argument(
        "--alignments-root",
        type=str,
        default="",
        help="Optional flattened alignments dir (OpenFold layout): <root>/<chain_id>/...",
    )
    ap.add_argument("--mmcif-root", type=str, default="data/mmcif", help="Directory containing mmCIF files")
    ap.add_argument("--manifest", type=str, required=True, help="train.txt or val.txt listing chain IDs")
    ap.add_argument(
        "--processed-features-dir",
        type=str,
        default="data/processed_features",
        help="Output directory for feature .npz files",
    )
    ap.add_argument(
        "--processed-labels-dir",
        type=str,
        default="data/processed_labels",
        help="Output directory for label .npz files",
    )
    ap.add_argument("--msa-name", type=str, default="uniref90_hits.a3m", help="Which MSA file to use")
    ap.add_argument("--max-msa-seqs", type=int, default=2048, help="Cap raw MSA depth to keep files smaller")
    ap.add_argument("--template-hhr-name", type=str, default="pdb70_hits.hhr", help="Template hits file name under chain dir")
    ap.add_argument("--max-templates", type=int, default=1, help="Maximum number of template hits to include per chain")
    ap.add_argument("--disable-templates", action="store_true", help="Do not attempt to include template features")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Require perfect sequence match between structure and MSA query. Recommended for a clean benchmark.",
    )
    ap.add_argument("--min-projection-seq-identity", type=float, default=0.90)
    ap.add_argument("--min-projection-coverage", type=float, default=0.90)
    ap.add_argument("--min-projection-aligned-fraction", type=float, default=0.90)
    ap.add_argument("--min-projection-valid-ca", type=int, default=32)
    ap.add_argument("--fail-fast", action="store_true")
    return ap.parse_args()


def _empty_template_features(L: int) -> Dict[str, np.ndarray]:
    return {
        "template_aatype": np.zeros((0, L), dtype=np.int32),
        "template_ca_coords": np.zeros((0, L, 3), dtype=np.float32),
        "template_ca_mask": np.zeros((0, L), dtype=bool),
    }


@dataclass(frozen=True)
class HHRHit:
    pdb_id: str
    chain_id: str
    query_aligned: str
    template_aligned: str


def _query_column_mask(query_aligned: str) -> np.ndarray:
    return np.asarray([ch != "-" for ch in query_aligned], dtype=bool)


def _ungap_query_columns(
    *,
    msa: np.ndarray,
    deletions: np.ndarray,
    query_aligned: str,
) -> Tuple[np.ndarray, np.ndarray, str]:
    mask = _query_column_mask(query_aligned)
    if mask.ndim != 1 or int(mask.shape[0]) != int(msa.shape[1]):
        raise ValueError(
            f"Query alignment length mismatch: len(query_aligned)={len(query_aligned)} vs msa.shape[1]={msa.shape[1]}"
        )
    target_seq = "".join(ch for ch in query_aligned if ch != "-")
    return msa[:, mask], deletions[:, mask], target_seq


def _parse_hhr_token(token: str) -> Optional[Tuple[str, str]]:
    token = token.strip()
    m = re.match(r"^([0-9A-Za-z]{4})[_\-]([0-9A-Za-z])$", token)
    if m:
        return m.group(1).lower(), m.group(2)

    m = re.match(r"^([0-9A-Za-z]{4})([0-9A-Za-z])$", token)
    if m:
        return m.group(1).lower(), m.group(2)
    return None


def _candidate_templates_from_hhr(hhr_path: Path) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for ln in hhr_path.read_text(errors="ignore").splitlines():
        if not ln.startswith(">"):
            continue
        token = ln[1:].strip().split()[0]
        parsed = _parse_hhr_token(token)
        if parsed is not None:
            out.append(parsed)
    return out


def _parse_hhr_hits(hhr_path: Path) -> List[HHRHit]:
    hits: List[HHRHit] = []
    current_template: Tuple[str, str] | None = None
    query_fragments: List[str] = []
    template_fragments: List[str] = []

    def flush_current() -> None:
        nonlocal current_template, query_fragments, template_fragments
        if current_template is None:
            return
        query_aligned = "".join(query_fragments)
        template_aligned = "".join(template_fragments)
        if query_aligned and template_aligned and len(query_aligned) == len(template_aligned):
            hits.append(
                HHRHit(
                    pdb_id=current_template[0],
                    chain_id=current_template[1],
                    query_aligned=query_aligned,
                    template_aligned=template_aligned,
                )
            )
        current_template = None
        query_fragments = []
        template_fragments = []

    for raw_line in hhr_path.read_text(errors="ignore").splitlines():
        line = raw_line.rstrip()
        if line.startswith("No "):
            flush_current()
            continue
        if line.startswith(">"):
            flush_current()
            token = line[1:].strip().split()[0]
            current_template = _parse_hhr_token(token)
            continue
        if current_template is None:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        if parts[0] == "Q" and parts[1] == "query":
            query_fragments.append(parts[3].strip())
        elif parts[0] == "T" and parts[1] not in {"Consensus", "ss_dssp", "ss_pred", "ss_conf"}:
            template_fragments.append(parts[3].strip())

    flush_current()
    return hits


def _make_global_aligner():
    if PairwiseAligner is None:
        return None
    aligner = PairwiseAligner(mode="global")
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -4.0
    aligner.extend_gap_score = -1.0
    return aligner


_GLOBAL_ALIGNER = _make_global_aligner()


def _global_alignment_pairs(query_seq: str, template_seq: str) -> List[Tuple[int, int]]:
    if not query_seq or not template_seq:
        return []

    if _GLOBAL_ALIGNER is None:
        matcher = difflib.SequenceMatcher(a=query_seq, b=template_seq, autojunk=False)
        pairs: List[Tuple[int, int]] = []
        for block in matcher.get_matching_blocks():
            for off in range(block.size):
                pairs.append((block.a + off, block.b + off))
        return pairs

    try:
        aln = _GLOBAL_ALIGNER.align(query_seq, template_seq)[0]
    except Exception:
        return []

    q_segments, t_segments = aln.aligned
    pairs = []
    for (q_start, q_end), (t_start, t_end) in zip(q_segments, t_segments):
        seg_len = min(int(q_end) - int(q_start), int(t_end) - int(t_start))
        if seg_len <= 0:
            continue
        q0 = int(q_start)
        t0 = int(t_start)
        for off in range(seg_len):
            pairs.append((q0 + off, t0 + off))
    return pairs


def _pairs_from_aligned_strings(query_aligned: str, template_aligned: str) -> Tuple[List[Tuple[int, int]], int]:
    if len(query_aligned) != len(template_aligned):
        raise ValueError("Aligned query/template strings must have the same length.")
    pairs: List[Tuple[int, int]] = []
    matches = 0
    q_idx = 0
    t_idx = 0
    for q_char, t_char in zip(query_aligned, template_aligned):
        q_has = q_char != "-"
        t_has = t_char != "-"
        if q_has and t_has:
            pairs.append((q_idx, t_idx))
            if q_char.upper() == t_char.upper():
                matches += 1
        if q_has:
            q_idx += 1
        if t_has:
            t_idx += 1
    return pairs, matches


def _project_structure_to_query(
    query_seq: str,
    structure_seq: str,
    structure_ca_coords: np.ndarray,
    structure_ca_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Map structure coordinates onto query positions via global sequence alignment."""
    pairs = _global_alignment_pairs(query_seq=query_seq, template_seq=structure_seq)
    if not pairs:
        raise ValueError("Could not align structure sequence to query sequence")

    qL = len(query_seq)
    out_ca = np.zeros((qL, 3), dtype=np.float32)
    out_mask = np.zeros((qL,), dtype=bool)
    for qi, si in pairs:
        if qi < 0 or qi >= qL:
            continue
        if si < 0 or si >= len(structure_seq):
            continue
        if bool(structure_ca_mask[si]):
            out_ca[qi] = structure_ca_coords[si]
            out_mask[qi] = True
    aligned_pairs = len(pairs)
    matches = sum(1 for qi, si in pairs if query_seq[qi].upper() == structure_seq[si].upper())
    stats = {
        "projection_seq_identity": (matches / float(aligned_pairs)) if aligned_pairs > 0 else 0.0,
        "projection_alignment_coverage": (aligned_pairs / float(len(query_seq))) if query_seq else 0.0,
        "projection_aligned_fraction": (
            aligned_pairs / float(max(len(query_seq), len(structure_seq)))
        ) if max(len(query_seq), len(structure_seq)) > 0 else 0.0,
        "projection_valid_ca_count": float(int(out_mask.sum())),
    }
    return out_ca, out_mask, stats


def _extract_template_features(
    chain_dir: Path,
    mmcif_root: Path,
    target_pdb_id: str,
    target_chain_id: str,
    target_seq: str,
    target_L: int,
    template_hhr_name: str,
    max_templates: int,
) -> Dict[str, np.ndarray]:
    if extract_chain_ca is None:
        raise ImportError("Template extraction requires optional dependency `gemmi`.")
    if max_templates <= 0:
        return _empty_template_features(target_L)

    hhr_path = None
    for p in chain_dir.rglob(template_hhr_name):
        hhr_path = p
        break
    if hhr_path is None:
        return _empty_template_features(target_L)

    hits = _parse_hhr_hits(hhr_path)
    if not hits:
        return _empty_template_features(target_L)

    template_aatype_list: List[np.ndarray] = []
    template_ca_list: List[np.ndarray] = []
    template_mask_list: List[np.ndarray] = []

    for hit in hits:
        if hit.pdb_id == target_pdb_id and hit.chain_id == target_chain_id:
            continue
        query_seq_from_hhr = hit.query_aligned.replace("-", "")
        template_seq_from_hhr = hit.template_aligned.replace("-", "")
        if query_seq_from_hhr.upper() != target_seq.upper():
            continue

        mmcif_path = mmcif_root / f"{hit.pdb_id}.cif"
        if not mmcif_path.exists():
            continue

        try:
            tpl = extract_chain_ca(
                mmcif_path=mmcif_path,
                pdb_id=hit.pdb_id,
                chain_id=hit.chain_id,
                expected_sequence=None,
                require_full_match=False,
            )
        except Exception:
            continue

        if template_seq_from_hhr.upper() != tpl.sequence.upper():
            continue

        pairs, _matches = _pairs_from_aligned_strings(hit.query_aligned, hit.template_aligned)
        if not pairs:
            continue

        t_aatype = np.full((target_L,), fill_value=20, dtype=np.int32)
        t_ca = np.zeros((target_L, 3), dtype=np.float32)
        t_mask = np.zeros((target_L,), dtype=bool)

        tpl_aatype = aatype_from_sequence(tpl.sequence)
        for qi, ti in pairs:
            if qi < 0 or qi >= target_L:
                continue
            if ti < 0 or ti >= len(tpl.sequence):
                continue
            t_aatype[qi] = tpl_aatype[ti]
            if bool(tpl.ca_mask[ti]):
                t_ca[qi] = tpl.ca_coords[ti]
                t_mask[qi] = True

        if int(t_mask.sum()) < 5:
            continue

        template_aatype_list.append(t_aatype)
        template_ca_list.append(t_ca)
        template_mask_list.append(t_mask)

        if len(template_aatype_list) >= max_templates:
            break

    if not template_aatype_list:
        return _empty_template_features(target_L)

    return {
        "template_aatype": np.stack(template_aatype_list, axis=0).astype(np.int32),
        "template_ca_coords": np.stack(template_ca_list, axis=0).astype(np.float32),
        "template_ca_mask": np.stack(template_mask_list, axis=0).astype(bool),
    }


def main() -> None:
    args = parse_args()
    if extract_chain_ca is None:
        raise ImportError("scripts/preprocess.py requires optional dependency `gemmi` to read mmCIF files.")

    raw_root = Path(args.raw_root)
    alignments_root = Path(args.alignments_root) if args.alignments_root else None
    mmcif_root = Path(args.mmcif_root)
    processed_features_dir = Path(args.processed_features_dir)
    processed_labels_dir = Path(args.processed_labels_dir)
    processed_features_dir.mkdir(parents=True, exist_ok=True)
    processed_labels_dir.mkdir(parents=True, exist_ok=True)

    manifest = Path(args.manifest)
    chain_ids = [ln.strip() for ln in manifest.read_text().splitlines() if ln.strip() and not ln.startswith("#")]

    n_ok = 0
    n_fail = 0

    for cid in tqdm(chain_ids, desc="preprocess"):
        try:
            pdb_id, chain_id = cid.split("_")
            pdb_id = pdb_id.lower()

            # Resolve alignment path:
            # - OpenFold flattened layout: <alignments_root>/<chain_id>/
            # - Legacy layout from scripts/prepare_data.py: <raw_root>/roda_pdb/<chain_id>/
            chain_dir = (alignments_root / cid) if alignments_root is not None else (raw_root / "roda_pdb" / cid)
            if not chain_dir.exists():
                raise FileNotFoundError(f"Missing alignment directory: {chain_dir}")
            # Find the msa file somewhere under chain_dir/*
            msa_path = None
            for p in chain_dir.rglob(args.msa_name):
                msa_path = p
                break
            if msa_path is None:
                raise FileNotFoundError(f"Could not find {args.msa_name} under {chain_dir}")

            mmcif_path = mmcif_root / f"{pdb_id}.cif"
            if not mmcif_path.exists():
                raise FileNotFoundError(f"Missing mmCIF: {mmcif_path}")

            a3m = read_a3m(msa_path)
            msa, deletions = a3m.to_tokens(max_seqs=args.max_msa_seqs)
            aligned_msa, _ = a3m.to_aligned_msa()
            query_aligned = aligned_msa[0]
            msa, deletions, query_sequence = _ungap_query_columns(
                msa=msa,
                deletions=deletions,
                query_aligned=query_aligned,
            )
            chain_struct = extract_chain_ca(
                mmcif_path=mmcif_path,
                pdb_id=pdb_id,
                chain_id=chain_id,
                expected_sequence=query_sequence if args.strict else None,
                require_full_match=args.strict,
            )

            if args.strict:
                if len(chain_struct.sequence) != msa.shape[1]:
                    raise ValueError(
                        f"Length mismatch for {cid}: msa L={msa.shape[1]} vs structure L={len(chain_struct.sequence)}"
                    )
                target_seq = chain_struct.sequence
                target_ca_coords = chain_struct.ca_coords.astype(np.float32)
                target_ca_mask = chain_struct.ca_mask.astype(bool)
                projection_stats = {
                    "projection_seq_identity": 1.0,
                    "projection_alignment_coverage": 1.0,
                    "projection_aligned_fraction": 1.0,
                    "projection_valid_ca_count": float(int(target_ca_mask.sum())),
                }
            else:
                target_seq = query_sequence
                target_ca_coords, target_ca_mask, projection_stats = _project_structure_to_query(
                    query_seq=query_sequence,
                    structure_seq=chain_struct.sequence,
                    structure_ca_coords=chain_struct.ca_coords,
                    structure_ca_mask=chain_struct.ca_mask,
                )
                if projection_stats["projection_seq_identity"] < float(args.min_projection_seq_identity):
                    raise ValueError(f"Projection seq identity below threshold for {cid}: {projection_stats}")
                if projection_stats["projection_alignment_coverage"] < float(args.min_projection_coverage):
                    raise ValueError(f"Projection coverage below threshold for {cid}: {projection_stats}")
                if projection_stats["projection_aligned_fraction"] < float(args.min_projection_aligned_fraction):
                    raise ValueError(f"Projection aligned fraction below threshold for {cid}: {projection_stats}")
                if int(projection_stats["projection_valid_ca_count"]) < int(args.min_projection_valid_ca):
                    raise ValueError(f"Projection valid C-alpha count below threshold for {cid}: {projection_stats}")

            features_out = {
                "aatype": aatype_from_sequence(target_seq),
                "msa": msa.astype(np.int32),
                "deletions": deletions.astype(np.int32),
                "projection_seq_identity": np.asarray(projection_stats["projection_seq_identity"], dtype=np.float32),
                "projection_alignment_coverage": np.asarray(
                    projection_stats["projection_alignment_coverage"], dtype=np.float32
                ),
                "projection_aligned_fraction": np.asarray(
                    projection_stats["projection_aligned_fraction"], dtype=np.float32
                ),
                "projection_valid_ca_count": np.asarray(
                    int(projection_stats["projection_valid_ca_count"]), dtype=np.int32
                ),
            }
            labels_out = {
                "ca_coords": target_ca_coords.astype(np.float32),
                "ca_mask": target_ca_mask.astype(bool),
            }
            if len(target_seq) != int(msa.shape[1]):
                raise ValueError(f"Post-processed sequence/MSA length mismatch for {cid}: {len(target_seq)} vs {msa.shape[1]}")
            if labels_out["ca_coords"].shape[0] != len(target_seq) or labels_out["ca_mask"].shape[0] != len(target_seq):
                raise ValueError(f"Label length mismatch for {cid} after preprocessing.")

            if args.disable_templates:
                features_out.update(_empty_template_features(len(target_seq)))
            else:
                tpl = _extract_template_features(
                    chain_dir=chain_dir,
                    mmcif_root=mmcif_root,
                    target_pdb_id=pdb_id,
                    target_chain_id=chain_id,
                    target_seq=target_seq,
                    target_L=len(target_seq),
                    template_hhr_name=args.template_hhr_name,
                    max_templates=int(args.max_templates),
                )
                features_out.update(tpl)

            np.savez_compressed(processed_features_dir / f"{cid}.npz", **features_out)
            np.savez_compressed(processed_labels_dir / f"{cid}.npz", **labels_out)
            err_marker = processed_features_dir / f"{cid}.error.txt"
            if err_marker.exists():
                err_marker.unlink()
            n_ok += 1
        except Exception as e:
            n_fail += 1
            if args.fail_fast:
                raise
            # Write an error marker file for debugging
            (processed_features_dir / f"{cid}.error.txt").write_text(str(e) + "\n")
            continue

    print(f"Preprocess complete: ok={n_ok}, fail={n_fail}")
    if n_fail > 0:
        print("Inspect *.error.txt files in", processed_features_dir)


if __name__ == "__main__":
    main()
