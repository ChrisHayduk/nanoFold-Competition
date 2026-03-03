from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from Bio import pairwise2
from tqdm import tqdm

from nanofold.a3m import read_a3m
from nanofold.mmcif import extract_chain_ca, aatype_from_sequence

"""Preprocess raw OpenProteinSet/OpenFold data into compact per-chain .npz files.

Input assumptions (adjust as needed):
- Either:
  - raw_root/roda_pdb/<chain_id>/ contains subdirectories for alignment files (A3M/HHR/etc), or
  - alignments_root/<chain_id>/ contains flattened OpenFold alignments from flatten_roda.sh
- mmcif_root contains mmCIF files named like <pdb_id>.cif (lowercase)

Output:
- processed_dir/<chain_id>.npz containing:
    aatype: (L,) int32
    msa: (N,L) int32
    deletions: (N,L) int32
    ca_coords: (L,3) float32
    ca_mask: (L,) bool
    template_aatype: (T,L) int32
    template_ca_coords: (T,L,3) float32
    template_ca_mask: (T,L) bool
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
    ap.add_argument("--processed-dir", type=str, default="data/processed", help="Output directory")
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
    ap.add_argument("--fail-fast", action="store_true")
    return ap.parse_args()


def _empty_template_features(L: int) -> Dict[str, np.ndarray]:
    return {
        "template_aatype": np.zeros((0, L), dtype=np.int32),
        "template_ca_coords": np.zeros((0, L, 3), dtype=np.float32),
        "template_ca_mask": np.zeros((0, L), dtype=bool),
    }


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


def _global_alignment_pairs(query_seq: str, template_seq: str) -> List[Tuple[int, int]]:
    aln = pairwise2.align.globalms(query_seq, template_seq, 2.0, -1.0, -4.0, -1.0, one_alignment_only=True)
    if not aln:
        return []

    q_aln = aln[0].seqA
    t_aln = aln[0].seqB

    pairs: List[Tuple[int, int]] = []
    qi = 0
    ti = 0
    for qch, tch in zip(q_aln, t_aln):
        q_has = qch != "-"
        t_has = tch != "-"
        if q_has and t_has:
            pairs.append((qi, ti))
        if q_has:
            qi += 1
        if t_has:
            ti += 1
    return pairs


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
    if max_templates <= 0:
        return _empty_template_features(target_L)

    hhr_path = None
    for p in chain_dir.rglob(template_hhr_name):
        hhr_path = p
        break
    if hhr_path is None:
        return _empty_template_features(target_L)

    candidates = _candidate_templates_from_hhr(hhr_path)
    if not candidates:
        return _empty_template_features(target_L)

    template_aatype_list: List[np.ndarray] = []
    template_ca_list: List[np.ndarray] = []
    template_mask_list: List[np.ndarray] = []

    for pdb_id, chain_id in candidates:
        if pdb_id == target_pdb_id and chain_id == target_chain_id:
            continue
        mmcif_path = mmcif_root / f"{pdb_id}.cif"
        if not mmcif_path.exists():
            continue

        try:
            tpl = extract_chain_ca(
                mmcif_path=mmcif_path,
                pdb_id=pdb_id,
                chain_id=chain_id,
                expected_sequence=None,
                require_full_match=False,
            )
        except Exception:
            continue

        pairs = _global_alignment_pairs(query_seq=target_seq, template_seq=tpl.sequence)
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

    raw_root = Path(args.raw_root)
    alignments_root = Path(args.alignments_root) if args.alignments_root else None
    mmcif_root = Path(args.mmcif_root)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

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

            # query sequence is first aligned row with gaps removed
            query_aligned = "".join(ch for ch in a3m.to_aligned_msa()[0][0] if ch != "-")
            chain_struct = extract_chain_ca(
                mmcif_path=mmcif_path,
                pdb_id=pdb_id,
                chain_id=chain_id,
                expected_sequence=query_aligned if args.strict else None,
                require_full_match=args.strict,
            )

            if len(chain_struct.sequence) != msa.shape[1]:
                # MSA aligned length should match structure-derived length if strict & no missing residues.
                # In practice, you may need alignment; for the benchmark, we recommend filtering out mismatches.
                raise ValueError(
                    f"Length mismatch for {cid}: msa L={msa.shape[1]} vs structure L={len(chain_struct.sequence)}"
                )

            out = {
                "aatype": aatype_from_sequence(chain_struct.sequence),
                "msa": msa.astype(np.int32),
                "deletions": deletions.astype(np.int32),
                "ca_coords": chain_struct.ca_coords.astype(np.float32),
                "ca_mask": chain_struct.ca_mask.astype(bool),
            }

            if args.disable_templates:
                out.update(_empty_template_features(len(chain_struct.sequence)))
            else:
                tpl = _extract_template_features(
                    chain_dir=chain_dir,
                    mmcif_root=mmcif_root,
                    target_pdb_id=pdb_id,
                    target_chain_id=chain_id,
                    target_seq=chain_struct.sequence,
                    target_L=len(chain_struct.sequence),
                    template_hhr_name=args.template_hhr_name,
                    max_templates=int(args.max_templates),
                )
                out.update(tpl)

            np.savez_compressed(processed_dir / f"{cid}.npz", **out)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            if args.fail_fast:
                raise
            # Write an error marker file for debugging
            (processed_dir / f"{cid}.error.txt").write_text(str(e) + "\n")
            continue

    print(f"Preprocess complete: ok={n_ok}, fail={n_fail}")
    if n_fail > 0:
        print("Inspect *.error.txt files in", processed_dir)


if __name__ == "__main__":
    main()
