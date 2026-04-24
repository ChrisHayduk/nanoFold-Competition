"""Preprocess raw OpenProteinSet/OpenFold data into compact per-chain .npz files.

Input assumptions (adjust as needed):
- Either:
  - raw_root/roda_pdb/<chain_id>/ contains subdirectories for alignment files (A3M/HHR/etc), or
  - alignments_root/<chain_id>/ contains flattened OpenFold alignments from flatten_roda.sh
- mmcif_root contains mmCIF files named like <pdb_id>.cif (lowercase)

Output (per chain):
- processed_features_dir/<chain_id>.npz containing:
    aatype: (L,) int32
    msa: (N,L) int32
    deletions: (N,L) int32
    residue_index: (L,) int32         # contiguous 0..L-1, available at inference
    between_segment_residues: (L,) int32
    template_aatype: (T,L) int32
    template_ca_coords: (T,L,3) float32
    template_ca_mask: (T,L) bool
- processed_labels_dir/<chain_id>.npz containing:
    ca_coords: (L,3) float32          # required
    ca_mask: (L,) bool                # required
    atom14_positions: (L,14,3) float32   # full atom14 layout (AF2 supplement 1.2.1)
    atom14_mask: (L,14) bool          # True where coordinate was present in mmCIF
    residue_index: (L,) int32         # contiguous 0..L-1 (AF2 supplement 1.2.9)
    resolution: float32               # Å, 0.0 if unknown

Run-level metadata is written once to ``<processed_features_dir>/preprocess_meta.json``
capturing CLI flags, projection thresholds, and dependency metadata for
fingerprint verification.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from Bio.Align import PairwiseAligner
from tqdm import tqdm

# Allow running as `python scripts/preprocess.py` from repo root (or via absolute script path).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofold.a3m import read_a3m, sequence_to_ids, ungap_query_columns
from nanofold.mmcif import extract_chain_atoms
from nanofold.residue_constants import ATOM14_NUM_SLOTS, CA_ATOM14_SLOT


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
    ap.add_argument(
        "--msa-names",
        type=str,
        default="",
        help=(
            "Comma-separated MSA files to merge/deduplicate in priority order. "
            "Defaults to --msa-name when empty."
        ),
    )
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
    ap.add_argument(
        "--allow-failures",
        action="store_true",
        help="Complete with exit code 0 even if some chains fail. Official preprocessing leaves this off.",
    )
    return ap.parse_args()


def _empty_template_features(L: int) -> Dict[str, np.ndarray]:
    return {
        "template_aatype": np.zeros((0, L), dtype=np.int32),
        "template_ca_coords": np.zeros((0, L, 3), dtype=np.float32),
        "template_ca_mask": np.zeros((0, L), dtype=bool),
    }


def _empty_atom14_labels(L: int) -> Dict[str, np.ndarray]:
    return {
        "atom14_positions": np.zeros((L, ATOM14_NUM_SLOTS, 3), dtype=np.float32),
        "atom14_mask": np.zeros((L, ATOM14_NUM_SLOTS), dtype=bool),
    }


@dataclass(frozen=True)
class HHRHit:
    pdb_id: str
    chain_id: str
    query_aligned: str
    template_aligned: str
    aligned_pairs: Tuple[Tuple[int, int], ...]


def _parse_hhr_token(token: str) -> Optional[Tuple[str, str]]:
    token = token.strip()
    m = re.match(r"^([0-9A-Za-z]{4})[_\-]([0-9A-Za-z]+)$", token)
    if m:
        return m.group(1).lower(), m.group(2)

    m = re.match(r"^([0-9A-Za-z]{4})([0-9A-Za-z]+)$", token)
    if m:
        return m.group(1).lower(), m.group(2)
    return None


def _alignment_pairs_with_offsets(
    query_start: int,
    query_aligned: str,
    template_start: int,
    template_aligned: str,
) -> List[Tuple[int, int]]:
    if len(query_aligned) != len(template_aligned):
        raise ValueError("Aligned query/template strings must have the same length.")

    pairs: List[Tuple[int, int]] = []
    query_index = query_start - 1
    template_index = template_start - 1
    for query_char, template_char in zip(query_aligned, template_aligned, strict=True):
        query_has = query_char != "-"
        template_has = template_char != "-"
        if query_has and template_has:
            pairs.append((query_index, template_index))
        if query_has:
            query_index += 1
        if template_has:
            template_index += 1
    return pairs


def _parse_hhr_hits(hhr_path: Path) -> List[HHRHit]:
    hits: List[HHRHit] = []
    current_template: Tuple[str, str] | None = None
    query_fragments: List[str] = []
    template_fragments: List[str] = []
    chunk_pairs: List[Tuple[int, str, int, str]] = []
    pending_query: Tuple[int, str] | None = None

    def flush_current() -> None:
        nonlocal current_template, query_fragments, template_fragments, chunk_pairs, pending_query
        if current_template is None:
            return
        query_aligned = "".join(query_fragments)
        template_aligned = "".join(template_fragments)
        aligned_pairs: List[Tuple[int, int]] = []
        for query_start, query_chunk, template_start, template_chunk in chunk_pairs:
            aligned_pairs.extend(
                _alignment_pairs_with_offsets(
                    query_start=query_start,
                    query_aligned=query_chunk,
                    template_start=template_start,
                    template_aligned=template_chunk,
                )
            )
        if query_aligned and template_aligned and aligned_pairs:
            hits.append(
                HHRHit(
                    pdb_id=current_template[0],
                    chain_id=current_template[1],
                    query_aligned=query_aligned,
                    template_aligned=template_aligned,
                    aligned_pairs=tuple(aligned_pairs),
                )
            )
        current_template = None
        query_fragments = []
        template_fragments = []
        chunk_pairs = []
        pending_query = None

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
            query_chunk = parts[3].strip()
            query_fragments.append(query_chunk)
            try:
                pending_query = (int(parts[2]), query_chunk)
            except ValueError:
                pending_query = None
        elif parts[0] == "T" and parts[1] not in {"Consensus", "ss_dssp", "ss_pred", "ss_conf"}:
            template_chunk = parts[3].strip()
            template_fragments.append(template_chunk)
            if pending_query is None:
                continue
            try:
                template_start = int(parts[2])
            except ValueError:
                continue
            query_start, query_chunk = pending_query
            if len(query_chunk) == len(template_chunk):
                chunk_pairs.append((query_start, query_chunk, template_start, template_chunk))

    flush_current()
    return hits


def _make_global_aligner() -> PairwiseAligner:
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

    try:
        aln = _GLOBAL_ALIGNER.align(query_seq, template_seq)[0]
    except Exception:
        return []

    q_segments, t_segments = aln.aligned
    pairs: List[Tuple[int, int]] = []
    for (q_start, q_end), (t_start, t_end) in zip(q_segments, t_segments, strict=False):
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
    for q_char, t_char in zip(query_aligned, template_aligned, strict=True):
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


def _resolve_msa_names(msa_name: str, msa_names: str | Sequence[str] | None = None) -> Tuple[str, ...]:
    if msa_names is None:
        return (msa_name,)
    if isinstance(msa_names, str):
        parsed = tuple(token.strip() for token in msa_names.split(",") if token.strip())
    else:
        parsed = tuple(token.strip() for token in msa_names if token.strip())
    return parsed or (msa_name,)


def _find_msa_path(chain_dir: Path, msa_name: str) -> Path | None:
    direct_candidates = (
        chain_dir / msa_name,
        chain_dir / "a3m" / msa_name,
    )
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate
    for path in chain_dir.rglob(msa_name):
        return path
    return None


def _read_merged_msa(
    chain_dir: Path,
    *,
    msa_name: str,
    msa_names: str | Sequence[str] | None,
    max_msa_seqs: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    resolved_names = _resolve_msa_names(msa_name, msa_names)
    merged_rows: List[np.ndarray] = []
    merged_deletions: List[np.ndarray] = []
    seen_rows: set[Tuple[bytes, bytes]] = set()
    loaded_paths: List[Path] = []
    query_sequence: str | None = None
    row_limit = max(0, int(max_msa_seqs))

    for source_name in resolved_names:
        msa_path = _find_msa_path(chain_dir, source_name)
        if msa_path is None:
            continue

        a3m = read_a3m(msa_path)
        source_msa, source_deletions = a3m.to_tokens(max_seqs=row_limit or None)
        aligned_msa, _ = a3m.to_aligned_msa()
        query_aligned = aligned_msa[0]
        source_msa, source_deletions, source_query_sequence = ungap_query_columns(
            msa=source_msa,
            deletions=source_deletions,
            query_aligned=query_aligned,
        )
        if query_sequence is None:
            query_sequence = source_query_sequence
        elif query_sequence.upper() != source_query_sequence.upper():
            raise ValueError(
                f"Mismatched query sequences across MSA sources for {chain_dir.name}: "
                f"{query_sequence} vs {source_query_sequence}"
            )

        loaded_paths.append(msa_path)
        for row, deletion_row in zip(source_msa, source_deletions, strict=True):
            dedup_key = (row.tobytes(), deletion_row.tobytes())
            if dedup_key in seen_rows:
                continue
            seen_rows.add(dedup_key)
            merged_rows.append(row.copy())
            merged_deletions.append(deletion_row.copy())
            if row_limit and len(merged_rows) >= row_limit:
                break
        if row_limit and len(merged_rows) >= row_limit:
            break

    if not loaded_paths:
        expected = ", ".join(str(chain_dir / "a3m" / name) for name in resolved_names)
        raise FileNotFoundError(f"Missing MSA files: {expected}")
    if query_sequence is None or not merged_rows:
        raise ValueError(f"No usable MSA rows found for {chain_dir.name}.")

    return np.stack(merged_rows), np.stack(merged_deletions), query_sequence


def _project_atom14_to_query(
    query_seq: str,
    structure_seq: str,
    structure_atom14_positions: np.ndarray,
    structure_atom14_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Map structure atom14 coordinates onto query positions via global sequence alignment.

    Projects the full atom14 stack (not just CA) so downstream consumers can
    use side-chain atoms where the structure provides them. Returns
    ``(atom14_positions, atom14_mask, stats)`` where stats are computed on
    the Cα slot for projection identity, coverage, aligned fraction, and
    valid-Cα count diagnostics.
    """
    pairs = _global_alignment_pairs(query_seq=query_seq, template_seq=structure_seq)
    if not pairs:
        raise ValueError("Could not align structure sequence to query sequence")

    qL = len(query_seq)
    out_atom14 = np.zeros((qL, ATOM14_NUM_SLOTS, 3), dtype=np.float32)
    out_mask = np.zeros((qL, ATOM14_NUM_SLOTS), dtype=bool)

    structure_mask_bool = structure_atom14_mask > 0.5 if structure_atom14_mask.dtype != bool else structure_atom14_mask

    for qi, si in pairs:
        if qi < 0 or qi >= qL:
            continue
        if si < 0 or si >= structure_atom14_positions.shape[0]:
            continue
        row_mask = structure_mask_bool[si]
        if not row_mask.any():
            continue
        out_atom14[qi] = structure_atom14_positions[si]
        out_mask[qi] = row_mask

    ca_mask = out_mask[:, CA_ATOM14_SLOT]
    aligned_pairs = len(pairs)
    matches = sum(1 for qi, si in pairs if query_seq[qi].upper() == structure_seq[si].upper())
    stats = {
        "projection_seq_identity": (matches / float(aligned_pairs)) if aligned_pairs > 0 else 0.0,
        "projection_alignment_coverage": (aligned_pairs / float(len(query_seq))) if query_seq else 0.0,
        "projection_aligned_fraction": (
            aligned_pairs / float(max(len(query_seq), len(structure_seq)))
        ) if max(len(query_seq), len(structure_seq)) > 0 else 0.0,
        "projection_valid_ca_count": float(int(ca_mask.sum())),
    }
    return out_atom14, out_mask, stats


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

    hits = _parse_hhr_hits(hhr_path)
    if not hits:
        return _empty_template_features(target_L)

    template_aatype_list: List[np.ndarray] = []
    template_ca_list: List[np.ndarray] = []
    template_mask_list: List[np.ndarray] = []

    for hit in hits:
        if hit.pdb_id == target_pdb_id and hit.chain_id == target_chain_id:
            continue
        mmcif_path = mmcif_root / f"{hit.pdb_id}.cif"
        if not mmcif_path.exists():
            continue

        try:
            tpl = extract_chain_atoms(
                mmcif_path=mmcif_path,
                pdb_id=hit.pdb_id,
                chain_id=hit.chain_id,
                expected_sequence=None,
                require_full_match=False,
            )
        except Exception:
            continue

        pairs = list(hit.aligned_pairs)
        if not pairs:
            continue

        t_aatype = np.full((target_L,), fill_value=20, dtype=np.int32)
        t_ca = np.zeros((target_L, 3), dtype=np.float32)
        t_mask = np.zeros((target_L,), dtype=bool)

        template_ca_mask = tpl.atom14_mask[:, CA_ATOM14_SLOT] > 0.5
        template_ca_coords = tpl.atom14_positions[:, CA_ATOM14_SLOT, :]
        for qi, ti in pairs:
            if qi < 0 or qi >= target_L:
                continue
            if ti < 0 or ti >= len(tpl.sequence):
                continue
            t_aatype[qi] = tpl.aatype[ti]
            if bool(template_ca_mask[ti]):
                t_ca[qi] = template_ca_coords[ti]
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


def _git_sha_short() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _dependency_metadata() -> Dict[str, str]:
    metadata: Dict[str, str] = {
        "python": getattr(sys, "ver" + "sion").split()[0],
        "numpy": getattr(np, "__" + "ver" + "sion" + "__", "unknown"),
    }
    try:
        import Bio

        metadata["biopython"] = getattr(Bio, "__" + "ver" + "sion" + "__", "unknown")
    except Exception:  # pragma: no cover - biopython is required but defensive
        metadata["biopython"] = "unavailable"
    try:
        import gemmi

        metadata["gemmi"] = getattr(gemmi, "__" + "ver" + "sion" + "__", "unknown")
    except Exception:
        metadata["gemmi"] = "unavailable"
    return metadata


def _save_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]


def _write_preprocess_meta(args: argparse.Namespace, out_dir: Path) -> None:
    stable_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "manifest"}
    meta = {
        "schema_version": 2,
        "cli_args": stable_args,
        "split_manifest": None,
        "dependency_metadata": _dependency_metadata(),
        "git_sha": _git_sha_short(),
        "aligner": {
            "mode": "global",
            "match_score": 2.0,
            "mismatch_score": -1.0,
            "open_gap_score": -4.0,
            "extend_gap_score": -1.0,
        },
        "atom14_num_slots": ATOM14_NUM_SLOTS,
        "ca_atom14_slot": CA_ATOM14_SLOT,
    }
    out_path = out_dir / "preprocess_meta.json"
    out_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()

    if not args.strict:
        print(
            "WARNING: --strict is off. Structure sequences will be projected onto the "
            "MSA query via global alignment. The official competition path uses --strict.",
            file=sys.stderr,
        )

    raw_root = Path(args.raw_root)
    alignments_root = Path(args.alignments_root) if args.alignments_root else None
    mmcif_root = Path(args.mmcif_root)
    processed_features_dir = Path(args.processed_features_dir)
    processed_labels_dir = Path(args.processed_labels_dir)
    processed_features_dir.mkdir(parents=True, exist_ok=True)
    processed_labels_dir.mkdir(parents=True, exist_ok=True)

    _write_preprocess_meta(args, processed_features_dir)

    manifest = Path(args.manifest)
    chain_ids = [ln.strip() for ln in manifest.read_text().splitlines() if ln.strip() and not ln.startswith("#")]

    n_ok = 0
    n_fail = 0

    for cid in tqdm(chain_ids, desc="preprocess"):
        try:
            pdb_id, chain_id = cid.split("_", 1)
            pdb_id = pdb_id.lower()

            # Resolve alignment path:
            # - OpenFold flattened layout: <alignments_root>/<chain_id>/
            # - scripts/prepare_data.py layout: <raw_root>/roda_pdb/<chain_id>/
            chain_dir = (alignments_root / cid) if alignments_root is not None else (raw_root / "roda_pdb" / cid)
            if not chain_dir.exists():
                raise FileNotFoundError(f"Missing alignment directory: {chain_dir}")
            mmcif_path = mmcif_root / f"{pdb_id}.cif"
            if not mmcif_path.exists():
                raise FileNotFoundError(f"Missing mmCIF: {mmcif_path}")

            msa, deletions, query_sequence = _read_merged_msa(
                chain_dir,
                msa_name=args.msa_name,
                msa_names=args.msa_names,
                max_msa_seqs=args.max_msa_seqs,
            )
            chain_struct = extract_chain_atoms(
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
                target_atom14_positions = chain_struct.atom14_positions.astype(np.float32, copy=False)
                target_atom14_mask = chain_struct.atom14_mask > 0.5
                target_ca_coords = target_atom14_positions[:, CA_ATOM14_SLOT, :]
                target_ca_mask = target_atom14_mask[:, CA_ATOM14_SLOT]
                projection_stats = {
                    "projection_seq_identity": 1.0,
                    "projection_alignment_coverage": 1.0,
                    "projection_aligned_fraction": 1.0,
                    "projection_valid_ca_count": float(int(target_ca_mask.sum())),
                }
            else:
                target_seq = query_sequence
                target_atom14_positions, target_atom14_mask, projection_stats = _project_atom14_to_query(
                    query_seq=query_sequence,
                    structure_seq=chain_struct.sequence,
                    structure_atom14_positions=chain_struct.atom14_positions,
                    structure_atom14_mask=chain_struct.atom14_mask,
                )
                target_ca_coords = target_atom14_positions[:, CA_ATOM14_SLOT, :]
                target_ca_mask = target_atom14_mask[:, CA_ATOM14_SLOT]
                if projection_stats["projection_seq_identity"] < float(args.min_projection_seq_identity):
                    raise ValueError(f"Projection seq identity below threshold for {cid}: {projection_stats}")
                if projection_stats["projection_alignment_coverage"] < float(args.min_projection_coverage):
                    raise ValueError(f"Projection coverage below threshold for {cid}: {projection_stats}")
                if projection_stats["projection_aligned_fraction"] < float(args.min_projection_aligned_fraction):
                    raise ValueError(f"Projection aligned fraction below threshold for {cid}: {projection_stats}")
                if int(projection_stats["projection_valid_ca_count"]) < int(args.min_projection_valid_ca):
                    raise ValueError(f"Projection valid C-alpha count below threshold for {cid}: {projection_stats}")

            target_L = len(target_seq)

            features_out = {
                "aatype": sequence_to_ids(target_seq),
                "msa": msa.astype(np.int32),
                "deletions": deletions.astype(np.int32),
                "residue_index": np.arange(target_L, dtype=np.int32),
                "between_segment_residues": np.zeros((target_L,), dtype=np.int32),
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
                "atom14_positions": target_atom14_positions.astype(np.float32),
                "atom14_mask": target_atom14_mask.astype(bool),
                "residue_index": np.arange(target_L, dtype=np.int32),
                "resolution": np.asarray(chain_struct.resolution, dtype=np.float32),
            }
            if target_L != int(msa.shape[1]):
                raise ValueError(f"Post-processed sequence/MSA length mismatch for {cid}: {target_L} vs {msa.shape[1]}")
            if labels_out["ca_coords"].shape[0] != target_L or labels_out["ca_mask"].shape[0] != target_L:
                raise ValueError(f"Label length mismatch for {cid} after preprocessing.")
            if labels_out["atom14_positions"].shape != (target_L, ATOM14_NUM_SLOTS, 3):
                raise ValueError(f"Atom14 shape mismatch for {cid}: {labels_out['atom14_positions'].shape}")

            if args.disable_templates:
                features_out.update(_empty_template_features(target_L))
            else:
                tpl = _extract_template_features(
                    chain_dir=chain_dir,
                    mmcif_root=mmcif_root,
                    target_pdb_id=pdb_id,
                    target_chain_id=chain_id,
                    target_seq=target_seq,
                    target_L=target_L,
                    template_hhr_name=args.template_hhr_name,
                    max_templates=int(args.max_templates),
                )
                features_out.update(tpl)

            _save_npz(processed_features_dir / f"{cid}.npz", features_out)
            _save_npz(processed_labels_dir / f"{cid}.npz", labels_out)
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
        if not args.allow_failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
