from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofold.chain_paths import chain_data_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and pin a raw-source lock for official data preparation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/openproteinset"))
    parser.add_argument("--manifests-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--chain-data-cache", type=Path, required=True)
    parser.add_argument("--structure-metadata", type=Path, required=True)
    parser.add_argument(
        "--metadata-source-lock",
        type=Path,
        default=Path("data/metadata_sources/structure_metadata_sources.lock.json"),
    )
    parser.add_argument(
        "--manifest-lock",
        type=Path,
        default=Path("leaderboard/official_manifest_source.lock.json"),
        help="Manifest lock to annotate when --annotate-manifest-lock is set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".nanofold_private/leaderboard/official_data_source.lock.json"),
    )
    parser.add_argument(
        "--hidden-manifest",
        type=Path,
        default=Path(".nanofold_private/manifests/hidden_val.txt"),
        help="Maintainer-only hidden manifest path used when --include-hidden is set.",
    )
    parser.add_argument("--msa-name", default="uniref90_hits.a3m")
    parser.add_argument(
        "--msa-names",
        default="",
        help="Comma-separated MSA filenames required for each official chain. Defaults to --msa-name.",
    )
    parser.add_argument("--template-hits-name", default="pdb70_hits.hhr")
    parser.add_argument("--enable-templates", action="store_true")
    parser.add_argument("--include-hidden", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--annotate-manifest-lock",
        action="store_true",
        help="Private maintainer mode: write this source lock path/hash into --manifest-lock.",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _display_path(path: Path, *, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _tree_sha256(files: list[Path], *, root: Path) -> str | None:
    present = [path for path in files if path.exists() and path.is_file()]
    if not present:
        return None
    hasher = hashlib.sha256()
    for path in sorted(present):
        try:
            rel = str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
        except ValueError:
            rel = str(path)
        file_hash = _sha256(path)
        if file_hash is None:
            continue
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\t")
        hasher.update(file_hash.encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _read_manifest(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def _manifest_paths(
    manifests_dir: Path,
    *,
    include_hidden: bool,
    hidden_manifest: Path,
) -> dict[str, Path]:
    out = {
        "train": manifests_dir / "train.txt",
        "val": manifests_dir / "val.txt",
        "all": manifests_dir / "all.txt",
    }
    if include_hidden:
        out["hidden_val"] = hidden_manifest
    return out


def _required_mmcif_files(data_root: Path, chain_ids: list[str]) -> list[Path]:
    mmcif_root = data_root / "pdb_data" / "mmcif_files"
    pdb_ids = sorted({chain_id.split("_", 1)[0].lower() for chain_id in chain_ids})
    return [mmcif_root / f"{pdb_id}.cif" for pdb_id in pdb_ids]


def _raw_alignment_files(data_root: Path, chain_ids: list[str]) -> list[Path]:
    roda_root = data_root / "roda_pdb"
    files: list[Path] = []
    for chain_id in chain_ids:
        chain_root = chain_data_dir(roda_root, chain_id)
        if not chain_root.exists():
            continue
        files.extend(path for path in chain_root.rglob("*") if path.is_file())
    return files


def _normalize_msa_names(msa_name: str, msa_names: str) -> list[str]:
    parsed = [token.strip() for token in msa_names.split(",") if token.strip()]
    return parsed or [msa_name]


def _structure_metadata_source(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    source = raw.get("source")
    return source if isinstance(source, dict) else {}


def _has_file(root: Path, filename: str) -> bool:
    if not root.exists():
        return False
    for _ in root.rglob(filename):
        return True
    return False


def _missing_alignment_assets(
    data_root: Path,
    chain_ids: list[str],
    *,
    msa_names: list[str],
    template_hits_name: str,
    require_template_hits: bool,
) -> tuple[list[str], list[str]]:
    missing_dirs: list[str] = []
    missing_files: list[str] = []
    roda_root = data_root / "roda_pdb"
    for chain_id in chain_ids:
        chain_root = chain_data_dir(roda_root, chain_id)
        if not chain_root.exists():
            missing_dirs.append(chain_id)
            missing_files.extend(f"{chain_id}:{msa_name}" for msa_name in msa_names)
            if require_template_hits:
                missing_files.append(f"{chain_id}:{template_hits_name}")
            continue
        for msa_name in msa_names:
            if not _has_file(chain_root, msa_name):
                missing_files.append(f"{chain_id}:{msa_name}")
        if require_template_hits and not _has_file(chain_root, template_hits_name):
            missing_files.append(f"{chain_id}:{template_hits_name}")
    return missing_dirs, missing_files


def _annotate_manifest_lock(manifest_lock: Path, source_lock: Path) -> None:
    if not manifest_lock.exists():
        return
    raw = json.loads(manifest_lock.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Manifest lock must be a JSON object: {manifest_lock}")
    raw["data_source_lock"] = _display_path(source_lock, repo_root=Path.cwd())
    raw["data_source_lock_sha256"] = _sha256(source_lock)
    manifest_lock.write_text(json.dumps(raw, indent=2) + "\n")


def build_lock(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path.cwd()
    manifests = _manifest_paths(
        args.manifests_dir,
        include_hidden=bool(args.include_hidden),
        hidden_manifest=args.hidden_manifest,
    )
    chain_ids: list[str] = []
    manifest_meta: dict[str, Any] = {}
    missing_manifests: list[str] = []
    for split_name, manifest_path in manifests.items():
        if not manifest_path.exists():
            missing_manifests.append(split_name)
        ids = _read_manifest(manifest_path)
        chain_ids.extend(ids)
        manifest_meta[split_name] = {
            "path": _display_path(manifest_path, repo_root=repo_root),
            "exists": manifest_path.exists(),
            "sha256": _sha256(manifest_path),
            "chain_count": len(ids),
        }
    unique_chain_ids = sorted(set(chain_ids))
    duplicate_file = args.data_root / "pdb_data" / "duplicate_pdb_chains.txt"
    mmcif_files = _required_mmcif_files(args.data_root, unique_chain_ids)
    alignment_files = _raw_alignment_files(args.data_root, unique_chain_ids)
    msa_names = _normalize_msa_names(str(args.msa_name), str(args.msa_names))
    missing_mmcifs = [_display_path(path, repo_root=repo_root) for path in mmcif_files if not path.exists()]
    missing_alignment_dirs, missing_alignment_files = _missing_alignment_assets(
        args.data_root,
        unique_chain_ids,
        msa_names=msa_names,
        template_hits_name=str(args.template_hits_name),
        require_template_hits=bool(args.enable_templates),
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifests": manifest_meta,
        "chain_data_cache": {
            "path": _display_path(args.chain_data_cache, repo_root=repo_root),
            "sha256": _sha256(args.chain_data_cache),
        },
        "duplicate_chains": {
            "path": _display_path(duplicate_file, repo_root=repo_root),
            "sha256": _sha256(duplicate_file),
        },
        "structure_metadata": {
            "path": _display_path(args.structure_metadata, repo_root=repo_root),
            "sha256": _sha256(args.structure_metadata),
            "source": _structure_metadata_source(args.structure_metadata),
        },
        "metadata_source_lock": {
            "path": _display_path(args.metadata_source_lock, repo_root=repo_root),
            "sha256": _sha256(args.metadata_source_lock),
        },
        "raw_assets": {
            "missing_manifest_count": len(missing_manifests),
            "missing_manifests": missing_manifests,
            "unique_chain_count": len(unique_chain_ids),
            "mmcif_file_count": len(mmcif_files) - len(missing_mmcifs),
            "missing_mmcif_count": len(missing_mmcifs),
            "missing_mmcifs": missing_mmcifs[:100],
            "mmcif_tree_sha256": _tree_sha256(mmcif_files, root=args.data_root),
            "alignment_file_count": len(alignment_files),
            "required_msa_names": msa_names,
            "template_hits_name": str(args.template_hits_name),
            "template_hits_required": bool(args.enable_templates),
            "missing_alignment_dir_count": len(missing_alignment_dirs),
            "missing_alignment_dirs": missing_alignment_dirs[:100],
            "missing_alignment_file_count": len(missing_alignment_files),
            "missing_alignment_files": missing_alignment_files[:100],
            "alignment_tree_sha256": _tree_sha256(alignment_files, root=args.data_root),
        },
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    lock = build_lock(args)
    raw_assets = lock["raw_assets"]
    if args.require_complete and (
        raw_assets["missing_manifest_count"] > 0
        or raw_assets["missing_mmcif_count"] > 0
        or raw_assets["missing_alignment_dir_count"] > 0
        or raw_assets["missing_alignment_file_count"] > 0
    ):
        raise SystemExit(
            "Official raw-source lock is incomplete: "
            f"missing_manifest_count={raw_assets['missing_manifest_count']} "
            f"missing_mmcif_count={raw_assets['missing_mmcif_count']} "
            f"missing_alignment_dir_count={raw_assets['missing_alignment_dir_count']} "
            f"missing_alignment_file_count={raw_assets['missing_alignment_file_count']}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    if args.annotate_manifest_lock:
        _annotate_manifest_lock(args.manifest_lock, args.output)
    print(f"Wrote official data source lock: {args.output}")


if __name__ == "__main__":
    main()
