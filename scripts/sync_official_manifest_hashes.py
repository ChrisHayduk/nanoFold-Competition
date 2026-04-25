from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class ManifestHashes:
    train: str
    val: str
    all: str
    hidden_val: str | None
    train_count: int
    val_count: int
    hidden_val_count: int | None
    unique_count: int


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Synchronize official manifest SHA256 values across track policy, lock metadata, and docs."
        )
    )
    ap.add_argument("--manifests-dir", type=str, default="data/manifests")
    ap.add_argument("--hidden-manifest", type=str, default=".nanofold_private/manifests/hidden_val.txt")
    ap.add_argument("--track-file", type=str, default="tracks/limited_large.yaml")
    ap.add_argument("--lock-file", type=str, default="leaderboard/official_manifest_source.lock.json")
    ap.add_argument("--readme", type=str, default="README.md")
    ap.add_argument("--competition-doc", type=str, default="docs/COMPETITION.md")
    ap.add_argument(
        "--check",
        action="store_true",
        help="Check-only mode. Exit non-zero if any file is out of sync.",
    )
    ap.add_argument(
        "--include-hidden",
        action="store_true",
        help="Private maintainer mode: also synchronize hidden manifest hash references.",
    )
    return ap.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_manifest_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line)
    return ids


def _compute_hashes(manifests_dir: Path, hidden_manifest: Path) -> ManifestHashes:
    train_path = manifests_dir / "train.txt"
    val_path = manifests_dir / "val.txt"
    all_path = manifests_dir / "all.txt"
    hidden_val_path = hidden_manifest
    for p in (train_path, val_path, all_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing manifest file: {p}")

    train_ids = _read_manifest_ids(train_path)
    val_ids = _read_manifest_ids(val_path)
    all_ids = _read_manifest_ids(all_path)
    hidden_val_ids = _read_manifest_ids(hidden_val_path) if hidden_val_path.exists() else None
    return ManifestHashes(
        train=_sha256(train_path),
        val=_sha256(val_path),
        all=_sha256(all_path),
        hidden_val=_sha256(hidden_val_path) if hidden_val_path.exists() else None,
        train_count=len(train_ids),
        val_count=len(val_ids),
        hidden_val_count=len(hidden_val_ids) if hidden_val_ids is not None else None,
        unique_count=len(set(all_ids)),
    )


def _replace_required(text: str, pattern: str, repl: str, *, label: str) -> str:
    updated, n = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE)
    if n != 1:
        raise ValueError(f"Could not locate {label} pattern for replacement.")
    return updated


def _update_readme(path: Path, hashes: ManifestHashes) -> str:
    text = path.read_text()
    text = _replace_required(
        text,
        r"(- `train\.txt`: `)[0-9a-f]{64}(`)",
        rf"\g<1>{hashes.train}\2",
        label="README train hash",
    )
    text = _replace_required(
        text,
        r"(- `val\.txt`: `)[0-9a-f]{64}(`)",
        rf"\g<1>{hashes.val}\2",
        label="README val hash",
    )
    text = _replace_required(
        text,
        r"(- `all\.txt`: `)[0-9a-f]{64}(`)",
        rf"\g<1>{hashes.all}\2",
        label="README all hash",
    )
    return text


def _update_competition_doc(path: Path, hashes: ManifestHashes) -> str:
    text = path.read_text()
    text = _replace_required(
        text,
        r"(- train: `)[0-9a-f]{64}(`)",
        rf"\g<1>{hashes.train}\2",
        label="COMPETITION train hash",
    )
    text = _replace_required(
        text,
        r"(- val: `)[0-9a-f]{64}(`)",
        rf"\g<1>{hashes.val}\2",
        label="COMPETITION val hash",
    )
    text = _replace_required(
        text,
        r"(- all: `)[0-9a-f]{64}(`)",
        rf"\g<1>{hashes.all}\2",
        label="COMPETITION all hash",
    )
    return text


def _update_track_yaml(path: Path, hashes: ManifestHashes, *, include_hidden: bool) -> str:
    text = path.read_text()
    replacements = {
        "train_manifest_sha256": hashes.train,
        "val_manifest_sha256": hashes.val,
        "all_manifest_sha256": hashes.all,
        "train_chain_count": str(hashes.train_count),
        "val_chain_count": str(hashes.val_count),
    }
    for key, value in replacements.items():
        text = _replace_required(
            text,
            rf"(^\s*{re.escape(key)}:\s*).*$",
            rf"\g<1>{value}",
            label=f"track {key}",
        )
    if include_hidden and hashes.hidden_val is not None:
        text = _replace_required(
            text,
            r"(^\s*hidden_manifest_sha256:\s*).*$",
            rf"\g<1>{hashes.hidden_val}",
            label="track hidden_manifest_sha256",
        )
        text = _replace_required(
            text,
            r"(^\s*hidden_chain_count:\s*).*$",
            rf"\g<1>{hashes.hidden_val_count}",
            label="track hidden_chain_count",
        )
    else:
        text = _replace_required(
            text,
            r"(^\s*hidden_manifest_sha256:\s*).*$",
            r"\g<1>null",
            label="track hidden_manifest_sha256",
        )
    return text


def _display_path(path: Path, *, lock_file: Path) -> str:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(resolved.relative_to(cwd))
    except ValueError:
        return os.path.relpath(resolved, start=lock_file.parent.resolve())


def _update_lock_json(
    path: Path,
    hashes: ManifestHashes,
    manifests_dir: Path,
    hidden_manifest: Path,
    *,
    include_hidden: bool,
) -> str:
    if path.exists():
        raw = json.loads(path.read_text())
    else:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Lock file must be a JSON object: {path}")

    if isinstance(raw.get("chain_data_cache_path"), str):
        cache_path = Path(str(raw["chain_data_cache_path"]))
        if cache_path.is_absolute():
            raw["chain_data_cache_path"] = _display_path(cache_path, lock_file=path)
    if isinstance(raw.get("data_source_lock"), str):
        source_lock_path = Path(str(raw["data_source_lock"]))
        if source_lock_path.is_absolute():
            raw["data_source_lock"] = _display_path(source_lock_path, lock_file=path)
    if not include_hidden:
        raw.pop("data_source_lock", None)
        raw.pop("data_source_lock_sha256", None)

    args = raw.get("args")
    if isinstance(args, dict):
        for key in ("cluster_tsv", "structure_metadata"):
            value = args.get(key)
            if isinstance(value, str) and value and Path(value).is_absolute():
                args[key] = _display_path(Path(value), lock_file=path)

    outputs = raw.get("outputs")
    if not isinstance(outputs, dict):
        outputs = {}
        raw["outputs"] = outputs

    train_path = manifests_dir / "train.txt"
    val_path = manifests_dir / "val.txt"
    all_path = manifests_dir / "all.txt"
    hidden_val_path = hidden_manifest
    outputs["train_manifest"] = _display_path(train_path, lock_file=path)
    outputs["val_manifest"] = _display_path(val_path, lock_file=path)
    outputs["all_manifest"] = _display_path(all_path, lock_file=path)
    if include_hidden and (hidden_val_path.exists() or "hidden_val_manifest" in outputs):
        outputs["hidden_val_manifest"] = _display_path(hidden_val_path, lock_file=path) if hidden_val_path.exists() else None
    else:
        outputs.pop("hidden_val_manifest", None)
    outputs["train_manifest_sha256"] = hashes.train
    outputs["val_manifest_sha256"] = hashes.val
    outputs["all_manifest_sha256"] = hashes.all
    if include_hidden and (hashes.hidden_val is not None or "hidden_val_manifest_sha256" in outputs):
        outputs["hidden_val_manifest_sha256"] = hashes.hidden_val
    else:
        outputs.pop("hidden_val_manifest_sha256", None)
    outputs["train_count"] = hashes.train_count
    outputs["val_count"] = hashes.val_count
    if include_hidden and (hashes.hidden_val_count is not None or "hidden_val_count" in outputs):
        outputs["hidden_val_count"] = hashes.hidden_val_count or 0
    else:
        outputs.pop("hidden_val_count", None)
    if not include_hidden:
        outputs.pop("split_quality_report", None)
        outputs.pop("split_quality_report_sha256", None)
    outputs["unique_count"] = hashes.unique_count

    if not include_hidden:
        stratification = raw.get("stratification")
        if isinstance(stratification, dict):
            split_distributions = stratification.get("split_distributions")
            if isinstance(split_distributions, dict):
                split_distributions.pop("hidden_val", None)
            quality_report = stratification.get("quality_report")
            if isinstance(quality_report, dict):
                quality_report.pop("selected_distribution", None)
                quality_report.pop("observed", None)
                for key in ("split_distribution", "jensen_shannon_divergence"):
                    value = quality_report.get(key)
                    if isinstance(value, dict):
                        value.pop("hidden_val", None)
    return json.dumps(raw, indent=2) + "\n"


def _write_or_check(path: Path, new_text: str, *, check_only: bool) -> bool:
    old_text = path.read_text() if path.exists() else ""
    changed = old_text != new_text
    if changed and not check_only:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_text)
    return changed


def main() -> None:
    args = parse_args()
    manifests_dir = Path(args.manifests_dir).resolve()
    hidden_manifest = Path(args.hidden_manifest).resolve()
    track_file = Path(args.track_file).resolve()
    lock_file = Path(args.lock_file).resolve()
    readme = Path(args.readme).resolve()
    competition_doc = Path(args.competition_doc).resolve()

    hashes = _compute_hashes(manifests_dir, hidden_manifest)
    updates: Dict[Path, str] = {
        track_file: _update_track_yaml(track_file, hashes, include_hidden=bool(args.include_hidden)),
        lock_file: _update_lock_json(
            lock_file,
            hashes,
            manifests_dir,
            hidden_manifest,
            include_hidden=bool(args.include_hidden),
        ),
        readme: _update_readme(readme, hashes),
        competition_doc: _update_competition_doc(competition_doc, hashes),
    }

    changed_paths: list[Path] = []
    for path, new_text in updates.items():
        if _write_or_check(path, new_text, check_only=bool(args.check)):
            changed_paths.append(path)

    if changed_paths:
        rel = [str(p.relative_to(Path.cwd())) if p.is_relative_to(Path.cwd()) else str(p) for p in changed_paths]
        if args.check:
            print("Out-of-sync official hash references:")
            for item in rel:
                print(f"- {item}")
            print("\nRun:\n  python scripts/sync_official_manifest_hashes.py")
            raise SystemExit(1)
        print("Updated files:")
        for item in rel:
            print(f"- {item}")
    else:
        print("All official hash references are already in sync.")


if __name__ == "__main__":
    main()
