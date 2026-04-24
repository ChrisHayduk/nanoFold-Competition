from __future__ import annotations

import argparse
import signal
import subprocess
import time
from pathlib import Path
from typing import Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

"""Download a fixed subset of OpenProteinSet/OpenFold training data.

This script is deliberately implementation-light and delegates heavy lifting to the AWS CLI
(RODA uses an anonymous S3 bucket).

By default, this downloads a filtered subset of each chain directory:
- one MSA file (`--msa-name`, default `uniref90_hits.a3m`)
- template hits (`--template-hits-name`, default `pdb70_hits.hhr`)

Use `--full-chain-dir` to download complete per-chain directories.

This script assumes:
- you have `aws` installed
- your manifests contain chain IDs like `7KDX_A`

The OpenFold docs describe the bucket layout and a helper script to flatten the RODA directory:
https://openfold.readthedocs.io/en/latest/OpenFold_Training_Setup.html

You will likely want to run `scripts/preprocess.py` after downloading to convert raw A3M + mmCIF into .npz.
"""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data/raw", help="Where to store downloaded raw data")
    ap.add_argument("--manifest", type=str, required=True, help="train.txt or val.txt listing chain IDs")
    ap.add_argument(
        "--duplicate-chains-file",
        type=str,
        default="",
        help="Optional duplicate_pdb_chains.txt for fallback to representative chains when a chain dir is missing.",
    )
    ap.add_argument("--bucket", type=str, default="s3://openfold", help="RODA bucket for OpenProteinSet")
    ap.add_argument("--msa-name", type=str, default="uniref90_hits.a3m", help="MSA filename to download per chain")
    ap.add_argument(
        "--template-hits-name",
        type=str,
        default="pdb70_hits.hhr",
        help="Template hits filename to download per chain when template hits are enabled",
    )
    ap.add_argument(
        "--no-template-hits",
        action="store_true",
        help="Skip downloading template hits file (pdb70_hits.hhr by default).",
    )
    ap.add_argument(
        "--full-chain-dir",
        action="store_true",
        help="Download the full chain directory instead of filtering to specific files.",
    )
    ap.add_argument(
        "--include-mmcif-zip",
        action="store_true",
        help="Also download pdb_mmcif.zip (very large). Consider hosting a smaller mmCIF subset yourself.",
    )
    ap.add_argument(
        "--download-mmcif-subset",
        action="store_true",
        help="Download only the manifest PDB mmCIF files from RCSB into pdb_data/mmcif_files.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running.",
    )
    ap.add_argument(
        "--download-retries",
        type=int,
        default=2,
        help="Number of retries for failed aws downloads (default: 2).",
    )
    ap.add_argument(
        "--download-retry-delay-seconds",
        type=float,
        default=2.0,
        help="Base delay for download retries in seconds (default: 2.0).",
    )
    return ap.parse_args()


def _format_returncode(returncode: int) -> str:
    if returncode < 0:
        sig_num = -returncode
        try:
            sig_name = signal.Signals(sig_num).name
        except Exception:
            sig_name = f"SIG{sig_num}"
        return f"terminated by {sig_name} ({sig_num})"
    return f"exit code {returncode}"


def run(
    cmd: list[str],
    dry_run: bool,
    *,
    retries: int = 0,
    retry_delay_seconds: float = 2.0,
    allow_fail: bool = False,
) -> bool:
    print("+", " ".join(cmd))
    if not dry_run:
        attempt = 0
        while True:
            try:
                subprocess.check_call(cmd)
                return True
            except subprocess.CalledProcessError as e:
                attempt += 1
                if attempt <= retries:
                    wait_s = max(0.0, retry_delay_seconds) * attempt
                    print(
                        f"WARNING: command failed ({_format_returncode(e.returncode)}), "
                        f"retrying {attempt}/{retries} after {wait_s:.1f}s"
                    )
                    time.sleep(wait_s)
                    continue
                if allow_fail:
                    print(f"WARNING: command failed and will be skipped ({_format_returncode(e.returncode)})")
                    return False
                raise
            except OSError as e:
                attempt += 1
                if attempt <= retries:
                    wait_s = max(0.0, retry_delay_seconds) * attempt
                    print(f"WARNING: command error ({e}), retrying {attempt}/{retries} after {wait_s:.1f}s")
                    time.sleep(wait_s)
                    continue
                if allow_fail:
                    print(f"WARNING: command error and will be skipped ({e})")
                    return False
                raise
    return True


def _has_file(root: Path, filename: str) -> bool:
    if not root.exists():
        return False
    for _ in root.rglob(filename):
        return True
    return False


def _has_required_files(root: Path, msa_name: str, template_hits_name: str, need_template_hits: bool) -> tuple[bool, bool]:
    has_msa = _has_file(root, msa_name)
    has_tpl = _has_file(root, template_hits_name) if need_template_hits else True
    return has_msa, has_tpl


def _load_duplicate_map(path: str) -> Dict[str, List[str]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"WARNING: duplicate chains file not found: {p}")
        return {}

    dup_map: Dict[str, List[str]] = {}
    for ln in p.read_text().splitlines():
        ids = [tok.strip() for tok in ln.split() if tok.strip()]
        if len(ids) < 2:
            continue
        for cid in ids:
            dup_map[cid] = ids
    return dup_map


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x in seen:
            continue
        out.append(x)
        seen.add(x)
    return out


def _structure_url_and_destination(cid: str, data_root: Path) -> tuple[str, Path]:
    pdb_id = cid.split("_", 1)[0]
    return (
        f"https://files.rcsb.org/download/{pdb_id.upper()}.cif",
        data_root / "pdb_data" / "mmcif_files" / f"{pdb_id.lower()}.cif",
    )


def _download_url(
    url: str,
    destination: Path,
    *,
    dry_run: bool,
    retries: int,
    retry_delay_seconds: float,
    allow_fail: bool,
) -> bool:
    print("+", url, "->", destination)
    if dry_run:
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while True:
        try:
            with urlopen(url) as response:
                destination.write_bytes(response.read())
            return True
        except (HTTPError, URLError, OSError) as exc:
            attempt += 1
            if attempt <= retries:
                wait_s = max(0.0, retry_delay_seconds) * attempt
                print(f"WARNING: download failed ({exc}), retrying {attempt}/{retries} after {wait_s:.1f}s")
                time.sleep(wait_s)
                continue
            if allow_fail:
                print(f"WARNING: download failed and will be skipped ({exc})")
                return False
            raise


def _download_mmcif_subset(
    chain_ids: List[str],
    data_root: Path,
    *,
    dry_run: bool,
    retries: int,
    retry_delay_seconds: float,
) -> None:
    seen_pdb_ids: set[str] = set()
    for cid in chain_ids:
        pdb_id = cid.split("_", 1)[0].lower()
        if pdb_id in seen_pdb_ids:
            continue
        seen_pdb_ids.add(pdb_id)
        url, destination = _structure_url_and_destination(cid, data_root)
        if destination.exists():
            continue
        _download_url(
            url,
            destination,
            dry_run=dry_run,
            retries=retries,
            retry_delay_seconds=retry_delay_seconds,
            allow_fail=True,
        )


def _download_chain(
    cid: str,
    out_dir: Path,
    *,
    bucket: str,
    msa_name: str,
    template_hits_name: str,
    no_template_hits: bool,
    full_chain_dir: bool,
    dry_run: bool,
    retries: int,
    retry_delay_seconds: float,
) -> bool:
    s3_path = f"{bucket}/pdb/{cid}"
    if full_chain_dir:
        return run(
            ["aws", "s3", "cp", s3_path, str(out_dir), "--recursive", "--no-sign-request"],
            dry_run,
            retries=retries,
            retry_delay_seconds=retry_delay_seconds,
            allow_fail=True,
        )

    cmd = [
        "aws",
        "s3",
        "cp",
        s3_path,
        str(out_dir),
        "--recursive",
        "--exclude",
        "*",
        "--include",
        msa_name,
        "--include",
        f"*/{msa_name}",
    ]
    if not no_template_hits:
        cmd.extend(
            [
                "--include",
                template_hits_name,
                "--include",
                f"*/{template_hits_name}",
            ]
        )
    cmd.append("--no-sign-request")
    return run(
        cmd,
        dry_run,
        retries=retries,
        retry_delay_seconds=retry_delay_seconds,
        allow_fail=True,
    )


def _replace_with_symlink(target: Path, source: Path) -> None:
    if target.resolve(strict=False) == source.resolve(strict=False):
        return

    if target.is_symlink():
        target.unlink()
    elif target.exists():
        if target.is_dir() and not any(target.iterdir()):
            target.rmdir()
        else:
            # Keep pre-existing non-empty directories to avoid destructive behavior.
            return

    target.symlink_to(source.resolve())


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    roda_root = data_root / "roda_pdb"
    roda_root.mkdir(parents=True, exist_ok=True)

    manifest = Path(args.manifest)
    chain_ids = [ln.strip() for ln in manifest.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    duplicate_map = _load_duplicate_map(args.duplicate_chains_file)
    known_missing_sources = set()

    for cid in chain_ids:
        out_dir = roda_root / cid
        out_dir.parent.mkdir(parents=True, exist_ok=True)

        need_tpl = not args.no_template_hits
        has_msa, has_tpl = _has_required_files(out_dir, args.msa_name, args.template_hits_name, need_tpl)
        if has_msa and has_tpl:
            continue

        candidates = [cid]
        if cid in duplicate_map:
            candidates.extend(duplicate_map[cid])
        candidates = _dedupe_keep_order(candidates)

        chosen_source = None
        chosen_has_tpl = False
        fallback_source = None
        fallback_has_tpl = False

        # First pass: prefer any already-present local candidate to avoid redundant AWS calls.
        for source_cid in candidates:
            source_dir = roda_root / source_cid
            source_dir.parent.mkdir(parents=True, exist_ok=True)
            src_msa, src_tpl = _has_required_files(source_dir, args.msa_name, args.template_hits_name, need_tpl)
            if src_msa and fallback_source is None:
                fallback_source = source_cid
                fallback_has_tpl = src_tpl

            if src_msa and src_tpl:
                chosen_source = source_cid
                chosen_has_tpl = src_tpl
                break

        # Second pass: download only if we still have no suitable local source.
        if chosen_source is None and fallback_source is None:
            for source_cid in candidates:
                if source_cid in known_missing_sources:
                    continue
                source_dir = roda_root / source_cid
                _download_chain(
                    source_cid,
                    source_dir,
                    bucket=args.bucket,
                    msa_name=args.msa_name,
                    template_hits_name=args.template_hits_name,
                    no_template_hits=args.no_template_hits,
                    full_chain_dir=args.full_chain_dir,
                    dry_run=args.dry_run,
                    retries=max(0, int(args.download_retries)),
                    retry_delay_seconds=max(0.0, float(args.download_retry_delay_seconds)),
                )
                src_msa, src_tpl = _has_required_files(source_dir, args.msa_name, args.template_hits_name, need_tpl)
                if not src_msa:
                    known_missing_sources.add(source_cid)
                    continue

                if fallback_source is None:
                    fallback_source = source_cid
                    fallback_has_tpl = src_tpl
                if src_tpl:
                    chosen_source = source_cid
                    chosen_has_tpl = src_tpl
                    break

        if chosen_source is None and fallback_source is not None:
            chosen_source = fallback_source
            chosen_has_tpl = fallback_has_tpl

        if chosen_source is None:
            print(f"WARNING: {cid} missing {args.msa_name} after download")
            if need_tpl:
                print(f"WARNING: {cid} missing {args.template_hits_name} after download")
            continue

        if chosen_source != cid and not args.dry_run:
            _replace_with_symlink(out_dir, roda_root / chosen_source)

        has_msa = _has_file(out_dir, args.msa_name)
        has_tpl = chosen_has_tpl if need_tpl else True
        if not has_msa:
            print(f"WARNING: {cid} missing {args.msa_name} after download")
        if not has_tpl:
            print(f"WARNING: {cid} missing {args.template_hits_name} after download")

    if args.include_mmcif_zip:
        run(["aws", "s3", "cp", f"{args.bucket}/pdb_mmcif.zip", str(data_root), "--no-sign-request"], args.dry_run)
    if args.download_mmcif_subset:
        _download_mmcif_subset(
            chain_ids,
            data_root,
            dry_run=args.dry_run,
            retries=max(0, int(args.download_retries)),
            retry_delay_seconds=max(0.0, float(args.download_retry_delay_seconds)),
        )

    print("Done. Next: run scripts/preprocess.py to build data/processed/*.npz")


if __name__ == "__main__":
    main()
