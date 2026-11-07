#!/usr/bin/env python3
"""Fetch only the pinned target-independent Protenix chemical CCD members."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

URL = "https://protenix.tos-cn-beijing.volces.com/common.tar.gz"
ARCHIVE_SHA256 = "08ea594f429df35494c062e3dfcacaf48fa761e4ea4a8bcb6d5107d211e64dbd"
ARCHIVE_BYTES = 475085654
REFERENCE_SHA256 = "b1c985dff08d873c628b3af61cf85c3668681ef9f4c3f2a163d71535f9b3d3e7"
PROTENIX_COMMIT = "c3bfc365b3e1341a11935eddfe7bfdc308092147"
EXPORTER_SHA256 = "f2a5a1517d2757e35e245df62cd6357d3e7739b874f819d52c39f691e2d40d79"
MEMBERS = {
    "common/components.cif": (490777362, "bb31ae5cf6c8bc669924313077cb4231ee5ffefd3a20118cd14f3ec89f8bb6a5"),
    "common/components.cif.rdkit_mol.pkl": (142498117, "d1cfb71f5993a3ebea7c47877022d7f597bbfbaf86e28a4770e957da6c50cd35"),
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def verify_file(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"not a regular nonsymlink file: {path}")
    if path.stat().st_size != expected_bytes or sha256(path) != expected_sha256:
        raise ValueError(f"chemical artifact size/hash differs: {path}")


def extract_chemistry(archive: Path, output_root: Path,
                      expected_members: dict[str, tuple[int, str]]) -> None:
    """Extract only exact regular members; never unpack other common metadata."""
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for name, (size, digest) in expected_members.items():
            selected = [member for member in members if member.name == name]
            if len(selected) != 1 or not selected[0].isfile() or selected[0].size != size:
                raise ValueError(f"missing, duplicate, linked or wrong-size chemical member: {name}")
            destination = output_root / name
            if destination.parent.is_symlink() or destination.is_symlink():
                raise ValueError(f"symlinked chemical destination: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                verify_file(destination, size, digest)
                continue
            descriptor, temporary = tempfile.mkstemp(prefix=".ccd-", dir=destination.parent)
            try:
                with os.fdopen(descriptor, "wb") as target, handle.extractfile(selected[0]) as source:
                    shutil.copyfileobj(source, target)
                    target.flush()
                    os.fsync(target.fileno())
                verify_file(Path(temporary), size, digest)
                os.replace(temporary, destination)
            finally:
                Path(temporary).unlink(missing_ok=True)


def verify_reference(path: Path) -> dict:
    value = json.loads(path.read_text())
    calculated = hashlib.sha256(json.dumps(value["residues"], sort_keys=True,
                                           separators=(",", ":")).encode()).hexdigest()
    if (value.get("schema_version") != 2 or value.get("protenix_commit") != PROTENIX_COMMIT
            or value.get("ccd_reference_sha256") != REFERENCE_SHA256
            or calculated != REFERENCE_SHA256 or len(value["residues"]) != 20):
        raise ValueError("derived chemical reference differs from historical AF3 reference")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protenix-root", type=Path, required=True)
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--reference-output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="already downloaded pinned archive")
    args = parser.parse_args()
    if args.output_root.is_symlink():
        raise ValueError("chemical output root must not be a symlink")
    args.output_root.mkdir(parents=True, exist_ok=True)
    archive = args.archive
    if archive is None:
        cache = args.output_root / ".download-cache"
        cache.mkdir(exist_ok=True)
        archive = cache / "common-08ea594f429df354.tar.gz"
        if not archive.exists():
            temporary = archive.with_suffix(".part")
            with urllib.request.urlopen(URL, timeout=60) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            verify_file(temporary, ARCHIVE_BYTES, ARCHIVE_SHA256)
            os.replace(temporary, archive)
    verify_file(archive, ARCHIVE_BYTES, ARCHIVE_SHA256)
    extract_chemistry(archive, args.output_root, MEMBERS)
    if sha256(args.exporter) != EXPORTER_SHA256:
        raise ValueError("frozen CCD exporter hash differs")
    commit = subprocess.check_output(["git", "-C", str(args.protenix_root),
                                      "rev-parse", "HEAD"], text=True).strip()
    if commit != PROTENIX_COMMIT:
        raise ValueError("Protenix checkout commit differs")
    environment = os.environ.copy()
    environment.pop("PROTENIX_CCD_REFERENCE_JSON", None)
    environment["PROTENIX_ROOT_DIR"] = str(args.output_root.absolute())
    environment["PYTHONPATH"] = str(args.protenix_root.absolute()) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["MPLCONFIGDIR"] = str(args.output_root / ".matplotlib")
    args.reference_output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, candidate = tempfile.mkstemp(prefix=".ccd-reference-", suffix=".json",
                                            dir=args.reference_output.parent)
    os.close(descriptor)
    try:
        subprocess.run([sys.executable, str(args.exporter), "--output", candidate],
                       env=environment, check=True)
        verify_reference(Path(candidate))
        if args.reference_output.exists():
            verify_reference(args.reference_output)
            if sha256(args.reference_output) != sha256(Path(candidate)):
                raise ValueError("existing reference bytes differ; refusing overwrite")
        else:
            os.replace(candidate, args.reference_output)
    finally:
        Path(candidate).unlink(missing_ok=True)
    receipt = {"schema_version": 1, "source_url": URL,
               "source_script": "https://raw.githubusercontent.com/bytedance/Protenix/" + PROTENIX_COMMIT + "/scripts/database/download_protenix_data.sh",
               "archive_sha256": ARCHIVE_SHA256, "archive_bytes": ARCHIVE_BYTES,
               "extracted_members": {name: {"bytes": size, "sha256": digest}
                                      for name, (size, digest) in MEMBERS.items()},
               "ccd_reference_sha256": REFERENCE_SHA256,
               "reference_file_sha256": sha256(args.reference_output),
               "downloaded_training_structures": False,
               "other_archive_members_extracted": False}
    (args.reference_output.parent / "ccd-bootstrap-provenance.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
