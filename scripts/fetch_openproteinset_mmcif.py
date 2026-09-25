"""Retrieve selected mmCIF members from an explicitly pinned OpenProteinSet ZIP.

This maintainer tool verifies transport and ZIP integrity only. It does not
validate target identity, biological suitability, geometry, or a revision cutoff.
It never falls back to downloading the complete archive or another source.
Transient transport and HTTP 429/5xx failures receive at most four attempts,
with 1, 2, and 4 second backoffs. Integrity validation failures are never retried.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import socket
import ssl
import stat
import struct
import sys
import tempfile
import time
import zlib
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from nanofold.structure_metadata import validate_author_chain_id

ARCHIVE_URL = "https://openfold.s3.amazonaws.com/pdb_mmcif.zip"
VALIDATOR_SOURCE = REPO_ROOT / "nanofold/structure_metadata.py"
MAX_RANGE = 2 * 1024**2
MAX_DIRECTORY = 64 * 1024**2
MAX_COMPRESSED = 64 * 1024**2
MAX_UNCOMPRESSED = 256 * 1024**2
MAX_ENTRIES = 1_000_000
MAX_MANIFEST = 10 * 1024**2
MAX_RANGE_ATTEMPTS = 4
U32 = 0xFFFFFFFF


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> NoReturn:
        # Reject before urllib drains an unbounded redirect body or opens another URL.
        raise ValueError("Archive redirects are not permitted")


open_range = build_opener(NoRedirect()).open


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def transient_transport_failure(error: OSError) -> bool:
    """Classify transport failures without treating invalid bytes as retryable."""
    if isinstance(error, HTTPError):
        return error.code == 429 or 500 <= error.code < 600
    reason = error.reason if isinstance(error, URLError) else error
    if isinstance(reason, ssl.SSLCertVerificationError):
        return False
    if isinstance(reason, (TimeoutError, ConnectionError, ssl.SSLEOFError, ssl.SSLZeroReturnError)):
        return True
    if isinstance(reason, socket.gaierror):
        return reason.errno == socket.EAI_AGAIN
    return isinstance(reason, OSError) and reason.errno in {
        errno.ETIMEDOUT, errno.ECONNRESET, errno.ECONNABORTED, errno.ECONNREFUSED,
        errno.EPIPE, errno.ENETRESET, errno.ENETDOWN, errno.ENETUNREACH, errno.EHOSTUNREACH,
    }


@dataclass(frozen=True)
class ArchivePin:
    size: int
    etag: str
    version_id: str | None

    def validate(self) -> None:
        require(self.size > 22, "Invalid pinned archive size")
        require(re.fullmatch(r'"[\x21\x23-\x7e]+"', self.etag) is not None, "A quoted strong ETag is required")
        require(self.version_id is None or re.fullmatch(r"[A-Za-z0-9_.~+/=-]+", self.version_id) is not None,
                "Invalid pinned version ID")

    @property
    def url(self) -> str:
        return ARCHIVE_URL if self.version_id is None else ARCHIVE_URL + "?" + urlencode({"versionId": self.version_id})


class RangeReader:
    def __init__(self, pin: ArchivePin):
        pin.validate()
        self.pin = pin
        self.ranges: list[dict[str, Any]] = []

    def _read_chunk(self, request: Request, offset: int, count: int, proof: dict[str, Any]) -> bytes:
        attempts: list[dict[str, Any]] = []
        proof["attempts"] = attempts
        for attempt in range(1, MAX_RANGE_ATTEMPTS + 1):
            outcome: dict[str, Any] = {"number": attempt, "verified": False}
            attempts.append(outcome)
            try:
                with open_range(request, timeout=90) as response:
                    proof.update(status=response.status, etag=response.headers.get("ETag"),
                                 version_id=response.headers.get("x-amz-version-id"),
                                 content_range=response.headers.get("Content-Range"),
                                 content_length=response.headers.get("Content-Length"))
                    require(response.status == 206, "Server did not honor the range; body was not read")
                    require(response.geturl() == self.pin.url, "Archive request was redirected")
                    for header in ("ETag", "Content-Range", "Content-Length", "x-amz-version-id", "Content-Encoding"):
                        require(len(response.headers.get_all(header, [])) <= 1, "Ambiguous response headers")
                    require(proof["etag"] == self.pin.etag, "Archive ETag changed")
                    require(proof["version_id"] == self.pin.version_id, "Archive version changed")
                    require(proof["content_range"] == f"bytes {offset}-{offset + count - 1}/{self.pin.size}",
                            "Incorrect Content-Range")
                    require(proof["content_length"] == str(count), "Incorrect Content-Length")
                    require(response.headers.get("Content-Encoding", "identity") == "identity", "Encoded range response")
                    data = response.read(count + 1)
                    require(len(data) == count, "Range body length mismatch")
                outcome["verified"] = True
                return data
            except OSError as error:
                retryable = transient_transport_failure(error)
                outcome.update(error_type=type(error).__name__, retryable=retryable)
                if isinstance(error, HTTPError):
                    outcome["http_status"] = error.code
                    error.close()  # Never read an error response body.
                if not retryable or attempt == MAX_RANGE_ATTEMPTS:
                    raise
                delay = min(2 ** (attempt - 1), 4)
                outcome["backoff_seconds"] = delay
                time.sleep(delay)
        raise RuntimeError("Range attempts exhausted")

    def read(self, start: int, length: int) -> bytes:
        require(0 <= start < self.pin.size and 0 < length <= MAX_DIRECTORY and start + length <= self.pin.size,
                "Archive range is outside permitted bounds")
        # Even tiny test archives are read in proper subsets, never as one full GET.
        chunk_limit = min(MAX_RANGE, self.pin.size - 1)
        pieces: list[bytes] = []
        for offset in range(start, start + length, chunk_limit):
            count = min(chunk_limit, start + length - offset)
            end = offset + count - 1
            proof: dict[str, Any] = {"start": offset, "length": count, "verified": False}
            self.ranges.append(proof)
            request = Request(self.pin.url, headers={
                "Range": f"bytes={offset}-{end}", "If-Match": self.pin.etag, "Accept-Encoding": "identity",
            })
            data = self._read_chunk(request, offset, count, proof)
            proof.update(sha256=sha256(data), verified=True)
            pieces.append(data)
        return b"".join(pieces)


def zip64_values(extra: bytes, values: list[tuple[int, int, str]]) -> list[int]:
    """Validate extra-field boundaries and resolve ordered ZIP64 sentinels."""
    at = 0
    zip64: bytes | None = None
    while at < len(extra):
        require(at + 4 <= len(extra), "Truncated ZIP extra field")
        kind, length = struct.unpack_from("<HH", extra, at)
        at += 4
        require(at + length <= len(extra), "ZIP extra field exceeds its bounds")
        if kind == 1:
            require(zip64 is None, "Duplicate ZIP64 extra field")
            zip64 = extra[at:at + length]
        at += length
    output = []
    at = 0
    for value, sentinel, fmt in values:
        if value == sentinel:
            size = struct.calcsize(fmt)
            require(zip64 is not None and at + size <= len(zip64), "Missing or truncated ZIP64 value")
            value = struct.unpack_from("<" + fmt, zip64, at)[0]  # type: ignore[arg-type]
            at += size
        output.append(value)
    require(zip64 is None or at == len(zip64), "Unexpected ZIP64 extra data")
    return output


@dataclass(frozen=True)
class Member:
    name: str
    version: int
    flags: int
    method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    offset: int
    next_offset: int


def read_directory(reader: RangeReader) -> tuple[list[Member], dict[str, Any]]:
    size = reader.pin.size
    tail_length = min(size, 65557 + 76)
    tail_start = size - tail_length
    tail = reader.read(tail_start, tail_length)
    eocd_at = tail.rfind(b"PK\x05\x06")
    require(eocd_at >= 0 and eocd_at + 22 <= len(tail), "Missing ZIP end record")
    _, disk, cd_disk, disk_count, count, cd_size, cd_start, comment_length = struct.unpack_from("<4s4H2IH", tail, eocd_at)
    require(eocd_at + 22 + comment_length == len(tail), "Invalid ZIP comment or trailing data")
    require(disk == cd_disk == 0 and disk_count == count, "Multi-disk archive is unsupported")
    records_start = tail_start + eocd_at
    is_zip64 = count == 65535 or cd_size == U32 or cd_start == U32
    has_locator = eocd_at >= 20 and tail[eocd_at - 20:eocd_at - 16] == b"PK\x06\x07"
    if is_zip64 or has_locator:
        require(has_locator, "Missing ZIP64 locator")
        _, zip_disk, record_start, total_disks = struct.unpack_from("<4sIQI", tail, eocd_at - 20)
        require(zip_disk == 0 and total_disks == 1, "Invalid ZIP64 disk layout")
        locator_start = records_start - 20
        require(record_start + 56 <= locator_start, "ZIP64 end record exceeds locator bounds")
        record = reader.read(record_start, 56)
        sig, record_size, _, version, z_disk, z_cd_disk, z_disk_count, z_count, z_size, z_start = struct.unpack("<4sQ2H2I4Q", record)
        require(sig == b"PK\x06\x06" and record_size == 44 and version <= 45,
                "Unsupported or malformed ZIP64 end record")
        require(record_start + 12 + record_size == locator_start, "ZIP64 end record has corrupt bounds")
        require(z_disk == z_cd_disk == 0 and z_disk_count == z_count, "Invalid ZIP64 disk counts")
        require((count == 65535 or count == z_count) and (cd_size == U32 or cd_size == z_size)
                and (cd_start == U32 or cd_start == z_start), "ZIP and ZIP64 end records disagree")
        count, cd_size, cd_start = z_count, z_size, z_start
        records_start = record_start
        is_zip64 = True
    require(0 < count <= MAX_ENTRIES and 0 < cd_size <= MAX_DIRECTORY, "Central directory exceeds limits")
    require(0 <= cd_start and cd_start + cd_size == records_start, "Central directory has corrupt bounds")
    central = reader.read(cd_start, cd_size)
    entries: list[Member] = []
    names: set[str] = set()
    at = 0
    for _ in range(count):
        require(at + 46 <= len(central), "Truncated central directory")
        fields = struct.unpack_from("<4s6H3I5H2I", central, at)
        (sig, _, version, flags, method, _, _, crc, compressed, uncompressed,
         name_length, extra_length, comment_length, disk, _, _, offset) = fields
        end = at + 46 + name_length + extra_length + comment_length
        require(sig == b"PK\x01\x02" and end <= len(central), "Invalid central directory entry")
        raw_name = central[at + 46:at + 46 + name_length]
        name = raw_name.decode("utf-8" if flags & 0x800 else "cp437")
        path = PurePosixPath(name)
        require(bool(name) and "\x00" not in name and "\\" not in name and not path.is_absolute()
                and all(part not in ("", ".", "..") for part in name.rstrip("/").split("/")), "Unsafe archive path")
        require(name.casefold() not in names, "Duplicate archive member alias")
        names.add(name.casefold())
        extra = central[at + 46 + name_length:at + 46 + name_length + extra_length]
        uncompressed, compressed, offset, disk = zip64_values(extra, [
            (uncompressed, U32, "Q"), (compressed, U32, "Q"), (offset, U32, "Q"), (disk, 65535, "I"),
        ])
        require(disk == 0 and flags in (0, 0x800) and method in (0, 8) and version <= 45,
                "Unsupported archive member encoding")
        require(0 <= offset and offset + 30 + name_length + compressed <= cd_start, "Member exceeds directory bounds")
        entries.append(Member(name, version, flags, method, crc, compressed, uncompressed, offset, 0))
        at = end
    require(at == len(central), "Central directory count or size mismatch")
    entries.sort(key=lambda entry: entry.offset)
    bounded: list[Member] = []
    for index, entry in enumerate(entries):
        next_offset = entries[index + 1].offset if index + 1 < len(entries) else cd_start
        require(entry.offset + 30 + entry.compressed_size <= next_offset, "Overlapping archive members")
        bounded.append(replace(entry, next_offset=next_offset))
    return bounded, {"zip64": is_zip64, "member_count": count, "central_directory_offset": cd_start,
                     "central_directory_size": cd_size, "central_directory_sha256": sha256(central)}


def retrieve_member(reader: RangeReader, member: Member) -> tuple[bytes, dict[str, Any]]:
    require(0 < member.compressed_size <= MAX_COMPRESSED and 0 < member.uncompressed_size <= MAX_UNCOMPRESSED,
            "Selected member exceeds size limits")
    first_range = len(reader.ranges)
    local = reader.read(member.offset, 30)
    sig, version, flags, method, _, _, crc, compressed, uncompressed, nlen, xlen = struct.unpack("<4s5H3I2H", local)
    require(sig == b"PK\x03\x04" and (version, flags, method, crc) ==
            (member.version, member.flags, member.method, member.crc32), "Local and central headers disagree")
    body_start = member.offset + 30 + nlen + xlen
    require(nlen > 0 and body_start + member.compressed_size <= member.next_offset, "Local member exceeds its bounds")
    name_extra = reader.read(member.offset + 30, nlen + xlen)
    require(name_extra[:nlen].decode("utf-8" if flags & 0x800 else "cp437") == member.name,
            "Local and central member names disagree")
    uncompressed, compressed = zip64_values(name_extra[nlen:], [(uncompressed, U32, "Q"), (compressed, U32, "Q")])
    require((compressed, uncompressed) == (member.compressed_size, member.uncompressed_size),
            "Local and central member sizes disagree")
    compressed_data = reader.read(body_start, compressed)
    if method == 8:
        inflater = zlib.decompressobj(-15)
        data = inflater.decompress(compressed_data, uncompressed + 1)
        require(inflater.eof and not inflater.unconsumed_tail and not inflater.unused_data,
                "Deflate stream is incomplete, oversized, or has trailing data")
    else:
        data = compressed_data
    require(len(data) == uncompressed, "Uncompressed member size mismatch")
    require(zlib.crc32(data) & U32 == crc, "Member CRC mismatch")
    return data, {"archive_member": asdict(member), "range_indices": list(range(first_range, len(reader.ranges))),
                  "compressed_sha256": sha256(compressed_data), "output_sha256": sha256(data),
                  "output_size_bytes": len(data), "crc32_verified": True, "local_header_verified": True}


def atomic_write(path: Path, data: bytes) -> None:
    """Publish a private completed file without ever replacing an existing path."""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".partial-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            os.link(temporary, path, follow_symlinks=False)
        finally:
            temporary.unlink(missing_ok=True)


def write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def no_symlinks(path: Path) -> None:
    require(not any(part.is_symlink() for part in (path, *path.parents)), "Symlink paths are not permitted")


def snapshot(path: Path) -> tuple[int, int, int, int, int]:
    info = path.stat()
    require(stat.S_ISREG(info.st_mode), "Source must be a regular file")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def run(args: argparse.Namespace) -> dict[str, Any]:
    pin = ArchivePin(args.archive_size, args.archive_etag,
                     None if args.archive_version_id == "absent" else args.archive_version_id)
    reader = RangeReader(pin)
    manifest = args.manifest.absolute()
    out = args.out_dir.absolute()
    no_symlinks(manifest)
    no_symlinks(out)
    manifest = manifest.resolve()
    out = out.resolve()
    require(not out.exists() and out.parent.is_dir(), "Output must be a fresh directory under an existing parent")
    require(not manifest.is_relative_to(out), "Output aliases a source path")
    require(not out.is_relative_to(REPO_ROOT) or out.is_relative_to(REPO_ROOT / ".nanofold_private"),
            "Repository outputs must be under the ignored private directory")
    manifest_snapshot = snapshot(manifest)
    require(manifest_snapshot[2] <= MAX_MANIFEST, "Manifest exceeds size limit")
    manifest_bytes = manifest.read_bytes()
    require(snapshot(manifest) == manifest_snapshot, "Manifest changed while reading")
    ids = [line.strip() for line in manifest_bytes.decode("utf-8").splitlines()
           if line.strip() and not line.lstrip().startswith("#")]
    require(bool(ids) and len(ids) == len(set(ids)), "Manifest is empty or contains duplicate targets")
    for chain_id in ids:
        validate_author_chain_id(chain_id)
    pdb_ids = {chain_id.split("_", 1)[0] for chain_id in ids}
    script = Path(__file__).resolve()
    script_snapshot = snapshot(script)
    script_hash = sha256(script.read_bytes())
    require(snapshot(script) == script_snapshot, "Retrieval source changed while reading")
    validator_snapshot = snapshot(VALIDATOR_SOURCE)
    validator_hash = sha256(VALIDATOR_SOURCE.read_bytes())
    require(snapshot(VALIDATOR_SOURCE) == validator_snapshot, "Manifest validator source changed while reading")
    inputs = {"manifest_path": str(manifest), "manifest_sha256": sha256(manifest_bytes),
              "script_sha256": script_hash, "manifest_validator_sha256": validator_hash}
    out.mkdir(mode=0o700)
    records: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "scope": "selective_archive_retrieval_only", "biological_identity_validated": False,
        "status": "failed", "archive_url": ARCHIVE_URL, "archive_pin": asdict(pin), "inputs": inputs,
        "selected_chain_count": len(ids), "requested_structure_count": len(pdb_ids),
        "started_at": datetime.now(timezone.utc).isoformat(), "members": records, "ranges": reader.ranges,
    }
    try:
        atomic_write(out / "manifest.txt", manifest_bytes)
        write_json(out / "request.json", {key: value for key, value in result.items() if key not in ("members", "ranges")})
        pdb_data = out / "pdb_data"
        pdb_data.mkdir(mode=0o700)
        cif_dir = pdb_data / "mmcif_files"
        cif_dir.mkdir(mode=0o700)
        members, directory = read_directory(reader)
        result["directory"] = directory
        selected: dict[str, Member] = {}
        for member in members:
            if re.fullmatch(r"mmcif_files/[A-Za-z0-9]{4}\.cif", member.name):
                pdb_id = PurePosixPath(member.name).stem.lower()
                if pdb_id in pdb_ids:
                    require(pdb_id not in selected, "Multiple archive members match one structure")
                    selected[pdb_id] = member
        require(set(selected) == pdb_ids, "Archive lacks one or more requested structures")
        for pdb_id, member in sorted(selected.items()):
            result["active_member"] = member.name
            data, record = retrieve_member(reader, member)
            destination = cif_dir / (pdb_id + ".cif")
            atomic_write(destination, data)
            records.append(record | {"output_path": str(destination.relative_to(out))})
        reader.read(0, 1)  # Confirm the pinned archive still exists after the last member.
        result["status"] = "retrieved_unvalidated_structures"
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "detail": str(exc)}
    finally:
        try:
            unchanged = (snapshot(manifest) == manifest_snapshot and sha256(manifest.read_bytes()) == inputs["manifest_sha256"]
                         and snapshot(script) == script_snapshot and sha256(script.read_bytes()) == script_hash
                         and snapshot(VALIDATOR_SOURCE) == validator_snapshot
                         and sha256(VALIDATOR_SOURCE.read_bytes()) == validator_hash)
        except (OSError, ValueError):
            unchanged = False
        result["source_inputs_unchanged"] = unchanged
        if not unchanged:
            result["status"] = "failed"
        result["retrieved_structure_count"] = len(records)
        result["downloaded_range_bytes"] = sum(row["length"] for row in reader.ranges if row["verified"])
        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        write_json(out / "retrieval-receipt.json", result)
    return result


class PrivateArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ValueError("Invalid arguments; use --help for the required explicit pins")


def main(argv: list[str] | None = None) -> int:
    parser = PrivateArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="Explicit chain manifest; distinct PDB entries are retrieved once.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Fresh private raw root, with receipts and pdb_data/mmcif_files.")
    parser.add_argument("--archive-size", required=True, type=int, help="Pinned complete archive size in bytes.")
    parser.add_argument("--archive-etag", required=True, help="Pinned strong ETag, including literal double quotes.")
    parser.add_argument("--archive-version-id", required=True,
                        help="Exact S3 version ID, including 'null'; use 'absent' only for an absent version header.")
    try:
        result = run(parser.parse_args(argv))
    except Exception:
        print("Retrieval rejected; check arguments, source paths, and fresh private output requirements.", file=sys.stderr)
        return 2
    print(json.dumps({key: result[key] for key in (
        "status", "selected_chain_count", "requested_structure_count", "retrieved_structure_count",
        "downloaded_range_bytes", "source_inputs_unchanged", "biological_identity_validated",
    )}, sort_keys=True))
    return 0 if result["status"] == "retrieved_unvalidated_structures" else 1


if __name__ == "__main__":
    raise SystemExit(main())
