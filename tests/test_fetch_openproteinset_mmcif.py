from __future__ import annotations

import errno
import io
import json
import os
import re
import socket
import ssl
import struct
import zipfile
import zlib
from email.message import Message
from http.client import HTTPMessage
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from scripts import fetch_openproteinset_mmcif as fetch

ETAG = '"immutable-fixture-2"'
PAYLOAD = b"data_0001\n#\n_entry.id 0001\n#\n" * 20


def make_zip(*, zip64: bool = True, stored: bool = False) -> bytes:
    """Small valid ZIP/ZIP64 files, independently checked by Python's ZIP reader."""
    body = bytearray()
    central = bytearray()
    for index in (1, 2):
        name = f"mmcif_files/000{index}.cif".encode()
        data = PAYLOAD if index == 1 else b"data_0002\n#\n"
        compressor = zlib.compressobj(wbits=-15)
        compressed = data if stored else compressor.compress(data) + compressor.flush()
        crc = zlib.crc32(data)
        offset = len(body)
        method = 0 if stored else 8
        version = 45 if zip64 else 20
        local_extra = struct.pack("<HHQQ", 1, 16, len(data), len(compressed)) if zip64 else b""
        csize, usize = (fetch.U32, fetch.U32) if zip64 else (len(compressed), len(data))
        body += struct.pack("<4s5H3I2H", b"PK\x03\x04", version, 0, method, 0, 0,
                            crc, csize, usize, len(name), len(local_extra)) + name + local_extra + compressed
        extra = struct.pack("<HHQQQ", 1, 24, len(data), len(compressed), offset) if zip64 else b""
        central += struct.pack("<4s6H3I5H2I", b"PK\x01\x02", version, version, 0, method, 0, 0,
                               crc, csize, usize, len(name), len(extra), 0, 0, 0, 0,
                               fetch.U32 if zip64 else offset) + name + extra
    cd_start = len(body)
    body += central
    if zip64:
        record_start = len(body)
        body += struct.pack("<4sQ2H2I4Q", b"PK\x06\x06", 44, 45, 45, 0, 0, 2, 2, len(central), cd_start)
        body += struct.pack("<4sIQI", b"PK\x06\x07", 0, record_start, 1)
    body += struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, 65535 if zip64 else 2, 65535 if zip64 else 2,
                        fetch.U32 if zip64 else len(central), fetch.U32 if zip64 else cd_start, 0)
    with zipfile.ZipFile(io.BytesIO(body)) as reference:
        assert reference.read("mmcif_files/0001.cif") == PAYLOAD
    return bytes(body)


class Response(io.BytesIO):
    def __init__(self, data: bytes, headers: Message, status: int, url: str):
        super().__init__(data)
        self.headers = headers
        self.status = status
        self.url = url
        self.read_calls = 0

    def geturl(self) -> str:
        return self.url

    def read(self, size: int = -1) -> bytes:
        assert 0 < size <= fetch.MAX_RANGE + 1
        self.read_calls += 1
        return super().read(size)


class Server:
    def __init__(self, data: bytes, *, version: str | None = "null", fault: str = "", fault_at: int = 1):
        self.data = data
        self.version = version
        self.fault = fault
        self.fault_at = fault_at
        self.requests: list[tuple[int, int]] = []
        self.responses: list[Response] = []
        self.on_request: Any = None

    def __call__(self, request: Any, timeout: int) -> Response:
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range"))
        assert match is not None
        start, end = map(int, match.groups())
        assert 0 <= start <= end < len(self.data)
        assert end - start + 1 <= fetch.MAX_RANGE and end - start + 1 < len(self.data)
        assert request.get_header("If-match") == ETAG
        assert request.get_header("Accept-encoding") == "identity"
        assert timeout == 90
        self.requests.append((start, end))
        if self.on_request is not None:
            self.on_request()
        headers = Message()
        headers["ETag"] = ETAG
        headers["Content-Range"] = f"bytes {start}-{end}/{len(self.data)}"
        headers["Content-Length"] = str(end - start + 1)
        if self.version is not None:
            headers["x-amz-version-id"] = self.version
        status = 206
        data = self.data[start:end + 1]
        url = request.full_url
        if len(self.requests) == self.fault_at:
            if self.fault == "status":
                status = 200
            elif self.fault == "etag":
                headers.replace_header("ETag", '"changed"')
            elif self.fault == "version":
                headers.replace_header("x-amz-version-id", "changed")
            elif self.fault == "range":
                headers.replace_header("Content-Range", f"bytes {start}-{end}/{len(self.data) + 1}")
            elif self.fault == "length":
                headers.replace_header("Content-Length", str(len(data) + 1))
            elif self.fault == "encoding":
                headers["Content-Encoding"] = "gzip"
            elif self.fault == "duplicate":
                headers["ETag"] = ETAG
            elif self.fault == "redirect":
                url = "https://example.org/not-the-archive"
            elif self.fault == "short":
                data = data[:-1]
            elif self.fault == "long":
                data += b"x"
        response = Response(data, headers, status, url)
        self.responses.append(response)
        return response


def install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data: bytes | None = None, **server_kwargs: Any) -> tuple[list[str], Server]:
    archive = make_zip() if data is None else data
    server = Server(archive, **server_kwargs)
    monkeypatch.setattr(fetch, "open_range", server)
    (tmp_path / "manifest.txt").write_text("0001_A\n0001_B\n")
    return ["--manifest", str(tmp_path / "manifest.txt"), "--out-dir", str(tmp_path / "raw"),
            "--archive-size", str(len(archive)), "--archive-etag", ETAG,
            "--archive-version-id", "absent" if server.version is None else server.version], server


def receipt(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "raw/retrieval-receipt.json").read_text())


@pytest.mark.parametrize("zip64,stored", [(True, False), (False, False), (True, True), (False, True)])
@pytest.mark.parametrize("version", ["null", None, "opaque.Version_123"])
def test_complete_private_retrieval_and_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                 capsys: pytest.CaptureFixture[str], zip64: bool, stored: bool,
                                                 version: str | None) -> None:
    args, server = install(tmp_path, monkeypatch, make_zip(zip64=zip64, stored=stored), version=version)
    assert fetch.main(args) == 0
    result = receipt(tmp_path)
    assert result["status"] == "retrieved_unvalidated_structures"
    assert not result["biological_identity_validated"]
    assert result["source_inputs_unchanged"] and result["selected_chain_count"] == 2
    assert result["requested_structure_count"] == result["retrieved_structure_count"] == 1
    assert result["directory"]["zip64"] == zip64
    assert result["archive_pin"]["version_id"] == version
    cif = tmp_path / "raw/pdb_data/mmcif_files/0001.cif"
    assert cif.read_bytes() == PAYLOAD
    assert list(cif.parent.iterdir()) == [cif]
    assert result["members"][0]["output_sha256"] == fetch.sha256(PAYLOAD)
    assert all(row["verified"] for row in result["ranges"])
    assert result["inputs"]["manifest_sha256"] == fetch.sha256((tmp_path / "manifest.txt").read_bytes())
    assert result["inputs"]["manifest_validator_sha256"] == fetch.sha256(fetch.VALIDATOR_SOURCE.read_bytes())
    for path in (tmp_path / "raw").rglob("*"):
        assert path.stat().st_mode & 0o077 == 0
    assert (tmp_path / "raw").stat().st_mode & 0o777 == 0o700
    output = capsys.readouterr()
    assert "0001" not in output.out + output.err
    assert len(server.requests) > 3
    before = {p: p.read_bytes() for p in (tmp_path / "raw").rglob("*") if p.is_file()}
    request_count = len(server.requests)
    assert fetch.main(args) == 2
    assert len(server.requests) == request_count
    assert all(path.read_bytes() == value for path, value in before.items())


@pytest.mark.parametrize("fault", ["status", "etag", "version", "range", "length", "encoding", "duplicate", "redirect", "short", "long"])
def test_range_rejection_preserves_failure_receipt_without_reading_untrusted_bodies(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    args, server = install(tmp_path, monkeypatch, fault=fault)
    assert fetch.main(args) == 1
    result = receipt(tmp_path)
    assert result["status"] == "failed" and result["retrieved_structure_count"] == 0
    assert not result["ranges"][0]["verified"]
    assert server.responses[0].read_calls == (1 if fault in ("short", "long") else 0)
    assert len(server.requests) == 1
    assert len(result["ranges"][0]["attempts"]) == 1
    assert not list((tmp_path / "raw").rglob("*.cif"))


def test_archive_change_after_metadata_is_not_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, server = install(tmp_path, monkeypatch, fault="etag", fault_at=5)
    assert fetch.main(args) == 1
    assert receipt(tmp_path)["status"] == "failed"
    assert len(server.requests) == 5
    assert server.responses[-1].read_calls == 0


@pytest.mark.parametrize("mutation", ["locator", "record_size", "cd_bounds", "count", "disk", "extra", "local_overlap",
                                       "duplicate_offset", "local_name", "local_crc", "local_size", "local_method", "trailing"])
def test_rejects_corrupt_zip64_and_local_headers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    data = bytearray(make_zip())
    z64 = data.index(b"PK\x06\x06")
    locator = data.index(b"PK\x06\x07")
    cd = data.index(b"PK\x01\x02")
    second_cd = data.index(b"PK\x01\x02", cd + 4)
    nlen = struct.unpack_from("<H", data, cd + 28)[0]
    if mutation == "locator":
        struct.pack_into("<Q", data, locator + 8, locator - 1)
    elif mutation == "record_size":
        struct.pack_into("<Q", data, z64 + 4, 43)
    elif mutation == "cd_bounds":
        struct.pack_into("<Q", data, z64 + 48, cd + 1)
    elif mutation == "count":
        struct.pack_into("<QQ", data, z64 + 24, 3, 3)
    elif mutation == "disk":
        struct.pack_into("<I", data, z64 + 16, 1)
    elif mutation == "extra":
        struct.pack_into("<H", data, cd + 46 + nlen + 2, 27)
    elif mutation == "local_overlap":
        struct.pack_into("<H", data, 28, 65535)
    elif mutation == "duplicate_offset":
        second_nlen = struct.unpack_from("<H", data, second_cd + 28)[0]
        struct.pack_into("<Q", data, second_cd + 46 + second_nlen + 20, 0)
    elif mutation == "local_name":
        data[30] = ord("x")
    elif mutation == "local_crc":
        struct.pack_into("<I", data, 14, 1)
    elif mutation == "local_size":
        struct.pack_into("<Q", data, 30 + nlen + 4, 1)
    elif mutation == "local_method":
        struct.pack_into("<H", data, 8, 0)
    else:
        data += b"unexpected"
    args, _ = install(tmp_path, monkeypatch, bytes(data))
    assert fetch.main(args) == 1
    assert receipt(tmp_path)["retrieved_structure_count"] == 0
    assert not list((tmp_path / "raw").rglob("*.cif"))


@pytest.mark.parametrize("mutation", ["crc", "bomb", "compressed_limit", "uncompressed_limit", "deflate_trailing", "truncated"])
def test_payload_integrity_is_checked_before_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    data = bytearray(make_zip(zip64=False))
    cd = data.index(b"PK\x01\x02")
    nlen = struct.unpack_from("<H", data, 26)[0]
    compressed = struct.unpack_from("<I", data, 18)[0]
    if mutation == "crc":
        struct.pack_into("<I", data, 14, 1)
        struct.pack_into("<I", data, cd + 16, 1)
    elif mutation == "bomb":
        struct.pack_into("<I", data, 22, 1)
        struct.pack_into("<I", data, cd + 24, 1)
    elif mutation == "compressed_limit":
        monkeypatch.setattr(fetch, "MAX_COMPRESSED", 1)
    elif mutation == "uncompressed_limit":
        monkeypatch.setattr(fetch, "MAX_UNCOMPRESSED", 1)
    elif mutation == "deflate_trailing":
        # Replace the compressed payload with a shorter valid stream followed by garbage.
        compressor = zlib.compressobj(wbits=-15)
        content = b"A" * len(PAYLOAD)
        packed = compressor.compress(content) + compressor.flush()
        assert len(packed) < compressed
        data[30 + nlen:30 + nlen + compressed] = packed + b"x" * (compressed - len(packed))
        struct.pack_into("<I", data, 14, zlib.crc32(content))
        struct.pack_into("<I", data, cd + 16, zlib.crc32(content))
    else:
        struct.pack_into("<I", data, 18, compressed - 1)
        struct.pack_into("<I", data, cd + 20, compressed - 1)
    args, _ = install(tmp_path, monkeypatch, bytes(data))
    assert fetch.main(args) == 1
    assert receipt(tmp_path)["retrieved_structure_count"] == 0


def test_manifest_mutation_fails_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, server = install(tmp_path, monkeypatch)
    server.on_request = lambda: (tmp_path / "manifest.txt").write_text("0002_A\n")
    assert fetch.main(args) == 1
    assert receipt(tmp_path)["source_inputs_unchanged"] is False
    assert (tmp_path / "raw/manifest.txt").read_text() == "0001_A\n0001_B\n"


def test_validator_source_mutation_fails_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, server = install(tmp_path, monkeypatch)
    source = tmp_path / "validator.py"
    source.write_bytes(fetch.VALIDATOR_SOURCE.read_bytes())
    monkeypatch.setattr(fetch, "VALIDATOR_SOURCE", source)
    server.on_request = lambda: source.write_text("# changed\n")
    assert fetch.main(args) == 1
    assert receipt(tmp_path)["status"] == "failed"
    assert receipt(tmp_path)["source_inputs_unchanged"] is False


@pytest.mark.parametrize("case", ["output_link", "parent_link", "manifest_link", "existing", "input_alias", "missing", "invalid", "duplicate"])
def test_path_and_manifest_rejections_do_not_modify_inputs_or_access_network(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, capsys: pytest.CaptureFixture[str]) -> None:
    args, server = install(tmp_path, monkeypatch)
    manifest = tmp_path / "manifest.txt"
    if case == "output_link":
        (tmp_path / "raw").symlink_to(tmp_path / "missing")
    elif case == "parent_link":
        (tmp_path / "alias").symlink_to(tmp_path, target_is_directory=True)
        args[3] = str(tmp_path / "alias/raw")
    elif case == "manifest_link":
        (tmp_path / "alias").symlink_to(manifest)
        args[1] = str(tmp_path / "alias")
    elif case == "existing":
        (tmp_path / "raw").mkdir()
    elif case == "input_alias":
        os.link(manifest, tmp_path / "raw")
    elif case == "missing":
        manifest.unlink()
    elif case == "invalid":
        manifest.write_text("private-hidden-invalid-target\n")
    else:
        manifest.write_text("0001_A\n0001_A\n")
    before = manifest.read_bytes() if manifest.exists() else None
    assert fetch.main(args) == 2
    assert not server.requests
    assert (manifest.read_bytes() if manifest.exists() else None) == before
    output = capsys.readouterr()
    assert "private-hidden-invalid-target" not in output.out + output.err


def test_atomic_publication_does_not_clobber_links_or_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"protected")
    for alias in (tmp_path / "regular", tmp_path / "symlink", tmp_path / "hardlink"):
        if alias.name == "symlink":
            alias.symlink_to(source)
        elif alias.name == "hardlink":
            os.link(source, alias)
        else:
            alias.write_bytes(b"protected")
        with pytest.raises(FileExistsError):
            fetch.atomic_write(alias, b"replacement")
        assert alias.read_bytes() == source.read_bytes() == b"protected"
    assert not list(tmp_path.glob(".partial-*"))


def test_large_metadata_ranges_are_chunked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = b"x" * (fetch.MAX_RANGE * 2 + 20)
    server = Server(data)
    monkeypatch.setattr(fetch, "open_range", server)
    reader = fetch.RangeReader(fetch.ArchivePin(len(data), ETAG, "null"))
    assert reader.read(3, fetch.MAX_RANGE * 2) == data[3:3 + fetch.MAX_RANGE * 2]
    assert len(server.requests) == 2


def test_selected_member_missing_fails_without_partial_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, _ = install(tmp_path, monkeypatch)
    (tmp_path / "manifest.txt").write_text("0001_A\n0003_A\n")
    assert fetch.main(args) == 1
    assert receipt(tmp_path)["retrieved_structure_count"] == 0


@pytest.mark.parametrize("etag", ["unquoted", 'W/"weak"', '"line\nbreak"'])
def test_explicit_strong_archive_pin_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, etag: str) -> None:
    args, server = install(tmp_path, monkeypatch)
    args[args.index("--archive-etag") + 1] = etag
    assert fetch.main(args) == 2
    assert not server.requests and not (tmp_path / "raw").exists()


@pytest.mark.parametrize("traversal", ["inside", "outside"])
def test_private_path_checks_use_canonical_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, traversal: str) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".nanofold_private").mkdir()
    (repo / "data").mkdir()
    (tmp_path / "other").mkdir()
    args, server = install(tmp_path, monkeypatch)
    monkeypatch.setattr(fetch, "REPO_ROOT", repo)
    output = (repo / ".nanofold_private/../data/new") if traversal == "inside" else (tmp_path / "other/../repo/data/new")
    args[3] = str(output)
    assert fetch.main(args) == 2
    assert not (repo / "data/new").exists() and not server.requests


def test_manifest_replacement_by_directory_retains_failed_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, server = install(tmp_path, monkeypatch)

    def replace_manifest() -> None:
        manifest = tmp_path / "manifest.txt"
        if manifest.is_file():
            manifest.unlink()
            manifest.mkdir()

    server.on_request = replace_manifest
    assert fetch.main(args) == 1
    assert receipt(tmp_path)["status"] == "failed"
    assert not receipt(tmp_path)["source_inputs_unchanged"]


def test_final_archive_pin_check_cannot_be_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, server = install(tmp_path, monkeypatch)
    # ZIP64 fixture: two tail ranges, end record, directory, header, name/extra, body, final pin.
    server.fault = "version"
    server.fault_at = 8
    assert fetch.main(args) == 1
    assert receipt(tmp_path)["status"] == "failed"
    assert receipt(tmp_path)["retrieved_structure_count"] == 1
    assert server.requests[-1] == (0, 0)


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_real_redirect_handler_rejects_before_draining_body_or_following(code: int) -> None:
    headers = HTTPMessage()
    headers["Location"] = "http://other.example/archive"
    response = Response(b"unbounded redirect body", headers, code, fetch.ARCHIVE_URL)
    handler = fetch.NoRedirect()
    request = Request(fetch.ARCHIVE_URL)
    # The stdlib calls redirect_request before fp.read() or parent.open(). A parent
    # is intentionally absent: following would raise AttributeError instead.
    with pytest.raises(ValueError, match="redirects are not permitted"):
        handler.http_error_302(request, response, code, "redirect", headers)
    assert response.read_calls == 0


@pytest.mark.parametrize("failure", ["reset", "timeout", "temporary_dns", "429", "500", "503"])
def test_transient_failures_retry_the_entire_pinned_range_with_finite_backoff(
        monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    server = Server(make_zip())
    delays: list[int] = []
    requests: list[tuple[str, str | None, str | None]] = []
    error_bodies: list[Response] = []

    def flaky(request: Request, timeout: int) -> Response:
        requests.append((request.full_url, request.get_header("Range"), request.get_header("If-match")))
        if len(requests) <= 3:
            if failure.isdigit():
                body = Response(b"do not read error bodies", Message(), int(failure), request.full_url)
                error_bodies.append(body)
                raise HTTPError(request.full_url, int(failure), "transient", Message(), body)
            if failure == "reset":
                reason: OSError = ConnectionResetError(errno.ECONNRESET, "Connection reset by peer")
            elif failure == "timeout":
                reason = TimeoutError("timed out")
            else:
                reason = socket.gaierror(socket.EAI_AGAIN, "temporary lookup failure")
            raise URLError(reason)
        return server(request, timeout)

    monkeypatch.setattr(fetch, "open_range", flaky)
    monkeypatch.setattr(fetch.time, "sleep", delays.append)
    reader = fetch.RangeReader(fetch.ArchivePin(len(server.data), ETAG, "null"))
    assert reader.read(5, 20) == server.data[5:25]
    assert len(requests) == 4 and len(set(requests)) == 1
    assert delays == [1, 2, 4]
    assert [row["verified"] for row in reader.ranges[0]["attempts"]] == [False, False, False, True]
    assert all(body.closed and body.read_calls == 0 for body in error_bodies)


def test_exhausted_transport_retries_leave_failed_receipt_and_no_cifs(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, _ = install(tmp_path, monkeypatch)
    requests: list[str | None] = []
    delays: list[int] = []

    def always_reset(request: Request, timeout: int) -> Response:
        requests.append(request.get_header("Range"))
        raise URLError(ConnectionResetError(errno.ECONNRESET, "Connection reset by peer"))

    monkeypatch.setattr(fetch, "open_range", always_reset)
    monkeypatch.setattr(fetch.time, "sleep", delays.append)
    assert fetch.main(args) == 1
    assert len(requests) == 4 and len(set(requests)) == 1 and delays == [1, 2, 4]
    result = receipt(tmp_path)
    assert result["status"] == "failed" and result["source_inputs_unchanged"]
    assert len(result["ranges"][0]["attempts"]) == 4
    assert not any(row["verified"] for row in result["ranges"][0]["attempts"])
    assert not list((tmp_path / "raw").rglob("*.cif"))


def test_connection_reset_after_partial_body_restarts_whole_range_before_publication(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, server = install(tmp_path, monkeypatch)
    partial_lengths: list[int] = []
    delays: list[int] = []

    def reset_member_body_once(request: Request, timeout: int) -> Response:
        response = server(request, timeout)
        if len(server.requests) == 7:
            def reset_after_prefix(size: int) -> bytes:
                partial_lengths.append(len(io.BytesIO.read(response, size // 2)))
                assert not (tmp_path / "raw/pdb_data/mmcif_files/0001.cif").exists()
                raise ConnectionResetError(errno.ECONNRESET, "Connection reset by peer")
            monkeypatch.setattr(response, "read", reset_after_prefix)
        return response

    monkeypatch.setattr(fetch, "open_range", reset_member_body_once)
    monkeypatch.setattr(fetch.time, "sleep", delays.append)
    assert fetch.main(args) == 0
    assert partial_lengths and partial_lengths[0] > 0 and delays == [1]
    assert server.requests[6] == server.requests[7]
    assert (tmp_path / "raw/pdb_data/mmcif_files/0001.cif").read_bytes() == PAYLOAD
    result = receipt(tmp_path)
    assert len(result["ranges"][6]["attempts"]) == 2
    assert result["members"][0]["crc32_verified"]


@pytest.mark.parametrize("failure", ["403", "404", "412", "certificate", "permanent_dns", "permission"])
def test_permanent_request_failures_are_not_retried(
        monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    calls = 0
    delays: list[int] = []

    def rejected(request: Request, timeout: int) -> Response:
        nonlocal calls
        calls += 1
        if failure.isdigit():
            raise HTTPError(request.full_url, int(failure), "permanent", Message(), io.BytesIO())
        if failure == "certificate":
            raise URLError(ssl.SSLCertVerificationError("certificate verification failed"))
        if failure == "permanent_dns":
            raise URLError(socket.gaierror(socket.EAI_NONAME, "permanent lookup failure"))
        raise PermissionError(errno.EACCES, "permission denied")

    monkeypatch.setattr(fetch, "open_range", rejected)
    monkeypatch.setattr(fetch.time, "sleep", delays.append)
    reader = fetch.RangeReader(fetch.ArchivePin(100, ETAG, "null"))
    with pytest.raises(OSError):
        reader.read(1, 5)
    assert calls == 1 and delays == []


@pytest.mark.parametrize("fault", ["etag", "version", "range", "length", "short", "long"])
def test_integrity_failure_after_transport_retry_is_terminal(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    args, server = install(tmp_path, monkeypatch, fault=fault)
    calls = 0
    delays: list[int] = []

    def reset_then_invalid(request: Request, timeout: int) -> Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise URLError(ConnectionResetError(errno.ECONNRESET, "Connection reset by peer"))
        return server(request, timeout)

    monkeypatch.setattr(fetch, "open_range", reset_then_invalid)
    monkeypatch.setattr(fetch.time, "sleep", delays.append)
    assert fetch.main(args) == 1
    assert calls == 2 and delays == [1]
    assert receipt(tmp_path)["error"]["type"] == "ValueError"
    assert not list((tmp_path / "raw").rglob("*.cif"))
