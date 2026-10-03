from __future__ import annotations

import hashlib
import io
import json
import threading
from pathlib import Path

import pytest

from nanofold.chain_paths import chain_data_dir
from nanofold.s3_transport import RECEIPT, PooledS3Transport
from scripts.prepare_data import _download_manifest_chains


class FakeS3:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects
        self.requests: list[dict] = []
        self.heads: list[str] = []
        self.short: set[str] = set()
        self.guard = threading.Lock()

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        return self

    def paginate(self, **kwargs):
        entries = [{"Key": key, "ETag": '"etag"', "Size": len(data)}
                   for key, data in self.objects.items() if key.startswith(kwargs["Prefix"])]
        yield {"Contents": entries[:1]}
        yield {"Contents": entries[1:]}

    def head_object(self, **kwargs):
        self.heads.append(kwargs["Key"])
        return {"ETag": '"etag"', "VersionId": "v1", "ContentLength": len(self.objects[kwargs["Key"]])}

    def get_object(self, **kwargs):
        with self.guard:
            self.requests.append(kwargs)
        data = self.objects[kwargs["Key"]]
        return {"ETag": '"etag"', "VersionId": "v1", "ContentLength": len(data),
                "Body": io.BytesIO(data[:-1] if kwargs["Key"] in self.short else data)}


def _download(transport: PooledS3Transport, out: Path, *, full: bool = False, retries: int = 0) -> bool:
    return transport.download_chain("1abc_A", out, bucket="s3://openfold", msa_names=["hits.a3m"],
                                    template_hits_name="pdb70_hits.hhr", no_template_hits=True,
                                    full_chain_dir=full, retries=retries, retry_delay_seconds=0)


@pytest.mark.parametrize("full", [False, True])
def test_paginated_recursive_filters_and_exact_object_provenance(tmp_path: Path, full: bool) -> None:
    objects = {"pdb/1abc_A/hits.a3m": b"root", "pdb/1abc_A/a/b/hits.a3m": b"nested",
               "pdb/1abc_A/a/pdb70_hits.hhr": b"template", "pdb/1abc_A/other.txt": b"other"}
    client = FakeS3(objects)
    transport = PooledS3Transport(tmp_path, workers=2, timeout=10, client=client)
    out = tmp_path / "chain"
    assert _download(transport, out, full=full)
    wanted = set(objects) - {"pdb/1abc_A/a/pdb70_hits.hhr"}
    if not full:
        wanted.remove("pdb/1abc_A/other.txt")
    assert {row["Key"] for row in client.requests} == wanted
    assert all(row["VersionId"] == "v1" and row["IfMatch"] == '"etag"' for row in client.requests)
    receipt = json.loads((out / RECEIPT).read_text())
    for row in receipt["objects"]:
        assert row["bucket"] == "openfold" and row["version_id"] == "v1"
        assert row["size"] == len(objects[row["key"]])
        assert row["sha256"] == hashlib.sha256(objects[row["key"]]).hexdigest()
    assert transport.complete(out)
    assert [json.loads(line)["status"] for line in (tmp_path / ".s3-acquisition.jsonl").read_text().splitlines()] == [
        "source_objects_pinned", "published",
    ]


def test_later_truncated_object_never_publishes_partial_chain(tmp_path: Path) -> None:
    client = FakeS3({"pdb/1abc_A/a/hits.a3m": b"complete", "pdb/1abc_A/z/hits.a3m": b"truncated"})
    client.short.add("pdb/1abc_A/z/hits.a3m")
    transport = PooledS3Transport(tmp_path, workers=1, timeout=10, client=client)
    assert not _download(transport, tmp_path / "chain", retries=1)
    assert not (tmp_path / "chain").exists()
    assert not list(tmp_path.glob(".s3-download-*"))
    assert len(client.requests) == 4 and len(client.heads) == 2


@pytest.mark.parametrize("unsafe", ["../hits.a3m", "a/../hits.a3m", "/hits.a3m", "a//hits.a3m"])
def test_unsafe_keys_fail_before_payload_get(tmp_path: Path, unsafe: str) -> None:
    client = FakeS3({"pdb/1abc_A/" + unsafe: b"unsafe"})
    transport = PooledS3Transport(tmp_path, workers=1, timeout=10, client=client)
    with pytest.raises(ValueError, match="Unsafe"):
        _download(transport, tmp_path / "chain")
    assert client.requests == []


def test_external_symlink_destination_is_rejected(tmp_path: Path) -> None:
    root, external = tmp_path / "root", tmp_path / "external"
    root.mkdir()
    external.mkdir()
    (root / "chain").symlink_to(external, target_is_directory=True)
    client = FakeS3({"pdb/1abc_A/hits.a3m": b"data"})
    transport = PooledS3Transport(root, workers=1, timeout=10, client=client)
    with pytest.raises(ValueError, match="escapes"):
        _download(transport, root / "chain")
    assert client.requests == [] and not list(external.iterdir())


def test_conflicting_existing_bytes_are_preserved(tmp_path: Path) -> None:
    out = tmp_path / "chain"
    out.mkdir()
    (out / "hits.a3m").write_bytes(b"original")
    client = FakeS3({"pdb/1abc_A/hits.a3m": b"different", "pdb/1abc_A/new/hits.a3m": b"new"})
    transport = PooledS3Transport(tmp_path, workers=1, timeout=10, client=client)
    with pytest.raises(ValueError, match="conflicts"):
        _download(transport, out)
    assert (out / "hits.a3m").read_bytes() == b"original"
    assert not (out / "new").exists() and not (out / RECEIPT).exists()


def test_parallel_duplicates_download_canonical_source_once(tmp_path: Path) -> None:
    client = FakeS3({"pdb/3abc_A/a3m/hits.a3m": b">q\nAAAA\n"})
    transport = PooledS3Transport(tmp_path, workers=2, timeout=10, client=client)
    missing = _download_manifest_chains(
        ["1abc_A", "2abc_A"], roda_root=tmp_path,
        duplicate_map={"1abc_A": ["3abc_A"], "2abc_A": ["3abc_A"]}, bucket="s3://openfold",
        msa_names=["hits.a3m"], template_hits_name="pdb70_hits.hhr", no_template_hits=True,
        full_chain_dir=False, dry_run=False, retries=0, retry_delay_seconds=0, workers=2, pooled_s3=transport,
    )
    assert missing == [] and len(client.requests) == 1
    assert chain_data_dir(tmp_path, "1abc_A").resolve() == chain_data_dir(tmp_path, "3abc_A")
    assert chain_data_dir(tmp_path, "2abc_A").resolve() == chain_data_dir(tmp_path, "3abc_A")


def test_unreceipted_existing_msa_cannot_satisfy_required_files(tmp_path: Path) -> None:
    from scripts.prepare_data import _has_required_files

    client = FakeS3({"pdb/1abc_A/hits.a3m": b"verified"})
    transport = PooledS3Transport(tmp_path, workers=1, timeout=10, client=client)
    out = tmp_path / "chain"
    assert _download(transport, out)
    (out / "hits2.a3m").write_bytes(b"unverified")
    assert _has_required_files(out, ["hits.a3m", "hits2.a3m"], "pdb70_hits.hhr", False, transport) == (False, True)


@pytest.mark.parametrize("changed", ["bucket", "full_directory"])
def test_changed_request_cannot_skip_pinned_acquisition(tmp_path: Path, changed: str) -> None:
    from scripts.prepare_data import _process_chain

    client = FakeS3({"pdb/1abc_A/hits.a3m": b"verified", "pdb/1abc_A/other.txt": b"extra"})
    transport = PooledS3Transport(tmp_path, workers=1, timeout=10, client=client)
    out = chain_data_dir(tmp_path, "1abc_A")
    assert _download(transport, out)
    client.requests.clear()
    assert _process_chain(
        "1abc_A", roda_root=tmp_path, duplicate_map={},
        bucket="s3://new-bucket" if changed == "bucket" else "s3://openfold",
        msa_names=["hits.a3m"], template_hits_name="pdb70_hits.hhr", no_template_hits=True,
        full_chain_dir=changed == "full_directory", dry_run=False,
        retries=0, retry_delay_seconds=0, pooled_s3=transport,
    ) == []
    assert client.requests
    if changed == "bucket":
        assert all(request["Bucket"] == "new-bucket" for request in client.requests)
    else:
        assert (out / "other.txt").read_bytes() == b"extra"


@pytest.mark.parametrize("target", ["receipt", "journal"])
def test_metadata_symlinks_cannot_read_or_write_outside_root(tmp_path: Path, target: str) -> None:
    root = tmp_path / "root"
    out = root / "chain"
    out.mkdir(parents=True)
    external = tmp_path / "external"
    external.write_bytes(b"preserve")
    location = out / RECEIPT if target == "receipt" else root / ".s3-acquisition.jsonl"
    location.symlink_to(external)
    client = FakeS3({"pdb/1abc_A/hits.a3m": b"data"})
    transport = PooledS3Transport(root, workers=1, timeout=10, client=client)
    with pytest.raises((ValueError, OSError)):
        _download(transport, out)
    assert external.read_bytes() == b"preserve" and not client.requests


@pytest.mark.parametrize("changed_field", ["VersionId", "ETag"])
def test_response_identity_change_never_publishes(tmp_path: Path, changed_field: str) -> None:
    class ChangedResponse(FakeS3):
        def get_object(self, **kwargs):
            response = super().get_object(**kwargs)
            response[changed_field] = "changed"
            return response

    client = ChangedResponse({"pdb/1abc_A/hits.a3m": b"data"})
    transport = PooledS3Transport(tmp_path, workers=1, timeout=10, client=client)
    with pytest.raises(ValueError, match="pinned object version"):
        _download(transport, tmp_path / "chain")
    assert not (tmp_path / "chain").exists()


def test_pooled_dry_run_does_not_import_client_or_create_output(tmp_path: Path, monkeypatch) -> None:
    import sys

    from scripts import prepare_data

    manifest = tmp_path / "manifest.txt"
    manifest.write_text("1abc_A\n")
    output = tmp_path / "raw"
    monkeypatch.setitem(sys.modules, "botocore", None)
    monkeypatch.setattr(sys, "argv", ["prepare_data.py", "--manifest", str(manifest),
                                     "--data-root", str(output), "--download-transport", "pooled-s3", "--dry-run"])
    prepare_data.main()
    assert not output.exists()
