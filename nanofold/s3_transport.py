"""Unsigned pooled S3 retrieval with pinned objects and durable chain receipts."""
from __future__ import annotations

import fnmatch
import hashlib
import importlib
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any, Sequence
from urllib.parse import urlsplit

RECEIPT = ".nanofold-s3-receipt.json"


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _relative(key: str, prefix: str) -> str:
    relative = key.removeprefix(prefix)
    parts = relative.split("/")
    if (not key.startswith(prefix) or not relative or "\\" in relative or "\x00" in relative
            or any(part in {"", ".", ".."} for part in parts) or PurePosixPath(relative).is_absolute()
            or RECEIPT in parts):
        raise ValueError("Unsafe or ambiguous S3 object path")
    return relative


class PooledS3Transport:
    def __init__(self, roda_root: Path, *, workers: int, timeout: float, client: Any = None):
        self.root = roda_root.resolve()
        if client is None:
            try:
                session = importlib.import_module("botocore.session")
                unsigned = importlib.import_module("botocore").UNSIGNED
                config = importlib.import_module("botocore.config").Config
            except ModuleNotFoundError as exc:
                raise RuntimeError("pooled-s3 requires the optional botocore package") from exc
            client = session.get_session().create_client(
                "s3", region_name="us-east-1",
                config=config(signature_version=unsigned, max_pool_connections=max(1, workers),
                              connect_timeout=timeout if timeout > 0 else None,
                              read_timeout=timeout if timeout > 0 else None, retries={"max_attempts": 0}),
            )
        self.client = client
        self.guard = threading.Lock()
        self.locks: dict[str, Any] = {}
        self.plans: dict[str, list[dict[str, Any]]] = {}

    def _lock(self, path: Path) -> Any:
        with self.guard:
            return self.locks.setdefault(str(path), threading.Lock())

    def _destination(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root) or resolved == self.root:
            raise ValueError("S3 destination escapes its raw-chain root")
        return resolved

    def _verified_receipt(self, directory: Path) -> dict[str, Any] | None:
        directory = self._destination(directory)
        receipt = directory / RECEIPT
        if receipt.is_symlink():
            raise ValueError("S3 receipt must not be a symlink")
        if not receipt.is_file():
            return None
        value = json.loads(receipt.read_text())
        if value.get("status") != "complete" or not value.get("objects"):
            return None
        for row in value["objects"]:
            if row["bucket"] != value["bucket"]:
                return None
            relative = _relative(row["key"], value["prefix"])
            path = directory / relative
            if not path.resolve().is_relative_to(directory) or not path.is_file():
                return None
            if path.stat().st_size != row["size"] or _sha(path) != row["sha256"]:
                return None
        return value

    def complete(self, directory: Path) -> bool:
        return self._verified_receipt(directory) is not None

    def required_files(self, directory: Path, *, bucket: str, msa_names: Sequence[str],
                       template_hits_name: str, need_template_hits: bool,
                       full_chain_dir: bool) -> tuple[bool, bool]:
        receipt = self._verified_receipt(directory)
        if receipt is None:
            return False, not need_template_hits
        parsed = urlsplit(bucket)
        # A local alias may use a duplicate's canonical source identity, but never
        # a different bucket or base prefix. Only receipt-covered bytes count.
        source = receipt["source_chain_id"]
        prefix = "/".join(part for part in (parsed.path.strip("/"), "pdb", source) if part) + "/"
        if (parsed.scheme != "s3" or parsed.query or parsed.fragment
                or receipt["bucket"] != parsed.netloc or receipt["prefix"] != prefix
                or (full_chain_dir and receipt["patterns"] is not None)
                or (full_chain_dir and need_template_hits and receipt.get("excluded_patterns"))):
            return False, not need_template_hits
        files = [_relative(row["key"], prefix) for row in receipt["objects"] if row["size"] > 0]

        def covered(name: str) -> bool:
            return any(fnmatch.fnmatchcase(path, name) or fnmatch.fnmatchcase(path, f"*/{name}") for path in files)

        return all(covered(name) for name in msa_names), not need_template_hits or covered(template_hits_name)

    def _event(self, value: dict[str, Any]) -> None:
        import fcntl

        self.root.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.root / ".s3-acquisition.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        with self.guard, os.fdopen(descriptor, "a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _plan(self, bucket: str, prefix: str, patterns: list[str] | None,
              excluded_patterns: list[str]) -> list[dict[str, Any]]:
        cache_key = json.dumps([bucket, prefix, patterns, excluded_patterns])
        if cache_key in self.plans:
            return self.plans[cache_key]
        objects = []
        seen = set()
        for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for entry in page.get("Contents", []):
                key = entry["Key"]
                if key.endswith("/"):
                    continue
                relative = _relative(key, prefix)
                if any(fnmatch.fnmatchcase(relative, pattern) for pattern in excluded_patterns):
                    continue
                if patterns is not None and not any(fnmatch.fnmatchcase(relative, pattern) for pattern in patterns):
                    continue
                if key in seen:
                    raise ValueError("S3 listing repeats an object")
                seen.add(key)
                head = self.client.head_object(Bucket=bucket, Key=key)
                if (head["ETag"] != entry["ETag"] or head["ContentLength"] != entry["Size"]
                        or not isinstance(head["ETag"], str) or not head["ETag"]):
                    raise ValueError("S3 object changed between listing and version pinning")
                objects.append({"bucket": bucket, "key": key, "etag": head["ETag"],
                                "version_id": head.get("VersionId"), "size": head["ContentLength"]})
        objects.sort(key=lambda row: row["key"])
        self._event({"status": "source_objects_pinned", "bucket": bucket, "prefix": prefix, "objects": objects})
        self.plans[cache_key] = objects
        return objects

    def _object(self, row: dict[str, Any], destination: Path) -> dict[str, Any]:
        request = {"Bucket": row["bucket"], "Key": row["key"], "IfMatch": row["etag"]}
        if row["version_id"] is not None:
            request["VersionId"] = row["version_id"]
        response = self.client.get_object(**request)
        body = response["Body"]
        try:
            if (response["ContentLength"] != row["size"] or response["ETag"] != row["etag"]
                    or response.get("VersionId") != row["version_id"]):
                raise ValueError("S3 response differs from its pinned object version")
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            size = 0
            with destination.open("xb") as handle:
                while block := body.read(1024 * 1024):
                    size += len(block)
                    if size > row["size"]:
                        raise OSError("S3 response exceeds pinned object size")
                    digest.update(block)
                    handle.write(block)
                handle.flush()
                os.fsync(handle.fileno())
            if size != row["size"]:
                raise OSError("S3 response is truncated")
            return row | {"sha256": digest.hexdigest()}
        finally:
            body.close()

    def download_chain(self, cid: str, out_dir: Path, *, bucket: str, msa_names: list[str],
                       template_hits_name: str, no_template_hits: bool, full_chain_dir: bool,
                       retries: int, retry_delay_seconds: float) -> bool:
        parsed = urlsplit(bucket)
        if (parsed.scheme != "s3" or not parsed.netloc or parsed.query or parsed.fragment
                or not re.fullmatch(r"[A-Za-z0-9]{4}_[^\s/\\]+", cid)):
            raise ValueError("Invalid S3 bucket or source chain identity")
        prefix = "/".join(part for part in (parsed.path.strip("/"), "pdb", cid) if part) + "/"
        names = [*msa_names, *([] if no_template_hits else [template_hits_name])]
        patterns = None if full_chain_dir else [pattern for name in names for pattern in (name, f"*/{name}")]
        excluded = [template_hits_name, f"*/{template_hits_name}"] if no_template_hits else []
        out_dir = self._destination(out_dir)
        with self._lock(out_dir):
            out_dir.parent.mkdir(parents=True, exist_ok=True)
            if self.complete(out_dir):
                cached = json.loads((out_dir / RECEIPT).read_text())
                if (cached["bucket"] == parsed.netloc and cached["prefix"] == prefix
                        and cached["patterns"] == patterns and cached.get("excluded_patterns", []) == excluded):
                    return True
            for attempt in range(max(0, retries) + 1):
                try:
                    objects = self._plan(parsed.netloc, prefix, patterns, excluded)
                    if not objects:
                        return True
                    with tempfile.TemporaryDirectory(dir=out_dir.parent, prefix=".s3-download-") as temporary:
                        staging = Path(temporary)
                        downloaded = [self._object(row, staging / _relative(row["key"], prefix)) for row in objects]
                        receipt = {"status": "complete", "source_chain_id": cid, "bucket": parsed.netloc,
                                   "prefix": prefix, "patterns": patterns, "excluded_patterns": excluded,
                                   "objects": downloaded}
                        receipt_path = staging / RECEIPT
                        with receipt_path.open("x") as handle:
                            handle.write(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
                            handle.flush()
                            os.fsync(handle.fileno())
                        # Validate every existing destination before publishing any new file.
                        for row in downloaded:
                            destination = out_dir / _relative(row["key"], prefix)
                            if not destination.resolve().is_relative_to(out_dir):
                                raise ValueError("S3 output path crosses a symlink outside its chain")
                            if destination.exists() and (not destination.is_file() or _sha(destination) != row["sha256"]):
                                raise ValueError("S3 acquisition conflicts with existing local source bytes")
                        if not out_dir.exists():
                            staging.rename(out_dir)
                        else:
                            for source in sorted(staging.rglob("*")):
                                if source.is_file() and source.name != RECEIPT:
                                    destination = out_dir / source.relative_to(staging)
                                    destination.parent.mkdir(parents=True, exist_ok=True)
                                    source.replace(destination)
                            receipt_path.replace(out_dir / RECEIPT)
                        directory_fd = os.open(out_dir.parent, os.O_RDONLY)
                        try:
                            os.fsync(directory_fd)
                        finally:
                            os.close(directory_fd)
                        self._event(receipt | {"status": "published", "output": str(out_dir)})
                    return True
                except Exception as exc:
                    self._event({"status": "failed_attempt", "source_chain_id": cid, "prefix": prefix,
                                 "attempt": attempt, "error_type": type(exc).__name__, "error": str(exc)})
                    if isinstance(exc, ValueError):
                        raise
                    if attempt < retries:
                        time.sleep(max(0., retry_delay_seconds) * (attempt + 1))
            return False
