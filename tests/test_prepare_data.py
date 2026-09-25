from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest


def _load_prepare_data_module():
    module_path = Path("scripts/prepare_data.py").resolve()
    spec = importlib.util.spec_from_file_location("prepare_data_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_structure_url_and_destination_are_manifest_chain_based(tmp_path: Path) -> None:
    module = _load_prepare_data_module()
    structure_url_and_destination = getattr(module, "_structure_url_and_destination")

    url, destination = structure_url_and_destination("1abc_A", tmp_path)

    assert url == "https://files.rcsb.org/download/1ABC.cif"
    assert destination == tmp_path / "pdb_data" / "mmcif_files" / "1abc.cif"


def test_download_mmcif_subset_deduplicates_pdb_ids(tmp_path: Path, monkeypatch) -> None:
    module = _load_prepare_data_module()
    download_mmcif_subset = getattr(module, "_download_mmcif_subset")
    calls: list[tuple[str, Path]] = []

    def fake_download_url(url, destination, **kwargs):  # noqa: ANN001, ANN202
        calls.append((url, destination))
        return True

    monkeypatch.setattr(module, "_download_url", fake_download_url)

    download_mmcif_subset(
        ["1abc_A", "1abc_B", "2xyz_A"],
        tmp_path,
        dry_run=False,
        retries=0,
        retry_delay_seconds=0.0,
        workers=1,
    )

    assert [url for url, _ in calls] == [
        "https://files.rcsb.org/download/1ABC.cif",
        "https://files.rcsb.org/download/2XYZ.cif",
    ]


def test_download_url_uses_timeout(tmp_path: Path, monkeypatch) -> None:
    module = _load_prepare_data_module()
    download_url = getattr(module, "_download_url")
    seen: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, *args):  # noqa: ANN002, ANN204
            return False

        def read(self) -> bytes:
            return b"data_test"

    def fake_urlopen(url, timeout):  # noqa: ANN001, ANN202
        seen["url"] = url
        seen["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(module, "urlopen", fake_urlopen)

    destination = tmp_path / "pdb_data" / "mmcif_files" / "1abc.cif"
    ok = download_url(
        "https://files.rcsb.org/download/1ABC.cif",
        destination,
        dry_run=False,
        retries=0,
        retry_delay_seconds=0.0,
        allow_fail=False,
        timeout_seconds=12.5,
    )

    assert ok is True
    assert seen == {"url": "https://files.rcsb.org/download/1ABC.cif", "timeout": 12.5}
    assert destination.read_bytes() == b"data_test"


@pytest.mark.parametrize("timeout_seconds", [0.0, -1.0])
def test_download_url_can_disable_timeout(tmp_path: Path, monkeypatch, timeout_seconds: float) -> None:
    module = _load_prepare_data_module()

    def fake_urlopen(url):
        return io.BytesIO(b"downloaded")

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    destination = tmp_path / "download.cif"
    assert module._download_url(
        "https://example.invalid/structure.cif", destination, dry_run=False,
        retries=0, retry_delay_seconds=0.0, allow_fail=False,
        timeout_seconds=timeout_seconds,
    )
    assert destination.read_bytes() == b"downloaded"


def test_download_timeout_retries_without_partial_destination(tmp_path: Path, monkeypatch) -> None:
    module = _load_prepare_data_module()
    attempts = []

    def fake_urlopen(url, timeout):
        attempts.append(timeout)
        raise TimeoutError("timed out")

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    destination = tmp_path / "download.cif"
    assert not module._download_url(
        "https://example.invalid/structure.cif", destination, dry_run=False,
        retries=2, retry_delay_seconds=0.0, allow_fail=True,
        timeout_seconds=3.0,
    )
    assert attempts == [3.0, 3.0, 3.0]
    assert not destination.exists()


def test_download_mmcif_subset_strict_fails_on_missing_download(tmp_path: Path, monkeypatch) -> None:
    module = _load_prepare_data_module()
    download_mmcif_subset = getattr(module, "_download_mmcif_subset")

    def fake_download_url(url, destination, **kwargs):  # noqa: ANN001, ANN202
        return False

    monkeypatch.setattr(module, "_download_url", fake_download_url)

    try:
        download_mmcif_subset(
            ["1abc_A"],
            tmp_path,
            dry_run=False,
            retries=0,
            retry_delay_seconds=0.0,
            workers=1,
            strict=True,
        )
    except SystemExit as exc:
        assert "Missing required mmCIF downloads" in str(exc)
    else:
        raise AssertionError("strict mmCIF subset download should fail on missing files")
