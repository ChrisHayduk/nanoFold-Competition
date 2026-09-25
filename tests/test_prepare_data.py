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


def test_prepare_dry_run_creates_no_directories(tmp_path: Path, monkeypatch) -> None:
    module = _load_prepare_data_module()
    manifest = tmp_path / "chains.txt"
    manifest.write_text("1abc_A\n")
    root = tmp_path / "not-created"
    monkeypatch.setattr(sys, "argv", ["prepare_data.py", "--manifest", str(manifest), "--data-root", str(root),
                                    "--dry-run", "--strict-downloads", "--download-mmcif-subset", "--no-template-hits"])
    module.main()
    assert not root.exists()


@pytest.mark.parametrize("invalid", ["directory", "empty"])
def test_required_alignment_assets_must_be_nonempty_regular_files(tmp_path: Path, invalid: str) -> None:
    module = _load_prepare_data_module()
    asset = tmp_path / "uniref90_hits.a3m"
    if invalid == "directory":
        asset.mkdir()
    else:
        asset.touch()
    assert module._has_required_files(tmp_path, [asset.name], "hits.hhr", False) == (False, True)


def test_failed_aws_download_does_not_publish_partial_assets(tmp_path: Path, monkeypatch) -> None:
    module = _load_prepare_data_module()
    out = tmp_path / "chain"
    out.mkdir()
    existing = out / "uniref90_hits.a3m"
    existing.write_text(">query\nCOMPLETE\n")

    def failed_transfer(command, *args, **kwargs):
        destination = Path(command[4])
        (destination / "uniref90_hits.a3m").write_text(">query\nPART")
        return False

    monkeypatch.setattr(module, "run", failed_transfer)
    assert not module._download_chain("1abc_A", out, bucket="s3://openfold", msa_names=[existing.name],
                                      template_hits_name="hits.hhr", no_template_hits=True,
                                      full_chain_dir=False, dry_run=False, retries=0, retry_delay_seconds=0)
    assert existing.read_text() == ">query\nCOMPLETE\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["chain"]


def test_partial_existing_chain_still_downloads_missing_templates(tmp_path: Path, monkeypatch) -> None:
    module = _load_prepare_data_module()
    root = tmp_path / "roda"
    chain = module.chain_data_dir(root, "1abc_A")
    chain.mkdir(parents=True)
    (chain / "uniref90_hits.a3m").write_text(">query\nAAAA\n")
    calls = []

    def transfer(cid, destination, **kwargs):
        calls.append(cid)
        (destination / "hits.hhr").write_text("template hits")
        return True

    monkeypatch.setattr(module, "_download_chain", transfer)
    missing = module._process_chain("1abc_A", roda_root=root, duplicate_map={}, bucket="s3://openfold",
                                    msa_names=["uniref90_hits.a3m"], template_hits_name="hits.hhr",
                                    no_template_hits=False, full_chain_dir=False, dry_run=False,
                                    retries=0, retry_delay_seconds=0)
    assert missing == []
    assert calls == ["1abc_A"]


def test_fallback_does_not_report_unlinked_template_as_present(tmp_path: Path) -> None:
    module = _load_prepare_data_module()
    root = tmp_path / "roda"
    target = module.chain_data_dir(root, "1abc_A")
    representative = module.chain_data_dir(root, "2abc_A")
    for chain in (target, representative):
        chain.mkdir(parents=True)
        (chain / "uniref90_hits.a3m").write_text(">query\nAAAA\n")
    (representative / "hits.hhr").write_text("template hits")
    missing = module._process_chain("1abc_A", roda_root=root, duplicate_map={"1abc_A": ["2abc_A"]},
                                    bucket="s3://openfold", msa_names=["uniref90_hits.a3m"],
                                    template_hits_name="hits.hhr", no_template_hits=False,
                                    full_chain_dir=False, dry_run=False, retries=0, retry_delay_seconds=0)
    assert missing == ["1abc_A:hits.hhr"]


def test_failed_response_body_leaves_no_partial_download(tmp_path: Path, monkeypatch) -> None:
    module = _load_prepare_data_module()

    class Interrupted(io.BytesIO):
        def read(self, *args, **kwargs):
            raise OSError("response interrupted")

    monkeypatch.setattr(module, "urlopen", lambda *args, **kwargs: Interrupted())
    assert not module._download_url("https://example.invalid/file", tmp_path / "file.cif", dry_run=False,
                                    retries=0, retry_delay_seconds=0, allow_fail=True)
    assert list(tmp_path.iterdir()) == []


def test_successful_retry_cannot_publish_previous_failed_attempt_bytes(tmp_path: Path, monkeypatch) -> None:
    module = _load_prepare_data_module()
    attempts = []

    def transfer(command):
        destination = Path(command[4])
        attempts.append(destination)
        if len(attempts) == 1:
            (destination / "uniref90_hits.a3m").write_text(">query\nPART")
            raise module.subprocess.CalledProcessError(1, command)
        # An empty recursive listing is a successful AWS command, but has no assets.

    monkeypatch.setattr(module.subprocess, "check_call", transfer)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    out = tmp_path / "chain"
    module._download_chain("1abc_A", out, bucket="s3://openfold", msa_names=["uniref90_hits.a3m"],
                           template_hits_name="hits.hhr", no_template_hits=True,
                           full_chain_dir=False, dry_run=False, retries=1, retry_delay_seconds=0)
    assert len(attempts) == 2
    assert attempts[0] != attempts[1]
    assert not module._has_file(out, "uniref90_hits.a3m")
    assert list(tmp_path.iterdir()) == []
