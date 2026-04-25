from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


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
