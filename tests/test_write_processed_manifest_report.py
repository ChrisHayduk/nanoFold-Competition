from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from nanofold.chain_paths import chain_error_path, chain_npz_path
from scripts.write_processed_manifest_report import build_report, main


def _touch_npz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, value=np.asarray(1, dtype=np.int32))


def test_write_processed_manifest_report_counts_missing_and_errors(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("1aaa_A\n2bbb_B\n3ccc_C\n")
    features = tmp_path / "features"
    labels = tmp_path / "labels"
    out_manifest = tmp_path / "processed.txt"

    _touch_npz(chain_npz_path(features, "1aaa_A"))
    _touch_npz(chain_npz_path(labels, "1aaa_A"))
    _touch_npz(chain_npz_path(features, "2bbb_B"))
    chain_error_path(features, "2bbb_B").write_text("Projection valid C-alpha count below threshold\nextra\n")
    chain_error_path(features, "3ccc_C").write_text("Missing MSA files\n")

    report = build_report(
        source_manifest=source,
        processed_features_dir=features,
        processed_labels_dir=labels,
        output_manifest=out_manifest,
        max_examples=10,
    )

    assert out_manifest.read_text() == "1aaa_A\n"
    assert report["requested_count"] == 3
    assert report["processed_count"] == 1
    assert report["missing_count"] == 2
    assert report["feature_only_count"] == 1
    assert report["error_summary"] == {
        "Projection valid C-alpha count below threshold": 1,
        "Missing MSA files": 1,
    }


@pytest.mark.parametrize("available,require_complete", [(True, False), (True, True), (False, False)])
def test_report_cli_writes_audit_before_reporting_incompleteness(
    tmp_path: Path, monkeypatch, available: bool, require_complete: bool,
) -> None:
    source = tmp_path / "source.txt"
    source.write_text("# requested order\n1aaa_A\n2bbb_B\n")
    features, labels = tmp_path / "features", tmp_path / "labels"
    if available:
        _touch_npz(chain_npz_path(features, "1aaa_A"))
        _touch_npz(chain_npz_path(labels, "1aaa_A"))
    _touch_npz(chain_npz_path(labels, "2bbb_B"))
    manifest, report_path = tmp_path / "audit/processed.txt", tmp_path / "audit/report.json"
    args = [
        "write_processed_manifest_report", "--source-manifest", str(source),
        "--processed-features-dir", str(features), "--processed-labels-dir", str(labels),
        "--output-manifest", str(manifest), "--report", str(report_path), "--max-examples", "0",
    ]
    if require_complete:
        args.append("--require-complete")
    monkeypatch.setattr(sys, "argv", args)
    if require_complete or not available:
        with pytest.raises(SystemExit, match="incomplete|No processed examples"):
            main()
    else:
        main()
    report = json.loads(report_path.read_text())
    assert report["processed_count"] == int(available)
    assert report["label_only_count"] == 1
    assert report["missing_examples"] == []
    assert manifest.read_text() == ("1aaa_A\n" if available else "")
    assert source.read_text() == "# requested order\n1aaa_A\n2bbb_B\n"
