from __future__ import annotations

import csv
from xml.etree import ElementTree as ET

from neurips_paper.scripts import plot_openproteinset_coverage_metrics as plots


def _write_csv(path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_create_metric_figures_writes_parseable_svg_and_preview_formats(tmp_path) -> None:
    table_dir = tmp_path / "tables"
    out_dir = tmp_path / "figures"
    table_dir.mkdir()
    _write_csv(
        table_dir / "embedding_nearest_train_quantiles.csv",
        [
            {"set": "all", "n": "110", "p50": "0.03", "p75": "0.05", "p90": "0.08", "p95": "0.10", "p99": "0.14"},
            {"set": "non_train", "n": "100", "p50": "0.03", "p75": "0.05", "p90": "0.08", "p95": "0.10", "p99": "0.14"},
            {"set": "public_val", "n": "10", "p50": "0.02", "p75": "0.04", "p90": "0.06", "p95": "0.07", "p99": "0.11"},
        ],
    )
    _write_csv(
        table_dir / "embedding_coverage_at_radii.csv",
        [
            {
                "radius_name": "full_10nn_median",
                "non_train_covered_fraction": "0.26",
                "public_val_covered_fraction": "0.25",
            },
            {
                "radius_name": "full_50nn_median",
                "non_train_covered_fraction": "0.44",
                "public_val_covered_fraction": "0.57",
            },
        ],
    )
    _write_csv(
        table_dir / "embedding_density_decile_coverage.csv",
        [
            {
                "density_decile": str(index),
                "median_local_d50": str(0.01 * index),
                "train_fraction": str(0.02 * index),
                "non_train_within_local_50_fraction": str(0.05 * index),
                "non_train_within_2x_local_50_fraction": str(min(0.08 * index, 1.0)),
            }
            for index in range(1, 11)
        ],
    )
    _write_csv(
        table_dir / "embedding_cluster_coverage_summary.csv",
        [
            {"k": "100", "clusters_touched_fraction": "0.87", "mass_in_touched_clusters_fraction": "0.90"},
            {"k": "200", "clusters_touched_fraction": "0.86", "mass_in_touched_clusters_fraction": "0.88"},
        ],
    )
    _write_csv(table_dir / "embedding_classifier_auc.csv", [{"auc": "0.73"}, {"auc": "0.75"}])

    figures = plots.create_figures(table_dir, out_dir, ["svg", "png"], "length-capped exact-unique sequences")

    assert set(figures) == {
        "embedding_space_coverage_diagnostics",
        "embedding_density_decile_diagnostics",
    }
    for paths in figures.values():
        for path in paths:
            assert path
    ET.parse(out_dir / "embedding_space_coverage_diagnostics.svg")
    ET.parse(out_dir / "embedding_density_decile_diagnostics.svg")
    assert "length-capped" in (out_dir / "embedding_space_coverage_diagnostics.svg").read_text()
    assert (out_dir / "embedding_space_coverage_diagnostics.png").stat().st_size > 0
    assert (out_dir / "embedding_density_decile_diagnostics.png").stat().st_size > 0
