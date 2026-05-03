from __future__ import annotations

import csv
from xml.etree import ElementTree as ET

from neurips_paper.scripts import plot_openproteinset_coverage_umap as umap_plot


def test_create_umap_plot_writes_parseable_svg_and_png(tmp_path) -> None:
    projection_path = tmp_path / "projection.csv"
    out_dir = tmp_path / "figures"
    rows = [
        {"record_id": f"background_{index}", "representative_chain_id": f"{index}_A", "split": "background", "length": "100", "chain_count": "1", "x": str(index % 5), "y": str(index // 5)}
        for index in range(30)
    ]
    rows.extend(
        [
            {"record_id": "train_0", "representative_chain_id": "1_A", "split": "train", "length": "120", "chain_count": "1", "x": "1.2", "y": "1.3"},
            {"record_id": "val_0", "representative_chain_id": "2_A", "split": "public_val", "length": "130", "chain_count": "1", "x": "2.2", "y": "2.3"},
        ]
    )
    with projection_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    written = umap_plot.create_plot(
        projection_path,
        out_dir,
        ["svg", "png"],
        "coverage",
        "length-capped exact-unique standard-AA sequences",
        "t-SNE",
    )

    assert len(written) == 2
    ET.parse(out_dir / "coverage.svg")
    assert "length-capped" in (out_dir / "coverage.svg").read_text()
    assert "t-SNE" in (out_dir / "coverage.svg").read_text()
    assert (out_dir / "coverage.png").stat().st_size > 0


def test_create_umap_plot_writes_contour_density_svg(tmp_path) -> None:
    projection_path = tmp_path / "projection.csv"
    out_dir = tmp_path / "figures"
    rows = [
        {
            "record_id": f"background_{index}",
            "representative_chain_id": f"{index}_A",
            "split": "background",
            "length": "100",
            "chain_count": "1",
            "x": str((index % 12) / 4),
            "y": str((index // 12) / 4),
        }
        for index in range(144)
    ]
    rows.extend(
        [
            {
                "record_id": "train_0",
                "representative_chain_id": "1_A",
                "split": "train",
                "length": "120",
                "chain_count": "1",
                "x": "1.2",
                "y": "1.3",
            },
            {
                "record_id": "val_0",
                "representative_chain_id": "2_A",
                "split": "public_val",
                "length": "130",
                "chain_count": "1",
                "x": "2.2",
                "y": "2.3",
            },
        ]
    )
    with projection_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    written = umap_plot.create_plot(
        projection_path,
        out_dir,
        ["svg"],
        "coverage_contour",
        density_style="contour",
        contour_bins=48,
        contour_sigma=1.1,
    )

    assert written == [str(out_dir / "coverage_contour.svg")]
    ET.parse(out_dir / "coverage_contour.svg")
    assert "Background contours" in (out_dir / "coverage_contour.svg").read_text()
