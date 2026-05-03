from __future__ import annotations

import math

import numpy as np

from neurips_paper.scripts import randomize_nanofold_splits as randomize
from scripts.build_manifests import Candidate


def _candidate(chain_id: str, *, cls: str = "alpha", length: int = 80) -> Candidate:
    return Candidate(
        chain_id=chain_id,
        sequence="A" * length,
        length=length,
        resolution=2.0,
        release_date=None,
        cluster_id=f"cluster_{chain_id}",
        pdb_id=chain_id.split("_", 1)[0].lower(),
        secondary_structure_class=cls,
        secondary_helix_fraction=0.7 if cls == "alpha" else 0.2,
        secondary_beta_fraction=0.05 if cls == "alpha" else 0.35,
        secondary_coil_fraction=0.25,
        domain_architecture_class=cls,
        domain_architecture_source="test",
        metadata_source_count=2,
    )


def test_nearest_train_from_graph_uses_first_train_neighbor_and_self_zero() -> None:
    indices = np.asarray(
        [
            [0, 1, 2],
            [1, 2, 0],
            [2, 1, 0],
            [3, 2, 1],
        ],
        dtype=np.int32,
    )
    distances = np.asarray(
        [
            [0.0, 0.2, 0.4],
            [0.0, 0.3, 0.5],
            [0.0, 0.1, 0.6],
            [0.0, 0.7, 0.8],
        ],
        dtype=np.float32,
    )
    train_mask = np.asarray([False, True, False, False])

    nearest = randomize.nearest_train_from_graph(indices, distances, train_mask)

    np.testing.assert_allclose(nearest, np.asarray([0.2, 0.0, 0.1, 0.8], dtype=np.float32))


def test_local_k_distances_from_graph_ignores_self() -> None:
    indices = np.asarray([[0, 2, 1], [1, 0, 2]], dtype=np.int32)
    distances = np.asarray([[0.0, 0.4, 0.2], [0.0, 0.3, 0.7]], dtype=np.float32)

    local = randomize.local_k_distances_from_graph(indices, distances, k_values=(1, 2))

    np.testing.assert_allclose(local[1], np.asarray([0.2, 0.3], dtype=np.float32))
    np.testing.assert_allclose(local[2], np.asarray([0.4, 0.7], dtype=np.float32))


def test_summarize_actual_against_random_reports_percentile_and_tail() -> None:
    actual = {"split_name": "actual", "js_public_max": 0.2, "coverage50_public_val_fraction": 0.8}
    randomized = [
        {"split_name": "random_0", "js_public_max": 0.1, "coverage50_public_val_fraction": 0.7},
        {"split_name": "random_1", "js_public_max": 0.2, "coverage50_public_val_fraction": 0.8},
        {"split_name": "random_2", "js_public_max": 0.3, "coverage50_public_val_fraction": 0.9},
    ]

    rows = {row["metric"]: row for row in randomize.summarize_actual_against_random(actual, randomized)}

    assert rows["js_public_max"]["direction"] == "lower"
    assert rows["coverage50_public_val_fraction"]["direction"] == "higher"
    assert math.isclose(rows["js_public_max"]["empirical_percentile_le"], 2 / 3)
    assert math.isclose(rows["js_public_max"]["two_sided_empirical_tail"], 1.0)


def test_sample_public_split_is_deterministic_and_sizes_are_exact() -> None:
    units = [
        _candidate("1aaa_A", cls="alpha"),
        _candidate("2aaa_A", cls="alpha"),
        _candidate("3aaa_A", cls="beta"),
        _candidate("4aaa_A", cls="beta"),
        _candidate("5aaa_A", cls="alpha"),
        _candidate("6aaa_A", cls="beta"),
    ]

    first = randomize.sample_public_split(
        units,
        train_size=3,
        val_size=1,
        hidden_val_size=1,
        seed=11,
        replicate_index=2,
    )
    second = randomize.sample_public_split(
        units,
        train_size=3,
        val_size=1,
        hidden_val_size=1,
        seed=11,
        replicate_index=2,
    )

    assert first == second
    assert len(first.train_ids) == 3
    assert len(first.val_ids) == 1
    assert set(first.train_ids).isdisjoint(first.val_ids)
