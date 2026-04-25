from __future__ import annotations

import json
from pathlib import Path

from nanofold.chain_paths import chain_error_path
from scripts.update_processability_exclusions import update_exclusions


def _write_projection_error(error_dir: Path, chain_id: str) -> None:
    chain_error_path(error_dir, chain_id).write_text(
        f"Projection coverage below threshold for {chain_id}: "
        "{'projection_seq_identity': 1.0, "
        "'projection_alignment_coverage': 0.5, "
        "'projection_aligned_fraction': 0.5, "
        "'projection_valid_ca_count': 64.0}\n"
    )


def test_update_exclusions_expands_to_pdb_entry(tmp_path: Path) -> None:
    error_dir = tmp_path / "errors"
    error_dir.mkdir()
    cache = tmp_path / "chain_data_cache.json"
    output = tmp_path / "official_processability_exclusions.txt"
    cache.write_text(
        json.dumps(
            {
                "1abc_A": {"seq": "A"},
                "1abc_B": {"seq": "A"},
                "2def_A": {"seq": "A"},
            }
        )
    )
    _write_projection_error(error_dir, "1abc_A")

    screened, added = update_exclusions(
        error_dir=error_dir,
        chain_data_cache=cache,
        output=output,
        min_projection_seq_identity=0.9,
        min_projection_coverage=0.7,
        min_projection_aligned_fraction=0.7,
        min_projection_valid_ca=32,
    )

    lines = {line for line in output.read_text().splitlines() if line and not line.startswith("#")}
    assert screened == 1
    assert added == 2
    assert lines == {"1abc_A", "1abc_B"}
