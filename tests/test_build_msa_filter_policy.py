from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from msa_filter_policy_helpers import write_valid_policy_case

from nanofold.msa_filter_policy import load_msa_filter_policy

REPO_ROOT = Path(__file__).resolve().parents[1]


def _command(paths: dict[str, Path], output: Path) -> list[str]:
    return [sys.executable, str(REPO_ROOT / "scripts/build_msa_filter_policy.py"),
            "--train-manifest", str(paths["train_manifest"]),
            "--val-manifest", str(paths["val_manifest"]),
            "--hidden-manifest", str(paths["hidden_manifest"]),
            "--train-msa-row-filter", str(paths["train_filter"]),
            "--val-msa-row-filter", str(paths["val_filter"]), "--output", str(output)]


def test_policy_cli_validates_and_deterministically_writes_directed_evidence(tmp_path: Path) -> None:
    paths = write_valid_policy_case(tmp_path / "source")
    output = tmp_path / "candidate" / "policy.json"
    subprocess.run(_command(paths, output), check=True, capture_output=True, text=True)
    policy = load_msa_filter_policy(output, expected_chain_ids=["trn1_A", "val1_A"])
    assert policy.assignments == {"trn1_A": "train", "val1_A": "val"}
    assert policy.filters["train"].path == paths["train_filter"].resolve()
    assert policy.filters["val"].path == paths["val_filter"].resolve()
    first = output.read_bytes()
    subprocess.run(_command(paths, output), check=True, capture_output=True, text=True)
    assert output.read_bytes() == first
    assert list(output.parent.iterdir()) == [output]


def test_policy_cli_rejects_swapped_filters_without_creating_outputs(tmp_path: Path) -> None:
    paths = write_valid_policy_case(tmp_path / "source")
    paths["train_filter"], paths["val_filter"] = paths["val_filter"], paths["train_filter"]
    output = tmp_path / "candidate" / "policy.json"
    result = subprocess.run(_command(paths, output), capture_output=True, text=True)
    assert result.returncode != 0
    assert not output.parent.exists()


@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_policy_cli_preserves_transitive_raw_source_aliases(tmp_path: Path, alias: str) -> None:
    paths = write_valid_policy_case(tmp_path / "source")
    source = paths["train_msa"]
    output = source
    if alias != "direct":
        output = tmp_path / "output.json"
        if alias == "symlink":
            output.symlink_to(source)
        else:
            output.hardlink_to(source)
    before = source.read_bytes()
    result = subprocess.run(_command(paths, output), capture_output=True, text=True)
    assert result.returncode != 0
    assert "immutable source dependency" in result.stderr
    assert source.read_bytes() == before
    assert output.read_bytes() == before
