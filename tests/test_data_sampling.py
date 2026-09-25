from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from nanofold import baseline_submission
from nanofold.data import _crop_single_example, collate_batch, read_manifest, sample_msa
from nanofold.model import NanoFoldBaseline
from submissions.template import submission as template_submission


def _example(chain_id: str, length: int, depth: int) -> dict[str, Any]:
    msa = torch.zeros((depth, length), dtype=torch.long)
    if depth > 1:
        msa[-1] = 21  # A biological row may be all gaps inside a crop.
    return {
        "chain_id": chain_id,
        "aatype": torch.zeros(length, dtype=torch.long),
        "msa": msa,
        "deletions": torch.ones_like(msa),
        "template_aatype": torch.zeros((0, length), dtype=torch.long),
        "template_ca_coords": torch.zeros((0, length, 3)),
        "template_ca_mask": torch.zeros((0, length), dtype=torch.bool),
    }


def test_manifest_rejects_duplicate_sampling_weight_but_preserves_chain_case(tmp_path: Path) -> None:
    path = tmp_path / "manifest.txt"
    path.write_text("1abc_A\n1abc_A\n2abc_A\n")
    with pytest.raises(ValueError, match="Duplicate chain ID"):
        read_manifest(path)
    path.write_text("# author chain IDs are case-sensitive\n1abc_A\n1abc_a\n")
    assert read_manifest(path) == ["1abc_A", "1abc_a"]


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_sampling_rejects_nonpositive_or_noninteger_sizes(size: Any) -> None:
    example = _example("a", 3, 2)
    with pytest.raises(ValueError, match="positive integer"):
        sample_msa(example["msa"], example["deletions"], size)
    with pytest.raises(ValueError, match="positive integer"):
        _crop_single_example(example, size, "center")


def test_sampling_validates_modes_even_without_cropping_or_subsampling() -> None:
    example = _example("a", 3, 2)
    with pytest.raises(ValueError, match="sample_mode"):
        sample_msa(example["msa"], example["deletions"], 10, "typo")
    with pytest.raises(ValueError, match="crop_mode"):
        _crop_single_example(example, 10, "typo")


@pytest.mark.parametrize("shape", [(0, 3), (2, 0), (3,)])
def test_sampling_rejects_empty_or_malformed_msa(shape: tuple[int, ...]) -> None:
    msa = torch.zeros(shape, dtype=torch.long)
    with pytest.raises(ValueError, match="matching nonempty"):
        sample_msa(msa, msa, 10)


def test_sampling_rejects_mismatched_deletions_before_early_return() -> None:
    with pytest.raises(ValueError, match="matching nonempty"):
        sample_msa(torch.zeros((2, 3)), torch.zeros((1, 3)), 10)


def test_collator_marks_real_gaps_valid_and_only_masks_padding() -> None:
    batch = collate_batch([_example("a", 3, 2), _example("b", 5, 4)], 5, 4)
    expected = torch.zeros((4, 5), dtype=torch.bool)
    expected[:2, :3] = True
    assert torch.equal(batch["msa_mask"][0], expected)
    assert bool((batch["msa"][0, 1, :3] == 21).all())
    assert bool(batch["msa_mask"][0, 1, :3].all())


@pytest.mark.parametrize("submission", [baseline_submission, template_submission])
def test_baseline_inference_is_invariant_to_batch_companion_depth_and_length(submission: Any) -> None:
    short = _example("a", 3, 2)
    alone = collate_batch([short], 5, 4)
    together = collate_batch([short, _example("b", 5, 4)], 5, 4)
    torch.manual_seed(7)
    model = NanoFoldBaseline(d_model=8, n_layers=1, n_heads=2, dropout=0).eval()
    with torch.no_grad():
        expected = submission.run_batch(model, alone, {}, training=False)["pred_atom14"][0]
        actual = submission.run_batch(model, together, {}, training=False)["pred_atom14"][0, :3]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
