from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from scripts import embed_crop_sequences as engine


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def parent_row(sequence: str) -> dict[str, str]:
    return {"sequence": sequence, "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest()}


def prepare_fixture(tmp_path: Path, sequences: list[str] | None = None) -> tuple[Path, dict[str, Any]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    parents = tmp_path / "parents.jsonl"
    write_rows(parents, [parent_row(sequence) for sequence in
                         (sequences if sequences is not None else ["A" * 256, "A" * 512, "ACDE"])])
    protocol = tmp_path / "protocol.json"
    protocol.write_text('{"purpose":"test"}\n')
    output = tmp_path / "prepared"
    receipt = engine.prepare(parents, sha256(parents), output, protocol, sha256(protocol))
    return output, receipt


@pytest.mark.parametrize("seed,start,stop,crop_sha", [
    (0, 59, 315, "48ad75b514139c0dcb0c5d54385c8c5bba0a43e9b12d7bd2d89a0c5ff12519a3"),
    (1, 77, 333, "50d231520051f3fc2d01066d9cfbcab908bf355047aab1d067aa9ad6d822946d"),
    (2, 139, 395, "48ad75b514139c0dcb0c5d54385c8c5bba0a43e9b12d7bd2d89a0c5ff12519a3"),
])
def test_crop_matches_frozen_contract_vectors(seed: int, start: int, stop: int, crop_sha: str) -> None:
    sequence = "ACDEFGHIKLMNPQRSTVWY" * 20
    crop = engine.deterministic_crop(sequence, seed)
    assert crop == {
        "parent_sequence_sha256": "e1f99b8004262ce170552c5a066d0fdc0d601fc5d8f410e3aefded3cf4507258",
        "crop_seed": seed, "start": start, "stop": stop,
        "sequence": sequence[start:stop], "crop_sequence_sha256": crop_sha,
    }


def test_short_parent_is_unchanged_for_every_seed() -> None:
    for seed in (0, 1, 2, 1000):
        crop = engine.deterministic_crop("ACDE", seed)
        assert crop["sequence"] == "ACDE"
        assert (crop["start"], crop["stop"], crop["crop_seed"]) == (0, 4, seed)
        assert crop["parent_sequence_sha256"] == crop["crop_sequence_sha256"] == (
            "f50b9a1db8767fb7cdc8b89cf1a78c9fac1e0e2d5bb5367aeec14709396d5c5e")


@pytest.mark.parametrize("sequence,seed", [("", 0), ("ACDE", -1), ("ACDE", 1.5), ("ACDE", True)])
def test_crop_rejects_invalid_requests(sequence: str, seed: Any) -> None:
    with pytest.raises(ValueError):
        engine.deterministic_crop(sequence, seed)


def test_preparation_deduplicates_crops_without_losing_parents_or_seeds(tmp_path: Path) -> None:
    output, receipt = prepare_fixture(tmp_path)
    manifest, inputs = engine.validate_prepared(output, sha256(output / "input-receipt.json"))
    assert manifest == receipt
    assert {row["sequence"] for row in inputs} == {"A" * 256, "ACDE"}
    assert len(inputs) == 2
    assert [row["id"] for row in inputs] == sorted({row["id"] for row in inputs})
    parents = {parent_row(sequence)["sequence_sha256"] for sequence in ["A" * 256, "A" * 512, "ACDE"]}
    homopolymer_crop = parent_row("A" * 256)["sequence_sha256"]
    for seed in (0, 1, 2):
        mapping = read_rows(output / f"mapping-seed-{seed}.jsonl")
        assert len(mapping) == len(parents)
        assert {row["parent_sequence_sha256"] for row in mapping} == parents
        assert all(row["crop_seed"] == seed for row in mapping)
        assert sum(row["crop_sequence_sha256"] == homopolymer_crop for row in mapping) == 2
        assert all(row["source_vector_record_id"] == row["crop_sequence_sha256"] for row in mapping)


def test_parent_input_permutation_does_not_change_crop_inputs_or_mappings(tmp_path: Path) -> None:
    sequences = ["ACDE", "ACDEFGHIKLMNPQRSTVWY" * 20, "A" * 512]
    first, _ = prepare_fixture(tmp_path / "first", sequences)
    second, _ = prepare_fixture(tmp_path / "second", list(reversed(sequences)))
    for name in ["inputs.jsonl", *(f"mapping-seed-{seed}.jsonl" for seed in (0, 1, 2))]:
        assert (first / name).read_bytes() == (second / name).read_bytes()


@pytest.mark.parametrize("corruption", ["duplicate_parent", "wrong_parent_hash", "wrong_input_pin", "wrong_protocol_pin"])
def test_prepare_rejects_unbound_or_duplicate_parents(tmp_path: Path, corruption: str) -> None:
    parents = tmp_path / "parents.jsonl"
    rows = [parent_row("ACDE"), parent_row("FGHI")]
    if corruption == "duplicate_parent":
        rows.append(rows[0])
    elif corruption == "wrong_parent_hash":
        rows[0]["sequence_sha256"] = "0" * 64
    write_rows(parents, rows)
    protocol = tmp_path / "protocol.json"
    protocol.write_text('{"purpose":"test"}')
    with pytest.raises(ValueError):
        engine.prepare(parents, "0" * 64 if corruption == "wrong_input_pin" else sha256(parents),
                       tmp_path / "prepared", protocol,
                       "0" * 64 if corruption == "wrong_protocol_pin" else sha256(protocol))


@pytest.mark.parametrize("name", ["inputs.jsonl", "mapping-seed-0.jsonl", "comparison-protocol.frozen.json", "input-receipt.json"])
def test_prepared_validation_rejects_changed_artifacts(tmp_path: Path, name: str) -> None:
    output, _ = prepare_fixture(tmp_path)
    receipt_sha = sha256(output / "input-receipt.json")
    artifact = output / name
    artifact.write_bytes(artifact.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        engine.validate_prepared(output, receipt_sha)


@pytest.mark.parametrize("corruption", ["input_sequence", "mapping_seed", "mapping_bounds", "mapping_parent"])
def test_prepared_validation_checks_meaning_even_when_file_hashes_are_updated(tmp_path: Path, corruption: str) -> None:
    output, manifest = prepare_fixture(tmp_path)
    if corruption == "input_sequence":
        artifact = output / "inputs.jsonl"
        rows = read_rows(artifact)
        rows[0]["sequence"] = "C" * len(rows[0]["sequence"])
        write_rows(artifact, rows)
        manifest["inputs_sha256"] = sha256(artifact)
    else:
        artifact = output / "mapping-seed-0.jsonl"
        rows = read_rows(artifact)
        if corruption == "mapping_seed":
            rows[0]["crop_seed"] = 1
        elif corruption == "mapping_bounds":
            rows[0]["stop"] += 1
        else:
            rows[0]["parent_sequence_sha256"] = rows[1]["parent_sequence_sha256"]
        write_rows(artifact, rows)
        manifest["mappings"]["0"]["sha256"] = sha256(artifact)
    receipt = output / "input-receipt.json"
    receipt.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        engine.validate_prepared(output, sha256(receipt))


def pooling_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    representations = torch.tensor([
        [[1000., 2000.], [1., 3.], [5., 7.], [-1000., -2000.], [100., 200.]],
        [[1000., 2000.], [9., 11.], [-1000., -2000.], [100., 200.], [300., 400.]],
    ], dtype=torch.float16)
    attention = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
    special = torch.tensor([[1, 0, 0, 1, 1], [1, 0, 1, 1, 1]])
    return representations, attention, special, torch.tensor([2, 1])


@pytest.mark.parametrize("mask_dtype", [torch.int64, torch.bool])
def test_pooling_excludes_special_tokens_and_padding_and_accumulates_float32(mask_dtype: torch.dtype) -> None:
    representations, attention, special, lengths = pooling_inputs()
    pooled = engine.masked_mean(representations, attention.to(mask_dtype), special.to(mask_dtype), lengths)
    assert pooled.dtype == torch.float32
    torch.testing.assert_close(pooled, torch.tensor([[3., 5.], [9., 11.]]), rtol=0, atol=0)


def test_pooling_converts_before_sum_to_avoid_half_precision_overflow() -> None:
    representations = torch.tensor([[[0.], [40000.], [40000.], [0.]]], dtype=torch.float16)
    pooled = engine.masked_mean(representations, torch.ones((1, 4), dtype=torch.int64),
                                torch.tensor([[1, 0, 0, 1]]), torch.tensor([2]))
    torch.testing.assert_close(pooled, torch.tensor([[40000.]]), rtol=0, atol=0)


@pytest.mark.parametrize("corruption", [
    "attention_nonbinary", "special_nonbinary", "attention_shape", "special_shape",
    "length_shape", "length_mismatch", "no_residues", "nonfinite_residue", "representation_shape",
])
def test_pooling_rejects_invalid_masks_or_residue_counts(corruption: str) -> None:
    representations, attention, special, lengths = pooling_inputs()
    if corruption == "attention_nonbinary":
        attention[0, 1] = 2
    elif corruption == "special_nonbinary":
        special[0, 1] = -1
    elif corruption == "attention_shape":
        attention = attention[:, :-1]
    elif corruption == "special_shape":
        special = special[:1]
    elif corruption == "length_shape":
        lengths = lengths[:1]
    elif corruption == "length_mismatch":
        lengths[0] = 3
    elif corruption == "no_residues":
        special[0] = 1
        lengths[0] = 0
    elif corruption == "nonfinite_residue":
        representations[0, 1, 0] = float("nan")
    else:
        representations = representations[0]
    with pytest.raises(ValueError):
        engine.masked_mean(representations, attention, special, lengths)


def chunk_inputs() -> tuple[list[dict[str, str]], np.ndarray]:
    part = [{"id": parent_row(sequence)["sequence_sha256"], "sequence": sequence} for sequence in ("ACDE", "FGHI")]
    embeddings = np.tile(np.array([[0.5], [-1.25]], dtype=np.float16), (1, 2560))
    return part, embeddings


def test_chunk_resume_is_bitwise_and_bound_to_contract_and_input_order(tmp_path: Path) -> None:
    part, embeddings = chunk_inputs()
    contract = "1" * 64
    assert engine.load_chunk(tmp_path, 0, part, contract) is None
    receipt = engine.save_chunk(tmp_path, 0, part, embeddings, contract, 0.125)
    loaded = engine.load_chunk(tmp_path, 0, part, contract)
    assert loaded is not None
    actual_receipt, actual = loaded
    assert actual_receipt == receipt
    assert actual.dtype == np.float16
    assert actual.tobytes() == embeddings.tobytes()
    with pytest.raises(ValueError):
        engine.load_chunk(tmp_path, 0, part, "2" * 64)
    with pytest.raises(ValueError):
        engine.load_chunk(tmp_path, 0, list(reversed(part)), contract)


@pytest.mark.parametrize("corruption", ["bytes", "unreceipted"])
def test_chunk_resume_rejects_corruption_and_unreceipted_vectors(tmp_path: Path, corruption: str) -> None:
    part, embeddings = chunk_inputs()
    engine.save_chunk(tmp_path, 0, part, embeddings, "1" * 64, 0.125)
    vector_file, = (tmp_path / "chunks").glob("*.npz")
    if corruption == "bytes":
        vector_file.write_bytes(vector_file.read_bytes() + b"corrupt")
    else:
        receipt_file, = (tmp_path / "chunks").glob("*.json")
        receipt_file.unlink()
    with pytest.raises(ValueError):
        engine.load_chunk(tmp_path, 0, part, "1" * 64)


@pytest.mark.parametrize("corruption", ["ids", "dtype", "shape", "nonfinite"])
def test_chunk_resume_validates_npz_contents_beyond_the_file_hash(tmp_path: Path, corruption: str) -> None:
    part, embeddings = chunk_inputs()
    receipt = engine.save_chunk(tmp_path, 0, part, embeddings, "1" * 64, 0.125)
    vector_file, = (tmp_path / "chunks").glob("*.npz")
    receipt_file, = (tmp_path / "chunks").glob("*.json")
    ids = [row["id"] for row in part]
    if corruption == "ids":
        ids.reverse()
    elif corruption == "dtype":
        embeddings = embeddings.astype(np.float32)
    elif corruption == "shape":
        embeddings = embeddings[:, :-1]
    else:
        embeddings[0, 0] = np.inf
    np.savez(vector_file, ids=np.asarray(ids), embeddings=embeddings)
    receipt["vectors_sha256"] = sha256(vector_file)
    receipt_file.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        engine.load_chunk(tmp_path, 0, part, "1" * 64)


@pytest.mark.parametrize("corruption", ["dtype", "width", "rows", "nonfinite"])
def test_chunk_save_rejects_invalid_embeddings(tmp_path: Path, corruption: str) -> None:
    part, embeddings = chunk_inputs()
    if corruption == "dtype":
        embeddings = embeddings.astype(np.float32)
    elif corruption == "width":
        embeddings = embeddings[:, :-1]
    elif corruption == "rows":
        embeddings = embeddings[:1]
    else:
        embeddings[0, 0] = np.nan
    with pytest.raises(ValueError):
        engine.save_chunk(tmp_path, 0, part, embeddings, "1" * 64, 0.125)


def asset_fixture(tmp_path: Path, *, omit_source: bool = False) -> tuple[Path, dict[str, str]]:
    directory = tmp_path / "assets"
    directory.mkdir()
    (directory / "README.md").write_text("Pinned embedding assets.\n")
    (directory / "source.py").write_text("VALUE = 1\n")
    names = ["README.md"] if omit_source else ["README.md", "source.py"]
    receipt = tmp_path / "asset-receipt.json"
    receipt.write_text(json.dumps({"code_revision": "x", "files": {
        name: sha256(directory / name) for name in names}}))
    return directory, {"path": str(receipt), "sha256": sha256(receipt)}


def test_asset_verification_rejects_source_omitted_from_pinned_inventory(tmp_path: Path) -> None:
    directory, receipt = asset_fixture(tmp_path, omit_source=True)
    with pytest.raises(ValueError, match="(?i)inventory|unlisted"):
        engine.verify_assets(directory, receipt, revision_field="code_revision", revision="x", files_field="files")


def test_complete_asset_inventory_accepts_exact_bytes_and_rejects_mutation(tmp_path: Path) -> None:
    directory, receipt = asset_fixture(tmp_path)
    assert engine.verify_assets(directory, receipt, "code_revision", "x", "files") == receipt["sha256"]
    (directory / "source.py").write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="Hash mismatch"):
        engine.verify_assets(directory, receipt, "code_revision", "x", "files")
