"""Prepare deterministic protein crops and embed them with pinned local ESMC assets.

This maintainer diagnostic never downloads assets or changes training data. Run
``prepare --help`` for input preparation and ``run --help`` for the JSON runtime
configuration. Generated files belong in an ignored analysis directory.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch

SEEDS = [0, 1, 2]
DIMENSIONS = 2560
MODEL_ID = "biohub/ESMC-6B"
MODEL_REVISION = "af1602ba7406f521b11bf8f81d52af378cde09e4"
SOURCE_REVISION = "43b4548b86762edfa747b07d5f440aad3c33acee"
POOLING = (
    "EsmcModel.last_hidden_state final layer80 after final LayerNorm; float32 mean over "
    "attention-mask AND NOT special-token-mask; equals crop residue count"
)
AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def read_json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def checked(path: Path, expected: str) -> Path:
    require(digest(path) == expected, f"Hash mismatch: {path.name}")
    return path


def atomic_bytes(path: Path, data: bytes) -> None:
    """Publish without overwriting existing or interrupted artifacts."""
    if path.exists():
        require(path.read_bytes() == data, f"Existing artifact differs: {path.name}")
        return
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    temporary.unlink()


def deterministic_crop(sequence: str, seed: int = 0) -> dict[str, Any]:
    """Choose an unbiased, exact-parent-keyed contiguous crop of at most 256 residues."""
    require(isinstance(sequence, str) and bool(sequence) and set(sequence) <= AMINO_ACIDS,
            "Crop requires a nonempty canonical amino acid sequence")
    require(type(seed) is int and seed >= 0, "Crop seed must be a nonnegative integer")
    parent = hashlib.sha256(sequence.encode("ascii")).hexdigest()
    starts = max(1, len(sequence) - 256 + 1)
    limit = 2 ** 256 - (2 ** 256 % starts)
    counter = 0
    while True:
        payload = f"nanofold-esmc-crop-v1\0{seed}\0{parent}\0{counter}".encode("ascii")
        value = int.from_bytes(hashlib.sha256(payload).digest(), "big")
        if value < limit:
            break
        counter += 1
    start = value % starts
    crop = sequence[start:start + 256]
    return {"parent_sequence_sha256": parent, "crop_seed": seed, "start": start,
            "stop": start + len(crop), "sequence": crop,
            "crop_sequence_sha256": hashlib.sha256(crop.encode("ascii")).hexdigest()}


def prepare(parents: Path, parents_sha256: str, output: Path,
            protocol: Path, protocol_sha256: str) -> dict[str, Any]:
    """Freeze unique inference inputs while retaining every parent in every seed."""
    checked(parents, parents_sha256)
    checked(protocol, protocol_sha256)
    read_json(protocol)
    source = sorted(read_rows(parents), key=lambda row: row["sequence_sha256"])
    require(source and len({row["sequence_sha256"] for row in source}) == len(source),
            "Parents must be nonempty distinct exact sequences")
    inputs: dict[str, str] = {}
    mappings: dict[str, list[dict[str, Any]]] = {}
    for seed in SEEDS:
        mapping = []
        for row in source:
            crop = deterministic_crop(row["sequence"], seed)
            require(crop["parent_sequence_sha256"] == row["sequence_sha256"], "Parent sequence hash mismatch")
            key, sequence = crop["crop_sequence_sha256"], crop.pop("sequence")
            require(key not in inputs or inputs[key] == sequence, "Crop hash collision")
            inputs[key] = sequence
            mapping.append({**crop, "source_vector_record_id": key, "parent_length": len(row["sequence"])})
        mappings[str(seed)] = mapping
    output.mkdir(parents=True, exist_ok=False)
    atomic_bytes(output / "comparison-protocol.frozen.json", protocol.read_bytes())
    mapping_receipts = {}
    for seed_key, rows in mappings.items():
        path = output / f"mapping-seed-{seed_key}.jsonl"
        atomic_bytes(path, b"".join(canonical(row) for row in rows))
        mapping_receipts[seed_key] = {"path": path.name, "sha256": digest(path), "parent_records": len(rows),
                                      "distinct_crop_inputs": len({row["crop_sequence_sha256"] for row in rows})}
    path = output / "inputs.jsonl"
    atomic_bytes(path, b"".join(canonical({"id": key, "sequence": inputs[key]}) for key in sorted(inputs)))
    receipt = {"schema": "nanofold-esmc-three-seed-inputs-v1", "status": "frozen_before_inference",
               "protocol_sha256": protocol_sha256, "parent_input_sha256": parents_sha256,
               "parent_records": len(source), "seeds": SEEDS, "primary_seed": 0, "mappings": mapping_receipts,
               "inputs_sha256": digest(path), "unique_inputs": len(inputs),
               "unique_residues": sum(map(len, inputs.values())),
               "unique_token_count_with_special": sum(len(sequence) + 2 for sequence in inputs.values()),
               "total_parent_seed_rows": len(source) * len(SEEDS),
               "residues_per_seed": {seed: sum(row["stop"] - row["start"] for row in rows)
                                     for seed, rows in mappings.items()},
               "prepare_script_sha256": digest(Path(__file__)),
               "deduplication": "Exact crop SHA256 with sequence collision checks; mappings retain equal-parent rows.",
               "scope": "Diagnostic sequence reference; source qualification and split ownership are separate."}
    checked(parents, parents_sha256)
    checked(protocol, protocol_sha256)
    atomic_bytes(output / "input-receipt.json", canonical(receipt))
    return receipt


def validate_prepared(root: Path, receipt_sha256: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(checked(root / "input-receipt.json", receipt_sha256))
    require(manifest["seeds"] == SEEDS and manifest["primary_seed"] == 0, "Unexpected crop seeds")
    checked(root / "comparison-protocol.frozen.json", manifest["protocol_sha256"])
    rows = read_rows(checked(root / "inputs.jsonl", manifest["inputs_sha256"]))
    ids = [row["id"] for row in rows]
    require(ids == sorted(set(ids)) and len(ids) == manifest["unique_inputs"] and bool(ids),
            "Input identities must be sorted, unique and complete")
    inputs = {}
    for row in rows:
        sequence = row["sequence"]
        require(set(row) == {"id", "sequence"} and 1 <= len(sequence) <= 256 and set(sequence) <= AMINO_ACIDS,
                "Invalid crop input sequence")
        require(hashlib.sha256(sequence.encode("ascii")).hexdigest() == row["id"], "Crop input hash mismatch")
        inputs[row["id"]] = sequence
    require(sum(map(len, inputs.values())) == manifest["unique_residues"], "Input residue count mismatch")
    parent_ids = None
    used: set[str] = set()
    for seed in SEEDS:
        item = manifest["mappings"][str(seed)]
        require(item["path"] == f"mapping-seed-{seed}.jsonl", "Unexpected mapping path")
        mapping = read_rows(checked(root / item["path"], item["sha256"]))
        parents = [row["parent_sequence_sha256"] for row in mapping]
        require(parents == sorted(set(parents)) and len(parents) == manifest["parent_records"] == item["parent_records"],
                "Mapping parent identities must be sorted, unique and complete")
        require(parent_ids is None or parents == parent_ids, "Parent identities differ between seeds")
        parent_ids = parents
        crop_ids = set()
        residues = 0
        for row in mapping:
            key = row["crop_sequence_sha256"]
            require(row["crop_seed"] == seed and row["source_vector_record_id"] == key and key in inputs,
                    "Mapping crop identity mismatch")
            length, start, stop = row["parent_length"], row["start"], row["stop"]
            require(all(type(x) is int for x in (length, start, stop)) and length > 0
                    and 0 <= start < stop <= length and stop - start == min(256, length) == len(inputs[key]),
                    "Invalid mapping crop bounds")
            if length <= 256:
                require(row["parent_sequence_sha256"] == key and start == 0, "Short parent identity mismatch")
            crop_ids.add(key)
            residues += len(inputs[key])
        require(len(crop_ids) == item["distinct_crop_inputs"]
                and residues == manifest["residues_per_seed"][str(seed)], "Mapping counts mismatch")
        used.update(crop_ids)
    require(used == set(inputs), "Inference inputs differ from mapping union")
    require(manifest["total_parent_seed_rows"] == manifest["parent_records"] * len(SEEDS), "Parent seed count mismatch")
    return manifest, rows


def masked_mean(representations: torch.Tensor, attention_mask: torch.Tensor,
                special_mask: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Pool residue states in float32, rejecting malformed token masks."""
    import torch

    require(representations.ndim == 3 and representations.shape[2] > 0, "Invalid representation shape")
    require(attention_mask.shape == special_mask.shape == representations.shape[:2], "Token mask shape mismatch")
    require(lengths.shape == (representations.shape[0],) and bool((lengths > 0).all())
            and bool((lengths == lengths.long()).all()), "Invalid residue lengths")
    for mask in (attention_mask, special_mask):
        require(bool(((mask == 0) | (mask == 1)).all()), "Token masks must be binary")
    valid = attention_mask.bool() & ~special_mask.bool()
    require(torch.equal(valid.sum(1), lengths) and torch.equal(attention_mask.sum(1), lengths + 2),
            "Token mask residue or special-token count mismatch")
    means = (representations.float() * valid.unsqueeze(-1)).sum(1) / lengths.unsqueeze(-1)
    require(bool(torch.isfinite(means).all()) and bool((means.norm(dim=1) > 0).all()), "Invalid pooled embedding")
    return means


def validate_vectors(ids: list[str], vectors: np.ndarray, part: list[dict[str, Any]]) -> None:
    require(ids == [row["id"] for row in part], "Chunk input identities differ")
    require(vectors.shape == (len(part), DIMENSIONS) and vectors.dtype == np.float16,
            "Chunk embedding shape or dtype differs")
    require(np.isfinite(vectors).all() and (np.linalg.norm(vectors.astype(np.float32), axis=1) > 0).all(),
            "Chunk embeddings must be finite and nonzero")


def chunk_path(output: Path, start: int, part: list[dict[str, Any]]) -> Path:
    return output / "chunks" / f"{start:06d}-{start + len(part):06d}.npz"


def load_chunk(output: Path, start: int, part: list[dict[str, Any]],
               contract_sha256: str) -> tuple[dict[str, Any], np.ndarray] | None:
    path = chunk_path(output, start, part)
    receipt_path = path.with_suffix(".json")
    if not receipt_path.exists():
        require(not path.exists() and not path.with_name(path.name + ".writing").exists(),
                "Unreceipted chunk retained for review")
        return None
    receipt = read_json(receipt_path)
    require(receipt["contract_sha256"] == contract_sha256, "Chunk contract differs")
    require(receipt["start"] == start and receipt["stop"] == start + len(part)
            and receipt["records"] == len(part) and receipt["path"] == path.name
            and receipt["residues"] == sum(len(row["sequence"]) for row in part), "Chunk receipt bounds differ")
    checked(path, receipt["vectors_sha256"])
    with np.load(path, allow_pickle=False) as data:
        vectors = data["embeddings"]
        validate_vectors(data["ids"].tolist(), vectors, part)
    return receipt, vectors


def save_vectors(path: Path, ids: list[str], vectors: np.ndarray) -> None:
    require(not path.exists(), f"Existing vector artifact: {path.name}")
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("xb") as handle:
        np.savez(handle, ids=np.asarray(ids), embeddings=vectors)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    temporary.unlink()


def save_chunk(output: Path, start: int, part: list[dict[str, Any]], embeddings: np.ndarray,
               contract_sha256: str, seconds: float) -> dict[str, Any]:
    ids = [row["id"] for row in part]
    validate_vectors(ids, embeddings, part)
    path = chunk_path(output, start, part)
    path.parent.mkdir(exist_ok=True)
    save_vectors(path, ids, embeddings)
    receipt = {"start": start, "stop": start + len(part), "records": len(part),
               "residues": sum(len(row["sequence"]) for row in part), "contract_sha256": contract_sha256,
               "vectors_sha256": digest(path), "path": path.name, "seconds": seconds}
    atomic_bytes(path.with_suffix(".json"), canonical(receipt))
    return receipt


def verify_assets(directory: Path, receipt_pin: dict[str, str], revision_field: str,
                  revision: str, files_field: str) -> str:
    path = checked(Path(receipt_pin["path"]), receipt_pin["sha256"])
    receipt = read_json(path)
    require(receipt[revision_field] == revision and bool(receipt[files_field]), "Asset revision or file list differs")
    require(directory.is_dir(), "Asset snapshot is not a directory")
    members = list(directory.rglob("*"))
    require(not any(member.is_symlink() for member in members), "Asset snapshot must contain regular files, not symlinks")
    inventory = {member.relative_to(directory).as_posix() for member in members if member.is_file()}
    require(inventory == set(receipt[files_field]), "Asset inventory differs: unlisted or missing files")
    for name, expected in receipt[files_field].items():
        asset = (directory / name).resolve()
        require(not Path(name).is_absolute() and asset.is_relative_to(directory.resolve()), "Asset path escapes root")
        checked(asset, expected)
    return digest(path)


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Embed prepared inputs with one visible CUDA GPU; resume only this exact contract."""
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(config, output)


def run_locked(config: dict[str, Any], output: Path) -> dict[str, Any]:
    started = time.monotonic()
    root = Path(config["inputs_dir"])
    manifest, rows = validate_prepared(root, config["input_receipt_sha256"])
    source, model_path = Path(config["source_dir"]), Path(config["model_dir"])
    source_sha = verify_assets(source, config["source_receipt"], "code_revision", SOURCE_REVISION, "files")
    model_sha = verify_assets(model_path, config["model_assets_receipt"], "revision", MODEL_REVISION, "files_sha256")
    batch_size, chunk_size = config.get("batch_size", 32), config.get("chunk_size", 1024)
    max_seconds = config.get("max_seconds", 3480)
    require(all(type(value) is int and value > 0 for value in (batch_size, chunk_size, max_seconds)),
            "Batch size, chunk size and time bound must be positive integers")
    import typing

    import torch
    import typing_extensions

    if not hasattr(typing, "Self"):
        setattr(typing, "Self", typing_extensions.Self)  # noqa: B010 - Python 3.10 compatibility for pinned ESM.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source.resolve()))
    esmc = importlib.import_module("esm.models.esmc")
    if esmc.__file__ is None or not Path(esmc.__file__).resolve().is_relative_to(source.resolve()):
        raise ValueError("ESMC imported outside pinned source")
    require(torch.cuda.device_count() == 1, "Exactly one visible CUDA GPU is required")
    properties = torch.cuda.get_device_properties(0)
    torch.manual_seed(0)
    environment = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                   "numpy": np.__version__, "gpu": properties.name, "gpu_total_bytes": properties.total_memory,
                   "optional_kernels": {name: getattr(esmc.kernels, name) for name in
                                        ("TE_INSTALLED", "XFORMERS_INSTALLED", "FLASH_ATTN_INSTALLED", "FLASH_ATTN_ROTARY_INSTALLED")},
                   "packages": {name: importlib.metadata.version(name) for name in
                                ("transformers", "huggingface-hub", "safetensors", "einops", "accelerate")},
                   "typing_self_alias": platform.python_version_tuple()[:2] == ("3", "10")}
    for key, value in config.get("expected_environment", {}).items():
        require(environment[key] == value, f"Runtime environment differs: {key}")
    contract = {"schema": "nanofold-esmc-uniform-cuda-v1", "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION, "esm_code_revision": SOURCE_REVISION,
                "model_assets_sha256": model_sha, "source_receipt_sha256": source_sha,
                "pooling": POOLING, "dimensions": DIMENSIONS, "inference_dtype": "bfloat16", "storage_dtype": "float16",
                "attention_implementation": "sdpa", "batch_size": batch_size, "chunk_size": chunk_size,
                "ordering": "inputs sorted by crop SHA; each chunk sorted descending residue count then crop SHA for batches; output restored input SHA order",
                "seeds": SEEDS, "primary_seed": 0, "inputs_sha256": manifest["inputs_sha256"],
                "input_receipt_sha256": config["input_receipt_sha256"], "protocol_sha256": manifest["protocol_sha256"],
                "runner_sha256": digest(Path(__file__)), "environment": environment,
                "reuse": "Only completed chunks from this exact contract are reused."}
    contract_bytes = canonical(contract)
    contract_sha = hashlib.sha256(contract_bytes).hexdigest()
    atomic_bytes(output / "contract.json", contract_bytes)
    attempt = uuid.uuid4().hex
    (output / "executions").mkdir(exist_ok=True)
    atomic_bytes(output / "executions" / f"{attempt}.json", canonical(
        {"slurm_job_id": os.environ.get("SLURM_JOB_ID"), "host": platform.node(),
         "contract_sha256": contract_sha, "max_seconds": max_seconds}))
    if (output / "embeddings.npz").exists() or (output / "provenance.json").exists():
        raise ValueError("Final artifacts already exist; retain them for verification, not another run")
    tokenizer = esmc.EsmcTokenizer.from_pretrained(str(model_path), local_files_only=True)
    loading = time.monotonic()
    model = esmc.EsmcModel.from_pretrained(str(model_path), device="cuda", dtype=torch.bfloat16,
                                         attn_implementation="sdpa", local_files_only=True).eval()
    require(model.config.hidden_size == DIMENSIONS and model.config.num_hidden_layers == 80, "Unexpected model size")
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - loading

    def forward(batch: list[dict[str, Any]]) -> np.ndarray:
        encoded = tokenizer([row["sequence"] for row in batch], padding=True, truncation=False,
                            return_tensors="pt", return_special_tokens_mask=True)
        special = encoded.pop("special_tokens_mask").to("cuda")
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        lengths = torch.tensor([len(row["sequence"]) for row in batch], device="cuda")
        representations = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
                                output_hidden_states=False, return_dict=True).last_hidden_state
        require(representations.shape[-1] == DIMENSIONS, "Unexpected embedding dimension")
        return masked_mean(representations, encoded["attention_mask"], special, lengths).cpu().numpy().astype(np.float16)

    chunks = []
    inference_start = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(rows), chunk_size):
            part = rows[start:start + chunk_size]
            existing = load_chunk(output, start, part, contract_sha)
            if existing is not None:
                chunks.append(existing[0])
                continue
            if time.monotonic() - started >= max_seconds:
                atomic_bytes(output / "executions" / f"{attempt}.incomplete.json", canonical(
                    {"status": "bounded_time_exit", "completed_rows": start, "contract_sha256": contract_sha}))
                raise TimeoutError("Time bound reached; completed chunks are retained for same-contract resume")
            tick = time.monotonic()
            ordered = sorted(part, key=lambda row: (-len(row["sequence"]), row["id"]))
            values: dict[str, np.ndarray] = {}
            for offset in range(0, len(ordered), batch_size):
                batch = ordered[offset:offset + batch_size]
                values.update((row["id"], value) for row, value in zip(batch, forward(batch), strict=True))
            matrix = np.stack([values[row["id"]] for row in part])
            receipt = save_chunk(output, start, part, matrix, contract_sha, time.monotonic() - tick)
            chunks.append(receipt)
            print(json.dumps({"event": "chunk_complete", "completed": start + len(part), "total": len(rows),
                              "chunk_seconds": receipt["seconds"]}), flush=True)
    matrix = np.empty((len(rows), DIMENSIONS), dtype=np.float16)
    for start in range(0, len(rows), chunk_size):
        part = rows[start:start + chunk_size]
        chunk = load_chunk(output, start, part, contract_sha)
        require(chunk is not None, "Chunk missing during final assembly")
        if chunk is not None:
            matrix[start:start + len(part)] = chunk[1]
    validate_prepared(root, config["input_receipt_sha256"])
    path = output / "embeddings.npz"
    save_vectors(path, [row["id"] for row in rows], matrix)
    provenance = {key: contract[key] for key in
                  ("model_id", "model_revision", "esm_code_revision", "pooling", "dimensions", "inference_dtype",
                   "storage_dtype", "inputs_sha256", "input_receipt_sha256", "protocol_sha256")}
    provenance.update({"schema": "nanofold-esmc-three-seed-inference-v1", "status": "complete", "backend": "cuda",
                       "seeds": SEEDS, "records": len(rows), "parent_records_per_seed": manifest["parent_records"],
                       "embeddings_sha256": digest(path), "contract_sha256": contract_sha, "mappings": manifest["mappings"],
                       "model_load_seconds": load_seconds, "inference_and_assembly_seconds": time.monotonic() - inference_start,
                       "total_seconds": time.monotonic() - started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "chunks": len(chunks),
                       "scope": "Uniform CUDA diagnostic embeddings; qualification and split ownership require separate joins."})
    atomic_bytes(output / "provenance.json", canonical(provenance))
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Freeze exact-parent crops for seeds 0, 1 and 2")
    prep.add_argument("--parents", type=Path, required=True, help="JSONL with sequence and sequence_sha256 per unique parent")
    prep.add_argument("--parents-sha256", required=True)
    prep.add_argument("--protocol", type=Path, required=True, help="Frozen analysis protocol JSON")
    prep.add_argument("--protocol-sha256", required=True)
    prep.add_argument("--output", type=Path, required=True, help="New ignored output directory")
    execute = commands.add_parser("run", help="Embed using supplied local assets; no downloads", description=(
        "Config JSON requires inputs_dir, input_receipt_sha256, output_dir, source_dir, model_dir, "
        "source_receipt and model_assets_receipt (each with path and sha256). Optional batch_size=32, "
        "chunk_size=1024, max_seconds=3480 and expected_environment (key/value constraints). Asset receipts use "
        "code_revision/files and revision/files_sha256 respectively, covering complete regular-file snapshots "
        "without symlinks or bytecode caches; keep receipts outside those snapshots. Exactly one CUDA GPU must be visible. "
        "The time bound is checked between chunks; use a scheduler hard limit for allocation control. "
        "Completed chunks resume only when the entire numerical contract is unchanged."))
    execute.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.parents, args.parents_sha256, args.output, args.protocol, args.protocol_sha256)
    else:
        result = run(read_json(args.config))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
