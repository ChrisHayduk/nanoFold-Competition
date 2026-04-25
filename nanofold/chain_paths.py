from __future__ import annotations

from pathlib import Path

CHAIN_FILE_PREFIX = "chain_"


def chain_id_to_stem(chain_id: str) -> str:
    normalized = str(chain_id)
    if not normalized:
        raise ValueError("Chain ID must be non-empty.")
    return f"{CHAIN_FILE_PREFIX}{normalized.encode('utf-8').hex()}"


def chain_id_from_stem(stem: str) -> str:
    if not stem.startswith(CHAIN_FILE_PREFIX):
        raise ValueError(f"Not a nanoFold chain file stem: {stem}")
    payload = stem[len(CHAIN_FILE_PREFIX) :]
    try:
        return bytes.fromhex(payload).decode("utf-8")
    except ValueError as exc:
        raise ValueError(f"Invalid nanoFold chain file stem: {stem}") from exc


def chain_npz_path(base_dir: str | Path, chain_id: str) -> Path:
    return Path(base_dir) / f"{chain_id_to_stem(chain_id)}.npz"


def chain_error_path(base_dir: str | Path, chain_id: str) -> Path:
    return Path(base_dir) / f"{chain_id_to_stem(chain_id)}.error.txt"


def chain_data_dir(base_dir: str | Path, chain_id: str) -> Path:
    return Path(base_dir) / chain_id_to_stem(chain_id)

