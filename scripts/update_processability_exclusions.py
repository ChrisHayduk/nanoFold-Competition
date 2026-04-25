"""Update the official processability exclusion list from preprocess error markers."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofold.chain_paths import chain_id_from_stem

STATS_RE = re.compile(r": (\{.*\})$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read preprocess *.error.txt markers and pin chains that fail the official "
            "atom14 label processability gate."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--error-dir", type=Path, default=Path("data/processed_features"))
    parser.add_argument(
        "--chain-data-cache",
        type=Path,
        help="OpenFold chain_data_cache.json used to expand a failing chain to its PDB entry.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/official_processability_exclusions.txt"),
    )
    parser.add_argument("--min-projection-seq-identity", type=float, default=0.90)
    parser.add_argument("--min-projection-coverage", type=float, default=0.70)
    parser.add_argument("--min-projection-aligned-fraction", type=float, default=0.70)
    parser.add_argument("--min-projection-valid-ca", type=int, default=32)
    return parser.parse_args(argv)


def _chain_id_from_error_path(path: Path) -> str:
    name = path.name
    if not name.endswith(".error.txt"):
        raise ValueError(f"Expected *.error.txt path: {path}")
    return chain_id_from_stem(name[: -len(".error.txt")])


def _parse_stats(text: str) -> dict[str, Any] | None:
    match = STATS_RE.search(text.strip())
    if match is None:
        return None
    payload = ast.literal_eval(match.group(1))
    return payload if isinstance(payload, dict) else None


def _as_float(stats: dict[str, Any], key: str) -> float:
    value = stats.get(key, 0.0)
    return float(value) if isinstance(value, (str, bytes, bytearray, int, float)) else 0.0


def _fails_gate(
    stats: dict[str, Any],
    *,
    min_projection_seq_identity: float,
    min_projection_coverage: float,
    min_projection_aligned_fraction: float,
    min_projection_valid_ca: int,
) -> bool:
    return (
        _as_float(stats, "projection_seq_identity") < min_projection_seq_identity
        or _as_float(stats, "projection_alignment_coverage") < min_projection_coverage
        or _as_float(stats, "projection_aligned_fraction") < min_projection_aligned_fraction
        or int(_as_float(stats, "projection_valid_ca_count")) < min_projection_valid_ca
    )


def _read_existing(path: Path) -> set[str]:
    if not path.exists():
        return set()
    chain_ids: set[str] = set()
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            chain_ids.add(line)
    return chain_ids


def _pdb_id(chain_id: str) -> str:
    return chain_id.split("_", 1)[0].lower()


def _load_pdb_index(path: Path | None) -> dict[str, set[str]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in chain data cache: {path}")
    pdb_index: dict[str, set[str]] = {}
    for chain_id in payload:
        if not isinstance(chain_id, str):
            continue
        pdb_index.setdefault(_pdb_id(chain_id), set()).add(chain_id)
    return pdb_index


def update_exclusions(
    *,
    error_dir: Path,
    chain_data_cache: Path | None,
    output: Path,
    min_projection_seq_identity: float,
    min_projection_coverage: float,
    min_projection_aligned_fraction: float,
    min_projection_valid_ca: int,
) -> tuple[int, int]:
    existing = _read_existing(output)
    pdb_index = _load_pdb_index(chain_data_cache)
    discovered: set[str] = set()
    screened = 0
    for path in sorted(error_dir.glob("chain_*.error.txt")):
        stats = _parse_stats(path.read_text())
        if stats is None:
            continue
        screened += 1
        if _fails_gate(
            stats,
            min_projection_seq_identity=min_projection_seq_identity,
            min_projection_coverage=min_projection_coverage,
            min_projection_aligned_fraction=min_projection_aligned_fraction,
            min_projection_valid_ca=min_projection_valid_ca,
        ):
            chain_id = _chain_id_from_error_path(path)
            discovered.add(chain_id)
            discovered.update(pdb_index.get(_pdb_id(chain_id), set()))

    chain_ids = sorted(existing | discovered)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(
            [
                "# Chain IDs excluded from official splitting because atom14 labels fail the processability gate.",
                "# PDB-entry siblings are excluded with a failing chain to keep biological examples consistent.",
                f"# min_projection_seq_identity={min_projection_seq_identity}",
                f"# min_projection_coverage={min_projection_coverage}",
                f"# min_projection_aligned_fraction={min_projection_aligned_fraction}",
                f"# min_projection_valid_ca={min_projection_valid_ca}",
                *chain_ids,
                "",
            ]
        )
    )
    return screened, len(discovered - existing)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    screened, added = update_exclusions(
        error_dir=args.error_dir,
        chain_data_cache=args.chain_data_cache,
        output=args.output,
        min_projection_seq_identity=float(args.min_projection_seq_identity),
        min_projection_coverage=float(args.min_projection_coverage),
        min_projection_aligned_fraction=float(args.min_projection_aligned_fraction),
        min_projection_valid_ca=int(args.min_projection_valid_ca),
    )
    print(f"Screened {screened} preprocess errors; added {added} processability exclusions.")
    print(f"Processability exclusions: {args.output}")


if __name__ == "__main__":
    main()
