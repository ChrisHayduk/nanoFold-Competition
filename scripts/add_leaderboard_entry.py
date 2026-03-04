from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Append a run result to leaderboard JSON and re-rank.")
    ap.add_argument("--result", type=str, required=True, help="Path to runs/<name>/result.json")
    ap.add_argument("--leaderboard", type=str, default="leaderboard/leaderboard.json")
    ap.add_argument("--description", type=str, default="", help="Optional description override.")
    ap.add_argument("--no-render", action="store_true", help="Do not render README leaderboard section.")
    ap.add_argument("--readme", type=str, default="README.md")
    return ap.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def _result_to_entry(result: Dict[str, Any], description_override: str) -> Dict[str, Any]:
    created_at = str(result.get("created_at", ""))[:10]
    commit = str(result.get("commit", "unknown"))[:7]
    submission = str(result.get("submission_name", "submission"))
    description = description_override.strip() or str(
        result.get("description", f"Official run for {submission}")
    )
    return {
        "score_lddt_ca": float(result["score_lddt_ca"]),
        "track": str(result.get("track", "limited")),
        "date": created_at,
        "commit": commit,
        "description": description,
        "run_name": str(result.get("run_name", "")),
    }


def _dedupe_entries(entries: List[Dict[str, Any]], new_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in entries:
        if (
            str(row.get("run_name", "")) == new_entry.get("run_name", "")
            and str(row.get("commit", "")) == new_entry.get("commit", "")
            and str(row.get("track", "")) == new_entry.get("track", "")
        ):
            continue
        out.append(row)
    out.append(new_entry)
    return out


def _rank_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = sorted(
        entries,
        key=lambda x: (-float(x.get("score_lddt_ca", float("-inf"))), str(x.get("date", ""))),
    )
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
    return ranked


def main() -> None:
    args = parse_args()
    result_path = Path(args.result)
    if not result_path.exists():
        raise FileNotFoundError(f"Result file not found: {result_path}")
    result = _load_json(result_path)
    if not isinstance(result, dict):
        raise ValueError(f"Result file must contain a JSON object: {result_path}")

    leaderboard_path = Path(args.leaderboard)
    if leaderboard_path.exists():
        entries = _load_json(leaderboard_path)
        if not isinstance(entries, list):
            raise ValueError(f"Leaderboard must be a JSON list: {leaderboard_path}")
    else:
        entries = []

    new_entry = _result_to_entry(result, args.description)
    merged = _dedupe_entries(entries, new_entry)
    ranked = _rank_entries(merged)
    _save_json(leaderboard_path, ranked)
    print(f"Updated leaderboard: {leaderboard_path.resolve()}")

    if not args.no_render:
        import subprocess
        import sys

        subprocess.run(
            [
                sys.executable,
                "scripts/render_leaderboard.py",
                "--leaderboard",
                str(leaderboard_path),
                "--readme",
                args.readme,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
