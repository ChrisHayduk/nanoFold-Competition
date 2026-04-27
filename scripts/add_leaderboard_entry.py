from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofold.leaderboard_identity import resolve_leaderboard_team
from nanofold.metrics import FOLDSCORE_CURVE_COMPONENT_NAMES


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Append a run result to leaderboard JSON and re-rank.")
    ap.add_argument("--result", type=str, required=True, help="Path to runs/<name>/result.json")
    ap.add_argument("--leaderboard", type=str, default="leaderboard/leaderboard.json")
    ap.add_argument("--description", type=str, default="", help="Optional description override.")
    ap.add_argument(
        "--team",
        type=str,
        default="",
        help="Optional team or individual submitter name override. If omitted, GitHub PR author metadata may be used.",
    )
    ap.add_argument("--no-render", action="store_true", help="Do not render README leaderboard section.")
    ap.add_argument("--readme", type=str, default="README.md")
    return ap.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def _result_to_entry(
    result: Dict[str, Any],
    description_override: str,
    team_override: str = "",
    env: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    created_at = str(result.get("created_at", ""))[:10]
    commit = str(result.get("commit", "unknown"))[:7]
    submission = str(result.get("submission_name", "submission"))
    team = resolve_leaderboard_team(
        explicit_team=team_override,
        result_team=str(result.get("team", "")),
        submission_name=submission,
        env=env,
    )
    description = description_override.strip() or str(
        result.get("description", f"Official run for {submission}")
    )
    def _to_float(value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return float("nan")

    hidden_score = result.get("final_hidden_foldscore", float("nan"))
    public_score = result.get("public_val_foldscore", float("nan"))
    rank_metric = str(result.get("rank_metric", "foldscore_auc_hidden"))
    default_rank_score = result.get("foldscore_auc_hidden", float("nan"))
    rank_score = result.get("rank_score", default_rank_score)
    rank_tiebreak_score = result.get("rank_tiebreak_score", hidden_score)
    entry = {
        "rank_metric": rank_metric,
        "rank_score": _to_float(rank_score),
        "rank_tiebreak_score": _to_float(rank_tiebreak_score),
        "final_hidden_foldscore": _to_float(hidden_score),
        "public_val_foldscore": _to_float(public_score),
        "foldscore_auc_hidden": _to_float(result.get("foldscore_auc_hidden", float("nan"))),
        "foldscore_at_samples": dict(result.get("foldscore_at_samples", {})) if isinstance(result.get("foldscore_at_samples"), dict) else {},
        "foldscore_at_steps": dict(result.get("foldscore_at_steps", {})) if isinstance(result.get("foldscore_at_steps"), dict) else {},
        "sample_budget": int(result.get("sample_budget", 0) or 0),
        "track": str(result.get("track", "limited")),
        "date": created_at,
        "commit": commit,
        "description": description,
        "team": team,
        "run_name": str(result.get("run_name", "")),
        "submission_name": submission,
    }
    for component_name in FOLDSCORE_CURVE_COMPONENT_NAMES:
        entry[f"final_hidden_{component_name}"] = _to_float(
            result.get(f"final_hidden_{component_name}", float("nan"))
        )
        entry[f"public_val_{component_name}"] = _to_float(
            result.get(f"public_val_{component_name}", float("nan"))
        )
        entry[f"{component_name}_auc_hidden"] = _to_float(
            result.get(f"{component_name}_auc_hidden", float("nan"))
        )
        samples = result.get(f"{component_name}_at_samples", {})
        steps = result.get(f"{component_name}_at_steps", {})
        entry[f"{component_name}_at_samples"] = dict(samples) if isinstance(samples, dict) else {}
        entry[f"{component_name}_at_steps"] = dict(steps) if isinstance(steps, dict) else {}
    return entry


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
    track_order = {"limited": 0, "research_large": 1, "unlimited": 2}

    def score_value(row: Dict[str, Any]) -> float:
        score = float(row.get("rank_score", float("nan")))
        if math.isnan(score):
            return float("-inf")
        return score

    def tiebreak_value(row: Dict[str, Any]) -> float:
        score = float(row.get("rank_tiebreak_score", float("nan")))
        if math.isnan(score):
            return float("-inf")
        return score

    ranked: List[Dict[str, Any]] = []
    tracks = sorted({str(row.get("track", "")) for row in entries}, key=lambda x: (track_order.get(x, 999), x))
    for track in tracks:
        track_rows = [row for row in entries if str(row.get("track", "")) == track]
        track_rows = sorted(
            track_rows,
            key=lambda x: (-score_value(x), -tiebreak_value(x), str(x.get("date", ""))),
        )
        for i, row in enumerate(track_rows, start=1):
            row["rank"] = i
        ranked.extend(track_rows)
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

    new_entry = _result_to_entry(result, args.description, args.team)
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
