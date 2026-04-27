from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

"""Render leaderboard/leaderboard.json into README.md.

We keep README as the main entry point for casual visitors, but store structured data in JSON.
"""


START = "<!-- LEADERBOARD_START -->"
END = "<!-- LEADERBOARD_END -->"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaderboard", type=str, default="leaderboard/leaderboard.json")
    ap.add_argument("--readme", type=str, default="README.md")
    return ap.parse_args()


def render_table(entries: List[Dict[str, Any]]) -> str:
    def _fmt_score(value: Any) -> str:
        try:
            f = float(value)
        except Exception:
            return "n/a"
        if math.isnan(f):
            return "n/a"
        return f"{f:.4f}"

    def _rank_score(entry: Dict[str, Any]) -> float:
        try:
            v = float(entry.get("rank_score", float("nan")))
        except Exception:
            return float("-inf")
        return v if not math.isnan(v) else float("-inf")

    def _track_sort_key(track: str) -> tuple[int, str]:
        order = {"limited": 0, "research_large": 1, "unlimited": 2}
        return (order.get(track, 999), track)

    lines: List[str] = []
    for track in sorted({str(e.get("track", "")) for e in entries}, key=_track_sort_key):
        if lines:
            lines.append("")
        track_entries = [entry for entry in entries if str(entry.get("track", "")) == track]
        sorted_entries = sorted(track_entries, key=lambda x: (-_rank_score(x), str(x.get("date", ""))))
        title = track or "unknown"
        lines.append(f"### `{title}`")
        lines.append("| # | Rank Score | Hidden FoldScore | Public FoldScore | Date | Commit | Description |")
        lines.append("|---:|---:|---:|---:|---|---|---|")
        for i, e in enumerate(sorted_entries, start=1):
            commit = e.get("commit", "")[:7]
            hidden = e.get("final_hidden_foldscore", float("nan"))
            public = e.get("public_val_foldscore", float("nan"))
            lines.append(
                f"| {i} | {_fmt_score(e.get('rank_score', float('nan')))} | {_fmt_score(hidden)} | {_fmt_score(public)} | {e.get('date','')} | `{commit}` | {e.get('description','')} |"
            )
    if not lines:
        lines.append("| Track | Status |")
        lines.append("|---|---|")
        lines.append("| `limited` | No accepted submissions yet |")
        lines.append("| `research_large` | No accepted submissions yet |")
        lines.append("| `unlimited` | No accepted submissions yet |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    entries = json.loads(Path(args.leaderboard).read_text())

    readme_path = Path(args.readme)
    text = readme_path.read_text()

    if START not in text or END not in text:
        raise ValueError("README missing leaderboard markers")

    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)

    table = render_table(entries)
    new_text = before + START + "\n" + table + "\n" + END + after
    readme_path.write_text(new_text)
    print(f"Updated {readme_path}")


if __name__ == "__main__":
    main()
