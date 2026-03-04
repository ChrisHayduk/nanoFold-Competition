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
            v = float(
                entry.get("rank_score", entry.get("score_hidden_lddt_ca", entry.get("score_lddt_ca", float("nan"))))
            )
        except Exception:
            return float("-inf")
        return v if not math.isnan(v) else float("-inf")

    lines = []
    lines.append("| # | Rank Score | Hidden Final | Hidden AUC | Public Val | Track | Date | Commit | Description |")
    lines.append("|---:|---:|---:|---:|---:|---|---|---|---|")
    sorted_entries = sorted(entries, key=lambda x: (-_rank_score(x), str(x.get("date", ""))))
    for i, e in enumerate(sorted_entries, start=1):
        commit = e.get("commit", "")[:7]
        hidden = e.get("score_hidden_lddt_ca", e.get("final_hidden_lddt_ca", float("nan")))
        public = e.get("score_public_val_lddt_ca", e.get("score_lddt_ca", float("nan")))
        lines.append(
            f"| {i} | {_fmt_score(e.get('rank_score', hidden))} | {_fmt_score(hidden)} | {_fmt_score(e.get('lddt_auc_hidden', float('nan')))} | {_fmt_score(public)} | {e.get('track','')} | {e.get('date','')} | `{commit}` | {e.get('description','')} |"
        )
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
