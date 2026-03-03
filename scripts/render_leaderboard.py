from __future__ import annotations

import argparse
import json
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
    lines = []
    lines.append("| # | Score (lDDT-Cα) | Track | Date | Commit | Description |")
    lines.append("|---:|---:|---|---|---|---|")
    for i, e in enumerate(sorted(entries, key=lambda x: (-x["score_lddt_ca"], x.get("date", ""))), start=1):
        commit = e.get("commit", "")[:7]
        lines.append(
            f"| {i} | {e['score_lddt_ca']:.4f} | {e.get('track','')} | {e.get('date','')} | `{commit}` | {e.get('description','')} |"
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
