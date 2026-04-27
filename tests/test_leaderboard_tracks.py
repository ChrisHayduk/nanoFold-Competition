from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script(path: str):
    module_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_leaderboard_ranks_entries_within_each_track() -> None:
    module = _load_script("scripts/add_leaderboard_entry.py")
    rank_entries = getattr(module, "_rank_entries")

    ranked = rank_entries(
        [
            {"track": "limited", "rank_score": 0.3, "rank_tiebreak_score": 0.0, "date": "2026-01-01"},
            {"track": "research_large", "rank_score": 0.9, "rank_tiebreak_score": 0.0, "date": "2026-01-01"},
            {"track": "limited", "rank_score": 0.4, "rank_tiebreak_score": 0.0, "date": "2026-01-01"},
        ]
    )

    limited_rows = [row for row in ranked if row["track"] == "limited"]
    research_rows = [row for row in ranked if row["track"] == "research_large"]
    assert [row["rank"] for row in limited_rows] == [1, 2]
    assert [row["rank"] for row in research_rows] == [1]
    assert [row["rank_score"] for row in limited_rows] == [0.4, 0.3]


def test_readme_leaderboard_renderer_groups_by_track() -> None:
    module = _load_script("scripts/render_leaderboard.py")
    render_table = getattr(module, "render_table")

    rendered = render_table(
        [
            {
                "track": "unlimited",
                "rank_score": 0.8,
                "final_hidden_foldscore": 0.8,
                "public_val_foldscore": 0.6,
                "date": "2026-01-02",
                "commit": "abcdef0",
                "description": "open run",
            },
            {
                "track": "limited",
                "rank_score": 0.5,
                "final_hidden_foldscore": 0.7,
                "public_val_foldscore": 0.4,
                "date": "2026-01-01",
                "commit": "1234567",
                "description": "small run",
            },
        ]
    )

    assert "### `limited`" in rendered
    assert "### `unlimited`" in rendered
    assert rendered.index("### `limited`") < rendered.index("### `unlimited`")
    assert "| # | Rank Score | Hidden FoldScore | Public FoldScore | Date | Commit | Description |" in rendered
