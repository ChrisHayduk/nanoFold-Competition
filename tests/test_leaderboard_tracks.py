from __future__ import annotations

import importlib.util
import json
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


def test_leaderboard_entry_carries_team_metadata() -> None:
    module = _load_script("scripts/add_leaderboard_entry.py")
    result_to_entry = getattr(module, "_result_to_entry")

    entry = result_to_entry(
        {
            "created_at": "2026-01-01T00:00:00+00:00",
            "commit": "abcdef012345",
            "submission_name": "submission_dir",
            "team": "Protein Geometry Lab",
            "rank_score": 0.5,
            "rank_tiebreak_score": 0.4,
            "final_hidden_foldscore": 0.4,
            "public_val_foldscore": 0.3,
        },
        "",
    )

    assert entry["team"] == "Protein Geometry Lab"
    assert entry["name"] == "submission_dir"
    assert entry["submission_path"] == "submissions/submission_dir"

    overridden = result_to_entry(
        {"submission_name": "submission_dir", "name": "Readable Name"},
        "",
        "Independent Researcher",
    )
    assert overridden["team"] == "Independent Researcher"
    assert overridden["name"] == "Readable Name"


def test_leaderboard_entry_uses_repo_relative_submission_path() -> None:
    module = _load_script("scripts/add_leaderboard_entry.py")
    result_to_entry = getattr(module, "_result_to_entry")

    submission_dir = Path("submissions/submission_dir").resolve()
    entry = result_to_entry(
        {
            "submission_name": "submission_dir",
            "submission_dir": str(submission_dir),
        },
        "",
        "",
        {},
    )

    assert entry["submission_path"] == "submissions/submission_dir"


def test_leaderboard_entry_can_fall_back_to_pr_author(tmp_path) -> None:
    module = _load_script("scripts/add_leaderboard_entry.py")
    result_to_entry = getattr(module, "_result_to_entry")
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps({"pull_request": {"user": {"login": "github-user"}}}))

    entry = result_to_entry(
        {"submission_name": "submission_dir"},
        "",
        "",
        {"GITHUB_EVENT_PATH": str(event_path)},
    )

    assert entry["team"] == "github-user"


def test_readme_leaderboard_renderer_groups_by_track() -> None:
    module = _load_script("scripts/render_leaderboard.py")
    render_table = getattr(module, "render_table")

    rendered = render_table(
        [
            {
                "track": "unlimited",
                "name": "open_submission",
                "submission_path": "submissions/open_submission",
                "rank_score": 0.8,
                "final_hidden_foldscore": 0.8,
                "public_val_foldscore": 0.6,
                "date": "2026-01-02",
                "commit": "abcdef0",
                "description": "open run",
                "team": "Open Team",
            },
            {
                "track": "limited",
                "name": "small_submission",
                "submission_path": "submissions/small_submission",
                "rank_score": 0.5,
                "final_hidden_foldscore": 0.7,
                "public_val_foldscore": 0.4,
                "date": "2026-01-01",
                "commit": "1234567",
                "description": "small run",
                "team": "Small Lab",
            },
        ]
    )

    assert "### `limited`" in rendered
    assert "### `unlimited`" in rendered
    assert rendered.index("### `limited`") < rendered.index("### `unlimited`")
    assert "| # | Name | Team | Rank Score | Hidden FoldScore | Public FoldScore | Date | Commit | Description |" in rendered
    assert "| 1 | [small_submission](submissions/small_submission) | Small Lab |" in rendered
