from __future__ import annotations

import json

from nanofold.leaderboard_identity import github_pr_author_from_env, resolve_leaderboard_team


def test_leaderboard_team_prefers_explicit_then_result_then_submission() -> None:
    assert (
        resolve_leaderboard_team(
            explicit_team="Protein Geometry Lab",
            result_team="PR Author",
            submission_name="submission_dir",
            env={},
        )
        == "Protein Geometry Lab"
    )
    assert (
        resolve_leaderboard_team(
            result_team="PR Author",
            submission_name="submission_dir",
            env={},
        )
        == "PR Author"
    )
    assert resolve_leaderboard_team(submission_name="submission_dir", env={}) == "submission_dir"


def test_github_pr_author_comes_from_pull_request_event(tmp_path) -> None:
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps({"pull_request": {"user": {"login": "alice"}}}))

    assert github_pr_author_from_env({"GITHUB_EVENT_PATH": str(event_path)}) == "alice"
    assert (
        resolve_leaderboard_team(
            submission_name="submission_dir",
            env={"GITHUB_EVENT_PATH": str(event_path)},
        )
        == "alice"
    )


def test_github_pr_author_can_be_supplied_by_maintainer_env() -> None:
    assert github_pr_author_from_env({"NANOFOLD_PR_AUTHOR": "bob"}) == "bob"


def test_github_actor_is_only_used_for_pr_events() -> None:
    assert (
        github_pr_author_from_env(
            {
                "GITHUB_EVENT_NAME": "pull_request",
                "GITHUB_ACTOR": "carol",
            }
        )
        == "carol"
    )
    assert (
        resolve_leaderboard_team(
            submission_name="submission_dir",
            env={
                "GITHUB_EVENT_NAME": "workflow_dispatch",
                "GITHUB_ACTOR": "maintainer",
            },
        )
        == "submission_dir"
    )
