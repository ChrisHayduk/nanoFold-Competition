from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

PR_AUTHOR_ENV = "NANOFOLD_PR_AUTHOR"
PR_EVENT_NAMES = {"pull_request", "pull_request_target"}


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def github_pr_author_from_env(env: Mapping[str, str] | None = None) -> str:
    source = os.environ if env is None else env

    explicit_author = _clean_text(source.get(PR_AUTHOR_ENV, ""))
    if explicit_author:
        return explicit_author

    event_path = _clean_text(source.get("GITHUB_EVENT_PATH", ""))
    if event_path:
        try:
            payload: Any = json.loads(Path(event_path).read_text())
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            pull_request = payload.get("pull_request")
            if isinstance(pull_request, dict):
                user = pull_request.get("user")
                if isinstance(user, dict):
                    login = _clean_text(user.get("login", ""))
                    if login:
                        return login

    event_name = _clean_text(source.get("GITHUB_EVENT_NAME", ""))
    actor = _clean_text(source.get("GITHUB_ACTOR", ""))
    if event_name in PR_EVENT_NAMES and actor:
        return actor

    return ""


def resolve_leaderboard_team(
    *,
    explicit_team: str = "",
    result_team: str = "",
    submission_name: str = "",
    env: Mapping[str, str] | None = None,
) -> str:
    return (
        _clean_text(explicit_team)
        or _clean_text(result_team)
        or github_pr_author_from_env(env)
        or _clean_text(submission_name)
        or "submission"
    )
