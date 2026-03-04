from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List

import yaml

# Allow running as `python scripts/run_official.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofold.competition_policy import DEFAULT_TRACK_ID, load_track_spec


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Canonical official runner for submission evaluation.")
    ap.add_argument("--submission", type=str, required=True, help="Submission directory, e.g. submissions/alice")
    ap.add_argument("--config", type=str, default="", help="Optional config path (defaults to <submission>/config.yaml)")
    ap.add_argument("--track", type=str, default=DEFAULT_TRACK_ID)
    ap.add_argument("--python", type=str, default=sys.executable, help="Python executable to use.")
    ap.add_argument("--commit", type=str, default="", help="Optional commit hash override.")
    ap.add_argument("--description", type=str, default="", help="Optional leaderboard description.")
    ap.add_argument("--update-leaderboard", action="store_true", help="Append result into leaderboard JSON and render README.")
    ap.add_argument("--leaderboard", type=str, default="leaderboard/leaderboard.json")
    ap.add_argument("--readme", type=str, default="README.md")
    return ap.parse_args()


def _run(cmd: List[str]) -> None:
    printable = " ".join(cmd)
    print(f"+ {printable}")
    subprocess.run(cmd, check=True)


def _resolve_commit(explicit: str) -> str:
    if explicit.strip():
        return explicit.strip()
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _read_per_chain(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    submission_dir = Path(args.submission).resolve()
    if not submission_dir.exists():
        raise FileNotFoundError(f"Submission dir not found: {submission_dir}")

    config_path = Path(args.config).resolve() if args.config else (submission_dir / "config.yaml").resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = yaml.safe_load(config_path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    run_name = str(cfg.get("run_name", "")).strip()
    if not run_name:
        raise ValueError("Config must define a non-empty `run_name`.")

    track_spec = load_track_spec(args.track)
    commit = _resolve_commit(args.commit)
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_summary_path = run_dir / "eval_val.json"
    per_chain_path = run_dir / "per_chain_scores.jsonl"
    result_path = run_dir / "result.json"
    ckpt_path = run_dir / "checkpoints" / "ckpt_last.pt"

    _run(
        [
            args.python,
            "scripts/validate_submission.py",
            "--submission",
            str(submission_dir),
            "--track",
            track_spec.track_id,
            "--strict",
        ]
    )

    _run(
        [
            args.python,
            "train.py",
            "--config",
            str(config_path),
            "--track",
            track_spec.track_id,
            "--official",
        ]
    )

    _run(
        [
            args.python,
            "eval.py",
            "--config",
            str(config_path),
            "--ckpt",
            str(ckpt_path),
            "--split",
            "val",
            "--track",
            track_spec.track_id,
            "--official",
            "--save",
            str(eval_summary_path),
            "--per-chain-out",
            str(per_chain_path),
        ]
    )

    eval_summary = _read_json(eval_summary_path)
    per_chain_rows = _read_per_chain(per_chain_path)
    lddt_values = [float(row["lddt_ca"]) for row in per_chain_rows]
    per_chain_summary = {
        "count": len(lddt_values),
        "mean": float(mean(lddt_values)) if lddt_values else float("nan"),
        "std": float(pstdev(lddt_values)) if len(lddt_values) > 1 else 0.0,
        "min": float(min(lddt_values)) if lddt_values else float("nan"),
        "max": float(max(lddt_values)) if lddt_values else float("nan"),
    }

    fingerprint_path = Path(track_spec.fingerprint_path) if track_spec.fingerprint_path else None
    train_metrics_path = run_dir / "metrics.json"
    train_metrics = _read_json(train_metrics_path) if train_metrics_path.exists() else {}

    result = {
        "run_name": run_name,
        "submission_name": submission_dir.name,
        "submission_dir": str(submission_dir),
        "config_path": str(config_path),
        "track": track_spec.track_id,
        "score_lddt_ca": float(eval_summary["mean_lddt_ca"]),
        "mean_loss": float(eval_summary.get("mean_loss", float("nan"))),
        "num_chains": int(eval_summary.get("num_chains", len(per_chain_rows))),
        "per_chain_summary": per_chain_summary,
        "eval_summary_path": str(eval_summary_path.resolve()),
        "per_chain_scores_path": str(per_chain_path.resolve()),
        "train_metrics_path": str(train_metrics_path.resolve()),
        "train_wall_time_seconds": float(train_metrics.get("wall_time_seconds", float("nan")))
        if isinstance(train_metrics, dict)
        else float("nan"),
        "eval_wall_time_seconds": float(eval_summary.get("eval_wall_time_seconds", float("nan"))),
        "fingerprint_path": str(fingerprint_path.resolve()) if fingerprint_path else None,
        "fingerprint_sha256": _sha256(fingerprint_path) if fingerprint_path and fingerprint_path.exists() else None,
        "config_sha256": _sha256(config_path),
        "commit": commit,
        "description": args.description.strip() or f"Official run for {submission_dir.name}",
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "train_env": train_metrics.get("env", {}) if isinstance(train_metrics, dict) else {},
        "eval_env": eval_summary.get("env", {}) if isinstance(eval_summary, dict) else {},
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote result artifact: {result_path.resolve()}")

    if args.update_leaderboard:
        _run(
            [
                args.python,
                "scripts/add_leaderboard_entry.py",
                "--result",
                str(result_path),
                "--leaderboard",
                args.leaderboard,
                "--readme",
                args.readme,
                "--description",
                result["description"],
            ]
        )


if __name__ == "__main__":
    main()
