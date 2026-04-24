from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

# Allow running as `python scripts/run_official.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofold.competition_policy import DEFAULT_TRACK_ID, load_track_spec
from nanofold.data import read_manifest
from nanofold.metrics import foldscore_auc, foldscore_components


DEFAULT_CHECKPOINT_STEPS = "0,1000,2000,5000,last"
SEALED_RUNTIME_ENV = "NANOFOLD_OFFICIAL_SEALED_RUNTIME"
RUNTIME_STAGE_ENV = "NANOFOLD_OFFICIAL_RUNTIME_STAGE"
HIDDEN_ENV_PREFIX = "NANOFOLD_HIDDEN_"


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
    ap.add_argument(
        "--checkpoint-steps",
        type=str,
        default=DEFAULT_CHECKPOINT_STEPS,
        help=f"Comma-separated checkpoint steps for hidden efficiency metrics (default: {DEFAULT_CHECKPOINT_STEPS}).",
    )
    ap.add_argument(
        "--hidden-lock-file",
        type=str,
        default="",
        help="Maintainer lock file containing expected hidden asset hashes.",
    )
    ap.add_argument(
        "--disable-hidden",
        action="store_true",
        help="Skip hidden evaluation/scoring and produce public-val-only artifact.",
    )
    ap.add_argument("--hidden-manifest", type=str, default="", help="Hidden manifest path override.")
    ap.add_argument("--hidden-features-dir", type=str, default="", help="Hidden processed features dir override.")
    ap.add_argument("--hidden-labels-dir", type=str, default="", help="Hidden processed labels dir override.")
    ap.add_argument("--hidden-fingerprint", type=str, default="", help="Hidden fingerprint JSON path override.")
    ap.add_argument("--skip-train", action="store_true", help="Skip train.py stage and reuse existing checkpoints.")
    ap.add_argument(
        "--skip-hidden-scoring",
        action="store_true",
        help="Run hidden prediction only; do not score or finalize the result artifact.",
    )
    ap.add_argument(
        "--score-hidden-only",
        action="store_true",
        help="Reuse existing public/hidden prediction artifacts and only run hidden scoring/finalization.",
    )
    return ap.parse_args()


def _run(cmd: List[str], *, env: Dict[str, str] | None = None) -> None:
    printable = " ".join(cmd)
    print(f"+ {printable}")
    subprocess.run(cmd, check=True, env=env)


def _sealed_runtime_enabled(env: Dict[str, str] | None = None) -> bool:
    source = os.environ if env is None else env
    return str(source.get(SEALED_RUNTIME_ENV, "")).strip() == "1"


def _require_sealed_hidden_runtime(*, disable_hidden: bool, env: Dict[str, str] | None = None) -> None:
    if disable_hidden:
        return
    if _sealed_runtime_enabled(env):
        return
    raise ValueError(
        "Official hidden leaderboard runs require a sealed runtime. "
        "Use scripts/run_official_docker.sh or run inside a sealed container with "
        f"{SEALED_RUNTIME_ENV}=1."
    )


def _scrub_hidden_env(*, stage: str, env: Dict[str, str] | None = None) -> Dict[str, str]:
    source = dict(os.environ if env is None else env)
    for key in list(source.keys()):
        if key.startswith(HIDDEN_ENV_PREFIX):
            source.pop(key, None)
    source[RUNTIME_STAGE_ENV] = stage
    return source


def _build_predict_command(
    *,
    python: str,
    config_path: Path,
    split: str,
    track_id: str,
    official: bool,
    pred_out_dir: Path,
    save_path: Path,
    ckpt: Path | None = None,
    ckpt_dir: Path | None = None,
    ckpt_steps: str = "",
    fingerprint: Path | None = None,
    hidden_manifest: Path | None = None,
    forbid_labels_dir: Path | None = None,
) -> List[str]:
    cmd = [
        python,
        "predict.py",
        "--config",
        str(config_path),
        "--split",
        split,
        "--track",
        track_id,
        "--pred-out-dir",
        str(pred_out_dir),
        "--save",
        str(save_path),
    ]
    if official:
        cmd.append("--official")
    if fingerprint is not None:
        cmd.extend(["--fingerprint", str(fingerprint)])
    if hidden_manifest is not None:
        cmd.extend(["--hidden-manifest", str(hidden_manifest)])
    if forbid_labels_dir is not None:
        cmd.extend(["--forbid-labels-dir", str(forbid_labels_dir)])
    if ckpt is not None:
        cmd.extend(["--ckpt", str(ckpt)])
    if ckpt_dir is not None:
        cmd.extend(["--ckpt-dir", str(ckpt_dir)])
    if ckpt_steps:
        cmd.extend(["--ckpt-steps", ckpt_steps])
    return cmd


def _build_score_command(
    *,
    python: str,
    prediction_summary: Path,
    labels_dir: Path,
    per_chain_out: Path,
    save_path: Path,
) -> List[str]:
    return [
        python,
        "score.py",
        "--prediction-summary",
        str(prediction_summary),
        "--labels-dir",
        str(labels_dir),
        "--per-chain-out",
        str(per_chain_out),
        "--save",
        str(save_path),
    ]


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


def _maybe_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return _sha256(path)


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


def _parse_checkpoint_steps(text: str) -> List[str]:
    tokens = [tok.strip().lower() for tok in text.split(",") if tok.strip()]
    out: List[str] = []
    for tok in tokens:
        if tok == "last":
            out.append(tok)
            continue
        if not tok.isdigit():
            raise ValueError(f"Invalid checkpoint token `{tok}` in --checkpoint-steps.")
        out.append(str(int(tok)))
    return out


def _resolve_hidden_asset(
    *,
    cli_value: str,
    track_value: str | None,
    env_key: str,
    label: str,
) -> Path:
    env_value = str(__import__("os").environ.get(env_key, "")).strip()
    value = cli_value.strip() or env_value or (track_value or "").strip()
    if not value:
        raise ValueError(f"Missing hidden asset for {label}. Set --{label} or environment variable {env_key}.")
    path = Path(value).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _require_pinned_hidden_track_metadata(track_spec: Any) -> None:
    if not getattr(track_spec, "hidden_manifest_sha256", None):
        raise ValueError(
            "Track metadata must pin `dataset.hidden_manifest_sha256` for hidden leaderboard runs."
        )
    if not getattr(track_spec, "hidden_fingerprint_sha256", None):
        raise ValueError(
            "Track metadata must pin `dataset.hidden_fingerprint_sha256` for hidden leaderboard runs."
        )


def _tree_sha256(root: Path) -> str:
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Expected directory for tree hash: {root}")
    hasher = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = str(path.relative_to(root)).replace("\\", "/")
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\t")
        hasher.update(_sha256(path).encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _validate_hidden_lock(
    *,
    lock_path: Path,
    hidden_manifest: Path,
    hidden_features_dir: Path,
    hidden_labels_dir: Path,
    hidden_fingerprint: Path,
    track_spec: Any,
) -> Dict[str, Any]:
    if not lock_path.exists():
        raise FileNotFoundError(f"Hidden lock file is required for hidden official runs: {lock_path}")

    raw = _read_json(lock_path)
    if not isinstance(raw, dict):
        raise ValueError(f"Hidden lock file must contain a JSON object: {lock_path}")

    expected = {
        "hidden_manifest_sha256": raw.get("hidden_manifest_sha256"),
        "hidden_features_fingerprint_sha256": raw.get("hidden_features_fingerprint_sha256"),
        "hidden_labels_fingerprint_sha256": raw.get("hidden_labels_fingerprint_sha256"),
        "hidden_fingerprint_sha256": raw.get("hidden_fingerprint_sha256"),
    }
    for key, value in expected.items():
        if not isinstance(value, str) or value.strip() == "":
            raise ValueError(f"Hidden lock file must define non-empty `{key}` for leaderboard runs.")

    actual = {
        "hidden_manifest_sha256": _sha256(hidden_manifest),
        "hidden_features_fingerprint_sha256": _tree_sha256(hidden_features_dir),
        "hidden_labels_fingerprint_sha256": _tree_sha256(hidden_labels_dir),
        "hidden_fingerprint_sha256": _sha256(hidden_fingerprint),
    }

    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            raise ValueError(
                f"Hidden lock mismatch for `{key}`: expected `{expected_value}`, got `{actual.get(key)}`."
            )
    if track_spec.hidden_manifest_sha256 and actual["hidden_manifest_sha256"] != track_spec.hidden_manifest_sha256:
        raise ValueError(
            "Track hidden manifest SHA256 mismatch: "
            f"expected `{track_spec.hidden_manifest_sha256}`, got `{actual['hidden_manifest_sha256']}`."
        )
    if track_spec.hidden_fingerprint_sha256 and actual["hidden_fingerprint_sha256"] != track_spec.hidden_fingerprint_sha256:
        raise ValueError(
            "Track hidden fingerprint SHA256 mismatch: "
            f"expected `{track_spec.hidden_fingerprint_sha256}`, got `{actual['hidden_fingerprint_sha256']}`."
        )

    return {
        "lock_file": str(lock_path.resolve()),
        "status": "validated",
        "expected": expected,
        "actual": actual,
    }


def _parse_step_from_ckpt_path(ckpt_path: str, *, max_steps: int) -> int | None:
    name = Path(ckpt_path).name
    m = re.search(r"ckpt_step_(\d+)\.pt$", name)
    if m:
        return int(m.group(1))
    if name == "ckpt_last.pt":
        return int(max_steps)
    return None


def _prediction_subdir(pred_root: Path, ckpt_path: str, n_ckpts: int) -> Path:
    if n_ckpts <= 1:
        return pred_root
    return pred_root / Path(ckpt_path).stem


def _center_crop_atom14_labels(
    atom14_positions: np.ndarray,
    atom14_mask: np.ndarray,
    crop_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    L = int(atom14_positions.shape[0])
    if L <= crop_size:
        return atom14_positions, atom14_mask
    start = (L - crop_size) // 2
    end = start + crop_size
    return atom14_positions[start:end], atom14_mask[start:end]


def _score_hidden_predictions(
    *,
    hidden_manifest: Path,
    hidden_labels_dir: Path,
    pred_root: Path,
    checkpoint_entries: List[Dict[str, Any]],
    crop_size: int,
    sample_budget: int,
    per_chain_out_path: Path,
) -> Dict[str, Any]:
    chain_ids = read_manifest(hidden_manifest)
    n_ckpts = len(checkpoint_entries)
    checkpoint_scores: List[Dict[str, Any]] = []
    per_chain_rows: List[Dict[str, Any]] = []

    for item in checkpoint_entries:
        ckpt_path = str(item["ckpt"])
        step = int(item.get("step", 0))
        cumulative_samples_seen = int(item.get("cumulative_samples_seen", -1))
        cumulative_cropped_residues_seen = int(item.get("cumulative_cropped_residues_seen", -1))
        cumulative_nonpad_residues_seen = int(item.get("cumulative_nonpad_residues_seen", -1))
        if cumulative_samples_seen < 0:
            raise ValueError(f"Checkpoint entry is missing `cumulative_samples_seen`: {item}")
        ckpt_pred_dir = _prediction_subdir(pred_root, ckpt_path, n_ckpts=n_ckpts)
        score_lists: Dict[str, List[float]] = {
            "foldscore": [],
            "lddt_ca": [],
            "lddt_backbone_atom14": [],
            "lddt_atom14": [],
        }

        for chain_id in chain_ids:
            pred_path = ckpt_pred_dir / f"{chain_id}.npz"
            label_path = hidden_labels_dir / f"{chain_id}.npz"
            if not pred_path.exists():
                raise FileNotFoundError(f"Missing prediction file for hidden scoring: {pred_path}")
            if not label_path.exists():
                raise FileNotFoundError(f"Missing hidden label file: {label_path}")

            with np.load(pred_path) as pred_npz:
                if "pred_atom14" not in pred_npz:
                    raise ValueError(f"Official hidden scoring requires pred_atom14 in {pred_path}")
                pred_atom14 = np.asarray(pred_npz["pred_atom14"], dtype=np.float32)
            with np.load(label_path) as label_npz:
                if "atom14_positions" not in label_npz or "atom14_mask" not in label_npz:
                    raise ValueError(f"Official hidden scoring requires atom14 labels in {label_path}")
                true_atom14 = np.asarray(label_npz["atom14_positions"], dtype=np.float32)
                atom14_mask = np.asarray(label_npz["atom14_mask"], dtype=bool)
            true_atom14, atom14_mask = _center_crop_atom14_labels(true_atom14, atom14_mask, crop_size=crop_size)

            L = min(pred_atom14.shape[0], true_atom14.shape[0], atom14_mask.shape[0])
            if L <= 0:
                metrics = {
                    "foldscore": float("nan"),
                    "lddt_ca": float("nan"),
                    "lddt_backbone_atom14": float("nan"),
                    "lddt_atom14": float("nan"),
                }
            else:
                comps = foldscore_components(
                    pred_atom14=torch.from_numpy(pred_atom14[:L]),
                    true_atom14=torch.from_numpy(true_atom14[:L]),
                    atom14_mask=torch.from_numpy(atom14_mask[:L]),
                )
                metrics = {name: float(value.detach().cpu()) for name, value in comps.items()}
            for name, value in metrics.items():
                if not np.isnan(value):
                    score_lists[name].append(float(value))
            per_chain_rows.append(
                {
                    "split": "hidden_val",
                    "ckpt": ckpt_path,
                    "step": step,
                    "cumulative_samples_seen": cumulative_samples_seen,
                    "chain_id": chain_id,
                    **metrics,
                }
            )

        mean_scores = {
            f"mean_{name}": float(sum(values) / len(values)) if values else float("nan")
            for name, values in score_lists.items()
        }
        checkpoint_scores.append(
            {
                "ckpt": ckpt_path,
                "step": step,
                "cumulative_samples_seen": cumulative_samples_seen,
                "cumulative_cropped_residues_seen": cumulative_cropped_residues_seen,
                "cumulative_nonpad_residues_seen": cumulative_nonpad_residues_seen,
                **mean_scores,
                "num_chains": len(chain_ids),
            }
        )

    per_chain_out_path.parent.mkdir(parents=True, exist_ok=True)
    with per_chain_out_path.open("w") as f:
        for row in per_chain_rows:
            f.write(json.dumps(row) + "\n")

    sorted_points = sorted(checkpoint_scores, key=lambda x: int(x["cumulative_samples_seen"]))
    previous_samples: int | None = None
    for row in sorted_points:
        current_samples = int(row["cumulative_samples_seen"])
        if previous_samples is not None and current_samples <= previous_samples:
            raise ValueError(
                "Hidden scoring requires strictly increasing cumulative_samples_seen values; "
                f"got {current_samples} after {previous_samples}."
            )
        previous_samples = current_samples

    final_hidden = float("nan")
    if sorted_points:
        final_hidden = float(sorted_points[-1]["mean_foldscore"])

    if not sorted_points or int(sorted_points[0]["cumulative_samples_seen"]) != 0:
        raise ValueError("Hidden scoring requires a step-0 checkpoint with cumulative_samples_seen=0.")
    if int(sorted_points[-1]["cumulative_samples_seen"]) != int(sample_budget):
        raise ValueError(
            "Hidden scoring requires a final checkpoint at the full sample budget "
            f"({sample_budget}); got {sorted_points[-1]['cumulative_samples_seen']}."
        )

    auc_hidden = foldscore_auc(
        (
            (
                int(row["step"]),
                int(row["cumulative_samples_seen"]),
                float(row["mean_foldscore"]),
            )
            for row in sorted_points
        ),
        sample_budget=sample_budget,
    )
    lddt_auc_hidden = foldscore_auc(
        (
            (
                int(row["step"]),
                int(row["cumulative_samples_seen"]),
                float(row.get("mean_lddt_ca", row["mean_foldscore"])),
            )
            for row in sorted_points
        ),
        sample_budget=sample_budget,
    )

    foldscore_at_steps = {
        str(int(row["step"])): float(row["mean_foldscore"])
        for row in sorted_points
    }
    foldscore_at_samples = {
        str(int(row["cumulative_samples_seen"])): float(row["mean_foldscore"])
        for row in sorted_points
    }
    lddt_at_steps = {
        str(int(row["step"])): float(row.get("mean_lddt_ca", row["mean_foldscore"]))
        for row in sorted_points
    }
    lddt_at_samples = {
        str(int(row["cumulative_samples_seen"])): float(row.get("mean_lddt_ca", row["mean_foldscore"]))
        for row in sorted_points
    }

    return {
        "final_hidden_foldscore": final_hidden,
        "foldscore_auc_hidden": auc_hidden,
        "foldscore_at_steps": foldscore_at_steps,
        "foldscore_at_samples": foldscore_at_samples,
        "final_hidden_lddt_ca": float(sorted_points[-1].get("mean_lddt_ca", final_hidden)),
        "lddt_auc_hidden": lddt_auc_hidden,
        "lddt_at_steps": lddt_at_steps,
        "lddt_at_samples": lddt_at_samples,
        "checkpoint_metrics_hidden": checkpoint_scores,
        "per_chain_scores_hidden_path": str(per_chain_out_path.resolve()),
    }


def _write_temp_predict_config(
    *,
    cfg: Dict[str, Any],
    hidden_features_dir: Path,
    config_path: Path,
    out_path: Path,
) -> Path:
    hidden_cfg = copy.deepcopy(cfg)
    data_cfg = hidden_cfg.get("data")
    if not isinstance(data_cfg, dict):
        raise ValueError("Config missing `data` section.")
    data_cfg["processed_features_dir"] = str(hidden_features_dir)
    data_cfg["processed_labels_dir"] = ""
    submission_cfg = hidden_cfg.get("submission")
    if isinstance(submission_cfg, dict):
        submission_path = submission_cfg.get("path")
        if isinstance(submission_path, str) and submission_path.strip():
            resolved_submission = Path(submission_path.strip())
            if not resolved_submission.is_absolute():
                resolved_submission = (config_path.parent / resolved_submission).resolve()
            submission_cfg["path"] = str(resolved_submission)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(hidden_cfg, sort_keys=False))
    return out_path


def main() -> None:
    args = parse_args()
    if args.skip_hidden_scoring and args.score_hidden_only:
        raise ValueError("`--skip-hidden-scoring` and `--score-hidden-only` are mutually exclusive.")
    _require_sealed_hidden_runtime(disable_hidden=bool(args.disable_hidden))
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
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    train_manifest_path = Path(str(data_cfg.get("train_manifest", track_spec.train_manifest))).resolve()
    val_manifest_path = Path(str(data_cfg.get("val_manifest", track_spec.val_manifest))).resolve()
    all_manifest_path = Path(str(track_spec.all_manifest)).resolve() if track_spec.all_manifest else None

    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    public_predict_summary_path = run_dir / "predict_val.json"
    eval_public_summary_path = run_dir / "eval_val.json"
    public_pred_dir = run_dir / "public_predictions"
    per_chain_public_path = run_dir / "per_chain_scores_val.jsonl"
    result_path = run_dir / "result.json"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_last_path = ckpt_dir / "ckpt_last.pt"
    hidden_predict_summary_path = run_dir / "predict_hidden.json"
    hidden_eval_summary_path = run_dir / "eval_hidden.json"
    hidden_eval_cfg_path = (config_path.parent / f".{run_name}_hidden_predict_config.yaml").resolve()
    hidden_pred_dir = run_dir / "hidden_predictions"
    forbid_labels_dir = run_dir / "_forbidden_eval_labels_mount"
    forbid_labels_dir.mkdir(parents=True, exist_ok=True)

    if not args.score_hidden_only:
        _run(
            [
                args.python,
                "scripts/validate_submission.py",
                "--submission",
                str(submission_dir),
                "--track",
                track_spec.track_id,
                "--strict",
            ],
            env=_scrub_hidden_env(stage="validate"),
        )

        if not args.skip_train:
            _run(
                [
                    args.python,
                    "train.py",
                    "--config",
                    str(config_path),
                    "--track",
                    track_spec.track_id,
                    "--official",
                ],
                env=_scrub_hidden_env(stage="train"),
            )

        if not ckpt_last_path.exists():
            raise FileNotFoundError(f"Missing checkpoint after training: {ckpt_last_path}")

        score_labels_dir = str(data_cfg.get("processed_labels_dir", "")).strip()
        if not score_labels_dir:
            raise ValueError(
                "Config data.processed_labels_dir is required for public validation scoring in the official runner."
            )

        _run(
            _build_predict_command(
                python=args.python,
                config_path=config_path,
                split="val",
                track_id=track_spec.track_id,
                official=True,
                pred_out_dir=public_pred_dir,
                save_path=public_predict_summary_path,
                ckpt=ckpt_last_path,
                forbid_labels_dir=forbid_labels_dir,
            ),
            env=_scrub_hidden_env(stage="predict"),
        )
        _run(
            _build_score_command(
                python=args.python,
                prediction_summary=public_predict_summary_path,
                labels_dir=Path(score_labels_dir).resolve(),
                per_chain_out=per_chain_public_path,
                save_path=eval_public_summary_path,
            ),
            env=_scrub_hidden_env(stage="score"),
        )

    eval_public_summary = _read_json(eval_public_summary_path)
    public_per_chain_rows = _read_per_chain(per_chain_public_path)
    public_scores = [float(row["foldscore"]) for row in public_per_chain_rows if "foldscore" in row]
    public_summary = {
        "count": len(public_scores),
        "mean": float(mean(public_scores)) if public_scores else float("nan"),
        "std": float(pstdev(public_scores)) if len(public_scores) > 1 else 0.0,
        "min": float(min(public_scores)) if public_scores else float("nan"),
        "max": float(max(public_scores)) if public_scores else float("nan"),
    }

    hidden_results: Dict[str, Any] = {
        "final_hidden_foldscore": float("nan"),
        "foldscore_auc_hidden": float("nan"),
        "foldscore_at_steps": {},
        "foldscore_at_samples": {},
        "final_hidden_lddt_ca": float("nan"),
        "lddt_auc_hidden": float("nan"),
        "lddt_at_steps": {},
        "lddt_at_samples": {},
        "checkpoint_metrics_hidden": [],
        "per_chain_scores_hidden_path": None,
    }
    hidden_assets_meta: Dict[str, Any] | None = None
    hidden_lock_meta: Dict[str, Any] | None = None

    if not args.disable_hidden:
        _require_pinned_hidden_track_metadata(track_spec)
        hidden_manifest = _resolve_hidden_asset(
            cli_value=args.hidden_manifest,
            track_value=track_spec.hidden_manifest,
            env_key="NANOFOLD_HIDDEN_MANIFEST",
            label="hidden-manifest",
        )
        hidden_features_dir = _resolve_hidden_asset(
            cli_value=args.hidden_features_dir,
            track_value=None,
            env_key="NANOFOLD_HIDDEN_FEATURES_DIR",
            label="hidden-features-dir",
        )
        hidden_fingerprint = _resolve_hidden_asset(
            cli_value=args.hidden_fingerprint,
            track_value=track_spec.hidden_fingerprint_path,
            env_key="NANOFOLD_HIDDEN_FINGERPRINT",
            label="hidden-fingerprint",
        )

        if not args.score_hidden_only:
            _write_temp_predict_config(
                cfg=cfg,
                hidden_features_dir=hidden_features_dir,
                config_path=config_path,
                out_path=hidden_eval_cfg_path,
            )

            checkpoint_tokens = _parse_checkpoint_steps(args.checkpoint_steps)
            checkpoint_steps_arg = ",".join(checkpoint_tokens)

            _run(
                _build_predict_command(
                    python=args.python,
                    config_path=hidden_eval_cfg_path,
                    split="hidden_val",
                    track_id=track_spec.track_id,
                    official=True,
                    pred_out_dir=hidden_pred_dir,
                    save_path=hidden_predict_summary_path,
                    ckpt_dir=ckpt_dir,
                    ckpt_steps=checkpoint_steps_arg,
                    fingerprint=hidden_fingerprint,
                    hidden_manifest=hidden_manifest,
                    forbid_labels_dir=forbid_labels_dir,
                ),
                env=_scrub_hidden_env(stage="predict"),
            )

        if args.skip_hidden_scoring:
            print(f"Hidden prediction complete. Intermediate summary: {hidden_predict_summary_path.resolve()}")
            return

        hidden_labels_dir = _resolve_hidden_asset(
            cli_value=args.hidden_labels_dir,
            track_value=None,
            env_key="NANOFOLD_HIDDEN_LABELS_DIR",
            label="hidden-labels-dir",
        )
        lock_file_value = args.hidden_lock_file.strip() or track_spec.hidden_lock_file or "leaderboard/official_hidden_assets.lock.json"
        lock_file = Path(lock_file_value).resolve()
        hidden_lock_meta = _validate_hidden_lock(
            lock_path=lock_file,
            hidden_manifest=hidden_manifest,
            hidden_features_dir=hidden_features_dir,
            hidden_labels_dir=hidden_labels_dir,
            hidden_fingerprint=hidden_fingerprint,
            track_spec=track_spec,
        )

        hidden_per_chain_path = run_dir / "per_chain_scores_hidden.jsonl"
        _run(
            _build_score_command(
                python=args.python,
                prediction_summary=hidden_predict_summary_path,
                labels_dir=hidden_labels_dir,
                per_chain_out=hidden_per_chain_path,
                save_path=hidden_eval_summary_path,
            ),
            env=_scrub_hidden_env(stage="score"),
        )
        hidden_eval_summary = _read_json(hidden_eval_summary_path)
        hidden_results = {
            "final_hidden_foldscore": float(hidden_eval_summary.get("final_hidden_foldscore", float("nan"))),
            "foldscore_auc_hidden": float(hidden_eval_summary.get("foldscore_auc_hidden", float("nan"))),
            "foldscore_at_steps": hidden_eval_summary.get("foldscore_at_steps", {}),
            "foldscore_at_samples": hidden_eval_summary.get("foldscore_at_samples", {}),
            "final_hidden_lddt_ca": float(hidden_eval_summary.get("final_hidden_lddt_ca", float("nan"))),
            "lddt_auc_hidden": float(hidden_eval_summary.get("lddt_auc_hidden", float("nan"))),
            "lddt_at_steps": hidden_eval_summary.get("lddt_at_steps", {}),
            "lddt_at_samples": hidden_eval_summary.get("lddt_at_samples", {}),
            "checkpoint_metrics_hidden": hidden_eval_summary.get("checkpoint_metrics_hidden", []),
            "per_chain_scores_hidden_path": hidden_eval_summary.get("per_chain_scores_path"),
        }

        hidden_assets_meta = {
            "manifest_path": str(hidden_manifest),
            "manifest_sha256": _sha256(hidden_manifest),
            "features_dir": str(hidden_features_dir),
            "features_dir_tree_sha256": _tree_sha256(hidden_features_dir),
            "labels_dir": str(hidden_labels_dir),
            "labels_dir_tree_sha256": _tree_sha256(hidden_labels_dir),
            "fingerprint_path": str(hidden_fingerprint),
            "fingerprint_sha256": _sha256(hidden_fingerprint),
            "lock": hidden_lock_meta,
        }

    fingerprint_path = Path(track_spec.fingerprint_path) if track_spec.fingerprint_path else None
    train_metrics_path = run_dir / "metrics.json"
    train_metrics = _read_json(train_metrics_path) if train_metrics_path.exists() else {}

    rank_metric = "foldscore_auc_hidden" if not args.disable_hidden else "public_val_foldscore"
    rank_score = (
        float(hidden_results["foldscore_auc_hidden"])
        if not args.disable_hidden
        else float(eval_public_summary["mean_foldscore"])
    )

    result = {
        "run_name": run_name,
        "submission_name": submission_dir.name,
        "submission_dir": str(submission_dir),
        "config_path": str(config_path),
        "track": track_spec.track_id,
        "rank_metric": rank_metric,
        "rank_score": rank_score,
        "rank_tiebreak_metric": "final_hidden_foldscore" if not args.disable_hidden else None,
        "rank_tiebreak_score": (
            float(hidden_results["final_hidden_foldscore"])
            if not args.disable_hidden
            else float(eval_public_summary["mean_foldscore"])
        ),
        "final_hidden_foldscore": float(hidden_results["final_hidden_foldscore"]),
        "foldscore_auc_hidden": float(hidden_results["foldscore_auc_hidden"]),
        "foldscore_at_steps": hidden_results["foldscore_at_steps"],
        "foldscore_at_samples": hidden_results.get("foldscore_at_samples", {}),
        "final_hidden_lddt_ca": float(hidden_results["final_hidden_lddt_ca"]),
        "lddt_auc_hidden": float(hidden_results["lddt_auc_hidden"]),
        "lddt_at_steps": hidden_results["lddt_at_steps"],
        "lddt_at_samples": hidden_results.get("lddt_at_samples", {}),
        "checkpoint_metrics_hidden": hidden_results["checkpoint_metrics_hidden"],
        "public_val_foldscore": float(eval_public_summary["mean_foldscore"]),
        "public_val_lddt_ca": float(eval_public_summary["mean_lddt_ca"]),
        "public_val_per_chain_summary": public_summary,
        "num_chains_public_val": int(eval_public_summary.get("num_chains", len(public_per_chain_rows))),
        "predict_public_summary_path": str(public_predict_summary_path.resolve()) if public_predict_summary_path.exists() else None,
        "eval_public_summary_path": str(eval_public_summary_path.resolve()),
        "per_chain_public_scores_path": str(per_chain_public_path.resolve()),
        "predict_hidden_summary_path": str(hidden_predict_summary_path.resolve()) if hidden_predict_summary_path.exists() else None,
        "eval_hidden_summary_path": str(hidden_eval_summary_path.resolve()) if hidden_eval_summary_path.exists() else None,
        "per_chain_hidden_scores_path": hidden_results["per_chain_scores_hidden_path"],
        "hidden_eval_config_path": str(hidden_eval_cfg_path.resolve()) if hidden_eval_cfg_path.exists() else None,
        "train_metrics_path": str(train_metrics_path.resolve()),
        "train_wall_time_seconds": float(train_metrics.get("wall_time_seconds", float("nan")))
        if isinstance(train_metrics, dict)
        else float("nan"),
        "eval_public_wall_time_seconds": float(
            eval_public_summary.get("score_wall_time_seconds", eval_public_summary.get("eval_wall_time_seconds", float("nan")))
        ),
        "sample_budget": int(train_metrics.get("sample_budget", 0)) if isinstance(train_metrics, dict) else 0,
        "residue_budget": int(train_metrics.get("residue_budget", 0)) if isinstance(train_metrics, dict) else 0,
        "cumulative_samples_seen": int(train_metrics.get("cumulative_samples_seen", 0)) if isinstance(train_metrics, dict) else 0,
        "cumulative_cropped_residues_seen": int(train_metrics.get("cumulative_cropped_residues_seen", 0))
        if isinstance(train_metrics, dict)
        else 0,
        "cumulative_nonpad_residues_seen": int(train_metrics.get("cumulative_nonpad_residues_seen", 0))
        if isinstance(train_metrics, dict)
        else 0,
        "fingerprint_path": str(fingerprint_path.resolve()) if fingerprint_path else None,
        "fingerprint_sha256": _sha256(fingerprint_path) if fingerprint_path and fingerprint_path.exists() else None,
        "hidden_assets": hidden_assets_meta,
        "manifest_paths": {
            "train_manifest": str(train_manifest_path),
            "val_manifest": str(val_manifest_path),
            "all_manifest": str(all_manifest_path) if all_manifest_path else None,
        },
        "manifest_sha256": {
            "train_manifest": _maybe_sha256(train_manifest_path),
            "val_manifest": _maybe_sha256(val_manifest_path),
            "all_manifest": _maybe_sha256(all_manifest_path) if all_manifest_path else None,
        },
        "track_manifest_sha256": {
            "train_manifest": track_spec.train_manifest_sha256,
            "val_manifest": track_spec.val_manifest_sha256,
            "all_manifest": track_spec.all_manifest_sha256,
            "hidden_manifest": track_spec.hidden_manifest_sha256,
        },
        "config_sha256": _sha256(config_path),
        "commit": commit,
        "description": args.description.strip() or f"Official run for {submission_dir.name}",
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "train_env": train_metrics.get("env", {}) if isinstance(train_metrics, dict) else {},
        "eval_public_env": eval_public_summary.get("env", {}) if isinstance(eval_public_summary, dict) else {},
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
