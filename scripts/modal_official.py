"""Run maintainer-only official nanoFold evaluation on Modal.

The Modal flow mirrors the local sealed runner:

1. Prediction mounts submission code, public data, hidden features, and hidden
   fingerprints. Hidden labels and hidden locks are absent.
2. Scoring mounts saved predictions, hidden labels, hidden features, hidden
   fingerprints, and the private hidden lock. It does not run submission hooks.

Public and hidden datasets live in separate Modal Volumes. Run outputs live in
the shared runs Volume used by scripts/modal_train.py.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

import modal  # pyright: ignore[reportMissingImports]
import yaml

from nanofold.leaderboard_identity import resolve_leaderboard_team

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/nanofold")

GPU_SPEC = os.environ.get("NANOFOLD_MODAL_GPU", "A10G")
PUBLIC_FEATURES_VOLUME_NAME = os.environ.get("NANOFOLD_MODAL_FEATURES_VOLUME", "nanofold-public-features")
PUBLIC_LABELS_VOLUME_NAME = os.environ.get("NANOFOLD_MODAL_LABELS_VOLUME", "nanofold-public-labels")
HIDDEN_FEATURES_VOLUME_NAME = os.environ.get("NANOFOLD_MODAL_HIDDEN_FEATURES_VOLUME", "nanofold-hidden-features")
HIDDEN_LABELS_VOLUME_NAME = os.environ.get("NANOFOLD_MODAL_HIDDEN_LABELS_VOLUME", "nanofold-hidden-labels")
HIDDEN_META_VOLUME_NAME = os.environ.get("NANOFOLD_MODAL_HIDDEN_META_VOLUME", "nanofold-hidden-meta")
RUNS_VOLUME_NAME = os.environ.get("NANOFOLD_MODAL_RUNS_VOLUME", "nanofold-runs")

PUBLIC_FEATURES_MOUNT = Path("/mnt/nanofold-public-features")
PUBLIC_LABELS_MOUNT = Path("/mnt/nanofold-public-labels")
HIDDEN_FEATURES_MOUNT = Path("/mnt/nanofold-hidden-features")
HIDDEN_LABELS_MOUNT = Path("/mnt/nanofold-hidden-labels")
HIDDEN_META_MOUNT = Path("/mnt/nanofold-hidden-meta")
STAGE_ROOT = Path(os.environ.get("NANOFOLD_MODAL_STAGE_DIR", "/tmp/nanofold-modal"))
ARCHIVE_DIR = "archives"
PUBLIC_FEATURES_ARCHIVE_NAME = "processed_features.tar"
PUBLIC_LABELS_ARCHIVE_NAME = "processed_labels.tar"
HIDDEN_FEATURES_ARCHIVE_NAME = "hidden_processed_features.tar"
HIDDEN_LABELS_ARCHIVE_NAME = "hidden_processed_labels.tar"

HIDDEN_MANIFEST_VOLUME_PATH = Path("/hidden_val.txt")
HIDDEN_FINGERPRINT_VOLUME_PATH = Path("/official_hidden_fingerprint.json")
HIDDEN_LOCK_VOLUME_PATH = Path("/private_hidden_assets.lock.json")
MODAL_RESULT_NAME = "modal_official_result.json"

app = modal.App("nanofold-official")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1",
        "numpy>=1.24",
        "tqdm>=4.66",
        "pyyaml>=6.0",
        "gemmi>=0.6",
        "biopython>=1.81",
    )
    .workdir(str(REMOTE_ROOT))
    .add_local_dir(str(REPO_ROOT / "nanofold"), remote_path=str(REMOTE_ROOT / "nanofold"))
    .add_local_dir(str(REPO_ROOT / "scripts"), remote_path=str(REMOTE_ROOT / "scripts"))
    .add_local_dir(str(REPO_ROOT / "submissions"), remote_path=str(REMOTE_ROOT / "submissions"))
    .add_local_dir(str(REPO_ROOT / "configs"), remote_path=str(REMOTE_ROOT / "configs"))
    .add_local_dir(str(REPO_ROOT / "tracks"), remote_path=str(REMOTE_ROOT / "tracks"))
    .add_local_dir(
        str(REPO_ROOT / "third_party" / "minAlphaFold2" / "minalphafold"),
        remote_path=str(REMOTE_ROOT / "third_party" / "minAlphaFold2" / "minalphafold"),
    )
    .add_local_dir(
        str(REPO_ROOT / "third_party" / "minAlphaFold2" / "configs"),
        remote_path=str(REMOTE_ROOT / "third_party" / "minAlphaFold2" / "configs"),
    )
    .add_local_file(str(REPO_ROOT / "train.py"), remote_path=str(REMOTE_ROOT / "train.py"))
    .add_local_file(str(REPO_ROOT / "eval.py"), remote_path=str(REMOTE_ROOT / "eval.py"))
    .add_local_file(str(REPO_ROOT / "predict.py"), remote_path=str(REMOTE_ROOT / "predict.py"))
    .add_local_file(str(REPO_ROOT / "score.py"), remote_path=str(REMOTE_ROOT / "score.py"))
    .add_local_file(
        str(REPO_ROOT / "data" / "manifests" / "train.txt"),
        remote_path=str(REMOTE_ROOT / "data" / "manifests" / "train.txt"),
    )
    .add_local_file(
        str(REPO_ROOT / "data" / "manifests" / "val.txt"),
        remote_path=str(REMOTE_ROOT / "data" / "manifests" / "val.txt"),
    )
    .add_local_file(
        str(REPO_ROOT / "data" / "manifests" / "all.txt"),
        remote_path=str(REMOTE_ROOT / "data" / "manifests" / "all.txt"),
    )
    .add_local_file(
        str(REPO_ROOT / "leaderboard" / "official_dataset_fingerprint.json"),
        remote_path=str(REMOTE_ROOT / "leaderboard" / "official_dataset_fingerprint.json"),
    )
    .add_local_file(
        str(REPO_ROOT / "leaderboard" / "research_large_dataset_fingerprint.json"),
        remote_path=str(REMOTE_ROOT / "leaderboard" / "research_large_dataset_fingerprint.json"),
    )
    .add_local_file(
        str(REPO_ROOT / "leaderboard" / "unlimited_dataset_fingerprint.json"),
        remote_path=str(REMOTE_ROOT / "leaderboard" / "unlimited_dataset_fingerprint.json"),
    )
    .add_local_file(
        str(REPO_ROOT / "leaderboard" / "official_manifest_source.lock.json"),
        remote_path=str(REMOTE_ROOT / "leaderboard" / "official_manifest_source.lock.json"),
    )
)

public_features_volume = modal.Volume.from_name(PUBLIC_FEATURES_VOLUME_NAME, create_if_missing=True)
public_labels_volume = modal.Volume.from_name(PUBLIC_LABELS_VOLUME_NAME, create_if_missing=True)
hidden_features_volume = modal.Volume.from_name(HIDDEN_FEATURES_VOLUME_NAME, create_if_missing=True)
hidden_labels_volume = modal.Volume.from_name(HIDDEN_LABELS_VOLUME_NAME, create_if_missing=True)
hidden_meta_volume = modal.Volume.from_name(HIDDEN_META_VOLUME_NAME, create_if_missing=True)
runs_volume = modal.Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True)


def _repo_relative(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    resolved = candidate.resolve()
    try:
        return resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise SystemExit(f"Path must live inside the repo: {resolved}") from exc


def _remote_repo_path(path: str | Path) -> str:
    return str(REMOTE_ROOT / _repo_relative(path))


def _require_local_dir(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    resolved = resolved.resolve()
    if not resolved.exists():
        raise SystemExit(f"{label} directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise SystemExit(f"{label} path is not a directory: {resolved}")
    return resolved


def _require_local_file(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    resolved = resolved.resolve()
    if not resolved.exists():
        raise SystemExit(f"{label} file does not exist: {resolved}")
    if not resolved.is_file():
        raise SystemExit(f"{label} path is not a file: {resolved}")
    return resolved


def _build_archive(*, local_dir: Path, archive_path: Path, label: str) -> None:
    files = [path for path in local_dir.rglob("*") if path.is_file()]
    if not files:
        raise SystemExit(f"{label} directory has no files to archive: {local_dir}")

    started = time.monotonic()
    print(f"[modal] building {label} archive with {len(files)} files", flush=True)
    with tarfile.open(archive_path, "w") as tar:
        for index, path in enumerate(sorted(files), start=1):
            tar.add(path, arcname=path.relative_to(local_dir).as_posix(), recursive=False)
            if index % 1000 == 0 or index == len(files):
                print(f"[modal] archived {index}/{len(files)} {label} files", flush=True)
    size_gb = archive_path.stat().st_size / (1024**3)
    elapsed = time.monotonic() - started
    print(f"[modal] built {label} archive: {size_gb:.2f} GiB in {elapsed:.1f}s", flush=True)


def _upload_archive(*, local_dir: Path, volume_name: str, archive_name: str, label: str) -> None:
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    with tempfile.TemporaryDirectory(prefix="nanofold-modal-upload-") as tmpdir:
        archive_path = Path(tmpdir) / archive_name
        _build_archive(local_dir=local_dir, archive_path=archive_path, label=label)
        remote_path = f"/{ARCHIVE_DIR}/{archive_name}"
        print(f"[modal] uploading {label} archive to `{volume_name}`:{remote_path}", flush=True)
        with volume.batch_upload(force=True) as batch:
            batch.put_file(str(archive_path), remote_path)
    print(f"[modal] uploaded {label} archive to `{volume_name}`", flush=True)


def _upload_file(*, local_file: Path, volume_name: str, remote_path: Path, label: str) -> None:
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    print(f"[modal] uploading {label} to `{volume_name}`:{remote_path}", flush=True)
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(local_file), str(remote_path))
    print(f"[modal] uploaded {label}", flush=True)


def _safe_extract_tar(archive_path: Path, target_dir: Path) -> None:
    target_root = target_dir.resolve()
    with tarfile.open(archive_path, "r") as tar:
        members = tar.getmembers()
        for member in members:
            member_path = (target_dir / member.name).resolve()
            if member_path != target_root and target_root not in member_path.parents:
                raise RuntimeError(f"Archive member escapes target directory: {member.name}")
        tar.extractall(target_dir, members=members)


def _extract_archive_to_stage(*, archive_path: Path, target_dir: Path, label: str) -> None:
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"[modal] using staged {label}: {target_dir}", flush=True)
        return

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target_dir.with_name(f"{target_dir.name}.tmp")
    if tmp_target.exists():
        shutil.rmtree(tmp_target)
    tmp_target.mkdir(parents=True)

    started = time.monotonic()
    print(f"[modal] extracting {label} archive from {archive_path} to {target_dir}", flush=True)
    _safe_extract_tar(archive_path, tmp_target)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    tmp_target.replace(target_dir)
    elapsed = time.monotonic() - started
    print(f"[modal] staged {label} in {elapsed:.1f}s", flush=True)


def _copy_volume_to_stage(*, source_dir: Path, target_dir: Path, label: str) -> None:
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"[modal] using staged {label}: {target_dir}", flush=True)
        return
    files = [path for path in source_dir.rglob("*") if path.is_file()]
    if not files:
        raise RuntimeError(f"No {label} files found in mounted volume: {source_dir}")

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target_dir.with_name(f"{target_dir.name}.tmp")
    if tmp_target.exists():
        shutil.rmtree(tmp_target)
    tmp_target.mkdir(parents=True)

    started = time.monotonic()
    print(f"[modal] copying {len(files)} {label} files from {source_dir} to {target_dir}", flush=True)
    for index, source_path in enumerate(sorted(files), start=1):
        relative_path = source_path.relative_to(source_dir)
        destination_path = tmp_target / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        if index % 1000 == 0 or index == len(files):
            print(f"[modal] copied {index}/{len(files)} {label} files", flush=True)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    tmp_target.replace(target_dir)
    elapsed = time.monotonic() - started
    print(f"[modal] staged {label} in {elapsed:.1f}s", flush=True)


def _stage_from_volume(*, mount: Path, archive_name: str, target_dir: Path, label: str) -> Path:
    archive_path = mount / ARCHIVE_DIR / archive_name
    if archive_path.exists():
        _extract_archive_to_stage(archive_path=archive_path, target_dir=target_dir, label=label)
    else:
        _copy_volume_to_stage(source_dir=mount, target_dir=target_dir, label=label)
    return target_dir


def _stage_public_data() -> tuple[Path, Path]:
    features = _stage_from_volume(
        mount=PUBLIC_FEATURES_MOUNT,
        archive_name=PUBLIC_FEATURES_ARCHIVE_NAME,
        target_dir=STAGE_ROOT / "processed_features",
        label="public processed features",
    )
    labels = _stage_from_volume(
        mount=PUBLIC_LABELS_MOUNT,
        archive_name=PUBLIC_LABELS_ARCHIVE_NAME,
        target_dir=STAGE_ROOT / "processed_labels",
        label="public processed labels",
    )
    return features, labels


def _stage_hidden_features() -> Path:
    return _stage_from_volume(
        mount=HIDDEN_FEATURES_MOUNT,
        archive_name=HIDDEN_FEATURES_ARCHIVE_NAME,
        target_dir=STAGE_ROOT / "hidden_processed_features",
        label="hidden processed features",
    )


def _stage_hidden_labels() -> Path:
    return _stage_from_volume(
        mount=HIDDEN_LABELS_MOUNT,
        archive_name=HIDDEN_LABELS_ARCHIVE_NAME,
        target_dir=STAGE_ROOT / "hidden_processed_labels",
        label="hidden processed labels",
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return raw


def _write_modal_config(
    *,
    config_path: Path,
    processed_features_dir: Path,
    processed_labels_dir: Path,
) -> Path:
    cfg = _load_yaml(config_path)
    data_cfg = cfg.get("data")
    if not isinstance(data_cfg, dict):
        raise ValueError(f"Config missing `data` section: {config_path}")
    data_cfg["processed_features_dir"] = str(processed_features_dir)
    data_cfg["processed_labels_dir"] = str(processed_labels_dir)

    submission_cfg = cfg.get("submission")
    if isinstance(submission_cfg, dict):
        submission_path = submission_cfg.get("path")
        if isinstance(submission_path, str) and submission_path.strip():
            resolved_submission = Path(submission_path.strip())
            if not resolved_submission.is_absolute():
                resolved_submission = (config_path.parent / resolved_submission).resolve()
            submission_cfg["path"] = str(resolved_submission)

    out_path = config_path.parent / f".modal_official_{config_path.name}"
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out_path


def _run_command(cmd: list[str], *, env: dict[str, str]) -> None:
    print(f"[modal] command: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)


def _resolve_local_commit(explicit: str) -> str:
    if explicit.strip():
        return explicit.strip()
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _official_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["NANOFOLD_OFFICIAL_SEALED_RUNTIME"] = "1"
    return env


def _run_name_from_config(path: Path) -> str:
    return str(_load_yaml(path).get("run_name", "run")).strip() or "run"


def _result_payload(*, config_path: Path) -> dict[str, Any]:
    run_name = _run_name_from_config(config_path)
    result_path = REMOTE_ROOT / "runs" / run_name / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Expected result artifact after official scoring: {result_path}")
    return {
        "run_name": run_name,
        "result_path": str(result_path),
        "result_json": result_path.read_text(),
    }


def _write_remote_result(payload: dict[str, Any]) -> Path:
    run_name = str(payload["run_name"])
    target = REMOTE_ROOT / "runs" / run_name / MODAL_RESULT_NAME
    target.write_text(str(payload["result_json"]))
    print(f"[modal] wrote result artifact to runs volume: {target}", flush=True)
    return target


def _print_background_call(*, label: str, call: Any) -> None:
    print(f"[modal] spawned {label} call: {call.object_id}", flush=True)
    print(f"[modal] call dashboard: {call.get_dashboard_url()}", flush=True)


@app.function(
    image=image,
    gpu=GPU_SPEC,
    volumes={
        str(PUBLIC_FEATURES_MOUNT): public_features_volume.read_only(),
        str(PUBLIC_LABELS_MOUNT): public_labels_volume.read_only(),
        str(HIDDEN_FEATURES_MOUNT): hidden_features_volume.read_only(),
        str(HIDDEN_META_MOUNT): hidden_meta_volume.read_only(),
        str(REMOTE_ROOT / "runs"): runs_volume,
    },
    timeout=60 * 60 * 24,
    block_network=True,
)
def run_prediction_stage(
    *,
    submission: str,
    config: str,
    track: str,
    description: str,
    team: str,
    checkpoint_steps: str,
    commit: str,
) -> None:
    os.chdir(REMOTE_ROOT)
    public_features, public_labels = _stage_public_data()
    hidden_features = _stage_hidden_features()
    hidden_manifest = HIDDEN_META_MOUNT / HIDDEN_MANIFEST_VOLUME_PATH.relative_to("/")
    hidden_fingerprint = HIDDEN_META_MOUNT / HIDDEN_FINGERPRINT_VOLUME_PATH.relative_to("/")
    modal_config = _write_modal_config(
        config_path=Path(config),
        processed_features_dir=public_features,
        processed_labels_dir=public_labels,
    )

    env = _official_env()
    env["NANOFOLD_HIDDEN_MANIFEST"] = str(hidden_manifest)
    env["NANOFOLD_HIDDEN_FEATURES_DIR"] = str(hidden_features)
    env["NANOFOLD_HIDDEN_FINGERPRINT"] = str(hidden_fingerprint)

    cmd = [
        sys.executable,
        "-u",
        "scripts/run_official.py",
        "--submission",
        submission,
        "--config",
        str(modal_config),
        "--track",
        track,
        "--description",
        description,
        "--team",
        team,
        "--commit",
        commit,
        "--skip-train",
        "--skip-hidden-scoring",
        "--hidden-manifest",
        str(hidden_manifest),
        "--hidden-features-dir",
        str(hidden_features),
        "--hidden-fingerprint",
        str(hidden_fingerprint),
        "--checkpoint-steps",
        checkpoint_steps,
    ]
    try:
        _run_command(cmd, env=env)
    finally:
        runs_volume.commit()


@app.function(
    image=image,
    gpu=GPU_SPEC,
    volumes={
        str(HIDDEN_FEATURES_MOUNT): hidden_features_volume.read_only(),
        str(HIDDEN_LABELS_MOUNT): hidden_labels_volume.read_only(),
        str(HIDDEN_META_MOUNT): hidden_meta_volume.read_only(),
        str(REMOTE_ROOT / "runs"): runs_volume,
    },
    timeout=60 * 60 * 12,
    block_network=True,
)
def run_scoring_stage(
    *,
    submission: str,
    config: str,
    track: str,
    description: str,
    team: str,
    commit: str,
) -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    hidden_features = _stage_hidden_features()
    hidden_labels = _stage_hidden_labels()
    hidden_manifest = HIDDEN_META_MOUNT / HIDDEN_MANIFEST_VOLUME_PATH.relative_to("/")
    hidden_fingerprint = HIDDEN_META_MOUNT / HIDDEN_FINGERPRINT_VOLUME_PATH.relative_to("/")
    hidden_lock = HIDDEN_META_MOUNT / HIDDEN_LOCK_VOLUME_PATH.relative_to("/")

    env = _official_env()
    env["NANOFOLD_HIDDEN_MANIFEST"] = str(hidden_manifest)
    env["NANOFOLD_HIDDEN_FEATURES_DIR"] = str(hidden_features)
    env["NANOFOLD_HIDDEN_LABELS_DIR"] = str(hidden_labels)
    env["NANOFOLD_HIDDEN_FINGERPRINT"] = str(hidden_fingerprint)
    env["NANOFOLD_HIDDEN_LOCK_FILE"] = str(hidden_lock)

    cmd = [
        sys.executable,
        "-u",
        "scripts/run_official.py",
        "--submission",
        submission,
        "--config",
        config,
        "--track",
        track,
        "--description",
        description,
        "--team",
        team,
        "--commit",
        commit,
        "--skip-train",
        "--score-hidden-only",
        "--hidden-manifest",
        str(hidden_manifest),
        "--hidden-features-dir",
        str(hidden_features),
        "--hidden-labels-dir",
        str(hidden_labels),
        "--hidden-fingerprint",
        str(hidden_fingerprint),
        "--hidden-lock-file",
        str(hidden_lock),
    ]
    try:
        _run_command(cmd, env=env)
        payload = _result_payload(config_path=Path(config))
        _write_remote_result(payload)
    finally:
        runs_volume.commit()
    return payload


def _write_local_result(payload: dict[str, Any], *, out_dir: Path) -> Path:
    run_name = str(payload["run_name"])
    result_json = str(payload["result_json"])
    target = out_dir / run_name / MODAL_RESULT_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(result_json)
    print(f"[modal] wrote local result artifact: {target}", flush=True)
    return target


def _update_local_leaderboard(*, result_path: Path, description: str, team: str) -> None:
    cmd = [
        sys.executable,
        "scripts/add_leaderboard_entry.py",
        "--result",
        str(result_path),
        "--leaderboard",
        "leaderboard/leaderboard.json",
        "--readme",
        "README.md",
        "--description",
        description,
        "--team",
        team,
    ]
    print(f"[modal] updating local leaderboard: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


@app.local_entrypoint()
def main(
    submission: str = "submissions/minalphafold2",
    config: str = "submissions/minalphafold2/config.yaml",
    track: str = "limited",
    description: str = "",
    team: str = "",
    commit: str = "",
    checkpoint_steps: str = "0,1000,2000,5000,last",
    upload_public_data: bool = False,
    upload_hidden_assets: bool = False,
    upload_only: bool = False,
    skip_predict: bool = False,
    skip_score: bool = False,
    update_leaderboard: bool = False,
    background_predict: bool = False,
    background_score: bool = False,
    public_features_dir: str = "data/processed_features",
    public_labels_dir: str = "data/processed_labels",
    hidden_manifest: str = ".nanofold_private/manifests/hidden_val.txt",
    hidden_features_dir: str = ".nanofold_private/hidden_processed_features",
    hidden_labels_dir: str = ".nanofold_private/hidden_processed_labels",
    hidden_fingerprint: str = ".nanofold_private/leaderboard/official_hidden_fingerprint.json",
    hidden_lock_file: str = ".nanofold_private/leaderboard/private_hidden_assets.lock.json",
    local_result_dir: str = "runs",
) -> None:
    """Upload maintainer assets and run official hidden evaluation on Modal."""

    if upload_public_data:
        _upload_archive(
            local_dir=_require_local_dir(public_features_dir, label="public processed features"),
            volume_name=PUBLIC_FEATURES_VOLUME_NAME,
            archive_name=PUBLIC_FEATURES_ARCHIVE_NAME,
            label="public processed features",
        )
        _upload_archive(
            local_dir=_require_local_dir(public_labels_dir, label="public processed labels"),
            volume_name=PUBLIC_LABELS_VOLUME_NAME,
            archive_name=PUBLIC_LABELS_ARCHIVE_NAME,
            label="public processed labels",
        )

    if upload_hidden_assets:
        _upload_archive(
            local_dir=_require_local_dir(hidden_features_dir, label="hidden processed features"),
            volume_name=HIDDEN_FEATURES_VOLUME_NAME,
            archive_name=HIDDEN_FEATURES_ARCHIVE_NAME,
            label="hidden processed features",
        )
        _upload_archive(
            local_dir=_require_local_dir(hidden_labels_dir, label="hidden processed labels"),
            volume_name=HIDDEN_LABELS_VOLUME_NAME,
            archive_name=HIDDEN_LABELS_ARCHIVE_NAME,
            label="hidden processed labels",
        )
        _upload_file(
            local_file=_require_local_file(hidden_manifest, label="hidden manifest"),
            volume_name=HIDDEN_META_VOLUME_NAME,
            remote_path=HIDDEN_MANIFEST_VOLUME_PATH,
            label="hidden manifest",
        )
        _upload_file(
            local_file=_require_local_file(hidden_fingerprint, label="hidden fingerprint"),
            volume_name=HIDDEN_META_VOLUME_NAME,
            remote_path=HIDDEN_FINGERPRINT_VOLUME_PATH,
            label="hidden fingerprint",
        )
        _upload_file(
            local_file=_require_local_file(hidden_lock_file, label="hidden lock"),
            volume_name=HIDDEN_META_VOLUME_NAME,
            remote_path=HIDDEN_LOCK_VOLUME_PATH,
            label="hidden lock",
        )

    if upload_only or (skip_predict and skip_score):
        print("[modal] upload complete; skipping official evaluation", flush=True)
        return

    remote_submission = _remote_repo_path(submission)
    remote_config = _remote_repo_path(config)
    run_description = description.strip() or "Modal official hidden evaluation"
    leaderboard_team = resolve_leaderboard_team(
        explicit_team=team,
        submission_name=Path(submission).name,
    )
    resolved_commit = _resolve_local_commit(commit)

    print(f"[modal] app=nanofold-official gpu={GPU_SPEC}", flush=True)
    print(f"[modal] public volumes: features={PUBLIC_FEATURES_VOLUME_NAME} labels={PUBLIC_LABELS_VOLUME_NAME}", flush=True)
    print(
        "[modal] hidden volumes: "
        f"features={HIDDEN_FEATURES_VOLUME_NAME} labels={HIDDEN_LABELS_VOLUME_NAME} meta={HIDDEN_META_VOLUME_NAME}",
        flush=True,
    )
    print(f"[modal] runs volume: {RUNS_VOLUME_NAME}", flush=True)

    if background_predict and not skip_score:
        raise SystemExit("background_predict must be used with skip_score; rerun later with skip_predict to score.")
    if background_score and not skip_predict:
        raise SystemExit("background_score must be used with skip_predict after prediction artifacts already exist.")
    if background_score and update_leaderboard:
        raise SystemExit(
            "background_score writes the result to the runs volume. Download that result and update the leaderboard locally."
        )

    if not skip_predict:
        prediction_kwargs = {
            "submission": remote_submission,
            "config": remote_config,
            "track": track,
            "description": run_description,
            "team": leaderboard_team,
            "checkpoint_steps": checkpoint_steps,
            "commit": resolved_commit,
        }
        if background_predict:
            call = run_prediction_stage.spawn(**prediction_kwargs)
            _print_background_call(label="official prediction", call=call)
        else:
            run_prediction_stage.remote(**prediction_kwargs)

    if not skip_score:
        scoring_kwargs = {
            "submission": remote_submission,
            "config": remote_config,
            "track": track,
            "description": run_description,
            "team": leaderboard_team,
            "commit": resolved_commit,
        }
        if background_score:
            call = run_scoring_stage.spawn(**scoring_kwargs)
            _print_background_call(label="official scoring", call=call)
            run_name = _run_name_from_config((REPO_ROOT / _repo_relative(config)).resolve())
            volume_path = f"/{run_name}/{MODAL_RESULT_NAME}"
            print(f"[modal] result will be written to `{RUNS_VOLUME_NAME}`:{volume_path}", flush=True)
            return
        payload = run_scoring_stage.remote(**scoring_kwargs)
        result_path = _write_local_result(payload, out_dir=(REPO_ROOT / local_result_dir).resolve())
        if update_leaderboard:
            _update_local_leaderboard(result_path=result_path, description=run_description, team=leaderboard_team)
