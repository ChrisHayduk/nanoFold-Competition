"""Run nanoFold training on a Modal GPU.

Public features and labels live in read-only Modal Volumes. Run outputs live
in a separate writable Volume so checkpoints survive local disconnects and
remote container restarts.
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

import modal  # pyright: ignore[reportMissingImports]

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/nanofold")

GPU_SPEC = os.environ.get("NANOFOLD_MODAL_GPU", "A10G")
FEATURES_VOLUME_NAME = os.environ.get("NANOFOLD_MODAL_FEATURES_VOLUME", "nanofold-public-features")
LABELS_VOLUME_NAME = os.environ.get("NANOFOLD_MODAL_LABELS_VOLUME", "nanofold-public-labels")
RUNS_VOLUME_NAME = os.environ.get("NANOFOLD_MODAL_RUNS_VOLUME", "nanofold-runs")
FEATURES_MOUNT = Path("/mnt/nanofold-public-features")
LABELS_MOUNT = Path("/mnt/nanofold-public-labels")
STAGE_ROOT = Path(os.environ.get("NANOFOLD_MODAL_STAGE_DIR", "/tmp/nanofold-modal"))
ARCHIVE_DIR = "archives"
FEATURES_ARCHIVE_NAME = "processed_features.tar"
LABELS_ARCHIVE_NAME = "processed_labels.tar"

app = modal.App("nanofold-train")

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

features_volume = modal.Volume.from_name(FEATURES_VOLUME_NAME, create_if_missing=True)
labels_volume = modal.Volume.from_name(LABELS_VOLUME_NAME, create_if_missing=True)
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
    resolved = (REPO_ROOT / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if not resolved.exists():
        raise SystemExit(f"{label} directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise SystemExit(f"{label} path is not a directory: {resolved}")
    return resolved


def _upload_directory(*, local_dir: Path, volume_name: str, label: str) -> None:
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    files = [path for path in local_dir.rglob("*") if path.is_file()]
    if not files:
        raise SystemExit(f"{label} directory has no files to upload: {local_dir}")
    print(f"[modal] uploading {len(files)} {label} files from {local_dir} to volume `{volume_name}`", flush=True)
    with volume.batch_upload(force=True) as batch:
        for index, path in enumerate(sorted(files), start=1):
            remote_path = "/" + path.relative_to(local_dir).as_posix()
            batch.put_file(str(path), remote_path)
            if index % 1000 == 0 or index == len(files):
                print(f"[modal] queued {index}/{len(files)} {label} files", flush=True)
    print(f"[modal] uploaded {label} to `{volume_name}`", flush=True)


def _build_archive(*, local_dir: Path, archive_path: Path, label: str) -> None:
    files = [path for path in local_dir.rglob("*") if path.is_file()]
    if not files:
        raise SystemExit(f"{label} directory has no files to archive: {local_dir}")

    started = time.monotonic()
    print(f"[modal] building {label} archive with {len(files)} files", flush=True)
    with tarfile.open(archive_path, "w") as tar:
        for index, path in enumerate(sorted(files), start=1):
            arcname = path.relative_to(local_dir).as_posix()
            tar.add(path, arcname=arcname, recursive=False)
            if index % 1000 == 0 or index == len(files):
                print(f"[modal] archived {index}/{len(files)} {label} files", flush=True)
    size_gb = archive_path.stat().st_size / (1024**3)
    elapsed = time.monotonic() - started
    print(f"[modal] built {label} archive: {archive_path} ({size_gb:.2f} GiB, {elapsed:.1f}s)", flush=True)


def _upload_archive(*, local_dir: Path, volume_name: str, label: str, archive_name: str) -> None:
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    with tempfile.TemporaryDirectory(prefix="nanofold-modal-upload-") as tmpdir:
        archive_path = Path(tmpdir) / archive_name
        _build_archive(local_dir=local_dir, archive_path=archive_path, label=label)
        remote_path = f"/{ARCHIVE_DIR}/{archive_name}"
        print(f"[modal] uploading {label} archive to volume `{volume_name}`:{remote_path}", flush=True)
        with volume.batch_upload(force=True) as batch:
            batch.put_file(str(archive_path), remote_path)
    print(f"[modal] uploaded {label} archive to `{volume_name}`", flush=True)


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


def _stage_public_data() -> tuple[Path, Path]:
    staged_features = STAGE_ROOT / "processed_features"
    staged_labels = STAGE_ROOT / "processed_labels"
    feature_archive = FEATURES_MOUNT / ARCHIVE_DIR / FEATURES_ARCHIVE_NAME
    label_archive = LABELS_MOUNT / ARCHIVE_DIR / LABELS_ARCHIVE_NAME

    if feature_archive.exists():
        _extract_archive_to_stage(archive_path=feature_archive, target_dir=staged_features, label="processed features")
    else:
        _copy_volume_to_stage(source_dir=FEATURES_MOUNT, target_dir=staged_features, label="processed feature")

    if label_archive.exists():
        _extract_archive_to_stage(archive_path=label_archive, target_dir=staged_labels, label="processed labels")
    else:
        _copy_volume_to_stage(source_dir=LABELS_MOUNT, target_dir=staged_labels, label="processed label")

    return staged_features, staged_labels


def _append_auto_resume(argv: list[str], *, config_path: Path) -> list[str]:
    if "--resume" in argv or "--reset-run" in argv:
        return argv
    import yaml

    cfg = yaml.safe_load(config_path.read_text())
    run_name = "run"
    if isinstance(cfg, dict):
        run_name = str(cfg.get("run_name") or run_name)
    candidate = REMOTE_ROOT / "runs" / run_name / "checkpoints" / "ckpt_last.pt"
    if candidate.exists():
        print(f"[modal] auto-resume: found {candidate}")
        return [*argv, "--resume", str(candidate)]
    print(f"[modal] auto-resume: no checkpoint found at {candidate}")
    return argv


@app.function(
    image=image,
    gpu=GPU_SPEC,
    volumes={
        str(FEATURES_MOUNT): features_volume.read_only(),
        str(LABELS_MOUNT): labels_volume.read_only(),
        str(REMOTE_ROOT / "runs"): runs_volume,
    },
    timeout=60 * 60 * 24,
)
def run_train(argv: list[str], *, remote_config_path: str, auto_resume: bool, requested_gpu: str) -> None:
    os.chdir(REMOTE_ROOT)
    staged_features, staged_labels = _stage_public_data()
    argv = [
        *argv,
        "--processed-features-dir",
        str(staged_features),
        "--processed-labels-dir",
        str(staged_labels),
    ]
    if auto_resume:
        argv = _append_auto_resume(argv, config_path=Path(remote_config_path))

    print(f"[modal] gpu={requested_gpu}", flush=True)
    print(f"[modal] command: {sys.executable} train.py {' '.join(argv)}", flush=True)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    try:
        subprocess.run([sys.executable, "-u", "train.py", *argv], check=True, env=env)
    finally:
        runs_volume.commit()


@app.local_entrypoint()
def main(
    config: str = "submissions/minalphafold2/config.yaml",
    track: str = "limited",
    upload_data: bool = False,
    skip_train: bool = False,
    reset_run: bool = False,
    auto_resume: bool = True,
    resume_path: str = "",
    upload_format: str = "archive",
    features_dir: str = "data/processed_features",
    labels_dir: str = "data/processed_labels",
) -> None:
    """Upload public data and launch official nanoFold training on Modal."""

    if upload_data:
        local_features_dir = _require_local_dir(features_dir, label="processed features")
        local_labels_dir = _require_local_dir(labels_dir, label="processed labels")
        if upload_format == "archive":
            _upload_archive(
                local_dir=local_features_dir,
                volume_name=FEATURES_VOLUME_NAME,
                label="processed feature",
                archive_name=FEATURES_ARCHIVE_NAME,
            )
            _upload_archive(
                local_dir=local_labels_dir,
                volume_name=LABELS_VOLUME_NAME,
                label="processed label",
                archive_name=LABELS_ARCHIVE_NAME,
            )
        elif upload_format == "files":
            _upload_directory(
                local_dir=local_features_dir,
                volume_name=FEATURES_VOLUME_NAME,
                label="processed feature",
            )
            _upload_directory(
                local_dir=local_labels_dir,
                volume_name=LABELS_VOLUME_NAME,
                label="processed label",
            )
        else:
            raise SystemExit("upload_format must be `archive` or `files`.")

    if skip_train:
        print("[modal] upload complete; skipping remote training launch", flush=True)
        return

    remote_config_path = _remote_repo_path(config)
    argv = [
        "--config",
        remote_config_path,
        "--track",
        track,
        "--official",
    ]
    if reset_run:
        argv.append("--reset-run")
    if resume_path:
        argv += ["--resume", resume_path]

    print(f"[modal] app=nanofold-train gpu={GPU_SPEC}", flush=True)
    print(f"[modal] data volumes: features={FEATURES_VOLUME_NAME} labels={LABELS_VOLUME_NAME}", flush=True)
    print(f"[modal] runs volume: {RUNS_VOLUME_NAME}", flush=True)
    run_train.remote(argv, remote_config_path=remote_config_path, auto_resume=auto_resume, requested_gpu=GPU_SPEC)
