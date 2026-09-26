"""Run nanoFold training on a Modal GPU.

Public features and labels live in read-only Modal Volumes. Run outputs live
in a separate writable Volume so checkpoints survive local disconnects and
remote container restarts.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import modal  # pyright: ignore[reportMissingImports]
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nanofold.modal_training import (
    BUNDLE_MANIFEST,
    CUDA_ENVIRONMENT,
    RUNTIME_PACKAGES,
    bind_execution,
    exclusive_run,
    extract_regular_archive,
    read_object,
    runtime_record,
    safe_relative,
    sha256,
    source_records,
    supervise_training,
    validate_bundle,
)

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

BUNDLE_PATH = os.environ.get("NANOFOLD_MODAL_TRAINING_BUNDLE", "")
# An installed bundle is discovered in the image without carrying a host path
# into the worker environment.
INSTALLED_BUNDLE = (REPO_ROOT / BUNDLE_MANIFEST).is_file()
BUNDLE = validate_bundle(REPO_ROOT if INSTALLED_BUNDLE else Path(BUNDLE_PATH), repo=REPO_ROOT,
                         installed=INSTALLED_BUNDLE) if (BUNDLE_PATH or INSTALLED_BUNDLE) else None
if BUNDLE is not None:
    names = (FEATURES_VOLUME_NAME, LABELS_VOLUME_NAME, RUNS_VOLUME_NAME)
    if len(set(names)) != 3 or any(name in {"nanofold-public-features", "nanofold-public-labels", "nanofold-runs"}
                                 for name in names):
        raise ValueError("Candidate training requires three distinct, explicitly named experiment volumes")

app = modal.App("nanofold-train")
image = (
    modal.Image.debian_slim(python_version="3.12").pip_install(*RUNTIME_PACKAGES).workdir(str(REMOTE_ROOT))
    .env({**CUDA_ENVIRONMENT, "NANOFOLD_MODAL_GPU": GPU_SPEC, "NANOFOLD_MODAL_FEATURES_VOLUME": FEATURES_VOLUME_NAME,
          "NANOFOLD_MODAL_LABELS_VOLUME": LABELS_VOLUME_NAME, "NANOFOLD_MODAL_RUNS_VOLUME": RUNS_VOLUME_NAME})
)
# Only executable sources and declared public metadata are sent to the image.
# In particular, do not recursively upload submissions/ with local outputs.
for relative in source_records(REPO_ROOT):
    image = image.add_local_file(str(REPO_ROOT / relative), remote_path=str(REMOTE_ROOT / relative))
if BUNDLE is None:
    metadata_files = ["data/manifests/train.txt", "data/manifests/val.txt", "data/manifests/all.txt",
                      "leaderboard/official_dataset_fingerprint.json",
                      "leaderboard/research_large_dataset_fingerprint.json",
                      "leaderboard/unlimited_dataset_fingerprint.json",
                      "leaderboard/official_manifest_source.lock.json"]
    for relative in metadata_files:
        image = image.add_local_file(str(REPO_ROOT / relative), remote_path=str(REMOTE_ROOT / relative))
else:
    bundle_root = REPO_ROOT if INSTALLED_BUNDLE else Path(BUNDLE_PATH)
    for relative in [*BUNDLE["files"], BUNDLE_MANIFEST]:
        image = image.add_local_file(str(bundle_root / relative), remote_path=str(REMOTE_ROOT / relative))

features_volume = modal.Volume.from_name(FEATURES_VOLUME_NAME, create_if_missing=True)
labels_volume = modal.Volume.from_name(LABELS_VOLUME_NAME, create_if_missing=True)
runs_volume = modal.Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True)
run_claims = modal.Dict.from_name("nanofold-training-writer-claims-v1", create_if_missing=True)


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


def _upload_directory(*, local_dir: Path, volume_name: str, label: str, files: list[Path]) -> None:
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
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


def _build_archive(*, local_dir: Path, archive_path: Path, label: str, files: list[Path]) -> None:
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


def _upload_archive(*, local_dir: Path, volume_name: str, label: str, archive_name: str, files: list[Path]) -> None:
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    with tempfile.TemporaryDirectory(prefix="nanofold-modal-upload-") as tmpdir:
        archive_path = Path(tmpdir) / archive_name
        _build_archive(local_dir=local_dir, archive_path=archive_path, label=label, files=files)
        remote_path = f"/{ARCHIVE_DIR}/{archive_name}"
        print(f"[modal] uploading {label} archive to volume `{volume_name}`:{remote_path}", flush=True)
        with volume.batch_upload(force=True) as batch:
            batch.put_file(str(archive_path), remote_path)
    print(f"[modal] uploaded {label} archive to `{volume_name}`", flush=True)


def _safe_extract_tar(archive_path: Path, target_dir: Path) -> None:
    extract_regular_archive(archive_path, target_dir)


def _extract_archive_to_stage(*, archive_path: Path, target_dir: Path, label: str) -> None:
    if target_dir.exists() and any(target_dir.iterdir()):
        raise ValueError("Training must use a fresh per-call staging directory")

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
        raise ValueError("Training must use a fresh per-call staging directory")
    if any(path.is_symlink() for path in source_dir.rglob("*")):
        raise ValueError("Mounted public data must not contain links")
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


def _stage_public_data(stage_root: Path) -> tuple[Path, Path]:
    staged_features = stage_root / "processed_features"
    staged_labels = stage_root / "processed_labels"
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
    cfg = yaml.safe_load(config_path.read_text())
    run_name = _run_name(cfg)
    candidate = REMOTE_ROOT / "runs" / run_name / "checkpoints" / "ckpt_last.pt"
    if candidate.exists():
        print(f"[modal] auto-resume: found {candidate}")
        return [*argv, "--resume", str(candidate)]
    print(f"[modal] auto-resume: no checkpoint found at {candidate}")
    return argv


def _run_name(cfg: object) -> str:
    import re

    name = cfg.get("run_name") if isinstance(cfg, dict) else None
    if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None:
        raise ValueError("run_name must be one safe directory name")
    return name


def _public_upload_files(local_dir: Path, manifests: list[Path], *, features: bool) -> list[Path]:
    from nanofold.chain_paths import chain_npz_path
    from nanofold.data import read_manifest

    ids = {chain for manifest in manifests for chain in read_manifest(manifest)}
    expected = {chain_npz_path(local_dir, chain) for chain in ids}
    if features and (local_dir / "preprocess_meta.json").is_file():
        expected.add(local_dir / "preprocess_meta.json")
    observed = {path for path in local_dir.rglob("*.npz") if path.is_file()}
    expected_npz = {path for path in expected if path.suffix == ".npz"}
    if observed != expected_npz or any(path.is_symlink() for path in local_dir.rglob("*")):
        raise ValueError("Upload must contain exactly the selected public train/validation assets and no links")
    return sorted(expected)


def _run_training_segment(argv: list[str], *, remote_config_path: str, auto_resume: bool,
              requested_gpu: str, max_runtime_seconds: float = 22 * 3600) -> dict[str, object]:
    started = time.monotonic()
    os.chdir(REMOTE_ROOT)
    config_path = Path(remote_config_path)
    cfg = yaml.safe_load(config_path.read_text())
    name = _run_name(cfg)
    if not math.isfinite(max_runtime_seconds) or not 0 < max_runtime_seconds <= 22 * 3600:
        raise ValueError("max_runtime_seconds must be positive and at most 22 hours")
    metadata = validate_bundle(REMOTE_ROOT, repo=REMOTE_ROOT, installed=True) if BUNDLE is not None else None
    if metadata is not None:
        run = metadata["runs"].get(name)
        if run is None or config_path != REMOTE_ROOT / run["config"] or "--official" in argv:
            raise ValueError("Remote config is not a declared candidate run")
        fingerprint = REMOTE_ROOT / metadata["fingerprint"]
        if "--verify-fingerprint" not in argv or "--deterministic" not in argv or str(fingerprint) not in argv:
            raise ValueError("Candidate training must verify its bound fingerprint deterministically")
    else:
        from nanofold.competition_policy import load_track_spec
        fingerprint = REMOTE_ROOT / str(load_track_spec(cfg.get("track", "limited")).fingerprint_path)
    binding = {
        "schema_version": 1, "source_files": source_records(REMOTE_ROOT),
        "config_sha256": sha256(config_path), "fingerprint_sha256": sha256(fingerprint),
        "bundle_sha256": sha256(REMOTE_ROOT / BUNDLE_MANIFEST) if metadata is not None else None,
        "runtime": runtime_record(), "requested_gpu": requested_gpu,
        "volumes": {"features": FEATURES_VOLUME_NAME, "labels": LABELS_VOLUME_NAME, "runs": RUNS_VOLUME_NAME},
    }
    binding_path = REMOTE_ROOT / "runs" / ".modal_execution" / f"{name}.json"
    run_dir = REMOTE_ROOT / "runs" / name
    if not binding_path.exists() and run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("Existing run has no Modal execution identity; choose a fresh run name")
    bind_execution(binding_path, binding)
    runs_volume.commit()
    checkpoint = run_dir / "checkpoints" / "ckpt_last.pt"
    if "--resume" in argv:
        resume = Path(argv[argv.index("--resume") + 1]).resolve()
        if not resume.is_relative_to((run_dir / "checkpoints").resolve()):
            raise ValueError("Resume checkpoint must belong to the same bound run")
    if auto_resume:
        argv = _append_auto_resume(argv, config_path=config_path)
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="call-", dir=STAGE_ROOT) as stage:
        manifests = [REMOTE_ROOT / cfg["data"][f"{role}_manifest"] for role in ("train", "val")]
        _check_public_mount(FEATURES_MOUNT, FEATURES_ARCHIVE_NAME, manifests, features=True)
        _check_public_mount(LABELS_MOUNT, LABELS_ARCHIVE_NAME, manifests, features=False)
        staged_features, staged_labels = _stage_public_data(Path(stage))
        for path, is_feature in ((staged_features, True), (staged_labels, False)):
            selected = _public_upload_files(path, manifests, features=is_feature)
            if {item for item in path.rglob("*") if item.is_file()} != set(selected):
                raise ValueError("Staged volume includes files outside public training assets")
        # Leave one hour below the platform deadline for graceful termination,
        # validation/checkpoint work and the final volume commit.
        remaining = 23 * 3600 - (time.monotonic() - started)
        if remaining <= 3600:
            raise TimeoutError("Data staging exhausted the safe training window")
        budget = min(max_runtime_seconds, remaining - 3600)
        command = [sys.executable, "-u", "train.py", *argv,
                   "--processed-features-dir", str(staged_features),
                   "--processed-labels-dir", str(staged_labels),
                   "--max-runtime-seconds", str(budget)]
        print(f"[modal] gpu={requested_gpu}; runtime budget={budget:.0f}s", flush=True)
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env.update(CUDA_ENVIRONMENT)
        code = supervise_training(command, cwd=REMOTE_ROOT, env=env, checkpoint=checkpoint,
                                  commit=runs_volume.commit, timeout_seconds=remaining)
    if code not in (0, 75):
        raise RuntimeError(f"Training exited with status {code}; inspect the bound run logs")
    result: dict[str, object] = {"run_name": name, "status": "paused" if code == 75 else "completed",
                                "exit_code": code, "checkpoint_committed": checkpoint.is_file()}
    if not checkpoint.is_file():
        raise RuntimeError("Training returned without a durable checkpoint")
    result_path = binding_path.with_suffix(".result.json")
    temporary = result_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(result_path)
    runs_volume.commit()
    print(f"[modal] {result['status']}; relaunch the same frozen run to continue if paused", flush=True)
    return result


def _check_public_mount(mount: Path, archive_name: str, manifests: list[Path], *, features: bool) -> None:
    observed = {path for path in mount.rglob("*") if path.is_file()}
    if any(path.is_symlink() for path in mount.rglob("*")):
        raise ValueError("Mounted public data must not contain links")
    archive = mount / ARCHIVE_DIR / archive_name
    allowed = {archive} if archive.is_file() else set(_public_upload_files(mount, manifests, features=features))
    if observed != allowed:
        raise ValueError("Mounted volume includes undeclared files; provision a clean public data volume")


@app.function(
    image=image,
    gpu=GPU_SPEC,
    volumes={
        str(FEATURES_MOUNT): features_volume.read_only(),
        str(LABELS_MOUNT): labels_volume.read_only(),
        str(REMOTE_ROOT / "runs"): runs_volume,
    },
    timeout=60 * 60 * 24,
    block_network=True,
)
def run_train(config_relative: str, *, track: str, auto_resume: bool, requested_gpu: str,
              reset_run: bool = False, resume_path: str = "",
              max_runtime_seconds: float = 22 * 3600) -> dict[str, object]:
    """Construct the complete trainer command from frozen inputs, never caller argv."""
    relative = safe_relative(config_relative).as_posix()
    metadata = validate_bundle(REMOTE_ROOT, repo=REMOTE_ROOT, installed=True) if BUNDLE is not None else None
    if metadata is None:
        if relative not in source_records(REMOTE_ROOT) or Path(relative).suffix not in {".yaml", ".yml"}:
            raise ValueError("Official config is not part of the packaged source inventory")
    elif relative not in {run["config"] for run in metadata["runs"].values()}:
        raise ValueError("Config is not a declared candidate run")
    config_path = REMOTE_ROOT / relative
    cfg = yaml.safe_load(config_path.read_text())
    name = _run_name(cfg)
    if track != cfg.get("track", "limited") or requested_gpu != GPU_SPEC:
        raise ValueError("Requested track or GPU differs from the frozen execution selection")
    if type(auto_resume) is not bool or type(reset_run) is not bool or not isinstance(resume_path, str):
        raise ValueError("Invalid runtime controls")
    if reset_run and resume_path:
        raise ValueError("reset_run and resume_path cannot be combined")
    if not math.isfinite(max_runtime_seconds) or not 0 < max_runtime_seconds <= 22 * 3600:
        raise ValueError("max_runtime_seconds must be positive and at most 22 hours")
    argv = ["--config", str(config_path), "--track", track]
    if metadata is None:
        argv.append("--official")
    else:
        argv += ["--verify-fingerprint", "--fingerprint", str(REMOTE_ROOT / metadata["fingerprint"]), "--deterministic"]
    if reset_run:
        argv.append("--reset-run")
    if resume_path:
        argv += ["--resume", resume_path]
    key = f"{RUNS_VOLUME_NAME}/{name}"
    marker = REMOTE_ROOT / "runs" / ".modal_execution" / f"{name}.active.json"
    with exclusive_run(claims=run_claims, key=key, marker=marker, reload_volume=runs_volume.reload,
                       commit=runs_volume.commit, call_id=modal.current_function_call_id()):
        return _run_training_segment(argv, remote_config_path=str(config_path), auto_resume=auto_resume,
                                     requested_gpu=requested_gpu, max_runtime_seconds=max_runtime_seconds)


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
    background: bool = False,
    run_name: str = "",
    max_runtime_seconds: float = 22 * 3600,
) -> None:
    """Train official defaults or select --run-name from a frozen candidate bundle."""
    if not math.isfinite(max_runtime_seconds) or not 0 < max_runtime_seconds <= 22 * 3600:
        raise ValueError("max_runtime_seconds must be positive and at most 22 hours")
    if reset_run and resume_path:
        raise ValueError("reset_run and resume_path cannot be combined")
    if BUNDLE is not None:
        metadata = validate_bundle(Path(BUNDLE_PATH), repo=REPO_ROOT)
        if run_name not in metadata["runs"]:
            raise ValueError("Select a declared candidate --run-name")
        if config != "submissions/minalphafold2/config.yaml":
            raise ValueError("A bundle selects config by --run-name; do not also pass --config")
        selected = metadata["runs"][run_name]
        if track != selected["track"]:
            raise ValueError("Requested track differs from the frozen candidate config")
        local_config = Path(BUNDLE_PATH) / selected["config"]
        remote_config_path = str(REMOTE_ROOT / selected["config"])
        fingerprint = Path(BUNDLE_PATH) / metadata["fingerprint"]
        metadata_root = Path(BUNDLE_PATH)
    else:
        if run_name:
            raise ValueError("--run-name requires NANOFOLD_MODAL_TRAINING_BUNDLE")
        relative_config = _repo_relative(config).as_posix()
        if relative_config not in source_records(REPO_ROOT):
            raise ValueError("Config is not packaged in the image; prepare a candidate bundle")
        local_config = REPO_ROOT / relative_config
        remote_config_path = str(REMOTE_ROOT / relative_config)
        from nanofold.competition_policy import load_track_spec
        fingerprint = REPO_ROOT / str(load_track_spec(track).fingerprint_path)
        metadata_root = REPO_ROOT
    cfg = yaml.safe_load(local_config.read_text())
    _run_name(cfg)
    if cfg.get("track", "limited") != track:
        raise ValueError("Requested track differs from config")
    manifests = [metadata_root / cfg["data"][f"{role}_manifest"] for role in ("train", "val")]
    if upload_format not in {"archive", "files"}:
        raise ValueError("upload_format must be archive or files")
    if upload_data:
        from nanofold.dataset_integrity import verify_dataset_against_fingerprint
        local_features_dir = _require_local_dir(features_dir, label="processed features")
        local_labels_dir = _require_local_dir(labels_dir, label="processed labels")
        feature_files = _public_upload_files(local_features_dir, manifests, features=True)
        label_files = _public_upload_files(local_labels_dir, manifests, features=False)
        fp = read_object(fingerprint)
        verify_dataset_against_fingerprint(
            processed_features_dir=local_features_dir, processed_labels_dir=local_labels_dir,
            train_manifest=manifests[0], val_manifest=manifests[1], expected_fingerprint_path=fingerprint,
            require_no_missing=True, require_labels=True, validate_schema=True, track_id=track,
            source_lock_path=metadata_root / fp["source_lock_path"],
        )
        for local_dir, volume_name, label, archive_name, files in (
            (local_features_dir, FEATURES_VOLUME_NAME, "processed feature", FEATURES_ARCHIVE_NAME, feature_files),
            (local_labels_dir, LABELS_VOLUME_NAME, "processed label", LABELS_ARCHIVE_NAME, label_files),
        ):
            if upload_format == "archive":
                _upload_archive(local_dir=local_dir, volume_name=volume_name, label=label, archive_name=archive_name, files=files)
            else:
                _upload_directory(local_dir=local_dir, volume_name=volume_name, label=label, files=files)
    if skip_train:
        print("[modal] skipping remote training launch", flush=True)
        return
    kwargs = {"track": track, "auto_resume": auto_resume, "reset_run": reset_run, "resume_path": resume_path,
              "requested_gpu": GPU_SPEC, "max_runtime_seconds": max_runtime_seconds}
    relative = Path(remote_config_path).relative_to(REMOTE_ROOT).as_posix()
    if background:
        call = run_train.spawn(relative, **kwargs)
        print(f"[modal] spawned training call: {call.object_id}", flush=True)
        print(f"[modal] call dashboard: {call.get_dashboard_url()}", flush=True)
    else:
        print(run_train.remote(relative, **kwargs), flush=True)
