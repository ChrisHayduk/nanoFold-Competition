from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
import sys
import tarfile
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from nanofold.chain_paths import chain_npz_path
from nanofold.dataset_integrity import build_dataset_fingerprint, verify_dataset_against_fingerprint
from nanofold.modal_training import (
    BUNDLE_MANIFEST,
    CUDA_ENVIRONMENT,
    SOURCE_ENTRYPOINTS,
    bind_execution,
    exclusive_run,
    extract_regular_archive,
    prepare_bundle,
    read_object,
    safe_relative,
    sha256,
    supervise_training,
    validate_bundle,
)


@pytest.fixture
def experiment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    for relative in (*SOURCE_ENTRYPOINTS, "submissions/example/submission.py", "configs/profile.toml"):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture source\n")
    manifests = repo / ".private/public"
    manifests.mkdir(parents=True)
    features, labels = repo / "features", repo / "labels"
    features.mkdir()
    labels.mkdir()
    for role, chain in (("train", "1abc_A"), ("val", "2def_B")):
        (manifests / f"{role}.txt").write_text(chain + "\n")
        np.savez(chain_npz_path(features, chain), aatype=np.zeros(6, dtype=np.int32),
                 msa=np.zeros((2, 6), dtype=np.int32), deletions=np.zeros((2, 6), dtype=np.int32),
                 template_aatype=np.zeros((0, 6), dtype=np.int32),
                 template_ca_coords=np.zeros((0, 6, 3), dtype=np.float32),
                 template_ca_mask=np.zeros((0, 6), dtype=bool))
        np.savez(chain_npz_path(labels, chain), ca_coords=np.zeros((6, 3), dtype=np.float32),
                 ca_mask=np.ones(6, dtype=bool), atom14_positions=np.zeros((6, 14, 3), dtype=np.float32),
                 atom14_mask=np.ones((6, 14), dtype=bool))
    lock = manifests / "source.lock.json"
    lock.write_text('{"args": {"hidden_val_size": 1}, "public": true}\n')
    fp = build_dataset_fingerprint(processed_features_dir=features, processed_labels_dir=labels,
                                   train_manifest=manifests / "train.txt", val_manifest=manifests / "val.txt",
                                   require_no_missing=True, track_id="limited", source_lock_path=lock)
    fp["source_lock_path"] = lock.relative_to(repo).as_posix()
    fingerprint = manifests / "fingerprint.json"
    fingerprint.write_text(json.dumps(fp))
    cfg = {"run_name": "replicate_17", "track": "limited", "seed": 17,
           "submission": {"module": "submissions.example.submission"},
           "model": {"profile_path": "configs/profile.toml"},
           "data": {"train_manifest": ".private/public/train.txt", "val_manifest": ".private/public/val.txt",
                    "processed_features_dir": str(features), "processed_labels_dir": str(labels)}}
    config = repo / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    bundle = tmp_path / "bundle"
    return dict(repo=repo, config=config, fingerprint=fingerprint, bundle=bundle,
                features=features, labels=labels, cfg=cfg)


def _prepare(experiment: dict[str, Any]) -> dict[str, Any]:
    return prepare_bundle(repo=experiment["repo"], configs=[experiment["config"]],
                          fingerprint=experiment["fingerprint"], output=experiment["bundle"])


def test_bundle_relocation_preserves_seed_source_lock_and_real_fingerprint(experiment: dict[str, Any],
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = _prepare(experiment)
    remote = experiment["repo"].parent / "worker"
    for relative in metadata["source_files"]:
        target = remote / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(experiment["repo"] / relative, target)
    shutil.copytree(experiment["bundle"], remote, dirs_exist_ok=True)
    validate_bundle(remote, repo=remote, installed=True)
    monkeypatch.chdir(remote)
    cfg = yaml.safe_load((remote / metadata["runs"]["replicate_17"]["config"]).read_text())
    assert cfg["seed"] == 17
    fp_path = remote / metadata["fingerprint"]
    fp = read_object(fp_path)
    assert (remote / fp["source_lock_path"]).read_bytes() == (experiment["repo"] / fp["source_lock_path"]).read_bytes()
    verify_dataset_against_fingerprint(processed_features_dir=experiment["features"],
                                       processed_labels_dir=experiment["labels"],
                                       train_manifest=cfg["data"]["train_manifest"], val_manifest=cfg["data"]["val_manifest"],
                                       expected_fingerprint_path=fp_path, require_no_missing=True, track_id="limited")
    assert not any(path.suffix in {".npz", ".pt"} for path in experiment["bundle"].rglob("*"))
    (remote / "configs/profile.toml").write_text("changed profile")
    with pytest.raises(ValueError, match="source differs"):
        validate_bundle(remote, repo=remote, installed=True)


@pytest.mark.parametrize("fault", ["hidden-lock", "hidden-fingerprint", "missing", "overlap", "path-submission", "seed-bool"])
def test_bundle_rejects_unsafe_or_incomplete_experiments(experiment: dict[str, Any], fault: str) -> None:
    cfg = experiment["cfg"]
    fp_path = experiment["fingerprint"]
    fp = read_object(fp_path)
    if fault == "hidden-lock":
        lock = experiment["repo"] / fp["source_lock_path"]
        lock.write_text('{"hidden_manifest": "secret"}')
        fp["source_lock_sha256"] = sha256(lock)
    elif fault == "hidden-fingerprint":
        fp["split_names"] = ["hidden_val"]
    elif fault == "missing":
        fp["missing_feature_chain_count"] = 1
    elif fault == "overlap":
        path = experiment["repo"] / cfg["data"]["val_manifest"]
        path.write_text("1abc_A\n")
        fp["manifests"]["val"]["sha256"] = sha256(path)
    elif fault == "path-submission":
        cfg["submission"] = {"path": "submission.py"}
    else:
        cfg["seed"] = True
    experiment["config"].write_text(yaml.safe_dump(cfg))
    fp_path.write_text(json.dumps(fp))
    with pytest.raises(ValueError):
        _prepare(experiment)


@pytest.mark.parametrize("fault", ["source", "bundle", "extra", "rehashed-hidden", "rehashed-count"])
def test_bundle_revalidation_checks_content_and_semantics(experiment: dict[str, Any], fault: str) -> None:
    metadata = _prepare(experiment)
    root = experiment["bundle"]
    if fault == "source":
        (experiment["repo"] / "submissions/example/submission.py").write_text("changed")
    elif fault == "bundle":
        (root / metadata["runs"]["replicate_17"]["config"]).write_text("changed")
    elif fault == "extra":
        (root / "hidden-labels.json").write_text("[]")
    else:
        path = root / metadata["fingerprint"]
        fp = read_object(path)
        if fault == "rehashed-hidden":
            fp["manifests"]["hidden_val"] = {}
        else:
            fp["present_label_chain_count"] = 3
        path.write_text(json.dumps(fp))
        metadata["files"][metadata["fingerprint"]] = sha256(path)
        (root / BUNDLE_MANIFEST).write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        validate_bundle(root, repo=experiment["repo"])


@pytest.mark.parametrize("value", ["", ".", "../x", "/etc/passwd", "a/../x", "a//b", "a\\b"])
def test_bundle_paths_cannot_escape(value: str) -> None:
    with pytest.raises(ValueError):
        safe_relative(value)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "traversal", "duplicate"])
def test_archive_rejects_links_traversal_and_duplicate_entries_before_writing(tmp_path: Path, kind: str) -> None:
    archive = tmp_path / "data.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("../escape" if kind == "traversal" else "chain.npz")
        if kind in {"symlink", "hardlink"}:
            info.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
            info.linkname = "../escape"
        tar.addfile(info)
        if kind == "duplicate":
            tar.addfile(info)
    target = tmp_path / "stage"
    target.mkdir()
    with pytest.raises(ValueError):
        extract_regular_archive(archive, target)
    assert not list(target.iterdir())
    assert not (tmp_path / "escape").exists()


def test_archive_extracts_regular_content(tmp_path: Path) -> None:
    archive = tmp_path / "data.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("nested/chain.npz")
        info.size = 5
        tar.addfile(info, io.BytesIO(b"bytes"))
    target = tmp_path / "stage"
    target.mkdir()
    extract_regular_archive(archive, target)
    assert (target / "nested/chain.npz").read_bytes() == b"bytes"


def test_run_binding_rejects_model_runtime_or_config_drift(tmp_path: Path) -> None:
    path = tmp_path / "bindings/run.json"
    original = {"source_files": {"model.py": "abc"}, "runtime": {"torch": "2"}, "config": "def"}
    bind_execution(path, original)
    bind_execution(path, original)
    for key in original:
        with pytest.raises(ValueError, match="identity changed"):
            bind_execution(path, {**original, key: "changed"})
    assert read_object(path) == original


def test_supervisor_commits_checkpoint_before_process_exit_and_preserves_pause(tmp_path: Path) -> None:
    checkpoint, finished = tmp_path / "ckpt.pt", tmp_path / "finished"
    committed_before_exit = []
    command = [sys.executable, "-c", "from pathlib import Path; import time; "
               "Path('tmp').write_bytes(b'checkpoint'); Path('tmp').replace('ckpt.pt'); "
               "time.sleep(.3); Path('finished').touch(); raise SystemExit(75)"]
    code = supervise_training(command, cwd=tmp_path, env=os.environ, checkpoint=checkpoint,
                               commit=lambda: committed_before_exit.append(checkpoint.exists() and not finished.exists()),
                               timeout_seconds=3, poll_seconds=.02)
    assert code == 75
    assert any(committed_before_exit)


def test_supervisor_stops_process_on_watchdog_or_commit_failure(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    command = [sys.executable, "-c", "from pathlib import Path; import os,time; "
               "Path('pid').write_text(str(os.getpid())); Path('ckpt.pt').touch(); time.sleep(30)"]
    with pytest.raises(TimeoutError):
        supervise_training(command, cwd=tmp_path, env=os.environ, checkpoint=checkpoint,
                           commit=lambda: None, timeout_seconds=.3, poll_seconds=.02)
    with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / "pid").read_text()), 0)
    checkpoint.unlink()
    commits = []

    def fail_first_commit() -> None:
        commits.append(True)
        if len(commits) == 1:
            raise RuntimeError("storage failed")

    with pytest.raises(RuntimeError, match="storage failed"):
        supervise_training(command, cwd=tmp_path, env=os.environ, checkpoint=checkpoint,
                           commit=fail_first_commit, timeout_seconds=3, poll_seconds=.02)
    with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / "pid").read_text()), 0)


@pytest.fixture
def launcher(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Load actual launcher with a no-network Modal facade, not source assertions."""
    class ModalFacade:
        def __getattr__(self, name: str) -> Any:
            if name in {"function", "local_entrypoint"}:
                return lambda **kwargs: lambda fn: fn
            return lambda *args, **kwargs: self

    fake = ModalFacade()
    monkeypatch.setitem(sys.modules, "modal", types.SimpleNamespace(App=lambda *args: fake, Image=fake, Volume=fake, Dict=fake,
                                                                     current_function_call_id=lambda: "test-call"))
    monkeypatch.delenv("NANOFOLD_MODAL_TRAINING_BUNDLE", raising=False)
    script = Path(__file__).resolve().parents[1] / "scripts/modal_train.py"
    spec = importlib.util.spec_from_file_location("modal_train_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_candidate_seed_uses_fingerprint_and_official_keeps_policy(launcher: Any,
                                                                          experiment: dict[str, Any],
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = _prepare(experiment)
    captured = []
    monkeypatch.setattr(launcher, "BUNDLE", metadata)
    monkeypatch.setattr(launcher, "BUNDLE_PATH", str(experiment["bundle"]))
    monkeypatch.setattr(launcher, "REPO_ROOT", experiment["repo"])
    monkeypatch.setattr(launcher, "run_train", types.SimpleNamespace(remote=lambda config, **kwargs: captured.append((config, kwargs))))
    launcher.main(run_name="replicate_17")
    assert captured[0][0] == metadata["runs"]["replicate_17"]["config"]
    assert captured[0][1]["track"] == "limited"
    with pytest.raises(ValueError, match="Requested track"):
        launcher.main(run_name="replicate_17", track="unlimited", upload_data=True)
    assert len(captured) == 1


def test_launcher_upload_allowlist_excludes_unrelated_metadata_and_rejects_hidden_npz(launcher: Any,
                                                                                   experiment: dict[str, Any]) -> None:
    features = experiment["features"]
    manifests = [experiment["repo"] / experiment["cfg"]["data"][f"{role}_manifest"] for role in ("train", "val")]
    (features / "audit.json").write_text('{"private_notes": "not uploaded"}')
    selected = launcher._public_upload_files(features, manifests, features=True)
    assert len(selected) == 2 and all(path.suffix == ".npz" for path in selected)
    chain_npz_path(features, "3ghi_C").write_bytes(b"hidden")
    with pytest.raises(ValueError, match="exactly"):
        launcher._public_upload_files(features, manifests, features=True)


def test_launcher_refuses_warm_stage_and_unsafe_run_paths(launcher: Any, tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "stale.npz").touch()
    with pytest.raises(ValueError, match="fresh"):
        launcher._extract_archive_to_stage(archive_path=tmp_path / "missing.tar", target_dir=stage, label="features")
    with pytest.raises(ValueError, match="safe directory"):
        launcher._run_name({"run_name": "../other"})


def test_bundle_accepts_explicit_external_configs_without_copying_neighbor_files(experiment: dict[str, Any]) -> None:
    external = experiment["repo"].parent / "plan/config.yaml"
    external.parent.mkdir()
    shutil.copyfile(experiment["config"], external)
    external.with_name("hidden.json").write_text("never packaged")
    experiment["config"] = external
    metadata = _prepare(experiment)
    assert metadata["runs"]["replicate_17"]["seed"] == 17
    assert not any("hidden" in name for name in metadata["files"])


def test_launcher_official_path_still_enforces_official_policy(launcher: Any,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    captured = []
    monkeypatch.setattr(launcher, "run_train", types.SimpleNamespace(remote=lambda config, **kwargs: captured.append((config, kwargs))))
    launcher.main()
    assert captured[0][0] == "submissions/minalphafold2/config.yaml"
    assert captured[0][1]["track"] == "limited"


def test_remote_launcher_pauses_resumes_and_checks_runtime_before_continuation(launcher: Any,
                                                                              experiment: dict[str, Any],
                                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = _prepare(experiment)
    remote = experiment["repo"].parent / "worker"
    for relative in metadata["source_files"]:
        target = remote / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(experiment["repo"] / relative, target)
    shutil.copytree(experiment["bundle"], remote, dirs_exist_ok=True)
    monkeypatch.setattr(launcher, "REMOTE_ROOT", remote)
    monkeypatch.setattr(launcher, "BUNDLE", metadata)
    monkeypatch.setattr(launcher, "STAGE_ROOT", remote / "stage")
    monkeypatch.setattr(launcher, "runtime_record", lambda: {"torch": "pinned"})
    staged_paths = []

    def stage_data(stage: Path) -> tuple[Path, Path]:
        staged_paths.append(stage)
        shutil.copytree(experiment["features"], stage / "features")
        shutil.copytree(experiment["labels"], stage / "labels")
        return stage / "features", stage / "labels"

    monkeypatch.setattr(launcher, "_stage_public_data", stage_data)
    commands = []
    commits = []
    monkeypatch.setattr(launcher, "runs_volume", types.SimpleNamespace(commit=lambda: commits.append(True), reload=lambda: None))
    monkeypatch.setattr(launcher, "run_claims", Claims())
    monkeypatch.setattr(launcher, "FEATURES_MOUNT", experiment["features"])
    monkeypatch.setattr(launcher, "LABELS_MOUNT", experiment["labels"])

    def child(command: list[str], **kwargs: Any) -> int:
        commands.append(command)
        assert all(kwargs["env"].get(key) == value for key, value in CUDA_ENVIRONMENT.items())
        checkpoint = kwargs["checkpoint"]
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"closed checkpoint")
        kwargs["commit"]()
        return 75 if len(commands) == 1 else 0

    monkeypatch.setattr(launcher, "supervise_training", child)
    config = metadata["runs"]["replicate_17"]["config"]
    first = launcher.run_train(config, track="limited", auto_resume=True, requested_gpu="A10G")
    assert first["status"] == "paused" and first["checkpoint_committed"] is True
    second = launcher.run_train(config, track="limited", auto_resume=True, requested_gpu="A10G")
    assert second["status"] == "completed"
    assert "--resume" not in commands[0] and "--resume" in commands[1]
    assert "--max-runtime-seconds" in commands[0]
    assert "--official" not in commands[0] and "--verify-fingerprint" in commands[0]
    assert commands[0].count("--config") == 1
    assert staged_paths[0] != staged_paths[1] and not any(path.exists() for path in staged_paths)
    assert len(commits) >= 6
    result = read_object(remote / "runs/.modal_execution/replicate_17.result.json")
    assert result["status"] == "completed"
    monkeypatch.setattr(launcher, "runtime_record", lambda: {"torch": "changed"})
    with pytest.raises(ValueError, match="identity changed"):
        launcher.run_train(config, track="limited", auto_resume=True, requested_gpu="A10G")
    assert len(commands) == 2


class Claims:
    def __init__(self) -> None:
        self.entries: dict[str, Any] = {}

    def put(self, key: str, value: Any, *, skip_if_exists: bool) -> bool:
        assert skip_if_exists
        if key in self.entries:
            return False
        self.entries[key] = value
        return True

    def get(self, key: str) -> Any:
        return self.entries.get(key)

    def pop(self, key: str) -> Any:
        return self.entries.pop(key)


def test_exclusive_writer_blocks_duplicates_and_retains_crash_marker_after_claim_expiry(tmp_path: Path) -> None:
    claims = Claims()
    marker = tmp_path / "run.active.json"
    kwargs: dict[str, Any] = dict(claims=claims, key="volume/run", marker=marker, reload_volume=lambda: None,
                  commit=lambda: None, call_id="test-call")
    with exclusive_run(**kwargs):
        assert marker.is_file()
        with pytest.raises(RuntimeError, match="already has"):
            with exclusive_run(**kwargs):
                pytest.fail("A duplicate writer entered")
    assert not marker.exists() and not claims.entries
    with pytest.raises(RuntimeError, match="training failed"):
        with exclusive_run(**kwargs):
            raise RuntimeError("training failed")
    assert marker.exists() and claims.entries
    claims.entries.clear()  # Simulate Modal Dict expiration after seven days.
    with pytest.raises(RuntimeError, match="persistent active marker"):
        with exclusive_run(**kwargs):
            pytest.fail("An expired claim bypassed the persistent crash marker")


def test_mounted_archive_cannot_hide_extra_label_files(launcher: Any, experiment: dict[str, Any], tmp_path: Path) -> None:
    mount = tmp_path / "mounted-labels"
    (mount / "archives").mkdir(parents=True)
    (mount / "archives/processed_labels.tar").touch()
    manifests = [experiment["repo"] / experiment["cfg"]["data"][f"{role}_manifest"] for role in ("train", "val")]
    launcher._check_public_mount(mount, "processed_labels.tar", manifests, features=False)
    (mount / "stale-hidden-labels.npz").write_bytes(b"must never be accessible")
    with pytest.raises(ValueError, match="undeclared"):
        launcher._check_public_mount(mount, "processed_labels.tar", manifests, features=False)


def test_remote_launcher_has_no_arbitrary_trainer_argument_channel(launcher: Any) -> None:
    with pytest.raises(TypeError):
        launcher.run_train("configs/run.yaml", track="limited", auto_resume=True, requested_gpu="A10G",
                           argv=["--config", "other.yaml", "--allow-resume-mismatch"])
