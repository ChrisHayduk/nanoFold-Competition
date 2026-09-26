"""Local validation and process supervision for public-only Modal training."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import tarfile
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

CUDA_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NVIDIA_TF32_OVERRIDE": "0",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
}
BUNDLE_MANIFEST = "modal_training_bundle.json"
CONFIG_PREFIX = "modal_training_configs"
RUNTIME_PACKAGES = (
    "torch==2.14.0", "numpy==2.3.5", "tqdm==4.70.1", "pyyaml==6.0.3",
    "gemmi==0.7.5", "biopython==1.88",
)
SOURCE_ROOTS = ("nanofold", "scripts", "submissions", "configs", "tracks",
                "third_party/minAlphaFold2/minalphafold", "third_party/minAlphaFold2/configs")
SOURCE_ENTRYPOINTS = ("train.py", "eval.py", "predict.py", "score.py",
                      "third_party/minAlphaFold2/minalphafold/stereo_chemical_props.txt")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, "Duplicate JSON field")
        result[key] = value
    return result


def read_object(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(), object_pairs_hook=_object_pairs,
                     parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    _require(isinstance(raw, dict), "Expected a JSON object")
    return raw


def safe_relative(value: str) -> Path:
    path = PurePosixPath(value)
    _require(bool(value) and not path.is_absolute() and ".." not in path.parts
             and "\\" not in value and path.parts != () and str(path) == value, "Unsafe bundle path")
    return Path(*path.parts)


def source_records(repo: Path) -> dict[str, str]:
    paths = [repo / name for name in SOURCE_ENTRYPOINTS]
    for name in SOURCE_ROOTS:
        paths.extend(path for path in (repo / name).rglob("*")
                     if path.suffix in {".py", ".yaml", ".yml", ".toml"} and path.is_file())
    _require(all(path.absolute() == path.resolve() and path.resolve().is_relative_to(repo.resolve()) for path in paths),
             "Training source contains a linked or external file")
    return {path.relative_to(repo).as_posix(): sha256(path) for path in sorted(paths)}


def _public_source_lock(value: Any, prefix: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            nested = (*prefix, key)
            _require(not ("hidden" in key.lower() and nested != ("args", "hidden_val_size")),
                     "Training bundle source lock contains hidden metadata")
            _require(key not in {"data_source_lock", "data_source_lock_sha256"},
                     "Training bundle requires the public manifest source lock")
            _public_source_lock(child, nested)
    elif isinstance(value, list):
        for child in value:
            _public_source_lock(child, prefix)


def _repo_path(repo: Path, value: str | Path) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else repo / path
    resolved = path.resolve(strict=True)
    _require(resolved.is_relative_to(repo.resolve()) and resolved.is_file(), "Bundle input must be a repo file")
    _require(path.absolute() == resolved, "Bundle input must not use symbolic links")
    return resolved


def _manifest(path: Path) -> list[str]:
    chains = [line.strip() for line in path.read_text().splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
    _require(bool(chains) and len(chains) == len(set(chains)), "Manifest must be nonempty and unique")
    return chains


def prepare_bundle(*, repo: Path, configs: Sequence[Path], fingerprint: Path, output: Path) -> dict[str, Any]:
    """Create a fresh metadata-only bundle; never copy tensors or hidden assets."""
    repo = repo.resolve()
    _require(bool(configs) and not output.exists(), "Use a fresh bundle output and at least one config")
    fp_path = _repo_path(repo, fingerprint)
    fp = read_object(fp_path)
    _require(fp.get("split_names") == ["train", "val"] and set(fp.get("manifests", {})) == {"train", "val"}
             and fp.get("require_labels") is True, "Bundle requires a complete public train/val fingerprint")
    for name in ("missing_feature_chain_count", "missing_label_chain_count"):
        _require(type(fp.get(name)) is int and fp[name] == 0, "Incomplete public fingerprint")
    _require(isinstance(fp.get("source_lock_path"), str), "Public fingerprint must bind its source lock")
    lock_relative = safe_relative(fp["source_lock_path"])
    lock_path = _repo_path(repo, lock_relative)
    _require(sha256(lock_path) == fp.get("source_lock_sha256"), "Public source lock digest mismatch")
    _public_source_lock(read_object(lock_path))
    copied: dict[str, bytes] = {}

    def add(path: Path) -> str:
        relative = path.relative_to(repo).as_posix()
        data = path.read_bytes()
        _require(relative not in copied or copied[relative] == data, "Conflicting bundle input")
        copied[relative] = data
        return relative

    fp_relative = add(fp_path)
    add(lock_path)
    entries = {}
    for config in configs:
        # Configs may live in a separate local experiment-planning directory.
        # Only their validated YAML content is copied; referenced inputs and
        # executable modules must still belong to the repository inventory.
        config_path = Path(config)
        config_path = config_path if config_path.is_absolute() else repo / config_path
        _require(config_path.is_file() and config_path.absolute() == config_path.resolve(),
                 "Config must be a regular file without symbolic links")
        cfg = yaml.safe_load(config_path.read_text())
        _require(isinstance(cfg, dict) and isinstance(cfg.get("data"), dict), "Invalid training config")
        run_name = cfg.get("run_name")
        _require(isinstance(run_name, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name) is not None,
                 "run_name must be one safe directory name")
        _require(run_name not in entries, "Duplicate training run_name")
        _require(cfg.get("track") == fp.get("track_id"), "Config track differs from fingerprint")
        _require(type(cfg.get("seed")) is int and cfg["seed"] >= 0, "Config requires a nonnegative integer seed")
        submission = cfg.get("submission")
        _require(isinstance(submission, dict) and set(submission) == {"module"},
                 "Portable training configs must use submission.module, preserving repository adapter layout")
        assert isinstance(submission, dict)
        module = submission["module"]
        _require(isinstance(module, str) and all(part.isidentifier() for part in module.split("."))
                 and module.startswith(("nanofold.", "submissions.")), "Unsupported submission module")
        _repo_path(repo, Path(*module.split(".")).with_suffix(".py"))
        profile = cfg.get("model", {}).get("profile_path")
        if profile is not None:
            relative_profile = safe_relative(profile).as_posix()
            _require(relative_profile in source_records(repo), "Model profile must be a bundled repository source")
        members = {}
        for role in ("train", "val"):
            path = _repo_path(repo, cfg["data"][f"{role}_manifest"])
            chains = _manifest(path)
            record = fp["manifests"][role]
            _require(record.get("sha256") == sha256(path) and type(record.get("chain_count")) is int
                     and record["chain_count"] == len(chains), "Config manifest differs from public fingerprint")
            cfg["data"][f"{role}_manifest"] = add(path)
            members[role] = set(chains)
        _require(members["train"].isdisjoint(members["val"]), "Train and validation manifests overlap")
        count = len(members["train"]) + len(members["val"])
        for key in ("unique_chain_count", "present_feature_chain_count", "present_label_chain_count"):
            _require(type(fp.get(key)) is int and fp[key] == count, "Fingerprint population mismatch")
        # Runtime overrides point at the per-call staged data. Stable placeholders
        # keep local machine paths out of the frozen remote config bytes.
        cfg["data"]["processed_features_dir"] = "data/processed_features"
        cfg["data"]["processed_labels_dir"] = "data/processed_labels"
        config_relative = f"{CONFIG_PREFIX}/{run_name}.yaml"
        copied[config_relative] = yaml.safe_dump(cfg, sort_keys=True).encode()
        entries[run_name] = {"config": config_relative, "track": cfg["track"], "seed": cfg["seed"]}
    result = {"schema_version": 1, "artifact_type": "public_modal_training_bundle", "fingerprint": fp_relative,
              "runs": entries, "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(copied.items())},
              "source_files": source_records(repo), "runtime_packages": list(RUNTIME_PACKAGES), "python_version": "3.12"}
    output.mkdir(parents=True, mode=0o700)
    for name, data in copied.items():
        target = output / safe_relative(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (output / BUNDLE_MANIFEST).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    validate_bundle(output, repo=repo)
    return result


def validate_bundle(bundle: Path, *, repo: Path, installed: bool = False) -> dict[str, Any]:
    """Verify metadata and current executable sources without reading the corpus."""
    metadata = read_object(bundle / BUNDLE_MANIFEST)
    _require(set(metadata) == {"schema_version", "artifact_type", "fingerprint", "runs", "files", "source_files",
                              "runtime_packages", "python_version"}
             and type(metadata["schema_version"]) is int and metadata["schema_version"] == 1
             and metadata["artifact_type"] == "public_modal_training_bundle", "Unsupported training bundle")
    files = metadata["files"]
    _require(isinstance(files, dict) and 4 <= len(files) <= 1000, "Invalid bundle inventory")
    for name, digest in files.items():
        path = bundle / safe_relative(name)
        _require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(bundle.resolve())
                 and sha256(path) == digest, "Training bundle input changed")
    if not installed:
        observed = {path.relative_to(bundle).as_posix() for path in bundle.rglob("*") if path.is_file()}
        _require(observed == set(files) | {BUNDLE_MANIFEST} and not any(path.is_symlink() for path in bundle.rglob("*")),
                 "Training bundle has undeclared files or links")
    _require(metadata["source_files"] == source_records(repo), "Training source differs from the prepared bundle")
    _require(metadata["runtime_packages"] == list(RUNTIME_PACKAGES) and metadata["python_version"] == "3.12",
             "Training runtime differs from the prepared bundle")
    _require(metadata["fingerprint"] in files, "Unbound fingerprint")
    fp = read_object(bundle / safe_relative(metadata["fingerprint"]))
    _require(fp.get("split_names") == ["train", "val"] and set(fp.get("manifests", {})) == {"train", "val"}
             and fp.get("require_labels") is True, "Not a complete public fingerprint")
    for key in ("missing_feature_chain_count", "missing_label_chain_count"):
        _require(type(fp.get(key)) is int and fp[key] == 0, "Incomplete public fingerprint")
    lock = safe_relative(fp["source_lock_path"])
    _require(lock.as_posix() in files and sha256(bundle / lock) == fp["source_lock_sha256"], "Unbound source lock")
    _public_source_lock(read_object(bundle / lock))
    allowed = {metadata["fingerprint"], lock.as_posix()}
    _require(isinstance(metadata["runs"], dict) and bool(metadata["runs"]), "Bundle has no runs")
    for name, run in metadata["runs"].items():
        _require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is not None, "Unsafe run name")
        relative = f"{CONFIG_PREFIX}/{name}.yaml"
        _require(isinstance(run, dict) and set(run) == {"config", "seed", "track"}
                 and run.get("config") == relative and relative in files, "Unbound config")
        cfg = yaml.safe_load((bundle / relative).read_text())
        _require(cfg["run_name"] == name and cfg["seed"] == run["seed"] and cfg["track"] == run["track"] == fp["track_id"],
                 "Config identity mismatch")
        _require(type(cfg["seed"]) is int and cfg["seed"] >= 0, "Invalid training seed")
        submission = cfg.get("submission")
        _require(isinstance(submission, dict) and set(submission) == {"module"}, "Bundle requires submission.module")
        assert isinstance(submission, dict)
        module = submission["module"]
        _require(isinstance(module, str) and all(part.isidentifier() for part in module.split("."))
                 and module.startswith(("nanofold.", "submissions.")), "Unsupported submission module")
        _require(Path(*module.split(".")).with_suffix(".py").as_posix() in metadata["source_files"],
                 "Submission module is not bound to executable sources")
        profile = cfg.get("model", {}).get("profile_path")
        if profile is not None:
            _require(safe_relative(profile).as_posix() in metadata["source_files"], "Unbound model profile")
        allowed.add(relative)
        members = {}
        for role in ("train", "val"):
            relative = safe_relative(cfg["data"][f"{role}_manifest"]).as_posix()
            _require(relative in files and sha256(bundle / relative) == fp["manifests"][role]["sha256"],
                     "Unbound public manifest")
            members[role] = set(_manifest(bundle / relative))
            record = fp["manifests"][role]
            _require(type(record.get("chain_count")) is int and record["chain_count"] == len(members[role]),
                     "Fingerprint manifest count mismatch")
            allowed.add(relative)
        _require(members["train"].isdisjoint(members["val"]), "Train and validation manifests overlap")
        for key in ("unique_chain_count", "present_feature_chain_count", "present_label_chain_count"):
            _require(type(fp.get(key)) is int and fp[key] == sum(map(len, members.values())), "Fingerprint population mismatch")
    _require(set(files) == allowed, "Bundle contains files outside public training metadata")
    return metadata


def extract_regular_archive(archive: Path, target: Path) -> None:
    """Extract only unique regular files/directories; tar links are never followed."""
    root = target.resolve()
    seen = set()
    with tarfile.open(archive, "r") as handle:
        members = handle.getmembers()
        for member in members:
            relative = safe_relative(member.name.rstrip("/"))
            _require(member.isfile() or member.isdir(), "Archive links and special files are forbidden")
            _require(str(relative) not in seen, "Duplicate archive member")
            seen.add(str(relative))
            _require((root / relative).resolve().is_relative_to(root), "Archive path escapes stage")
        for member in members:
            path = root / safe_relative(member.name.rstrip("/"))
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                source = handle.extractfile(member)
                _require(source is not None, "Archive member has no data")
                assert source is not None
                with source, path.open("xb") as destination:
                    shutil.copyfileobj(source, destination)


def supervise_training(command: Sequence[str], *, cwd: Path, env: Mapping[str, str], checkpoint: Path,
                       commit: Callable[[], Any], timeout_seconds: float, poll_seconds: float = 15.0) -> int:
    """Commit closed atomic checkpoints while training; stop on durability failure."""
    _require(math.isfinite(timeout_seconds) and timeout_seconds > 0
             and math.isfinite(poll_seconds) and poll_seconds > 0, "Invalid supervisor deadline")
    process = subprocess.Popen(list(command), cwd=cwd, env=dict(env), start_new_session=True)
    started = time.monotonic()
    last = None

    def flush_checkpoint() -> None:
        nonlocal last
        if checkpoint.is_file():
            stat = checkpoint.stat()
            signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if signature != last:
                commit()
                last = signature

    try:
        while True:
            flush_checkpoint()
            remaining = timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Modal training watchdog expired; resume the last committed checkpoint")
            try:
                code = process.wait(timeout=min(poll_seconds, remaining))
                flush_checkpoint()
                return code
            except subprocess.TimeoutExpired:
                pass
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        commit()


def bind_execution(path: Path, record: Mapping[str, Any]) -> None:
    """A run name may resume only under its original code/data/runtime identity."""
    if path.exists():
        _require(read_object(path) == dict(record),
                 "Modal execution identity changed; choose a new run name and never resume this checkpoint")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation catches local duplicate launches; cross-container scheduling
    # still requires one writer per run because Volume commits are not atomic leases.
    with path.open("x") as handle:
        json.dump(dict(record), handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def runtime_record() -> dict[str, Any]:
    """Bind the resolved environment, including transitive scientific dependencies."""
    packages = {dist.metadata["Name"].lower(): dist.version for dist in importlib.metadata.distributions()
                if dist.metadata["Name"]}
    for requirement in RUNTIME_PACKAGES:
        name, version = requirement.split("==")
        _require(packages.get(name, "").split("+")[0] == version, "Installed training dependency differs from pinned image")
    _require(platform.python_version_tuple()[:2] == ("3", "12"), "Training image requires Python 3.12")
    _require(all(os.environ.get(key) == value for key, value in CUDA_ENVIRONMENT.items()),
             "CUDA determinism/precision environment differs from the pinned policy")
    import torch

    _require(torch.cuda.is_available(), "Modal training requires a working CUDA GPU")
    return {"python": platform.python_version(), "packages": dict(sorted(packages.items())),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "precision_policy": {"environment": dict(CUDA_ENVIRONMENT), "effective_tf32": False,
                                 "cuda_matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
                                 "cudnn_fp32_precision": torch.backends.cudnn.fp32_precision}}


@contextmanager
def exclusive_run(*, claims: Any, key: str, marker: Path, reload_volume: Callable[[], Any],
                  commit: Callable[[], Any], call_id: str) -> Iterator[None]:
    """Atomic claim plus a durable crash marker; failures require operator recovery."""
    token = uuid.uuid4().hex
    record = {"token": token, "call_id": call_id}
    if claims.put(key, record, skip_if_exists=True) is not True:
        raise RuntimeError("This run already has an active or interrupted Modal writer")
    # Modal Dict entries expire after inactivity. The committed marker outlives
    # that TTL, so a crashed run cannot silently acquire a new writer later.
    reload_volume()
    if marker.exists():
        raise RuntimeError("Interrupted Modal run requires operator recovery of its persistent active marker")
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open("x") as handle:
        json.dump(record, handle)
    commit()
    yield
    # Reach this only after training has stopped and its final result committed.
    # Keep both claims on any exception, including failed checkpoint commits.
    if claims.get(key) != record or read_object(marker) != record:
        raise RuntimeError("Modal run ownership changed during execution")
    marker.unlink()
    commit()
    claims.pop(key)
