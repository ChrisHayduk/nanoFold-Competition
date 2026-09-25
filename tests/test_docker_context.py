from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

from nanofold.chain_paths import chain_npz_path

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_ID = "sha256:" + "a" * 64
PUBLIC_SCORE_OUTPUTS = ("result.json", "eval_val.json", "per_chain_scores_val.jsonl")
HIDDEN_SCORE_OUTPUTS = ("eval_hidden.json", "per_chain_scores_hidden.jsonl")


class DockerHarness:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.log = root / "docker.jsonl"
        self.submission = root / "submission"
        self.submission.mkdir()
        self.config = self.submission / "config.yaml"
        self.write_config()
        (self.submission / "submission.py").write_text("# Participant-controlled code.\n")
        self.features = root / "public_features"
        self.labels = root / "public_labels"
        self.runs = root / "runs"
        self.private = root / "arbitrarily_named_private_assets"
        for path in (
            self.features,
            self.labels,
            self.runs,
            self.private / "manifests",
            self.private / "hidden_processed_features",
            self.private / "hidden_processed_labels",
            self.private / "leaderboard",
        ):
            path.mkdir(parents=True)
        (self.private / "manifests/hidden_val.txt").write_text("1abc_A\n")
        (self.private / "leaderboard/official_hidden_fingerprint.json").write_text("{}\n")
        (self.private / "leaderboard/private_hidden_assets.lock.json").write_text("{}\n")
        self.canary = root / "do_not_overwrite.txt"
        self.canary.write_text("maintainer data\n")
        docker = root / "docker"
        docker.write_text(
            f"#!{sys.executable}\n"
            + textwrap.dedent(
                """\
                import hashlib
                import json
                import os
                import sys
                from pathlib import Path

                argv = sys.argv[1:]
                event = {"argv": argv}
                if argv[0] == "build":
                    context = Path(argv[-1])
                    event["context_files"] = sorted(
                        str(path.relative_to(context)) for path in context.rglob("*") if path.is_file()
                    )
                    event["fingerprint_source_locks"] = {}
                    for fingerprint_path in sorted((context / "leaderboard").glob("*dataset_fingerprint.json")):
                        fingerprint = json.loads(fingerprint_path.read_text())
                        lock_path = (context / fingerprint["source_lock_path"]).resolve()
                        if not lock_path.is_relative_to(context.resolve()):
                            raise RuntimeError("Packaged fingerprint refers outside the image context")
                        lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
                        if lock_sha256 != fingerprint["source_lock_sha256"]:
                            raise RuntimeError("Packaged fingerprint source-lock hash does not match its content")
                        event["fingerprint_source_locks"][fingerprint_path.name] = {
                            "path": fingerprint["source_lock_path"],
                            "sha256": lock_sha256,
                        }
                    iid = argv[argv.index("--iidfile") + 1]
                    Path(iid).write_text(os.environ.get("DOCKER_TEST_IMAGE_ID", "sha256:" + "a" * 64) + "\\n")
                if argv[0] == "run" and "--score-hidden-only" in argv:
                    event["score_outputs"] = {}
                    for index, arg in enumerate(argv):
                        if arg in ("-v", "--volume"):
                            source, target, mode = argv[index + 1].rsplit(":", 2)
                            if target.startswith("/workspace/runs/test_run/") and mode == "rw":
                                output = Path(source)
                                event["score_outputs"][target] = {
                                    "is_file": output.is_file(),
                                    "is_symlink": output.is_symlink(),
                                    "size": output.stat().st_size,
                                }
                                if not os.environ.get("DOCKER_TEST_SKIP_SCORE_OUTPUT"):
                                    output.write_text('{}\\n')
                with Path(os.environ["DOCKER_LOG"]).open("a") as log:
                    log.write(json.dumps(event) + "\\n")
                if argv[0] == "run" and "--score-hidden-only" not in argv:
                    for index, arg in enumerate(argv):
                        if arg in ("-v", "--volume"):
                            source, target, mode = argv[index + 1].rsplit(":", 2)
                            if target == "/workspace/runs/test_run" and mode == "rw":
                                run_dir = Path(source)
                                (run_dir / "score.py").write_text("raise RuntimeError('untrusted scorer')\\n")
                                (run_dir / "nanofold.py").write_text("raise RuntimeError('untrusted package')\\n")
                                poison = os.environ.get("DOCKER_TEST_POISON_OUTPUT")
                                if poison:
                                    output = run_dir / poison
                                    if output.exists() or output.is_symlink():
                                        output.unlink()
                                    output.symlink_to(os.environ["DOCKER_TEST_CANARY"])
                """
            )
        )
        docker.chmod(0o755)
        self.env = dict(os.environ)
        for key in list(self.env):
            if key.startswith("NANOFOLD_") or key.startswith("DOCKER_TEST_"):
                self.env.pop(key)
        self.env.update(
            PATH=f"{root}:{Path(sys.executable).parent}:{self.env['PATH']}",
            DOCKER_LOG=str(self.log),
            DOCKER_TEST_CANARY=str(self.canary),
            IMAGE_NAME="nanofold-integration-test",
            NANOFOLD_TRUSTED_ROOT=str(REPO_ROOT),
            NANOFOLD_RUNS_ROOT=str(self.runs),
            NANOFOLD_PUBLIC_FEATURES_DIR=str(self.features),
            NANOFOLD_PUBLIC_LABELS_DIR=str(self.labels),
            NANOFOLD_PRIVATE_ROOT=str(self.private),
        )

    def write_config(self, **overrides: Any) -> None:
        config: dict[str, Any] = {
            "run_name": "test_run",
            "data": {
                "processed_features_dir": "data/processed_features",
                "processed_labels_dir": "data/processed_labels",
            },
        }
        config.update(overrides)
        self.config.write_text(yaml.safe_dump(config))

    def run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "/bin/bash",
                str(REPO_ROOT / "scripts/run_official_docker.sh"),
                "--submission",
                str(self.submission),
                "--track",
                "limited",
                *args,
            ],
            cwd=self.root,
            env=self.env,
            text=True,
            capture_output=True,
            check=False,
        )

    def events(self) -> list[dict[str, Any]]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def runs_logged(self) -> list[list[str]]:
        return [event["argv"] for event in self.events() if event["argv"][0] == "run"]


@pytest.fixture
def docker_harness(tmp_path: Path) -> DockerHarness:
    return DockerHarness(tmp_path)


def _mounts(argv: list[str]) -> dict[str, tuple[Path, str]]:
    result: dict[str, tuple[Path, str]] = {}
    for index, token in enumerate(argv):
        if token in ("-v", "--volume"):
            source, destination, mode = argv[index + 1].rsplit(":", 2)
            assert destination not in result, f"Duplicate mount destination: {destination}"
            result[destination] = (Path(source), mode)
    return result


def _assert_success(proc: subprocess.CompletedProcess[str]) -> None:
    assert proc.returncode == 0, proc.stdout + proc.stderr


def _make_trusted_repository(harness: DockerHarness) -> Path:
    root = harness.root / "trusted-source"
    root.mkdir()
    source_files = (
        "Dockerfile.official",
        "requirements.txt",
        "requirements-dev.txt",
        "train.py",
        "eval.py",
        "predict.py",
        "score.py",
        "nanofold/__init__.py",
        "scripts/run_official.py",
        "tracks/limited.yaml",
        "data/manifests/train.txt",
        "data/manifests/val.txt",
        "data/manifests/all.txt",
        "leaderboard/official_dataset_fingerprint.json",
        "leaderboard/research_large_dataset_fingerprint.json",
        "leaderboard/unlimited_dataset_fingerprint.json",
        "leaderboard/official_manifest_source.lock.json",
        "leaderboard/leaderboard.json",
        "README.md",
    )
    for name in source_files:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / name, destination)
    for name in (
        "unknown-private-location/secret-labels.npz",
        "unknown-private-location/hidden-manifest.txt",
        "submissions/attacker/score.py",
        "runs/attacker/nanofold.py",
    ):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("must never be part of the trusted runner image\n")
    for command in (["git", "init", "-q"], ["git", "add", "."]):
        subprocess.run(command, cwd=root, check=True, capture_output=True)
    (root / "nanofold/untracked.py").write_text("must not be copied from an unreviewed file\n")
    harness.env["NANOFOLD_TRUSTED_ROOT"] = str(root)
    return root


def test_official_docker_context_excludes_generated_data_and_private_assets() -> None:
    patterns = {
        line.strip()
        for line in (REPO_ROOT / ".dockerignore").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert {
        ".nanofold_private/",
        ".venv/",
        ".conda/",
        "data/openproteinset/",
        "data/processed_features/",
        "data/processed_labels/",
        "data/hidden_processed_features/",
        "data/hidden_processed_labels/",
        "runs/",
        "artifacts/",
        "*.npz",
        "*.pt",
        "*.ckpt",
    } <= patterns


def test_official_dockerfile_installs_requirements_before_source() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile.official").read_text()
    requirements = dockerfile.index("COPY requirements.txt requirements-dev.txt ./")
    install = dockerfile.index("RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt")
    source = next(dockerfile.index(line) for line in dockerfile.splitlines() if line.startswith("COPY ") and "score.py" in line)
    assert "COPY . ." not in dockerfile
    assert requirements < install < source
    assert "COPY leaderboard/official_manifest_source.lock.json ./leaderboard/" in dockerfile
    vendor_sources = [
        source
        for line in dockerfile.splitlines()
        if line.startswith("COPY ")
        for source in shlex.split(line)[1:-1]
        if source.startswith("third_party/")
    ]
    required_vendor_sources = {
        "third_party/minAlphaFold2/minalphafold",
        "third_party/minAlphaFold2/configs",
    }
    assert required_vendor_sources <= set(vendor_sources)
    assert set(vendor_sources) <= required_vendor_sources | {"third_party/minAlphaFold2/LICENSE"}


@pytest.mark.parametrize("hidden", [False, True])
def test_official_wrapper_separates_prediction_and_immutable_scoring(
    docker_harness: DockerHarness, hidden: bool
) -> None:
    harness = docker_harness
    _assert_success(harness.run(*([] if hidden else ["--disable-hidden"])))
    events = harness.events()
    build = [event for event in events if event["argv"][0] == "build"]
    assert len(build) == 1
    assert "--iidfile" in build[0]["argv"]
    assert Path(build[0]["argv"][-1]) != REPO_ROOT
    context_files = set(build[0]["context_files"])
    assert {"score.py", "scripts/run_official.py", "nanofold/__init__.py"} <= context_files
    assert "leaderboard/official_manifest_source.lock.json" in context_files
    locks = build[0]["fingerprint_source_locks"]
    assert set(locks) == {
        "official_dataset_fingerprint.json",
        "research_large_dataset_fingerprint.json",
        "unlimited_dataset_fingerprint.json",
    }
    assert all(record["path"] == "leaderboard/official_manifest_source.lock.json" for record in locks.values())
    assert not any(name.startswith(("submissions/", "runs/", ".nanofold_private/")) for name in context_files)
    vendor_prefix = "third_party/minAlphaFold2/"
    vendor_files = {name.removeprefix(vendor_prefix) for name in context_files if name.startswith(vendor_prefix)}
    if (REPO_ROOT / vendor_prefix / "minalphafold").is_dir():
        assert "minalphafold/model.py" in vendor_files
    assert all(name.startswith(("minalphafold/", "configs/", "LICENSE")) for name in vendor_files)
    assert not any(name.lower().endswith((".pdb", ".pdb.gz", ".cif", ".cif.gz", ".mmcif")) for name in vendor_files)

    runs = harness.runs_logged()
    assert len(runs) == (3 if hidden else 2)
    public_prediction, scoring = runs[0], runs[-1]
    predictions = runs[:-1]
    for args in runs:
        assert "--read-only" in args
        assert "--network=none" in args
        assert "--cap-drop=ALL" in args
        assert "--security-opt=no-new-privileges" in args
        assert args[args.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
        assert "--privileged" not in args
        assert IMAGE_ID in args
        assert "nanofold-integration-test" not in args
        mounts = _mounts(args)
        assert "/workspace" not in mounts
        assert not any(path in mounts for path in ("/workspace/scripts", "/workspace/nanofold", "/workspace/score.py"))
        assert mounts["/workspace/submissions/submission"] == (harness.submission, "ro")
        assert mounts["/workspace/data/processed_features"] == (harness.features, "ro")
    assert _mounts(public_prediction)["/workspace/data/processed_labels"] == (harness.labels, "ro")
    assert _mounts(scoring)["/workspace/data/processed_labels"] == (harness.labels, "ro")
    assert "--disable-hidden" in public_prediction
    assert ("--disable-hidden" in scoring) is not hidden
    for prediction in predictions:
        assert "--score-hidden-only" not in prediction
    assert "--score-hidden-only" in scoring
    assert "--skip-train" in scoring
    assert "--skip-hidden-scoring" not in scoring
    scoring_mounts = _mounts(scoring)
    run_target = "/workspace/runs/test_run"
    run_dir = harness.runs / "test_run"
    assert scoring_mounts[run_target] == (run_dir, "ro")
    for prediction in predictions:
        prediction_mounts = _mounts(prediction)
        assert prediction_mounts[run_target] == (run_dir, "rw")
        assert {target for target, (_, mode) in prediction_mounts.items() if mode == "rw"} == {run_target}
    output_names = PUBLIC_SCORE_OUTPUTS + (HIDDEN_SCORE_OUTPUTS if hidden else ())
    assert {target for target, (_, mode) in scoring_mounts.items() if mode == "rw"} == {
        f"{run_target}/{name}" for name in output_names
    }
    score_event = [event for event in events if event["argv"][0] == "run"][-1]
    for name in output_names:
        source, mode = scoring_mounts[f"{run_target}/{name}"]
        assert mode == "rw"
        assert source.parent != run_dir
        assert score_event["score_outputs"][f"{run_target}/{name}"] == {
            "is_file": True,
            "is_symlink": False,
            "size": 0,
        }
        assert (run_dir / name).read_text() == "{}\n"
    # Fake prediction plants importable code in its writable directory. Scoring
    # keeps that directory read-only, outside its immutable source location.
    assert (run_dir / "score.py").is_file()
    assert (run_dir / "nanofold.py").is_file()
    assert scoring[scoring.index("-w") + 1] == "/workspace"
    assert not any("PYTHONPATH=" in arg for arg in scoring)
    for target in ("/nanofold_hidden/labels", "/nanofold_hidden/private_hidden_assets.lock.json"):
        assert all(target not in _mounts(prediction) for prediction in predictions)
        assert (target in scoring_mounts) is hidden
        if hidden:
            assert scoring_mounts[target][1] == "ro"
    if hidden:
        hidden_prediction = runs[1]
        hidden_mounts = _mounts(hidden_prediction)
        assert "--predict-hidden-only" in hidden_prediction
        assert "--skip-train" in hidden_prediction
        assert "--skip-hidden-scoring" in hidden_prediction
        assert "/workspace/data/processed_labels" not in hidden_mounts
        for target in (
            "/nanofold_hidden/hidden_val.txt",
            "/nanofold_hidden/features",
            "/nanofold_hidden/official_hidden_fingerprint.json",
        ):
            assert target not in _mounts(public_prediction)
            assert hidden_mounts[target][1] == "ro"
            assert scoring_mounts[target][1] == "ro"


@pytest.mark.parametrize("name", PUBLIC_SCORE_OUTPUTS + HIDDEN_SCORE_OUTPUTS)
def test_prediction_cannot_redirect_scoring_outputs(docker_harness: DockerHarness, name: str) -> None:
    harness = docker_harness
    harness.env["DOCKER_TEST_POISON_OUTPUT"] = name
    proc = harness.run()
    assert proc.returncode != 0
    assert len(harness.runs_logged()) == 2
    assert harness.canary.read_text() == "maintainer data\n"


@pytest.mark.parametrize(
    "args",
    [
        ("--python", "/workspace/runs/test_run/evil-python"),
        ("--skip-hidden-scoring",),
        ("--score-hidden-only",),
        ("--disable-hidden", "--disable-hidden"),
        ("--leaderboard", "/tmp/custom-leaderboard.json"),
        ("--readme", "/tmp/custom-readme.md"),
        ("--hidden-manifest", "/tmp/hidden.txt"),
        ("--hidden-features-dir", "/tmp/hidden-features"),
        ("--hidden-labels-dir", "/tmp/hidden-labels"),
        ("--hidden-fingerprint", "/tmp/hidden.json"),
        ("--hidden-lock-file", "/tmp/hidden.lock.json"),
        ("--hidden-labels-dir=/tmp/hidden-labels",),
    ],
)
def test_official_wrapper_rejects_stage_and_path_overrides(
    docker_harness: DockerHarness, args: tuple[str, ...]
) -> None:
    proc = docker_harness.run(*args)
    assert proc.returncode != 0
    assert not docker_harness.runs_logged()


@pytest.mark.parametrize("run_name", ["../escape", "/tmp/escape", "with/slash", ".."])
def test_official_wrapper_rejects_unsafe_run_names(docker_harness: DockerHarness, run_name: str) -> None:
    docker_harness.write_config(run_name=run_name)
    proc = docker_harness.run("--disable-hidden")
    assert proc.returncode != 0
    assert not docker_harness.runs_logged()


@pytest.mark.parametrize("field", ["processed_features_dir", "processed_labels_dir"])
def test_official_wrapper_rejects_config_data_path_overrides(
    docker_harness: DockerHarness, field: str
) -> None:
    data = {
        "processed_features_dir": "data/processed_features",
        "processed_labels_dir": "data/processed_labels",
        field: "/nanofold_hidden/labels",
    }
    docker_harness.write_config(data=data)
    proc = docker_harness.run("--disable-hidden")
    assert proc.returncode != 0
    assert not docker_harness.runs_logged()


def test_official_wrapper_rejects_config_outside_submission(docker_harness: DockerHarness) -> None:
    outside = docker_harness.root / "outside.yaml"
    outside.write_text(docker_harness.config.read_text())
    proc = docker_harness.run("--disable-hidden", "--config", str(outside))
    assert proc.returncode != 0
    assert not docker_harness.runs_logged()


def test_official_wrapper_rejects_config_symlink_escape(docker_harness: DockerHarness) -> None:
    outside = docker_harness.root / "outside.yaml"
    outside.write_text(docker_harness.config.read_text())
    docker_harness.config.unlink()
    docker_harness.config.symlink_to(outside)
    proc = docker_harness.run("--disable-hidden")
    assert proc.returncode != 0
    assert not docker_harness.runs_logged()


def test_official_wrapper_requires_digest_image_id(docker_harness: DockerHarness) -> None:
    docker_harness.env["DOCKER_TEST_IMAGE_ID"] = "mutable-image:latest"
    proc = docker_harness.run("--disable-hidden")
    assert proc.returncode != 0
    assert not docker_harness.runs_logged()


def test_build_context_allows_only_tracked_runtime_source(docker_harness: DockerHarness) -> None:
    trusted_root = _make_trusted_repository(docker_harness)
    _assert_success(docker_harness.run("--disable-hidden"))
    build = next(event for event in docker_harness.events() if event["argv"][0] == "build")
    assert Path(build["argv"][-1]) != trusted_root
    context_files = set(build["context_files"])
    assert {"score.py", "nanofold/__init__.py", "tracks/limited.yaml"} <= context_files
    assert not any(name.startswith(("unknown-private-location/", "submissions/", "runs/", ".git/")) for name in context_files)
    assert "nanofold/untracked.py" not in context_files


def test_leaderboard_is_writable_only_by_trusted_scoring(docker_harness: DockerHarness) -> None:
    trusted_root = _make_trusted_repository(docker_harness)
    _assert_success(docker_harness.run("--update-leaderboard"))
    runs = docker_harness.runs_logged()
    assert len(runs) == 3
    for prediction in runs[:-1]:
        assert "--update-leaderboard" not in prediction
        mounts = _mounts(prediction)
        assert "/workspace/README.md" not in mounts
        assert "/workspace/leaderboard/leaderboard.json" not in mounts
    scoring = runs[-1]
    assert "--update-leaderboard" in scoring
    mounts = _mounts(scoring)
    assert mounts["/workspace/README.md"] == (trusted_root / "README.md", "rw")
    assert mounts["/workspace/leaderboard/leaderboard.json"] == (trusted_root / "leaderboard/leaderboard.json", "rw")


def test_scoring_must_produce_outputs_before_publishing(docker_harness: DockerHarness) -> None:
    harness = docker_harness
    run_dir = harness.runs / "test_run"
    run_dir.mkdir()
    (run_dir / "result.json").write_text('{"previous_result": true}\n')
    harness.env["DOCKER_TEST_SKIP_SCORE_OUTPUT"] = "1"
    proc = harness.run("--disable-hidden")
    assert proc.returncode != 0
    assert len(harness.runs_logged()) == 2
    assert (run_dir / "result.json").read_text() == '{"previous_result": true}\n'


def test_build_rejects_tracked_source_symlink(docker_harness: DockerHarness) -> None:
    trusted_root = _make_trusted_repository(docker_harness)
    (trusted_root / "score.py").unlink()
    (trusted_root / "score.py").symlink_to(docker_harness.canary)
    proc = docker_harness.run("--disable-hidden")
    assert proc.returncode != 0
    assert not docker_harness.events()


def test_wrapper_rejects_run_directory_symlink(docker_harness: DockerHarness) -> None:
    (docker_harness.runs / "test_run").symlink_to(docker_harness.submission, target_is_directory=True)
    proc = docker_harness.run("--disable-hidden")
    assert proc.returncode != 0
    assert not docker_harness.runs_logged()


def test_wrapper_rejects_hidden_labels_inside_visible_submission(docker_harness: DockerHarness) -> None:
    hidden_labels = docker_harness.submission / "labels"
    hidden_labels.mkdir()
    docker_harness.env["NANOFOLD_HIDDEN_LABELS_DIR"] = str(hidden_labels)
    proc = docker_harness.run()
    assert proc.returncode != 0
    assert not docker_harness.runs_logged()


def test_public_only_run_needs_no_hidden_assets(docker_harness: DockerHarness) -> None:
    shutil.rmtree(docker_harness.private)
    _assert_success(docker_harness.run("--disable-hidden"))
    runs = docker_harness.runs_logged()
    assert len(runs) == 2
    assert all(not any("/nanofold_hidden/" in arg for arg in args) for args in runs)


def test_wrapper_rejects_hidden_container_path_override(docker_harness: DockerHarness) -> None:
    docker_harness.env["NANOFOLD_HIDDEN_ROOT_CONTAINER"] = "/workspace/submissions/submission"
    proc = docker_harness.run()
    assert proc.returncode != 0
    assert not docker_harness.runs_logged()


def _processed_assets(harness: DockerHarness) -> dict[str, tuple[Path, str]]:
    public_chain = next(
        line.strip()
        for line in (REPO_ROOT / "data/manifests/all.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    return {
        "public_features": (harness.features, public_chain),
        "public_labels": (harness.labels, public_chain),
        "hidden_features": (harness.private / "hidden_processed_features", "1abc_A"),
        "hidden_labels": (harness.private / "hidden_processed_labels", "1abc_A"),
    }


@pytest.mark.parametrize("asset", ["public_features", "public_labels", "hidden_features", "hidden_labels"])
@pytest.mark.parametrize("injection", ["extra_npz", "nested_npz", "symlink", "python", "fifo"])
def test_wrapper_rejects_unapproved_processed_asset_contents(
    docker_harness: DockerHarness, asset: str, injection: str
) -> None:
    directory, chain_id = _processed_assets(docker_harness)[asset]
    expected_path = chain_npz_path(directory, chain_id)
    if injection == "extra_npz":
        chain_npz_path(directory, "not_a_manifest_chain").write_bytes(b"unapproved protein data")
    elif injection == "nested_npz":
        nested = directory / "additional_data"
        nested.mkdir()
        chain_npz_path(nested, chain_id).write_bytes(b"nested data")
    elif injection == "symlink":
        expected_path.symlink_to(docker_harness.canary)
    elif injection == "python":
        (directory / "sitecustomize.py").write_text("raise RuntimeError('untrusted code')\n")
    else:
        os.mkfifo(expected_path)
    proc = docker_harness.run()
    assert proc.returncode != 0
    assert not docker_harness.events(), "Unapproved contents must fail before Docker builds or runs"


@pytest.mark.parametrize("asset", ["public_labels", "hidden_labels"])
@pytest.mark.parametrize("sidecar", [
    "preprocess_meta.json", "msa_row_filter_audit.json",
    "msa_row_filter_audit.train.json", "msa_row_filter_audit.val.json",
])
def test_wrapper_does_not_allow_feature_sidecars_in_labels(
    docker_harness: DockerHarness, asset: str, sidecar: str
) -> None:
    directory, _ = _processed_assets(docker_harness)[asset]
    (directory / sidecar).write_text("{}\n")
    proc = docker_harness.run()
    assert proc.returncode != 0
    assert not docker_harness.events()


@pytest.mark.parametrize("sidecar", ["msa_row_filter_audit.train.json", "msa_row_filter_audit.val.json"])
def test_wrapper_rejects_public_filter_sidecars_in_hidden_features(
    docker_harness: DockerHarness, sidecar: str,
) -> None:
    directory, _ = _processed_assets(docker_harness)["hidden_features"]
    (directory / sidecar).write_text("{}\n")
    proc = docker_harness.run()
    assert proc.returncode != 0
    assert not docker_harness.events()


def test_wrapper_accepts_manifest_files_and_feature_sidecars(docker_harness: DockerHarness) -> None:
    for asset, (directory, chain_id) in _processed_assets(docker_harness).items():
        # The wrapper validates the mount contents; the container's fingerprint
        # checks remain responsible for NPZ content and completeness.
        chain_npz_path(directory, chain_id).touch()
        if asset.endswith("features"):
            for sidecar in ("preprocess_meta.json", "msa_row_filter_audit.json"):
                (directory / sidecar).write_text("{}\n")
        if asset == "public_features":
            for sidecar in ("msa_row_filter_audit.train.json", "msa_row_filter_audit.val.json"):
                (directory / sidecar).write_text("{}\n")
    _assert_success(docker_harness.run())
    assert len(docker_harness.runs_logged()) == 3


@pytest.mark.parametrize("absolute", [False, True])
def test_wrapper_rejects_private_source_lock_in_hidden_fingerprint(
    docker_harness: DockerHarness, absolute: bool
) -> None:
    private_reference = "leaderboard/private_hidden_assets.lock.json"
    if absolute:
        private_reference = str(docker_harness.private / private_reference)
    fingerprint = docker_harness.private / "leaderboard/official_hidden_fingerprint.json"
    fingerprint.write_text(json.dumps({"source_lock_path": private_reference}))
    proc = docker_harness.run()
    assert proc.returncode != 0
    assert not docker_harness.events()
