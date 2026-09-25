#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Isolated host Python parses YAML as data; submission code is never imported.
python -I - "$REPO_ROOT" "$@" <<'PY'
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

repo = Path(sys.argv[1])
arguments = sys.argv[2:]
if os.environ.get("NANOFOLD_HIDDEN_ROOT_CONTAINER", "/nanofold_hidden") != "/nanofold_hidden":
    raise SystemExit("ERROR: hidden container paths are fixed; configure host paths with NANOFOLD_HIDDEN_* variables")
if arguments in (["--help"], ["-h"]):
    print("""Usage: run_official_docker.sh --submission DIR [official runner options]

Invoke this wrapper from a trusted checkout, never a submission branch.
NANOFOLD_TRUSTED_ROOT selects trusted runner source (defaults to this checkout).
The image contains only selected tracked source, and all stages pin its image ID.
Training sees public assets; hidden prediction sees hidden features only; a fresh
container scores read-only predictions with hidden labels and fresh output files.

Host locations: NANOFOLD_PUBLIC_FEATURES_DIR, NANOFOLD_PUBLIC_LABELS_DIR,
NANOFOLD_RUNS_ROOT, NANOFOLD_PRIVATE_ROOT and NANOFOLD_HIDDEN_* variables.
Config must remain inside the submission directory and use data/processed_features
and data/processed_labels. Stage flags, Python and leaderboard path overrides are
not accepted. --update-leaderboard writes trusted README/leaderboard files only
from the scoring container, after all submission processes have exited.""")
    raise SystemExit(0)

value_flags = {"--submission", "--config", "--track", "--commit", "--description", "--team", "--checkpoint-steps"}
boolean_flags = {"--skip-train", "--disable-hidden", "--update-leaderboard"}
options = {}
while arguments:
    name, separator, value = arguments.pop(0).partition("=")
    if name in options or name not in value_flags | boolean_flags:
        raise SystemExit(f"ERROR: unsupported or repeated wrapper option: {name}")
    if name in boolean_flags:
        if separator:
            raise SystemExit(f"ERROR: {name} does not accept a value")
        options[name] = True
    else:
        if not separator:
            if not arguments or arguments[0].startswith("--"):
                raise SystemExit(f"ERROR: missing value for {name}")
            value = arguments.pop(0)
        options[name] = value
if not options.get("--submission"):
    raise SystemExit("ERROR: --submission is required")


def path(value):
    result = Path(value).expanduser().resolve()
    if any(char in str(result) for char in "\n\r\x00,:"):
        raise SystemExit("ERROR: mount paths cannot contain newlines, commas or colons")
    return result


def require(value, *, directory=False):
    result = path(value)
    if not (result.is_dir() if directory else result.is_file()):
        raise SystemExit(f"ERROR: required {'directory' if directory else 'file'} is missing: {result}")
    return result


def within(child, parent):
    return child == parent or parent in child.parents


def overlaps(first, second):
    return within(first, second) or within(second, first)


def processed_directory(root, manifest, *, feature_sidecars=False, public_filter_sidecars=False):
    chain_ids = {
        line.strip() for line in manifest.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    allowed = {"chain_" + chain_id.encode("utf-8").hex() + ".npz" for chain_id in chain_ids}
    if feature_sidecars:
        allowed.update({"preprocess_meta.json", "msa_row_filter_audit.json"})
    if public_filter_sidecars:
        allowed.update({"msa_row_filter_audit.train.json", "msa_row_filter_audit.val.json"})
    for entry in root.iterdir():
        if entry.name not in allowed or not stat.S_ISREG(entry.lstat().st_mode):
            raise SystemExit("ERROR: processed asset directory contains files outside its manifest or unexpected subdirectories/links")


trusted = require(os.environ.get("NANOFOLD_TRUSTED_ROOT", str(repo)), directory=True)
submission = require(options["--submission"], directory=True)
safe_name = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
if not safe_name.fullmatch(submission.name):
    raise SystemExit("ERROR: submission directory must have a simple name")
config = require(options.get("--config", str(submission / "config.yaml")))
if not within(config, submission):
    raise SystemExit("ERROR: config must resolve inside the submission directory")
cfg = yaml.safe_load(config.read_text())
if not isinstance(cfg, dict) or not isinstance(cfg.get("data"), dict):
    raise SystemExit("ERROR: config must contain a data mapping")
run_name = cfg.get("run_name")
if not isinstance(run_name, str) or not safe_name.fullmatch(run_name):
    raise SystemExit("ERROR: run_name must be a simple filename without path separators")
for key, expected in (("processed_features_dir", "data/processed_features"), ("processed_labels_dir", "data/processed_labels")):
    if cfg["data"].get(key) != expected:
        raise SystemExit(f"ERROR: config data.{key} must be {expected}; relocate host data with NANOFOLD_PUBLIC_*_DIR")
features = require(os.environ.get("NANOFOLD_PUBLIC_FEATURES_DIR", str(trusted / "data/processed_features")), directory=True)
labels = require(os.environ.get("NANOFOLD_PUBLIC_LABELS_DIR", str(trusted / "data/processed_labels")), directory=True)
if overlaps(features, labels):
    raise SystemExit("ERROR: public features and labels must be separate directories")
public_manifest = require(trusted / "data/manifests/all.txt")
processed_directory(features, public_manifest, feature_sidecars=True, public_filter_sidecars=True)
processed_directory(labels, public_manifest)
runs_argument = Path(os.environ.get("NANOFOLD_RUNS_ROOT", str(trusted / "runs"))).expanduser()
if runs_argument.is_symlink():
    raise SystemExit("ERROR: runs root cannot be a symlink")
runs = path(runs_argument)
run = runs / run_name
if run.is_symlink() or (run.exists() and not run.is_dir()):
    raise SystemExit("ERROR: run output must be a directory, not a symlink or file")
if within(run, trusted) and not within(run, trusted / "runs"):
    raise SystemExit("ERROR: run outputs cannot overlap trusted source files")
for source in (submission, features, labels, trusted):
    if within(source, run) or (source != trusted and within(run, source)):
        raise SystemExit("ERROR: run outputs overlap an input directory")

private = path(os.environ.get("NANOFOLD_PRIVATE_ROOT", str(trusted / ".nanofold_private")))
defaults = {
    "MANIFEST": private / "manifests/hidden_val.txt",
    "FEATURES_DIR": private / "hidden_processed_features",
    "LABELS_DIR": private / "hidden_processed_labels",
    "FINGERPRINT": private / "leaderboard/official_hidden_fingerprint.json",
    "LOCK_FILE": private / "leaderboard/private_hidden_assets.lock.json",
}
hidden = {}
if not options.get("--disable-hidden"):
    for key, default in defaults.items():
        hidden[key] = require(os.environ.get("NANOFOLD_HIDDEN_" + key, str(default)), directory=key.endswith("_DIR"))
    if overlaps(hidden["FEATURES_DIR"], hidden["LABELS_DIR"]):
        raise SystemExit("ERROR: hidden features and labels must be separate directories")
    processed_directory(hidden["FEATURES_DIR"], hidden["MANIFEST"], feature_sidecars=True)
    processed_directory(hidden["LABELS_DIR"], hidden["MANIFEST"])
    for visible in (submission, features, labels, run, hidden["FEATURES_DIR"]):
        for secret in (hidden["LABELS_DIR"], hidden["LOCK_FILE"]):
            if overlaps(visible, secret):
                raise SystemExit("ERROR: prediction-visible mount overlaps hidden labels or the private lock")
    fingerprint = json.loads(hidden["FINGERPRINT"].read_text())
    if not isinstance(fingerprint, dict) or fingerprint.get("source_lock_path") not in (
        None, "", "leaderboard/official_manifest_source.lock.json",
    ):
        raise SystemExit("ERROR: hidden fingerprint must reference the packaged public manifest source lock")
track = str(options.get("--track", "limited"))
if not safe_name.fullmatch(track) or not (trusted / "tracks" / (track + ".yaml")).is_file():
    raise SystemExit("ERROR: track must name a trusted track definition")

submission_container = "/workspace/submissions/" + submission.name
run_container = "/workspace/runs/" + run_name
runner_args = ["--submission", submission_container, "--config", submission_container + "/" + str(config.relative_to(submission))]
for key, value in options.items():
    if key in {"--submission", "--config", "--update-leaderboard", "--skip-train", "--disable-hidden"}:
        continue
    runner_args.append(key)
    if value is not True:
        runner_args.append(str(value))
if "--commit" not in options:
    commit = subprocess.run(["git", "-C", str(submission), "rev-parse", "HEAD"], text=True, capture_output=True)
    if commit.returncode == 0:
        runner_args.extend(["--commit", commit.stdout.strip()])


def mount(source, destination, *, writable=False):
    return ["-v", f"{source}:{destination}:{'rw' if writable else 'ro'}"]


def command(args):
    print("+ " + " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)


with tempfile.TemporaryDirectory(prefix="nanofold-sealed-") as temporary:
    staging = Path(temporary)
    context = staging / "context"
    context.mkdir()
    required = ["Dockerfile.official", "requirements.txt", "requirements-dev.txt", "train.py", "eval.py", "predict.py", "score.py"]
    selected = required + ["nanofold", "scripts", "tracks"]
    selected += ["data/manifests/" + name + ".txt" for name in ("train", "val", "all")]
    selected += ["leaderboard/" + name for name in (
        "official_dataset_fingerprint.json", "research_large_dataset_fingerprint.json", "unlimited_dataset_fingerprint.json",
        "official_manifest_source.lock.json",
    )]
    tracked = subprocess.check_output(["git", "-C", str(trusted), "ls-files", "-z", "--", *selected]).split(b"\0")

    def copy_tracked(source_root, names, prefix=Path()):
        for raw in names:
            if not raw:
                continue
            relative = Path(os.fsdecode(raw))
            source = source_root / relative
            if not within(source.resolve(), source_root) or source.is_symlink():
                raise SystemExit(f"ERROR: trusted source has a symlink or escaping path: {relative}")
            if not source.is_file():
                raise SystemExit(f"ERROR: tracked source file is missing: {relative}")
            target = context / prefix / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    copy_tracked(trusted, tracked)
    for name in required:
        if not (context / name).is_file():
            raise SystemExit(f"ERROR: required runner source is not tracked: {name}")
    vendor = trusted / "third_party/minAlphaFold2"
    for directory in ("minalphafold", "configs"):
        (context / "third_party/minAlphaFold2" / directory).mkdir(parents=True)
    if (vendor / "minalphafold").is_dir():
        names = subprocess.check_output([
            "git", "-C", str(vendor), "ls-files", "-z", "--", "minalphafold", "configs", "LICENSE*",
        ]).split(b"\0")
        for raw in names:
            relative = Path(os.fsdecode(raw))
            if raw and not (
                (relative.parts[0] == "minalphafold" and relative.suffix == ".py")
                or (relative.parts[0] == "configs" and relative.suffix == ".toml")
                or str(relative) == "minalphafold/stereo_chemical_props.txt"
                or (len(relative.parts) == 1 and relative.name.startswith("LICENSE"))
            ):
                raise SystemExit("ERROR: vendored runner source contains an unexpected non-code asset")
        copy_tracked(vendor, names, Path("third_party/minAlphaFold2"))
        if not (context / "third_party/minAlphaFold2/LICENSE").is_file():
            raise SystemExit("ERROR: minAlphaFold2 source is missing its tracked license")
    else:
        (context / "third_party/minAlphaFold2/LICENSE").write_text("minAlphaFold2 source is not included in this image.\n")
    image_file = staging / "image.id"
    command(["docker", "build", "-f", context / "Dockerfile.official", "--iidfile", image_file,
             "-t", os.environ.get("IMAGE_NAME", "nanofold-official-runner"), context])
    image_id = image_file.read_text().strip()
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", image_id):
        raise SystemExit("ERROR: Docker did not return an immutable image ID")

    run.mkdir(parents=True, exist_ok=True)
    forbidden = run / "_forbidden_eval_labels_mount"
    if forbidden.is_symlink():
        raise SystemExit("ERROR: forbidden-label marker cannot be a symlink")
    forbidden.mkdir(exist_ok=True)
    common = ["docker", "run", "--rm", "--network=none", "--cap-drop=ALL", "--read-only", "--security-opt=no-new-privileges",
              "--user", f"{os.getuid()}:{os.getgid()}", "--tmpfs", "/tmp:rw,nosuid,nodev",
              "-e", "NANOFOLD_OFFICIAL_SEALED_RUNTIME=1", "-w", "/workspace"]
    common += mount(submission, submission_container)
    public_features_mount = mount(features, "/workspace/data/processed_features")
    public_labels_mount = mount(labels, "/workspace/data/processed_labels")
    public_args = runner_args + ["--disable-hidden"]
    if options.get("--skip-train"):
        public_args.append("--skip-train")
    command(common + public_features_mount + public_labels_mount + mount(run, run_container, writable=True) + [image_id] + public_args)

    hidden_common = []
    if hidden:
        for key, destination in (("MANIFEST", "hidden_val.txt"), ("FEATURES_DIR", "features"), ("FINGERPRINT", "official_hidden_fingerprint.json")):
            container_path = "/nanofold_hidden/" + destination
            hidden_common += ["-e", "NANOFOLD_HIDDEN_" + key + "=" + container_path]
            hidden_common += mount(hidden[key], container_path)
        command(common + public_features_mount + hidden_common + mount(run, run_container, writable=True) + [image_id] + runner_args
                + ["--predict-hidden-only", "--skip-train", "--skip-hidden-scoring"])

    # No submission process survives into scoring. Refuse links/special files
    # before exposing artifacts to the scorer; outputs live on separate mounts.
    if run.is_symlink() or not run.is_dir():
        raise SystemExit("ERROR: prediction replaced the run directory")
    for root, directories, files in os.walk(run, followlinks=False):
        for name in directories + files:
            mode = (Path(root) / name).lstat().st_mode
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise SystemExit("ERROR: prediction artifacts cannot contain symlinks or special files")
    score_outputs = staging / "score_outputs"
    score_outputs.mkdir()
    score_args = common + public_features_mount + public_labels_mount + hidden_common + mount(run, run_container)
    output_names = ["result.json", "eval_val.json", "per_chain_scores_val.jsonl"]
    if hidden:
        output_names += ["eval_hidden.json", "per_chain_scores_hidden.jsonl"]
    for name in output_names:
        target = run / name
        if target.exists() and not target.is_file():
            raise SystemExit("ERROR: scorer output path is not a regular file")
        target.touch(exist_ok=True)
        (score_outputs / name).touch()
        score_args += mount(score_outputs / name, run_container + "/" + name, writable=True)
    if hidden:
        for key, destination in (("LABELS_DIR", "labels"), ("LOCK_FILE", "private_hidden_assets.lock.json")):
            container_path = "/nanofold_hidden/" + destination
            score_args += ["-e", "NANOFOLD_HIDDEN_" + key + "=" + container_path]
            score_args += mount(hidden[key], container_path)
    final_args = runner_args + ["--score-hidden-only", "--skip-train"]
    if not hidden:
        final_args.append("--disable-hidden")
    if options.get("--update-leaderboard"):
        for relative in ("README.md", "leaderboard/leaderboard.json"):
            source = trusted / relative
            if source.is_symlink() or not source.is_file():
                raise SystemExit("ERROR: leaderboard outputs must be trusted regular files")
            score_args += mount(source, "/workspace/" + relative, writable=True)
        final_args.append("--update-leaderboard")
    command(score_args + [image_id] + final_args)
    if any(not (score_outputs / name).stat().st_size for name in output_names):
        raise SystemExit("ERROR: scoring did not produce all required output files")
    # Publish the final result last, after all required score files are present.
    for name in [name for name in output_names if name != "result.json"] + ["result.json"]:
        source = score_outputs / name
        fd, temporary_file = tempfile.mkstemp(prefix=".scored-", dir=run)
        with os.fdopen(fd, "wb") as target, source.open("rb") as data:
            shutil.copyfileobj(data, target)
        os.replace(temporary_file, run / source.name)
PY
