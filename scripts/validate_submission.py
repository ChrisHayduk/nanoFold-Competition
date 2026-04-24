from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml

# Allow running as `python scripts/validate_submission.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofold.competition_policy import (
    DEFAULT_TRACK_ID,
    TrackSpec,
    apply_track_policy,
    compute_effective_batch_size,
    compute_residue_budget,
    compute_sample_budget,
    enforce_model_param_limit,
    load_track_spec,
    validate_track_policy,
)
from nanofold.submission_runtime import load_submission_hooks, run_submission_batch, strip_supervision_from_batch
from nanofold.utils import count_parameters

REQUIRED_TOP_LEVEL = ("run_name", "seed", "submission", "data", "train")
REQUIRED_DATA_KEYS = (
    "processed_features_dir",
    "processed_labels_dir",
    "train_manifest",
    "val_manifest",
    "crop_size",
    "msa_depth",
    "batch_size",
)
REQUIRED_TRAIN_KEYS = ("max_steps", "eval_every", "save_every")

METADATA_FIELDS = (
    "max_steps",
    "effective_batch_size",
    "sample_budget",
    "residue_budget",
    "crop_size",
    "seed",
    "hardware",
    "wall_clock_time",
    "commit",
)

MAX_ALLOWED_FILE_SIZE_BYTES = 20 * 1024 * 1024
FORBIDDEN_WEIGHT_EXTENSIONS = {".pt", ".pth", ".ckpt", ".safetensors", ".npz"}
SUSPICIOUS_IMPORTS = {
    "requests",
    "urllib",
    "urllib.request",
    "torch.hub",
    "huggingface_hub",
}


@dataclass
class Diagnostic:
    level: str  # "error" | "warning"
    message: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate a competition submission folder.")
    ap.add_argument(
        "--submission",
        type=str,
        required=True,
        help="Path to submission directory, e.g. submissions/alice",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures (non-zero exit).",
    )
    ap.add_argument(
        "--track",
        type=str,
        default=DEFAULT_TRACK_ID,
        help=f"Track id to validate against (default: {DEFAULT_TRACK_ID})",
    )
    return ap.parse_args()


def add_error(diags: List[Diagnostic], message: str) -> None:
    diags.append(Diagnostic(level="error", message=message))


def add_warning(diags: List[Diagnostic], message: str) -> None:
    diags.append(Diagnostic(level="warning", message=message))


def _require_keys(diags: List[Diagnostic], mapping: Dict[str, Any], keys: tuple[str, ...], scope: str) -> None:
    for key in keys:
        if key not in mapping:
            if scope:
                add_error(diags, f"Missing key `{scope}.{key}`")
            else:
                add_error(diags, f"Missing key `{key}`")


def _validate_numeric(diags: List[Diagnostic], cfg: Dict[str, Any]) -> Tuple[int, int, int]:
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    def require_int(path: str, value: Any, min_value: int = 0, strictly_positive: bool = False) -> int:
        if not isinstance(value, int):
            add_error(diags, f"`{path}` must be an integer (got {type(value).__name__})")
            return 0
        if strictly_positive and value <= min_value:
            add_error(diags, f"`{path}` must be > {min_value}")
        elif not strictly_positive and value < min_value:
            add_error(diags, f"`{path}` must be >= {min_value}")
        return value

    require_int("seed", cfg["seed"], min_value=0, strictly_positive=False)

    crop_size = require_int("data.crop_size", data_cfg["crop_size"], min_value=0, strictly_positive=True)
    msa_depth = require_int("data.msa_depth", data_cfg["msa_depth"], min_value=0, strictly_positive=True)
    batch_size = require_int("data.batch_size", data_cfg["batch_size"], min_value=0, strictly_positive=True)
    if "num_workers" in data_cfg:
        require_int("data.num_workers", data_cfg["num_workers"], min_value=0, strictly_positive=False)

    max_steps = require_int("train.max_steps", train_cfg["max_steps"], min_value=0, strictly_positive=True)
    eval_every = require_int("train.eval_every", train_cfg["eval_every"], min_value=0, strictly_positive=True)
    save_every = require_int("train.save_every", train_cfg["save_every"], min_value=0, strictly_positive=True)
    if "log_every" in train_cfg:
        require_int("train.log_every", train_cfg["log_every"], min_value=0, strictly_positive=True)
    if eval_every > max_steps:
        add_warning(diags, "`train.eval_every` is greater than `train.max_steps`; no intermediate evals will run.")
    if save_every > max_steps:
        add_warning(diags, "`train.save_every` is greater than `train.max_steps`; no intermediate checkpoints will be saved.")

    grad_accum_steps = train_cfg.get("grad_accum_steps", 1)
    grad_accum_steps = require_int("train.grad_accum_steps", grad_accum_steps, min_value=0, strictly_positive=True)
    effective_batch_size = compute_effective_batch_size(batch_size=batch_size, grad_accum_steps=grad_accum_steps)
    sample_budget = compute_sample_budget(
        max_steps=max_steps,
        effective_batch_size=effective_batch_size,
    )
    residue_budget = compute_residue_budget(
        max_steps=max_steps,
        effective_batch_size=effective_batch_size,
        crop_size=crop_size,
    )
    if crop_size <= 0 or msa_depth <= 0:
        return 0, 0, 0
    return effective_batch_size, sample_budget, residue_budget


def _validate_submission_entrypoint(
    diags: List[Diagnostic],
    cfg: Dict[str, Any],
    config_path: Path,
    submission_dir: Path,
    allow_template_fields: bool,
) -> None:
    sub_cfg = cfg.get("submission")
    if not isinstance(sub_cfg, dict):
        add_error(diags, "`submission` must be a mapping with exactly one of `path` or `module`.")
        return

    has_path = isinstance(sub_cfg.get("path"), str) and sub_cfg.get("path", "").strip() != ""
    has_module = isinstance(sub_cfg.get("module"), str) and sub_cfg.get("module", "").strip() != ""

    if has_path == has_module:
        add_error(diags, "`submission` must set exactly one of `path` or `module`.")
        return

    if not allow_template_fields and not has_path:
        add_error(diags, "Submission must set `submission.path` to a local file under the submission directory.")
        return

    if has_path:
        entry_path = Path(sub_cfg["path"].strip())
        if not entry_path.is_absolute():
            entry_path = config_path.parent / entry_path
        entry_path = entry_path.resolve()
        if not entry_path.exists():
            add_error(diags, f"Submission entrypoint file does not exist: {entry_path}")
            return
        if not entry_path.is_file():
            add_error(diags, f"Submission entrypoint is not a file: {entry_path}")
            return
        if not allow_template_fields and submission_dir not in entry_path.parents:
            add_error(diags, f"`submission.path` must resolve inside `{submission_dir}` (got `{entry_path}`).")


def _make_synthetic_batch(crop_size: int, msa_depth: int) -> Dict[str, Any]:
    B = 1
    L = min(crop_size, 32)
    N = min(msa_depth, 8)
    T = 1
    return {
        "chain_id": ["SYNTHETIC_A"],
        "aatype": torch.randint(low=0, high=21, size=(B, L), dtype=torch.long),
        "msa": torch.randint(low=0, high=23, size=(B, N, L), dtype=torch.long),
        "deletions": torch.zeros((B, N, L), dtype=torch.long),
        "template_aatype": torch.randint(low=0, high=21, size=(B, T, L), dtype=torch.long),
        "template_ca_coords": torch.randn(B, T, L, 3, dtype=torch.float32),
        "template_ca_mask": torch.ones(B, T, L, dtype=torch.bool),
        "ca_coords": torch.randn(B, L, 3, dtype=torch.float32),
        "ca_mask": torch.ones(B, L, dtype=torch.bool),
        "atom14_positions": torch.randn(B, L, 14, 3, dtype=torch.float32),
        "atom14_mask": torch.ones(B, L, 14, dtype=torch.bool),
        "residue_mask": torch.ones(B, L, dtype=torch.bool),
    }


def _validate_submission_interface(
    diags: List[Diagnostic],
    cfg: Dict[str, Any],
    config_path: Path,
    *,
    track_spec: TrackSpec,
) -> None:
    try:
        hooks = load_submission_hooks(cfg, config_path, allowed_root=config_path.parent)
    except Exception as exc:  # noqa: BLE001
        add_error(diags, f"Failed loading submission entrypoint: {exc}")
        return

    try:
        model = hooks.build_model(cfg)
    except Exception as exc:  # noqa: BLE001
        add_error(diags, f"`build_model(cfg)` failed: {exc}")
        return

    try:
        enforce_model_param_limit(track_spec=track_spec, n_params=count_parameters(model))
    except Exception as exc:  # noqa: BLE001
        add_error(diags, str(exc))
        return

    try:
        optimizer = hooks.build_optimizer(cfg, model)
    except Exception as exc:  # noqa: BLE001
        add_error(diags, f"`build_optimizer(cfg, model)` failed: {exc}")
        return

    if not callable(getattr(optimizer, "step", None)) or not callable(getattr(optimizer, "zero_grad", None)):
        add_error(diags, "`build_optimizer` must return an optimizer-like object with `step()` and `zero_grad()`.")
        return

    crop_size = int(cfg["data"]["crop_size"])
    msa_depth = int(cfg["data"]["msa_depth"])
    synthetic_batch = _make_synthetic_batch(crop_size=crop_size, msa_depth=msa_depth)

    model.train()
    try:
        _ = run_submission_batch(
            hooks,
            model=model,
            batch=synthetic_batch,
            cfg=cfg,
            training=True,
        )
    except Exception as exc:  # noqa: BLE001
        add_error(diags, f"`run_batch(..., training=True)` failed validation: {exc}")
        return

    model.eval()
    try:
        with torch.no_grad():
            _ = run_submission_batch(
                hooks,
                model=model,
                batch=strip_supervision_from_batch(synthetic_batch),
                cfg=cfg,
                training=False,
            )
    except Exception as exc:  # noqa: BLE001
        add_error(diags, f"`run_batch(..., training=False)` failed validation: {exc}")


def _iter_submission_files(submission_dir: Path) -> List[Path]:
    files: List[Path] = []
    for path in sorted(submission_dir.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        files.append(path)
    return files


def _validate_submission_artifacts(diags: List[Diagnostic], submission_dir: Path) -> None:
    for path in _iter_submission_files(submission_dir):
        ext = path.suffix.lower()
        size = path.stat().st_size
        if size > MAX_ALLOWED_FILE_SIZE_BYTES:
            add_error(
                diags,
                f"File exceeds {MAX_ALLOWED_FILE_SIZE_BYTES // (1024 * 1024)}MB limit: {path} ({size} bytes).",
            )
        if ext in FORBIDDEN_WEIGHT_EXTENSIONS:
            add_error(
                diags,
                f"Forbidden model artifact extension detected: {path}. "
                "Pretrained weights are not allowed in submissions.",
            )


def _collect_import_names(py_path: Path) -> List[str]:
    text = py_path.read_text()
    tree = ast.parse(text, filename=str(py_path))
    names: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module)
    return names


def _validate_suspicious_imports(diags: List[Diagnostic], submission_dir: Path) -> None:
    for path in _iter_submission_files(submission_dir):
        if path.suffix.lower() != ".py":
            continue
        try:
            imports = _collect_import_names(path)
        except Exception as exc:  # noqa: BLE001
            add_warning(diags, f"Could not parse imports for {path}: {exc}")
            continue
        for name in imports:
            for suspicious in SUSPICIOUS_IMPORTS:
                if name == suspicious or name.startswith(f"{suspicious}."):
                    add_warning(
                        diags,
                        f"Suspicious import `{name}` in {path}; network/model download paths are disallowed in official runs.",
                    )


def _extract_metadata_value(notes_text: str, field: str) -> str | None:
    pattern = re.compile(rf"^\s*-\s*{re.escape(field)}\s*:\s*(.*)$", flags=re.MULTILINE)
    match = pattern.search(notes_text)
    if not match:
        return None
    return match.group(1).strip()


def _validate_notes(
    diags: List[Diagnostic],
    notes_text: str,
    allow_template_fields: bool,
    effective_batch_size: int,
    sample_budget: int,
    residue_budget: int,
) -> None:
    required_sections = (
        "## Submission Summary",
        "## Method Rationale",
        "## How to run",
    )
    for section in required_sections:
        if section not in notes_text:
            add_error(diags, f"`notes.md` missing section `{section}`")

    template_guidance_snippets = (
        "Template submission scaffold for participants.",
        "Summarize the expected gains and intuition behind your method.",
    )
    for snippet in template_guidance_snippets:
        if snippet in notes_text and not allow_template_fields:
            add_error(diags, "notes.md still contains template guidance text; write submission-specific details.")

    for field in METADATA_FIELDS:
        value = _extract_metadata_value(notes_text, field)
        if value is None:
            add_warning(diags, f"`notes.md` missing metadata line `- {field}: ...`")
            continue
        if value == "":
            if allow_template_fields:
                add_warning(diags, f"`notes.md` metadata `{field}` is empty.")
            else:
                add_error(diags, f"`notes.md` metadata `{field}` is empty.")

    ebs_value = _extract_metadata_value(notes_text, "effective_batch_size")
    if ebs_value and ebs_value.isdigit():
        if int(ebs_value) != effective_batch_size:
            add_warning(
                diags,
                f"`notes.md` effective_batch_size={ebs_value} does not match config-derived value {effective_batch_size}.",
            )

    bsample_value = _extract_metadata_value(notes_text, "sample_budget")
    if bsample_value and bsample_value.isdigit():
        if int(bsample_value) != sample_budget:
            add_warning(
                diags,
                f"`notes.md` sample_budget={bsample_value} does not match config-derived value {sample_budget}.",
            )

    bres_value = _extract_metadata_value(notes_text, "residue_budget")
    if bres_value and bres_value.isdigit():
        if int(bres_value) != residue_budget:
            add_warning(
                diags,
                f"`notes.md` residue_budget={bres_value} does not match config-derived value {residue_budget}.",
            )

    checklist_items = (
        "Used only the provided benchmark data",
        "Kept dataset manifests fixed",
        "Model outputs atom14 coordinates",
    )
    for item in checklist_items:
        matched = re.search(rf"^\s*-\s*\[(x|X)\]\s*.*{re.escape(item)}", notes_text, flags=re.MULTILINE)
        if not matched:
            add_warning(diags, f"Compliance checklist item not checked in notes.md: `{item}`")


def validate_submission(submission_dir: Path, strict: bool, track_id: str) -> int:
    diags: List[Diagnostic] = []
    submission_dir = submission_dir.resolve()

    try:
        track_spec = load_track_spec(track_id)
    except Exception as exc:  # noqa: BLE001
        add_error(diags, f"Failed to load track `{track_id}`: {exc}")
        _print_diagnostics(diags)
        return 1

    if not submission_dir.exists():
        add_error(diags, f"Submission directory does not exist: {submission_dir}")
    elif not submission_dir.is_dir():
        add_error(diags, f"Submission path is not a directory: {submission_dir}")

    if any(d.level == "error" for d in diags):
        _print_diagnostics(diags)
        return 1

    submission_name = submission_dir.name
    allow_template_fields = submission_name == "template"

    config_path = submission_dir / "config.yaml"
    notes_path = submission_dir / "notes.md"

    cfg: Dict[str, Any] = {}
    if not config_path.exists():
        add_error(diags, f"Missing required file: {config_path}")
    else:
        try:
            loaded = yaml.safe_load(config_path.read_text())
            if not isinstance(loaded, dict):
                add_error(diags, f"`{config_path}` must contain a YAML mapping at top level.")
            else:
                cfg = loaded
        except Exception as exc:  # noqa: BLE001
            add_error(diags, f"Failed to parse `{config_path}`: {exc}")

    if not notes_path.exists():
        add_error(diags, f"Missing required file: {notes_path}")
        notes_text = ""
    else:
        notes_text = notes_path.read_text()

    if cfg:
        _require_keys(diags, cfg, REQUIRED_TOP_LEVEL, scope="")
        if "data" in cfg and isinstance(cfg["data"], dict):
            _require_keys(diags, cfg["data"], REQUIRED_DATA_KEYS, scope="data")
        else:
            add_error(diags, "`data` must be a mapping")
        if "train" in cfg and isinstance(cfg["train"], dict):
            _require_keys(diags, cfg["train"], REQUIRED_TRAIN_KEYS, scope="train")
        else:
            add_error(diags, "`train` must be a mapping")

    if cfg and all(k in cfg for k in ("submission", "data", "train")):
        cfg = apply_track_policy(cfg, track_spec=track_spec)
        effective_batch_size, sample_budget, residue_budget = _validate_numeric(diags, cfg)
        policy_errors = validate_track_policy(
            cfg=cfg,
            track_spec=track_spec,
            enforce_manifest_paths=True,
            enforce_manifest_hashes=True,
        )
        for msg in policy_errors:
            add_error(diags, msg)
        _validate_submission_entrypoint(
            diags,
            cfg,
            config_path=config_path,
            submission_dir=submission_dir,
            allow_template_fields=allow_template_fields,
        )
        _validate_submission_interface(diags, cfg, config_path=config_path, track_spec=track_spec)
        _validate_submission_artifacts(diags, submission_dir=submission_dir)
        _validate_suspicious_imports(diags, submission_dir=submission_dir)
        run_name = str(cfg.get("run_name", "")).strip()
        if run_name == "":
            add_error(diags, "`run_name` must be non-empty.")
        if not allow_template_fields and run_name in {"your_name_run1", "baseline"}:
            add_warning(diags, f"`run_name` looks like a template/default value: `{run_name}`")
    else:
        effective_batch_size = 0
        sample_budget = 0
        residue_budget = 0

    if notes_text:
        _validate_notes(
            diags,
            notes_text,
            allow_template_fields=allow_template_fields,
            effective_batch_size=effective_batch_size,
            sample_budget=sample_budget,
            residue_budget=residue_budget,
        )

    _print_diagnostics(diags)

    n_errors = sum(d.level == "error" for d in diags)
    n_warnings = sum(d.level == "warning" for d in diags)

    if n_errors > 0:
        return 1
    if strict and n_warnings > 0:
        return 1
    return 0


def _print_diagnostics(diags: List[Diagnostic]) -> None:
    errors = [d for d in diags if d.level == "error"]
    warnings = [d for d in diags if d.level == "warning"]

    print(f"Validation summary: {len(errors)} error(s), {len(warnings)} warning(s)")
    for d in errors:
        print(f"ERROR: {d.message}")
    for d in warnings:
        print(f"WARN:  {d.message}")

    if not errors and not warnings:
        print("Submission looks valid.")


def main() -> None:
    args = parse_args()
    code = validate_submission(Path(args.submission), strict=bool(args.strict), track_id=args.track)
    sys.exit(code)


if __name__ == "__main__":
    main()
