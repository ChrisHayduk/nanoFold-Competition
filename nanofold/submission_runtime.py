from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Optional

import torch

from .residue_constants import ATOM14_NUM_SLOTS, CA_ATOM14_SLOT
from .utils import sha256_file


INFERENCE_BATCH_KEYS = frozenset(
    {
        "chain_id",
        "aatype",
        "msa",
        "deletions",
        "template_aatype",
        "template_ca_coords",
        "template_ca_mask",
        "residue_index",
        "between_segment_residues",
        "residue_mask",
    }
)


@dataclass(frozen=True)
class SubmissionHooks:
    module_ref: str
    module: ModuleType
    source_path: str | None
    source_sha256: str | None
    build_model: Callable[[Dict[str, Any]], torch.nn.Module]
    build_optimizer: Callable[[Dict[str, Any], torch.nn.Module], torch.optim.Optimizer]
    run_batch: Callable[..., Dict[str, torch.Tensor]]
    build_scheduler: Optional[Callable[[Dict[str, Any], torch.optim.Optimizer], Any]]


def _import_module_from_name(module_name: str) -> tuple[str, ModuleType]:
    module = importlib.import_module(module_name)
    return f"module:{module_name}", module


def _import_module_from_path(module_path: Path) -> tuple[str, ModuleType]:
    resolved = module_path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Submission module path does not exist: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Submission module path is not a file: {resolved}")

    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]
    module_name = f"nanofold_submission_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, str(resolved))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for submission module: {resolved}")

    module = importlib.util.module_from_spec(spec)
    added = False
    module_dir = str(resolved.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
        added = True
    try:
        spec.loader.exec_module(module)
    finally:
        if added:
            sys.path.pop(0)
    return f"path:{resolved}", module


def _resolve_module_source_path(module: ModuleType) -> str | None:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or module_file.strip() == "":
        return None
    resolved = Path(module_file).resolve()
    if not resolved.exists() or not resolved.is_file():
        return None
    return str(resolved)


def _ensure_path_within_root(path: Path, allowed_root: Path) -> None:
    try:
        path.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(
            f"Submission module path must resolve inside `{allowed_root}` (got `{path}`)."
        ) from exc


def load_submission_hooks(
    cfg: Dict[str, Any],
    config_path: str | Path,
    *,
    allowed_root: str | Path | None = None,
) -> SubmissionHooks:
    """Load required submission hooks from cfg['submission'].

    Supported config formats:
      submission:
        module: "python.import.path"
    or
      submission:
        path: "relative/or/absolute/path/to/submission.py"
    """
    config_path = Path(config_path).resolve()
    submission_cfg = cfg.get("submission")

    if not isinstance(submission_cfg, dict):
        raise ValueError("Config must contain a `submission` mapping with either `module` or `path`.")

    module_name = submission_cfg.get("module")
    module_path = submission_cfg.get("path")
    has_name = isinstance(module_name, str) and module_name.strip() != ""
    has_path = isinstance(module_path, str) and module_path.strip() != ""

    if has_name == has_path:
        raise ValueError("`submission` must set exactly one of `module` or `path`.")

    if has_name:
        assert isinstance(module_name, str)
        module_ref, module = _import_module_from_name(module_name.strip())
    else:
        assert isinstance(module_path, str)
        path = Path(module_path.strip())
        if not path.is_absolute():
            path = config_path.parent / path
        path = path.resolve()
        if allowed_root is not None:
            _ensure_path_within_root(path, Path(allowed_root).resolve())
        module_ref, module = _import_module_from_path(path)

    required = ("build_model", "build_optimizer", "run_batch")
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise AttributeError(f"Submission module missing required callables: {', '.join(missing)}")

    build_scheduler = getattr(module, "build_scheduler", None)
    if build_scheduler is not None and not callable(build_scheduler):
        raise TypeError("`build_scheduler` exists but is not callable.")

    source_path = _resolve_module_source_path(module)
    source_sha256 = sha256_file(source_path) if source_path is not None else None

    return SubmissionHooks(
        module_ref=module_ref,
        module=module,
        source_path=source_path,
        source_sha256=source_sha256,
        build_model=getattr(module, "build_model"),
        build_optimizer=getattr(module, "build_optimizer"),
        run_batch=getattr(module, "run_batch"),
        build_scheduler=build_scheduler,
    )


def strip_supervision_from_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in batch.items()
        if key in INFERENCE_BATCH_KEYS
    }


def run_submission_batch(
    hooks: SubmissionHooks,
    model: torch.nn.Module,
    batch: Dict[str, Any],
    cfg: Dict[str, Any],
    training: bool,
) -> Dict[str, torch.Tensor]:
    run_batch_input = batch
    if not training:
        # Runtime-level safety: never expose supervision tensors to submission code
        # in inference mode, even if a caller accidentally passes them through.
        run_batch_input = strip_supervision_from_batch(batch)

    out = hooks.run_batch(model=model, batch=run_batch_input, cfg=cfg, training=training)
    if not isinstance(out, dict):
        raise TypeError("`run_batch` must return a dict with required key `pred_atom14`.")

    aatype_shape: tuple[int, int] | None = None
    if "aatype" in run_batch_input and torch.is_tensor(run_batch_input["aatype"]):
        B, L = run_batch_input["aatype"].shape[:2]
        aatype_shape = (int(B), int(L))

    pred_atom14 = out.get("pred_atom14")
    if pred_atom14 is None:
        raise KeyError("`run_batch` output is missing required key `pred_atom14`.")
    if not torch.is_tensor(pred_atom14):
        raise TypeError("`pred_atom14` must be a torch.Tensor.")
    if pred_atom14.ndim != 4 or pred_atom14.shape[2:] != (ATOM14_NUM_SLOTS, 3):
        raise ValueError(
            f"`pred_atom14` must have shape (B, L, 14, 3), got {tuple(pred_atom14.shape)}"
        )
    if aatype_shape is not None:
        B, L = aatype_shape
        if pred_atom14.shape[0] != B or pred_atom14.shape[1] != L:
            raise ValueError(
                f"`pred_atom14` must match aatype shape (B={B}, L={L}); "
                f"got {tuple(pred_atom14.shape)}"
            )
    if not torch.is_floating_point(pred_atom14):
        raise TypeError("`pred_atom14` tensor must have floating point dtype.")
    if not torch.isfinite(pred_atom14).all():
        raise ValueError("`pred_atom14` contains NaN/Inf.")

    out = dict(out)
    out["pred_ca"] = pred_atom14[:, :, CA_ATOM14_SLOT, :]

    pred_ca = out["pred_ca"]
    if not torch.is_tensor(pred_ca):
        raise TypeError("`pred_ca` must be a torch.Tensor.")
    if pred_ca.ndim != 3 or pred_ca.shape[-1] != 3:
        raise ValueError(f"`pred_ca` must have shape (B, L, 3), got {tuple(pred_ca.shape)}")

    if aatype_shape is not None:
        B, L = aatype_shape
        if pred_ca.shape[0] != B or pred_ca.shape[1] != L:
            raise ValueError(
                f"`pred_ca` must match aatype shape (B={B}, L={L}); got {tuple(pred_ca.shape)}"
            )

    if not torch.is_floating_point(pred_ca):
        raise TypeError("`pred_ca` tensor must have floating point dtype.")
    if not torch.isfinite(pred_ca).all():
        raise ValueError("`pred_ca` contains NaN/Inf.")

    if training and "loss" not in out:
        raise KeyError("`run_batch(training=True)` must include scalar `loss` in output dict.")

    if "loss" in out:
        loss = out["loss"]
        if not torch.is_tensor(loss):
            raise TypeError("`loss` must be a torch.Tensor.")
        if loss.ndim != 0:
            raise ValueError(f"`loss` must be a scalar tensor, got shape {tuple(loss.shape)}")
        if not torch.isfinite(loss):
            raise ValueError("`loss` is NaN/Inf.")
        if training and not loss.requires_grad:
            raise ValueError("`loss` must require gradients when training=True.")

    return out
