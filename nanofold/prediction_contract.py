"""Bindings shared by prediction writers and the independent scorer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

PREDICTION_FORMAT_VERSION = 1


def validate_checkpoint_for_prediction(
    checkpoint: dict[str, Any],
    cfg: dict[str, Any],
    *,
    submission_source_sha256: str | None,
    require_metadata: bool,
    training_fingerprint_sha256: str | None = None,
    track_id: str | None = None,
) -> bool:
    """Check semantic settings as well as the state-dict's eventual shape check.

    File locations may change between training and a sealed prediction runtime.
    Legacy research checkpoints may lack metadata; their unverified status must
    remain explicit in the prediction summary.
    """
    saved_cfg = checkpoint.get("config")
    saved_source = checkpoint.get("submission_entrypoint_sha256")
    if saved_cfg is not None:
        if not isinstance(saved_cfg, dict):
            raise ValueError("Checkpoint config must be a mapping.")
        if saved_cfg.get("model", {}) != cfg.get("model", {}):
            raise ValueError("Checkpoint model config does not match prediction config.")
        for config in (saved_cfg, cfg):
            if not isinstance(config.get("submission", {}), dict):
                raise ValueError("Checkpoint and prediction submission settings must be mappings.")
        saved_submission = {k: v for k, v in saved_cfg.get("submission", {}).items() if k not in {"module", "path"}}
        requested_submission = {k: v for k, v in cfg.get("submission", {}).items() if k not in {"module", "path"}}
        if saved_submission != requested_submission:
            raise ValueError("Checkpoint submission settings do not match prediction config.")
    if saved_source is not None and saved_source != submission_source_sha256:
        raise ValueError("Checkpoint submission source hash does not match loaded prediction code.")
    verified = isinstance(saved_cfg, dict) and isinstance(saved_source, str) and bool(saved_source)
    if require_metadata and not verified:
        raise ValueError("Official prediction requires checkpoint config and submission source hash metadata.")
    if training_fingerprint_sha256 is not None and checkpoint.get("fingerprint_sha256") != training_fingerprint_sha256:
        raise ValueError("Checkpoint training fingerprint does not match the trusted public dataset fingerprint.")
    if track_id is not None and checkpoint.get("track_id") != track_id:
        raise ValueError("Checkpoint training track does not match the prediction track.")
    return verified


def validate_checkpoint_output_names(checkpoints: list[Path]) -> None:
    if len(checkpoints) != len(set(checkpoints)):
        raise ValueError("Checkpoint list contains duplicate paths.")
    if len(checkpoints) > 1 and len({path.stem for path in checkpoints}) != len(checkpoints):
        raise ValueError("Multi-checkpoint prediction requires distinct checkpoint stems to avoid output collisions.")


def prepare_prediction_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"Prediction directory must be empty; refusing to mix or overwrite predictions: {path}")


def validate_complete_chain_ids(actual: list[str], expected: list[str]) -> None:
    if len(actual) != len(set(actual)) or len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError("Prediction/scoring must cover every manifest chain exactly once.")


def load_prediction(
    path: Path,
    *,
    chain_id: str,
    checkpoint_sha256: str | None,
    require_binding: bool,
) -> torch.Tensor:
    with np.load(path, allow_pickle=False) as data:
        if "pred_atom14" not in data:
            raise ValueError(f"Prediction file is missing required `pred_atom14`: {path}")
        coords = np.asarray(data["pred_atom14"])
        if coords.dtype.kind != "f" or not np.isfinite(coords).all():
            raise ValueError("Prediction coordinates must be finite floating point values.")
        if coords.ndim != 3 or coords.shape[1:] != (14, 3) or coords.shape[0] == 0:
            raise ValueError("Prediction coordinates must have nonempty shape (L, 14, 3).")
        if require_binding:
            if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64 or any(
                char not in "0123456789abcdef" for char in checkpoint_sha256
            ):
                raise ValueError("Prediction summary is missing a valid checkpoint SHA256 binding.")
            for key, expected in (("chain_id", chain_id), ("ckpt_sha256", checkpoint_sha256)):
                if key not in data or data[key].shape != () or str(data[key].item()) != expected:
                    raise ValueError(f"Prediction {key} does not match its scoring identity: {path}")
        if "masked_length" in data:
            length = data["masked_length"]
            if length.shape != () or length.dtype.kind not in "iu" or int(length.item()) != coords.shape[0]:
                raise ValueError(f"Prediction masked_length does not match its coordinate length: {path}")
        elif require_binding:
            raise ValueError(f"Prediction is missing masked_length: {path}")
        # The standalone and in-process scorers both compute in float32.
        with np.errstate(over="ignore", invalid="ignore"):
            coords32 = coords.astype(np.float32)
        if not np.isfinite(coords32).all():
            raise ValueError("Prediction coordinates must remain finite in float32.")
        return torch.from_numpy(coords32)
