from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


EXPECTED_TRAIN_MANIFEST_PARTS = ("data", "manifests", "train.txt")
EXPECTED_VAL_MANIFEST_PARTS = ("data", "manifests", "val.txt")

OFFICIAL_DATASET_FINGERPRINT_PATH = "leaderboard/official_dataset_fingerprint.json"


@dataclass(frozen=True)
class OfficialLimitedSpec:
    name: str
    seed: int
    crop_size: int
    msa_depth: int
    effective_batch_size: int
    max_steps: int
    val_crop_mode: str
    val_msa_sample_mode: str

    @property
    def residue_budget(self) -> int:
        return compute_residue_budget(
            max_steps=self.max_steps,
            effective_batch_size=self.effective_batch_size,
            crop_size=self.crop_size,
        )


OFFICIAL_LARGE_V3_SPEC = OfficialLimitedSpec(
    name="official_limited_large_v3",
    seed=0,
    crop_size=256,
    msa_depth=192,
    effective_batch_size=2,
    max_steps=10000,
    val_crop_mode="center",
    val_msa_sample_mode="top",
)


def compute_effective_batch_size(batch_size: int, grad_accum_steps: int) -> int:
    return int(batch_size) * int(grad_accum_steps)


def compute_residue_budget(max_steps: int, effective_batch_size: int, crop_size: int) -> int:
    return int(max_steps) * int(effective_batch_size) * int(crop_size)


def resolve_val_crop_mode(cfg: Dict[str, Any]) -> str:
    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, dict):
        return "center"
    return str(data_cfg.get("val_crop_mode", "center"))


def resolve_val_msa_sample_mode(cfg: Dict[str, Any]) -> str:
    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, dict):
        return "top"
    return str(data_cfg.get("val_msa_sample_mode", "top"))


def _normalized_parts(path_str: str) -> Tuple[str, ...]:
    path = Path(path_str)
    raw_parts = [p for p in path.parts if p not in ("", ".")]
    return tuple(raw_parts)


def path_endswith_parts(path_str: str, expected_parts: Tuple[str, ...]) -> bool:
    parts = _normalized_parts(path_str)
    if len(parts) < len(expected_parts):
        return False
    return tuple(parts[-len(expected_parts) :]) == expected_parts


def _get_nested(cfg: Dict[str, Any], path: str) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _require_int(cfg: Dict[str, Any], path: str, errors: List[str], default: int | None = None) -> int | None:
    value = _get_nested(cfg, path)
    if value is None:
        if default is None:
            errors.append(f"Missing key `{path}`.")
            return None
        value = default
    if not isinstance(value, int):
        errors.append(f"`{path}` must be an integer (got {type(value).__name__}).")
        return None
    return value


def _require_str(cfg: Dict[str, Any], path: str, errors: List[str]) -> str | None:
    value = _get_nested(cfg, path)
    if value is None:
        errors.append(f"Missing key `{path}`.")
        return None
    if not isinstance(value, str) or value.strip() == "":
        errors.append(f"`{path}` must be a non-empty string.")
        return None
    return value


def validate_official_limited_config(
    cfg: Dict[str, Any],
    *,
    spec: OfficialLimitedSpec = OFFICIAL_LARGE_V3_SPEC,
    enforce_manifest_paths: bool = True,
) -> List[str]:
    errors: List[str] = []
    if not isinstance(cfg, dict):
        return ["Config root must be a mapping."]

    seed = _require_int(cfg, "seed", errors)
    crop_size = _require_int(cfg, "data.crop_size", errors)
    msa_depth = _require_int(cfg, "data.msa_depth", errors)
    batch_size = _require_int(cfg, "data.batch_size", errors)
    grad_accum_steps = _require_int(cfg, "train.grad_accum_steps", errors, default=1)
    max_steps = _require_int(cfg, "train.max_steps", errors)

    if seed is not None and seed != spec.seed:
        errors.append(f"`seed` must be {spec.seed} for `{spec.name}` (got {seed}).")
    if crop_size is not None and crop_size != spec.crop_size:
        errors.append(f"`data.crop_size` must be {spec.crop_size} for `{spec.name}` (got {crop_size}).")
    if msa_depth is not None and msa_depth != spec.msa_depth:
        errors.append(f"`data.msa_depth` must be {spec.msa_depth} for `{spec.name}` (got {msa_depth}).")
    if max_steps is not None and max_steps != spec.max_steps:
        errors.append(f"`train.max_steps` must be {spec.max_steps} for `{spec.name}` (got {max_steps}).")

    val_crop_mode = resolve_val_crop_mode(cfg)
    if val_crop_mode != spec.val_crop_mode:
        errors.append(
            f"`data.val_crop_mode` must resolve to `{spec.val_crop_mode}` for `{spec.name}` "
            f"(got `{val_crop_mode}`)."
        )

    val_msa_sample_mode = resolve_val_msa_sample_mode(cfg)
    if val_msa_sample_mode != spec.val_msa_sample_mode:
        errors.append(
            f"`data.val_msa_sample_mode` must resolve to `{spec.val_msa_sample_mode}` for `{spec.name}` "
            f"(got `{val_msa_sample_mode}`)."
        )

    effective_batch_size: int | None = None
    if batch_size is not None and grad_accum_steps is not None:
        effective_batch_size = compute_effective_batch_size(batch_size=batch_size, grad_accum_steps=grad_accum_steps)
        if effective_batch_size != spec.effective_batch_size:
            errors.append(
                "`effective_batch_size` must be "
                f"{spec.effective_batch_size} for `{spec.name}` "
                f"(got {effective_batch_size} from data.batch_size={batch_size} "
                f"and train.grad_accum_steps={grad_accum_steps})."
            )

    if max_steps is not None and crop_size is not None and effective_batch_size is not None:
        residue_budget = compute_residue_budget(
            max_steps=max_steps,
            effective_batch_size=effective_batch_size,
            crop_size=crop_size,
        )
        if residue_budget != spec.residue_budget:
            errors.append(
                "`B_res` must be "
                f"{spec.residue_budget} for `{spec.name}` "
                f"(got {residue_budget} from train.max_steps={max_steps}, "
                f"effective_batch_size={effective_batch_size}, data.crop_size={crop_size})."
            )

    if enforce_manifest_paths:
        train_manifest = _require_str(cfg, "data.train_manifest", errors)
        val_manifest = _require_str(cfg, "data.val_manifest", errors)
        if train_manifest is not None and not path_endswith_parts(train_manifest, EXPECTED_TRAIN_MANIFEST_PARTS):
            errors.append(
                "Competition requires `data.train_manifest` to target `data/manifests/train.txt`."
            )
        if val_manifest is not None and not path_endswith_parts(val_manifest, EXPECTED_VAL_MANIFEST_PARTS):
            errors.append("Competition requires `data.val_manifest` to target `data/manifests/val.txt`.")
        if train_manifest is not None and val_manifest is not None and train_manifest == val_manifest:
            errors.append("`data.train_manifest` and `data.val_manifest` must be different files.")

    return errors


def assert_official_limited_config(
    cfg: Dict[str, Any],
    *,
    spec: OfficialLimitedSpec = OFFICIAL_LARGE_V3_SPEC,
    enforce_manifest_paths: bool = True,
) -> None:
    errors = validate_official_limited_config(
        cfg=cfg,
        spec=spec,
        enforce_manifest_paths=enforce_manifest_paths,
    )
    if errors:
        joined = "\n".join(f"- {msg}" for msg in errors)
        raise ValueError(f"Official limited-track config validation failed:\n{joined}")
