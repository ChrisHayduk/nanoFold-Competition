from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


EXPECTED_TRAIN_MANIFEST_PARTS = ("data", "manifests", "train.txt")
EXPECTED_VAL_MANIFEST_PARTS = ("data", "manifests", "val.txt")

DEFAULT_TRACK_ID = "limited_large_v3"
TRACKS_DIR = Path(__file__).resolve().parents[1] / "tracks"


@dataclass(frozen=True)
class TrackSpec:
    track_id: str
    name: str
    mode: str
    official: bool
    train_manifest: str
    val_manifest: str
    fingerprint_path: str | None
    train_chain_count: int | None
    val_chain_count: int | None
    seed: int | None
    crop_size: int | None
    msa_depth: int | None
    effective_batch_size: int | None
    max_steps: int | None
    val_crop_mode: str | None
    val_msa_sample_mode: str | None

    @property
    def residue_budget(self) -> int | None:
        if self.max_steps is None or self.effective_batch_size is None or self.crop_size is None:
            return None
        return compute_residue_budget(
            max_steps=self.max_steps,
            effective_batch_size=self.effective_batch_size,
            crop_size=self.crop_size,
        )


@dataclass(frozen=True)
class OfficialLimitedSpec:
    # Backward-compatible view used by existing validation tooling.
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


def _tracks_dir() -> Path:
    if not TRACKS_DIR.exists():
        raise FileNotFoundError(f"Tracks directory not found: {TRACKS_DIR}")
    return TRACKS_DIR


def list_track_ids() -> List[str]:
    return sorted(p.stem for p in _tracks_dir().glob("*.yaml"))


def track_file_path(track_id: str) -> Path:
    path = _tracks_dir() / f"{track_id}.yaml"
    if not path.exists():
        available = ", ".join(list_track_ids())
        raise FileNotFoundError(f"Unknown track `{track_id}`. Available tracks: {available}")
    return path


def load_track_spec(track_id: str = DEFAULT_TRACK_ID) -> TrackSpec:
    path = track_file_path(track_id)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Track file must contain a YAML mapping: {path}")

    dataset = raw.get("dataset", {})
    train = raw.get("train", {})
    if not isinstance(dataset, dict):
        raise ValueError(f"`dataset` must be a mapping in {path}")
    if not isinstance(train, dict):
        raise ValueError(f"`train` must be a mapping in {path}")

    return TrackSpec(
        track_id=str(raw.get("id") or track_id),
        name=str(raw.get("name") or track_id),
        mode=str(raw.get("mode") or "limited"),
        official=bool(raw.get("official", False)),
        train_manifest=str(dataset.get("train_manifest", "data/manifests/train.txt")),
        val_manifest=str(dataset.get("val_manifest", "data/manifests/val.txt")),
        fingerprint_path=_opt_str(dataset.get("fingerprint")),
        train_chain_count=_opt_int(dataset.get("train_chain_count")),
        val_chain_count=_opt_int(dataset.get("val_chain_count")),
        seed=_opt_int(train.get("seed")),
        crop_size=_opt_int(train.get("crop_size")),
        msa_depth=_opt_int(train.get("msa_depth")),
        effective_batch_size=_opt_int(train.get("effective_batch_size")),
        max_steps=_opt_int(train.get("max_steps")),
        val_crop_mode=_opt_str(train.get("val_crop_mode")),
        val_msa_sample_mode=_opt_str(train.get("val_msa_sample_mode")),
    )


def _opt_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Expected integer or null, got {type(value).__name__}")
    return value


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value.strip() == "":
        raise TypeError("Expected non-empty string or null")
    return value


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


def _require_manifest_suffix(path_str: str, expected: str) -> bool:
    expected_parts = tuple(p for p in Path(expected).parts if p not in ("", "."))
    return path_endswith_parts(path_str, expected_parts)


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


def validate_config_against_track(
    cfg: Dict[str, Any],
    *,
    track_spec: TrackSpec,
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

    if track_spec.seed is not None and seed is not None and seed != track_spec.seed:
        errors.append(f"`seed` must be {track_spec.seed} for `{track_spec.track_id}` (got {seed}).")
    if track_spec.crop_size is not None and crop_size is not None and crop_size != track_spec.crop_size:
        errors.append(
            f"`data.crop_size` must be {track_spec.crop_size} for `{track_spec.track_id}` (got {crop_size})."
        )
    if track_spec.msa_depth is not None and msa_depth is not None and msa_depth != track_spec.msa_depth:
        errors.append(
            f"`data.msa_depth` must be {track_spec.msa_depth} for `{track_spec.track_id}` (got {msa_depth})."
        )
    if track_spec.max_steps is not None and max_steps is not None and max_steps != track_spec.max_steps:
        errors.append(
            f"`train.max_steps` must be {track_spec.max_steps} for `{track_spec.track_id}` (got {max_steps})."
        )

    val_crop_mode = resolve_val_crop_mode(cfg)
    if track_spec.val_crop_mode is not None and val_crop_mode != track_spec.val_crop_mode:
        errors.append(
            f"`data.val_crop_mode` must resolve to `{track_spec.val_crop_mode}` for `{track_spec.track_id}` "
            f"(got `{val_crop_mode}`)."
        )

    val_msa_sample_mode = resolve_val_msa_sample_mode(cfg)
    if (
        track_spec.val_msa_sample_mode is not None
        and val_msa_sample_mode != track_spec.val_msa_sample_mode
    ):
        errors.append(
            f"`data.val_msa_sample_mode` must resolve to `{track_spec.val_msa_sample_mode}` "
            f"for `{track_spec.track_id}` (got `{val_msa_sample_mode}`)."
        )

    effective_batch_size: int | None = None
    if batch_size is not None and grad_accum_steps is not None:
        effective_batch_size = compute_effective_batch_size(
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
        )
        if (
            track_spec.effective_batch_size is not None
            and effective_batch_size != track_spec.effective_batch_size
        ):
            errors.append(
                "`effective_batch_size` must be "
                f"{track_spec.effective_batch_size} for `{track_spec.track_id}` "
                f"(got {effective_batch_size} from data.batch_size={batch_size} "
                f"and train.grad_accum_steps={grad_accum_steps})."
            )

    if max_steps is not None and crop_size is not None and effective_batch_size is not None:
        residue_budget = compute_residue_budget(
            max_steps=max_steps,
            effective_batch_size=effective_batch_size,
            crop_size=crop_size,
        )
        if track_spec.residue_budget is not None and residue_budget != track_spec.residue_budget:
            errors.append(
                "`B_res` must be "
                f"{track_spec.residue_budget} for `{track_spec.track_id}` "
                f"(got {residue_budget} from train.max_steps={max_steps}, "
                f"effective_batch_size={effective_batch_size}, data.crop_size={crop_size})."
            )

    if enforce_manifest_paths:
        train_manifest = _require_str(cfg, "data.train_manifest", errors)
        val_manifest = _require_str(cfg, "data.val_manifest", errors)
        if train_manifest is not None and not _require_manifest_suffix(train_manifest, track_spec.train_manifest):
            errors.append(
                f"Track `{track_spec.track_id}` requires `data.train_manifest` to target "
                f"`{track_spec.train_manifest}`."
            )
        if val_manifest is not None and not _require_manifest_suffix(val_manifest, track_spec.val_manifest):
            errors.append(
                f"Track `{track_spec.track_id}` requires `data.val_manifest` to target "
                f"`{track_spec.val_manifest}`."
            )
        if train_manifest is not None and val_manifest is not None and train_manifest == val_manifest:
            errors.append("`data.train_manifest` and `data.val_manifest` must be different files.")

    return errors


def assert_config_matches_track(
    cfg: Dict[str, Any],
    *,
    track_spec: TrackSpec,
    enforce_manifest_paths: bool = True,
) -> None:
    errors = validate_config_against_track(
        cfg=cfg,
        track_spec=track_spec,
        enforce_manifest_paths=enforce_manifest_paths,
    )
    if errors:
        joined = "\n".join(f"- {msg}" for msg in errors)
        raise ValueError(f"Track `{track_spec.track_id}` config validation failed:\n{joined}")


# Backward-compatible aliases used by existing code paths.
OFFICIAL_LARGE_V3_TRACK = load_track_spec(DEFAULT_TRACK_ID)
OFFICIAL_DATASET_FINGERPRINT_PATH = OFFICIAL_LARGE_V3_TRACK.fingerprint_path or "leaderboard/official_dataset_fingerprint.json"
OFFICIAL_LARGE_V3_SPEC = OfficialLimitedSpec(
    name=f"official_{OFFICIAL_LARGE_V3_TRACK.track_id}",
    seed=int(OFFICIAL_LARGE_V3_TRACK.seed or 0),
    crop_size=int(OFFICIAL_LARGE_V3_TRACK.crop_size or 0),
    msa_depth=int(OFFICIAL_LARGE_V3_TRACK.msa_depth or 0),
    effective_batch_size=int(OFFICIAL_LARGE_V3_TRACK.effective_batch_size or 0),
    max_steps=int(OFFICIAL_LARGE_V3_TRACK.max_steps or 0),
    val_crop_mode=str(OFFICIAL_LARGE_V3_TRACK.val_crop_mode or "center"),
    val_msa_sample_mode=str(OFFICIAL_LARGE_V3_TRACK.val_msa_sample_mode or "top"),
)


def validate_official_limited_config(
    cfg: Dict[str, Any],
    *,
    spec: OfficialLimitedSpec = OFFICIAL_LARGE_V3_SPEC,
    enforce_manifest_paths: bool = True,
) -> List[str]:
    # `spec` is kept for API compatibility; validation uses the official track registry.
    _ = spec
    return validate_config_against_track(
        cfg=cfg,
        track_spec=OFFICIAL_LARGE_V3_TRACK,
        enforce_manifest_paths=enforce_manifest_paths,
    )


def assert_official_limited_config(
    cfg: Dict[str, Any],
    *,
    spec: OfficialLimitedSpec = OFFICIAL_LARGE_V3_SPEC,
    enforce_manifest_paths: bool = True,
) -> None:
    _ = spec
    assert_config_matches_track(
        cfg=cfg,
        track_spec=OFFICIAL_LARGE_V3_TRACK,
        enforce_manifest_paths=enforce_manifest_paths,
    )
