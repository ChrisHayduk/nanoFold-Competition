from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from .metrics import FOLDSCORE_WEIGHT_BY_COMPONENT
from .utils import sha256_file

EXPECTED_TRAIN_MANIFEST_PARTS = ("data", "manifests", "train.txt")
EXPECTED_VAL_MANIFEST_PARTS = ("data", "manifests", "val.txt")

DEFAULT_TRACK_ID = "limited"
TRACKS_DIR = Path(__file__).resolve().parents[1] / "tracks"


@dataclass(frozen=True)
class TrackSpec:
    track_id: str
    name: str
    mode: str
    official: bool
    train_manifest: str
    val_manifest: str
    all_manifest: str | None
    train_manifest_sha256: str | None
    val_manifest_sha256: str | None
    all_manifest_sha256: str | None
    fingerprint_path: str | None
    hidden_manifest: str | None
    hidden_manifest_sha256: str | None
    hidden_fingerprint_path: str | None
    hidden_fingerprint_sha256: str | None
    hidden_lock_file: str | None
    train_chain_count: int | None
    val_chain_count: int | None
    hidden_chain_count: int | None
    rank_metric: str | None
    rank_tiebreak_metric: str | None
    foldscore_weights: Dict[str, float]
    template_policy: str | None
    templates_enabled: bool | None
    seed: int | None
    crop_size: int | None
    msa_depth: int | None
    effective_batch_size: int | None
    max_steps: int | None
    val_crop_mode: str | None
    val_msa_sample_mode: str | None
    max_params: int | None

    @property
    def sample_budget(self) -> int | None:
        if self.max_steps is None or self.effective_batch_size is None:
            return None
        return compute_sample_budget(
            max_steps=self.max_steps,
            effective_batch_size=self.effective_batch_size,
        )

    @property
    def residue_budget(self) -> int | None:
        if self.max_steps is None or self.effective_batch_size is None or self.crop_size is None:
            return None
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
    model = raw.get("model", {})
    scoring = raw.get("scoring", {})
    templates = raw.get("templates", {})
    if not isinstance(dataset, dict):
        raise ValueError(f"`dataset` must be a mapping in {path}")
    if not isinstance(train, dict):
        raise ValueError(f"`train` must be a mapping in {path}")
    if not isinstance(model, dict):
        raise ValueError(f"`model` must be a mapping in {path}")
    if not isinstance(scoring, dict):
        raise ValueError(f"`scoring` must be a mapping in {path}")
    if not isinstance(templates, dict):
        raise ValueError(f"`templates` must be a mapping in {path}")

    return TrackSpec(
        track_id=str(raw.get("id") or track_id),
        name=str(raw.get("name") or track_id),
        mode=str(raw.get("mode") or "limited"),
        official=bool(raw.get("official", False)),
        train_manifest=str(dataset.get("train_manifest", "data/manifests/train.txt")),
        val_manifest=str(dataset.get("val_manifest", "data/manifests/val.txt")),
        all_manifest=_opt_str(dataset.get("all_manifest")),
        train_manifest_sha256=_opt_sha256(dataset.get("train_manifest_sha256"), field_name="dataset.train_manifest_sha256"),
        val_manifest_sha256=_opt_sha256(dataset.get("val_manifest_sha256"), field_name="dataset.val_manifest_sha256"),
        all_manifest_sha256=_opt_sha256(dataset.get("all_manifest_sha256"), field_name="dataset.all_manifest_sha256"),
        fingerprint_path=_opt_str(dataset.get("fingerprint")),
        hidden_manifest=_opt_str(dataset.get("hidden_manifest")),
        hidden_manifest_sha256=_opt_sha256(
            dataset.get("hidden_manifest_sha256"),
            field_name="dataset.hidden_manifest_sha256",
        ),
        hidden_fingerprint_path=_opt_str(dataset.get("hidden_fingerprint")),
        hidden_fingerprint_sha256=_opt_sha256(
            dataset.get("hidden_fingerprint_sha256"),
            field_name="dataset.hidden_fingerprint_sha256",
        ),
        hidden_lock_file=_opt_str(dataset.get("hidden_lock_file")),
        train_chain_count=_opt_int(dataset.get("train_chain_count")),
        val_chain_count=_opt_int(dataset.get("val_chain_count")),
        hidden_chain_count=_opt_int(dataset.get("hidden_chain_count")),
        rank_metric=_opt_str(scoring.get("rank_metric")),
        rank_tiebreak_metric=_opt_str(scoring.get("rank_tiebreak_metric")),
        foldscore_weights=_float_map(
            scoring.get("foldscore_weights"),
            default=FOLDSCORE_WEIGHT_BY_COMPONENT,
        ),
        template_policy=_opt_str(templates.get("policy")),
        templates_enabled=_opt_bool(templates.get("enabled"), default=None),
        seed=_opt_int(train.get("seed")),
        crop_size=_opt_int(train.get("crop_size")),
        msa_depth=_opt_int(train.get("msa_depth")),
        effective_batch_size=_opt_int(train.get("effective_batch_size")),
        max_steps=_opt_int(train.get("max_steps")),
        val_crop_mode=_opt_str(train.get("val_crop_mode")),
        val_msa_sample_mode=_opt_str(train.get("val_msa_sample_mode")),
        max_params=_opt_int(model.get("max_params")),
    )


def _opt_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Expected integer or null, got {type(value).__name__}")
    return value


def _opt_bool(value: Any, *, default: bool | None) -> bool | None:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise TypeError(f"Expected boolean or null, got {type(value).__name__}")
    return value


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value.strip() == "":
        raise TypeError("Expected non-empty string or null")
    return value


def _opt_sha256(value: Any, *, field_name: str) -> str | None:
    text = _opt_str(value)
    if text is None:
        return None
    lowered = text.strip().lower()
    if len(lowered) != 64 or any(ch not in "0123456789abcdef" for ch in lowered):
        raise TypeError(f"`{field_name}` must be a 64-char lowercase hex SHA256 digest.")
    return lowered


def _float_map(value: Any, *, default: Dict[str, float]) -> Dict[str, float]:
    if value is None:
        return dict(default)
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping or null, got {type(value).__name__}")
    out: Dict[str, float] = {}
    for key, raw in value.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(f"Expected numeric value for `{key}`, got {type(raw).__name__}")
        out[str(key)] = float(raw)
    return out


def compute_effective_batch_size(batch_size: int, grad_accum_steps: int) -> int:
    return int(batch_size) * int(grad_accum_steps)


def compute_sample_budget(max_steps: int, effective_batch_size: int) -> int:
    return int(max_steps) * int(effective_batch_size)


def compute_residue_budget(max_steps: int, effective_batch_size: int, crop_size: int) -> int:
    return compute_sample_budget(max_steps=max_steps, effective_batch_size=effective_batch_size) * int(crop_size)


def _ensure_mapping(root: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = root.get(key)
    if isinstance(value, dict):
        return value
    value = {}
    root[key] = value
    return value


def apply_track_policy(cfg: Dict[str, Any], *, track_spec: TrackSpec) -> Dict[str, Any]:
    """Apply immutable track constants onto a config (override + validate model)."""
    if not isinstance(cfg, dict):
        raise TypeError("Config root must be a mapping.")
    out = copy.deepcopy(cfg)
    out["track"] = track_spec.track_id

    if track_spec.seed is not None:
        out["seed"] = int(track_spec.seed)

    data_cfg = _ensure_mapping(out, "data")
    train_cfg = _ensure_mapping(out, "train")

    if track_spec.crop_size is not None:
        data_cfg["crop_size"] = int(track_spec.crop_size)
    if track_spec.msa_depth is not None:
        data_cfg["msa_depth"] = int(track_spec.msa_depth)
    if track_spec.max_steps is not None:
        train_cfg["max_steps"] = int(track_spec.max_steps)
    if track_spec.val_crop_mode is not None:
        data_cfg["val_crop_mode"] = str(track_spec.val_crop_mode)
    if track_spec.val_msa_sample_mode is not None:
        data_cfg["val_msa_sample_mode"] = str(track_spec.val_msa_sample_mode)

    # Normalize effective_batch_size by adjusting grad_accum_steps first.
    if track_spec.effective_batch_size is not None:
        target = int(track_spec.effective_batch_size)
        batch_size = int(data_cfg.get("batch_size", 1))
        if batch_size <= 0 or target <= 0:
            raise ValueError("`batch_size` and target `effective_batch_size` must be positive.")
        if target % batch_size == 0:
            train_cfg["grad_accum_steps"] = target // batch_size
        else:
            data_cfg["batch_size"] = 1
            train_cfg["grad_accum_steps"] = target
    return out


def validate_track_policy(
    cfg: Dict[str, Any],
    *,
    track_spec: TrackSpec,
    enforce_manifest_paths: bool = True,
    enforce_manifest_hashes: bool = True,
) -> List[str]:
    return validate_config_against_track(
        cfg=cfg,
        track_spec=track_spec,
        enforce_manifest_paths=enforce_manifest_paths,
        enforce_manifest_hashes=enforce_manifest_hashes,
    )


def assert_track_policy(
    cfg: Dict[str, Any],
    *,
    track_spec: TrackSpec,
    enforce_manifest_paths: bool = True,
    enforce_manifest_hashes: bool = True,
) -> None:
    errors = validate_track_policy(
        cfg,
        track_spec=track_spec,
        enforce_manifest_paths=enforce_manifest_paths,
        enforce_manifest_hashes=enforce_manifest_hashes,
    )
    if errors:
        joined = "\n".join(f"- {msg}" for msg in errors)
        raise ValueError(f"Track `{track_spec.track_id}` policy validation failed:\n{joined}")


def enforce_model_param_limit(*, track_spec: TrackSpec, n_params: int) -> None:
    if track_spec.max_params is None:
        return
    if int(n_params) > int(track_spec.max_params):
        raise ValueError(
            f"Track `{track_spec.track_id}` allows at most {track_spec.max_params:,} trainable parameters; "
            f"got {int(n_params):,}."
        )


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


def _resolve_local_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _validate_manifest_hash(
    *,
    errors: List[str],
    path_str: str,
    expected_sha256: str | None,
    label: str,
) -> None:
    if expected_sha256 is None:
        return
    path = _resolve_local_path(path_str)
    if not path.exists():
        errors.append(f"{label} is missing on disk: `{path}`.")
        return
    if not path.is_file():
        errors.append(f"{label} must be a file: `{path}`.")
        return
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        errors.append(
            f"{label} SHA256 mismatch for `{path}`: expected `{expected_sha256}`, got `{actual_sha256}`."
        )


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
    enforce_manifest_hashes: bool = False,
) -> List[str]:
    errors: List[str] = []

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

    train_manifest: str | None = None
    val_manifest: str | None = None
    if enforce_manifest_paths or enforce_manifest_hashes:
        train_manifest = _require_str(cfg, "data.train_manifest", errors)
        val_manifest = _require_str(cfg, "data.val_manifest", errors)

    if enforce_manifest_paths:
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

    if enforce_manifest_hashes:
        if train_manifest is not None:
            _validate_manifest_hash(
                errors=errors,
                path_str=train_manifest,
                expected_sha256=track_spec.train_manifest_sha256,
                label="Train manifest",
            )
        if val_manifest is not None:
            _validate_manifest_hash(
                errors=errors,
                path_str=val_manifest,
                expected_sha256=track_spec.val_manifest_sha256,
                label="Val manifest",
            )
        if track_spec.all_manifest_sha256 is not None:
            if not track_spec.all_manifest:
                errors.append(
                    f"Track `{track_spec.track_id}` declares `dataset.all_manifest_sha256` "
                    "but not `dataset.all_manifest`."
                )
            else:
                _validate_manifest_hash(
                    errors=errors,
                    path_str=track_spec.all_manifest,
                    expected_sha256=track_spec.all_manifest_sha256,
                    label="All-chain manifest",
                )

    return errors


def assert_config_matches_track(
    cfg: Dict[str, Any],
    *,
    track_spec: TrackSpec,
    enforce_manifest_paths: bool = True,
    enforce_manifest_hashes: bool = False,
) -> None:
    errors = validate_config_against_track(
        cfg=cfg,
        track_spec=track_spec,
        enforce_manifest_paths=enforce_manifest_paths,
        enforce_manifest_hashes=enforce_manifest_hashes,
    )
    if errors:
        joined = "\n".join(f"- {msg}" for msg in errors)
        raise ValueError(f"Track `{track_spec.track_id}` config validation failed:\n{joined}")


OFFICIAL_TRACK = load_track_spec(DEFAULT_TRACK_ID)
OFFICIAL_DATASET_FINGERPRINT_PATH = OFFICIAL_TRACK.fingerprint_path or "leaderboard/official_dataset_fingerprint.json"
