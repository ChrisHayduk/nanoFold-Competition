from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from nanofold.metrics import foldscore_components, foldscore_auc
from nanofold.utils import get_env_metadata, utc_now_iso


HIDDEN_SPLITS = {"hidden_val", "test_hidden"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Score saved predictions against labels.")
    ap.add_argument("--prediction-summary", type=str, required=True, help="Path to predict.py JSON summary.")
    ap.add_argument("--labels-dir", type=str, required=True, help="Directory with label NPZ files.")
    ap.add_argument("--manifest", type=str, default="", help="Optional manifest override.")
    ap.add_argument("--crop-size", type=int, default=-1, help="Optional crop-size override.")
    ap.add_argument(
        "--crop-mode",
        type=str,
        default="",
        help="Optional crop mode override (defaults to prediction summary crop_mode).",
    )
    ap.add_argument(
        "--per-chain-out",
        type=str,
        default="",
        help="Optional JSONL output path for per-chain FoldScore/component records.",
    )
    ap.add_argument(
        "--save",
        type=str,
        default="",
        help="Optional path to write scoring summary JSON.",
    )
    return ap.parse_args()


def _load_label_crop(
    *,
    labels_dir: Path,
    chain_id: str,
    crop_size: int,
    crop_mode: str,
) -> Dict[str, torch.Tensor]:
    label_path = labels_dir / f"{chain_id}.npz"
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file for scoring: {label_path}")
    with np.load(label_path) as data:
        missing = [key for key in ("ca_coords", "ca_mask", "atom14_positions", "atom14_mask") if key not in data]
        if missing:
            raise ValueError(f"Label file {label_path} is missing required keys: {', '.join(missing)}")
        labels: Dict[str, torch.Tensor] = {
            "ca_coords": torch.from_numpy(data["ca_coords"]).float(),
            "ca_mask": torch.from_numpy(data["ca_mask"]).bool(),
            "atom14_positions": torch.from_numpy(np.asarray(data["atom14_positions"], dtype=np.float32)),
            "atom14_mask": torch.from_numpy(np.asarray(data["atom14_mask"], dtype=bool)),
        }
    ca_coords = labels["ca_coords"]
    ca_mask = labels["ca_mask"]
    if ca_coords.ndim != 2 or ca_coords.shape[-1] != 3:
        raise ValueError(f"Invalid ca_coords shape in {label_path}: {tuple(ca_coords.shape)}")
    if ca_mask.ndim != 1:
        raise ValueError(f"Invalid ca_mask shape in {label_path}: {tuple(ca_mask.shape)}")
    if ca_coords.shape[0] != ca_mask.shape[0]:
        raise ValueError(f"Label length mismatch in {label_path}")
    atom14_positions = labels["atom14_positions"]
    if atom14_positions.ndim != 3 or atom14_positions.shape[1:] != (14, 3):
        raise ValueError(f"Invalid atom14_positions shape in {label_path}: {tuple(atom14_positions.shape)}")
    if atom14_positions.shape[0] != ca_coords.shape[0]:
        raise ValueError(f"atom14_positions length mismatch in {label_path}")
    atom14_mask = labels["atom14_mask"]
    if atom14_mask.ndim != 2 or atom14_mask.shape[1] != 14:
        raise ValueError(f"Invalid atom14_mask shape in {label_path}: {tuple(atom14_mask.shape)}")
    if atom14_mask.shape[0] != ca_coords.shape[0]:
        raise ValueError(f"atom14_mask length mismatch in {label_path}")

    L = int(ca_coords.shape[0])
    if L <= crop_size:
        return labels
    if crop_mode == "center":
        start = (L - crop_size) // 2
    elif crop_mode == "random":
        raise ValueError("Scoring external labels with random crop is unsupported; use deterministic crop mode.")
    else:
        raise ValueError(f"Unsupported crop_mode={crop_mode!r}")
    end = start + crop_size
    cropped = dict(labels)
    cropped["ca_coords"] = labels["ca_coords"][start:end]
    cropped["ca_mask"] = labels["ca_mask"][start:end]
    cropped["atom14_positions"] = labels["atom14_positions"][start:end]
    cropped["atom14_mask"] = labels["atom14_mask"][start:end]
    return cropped


def _score_chain(
    *,
    pred_atom14: torch.Tensor,
    labels: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    has_atom14_labels = "atom14_positions" in labels and "atom14_mask" in labels
    if not has_atom14_labels:
        raise ValueError("FoldScore scoring requires label atom14_positions/atom14_mask.")
    comps = foldscore_components(
        pred_atom14=pred_atom14,
        true_atom14=labels["atom14_positions"],
        atom14_mask=labels["atom14_mask"],
    )
    return {name: float(value.detach().cpu()) for name, value in comps.items()}


def _prediction_subdir(pred_root: Path, ckpt_path: str, n_ckpts: int) -> Path:
    if n_ckpts <= 1:
        return pred_root
    return pred_root / Path(ckpt_path).stem


def _validate_hidden_axis(checkpoint_rows: List[Dict[str, Any]], sample_budget: int) -> List[Dict[str, Any]]:
    if not checkpoint_rows:
        raise ValueError("Hidden scoring requires checkpoint metadata.")
    previous_samples: int | None = None
    for row in checkpoint_rows:
        current_samples = int(row.get("cumulative_samples_seen", -1))
        if current_samples < 0:
            raise ValueError(f"Checkpoint entry is missing `cumulative_samples_seen`: {row}")
        if previous_samples is not None and current_samples <= previous_samples:
            raise ValueError(
                "Hidden scoring requires strictly increasing cumulative_samples_seen values; "
                f"got {current_samples} after {previous_samples}."
            )
        previous_samples = current_samples
    if int(checkpoint_rows[0]["cumulative_samples_seen"]) != 0:
        raise ValueError("Hidden scoring requires a step-0 checkpoint with cumulative_samples_seen=0.")
    if int(checkpoint_rows[-1]["cumulative_samples_seen"]) != int(sample_budget):
        raise ValueError(
            "Hidden scoring requires a final checkpoint at the full sample budget "
            f"({sample_budget}); got {checkpoint_rows[-1]['cumulative_samples_seen']}."
        )
    return checkpoint_rows


def _hidden_curve_metrics(checkpoint_rows: List[Dict[str, Any]], *, sample_budget: int) -> Dict[str, Any]:
    rows = _validate_hidden_axis(checkpoint_rows, sample_budget=sample_budget)
    final_hidden = float(rows[-1]["mean_foldscore"])
    auc_hidden = foldscore_auc(
        (
            (
                int(row["step"]),
                int(row["cumulative_samples_seen"]),
                float(row["mean_foldscore"]),
            )
            for row in rows
        ),
        sample_budget=sample_budget,
    )
    lddt_auc = foldscore_auc(
        (
            (
                int(row["step"]),
                int(row["cumulative_samples_seen"]),
                float(row.get("mean_lddt_ca", row["mean_foldscore"])),
            )
            for row in rows
        ),
        sample_budget=sample_budget,
    )
    return {
        "final_hidden_foldscore": final_hidden,
        "foldscore_auc_hidden": auc_hidden,
        "foldscore_at_steps": {str(int(row["step"])): float(row["mean_foldscore"]) for row in rows},
        "foldscore_at_samples": {
            str(int(row["cumulative_samples_seen"])): float(row["mean_foldscore"]) for row in rows
        },
        "final_hidden_lddt_ca": float(rows[-1].get("mean_lddt_ca", final_hidden)),
        "lddt_auc_hidden": lddt_auc,
        "lddt_at_steps": {str(int(row["step"])): float(row.get("mean_lddt_ca", row["mean_foldscore"])) for row in rows},
        "lddt_at_samples": {
            str(int(row["cumulative_samples_seen"])): float(row.get("mean_lddt_ca", row["mean_foldscore"]))
            for row in rows
        },
        "checkpoint_metrics_hidden": rows,
    }


def main() -> None:
    args = parse_args()
    prediction_summary_path = Path(args.prediction_summary).resolve()
    prediction_summary = json.loads(prediction_summary_path.read_text())
    if not isinstance(prediction_summary, dict):
        raise ValueError(f"Prediction summary must be a JSON object: {prediction_summary_path}")
    if str(prediction_summary.get("mode", "")) != "predict":
        raise ValueError(f"Prediction summary must come from predict.py: {prediction_summary_path}")

    checkpoints = prediction_summary.get("checkpoints", [])
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError(f"Prediction summary missing checkpoints: {prediction_summary_path}")

    manifest_path = Path(args.manifest).resolve() if args.manifest else Path(str(prediction_summary["manifest_path"])).resolve()
    chain_ids = [
        line.strip()
        for line in manifest_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not chain_ids:
        raise ValueError(f"Empty manifest for scoring: {manifest_path}")

    labels_dir = Path(args.labels_dir).resolve()
    pred_root = Path(str(prediction_summary["pred_out_dir"])).resolve()
    crop_size = int(args.crop_size) if args.crop_size > 0 else int(prediction_summary["crop_size"])
    crop_mode = args.crop_mode.strip() or str(prediction_summary.get("crop_mode", "center"))
    split = str(prediction_summary["split"])
    n_ckpts = len(checkpoints)
    per_chain_rows: List[Dict[str, Any]] = []
    scored_checkpoints: List[Dict[str, Any]] = []
    scoring_start = time.perf_counter()

    for item in checkpoints:
        ckpt_path = str(item["ckpt"])
        pred_dir = _prediction_subdir(pred_root, ckpt_path, n_ckpts=n_ckpts)
        chain_metrics: Dict[str, List[float]] = {
            "foldscore": [],
            "lddt_ca": [],
            "lddt_backbone_atom14": [],
            "lddt_atom14": [],
        }
        ckpt_start = time.perf_counter()

        for chain_id in chain_ids:
            pred_path = pred_dir / f"{chain_id}.npz"
            if not pred_path.exists():
                raise FileNotFoundError(f"Missing prediction file for scoring: {pred_path}")
            with np.load(pred_path) as pred_npz:
                if "pred_atom14" not in pred_npz:
                    raise ValueError(f"Prediction file is missing required `pred_atom14`: {pred_path}")
                pred_atom14 = torch.from_numpy(np.asarray(pred_npz["pred_atom14"], dtype=np.float32))
            labels = _load_label_crop(
                labels_dir=labels_dir,
                chain_id=chain_id,
                crop_size=crop_size,
                crop_mode=crop_mode,
            )
            metrics = _score_chain(
                pred_atom14=pred_atom14,
                labels=labels,
            )
            for name, value in metrics.items():
                if not np.isnan(value):
                    chain_metrics[name].append(float(value))
            per_chain_rows.append(
                {
                    "split": split,
                    "ckpt": ckpt_path,
                    "step": int(item.get("step", 0)),
                    "cumulative_samples_seen": int(item.get("cumulative_samples_seen", -1)),
                    "chain_id": chain_id,
                    **metrics,
                }
            )

        scored_row = dict(item)
        for name, values in chain_metrics.items():
            scored_row[f"mean_{name}"] = float(sum(values) / len(values)) if values else float("nan")
        scored_row["num_chains"] = len(chain_ids)
        scored_row["score_wall_time_seconds"] = float(time.perf_counter() - ckpt_start)
        scored_checkpoints.append(scored_row)

    per_chain_out_path = Path(args.per_chain_out).resolve() if args.per_chain_out else None
    if per_chain_out_path:
        per_chain_out_path.parent.mkdir(parents=True, exist_ok=True)
        with per_chain_out_path.open("w") as f:
            for row in per_chain_rows:
                f.write(json.dumps(row) + "\n")
        print(f"Wrote per-chain scores to {per_chain_out_path.resolve()}")

    final_row = scored_checkpoints[-1]
    out: Dict[str, Any] = {
        "mode": "score",
        "split": split,
        "track": str(prediction_summary["track"]),
        "official_mode": bool(prediction_summary.get("official_mode", False)),
        "prediction_summary_path": str(prediction_summary_path),
        "labels_dir": str(labels_dir),
        "manifest_path": str(manifest_path),
        "num_checkpoints": len(scored_checkpoints),
        "checkpoints": scored_checkpoints,
        "mean_foldscore": float(final_row["mean_foldscore"]),
        "mean_lddt_ca": float(final_row["mean_lddt_ca"]),
        "mean_lddt_backbone_atom14": float(final_row.get("mean_lddt_backbone_atom14", float("nan"))),
        "mean_lddt_atom14": float(final_row.get("mean_lddt_atom14", float("nan"))),
        "num_chains": int(final_row["num_chains"]),
        "effective_batch_size": int(prediction_summary.get("effective_batch_size", 0)),
        "sample_budget": int(prediction_summary.get("sample_budget", 0)),
        "residue_budget": int(prediction_summary.get("residue_budget", 0)),
        "cumulative_samples_seen": int(final_row.get("cumulative_samples_seen", 0)),
        "cumulative_cropped_residues_seen": int(final_row.get("cumulative_cropped_residues_seen", 0)),
        "cumulative_nonpad_residues_seen": int(final_row.get("cumulative_nonpad_residues_seen", 0)),
        "env": get_env_metadata(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "per_chain_scores_path": str(per_chain_out_path.resolve()) if per_chain_out_path else None,
        "finished_at": utc_now_iso(),
        "score_wall_time_seconds": float(time.perf_counter() - scoring_start),
    }

    if split in HIDDEN_SPLITS:
        out.update(
            _hidden_curve_metrics(
                scored_checkpoints,
                sample_budget=int(prediction_summary.get("sample_budget", 0)),
            )
        )

    print(json.dumps(out, indent=2))
    if args.save:
        save_path = Path(args.save).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(out, indent=2) + "\n")
        print(f"Wrote scoring summary to {save_path.resolve()}")


if __name__ == "__main__":
    main()
