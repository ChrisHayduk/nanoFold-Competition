#!/usr/bin/env python3
"""Checksum-gated target-free inference and NanoFold atom14 export."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ARCH_ROOT = HERE.parent
sys.path.insert(0, str(ARCH_ROOT / "common"))
sys.path.insert(0, str(HERE))

from adapter import (  # noqa: E402
    ccd_reference_sha256,
    load_protenix_ccd_reference,
    nanofold_inputs_to_protenix,
    prediction_to_nanofold,
)
from checkpoint import BoundedCheckpointManager, sha256_file  # noqa: E402
from config import PROTENIX_COMMIT, build_protenix_config  # noqa: E402
from data import chain_npz_path, read_manifest  # noqa: E402
from train import canonical_json_sha256, to_device  # noqa: E402


def seed_inference(seed: int) -> None:
    """Seed every RNG used by Protenix inference on one process."""

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Fixed algorithm selection avoids autotuner-induced variation. Some
    # upstream fused kernels are not guaranteed bitwise deterministic across
    # CUDA/driver versions, so the exported metadata says seed-controlled
    # rather than claiming cross-platform bitwise identity.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False


def load_target_free_example(
    data_root: Path,
    chain_id: str,
    *,
    crop_size: int,
    msa_depth: int,
) -> tuple[dict, Path]:
    """Load only the public feature NPZ; never construct or open a label path."""

    if crop_size < 1 or msa_depth < 1:
        raise ValueError("crop_size and msa_depth must be positive")
    feature_path = chain_npz_path(data_root / "processed_features", chain_id)
    with np.load(feature_path, allow_pickle=False) as features:
        aatype_full = np.asarray(features["aatype"], dtype=np.int64)
        msa_full = np.asarray(features["msa"], dtype=np.int64)
        deletions_full = np.asarray(features["deletions"], dtype=np.float32)
        residue_index_full = np.asarray(features["residue_index"], dtype=np.int64)
    if np.any((msa_full < 0) | (msa_full > 21)):
        bad = np.unique(msa_full[(msa_full < 0) | (msa_full > 21)]).tolist()
        raise ValueError(f"unexpected NanoFold MSA tokens: {bad}")
    msa_full = msa_full.copy()
    msa_full[msa_full == 21] = 31
    length = int(aatype_full.shape[0])
    if aatype_full.shape != (length,) or residue_index_full.shape != (length,):
        raise ValueError(f"{feature_path}: inconsistent one-dimensional features")
    if msa_full.ndim != 2 or msa_full.shape[1] != length:
        raise ValueError(f"{feature_path}: inconsistent MSA dimensions")
    if deletions_full.shape != msa_full.shape:
        raise ValueError(f"{feature_path}: MSA/deletion shape mismatch")
    if np.any((aatype_full < 0) | (aatype_full >= 20)):
        bad = np.unique(aatype_full[(aatype_full < 0) | (aatype_full >= 20)]).tolist()
        raise ValueError(f"noncanonical validation residue types: {bad}")
    if msa_full.shape[0] < 1:
        raise ValueError(f"{feature_path}: MSA must contain a query row")

    # Match the common loader's evaluation policy exactly: summary statistics
    # use the complete MSA, the residue crop is centered, and MSA rows are the
    # deterministic prefix with the query retained at row zero.
    profile_full = np.zeros((length, 32), dtype=np.float32)
    for token in range(32):
        profile_full[:, token] = np.mean(msa_full == token, axis=0)
    deletion_mean_full = np.mean(deletions_full, axis=0, dtype=np.float32)
    crop_length = min(length, crop_size)
    crop_start = (length - crop_length) // 2
    crop_slice = slice(crop_start, crop_start + crop_length)
    keep_count = min(int(msa_full.shape[0]), msa_depth)
    aatype = torch.from_numpy(aatype_full[crop_slice]).long()
    deletions = torch.from_numpy(
        deletions_full[:keep_count, crop_slice]
    ).float()
    token_mask = torch.ones(crop_length, dtype=torch.bool)
    example = {
        "chain_id": chain_id,
        "source_length": length,
        "crop_start": crop_start,
        "crop_length": crop_length,
        "aatype": aatype,
        "restype": F.one_hot(aatype, num_classes=32).float(),
        "msa": torch.from_numpy(msa_full[:keep_count, crop_slice]).long(),
        "msa_mask": torch.ones((keep_count, crop_length), dtype=torch.bool),
        "has_deletion": (deletions > 0).float(),
        "deletion_value": ((2.0 / np.pi) * torch.atan(deletions / 3.0)).float(),
        "profile": torch.from_numpy(profile_full[crop_slice]).float(),
        "deletion_mean": torch.from_numpy(
            deletion_mean_full[crop_slice]
        ).float(),
        "residue_index": torch.from_numpy(
            residue_index_full[crop_slice]
        ).long(),
        "token_mask": token_mask,
    }
    return example, feature_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protenix-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chain-id")
    parser.add_argument("--diffusion-steps", type=int, default=200)
    parser.add_argument("--diffusion-samples", type=int, default=1)
    parser.add_argument("--model-seeds", type=int, default=1)
    parser.add_argument("--mc-dropout-apply-rate", type=float, default=0.0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20_260_725)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--msa-depth", type=int, default=192)
    args = parser.parse_args()
    if args.diffusion_steps < 1:
        raise ValueError("--diffusion-steps must be positive")
    if args.diffusion_samples < 1:
        raise ValueError("--diffusion-samples must be positive")
    if args.model_seeds < 1:
        raise ValueError("--model-seeds must be positive")
    if not 0.0 <= args.mc_dropout_apply_rate <= 1.0:
        raise ValueError("--mc-dropout-apply-rate must be in [0,1]")
    total_samples = args.diffusion_samples * args.model_seeds
    if not 0 <= args.sample_index < total_samples:
        raise ValueError(
            f"--sample-index must be in [0,{total_samples - 1}] for this sampling contract"
        )
    sys.path.insert(0, str(args.protenix_root))

    git_head = subprocess.run(
        ["git", "-C", str(args.protenix_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if git_head != PROTENIX_COMMIT:
        raise RuntimeError(f"Protenix commit mismatch: {git_head} != {PROTENIX_COMMIT}")
    git_tree = subprocess.run(
        ["git", "-C", str(args.protenix_root), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    from protenix.model.protenix import Protenix

    manager = BoundedCheckpointManager(args.checkpoint_dir)
    checkpoint = manager.load_latest()  # verifies latest pointer and SHA sidecar
    if checkpoint is None:
        raise FileNotFoundError("no latest checkpoint")
    if checkpoint.get("protenix_commit") != PROTENIX_COMMIT:
        raise ValueError("checkpoint Protenix commit does not match pinned inference code")
    if checkpoint.get("protenix_git_tree") != git_tree:
        raise ValueError("checkpoint Protenix Git tree does not match inference checkout")
    tracked_changes = subprocess.run(
        [
            "git",
            "-C",
            str(args.protenix_root),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tracked_changes:
        raise RuntimeError(
            "pinned Protenix checkout has tracked working-tree changes:\n"
            f"{tracked_changes}"
        )
    config = build_protenix_config()
    if checkpoint.get("resolved_config_sha256") != canonical_json_sha256(
        config.to_dict()
    ):
        raise ValueError("checkpoint training config does not match inference package")
    expected_sources = checkpoint.get("source_sha256", {})
    inference_sources = {
        "adapter": HERE / "adapter.py",
        "config_python": HERE / "config.py",
        "prediction_export": HERE / "predict_one.py",
        "common_data": ARCH_ROOT / "common" / "data.py",
    }
    for name, path in inference_sources.items():
        if expected_sources.get(name) != sha256_file(path):
            raise ValueError(f"checkpoint source hash mismatch for {name}")
    ccd_reference = load_protenix_ccd_reference()
    if checkpoint.get("ccd_reference_sha256") != ccd_reference_sha256(
        ccd_reference
    ):
        raise ValueError("checkpoint CCD reference does not match inference runtime")
    config.sample_diffusion.N_step = args.diffusion_steps
    config.sample_diffusion.N_sample = args.diffusion_samples
    config.model.N_model_seed = args.model_seeds
    config.deterministic = True
    config.deterministic_seed = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Protenix(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    validation_ids = read_manifest(args.data_root / "manifests" / "val.txt")
    chain_id = args.chain_id or validation_ids[0]
    if chain_id not in set(validation_ids):
        raise ValueError(f"{chain_id} is not in the official validation manifest")
    source, feature_path = load_target_free_example(
        args.data_root,
        chain_id,
        crop_size=args.crop_size,
        msa_depth=args.msa_depth,
    )
    item = nanofold_inputs_to_protenix(
        source,
        seed=args.seed,
        step=0,
        rank=0,
        augment_reference=False,
        ccd_reference=ccd_reference,
    )
    features = to_device(item["input_feature_dict"], device)
    amp = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    # Model construction and checkpoint loading may consume RNG state. Reset
    # immediately before the stochastic MSA/diffusion path.
    seed_inference(args.seed)
    with torch.no_grad(), amp:
        prediction, _, _ = model(
            input_feature_dict=features,
            label_full_dict=None,
            label_dict=None,
            mode="inference",
            current_step=None,
            symmetric_permutation=None,
            mc_dropout_apply_rate=args.mc_dropout_apply_rate,
        )
    exported = prediction_to_nanofold(
        prediction, item, sample_index=args.sample_index
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".npz", dir=args.output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary,
            pred_atom14=exported["pred_atom14"].float().cpu().numpy(),
            pred_ca=exported["pred_ca"].float().cpu().numpy(),
            chain_id=np.asarray(chain_id),
        )
        os.replace(temporary, args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "chain_id": chain_id,
                "checkpoint_completed_step": checkpoint["completed_step"],
                "checkpoint_provenance_sha256": checkpoint["provenance_sha256"],
                "protenix_git_tree": git_tree,
                "resolved_config_sha256": checkpoint["resolved_config_sha256"],
                "ccd_reference_sha256": checkpoint["ccd_reference_sha256"],
                "feature_path": str(feature_path),
                "feature_sha256": sha256_file(feature_path),
                "target_files_accessed": False,
                "seed": args.seed,
                "diffusion_steps": args.diffusion_steps,
                "diffusion_samples": args.diffusion_samples,
                "model_seeds": args.model_seeds,
                "mc_dropout_apply_rate": args.mc_dropout_apply_rate,
                "sample_index": args.sample_index,
                "reproducibility": "seed-controlled; no cross-platform bitwise guarantee",
                "output": str(args.output),
                "output_sha256": sha256_file(args.output),
                "pred_atom14_shape": list(exported["pred_atom14"].shape),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
