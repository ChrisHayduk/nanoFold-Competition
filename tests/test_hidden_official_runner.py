from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_run_official_module():
    module_path = Path("scripts/run_official.py").resolve()
    spec = importlib.util.spec_from_file_location("run_official_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hidden_scoring_with_synthetic_predictions(tmp_path: Path) -> None:
    module = _load_run_official_module()
    score_fn = getattr(module, "_score_hidden_predictions")

    manifest = tmp_path / "hidden_manifest.txt"
    manifest.write_text("1abc_A\n")

    labels_dir = tmp_path / "hidden_labels"
    labels_dir.mkdir()
    true_ca = np.zeros((8, 3), dtype=np.float32)
    ca_mask = np.ones((8,), dtype=bool)
    np.savez_compressed(labels_dir / "1abc_A.npz", ca_coords=true_ca, ca_mask=ca_mask)

    pred_root = tmp_path / "hidden_preds"
    pred_root.mkdir()
    ckpt_1 = "/tmp/ckpt_step_1000.pt"
    ckpt_2 = "/tmp/ckpt_step_2000.pt"
    (pred_root / "ckpt_step_1000").mkdir()
    (pred_root / "ckpt_step_2000").mkdir()
    np.savez_compressed(pred_root / "ckpt_step_1000" / "1abc_A.npz", pred_ca=true_ca, masked_length=np.array(8))
    np.savez_compressed(pred_root / "ckpt_step_2000" / "1abc_A.npz", pred_ca=true_ca, masked_length=np.array(8))

    result = score_fn(
        hidden_manifest=manifest,
        hidden_labels_dir=labels_dir,
        pred_root=pred_root,
        checkpoint_entries=[{"ckpt": ckpt_1}, {"ckpt": ckpt_2}],
        crop_size=8,
        max_steps=2000,
        per_chain_out_path=tmp_path / "per_chain_hidden.jsonl",
    )

    assert result["final_hidden_lddt_ca"] == 1.0
    assert result["lddt_auc_hidden"] == 1.0
    assert result["lddt_at_steps"]["1000"] == 1.0
    assert result["lddt_at_steps"]["2000"] == 1.0
