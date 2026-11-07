#!/usr/bin/env python3
"""Independent coordinate replay for new AF3-scale public-validation outputs."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import types


METRIC_HASH = "1926ccffd0628b99b0dc42a2520560d6a645f66f5e96c6e4a53285c2d1a4930d"
CONSTANT_HASH = "6f79a66d8c68c3d16eebfba8876e77ced4bd682c6486a1e15fae5e94a4ebd473"
VALIDATION_HASH = "d4a0265bcd0a021e116c0c889f21e86bc24006460bcc42dec2f9a80b70c8812b"
METRICS = ("foldscore", "gdt_ha_ca", "lddt_atom14", "cad_atom14", "sg_atom14", "sc_atom14", "molprobity_clash_atom14", "bb_atom14", "dipdiff_atom14", "gdt_ts_ca", "lddt_ca", "lddt_backbone_atom14", "rmsd_ca")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 << 20): h.update(block)
    return h.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--nanofold-source-root", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, choices=(27000, 30000), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    import torch.distributed as dist
    rank, local, world = (int(os.environ.get(name, default)) for name, default in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")))
    torch.cuda.set_device(local)
    if world > 1: dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    # The pinned Protenix constructor enables TF32 before the original scorer
    # runs. Reproduce that explicit state without constructing the model here.
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda", local)
    manifest = args.data_root / "manifests/val.txt"
    if digest(manifest) != VALIDATION_HASH: raise ValueError("public cohort digest differs")
    ids = [line.strip() for line in manifest.read_text().splitlines() if line.strip()]
    if len(ids) != 1000 or len(set(ids)) != 1000: raise ValueError("expected exact 1000-chain cohort")
    root = args.nanofold_source_root / "nanofold"
    if digest(root / "metrics.py") != METRIC_HASH or digest(root / "residue_constants.py") != CONSTANT_HASH: raise ValueError("metric source differs")
    package = types.ModuleType("_af3_independent_metric")
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    load(package.__name__ + ".residue_constants", root / "residue_constants.py")
    metrics = load(package.__name__ + ".metrics", root / "metrics.py")
    if tuple(metrics.FOLDSCORE_COMPONENT_NAMES) != METRICS[:-1]: raise ValueError("metric component set differs")
    summary = json.loads((args.evaluation_root / "summary.json").read_text())
    if summary["checkpoint_completed_step"] != args.expected_step or summary["num_chains"] != 1000: raise ValueError("summary endpoint/cohort differs")
    predictions = args.evaluation_root / "predictions/model-seed-000/diffusion-sample-000"
    records = args.evaluation_root / "records/model-seed-000/diffusion-sample-000"
    names = {f"chain_{chain.encode().hex()}" for chain in ids}
    if {p.stem for p in predictions.glob("*.npz")} != names or {p.stem for p in records.glob("*.json")} != names: raise ValueError("prediction/record roster differs")
    sums = {name: 0. for name in METRICS}
    maxima = {name: 0. for name in METRICS}
    mismatches = []
    rows = []
    with torch.inference_mode():
        for chain in ids[rank::world]:
            stem = f"chain_{chain.encode().hex()}"
            prediction_path, record_path = predictions / (stem + ".npz"), records / (stem + ".json")
            record = json.loads(record_path.read_text())
            if record["checkpoint_completed_step"] != args.expected_step or record["chain_id"] != chain: raise ValueError("record endpoint differs")
            with np.load(prediction_path, allow_pickle=False) as saved:
                pred = torch.as_tensor(np.asarray(saved["pred_atom14"], dtype=np.float32), device=device)
                aatype = torch.as_tensor(np.asarray(saved["aatype"], dtype=np.int64), device=device)
                start, length = int(saved["crop_start"]), int(saved["crop_length"])
            label_path = args.data_root / "processed_labels" / (stem + ".npz")
            with np.load(label_path, allow_pickle=False) as label:
                true = torch.as_tensor(np.asarray(label["atom14_positions"][start:start+length], dtype=np.float32), device=device)
                mask = torch.as_tensor(np.asarray(label["atom14_mask"][start:start+length], dtype=np.bool_), device=device)
            if pred.shape != true.shape or pred.shape != (length, 14, 3): raise ValueError("coordinate shape differs")
            values = {name: float(value.float().cpu()) for name, value in metrics.foldscore_components(pred_atom14=pred, true_atom14=true, atom14_mask=mask, aatype=aatype).items()}
            active = mask[:, 1]
            if int(active.sum()) < 3: raise ValueError("fewer than three C-alpha atoms")
            x, y = pred[:, 1][active], true[:, 1][active]
            x, y = x - x.mean(0), y - y.mean(0)
            u, _, vh = torch.linalg.svd(x.T @ y, full_matrices=False)
            correction = torch.ones(3, device=device)
            if torch.det(u @ vh) < 0: correction[-1] = -1
            rotation = u @ torch.diag(correction) @ vh
            values["rmsd_ca"] = float(((x @ rotation - y).square().sum(-1).mean().clamp_min(0)).sqrt().cpu())
            if tuple(values) != METRICS or not all(math.isfinite(v) for v in values.values()): raise ValueError("metric set/finite check differs")
            for name, value in values.items():
                delta = abs(value - float(record[name]))
                sums[name] += value
                maxima[name] = max(maxima[name], delta)
                if delta > 1e-6 and len(mismatches) < 25: mismatches.append({"chain_id": chain, "metric": name, "absolute_difference": delta})
            rows.append({"chain_id": chain, "prediction_sha256": digest(prediction_path), "label_sha256": digest(label_path), **values})
    local_result = {"rank": rank, "sums": sums, "maxima": maxima, "mismatches": mismatches, "rows": rows}
    gathered = [None] * world
    if world > 1: dist.all_gather_object(gathered, local_result)
    else: gathered[0] = local_result
    failed = any(row["mismatches"] for row in gathered)
    if rank == 0:
        all_rows = sorted([row for shard in gathered for row in shard["rows"]], key=lambda row: row["chain_id"])
        if {row["chain_id"] for row in all_rows} != set(ids) or len(all_rows) != 1000: raise ValueError("independent row census differs")
        result = {"schema_version": 1, "state": "FAILED" if failed else "PASS", "checkpoint_completed_step": args.expected_step,
            "num_chains": 1000, "world_size": world, "metric_tolerance": 1e-6,
            "maximum_absolute_difference": {name: max(row["maxima"][name] for row in gathered) for name in METRICS},
            "means": {name: math.fsum(row["sums"][name] for row in gathered) / 1000 for name in METRICS},
            "mismatch_examples": [item for row in gathered for item in row["mismatches"]][:25],
            "tf32": torch.backends.cuda.matmul.allow_tf32, "torch": str(torch.__version__),
            "metrics_sha256": METRIC_HASH, "residue_constants_sha256": CONSTANT_HASH,
            "summary_sha256": digest(args.evaluation_root / "summary.json"), "script_sha256": digest(Path(__file__)),
            "rows": all_rows}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle: handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))
    if world > 1: dist.destroy_process_group()
    if failed: raise SystemExit(1)


if __name__ == "__main__": main()
