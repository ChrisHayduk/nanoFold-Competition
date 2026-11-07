#!/usr/bin/env python3
"""Fail-closed AF3 dependency, declared GPU and fused-layernorm smoke."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from download_af3_ccd import PROTENIX_COMMIT, REFERENCE_SHA256, MEMBERS, verify_file


GPU_PROFILES = {"h100": ("H100", (9, 0)), "a40": ("A40", (8, 6)), "a100": ("A100", (8, 0))}


def validate_gpu_hardware(name: str, capability: tuple[int, int], profile: str) -> None:
    expected_name, expected_capability = GPU_PROFILES[profile]
    if expected_name not in name or tuple(capability) != expected_capability:
        raise RuntimeError(f"GPU differs from declared {profile} profile: {name}, {capability}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protenix-root", type=Path, required=True)
    parser.add_argument("--af3-root", type=Path, required=True)
    parser.add_argument("--ccd-root", type=Path, required=True)
    parser.add_argument("--expected-gpus", type=int, required=True)
    parser.add_argument("--gpu-profile", choices=tuple(GPU_PROFILES), default="h100")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64" or sys.version_info[:2] != (3, 12):
        raise RuntimeError("requires Linux x86_64 and Python 3.12")
    if args.expected_gpus < 1:
        raise ValueError("expected GPU count must be positive")
    version = subprocess.check_output(["git", "-C", str(args.protenix_root), "rev-parse", "HEAD"], text=True).strip()
    if version != PROTENIX_COMMIT:
        raise ValueError("upstream commit differs")
    for name, (size, digest) in MEMBERS.items():
        verify_file(args.ccd_root / name, size, digest)
    versions = {}
    pins = Path(__file__).with_name("af3-minimal-requirements.lock.txt")
    for line in pins.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==")
        versions[name] = importlib.metadata.version(name)
        if versions[name] != expected:
            raise ValueError(f"package version differs: {name}={versions[name]} != {expected}")
    import torch
    if torch.__version__ != "2.7.1+cu126" or torch.version.cuda != "12.6":
        raise RuntimeError(f"Torch/CUDA differs: {torch.__version__}, {torch.version.cuda}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != args.expected_gpus:
        raise RuntimeError("CUDA unavailable or GPU count differs")
    sys.path.insert(0, str(args.protenix_root))
    sys.path.insert(0, str(args.af3_root))
    os.environ["PROTENIX_ROOT_DIR"] = str(args.ccd_root)
    os.environ.pop("PROTENIX_CCD_REFERENCE_JSON", None)
    from adapter import ccd_reference_sha256, load_protenix_ccd_reference
    from config import build_protenix_config
    table = load_protenix_ccd_reference()
    if ccd_reference_sha256(table) != REFERENCE_SHA256:
        raise ValueError("chemical reference digest differs")
    config = build_protenix_config()
    # Import compiles/loads the exact pinned CUDA extension; no model or labels.
    from protenix.model.protenix import Protenix  # noqa: F401
    from protenix.model.layer_norm.layer_norm import FusedLayerNorm, fast_layer_norm_cuda_v2
    from cuequivariance_torch.primitives.triangle import triangle_attention, triangle_multiplicative_update  # noqa: F401
    gpu_rows = []
    for index in range(args.expected_gpus):
        validate_gpu_hardware(torch.cuda.get_device_name(index), torch.cuda.get_device_capability(index), args.gpu_profile)
        device = torch.device("cuda", index)
        values = torch.randn(4, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
        module = FusedLayerNorm(32).to(device=device, dtype=torch.bfloat16)
        result = module(values)
        result.float().square().mean().backward()
        torch.cuda.synchronize(device)
        if not torch.isfinite(result).all() or values.grad is None or not torch.isfinite(values.grad).all():
            raise RuntimeError("fused layernorm forward/backward smoke failed")
        gpu_rows.append({"index": index, "name": torch.cuda.get_device_name(index),
                         "capability": list(torch.cuda.get_device_capability(index)),
                         "fused_layernorm_forward_backward": "PASS"})
    extension = Path(fast_layer_norm_cuda_v2.__file__)
    extension_info = {"path": str(extension), "sha256": hashlib.sha256(extension.read_bytes()).hexdigest()}
    payload = {"schema_version": 1, "state": "RUNTIME_PREFLIGHT_PASSED",
               "training_started": False, "model_forward_run": False,
               "triangle_kernel_execution_verified": False,
               "remaining_gate": "outcome-blind complete-model timing/forward-backward smoke and distributed NCCL",
               "python": sys.version, "platform": platform.platform(), "packages": versions,
               "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
               "cudnn": torch.backends.cudnn.version(), "gpus": gpu_rows,
               "declared_gpu_profile": args.gpu_profile,
               "nvcc": subprocess.check_output(["nvcc", "--version"], text=True),
               "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True),
               "ccd_reference_sha256": REFERENCE_SHA256,
               "protenix_commit": PROTENIX_COMMIT,
               "resolved_config_sha256": hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
               "layernorm_extension": extension_info,
               "tf32_after_import": torch.backends.cuda.matmul.allow_tf32,
               "tf32_model_constructor_contract": bool(config.enable_tf32)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
