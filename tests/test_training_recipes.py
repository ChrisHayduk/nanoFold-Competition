from __future__ import annotations

import json
import os
import runpy
import shlex
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest
import torch

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments" / "neurips-2026"
RF_ROOT = EXPERIMENTS / "rfdiffusion"
AF3_ROOT = EXPERIMENTS / "af3"
RF_SMALL = Path("small/architecture-scaling/rfdiffusion-style")
RF_LARGE = Path("large/architecture-reconstructions/rfdiffusion-style")
PROFILES = ("tiny", "medium", "large")


def _dry_run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> list[str]:
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, check=True)
    return shlex.split(result.stdout)


def _flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("phase", ("train", "evaluate"))
def test_rf_dry_run_routes_profiles(tmp_path: Path, profile: str, phase: str) -> None:
    paths = {"--data-root": tmp_path / "public tensors"}
    if phase == "train":
        paths["--run-dir"] = tmp_path / "training run"
    else:
        paths.update(
            {
                "--checkpoint": tmp_path / "step-0030000.pt",
                "--scorer-root": tmp_path / "official scorer",
                "--output-dir": tmp_path / "evaluation",
            }
        )
    command = [sys.executable, str(RF_ROOT / "run_rf.py"), phase, profile, "--dry-run"]
    for flag, path in paths.items():
        command.extend((flag, str(path)))
    argv = _dry_run(command, cwd=tmp_path)

    if profile == "large":
        ranks = 8
        script = RF_LARGE / f"{phase}.py"
        config = RF_LARGE / ("configs/full.json" if phase == "train" else "configs/eval-final.json")
    else:
        ranks = 2 if phase == "train" else 4
        script = RF_SMALL / f"runtime/{phase}_accumulated.py"
        config = RF_SMALL / (
            f"configs/{profile}-no-checkpoint.json" if phase == "train" else "runtime/eval-world4.json"
        )
    assert argv[:6] == [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={ranks}",
        str(RF_ROOT / script),
    ]
    assert Path(argv[5]).is_file()
    assert _flag(argv, "--config") == str(RF_ROOT / config)
    assert _flag(argv, "--dataset-verification") == str(RF_ROOT / "data-lock.json")
    for flag, path in paths.items():
        forwarded_flag = "--nanofold-source-root" if flag == "--scorer-root" else flag
        assert _flag(argv, forwarded_flag) == str(path)

    recipe = json.loads((RF_ROOT / config).read_text())
    if phase == "train":
        assert recipe["training"]["seed"] == 0
        assert recipe["training"]["steps"] == 30_000
        assert recipe["training"]["world_size"] == 8
        assert "--resume" not in argv
        if profile != "large":
            assert _flag(argv, "--physical-world-size") == "2"
    else:
        assert recipe["runtime"]["world_size"] == ranks
        assert recipe["sampling"]["num_samples"] == 5
        assert recipe["sampling"]["sampling_steps"] == 50


def _af3_dry_run(tmp_path: Path, profile: str, phase: str) -> tuple[list[str], dict[str, str]]:
    settings = {
        "PROTENIX_ROOT": str(tmp_path / "pinned Protenix"),
        "PROTENIX_ROOT_DIR": str(tmp_path / "pinned CCD"),
        "NANOFOLD_DATA_ROOT": str(tmp_path / "public tensors"),
        "NANOFOLD_VERIFY_REPORT": str(tmp_path / "data report.json"),
        "NANOFOLD_RUN_DIR": str(tmp_path / "training run"),
        "NANOFOLD_SCORER_ROOT": str(tmp_path / "official scorer"),
    }
    argv = _dry_run(
        ["bash", str(AF3_ROOT / "run_af3.sh"), profile, phase],
        cwd=tmp_path,
        env={**os.environ, **settings, "AF3_PRINT_COMMAND": "1", "AF3_PYTHON": sys.executable},
    )
    return argv, settings


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("phase", ("train", "evaluate"))
def test_af3_dry_run_routes_profiles(tmp_path: Path, profile: str, phase: str) -> None:
    argv, settings = _af3_dry_run(tmp_path, profile, phase)
    runtime = AF3_ROOT / "runtime" / profile / "protenix-af3-style"
    assert argv[:7] == [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=8",
        str(runtime / f"{phase}.py"),
    ]
    assert Path(argv[6]).is_file()
    assert _flag(argv, "--protenix-root") == settings["PROTENIX_ROOT"]
    assert _flag(argv, "--data-root") == settings["NANOFOLD_DATA_ROOT"]
    run_dir = Path(settings["NANOFOLD_RUN_DIR"])
    if phase == "train":
        assert _flag(argv, "--dataset-verification-report") == settings["NANOFOLD_VERIFY_REPORT"]
        assert _flag(argv, "--run-dir") == str(run_dir)
        assert _flag(argv, "--resume") == "never"
        assert _flag(argv, "--max-steps") == "30000"
        assert _flag(argv, "--seed") == "20260725"
        assert "--smoke-steps" not in argv
    else:
        name = "eval-final.json" if profile == "large" else "eval-final-world8.json"
        config_path = runtime / "configs" / name
        assert _flag(argv, "--config") == str(config_path)
        assert _flag(argv, "--dataset-verification") == settings["NANOFOLD_VERIFY_REPORT"]
        assert _flag(argv, "--checkpoint") == str(run_dir / "checkpoints/completed-step-0030000.pt")
        assert _flag(argv, "--training-provenance") == str(run_dir / "provenance.json")
        assert _flag(argv, "--nanofold-source-root") == settings["NANOFOLD_SCORER_ROOT"]


def test_af3_large_prefix_and_continuation_share_terminal_budget(tmp_path: Path) -> None:
    prefix, _ = _af3_dry_run(tmp_path, "large", "prefix250")
    continuation, _ = _af3_dry_run(tmp_path, "large", "continue")
    assert _flag(prefix, "--max-steps") == "30000"
    assert _flag(prefix, "--resume") == "never"
    assert _flag(prefix, "--smoke-steps") == "250"
    assert _flag(continuation, "--resume") == "required"
    assert "--smoke-steps" not in continuation
    assert _flag(prefix, "--run-dir") == _flag(continuation, "--run-dir")
    assert _flag(prefix, "--max-steps") == _flag(continuation, "--max-steps")


def test_rf_two_rank_accumulation_preserves_logical_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    path = RF_ROOT / RF_SMALL / "runtime/train_accumulated.py"
    monkeypatch.syspath_prepend(str(path.parent))
    for name in ("data", "geometry", "model", "runtime", "verify_dataset"):
        # Track originally absent modules so cleanup also removes new imports.
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
        del sys.modules[name]
    trainer = runpy.run_path(str(path))
    contract = trainer["execution_contract"](2)
    assert contract["gradient_accumulation_steps"] == 4
    schedule = trainer["DeterministicBatchSchedule"](tuple(f"chain-{i}" for i in range(16)), 0, 8)
    logical_rank = trainer["logical_rank_for"]
    for step in (0, 1, 2, 29_999):
        two_rank = [
            schedule.chain_at(step, logical_rank(physical_rank=rank, microstep=micro, physical_world_size=2))
            for micro in range(4)
            for rank in range(2)
        ]
        assert two_rank == [schedule.chain_at(step, rank) for rank in range(8)]
        assert len({row[0] for row in two_rank}) == 8
    metadata = trainer["execution_metadata"]({"_execution_contract": contract}, 2, 30_000)
    assert metadata["optimizer_updates"] == 30_000
    assert metadata["samples_seen"] == 240_000


@pytest.mark.parametrize("profile", ("tiny", "medium"))
def test_af3_accumulated_step_averages_gradients_and_updates_once(profile: str) -> None:
    recipe = runpy.run_path(str(AF3_ROOT / "runtime" / profile / "common/effective_batch.py"))

    class ProbeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
            self.sync = True

        @contextmanager
        def no_sync(self):
            self.sync = False
            try:
                yield
            finally:
                self.sync = True

    model = ProbeModel()
    model.weight.grad = torch.tensor(99.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    initial_epoch = scheduler.last_epoch
    calls = []

    def forward_loss(rank: int) -> torch.Tensor:
        calls.append((rank, model.sync))
        return model.weight * (rank + 1)

    def clip_grad() -> torch.Tensor:
        assert model.weight.grad is not None
        return model.weight.grad.abs()

    def require_finite(value: torch.Tensor) -> None:
        assert torch.isfinite(value)

    loss, norm = recipe["accumulated_optimizer_step"](
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        physical_rank=1,
        physical_world_size=2,
        forward_loss=forward_loss,
        require_finite=require_finite,
        clip_grad=clip_grad,
    )
    assert calls == [(1, False), (3, False), (5, False), (7, True)]
    assert loss.item() == pytest.approx(10.0)
    assert norm.item() == pytest.approx(5.0)
    assert model.weight.item() == pytest.approx(1.5)
    assert scheduler.last_epoch == initial_epoch + 1
