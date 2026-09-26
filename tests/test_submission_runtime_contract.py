from __future__ import annotations

from types import ModuleType
from typing import Any, Dict

import pytest
import torch

from nanofold.submission_runtime import (
    SubmissionHooks,
    load_submission_hooks,
    run_submission_batch,
    strip_supervision_from_batch,
)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        B, L = batch["aatype"].shape
        base = torch.zeros((B, L, 3), dtype=torch.float32, device=self.scale.device)
        return base * self.scale


def _batch() -> Dict[str, Any]:
    B, L, N, T = 1, 8, 4, 1
    return {
        "chain_id": ["TEST_A"],
        "aatype": torch.zeros((B, L), dtype=torch.long),
        "msa": torch.zeros((B, N, L), dtype=torch.long),
        "deletions": torch.zeros((B, N, L), dtype=torch.long),
        "template_aatype": torch.zeros((B, T, L), dtype=torch.long),
        "template_ca_coords": torch.zeros((B, T, L, 3), dtype=torch.float32),
        "template_ca_mask": torch.ones((B, T, L), dtype=torch.bool),
        "residue_index": torch.arange(L, dtype=torch.long).expand(B, -1),
        "between_segment_residues": torch.zeros((B, L), dtype=torch.long),
        "ca_coords": torch.zeros((B, L, 3), dtype=torch.float32),
        "ca_mask": torch.ones((B, L), dtype=torch.bool),
        "atom14_positions": torch.zeros((B, L, 14, 3), dtype=torch.float32),
        "atom14_mask": torch.ones((B, L, 14), dtype=torch.bool),
        "residue_mask": torch.ones((B, L), dtype=torch.bool),
    }


def _hooks(run_batch_fn):
    mod = ModuleType("submission")
    return SubmissionHooks(
        module_ref="test:submission",
        module=mod,
        source_path=None,
        source_sha256=None,
        build_model=lambda cfg: TinyModel(),
        build_optimizer=lambda cfg, model: torch.optim.Adam(model.parameters(), lr=1e-3),
        run_batch=run_batch_fn,
        build_scheduler=None,
    )


def _atom14_from_ca(pred_ca: torch.Tensor) -> torch.Tensor:
    pred_atom14 = pred_ca.unsqueeze(2).expand(-1, -1, 14, -1).contiguous()
    pred_atom14[:, :, 1, :] = pred_ca
    return pred_atom14


def test_run_submission_batch_accepts_valid_output() -> None:
    def run_batch(model, batch, cfg, training):  # noqa: ANN001, ANN201
        pred = model(batch)
        out = {"pred_atom14": _atom14_from_ca(pred)}
        if training:
            out["loss"] = pred.sum() * 0.0
        return out

    hooks = _hooks(run_batch)
    model = TinyModel()
    out = run_submission_batch(hooks, model=model, batch=_batch(), cfg={}, training=True)
    assert "pred_ca" in out
    assert "loss" in out


def test_run_submission_batch_rejects_wrong_shape() -> None:
    def run_batch(model, batch, cfg, training):  # noqa: ANN001, ANN201
        return {"pred_atom14": torch.zeros((1, 8, 13, 3), dtype=torch.float32), "loss": torch.tensor(0.0)}

    hooks = _hooks(run_batch)
    with pytest.raises(ValueError):
        run_submission_batch(hooks, model=TinyModel(), batch=_batch(), cfg={}, training=True)


def test_run_submission_batch_rejects_nan_prediction() -> None:
    def run_batch(model, batch, cfg, training):  # noqa: ANN001, ANN201
        pred_atom14 = torch.zeros((1, 8, 14, 3), dtype=torch.float32)
        pred_atom14[0, 0, 1, 0] = float("nan")
        return {"pred_atom14": pred_atom14, "loss": torch.tensor(0.0)}

    hooks = _hooks(run_batch)
    with pytest.raises(ValueError):
        run_submission_batch(hooks, model=TinyModel(), batch=_batch(), cfg={}, training=True)


def test_strip_supervision_from_batch_for_inference() -> None:
    batch = _batch()
    out = strip_supervision_from_batch(batch)
    assert "ca_coords" not in out
    assert "ca_mask" not in out
    assert "atom14_positions" not in out
    assert "atom14_mask" not in out
    assert set(out.keys()) == {
        "chain_id",
        "aatype",
        "msa",
        "deletions",
        "template_aatype",
        "template_ca_coords",
        "template_ca_mask",
        "residue_index",
        "between_segment_residues",
        "residue_mask",
    }


def test_run_submission_batch_strips_supervision_internally() -> None:
    observed_keys: list[str] = []

    def run_batch(model, batch, cfg, training):  # noqa: ANN001, ANN201
        observed_keys.extend(list(batch.keys()))
        pred = torch.zeros((1, 8, 3), dtype=torch.float32)
        return {"pred_atom14": _atom14_from_ca(pred)}

    hooks = _hooks(run_batch)
    _ = run_submission_batch(hooks, model=TinyModel(), batch=_batch(), cfg={}, training=False)
    assert "ca_coords" not in observed_keys
    assert "ca_mask" not in observed_keys
    assert "atom14_positions" not in observed_keys
    assert "atom14_mask" not in observed_keys


def test_run_submission_batch_requires_atom14() -> None:
    def run_batch(model, batch, cfg, training):  # noqa: ANN001, ANN201
        pred = model(batch)
        return {"pred_ca": pred}

    hooks = _hooks(run_batch)
    with pytest.raises(KeyError, match="pred_atom14"):
        run_submission_batch(
            hooks,
            model=TinyModel(),
            batch=_batch(),
            cfg={},
            training=False,
        )


def test_run_submission_batch_derives_pred_ca_from_atom14_slot() -> None:
    def run_batch(model, batch, cfg, training):  # noqa: ANN001, ANN201
        B, L = batch["aatype"].shape
        pred_atom14 = torch.zeros((B, L, 14, 3), dtype=torch.float32)
        pred_atom14[:, :, 1, 0] = 7.0
        return {"pred_atom14": pred_atom14}

    hooks = _hooks(run_batch)
    out = run_submission_batch(
        hooks,
        model=TinyModel(),
        batch=_batch(),
        cfg={},
        training=False,
    )
    assert torch.allclose(out["pred_ca"], out["pred_atom14"][:, :, 1, :])


def test_run_submission_batch_training_loss_requires_grad() -> None:
    def run_batch(model, batch, cfg, training):  # noqa: ANN001, ANN201
        pred = torch.zeros((1, 8, 3), dtype=torch.float32)
        return {"pred_atom14": _atom14_from_ca(pred), "loss": torch.tensor(0.0)}

    hooks = _hooks(run_batch)
    with pytest.raises(ValueError, match="must require gradients"):
        run_submission_batch(hooks, model=TinyModel(), batch=_batch(), cfg={}, training=True)


def test_load_submission_hooks_rejects_out_of_tree_path(tmp_path) -> None:
    config_dir = tmp_path / "submission"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    config_path.write_text("run_name: test\n")

    outside_entry = tmp_path / "outside.py"
    outside_entry.write_text(
        "\n".join(
            [
                "import torch",
                "def build_model(cfg):",
                "    return torch.nn.Identity()",
                "def build_optimizer(cfg, model):",
                "    return torch.optim.Adam([torch.nn.Parameter(torch.tensor(1.0))])",
                "def run_batch(model, batch, cfg, training):",
                "    pred = torch.zeros((1, 1, 3), dtype=torch.float32)",
                "    out = {'pred_atom14': pred.unsqueeze(2).expand(-1, -1, 14, -1).contiguous()}",
                "    if training:",
                "        out['loss'] = pred.sum() * 0.0",
                "    return out",
            ]
        )
        + "\n"
    )

    cfg = {"submission": {"path": str(outside_entry)}}
    with pytest.raises(ValueError, match="inside"):
        load_submission_hooks(cfg, config_path, allowed_root=config_dir)


def test_path_submission_executes_current_source_despite_valid_stale_bytecode(tmp_path) -> None:
    import hashlib
    import importlib.util
    import os
    import py_compile
    import sys
    from pathlib import Path

    source = tmp_path / "submission.py"
    original = (
        "VALUE = 1\n"
        "def build_model(cfg): return VALUE\n"
        "def build_optimizer(cfg, model): return VALUE\n"
        "def run_batch(model, batch, cfg, training): return VALUE\n"
    )
    source.write_text(original)
    cache = py_compile.compile(str(source), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    assert cache is not None
    timestamp = source.stat()
    cache_bytes = Path(cache).read_bytes()
    changed = original.replace("VALUE = 1", "VALUE = 2")
    assert len(changed) == len(original)
    source.write_text(changed)
    os.utime(source, ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns))

    # Establish that the fixture actually defeats normal timestamp validation.
    stale_spec = importlib.util.spec_from_file_location("stale_control", source)
    assert stale_spec is not None and stale_spec.loader is not None
    stale = importlib.util.module_from_spec(stale_spec)
    stale_spec.loader.exec_module(stale)
    assert stale.VALUE == 1

    paths_before = list(sys.path)
    hooks = load_submission_hooks({"submission": {"path": str(source)}}, tmp_path / "config.yaml")
    expected_name = "nanofold_submission_" + hashlib.sha1(str(source.resolve()).encode()).hexdigest()[:12]
    assert hooks.build_model({}) == 2
    assert hooks.module.__name__ == hooks.build_model.__module__ == expected_name
    assert hooks.module.__spec__ is not None and hooks.module.__spec__.name == expected_name
    assert hooks.module.__file__ == hooks.source_path == str(source.resolve())
    assert hooks.module_ref == f"path:{source.resolve()}"
    assert hooks.source_sha256 == hashlib.sha256(changed.encode()).hexdigest()
    assert Path(cache).read_bytes() == cache_bytes
    assert sys.path == paths_before


@pytest.mark.parametrize("preloaded", [False, True])
def test_named_submission_uses_current_source_and_preserves_package_imports(tmp_path, monkeypatch, preloaded) -> None:
    import hashlib
    import importlib
    import os
    import py_compile
    import sys

    package = tmp_path / "source_bound_fixture"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "helper.py").write_text("OFFSET = 40\n")
    source = package / "submission.py"
    original = (
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "from .helper import OFFSET\n"
        "VALUE = 1\n"
        "@dataclass\n"
        "class State:\n"
        "    value: int = VALUE\n"
        "def build_model(cfg): return State().value + OFFSET\n"
        "def build_optimizer(cfg, model): return VALUE\n"
        "def run_batch(model, batch, cfg, training): return VALUE\n"
    )
    source.write_text(original)
    py_compile.compile(str(source), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    monkeypatch.syspath_prepend(str(tmp_path))
    module_name = "source_bound_fixture.submission"
    try:
        if preloaded:
            old = importlib.import_module(module_name)
            assert old.build_model({}) == 41
        timestamp = source.stat()
        changed = original.replace("VALUE = 1", "VALUE = 2")
        source.write_text(changed)
        os.utime(source, ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns))
        # Both variants start with a stale normally-imported module: one loaded
        # before mutation, one loaded afterward from the timestamp-valid cache.
        stale = importlib.import_module(module_name)
        assert stale.build_model({}) == 41
        hooks = load_submission_hooks({"submission": {"module": module_name}}, tmp_path / "config.yaml")
        assert hooks.build_model({}) == 42
        assert hooks.module.__package__ == "source_bound_fixture"
        assert hooks.module.__name__ == hooks.module_ref.removeprefix("module:") == module_name
        assert sys.modules[module_name] is hooks.module
        assert getattr(sys.modules["source_bound_fixture"], "submission") is hooks.module
        assert hooks.source_sha256 == hashlib.sha256(changed.encode()).hexdigest()
        assert hooks.source_path == str(source.resolve())
    finally:
        for key in list(sys.modules):
            if key == "source_bound_fixture" or key.startswith("source_bound_fixture."):
                sys.modules.pop(key, None)


@pytest.mark.parametrize("mode", ["path", "module"])
def test_submission_source_mutation_during_import_is_not_misattributed(tmp_path, monkeypatch, mode) -> None:
    import sys

    source = tmp_path / "self_mutating_submission.py"
    source.write_text(
        "from pathlib import Path\n"
        "Path(__file__).write_text('# changed during import\\n')\n"
        "def build_model(cfg): return 1\n"
        "def build_optimizer(cfg, model): return 1\n"
        "def run_batch(model, batch, cfg, training): return 1\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    value = str(source) if mode == "path" else source.stem
    with pytest.raises(ImportError, match="changed during import"):
        load_submission_hooks({"submission": {mode: value}}, tmp_path / "config.yaml")
    assert source.stem not in sys.modules


def test_named_non_source_submission_is_rejected(tmp_path) -> None:
    with pytest.raises(ImportError, match="readable Python source"):
        load_submission_hooks({"submission": {"module": "sys"}}, tmp_path / "config.yaml")
