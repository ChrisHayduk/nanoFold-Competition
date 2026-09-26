from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import torch

from nanofold.utils import load_torch_checkpoint, serialize_numpy_rng_state, set_seed


def test_checkpoint_payload_loads_with_restricted_torch_loader(tmp_path) -> None:
    numpy_state = serialize_numpy_rng_state(cast(tuple[Any, ...], np.random.get_state()))
    path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "model": {"weight": torch.ones(1)},
            "rng_state": {
                "numpy": numpy_state,
                "torch": torch.get_rng_state(),
            },
        },
        path,
    )

    ckpt = load_torch_checkpoint(path, map_location="cpu")

    assert torch.equal(ckpt["model"]["weight"], torch.ones(1))
    assert isinstance(ckpt["rng_state"]["numpy"][1], list)
    np.random.set_state(ckpt["rng_state"]["numpy"])


def test_requested_determinism_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(mode: bool) -> None:
        raise RuntimeError("deterministic backend unavailable")

    monkeypatch.setattr(torch, "use_deterministic_algorithms", unavailable)
    with pytest.raises(RuntimeError, match="deterministic backend unavailable"):
        set_seed(0, deterministic=True)
