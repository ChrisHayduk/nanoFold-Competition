from __future__ import annotations

import torch

from neurips_paper.submission_common.minalphafold2_experiment import (
    _fape_clamp_weight,
    _n_cycles_for_batch,
)


def _features(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {"aatype": torch.zeros((batch_size, 8), dtype=torch.long)}


def test_paper_recycle_default_samples_train_but_uses_max_for_eval() -> None:
    cfg = {"seed": 3, "model": {"n_cycles": 4}}
    features = _features()

    sampled = {
        _n_cycles_for_batch({**cfg, "_runtime": {"step": step}}, features, training=True)
        for step in range(32)
    }

    assert sampled <= {1, 2, 3, 4}
    assert len(sampled) > 1
    assert _n_cycles_for_batch(cfg, features, training=False) == 4


def test_paper_recycle_fixed_mode_is_explicit_train_ablation() -> None:
    cfg = {
        "model": {"n_cycles": 4},
        "experiment": {"recycle": {"train_mode": "fixed"}},
    }

    assert _n_cycles_for_batch(cfg, _features(), training=True) == 4
    assert _n_cycles_for_batch(cfg, _features(), training=False) == 4


def test_paper_fape_default_is_af2_batchwise_with_expected_eval_mix() -> None:
    cfg = {"seed": 11, "model": {"n_cycles": 2}, "_runtime": {"step": 17}}
    features = _features(batch_size=4)

    first = _fape_clamp_weight(cfg, features, training=True)
    second = _fape_clamp_weight(cfg, features, training=True)
    eval_weight = _fape_clamp_weight(cfg, features, training=False)

    assert first is not None
    assert second is not None
    assert float(first.item()) in {0.0, 1.0}
    assert torch.equal(first, second)
    assert eval_weight is not None
    assert torch.isclose(eval_weight, torch.tensor(0.9))


def test_paper_fape_samplewise_mode_is_only_for_explicit_ablation() -> None:
    cfg = {
        "seed": 11,
        "model": {"n_cycles": 2},
        "_runtime": {"step": 17},
        "experiment": {
            "fape": {
                "backbone_clamp_mode": "samplewise",
                "clamped_probability": 0.9,
            }
        },
    }
    features = _features(batch_size=4)

    train_weight = _fape_clamp_weight(cfg, features, training=True)
    eval_weight = _fape_clamp_weight(cfg, features, training=False)

    assert train_weight is not None
    assert train_weight.shape == (4,)
    assert set(train_weight.tolist()) <= {0.0, 1.0}
    assert eval_weight is not None
    assert torch.isclose(eval_weight, torch.tensor(0.9))
