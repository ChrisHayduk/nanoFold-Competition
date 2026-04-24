"""Ensure predict.py sanitizes data.processed_labels_dir before any downstream use."""

from __future__ import annotations

import importlib


def test_sanitize_predict_config_clears_labels_dir() -> None:
    predict = importlib.import_module("predict")
    cfg = {
        "data": {
            "processed_features_dir": "/tmp/feat",
            "processed_labels_dir": "/tmp/lab",
            "train_manifest": "manifests/train.txt",
            "val_manifest": "manifests/val.txt",
        },
    }
    sanitized = predict._sanitize_predict_config(cfg)
    assert sanitized["data"]["processed_labels_dir"] == ""
    # Original config must not be mutated.
    assert cfg["data"]["processed_labels_dir"] == "/tmp/lab"
    # Other keys pass through.
    assert sanitized["data"]["processed_features_dir"] == "/tmp/feat"


def test_sanitize_predict_config_requires_data_section() -> None:
    import pytest

    predict = importlib.import_module("predict")
    with pytest.raises(ValueError):
        predict._sanitize_predict_config({"not_data": {}})
