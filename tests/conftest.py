from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def noisy_image() -> np.ndarray:
    rng = np.random.default_rng(42)
    clean = np.zeros((12, 12), dtype=np.float32)
    clean[3:9, 3:9] = 1.0
    return clean + rng.normal(0, 0.08, clean.shape).astype(np.float32)


def tiny_config(method: str, axes: str = "yx", **extra):
    config = {
        "method": method,
        "effort": "quick",
        "device": "cpu",
        "axes": axes,
        "seed": 17,
        "method_config": {"steps": 2, "report_interval": 1},
    }
    config.update(extra)
    if method == "structn2v":
        volumetric = "z" in axes
        config["method_config"].update(
            {
                "noise_structure": "local_isotropic",
                "mask_fraction": 0.02,
                "patch_size": [2, 8, 8] if volumetric else [8, 8],
                "patch_count": 1,
                "batch_size": 1,
                "epochs": 1,
                "selection_steps": 1,
            }
        )
    elif method == "zs_n2n":
        config["method_config"]["features"] = 4
    return config
