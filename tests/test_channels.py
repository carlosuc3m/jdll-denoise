from __future__ import annotations

import numpy as np
from conftest import tiny_config

from jdll_denoise import denoise


def test_channels_preserved_with_independent_scales() -> None:
    rng = np.random.default_rng(8)
    source = np.stack(
        (rng.random((8, 8), dtype=np.float32), 1000 + 200 * rng.random((8, 8), dtype=np.float32))
    )
    result = denoise(source, tiny_config("zs_n2n", "cyx"))
    assert result["denoised"].shape == source.shape
    assert result["metadata"]["channels"] == 2
    assert len(result["metadata"]["normalization"]) == 2
    assert result["denoised"][1].mean() > 100
    assert result["denoised"][0].mean() < 10


def test_time_and_axis_order_restored() -> None:
    rng = np.random.default_rng(9)
    source = rng.random((8, 8, 2, 2), dtype=np.float32)  # Y, X, T, C
    result = denoise(source, tiny_config("noise2fast", "yxtc"))
    assert result["denoised"].shape == source.shape
    assert result["metadata"]["axes"] == "yxtc"
    assert len(result["metadata"]["jobs"]) == 4
