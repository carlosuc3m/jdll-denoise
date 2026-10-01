from __future__ import annotations

import numpy as np
from conftest import tiny_config

from jdll_denoise import denoise, resolve_config


def test_stack_noise2fast_is_honestly_slice_wise() -> None:
    source = np.random.default_rng(2).random((2, 8, 8), dtype=np.float32)
    result = denoise(source, tiny_config("noise2fast", "zyx"))
    assert result["metadata"]["dimension_mode"] == "slice_wise_2d"
    assert result["denoised"].shape == source.shape


def test_struct_auto_dimension_route() -> None:
    assert resolve_config({"method": "structn2v", "axes": "zyx"}, (4, 8, 8))["dimensions"] == "2.5d"
    assert resolve_config({"method": "structn2v", "axes": "zyx"}, (8, 8, 8))["dimensions"] == "3d"


def test_struct_25d_and_3d_smoke() -> None:
    rng = np.random.default_rng(3)
    for mode, depth in [("2.5d", 3), ("3d", 2)]:
        source = rng.random((depth, 8, 8), dtype=np.float32)
        config = tiny_config("structn2v", "zyx", dimensions=mode)
        config["method_config"]["steps"] = 1
        result = denoise(source, config)
        assert result["denoised"].shape == source.shape
        assert result["metadata"]["dimension_mode"] == mode
