from __future__ import annotations

import numpy as np
import pytest
from conftest import tiny_config

from jdll_denoise import InputFormatError, denoise, preview


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_same_shape_float32_and_scale(dtype) -> None:
    source = np.arange(64).reshape(8, 8).astype(dtype)
    original = source.copy()
    config = tiny_config("noise2fast")
    config_original = {**config, "method_config": dict(config["method_config"])}
    result = denoise(source, config)
    assert result["denoised"].shape == source.shape
    assert result["denoised"].dtype == np.float32
    assert np.isfinite(result["denoised"]).all()
    assert np.array_equal(source, original)
    assert config == config_original
    assert result["metadata"]["source_dtype"] == str(source.dtype)


def test_constant_image_is_exact() -> None:
    source = np.full((9, 11), -7.5, dtype=np.float32)
    result = denoise(source, tiny_config("noise2fast"))
    np.testing.assert_array_equal(result["denoised"], source)


def test_non_finite_rejected() -> None:
    source = np.ones((8, 8), dtype=np.float32)
    source[0, 0] = np.nan
    with pytest.raises(InputFormatError, match="NaN or Inf"):
        denoise(source, tiny_config("noise2fast"))


def test_preview_metadata() -> None:
    result = preview(np.ones((8, 8), dtype=np.float32), tiny_config("noise2fast"))
    metadata = result["metadata"]
    assert metadata["cache_reusable_for_full_input"] is True
    assert metadata["preview_region"] == {"z_start": 0, "z_end": 1, "center_z": 0}


def test_volumetric_preview_returns_center_plane() -> None:
    source = np.random.default_rng(2).random((3, 8, 8), dtype=np.float32)
    config = tiny_config("structn2v", "zyx", dimensions="2.5d")
    config["method_config"]["steps"] = 1
    result = preview(source, config)
    assert result["denoised"].shape == (8, 8)
    assert result["metadata"]["output_axes"] == "yx"
    assert result["metadata"]["preview_region"]["center_z"] == 1


def test_result_is_contiguous(noisy_image: np.ndarray) -> None:
    result = denoise(noisy_image[:, ::-1], tiny_config("zs_n2n"))
    assert result["denoised"].flags.c_contiguous


def test_seeded_cpu_run_is_deterministic(noisy_image: np.ndarray) -> None:
    first = denoise(noisy_image, tiny_config("noise2fast"))["denoised"]
    second = denoise(noisy_image, tiny_config("noise2fast"))["denoised"]
    np.testing.assert_array_equal(first, second)
