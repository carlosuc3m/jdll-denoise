from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from jdll_denoise import MethodUnavailableError, denoise
from jdll_denoise.methods.bm import estimate_noise_sigma


def test_sigma_estimate_is_bounded() -> None:
    sigma, warning = estimate_noise_sigma(np.ones((16, 16), dtype=np.float32))
    assert sigma == pytest.approx(0.001)
    assert warning


@pytest.mark.skipif(importlib.util.find_spec("bm3d") is not None, reason="bm3d is installed")
def test_missing_optional_bm_package_is_clear() -> None:
    with pytest.raises(MethodUnavailableError, match="Tampere license"):
        denoise(
            np.arange(64, dtype=np.float32).reshape(8, 8), {"method": "bm3d_bm4d", "axes": "yx"}
        )
