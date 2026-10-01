from __future__ import annotations

import numpy as np
import pytest
from conftest import tiny_config

from jdll_denoise import DenoisingCancelled, denoise


class CountingCancellation:
    def __init__(self, limit: int):
        self.calls = 0
        self.limit = limit

    def is_cancelled(self) -> bool:
        self.calls += 1
        return self.calls >= self.limit


def test_cancel_during_neural_work_and_run_again() -> None:
    source = np.random.default_rng(5).random((8, 8), dtype=np.float32)
    with pytest.raises(DenoisingCancelled):
        denoise(source, tiny_config("noise2fast"), cancellation=CountingCancellation(5))
    assert denoise(source, tiny_config("noise2fast"))["denoised"].shape == source.shape


def test_cancel_before_validation() -> None:
    with pytest.raises(DenoisingCancelled):
        denoise(np.ones((8, 8)), tiny_config("zs_n2n"), cancellation=lambda: True)
