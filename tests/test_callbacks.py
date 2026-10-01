from __future__ import annotations

import numpy as np
from conftest import tiny_config

from jdll_denoise import denoise


def test_callback_schema_and_order() -> None:
    events = []
    denoise(
        np.arange(64, dtype=np.float32).reshape(8, 8),
        tiny_config("noise2fast"),
        callbacks=events.append,
    )
    assert events[0]["phase"] == "validating"
    assert any(event["type"] == "resolved_config" for event in events)
    assert events[-1]["phase"] == "completed"
    required = {"type", "phase", "method", "message", "current", "maximum", "metrics", "info"}
    assert all(required <= event.keys() for event in events)


def test_callback_failure_is_isolated_and_reported() -> None:
    def broken(event):
        raise RuntimeError("bridge disappeared")

    result = denoise(
        np.arange(64, dtype=np.float32).reshape(8, 8), tiny_config("zs_n2n"), callbacks=broken
    )
    assert result["denoised"].shape == (8, 8)
    assert any("bridge disappeared" in warning for warning in result["metadata"]["warnings"])
