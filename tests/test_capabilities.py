from __future__ import annotations

import importlib


def test_neural_capability_devices_follow_runtime(monkeypatch) -> None:
    module = importlib.import_module("jdll_denoise.capabilities")
    monkeypatch.setattr(module, "runtime_devices", lambda: {"cuda": False, "mps": True})
    result = module.capabilities()
    assert result["cuda_available"] is False
    assert result["mps_available"] is True
    assert result["methods"]["noise2fast"]["devices"] == ["cpu", "mps"]
    assert result["methods"]["bm3d_bm4d"]["devices"] == ["cpu"]
    assert result["methods"]["zs_n2n"]["channel_modes"] == [
        "independent",
        "native_color",
    ]
