from __future__ import annotations

import pytest

from jdll_denoise import ConfigurationError, resolve_config
from jdll_denoise.presets import PRESETS


def test_default_method_is_noise2fast() -> None:
    resolved = resolve_config({}, (32, 32))
    assert resolved["method"] == "noise2fast"
    assert resolved["effort"] == "balanced"
    assert resolved["axes"] == "yx"


@pytest.mark.parametrize("method", PRESETS)
def test_step_presets_are_monotonic(method: str) -> None:
    presets = PRESETS[method]
    if "steps" in presets["quick"]:
        assert presets["quick"]["steps"] < presets["balanced"]["steps"]
        assert presets["balanced"]["steps"] <= presets["thorough"]["steps"]


def test_unknown_key_is_actionable() -> None:
    with pytest.raises(ConfigurationError, match="unknown configuration key.*surprise"):
        resolve_config({"surprise": 1}, (8, 8))


def test_extensions_are_retained_without_mutating_input() -> None:
    config = {"extensions": {"future": [1, 2]}}
    resolved = resolve_config(config, (8, 8))
    resolved["extensions"]["future"].append(3)
    assert config == {"extensions": {"future": [1, 2]}}


def test_ambiguous_rank_requires_axes() -> None:
    with pytest.raises(ConfigurationError, match="axes is required"):
        resolve_config({}, (2, 3, 8, 8))


def test_unsupported_dimension_fails() -> None:
    with pytest.raises(ConfigurationError, match="does not support dimensions=3d"):
        resolve_config({"method": "noise2fast", "axes": "zyx", "dimensions": "3d"}, (3, 8, 8))


def test_struct_auto_values_become_concrete() -> None:
    config = {
        "method": "structn2v",
        "axes": "zyx",
        "dimensions": "3d",
        "method_config": {
            "steps": "auto",
            "learning_rate": "auto",
            "mask_fraction": "auto",
            "correlation_radius": "auto",
            "period": "auto",
            "patch_size": "auto",
            "batch_size": "auto",
            "architecture": "auto",
        },
    }
    resolved = resolve_config(config, (8, 16, 16))
    for key in config["method_config"]:
        assert resolved["method_config"][key] != "auto"


def test_native_color_is_only_exposed_for_zs_n2n() -> None:
    resolved = resolve_config(
        {"method": "zs_n2n", "axes": "cyx", "channel_mode": "native_color"},
        (3, 8, 8),
    )
    assert resolved["channel_mode"] == "native_color"
    with pytest.raises(ConfigurationError, match="validated only for ZS-N2N"):
        resolve_config(
            {"method": "noise2fast", "axes": "cyx", "channel_mode": "native_color"},
            (3, 8, 8),
        )
