"""Honest runtime capability reporting."""

from __future__ import annotations

import importlib.metadata
import importlib.util

from .devices import runtime_devices
from .presets import PRESETS, ZS_N2N_EFFORT_PRESETS_VERSION


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def capabilities() -> dict:
    devices = runtime_devices()
    neural_devices = ["cpu"]
    if devices["cuda"]:
        neural_devices.append("cuda")
    if devices["mps"]:
        neural_devices.append("mps")
    bm3d = importlib.util.find_spec("bm3d") is not None
    bm4d = importlib.util.find_spec("bm4d") is not None
    return {
        "methods": {
            "bm3d_bm4d": {
                "available": bm3d,
                "devices": ["cpu"],
                "dimensions": (["2d"] if bm3d else []) + (["3d"] if bm4d else []),
                "bm3d_available": bm3d,
                "bm4d_available": bm4d,
                "license": "Tampere restricted non-commercial scientific/educational license",
                "versions": {"bm3d": _version("bm3d"), "bm4d": _version("bm4d")},
                "channel_modes": ["independent"],
            },
            "zs_n2n": {
                "effort_presets_version": ZS_N2N_EFFORT_PRESETS_VERSION,
                "efforts": list(PRESETS["zs_n2n"]),
                "available": True,
                "devices": neural_devices,
                "dimensions": ["2d"],
                "channel_modes": ["independent", "native_color"],
            },
            "noise2fast": {
                "available": True,
                "devices": neural_devices,
                "dimensions": ["2d"],
                "channel_modes": ["independent"],
            },
            "structn2v": {
                "available": True,
                "devices": neural_devices,
                "dimensions": ["2d", "2.5d", "3d"],
                "channel_modes": ["independent"],
            },
        },
        "cuda_available": devices["cuda"],
        "mps_available": devices["mps"],
    }
