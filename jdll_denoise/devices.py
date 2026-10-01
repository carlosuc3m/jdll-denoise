"""Runtime device discovery and resolution."""

from __future__ import annotations

import torch

from .errors import DeviceUnavailableError


def runtime_devices() -> dict[str, bool]:
    return {
        "cuda": bool(torch.cuda.is_available()),
        "mps": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    }


def resolve_device(
    requested: str, *, neural: bool, allow_cpu_fallback: bool
) -> tuple[str, str | None]:
    available = runtime_devices()
    if not neural:
        if requested in {"cpu", "auto"}:
            return "cpu", None
        if allow_cpu_fallback:
            return "cpu", f"{requested} is unsupported by the classical BM adapter; using CPU"
        raise DeviceUnavailableError(
            f"method=bm3d_bm4d device={requested} phase=resolving_config: "
            "the BM adapter is CPU-only; select cpu or enable allow_cpu_fallback"
        )
    if requested == "auto":
        if available["cuda"]:
            return "cuda", None
        if available["mps"]:
            return "mps", None
        return "cpu", None
    if requested == "cpu" or available.get(requested, False):
        return requested, None
    if allow_cpu_fallback:
        return "cpu", f"requested {requested} device is unavailable; using CPU"
    raise DeviceUnavailableError(
        f"device={requested} phase=resolving_config: device is unavailable; "
        "select cpu/auto or enable allow_cpu_fallback"
    )
