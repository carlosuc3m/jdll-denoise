"""Optional adapter for the license-restricted official BM3D/BM4D packages."""

from __future__ import annotations

import importlib
import importlib.metadata
from typing import Any

import numpy as np
from scipy.ndimage import laplace

from ..cancellation import checkpoint
from ..errors import MethodUnavailableError, ThirdPartyBackendError
from .base import MethodContext, MethodResult


def estimate_noise_sigma(image: np.ndarray) -> tuple[float, str | None]:
    # A robust high-frequency MAD estimate.  The discrete Laplacian amplifies
    # independent Gaussian noise by sqrt(20) in 2D and sqrt(42) in 3D.
    high_frequency = laplace(image.astype(np.float64), mode="reflect")
    median = float(np.median(high_frequency))
    mad = float(np.median(np.abs(high_frequency - median)))
    kernel_gain = np.sqrt(20.0 if image.ndim == 2 else 42.0)
    sigma = mad / (0.6744897501960817 * kernel_gain)
    warning = None
    if not np.isfinite(sigma) or sigma <= 1e-6:
        sigma = 1e-3
        warning = (
            "automatic noise estimate was degenerate and was clamped to 0.001 normalized units"
        )
    elif sigma > 0.5:
        sigma = 0.5
        warning = "automatic noise estimate exceeded 0.5 and was conservatively clamped"
    return sigma, warning


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def run(image: np.ndarray, context: MethodContext) -> MethodResult:
    dimensions = context.config["dimensions"]
    package = "bm4d" if dimensions == "3d" else "bm3d"
    try:
        module = importlib.import_module(package)
    except ImportError as exc:
        raise MethodUnavailableError(
            f"method={package} device=cpu phase=preparing: optional package '{package}' is not "
            "installed; install jdll-denoise[bm] after accepting the Tampere license restriction"
        ) from exc
    checkpoint(context.cancellation, "estimating_noise")
    method_config = context.config["method_config"]
    sigma_value = method_config.get("sigma", "auto")
    warning = None
    if sigma_value == "auto":
        sigma, warning = estimate_noise_sigma(image)
    else:
        sigma = float(sigma_value)
    if warning:
        context.emitter.warning("estimating_noise", warning, method=context.config["method"])
    context.emitter.status(
        "estimating_noise",
        f"Using normalized noise sigma={sigma:.6g}",
        method=context.config["method"],
    )
    checkpoint(context.cancellation, "before_bm_call")
    context.emitter.status(
        "inference",
        f"Running {package.upper()}; cancellation is checked after this atomic stage",
        method=context.config["method"],
    )
    profile = method_config["profile"]
    stages = method_config["stages"]
    try:
        kwargs: dict[str, Any] = {"sigma_psd": sigma, "profile": profile}
        if stages == "hard_thresholding":
            stage_enum = getattr(module, f"{package.upper()}Stages", None)
            if stage_enum is not None:
                kwargs["stage_arg"] = stage_enum.HARD_THRESHOLDING
        function = getattr(module, package)
        output = function(image.astype(np.float64, copy=False), **kwargs)
    except Exception as exc:
        raise ThirdPartyBackendError(
            f"method={package} device=cpu phase=inference: official backend failed: {exc}; "
            "verify the optional package version and input size"
        ) from exc
    checkpoint(context.cancellation, "after_bm_call")
    result = np.ascontiguousarray(output, dtype=np.float32)
    if result.shape != image.shape or not np.isfinite(result).all():
        raise ThirdPartyBackendError(
            f"method={package} device=cpu phase=inference: backend returned invalid shape or values"
        )
    return MethodResult(
        result,
        {
            "third_party_versions": {package: _package_version(package)},
            "bm_profile": profile,
            "bm_stages": stages,
            "noise_sigma_normalized": sigma,
        },
    )
