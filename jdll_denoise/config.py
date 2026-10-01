"""Configuration validation and automatic preset resolution."""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any

from .devices import resolve_device
from .errors import ConfigurationError
from .presets import PRESETS, ZS_N2N_EFFORT_PRESETS_VERSION, method_preset

METHODS = ("bm3d_bm4d", "zs_n2n", "noise2fast", "structn2v")
DEVICES = ("cpu", "cuda", "mps", "auto")
DIMENSIONS = ("auto", "2d", "2.5d", "3d")
AXES = set("tczyx")

_COMMON_KEYS = {
    "method",
    "effort",
    "device",
    "dimensions",
    "axes",
    "seed",
    "normalization",
    "tiling",
    "method_config",
    "allow_cpu_fallback",
    "channel_mode",
    "extensions",
}
_NORMALIZATION_KEYS = {"mode", "low", "high", "per_channel", "clip"}
_TILING_KEYS = {"enabled", "tile_size", "overlap", "blend", "batch_size"}
_METHOD_KEYS = {
    "bm3d_bm4d": {"sigma", "profile", "stages"},
    "zs_n2n": {
        "steps",
        "learning_rate",
        "hidden_width",
        "middle_groups",
        "output_init",
        "output_init_std",
        "batch_pairs",
        # Backward-compatible advanced alias. Standard presets never use it.
        "features",
        "scheduler_step_size",
        "scheduler_gamma",
        "report_interval",
        "early_stopping",
        "patience",
        "min_delta",
        "restore_best_weights",
    },
    "noise2fast": {
        "steps",
        "report_interval",
        "hidden_width",
        "patience",
    },
    "structn2v": {
        "noise_structure",
        "mask_fraction",
        "correlation_radius",
        "period",
        "architecture",
        "patch_size",
        "patch_count",
        "batch_size",
        "learning_rate",
        "steps",
        "epochs",
        "report_interval",
        "neighborhood_radius",
        "selection_steps",
    },
}


def _enum(name: str, value: Any, accepted: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in accepted:
        raise ConfigurationError(f"{name} must be one of {', '.join(accepted)}; got {value!r}")
    return value


def validate_axes(axes: Any, ndim: int | None = None) -> str:
    if not isinstance(axes, str):
        raise ConfigurationError("axes must be a string such as 'yx', 'cyx', or 'tczyx'")
    axes = axes.lower()
    if ndim is not None and len(axes) != ndim:
        raise ConfigurationError(
            f"axes={axes!r} has {len(axes)} entries but input has {ndim} dimensions"
        )
    if len(set(axes)) != len(axes) or not set(axes) <= AXES:
        raise ConfigurationError("axes may contain each of t, c, z, y, x at most once")
    if "y" not in axes or "x" not in axes:
        raise ConfigurationError("axes must explicitly contain spatial y and x axes")
    return axes


def _default_axes(input_shape: tuple[int, ...] | None) -> str:
    if input_shape is None:
        return "cyx"
    if len(input_shape) == 2:
        return "yx"
    if len(input_shape) == 3:
        return "cyx"
    raise ConfigurationError(
        "axes is required for arrays with four or more dimensions because C/Z/T are ambiguous"
    )


def _resolve_dimension(
    method: str, requested: str, axes: str, shape: tuple[int, ...] | None
) -> str:
    has_volume = "z" in axes and (shape is None or shape[axes.index("z")] > 1)
    if requested == "auto":
        if not has_volume:
            return "2d"
        if method == "bm3d_bm4d":
            return "3d"
        if method in {"zs_n2n", "noise2fast"}:
            return "2d"
        depth = shape[axes.index("z")] if shape is not None else 8
        return "3d" if depth >= 8 else "2.5d"
    if requested in {"2.5d", "3d"} and not has_volume:
        raise ConfigurationError(
            f"dimensions={requested} requires a z axis with depth greater than one"
        )
    supported = {
        "bm3d_bm4d": {"2d", "3d"},
        "zs_n2n": {"2d"},
        "noise2fast": {"2d"},
        "structn2v": {"2d", "2.5d", "3d"},
    }[method]
    if requested not in supported:
        raise ConfigurationError(f"method={method} does not support dimensions={requested}")
    return requested


def resolve_config(
    config: dict[str, Any], input_shape: tuple[int, ...] | None = None
) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ConfigurationError("config must be a dictionary")
    source = deepcopy(config)
    unknown = set(source) - _COMMON_KEYS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ConfigurationError(
            f"unknown configuration key(s): {names}; place future keys in extensions"
        )

    method = _enum("method", source.get("method", "noise2fast"), METHODS)
    requested_effort = source.get("effort", "balanced")
    if method == "zs_n2n" and requested_effort == "balanced-high":
        requested_effort = "balanced_high"
    effort = _enum("effort", requested_effort, tuple(PRESETS[method]))
    requested_device = _enum("device", source.get("device", "cpu"), DEVICES)
    requested_dimensions = _enum("dimensions", source.get("dimensions", "auto"), DIMENSIONS)
    axes_value = source["axes"] if "axes" in source else _default_axes(input_shape)
    axes = validate_axes(axes_value, None if input_shape is None else len(input_shape))
    allow_fallback = source.get("allow_cpu_fallback", False)
    if not isinstance(allow_fallback, bool):
        raise ConfigurationError("allow_cpu_fallback must be true or false")
    device, device_warning = resolve_device(
        requested_device, neural=method != "bm3d_bm4d", allow_cpu_fallback=allow_fallback
    )
    dimensions = _resolve_dimension(method, requested_dimensions, axes, input_shape)

    seed = source.get("seed", 5489)
    if not isinstance(seed, int) or seed < 0:
        raise ConfigurationError("seed must be a non-negative integer")

    norm: dict[str, Any] = {
        "mode": "percentile",
        "low": 0.0 if method in {"noise2fast", "zs_n2n"} else 0.1,
        "high": 100.0 if method in {"noise2fast", "zs_n2n"} else 99.9,
        "per_channel": True,
        "clip": method != "structn2v",
    }
    supplied_norm = source.get("normalization", {})
    if not isinstance(supplied_norm, dict):
        raise ConfigurationError("normalization must be an object")
    norm_unknown = set(supplied_norm) - _NORMALIZATION_KEYS
    if norm_unknown:
        raise ConfigurationError(f"unknown normalization key(s): {', '.join(sorted(norm_unknown))}")
    norm.update(supplied_norm)
    if norm["mode"] != "percentile":
        raise ConfigurationError("normalization.mode currently supports only 'percentile'")
    if not (0 <= float(norm["low"]) < float(norm["high"]) <= 100):
        raise ConfigurationError("normalization percentiles must satisfy 0 <= low < high <= 100")
    if norm["per_channel"] is not True:
        raise ConfigurationError(
            "v1 requires normalization.per_channel=true to preserve biological channels"
        )

    tiling = {
        "enabled": "auto",
        "tile_size": 512,
        "overlap": 32,
        "blend": "gaussian",
        "batch_size": "auto",
    }
    supplied_tiling = source.get("tiling", {})
    if not isinstance(supplied_tiling, dict):
        raise ConfigurationError("tiling must be an object")
    tiling_unknown = set(supplied_tiling) - _TILING_KEYS
    if tiling_unknown:
        raise ConfigurationError(f"unknown tiling key(s): {', '.join(sorted(tiling_unknown))}")
    tiling.update(supplied_tiling)
    if tiling["enabled"] == "auto":
        spatial_sizes = [input_shape[axes.index(axis)] for axis in "yx"] if input_shape else [0, 0]
        tiling["enabled"] = False if method == "noise2fast" else max(spatial_sizes) > 1024
    if tiling["tile_size"] == "auto":
        tiling["tile_size"] = 512
    if tiling["overlap"] == "auto":
        tiling["overlap"] = 32
    if tiling["batch_size"] == "auto":
        tiling["batch_size"] = 4 if device in {"cuda", "mps"} else 1
    if not isinstance(tiling["enabled"], bool):
        raise ConfigurationError("tiling.enabled must be true, false, or 'auto'")
    if method == "noise2fast" and tiling["enabled"]:
        raise ConfigurationError(
            "Noise2Fast inference tiling is not part of the published reconstruction; "
            "tiling.enabled must be false for paper fidelity"
        )
    if not isinstance(tiling["tile_size"], int) or tiling["tile_size"] < 32:
        raise ConfigurationError("tiling.tile_size must be an integer >= 32")
    if not isinstance(tiling["overlap"], int) or not 0 <= tiling["overlap"] < tiling["tile_size"]:
        raise ConfigurationError("tiling.overlap must be >= 0 and smaller than tile_size")
    if tiling["blend"] not in {"gaussian", "linear"}:
        raise ConfigurationError("tiling.blend must be gaussian or linear")
    if not isinstance(tiling["batch_size"], int) or tiling["batch_size"] < 1:
        raise ConfigurationError("tiling.batch_size must be a positive integer or 'auto'")

    method_config = method_preset(method, effort)
    supplied_method = source.get("method_config", {})
    if not isinstance(supplied_method, dict):
        raise ConfigurationError("method_config must be an object")
    method_unknown = set(supplied_method) - _METHOD_KEYS[method]
    if method_unknown:
        raise ConfigurationError(
            f"unknown method_config key(s) for {method}: {', '.join(sorted(method_unknown))}"
        )
    method_config.update(supplied_method)
    if method == "zs_n2n" and "features" in supplied_method:
        for key in ("features", "hidden_width"):
            if key in supplied_method:
                value = supplied_method[key]
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ConfigurationError(f"method_config.{key} must be a positive integer")
        if (
            "hidden_width" in supplied_method
            and supplied_method["hidden_width"] != supplied_method["features"]
        ):
            raise ConfigurationError("method_config.features and hidden_width must agree")
        method_config["hidden_width"] = supplied_method["features"]
    # Resolve StructN2V to the two schedules reported in the paper.  A positive
    # explicit `steps` value remains available for deterministic test/debug
    # runs, and is reported as a non-canonical override by the implementation.
    if method == "structn2v":
        volumetric = dimensions in {"2.5d", "3d"}
        automatic = {
            "mask_fraction": 0.02,
            "correlation_radius": 8 if volumetric else 4,
            "period": 4,
            "architecture": "paper_unet",
            "patch_size": [32, 64, 64] if volumetric else [256, 256],
            "patch_count": 1900 if volumetric else 1600,
            "batch_size": 10 if volumetric else 4,
            "learning_rate": 4e-6 if volumetric else 2e-5,
            "epochs": 100 if volumetric else 600,
            "steps": 0,
            "neighborhood_radius": 5,
            "selection_steps": 8,
        }
        for key, value in automatic.items():
            if method_config.get(key, "auto") == "auto":
                method_config[key] = value
    if "steps" in method_config:
        minimum = 0 if method in {"noise2fast", "structn2v"} else 1
        if not isinstance(method_config["steps"], int) or method_config["steps"] < minimum:
            qualifier = "a non-negative" if minimum == 0 else "a positive"
            raise ConfigurationError(f"method_config.steps must be {qualifier} integer")
    if method == "zs_n2n":
        method_config.setdefault("report_interval", 10)
        method_config.setdefault("early_stopping", False)
        method_config.setdefault("restore_best_weights", False)
        for key in (
            "hidden_width",
            "middle_groups",
            "steps",
            "scheduler_step_size",
            "report_interval",
            "patience",
        ):
            if key not in method_config:
                continue
            value = method_config[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigurationError(f"method_config.{key} must be a positive integer")
        if method_config["hidden_width"] % method_config["middle_groups"]:
            raise ConfigurationError("method_config.middle_groups must divide hidden_width")
        method_config["output_init"] = _enum(
            "method_config.output_init", method_config["output_init"], ("native", "near_zero")
        )
        for key in ("batch_pairs", "early_stopping", "restore_best_weights"):
            if not isinstance(method_config[key], bool):
                raise ConfigurationError(f"method_config.{key} must be true or false")
        for key in ("output_init_std", "learning_rate", "scheduler_gamma", "min_delta"):
            if key not in method_config:
                continue
            value = method_config[key]
            qualifier = "non-negative" if key == "min_delta" else "positive"
            message = f"method_config.{key} must be a finite {qualifier} number"
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigurationError(message)
            try:
                number = float(value)
            except OverflowError:
                raise ConfigurationError(message) from None
            if not isfinite(number) or (number < 0 if key == "min_delta" else number <= 0):
                raise ConfigurationError(message)
            method_config[key] = number
        if method_config["scheduler_gamma"] > 1:
            raise ConfigurationError("method_config.scheduler_gamma must be in (0, 1]")
    if method == "noise2fast":
        method_config.setdefault("steps", 0)
        method_config.setdefault("report_interval", 10)
        for key in ("hidden_width", "patience", "report_interval", "steps"):
            value = method_config[key]
            minimum = 0 if key == "steps" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                qualifier = "a non-negative" if minimum == 0 else "a positive"
                raise ConfigurationError(
                    f"method_config.{key} must be {qualifier} integer; got {value!r}"
                )
        method_config["output_window"] = 100
    if method == "bm3d_bm4d":
        method_config.setdefault("sigma", "auto")
        if method_config["sigma"] != "auto" and (
            not isinstance(method_config["sigma"], (int, float)) or method_config["sigma"] <= 0
        ):
            raise ConfigurationError("method_config.sigma must be 'auto' or a positive number")
    if method == "structn2v":
        method_config.setdefault("noise_structure", "auto")
        if method_config["noise_structure"] not in {
            "auto",
            "local_isotropic",
            "horizontal",
            "vertical",
            "periodic_grid",
        }:
            raise ConfigurationError("invalid StructN2V noise_structure")
        method_config.setdefault("mask_fraction", 0.02)
        method_config.setdefault("correlation_radius", 4)
        method_config.setdefault("period", 4)
        method_config.setdefault("architecture", "paper_unet")
        if method_config["architecture"] != "paper_unet":
            raise ConfigurationError("StructN2V architecture must be 'paper_unet'")
        if not 0 < float(method_config["mask_fraction"]) < 1:
            raise ConfigurationError("method_config.mask_fraction must be between zero and one")
        if int(method_config["correlation_radius"]) < 1:
            raise ConfigurationError("method_config.correlation_radius must be positive")
        if int(method_config["period"]) < 1:
            raise ConfigurationError("method_config.period must be positive")
        patch_size = method_config["patch_size"]
        expected_dims = 2 if dimensions == "2d" else 3
        if isinstance(patch_size, int):
            patch_size = [patch_size] * expected_dims
        if (
            not isinstance(patch_size, (list, tuple))
            or len(patch_size) != expected_dims
            or not all(isinstance(value, int) and value >= 1 for value in patch_size)
        ):
            raise ConfigurationError(
                f"method_config.patch_size must contain {expected_dims} positive integers"
            )
        method_config["patch_size"] = list(patch_size)
        integer_keys = (
            "patch_count",
            "batch_size",
            "epochs",
            "neighborhood_radius",
            "selection_steps",
        )
        for key in integer_keys:
            if not isinstance(method_config[key], int) or method_config[key] < 1:
                raise ConfigurationError(f"method_config.{key} must be a positive integer")
    method_config.setdefault("report_interval", 10)

    channel_mode = source.get("channel_mode", "auto")
    if channel_mode not in {"auto", "independent", "native_color"}:
        raise ConfigurationError("channel_mode must be auto, independent, or native_color")
    if channel_mode == "native_color" and method != "zs_n2n":
        raise ConfigurationError(
            "channel_mode=native_color is currently validated only for ZS-N2N; "
            "use independent for this method"
        )

    return {
        "method": method,
        **({"effort_presets_version": ZS_N2N_EFFORT_PRESETS_VERSION} if method == "zs_n2n" else {}),
        "effort": effort,
        "device": device,
        "device_requested": requested_device,
        "device_warning": device_warning,
        "dimensions": dimensions,
        "dimensions_requested": requested_dimensions,
        "axes": axes,
        "seed": seed,
        "normalization": norm,
        "tiling": tiling,
        "method_config": method_config,
        "allow_cpu_fallback": allow_fallback,
        "channel_mode": channel_mode,
        "extensions": _extensions(source.get("extensions", {})),
        "estimated_memory_bytes": _estimate_memory(input_shape, method, dimensions),
        "cancellation_granularity": "stage" if method == "bm3d_bm4d" else "optimization_step",
    }


def _estimate_memory(shape: tuple[int, ...] | None, method: str, dimensions: str) -> int | None:
    if shape is None:
        return None
    elements = 1
    for value in shape:
        elements *= value
    multiplier = 6 if method == "bm3d_bm4d" else (32 if dimensions == "3d" else 18)
    return int(elements * 4 * multiplier)


def _extensions(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError("extensions must be an object")
    return deepcopy(value)
