"""Array-first public orchestration API."""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

import numpy as np

from .callbacks import Callbacks, EventEmitter
from .cancellation import CancellationLike, checkpoint
from .cleanup import cleanup_device
from .config import resolve_config
from .errors import DenoisingCancelled, DenoisingError, InputFormatError
from .io import ArrayLayout
from .methods import METHODS
from .methods.base import MethodContext
from .normalization import Normalizer

BACKEND_VERSION = "0.1.0"


def _dimension_label(config: dict[str, Any], axes: str, shape: tuple[int, ...]) -> str:
    has_stack = "z" in axes and shape[axes.index("z")] > 1
    if has_stack and config["method"] in {"zs_n2n", "noise2fast"}:
        return "slice_wise_2d"
    return config["dimensions"]


def _run(
    image: object,
    config: dict[str, Any],
    *,
    callbacks: Callbacks = None,
    cancellation: CancellationLike = None,
    preview_mode: bool = False,
) -> dict[str, Any]:
    api_started = time.perf_counter()
    emitter = EventEmitter(callbacks)
    resolved: dict[str, Any] | None = None
    warnings: list[str] = []
    try:
        checkpoint(cancellation, "before_validation")
        emitter.status("validating", "Validating input array and configuration")
        try:
            shape = tuple(np.asarray(image).shape)
        except Exception as exc:
            raise InputFormatError(
                f"phase=validating: input cannot be converted to an array: {exc}"
            ) from exc
        resolved = resolve_config(deepcopy(config), shape)
        checkpoint(cancellation, "resolving_config")
        emitter.emit(
            "resolved_config",
            "resolving_config",
            "Resolved denoising configuration",
            method=resolved["method"],
            info={"resolved_config": deepcopy(resolved)},
        )
        if resolved["device_warning"]:
            warnings.append(resolved["device_warning"])
            emitter.warning(
                "resolving_config", resolved["device_warning"], method=resolved["method"]
            )
        layout = ArrayLayout.from_input(image, resolved["axes"])
        source = layout.array
        joint_channels = resolved["channel_mode"] == "native_color"
        if joint_channels:
            if "c" not in resolved["axes"]:
                raise InputFormatError("channel_mode=native_color requires an explicit c axis")
            if source.shape[resolved["axes"].index("c")] != 3:
                raise InputFormatError(
                    "channel_mode=native_color currently requires exactly three RGB channels"
                )
        checkpoint(cancellation, "before_output_allocation")
        output = np.empty(source.shape, dtype=np.float32)
        normalizations: list[dict[str, Any]] = []
        jobs_metadata: list[dict[str, Any]] = []
        channel_jobs = (
            1
            if joint_channels
            else (source.shape[resolved["axes"].index("c")] if "c" in resolved["axes"] else 1)
        )
        total_jobs = channel_jobs * (
            source.shape[resolved["axes"].index("t")] if "t" in resolved["axes"] else 1
        )
        completed_jobs = 0
        jobs = layout.joint_channel_jobs() if joint_channels else layout.jobs()
        for job in jobs:
            checkpoint(cancellation, "between_channels_or_timepoints")
            job_message = (
                f"Preparing channel {job.channel + 1}/{job.channels}, "
                f"time {job.time + 1}/{job.times}"
            )
            emitter.status(
                "preparing",
                job_message,
                method=resolved["method"],
                channel=job.channel + 1,
                channels=job.channels,
            )
            if joint_channels:
                channel_normalizers = [
                    Normalizer.fit(job.data[channel], resolved["normalization"])
                    for channel in range(job.channels)
                ]
                for channel, channel_normalizer in enumerate(channel_normalizers):
                    normalizations.append(
                        {
                            "channel": channel,
                            "time": job.time,
                            **channel_normalizer.metadata(),
                        }
                    )
                normalized = np.stack(
                    [
                        channel_normalizer.normalize(job.data[channel])
                        for channel, channel_normalizer in enumerate(channel_normalizers)
                    ]
                )
                constant_job = all(item.constant for item in channel_normalizers)
            else:
                normalizer = Normalizer.fit(job.data, resolved["normalization"])
                normalizations.append(
                    {"channel": job.channel, "time": job.time, **normalizer.metadata()}
                )
                normalized = normalizer.normalize(job.data)
                constant_job = normalizer.constant
            context = MethodContext(
                config=deepcopy(resolved),
                emitter=emitter,
                cancellation=cancellation,
                seed=resolved["seed"] + job.time * 1_000_003 + job.channel * 1_009,
                channel=job.channel,
                channels=job.channels,
                time=job.time,
                times=job.times,
            )
            if constant_job and resolved["method"] != "bm3d_bm4d":
                denoised_normalized = normalized.copy()
                method_metadata = {"constant_short_circuit": True}
            else:
                method_result = METHODS[resolved["method"]](normalized, context)
                denoised_normalized = method_result.data
                method_metadata = method_result.metadata
            if (
                denoised_normalized.shape != job.data.shape
                or not np.isfinite(denoised_normalized).all()
            ):
                raise DenoisingError(
                    f"method={resolved['method']} device={resolved['device']} phase=inference: "
                    "method returned a mismatched shape or non-finite values"
                )
            if joint_channels:
                restored = np.stack(
                    [
                        channel_normalizer.invert(denoised_normalized[channel])
                        for channel, channel_normalizer in enumerate(channel_normalizers)
                    ]
                )
                layout.put_joint_channels(output, job, restored)
            else:
                restored = normalizer.invert(denoised_normalized)
                layout.put(output, job, restored)
            jobs_metadata.append(
                {"channel": job.channel, "time": job.time, "seed": context.seed, **method_metadata}
            )
            warnings.extend(_nested_values(method_metadata, "warnings"))
            warnings.extend(_nested_values(method_metadata, "fidelity_warnings"))
            completed_jobs += 1
            emitter.progress(
                "finalizing",
                f"Completed channel/time job {completed_jobs}/{total_jobs}",
                completed_jobs,
                total_jobs,
                method=resolved["method"],
                channel=job.channel + 1,
                channels=job.channels,
            )
        checkpoint(cancellation, "before_return")
        output = np.ascontiguousarray(output, dtype=np.float32)
        output_axes = resolved["axes"]
        preview_depth = source.shape[resolved["axes"].index("z")] if "z" in resolved["axes"] else 1
        if preview_mode and preview_depth > 1:
            z_axis = resolved["axes"].index("z")
            output = np.ascontiguousarray(
                np.take(output, (preview_depth - 1) // 2, axis=z_axis), dtype=np.float32
            )
            output_axes = output_axes.replace("z", "")
        warnings.extend(emitter.failures)
        actual_devices = _nested_values({"jobs": jobs_metadata}, "device_used")
        devices_used = list(dict.fromkeys(str(item) for item in actual_devices))
        device_used = (
            devices_used[0]
            if len(devices_used) == 1
            else ("mixed" if devices_used else str(resolved["device"]))
        )
        metadata: dict[str, Any] = {
            "backend": "jdll-denoise",
            "backend_version": BACKEND_VERSION,
            "method": resolved["method"],
            "effort": resolved["effort"],
            "device_requested": resolved["device_requested"],
            "device_resolved": resolved["device"],
            "device_used": device_used,
            "devices_used": devices_used or [str(resolved["device"])],
            "dimension_mode": _dimension_label(resolved, resolved["axes"], source.shape),
            "dimensions": _dimension_label(resolved, resolved["axes"], source.shape),
            "axes": resolved["axes"],
            "output_axes": output_axes,
            "input_shape": list(source.shape),
            "output_shape": list(output.shape),
            "source_dtype": str(source.dtype),
            "source_intensity_range": [float(source.min()), float(source.max())],
            "channels": source.shape[resolved["axes"].index("c")] if "c" in resolved["axes"] else 1,
            "channel_mode": "native_color" if joint_channels else "independent",
            "normalization": normalizations,
            "resolved_config": deepcopy(resolved),
            "seed": resolved["seed"],
            "warnings": warnings,
            "jobs": jobs_metadata,
            "third_party_versions": _merge_versions(jobs_metadata),
            "phase_timings_seconds": {
                "api_end_to_end": time.perf_counter() - api_started,
            },
        }
        if preview_mode:
            # Preview uses the same resolved budget and architecture as denoise.
            metadata.update(
                {
                    "cache_reusable_for_full_input": preview_depth == 1,
                    "preview_region": {
                        "z_start": 0,
                        "z_end": preview_depth,
                        "center_z": (preview_depth - 1) // 2,
                    },
                    "preview_is_representative_only": preview_depth > 1,
                }
            )
        callback_failures_before_completion = len(emitter.failures)
        if resolved["method"] == "zs_n2n":
            metadata["effort_presets_version"] = resolved["effort_presets_version"]
        emitter.progress("completed", "Denoising completed", 1, 1, method=resolved["method"])
        warnings.extend(emitter.failures[callback_failures_before_completion:])
        metadata["phase_timings_seconds"]["api_end_to_end"] = time.perf_counter() - api_started
        return {"denoised": output, "metadata": metadata}
    except DenoisingCancelled:
        emitter.status(
            "cancelled",
            "Denoising cancelled; no partial result returned",
            method=resolved["method"] if resolved else None,
        )
        raise
    except Exception as exc:
        emitter.status(
            "failed",
            f"Denoising failed: {exc}",
            method=resolved["method"] if resolved else None,
        )
        raise
    finally:
        cleanup_device(resolved["device"] if resolved else None)


def _merge_versions(jobs: list[dict[str, Any]]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for job in jobs:
        versions.update(job.get("third_party_versions", {}))
    return versions


def _nested_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for nested_key, nested_value in value.items():
            if nested_key == key:
                if isinstance(nested_value, list) and key.endswith("warnings"):
                    found.extend(nested_value)
                else:
                    found.append(nested_value)
            else:
                found.extend(_nested_values(nested_value, key))
    elif isinstance(value, list):
        for nested_value in value:
            found.extend(_nested_values(nested_value, key))
    return found


def denoise(
    image: object,
    config: dict[str, Any],
    *,
    callbacks: Callbacks = None,
    cancellation: CancellationLike = None,
) -> dict[str, Any]:
    return _run(image, config, callbacks=callbacks, cancellation=cancellation)


def preview(
    image: object,
    config: dict[str, Any],
    *,
    callbacks: Callbacks = None,
    cancellation: CancellationLike = None,
) -> dict[str, Any]:
    return _run(image, config, callbacks=callbacks, cancellation=cancellation, preview_mode=True)
