"""Noise2Fast single-image denoising with configurable width and patience.

The four directed checkerboard pairs and feed-forward CNN follow Lequyer et
al., *Noise2Fast: Fast Self-Supervised Single Image Blind Denoising* (2022).
JDLL effort presets vary hidden width and validation patience while retaining
the existing training loop and 100-prediction reconstruction. Width 64 uses the
released architecture; other widths are variants with the same layer structure.
"""

from __future__ import annotations

import time
from collections import deque
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..cancellation import checkpoint
from ..cleanup import cleanup_device
from ..errors import OutOfMemoryDenoisingError
from .base import MethodContext, MethodResult, seed_everything


class _DeviceOOM(RuntimeError):
    pass


class Noise2FastBlock(nn.Module):
    """Two 3x3 convolutions and ReLUs at the requested hidden width."""

    def __init__(self, in_channels: int, hidden_width: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_width, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_width, hidden_width, kernel_size=3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = F.relu(self.conv1(value))
        return F.relu(self.conv2(value))


class Noise2FastNet(nn.Module):
    """Eight 3x3 convolutions, a 1x1 sigmoid head, and a 17x17 receptive field.

    Direct callers retain the released 64-feature constructor default. The
    public denoise/preview paths explicitly pass their resolved preset width.
    """

    def __init__(self, hidden_width: int = 64) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            Noise2FastBlock(1, hidden_width),
            Noise2FastBlock(hidden_width, hidden_width),
            Noise2FastBlock(hidden_width, hidden_width),
            Noise2FastBlock(hidden_width, hidden_width),
        )
        self.output = nn.Conv2d(hidden_width, 1, kernel_size=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.output(self.blocks(value)))


def checkerboard_pairs(
    value: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Construct the four directed up/left pairs from paper equations 18--21.

    The reference implementation removes the final row and/or column when an
    input dimension is odd.  This crop is used only for training; inference is
    performed on the original full-sized plane.
    """

    if value.ndim != 4:
        raise ValueError("Noise2Fast checkerboard input must have shape N,C,Y,X")
    height = value.shape[-2] - value.shape[-2] % 2
    width = value.shape[-1] - value.shape[-1] % 2
    if height < 2 or width < 2:
        raise ValueError("Noise2Fast requires both spatial dimensions to be at least 2 pixels")
    cropped = value[..., :height, :width]

    even_rows = cropped[..., 0::2, :]
    odd_rows = cropped[..., 1::2, :]
    column_parity = (torch.arange(width, device=value.device) % 2 == 0)[None, None, None]
    squeeze_up_odd = torch.where(column_parity, odd_rows, even_rows)
    squeeze_up_even = torch.where(column_parity, even_rows, odd_rows)

    even_columns = cropped[..., :, 0::2]
    odd_columns = cropped[..., :, 1::2]
    row_parity = (torch.arange(height, device=value.device) % 2 == 0)[None, None, :, None]
    squeeze_left_odd = torch.where(row_parity, odd_columns, even_columns)
    squeeze_left_even = torch.where(row_parity, even_columns, odd_columns)

    return (
        (squeeze_up_even, squeeze_up_odd),
        (squeeze_up_odd, squeeze_up_even),
        (squeeze_left_odd, squeeze_left_even),
        (squeeze_left_even, squeeze_left_odd),
    )


def checkerboard_views(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility helper returning the first complementary checkerboard pair."""

    return checkerboard_pairs(value)[0]


def _is_accelerator_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )


def _run_plane_on_device(
    image: np.ndarray, context: MethodContext, plane_index: int, device: str
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train one fresh model and average the last 100 available predictions."""

    seed = context.seed + plane_index
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    cfg = context.config["method_config"]
    hidden_width = int(cfg["hidden_width"])
    patience = int(cfg["patience"])
    output_window = int(cfg["output_window"])
    maximum_steps = int(cfg["steps"])
    report_interval = max(1, int(cfg["report_interval"]))
    started = time.perf_counter()

    context.emitter.status(
        "preparing",
        f"Noise2Fast {context.config['effort'].capitalize()}: {hidden_width} features, "
        f"patience {patience} steps, averaging {output_window} predictions, "
        f"learning rate 0.001, device {device}.",
        method="noise2fast",
        channel=context.channel + 1,
        channels=context.channels,
        z=plane_index,
    )
    model = Noise2FastNet(hidden_width=hidden_width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    full = torch.from_numpy(np.ascontiguousarray(image))[None, None].to(
        device=device, dtype=torch.float32
    )
    pairs = checkerboard_pairs(full)
    recent: deque[np.ndarray] = deque(maxlen=output_window)
    best_validation = -float("inf")
    best_step = 0
    stale = 0
    step = 0
    last_loss = float("nan")
    smooth_loss = float("nan")
    optimization_started = time.perf_counter()

    try:
        model.train()
        while stale < patience and (maximum_steps == 0 or step < maximum_steps):
            checkpoint(context.cancellation, "optimization")
            step += 1
            source, target = pairs[int(rng.integers(0, 4))]
            try:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(source)
                loss = F.binary_cross_entropy(prediction, target)
                loss.backward()
                optimizer.step()

                # Full-image validation after every optimizer update is part of
                # Noise2Fast's over-fitting detector and reconstruction rule.
                with torch.no_grad():
                    validation = model(full)
                    validation_score_tensor = -F.mse_loss(validation, full)
                    recent.append(validation[0, 0].detach().cpu().numpy().copy())
                    validation_score = float(validation_score_tensor.detach().cpu())
            except Exception as exc:
                if _is_accelerator_oom(exc):
                    raise _DeviceOOM(str(exc)) from exc
                raise

            if validation_score > best_validation:
                best_validation = validation_score
                best_step = step
                stale = 0
            else:
                stale += 1

            should_report = step == 1 or step % report_interval == 0
            should_report = should_report or stale == patience
            should_report = should_report or (maximum_steps > 0 and step == maximum_steps)
            if should_report:
                last_loss = float(loss.detach().cpu())
                smooth_loss = (
                    last_loss
                    if not np.isfinite(smooth_loss)
                    else 0.95 * smooth_loss + 0.05 * last_loss
                )
                context.emitter.progress(
                    "optimization",
                    f"Noise2Fast step {step}; no improvement for {stale}/{patience} steps.",
                    step,
                    maximum_steps,  # Zero denotes an unknown optimization total.
                    method="noise2fast",
                    channel=context.channel + 1,
                    channels=context.channels,
                    metrics={
                        "loss": last_loss,
                        "smoothed_loss": smooth_loss,
                        "learning_rate": 1e-3,
                        "elapsed_seconds": time.perf_counter() - optimization_started,
                        "best_step": best_step,
                        "validation_negative_mse": validation_score,
                        "validation_patience": patience,
                        # Historical key: this counter counts optimizer steps.
                        "epochs_without_validation_improvement": stale,
                    },
                )

        if not recent:
            raise RuntimeError("Noise2Fast produced no validation predictions")
        output = np.mean(np.stack(tuple(recent), axis=0), axis=0, dtype=np.float64).astype(
            np.float32
        )
        stopped_by = "validation_patience" if stale >= patience else "configured_step_limit"
        metadata: dict[str, Any] = {
            "algorithm_fidelity": (
                "Noise2Fast released 64-feature architecture; JDLL training loop"
                if hidden_width == 64
                else "Noise2Fast width variant; JDLL training loop"
            ),
            "architecture": f"8x3x3_conv_{hidden_width}_relu+1x1_sigmoid",
            "hidden_width": hidden_width,
            "training_pair_count": 4,
            "training_batch_size": 1,
            "training_loss": "binary_cross_entropy",
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "validation_metric": "negative_full_image_mse_to_noisy_input",
            "validation_patience": patience,
            "output_average_window": output_window,
            "output_average_count": len(recent),
            "optimization_steps_completed": step,
            "completed_steps": step,
            "best_step": best_step,
            "best_validation_negative_mse": best_validation,
            "epochs_without_validation_improvement": stale,
            "stopped_by": stopped_by,
            "configured_step_limit": maximum_steps,
            "model_device": str(next(model.parameters()).device),
            "training_input_device": str(pairs[0][0].device),
            "prediction_input_device": str(full.device),
            "device_used": device,
            "device_fallback": None,
            "training_view_shapes": [list(pair[0].shape) for pair in pairs],
            "phase_timings_seconds": {
                "optimization_and_validation": time.perf_counter() - optimization_started,
                "method_total": time.perf_counter() - started,
            },
        }
        if stopped_by == "configured_step_limit":
            metadata["warnings"] = [
                f"Noise2Fast stopped at the explicitly configured step limit ({maximum_steps}) "
                f"before validation patience ({patience} steps without improvement) was reached"
            ]
        return np.ascontiguousarray(output), metadata
    finally:
        del pairs, full, optimizer, model


def _run_plane(
    image: np.ndarray, context: MethodContext, plane_index: int
) -> tuple[np.ndarray, dict[str, Any]]:
    requested_resolved_device = context.config["device"]
    try:
        return _run_plane_on_device(image, context, plane_index, requested_resolved_device)
    except _DeviceOOM as exc:
        if requested_resolved_device != "cpu" and context.config["allow_cpu_fallback"]:
            warning = (
                f"Noise2Fast accelerator OOM on {requested_resolved_device}; explicitly "
                "falling back to CPU and restarting this plane"
            )
            context.emitter.warning("optimization", warning, method="noise2fast")
            cleanup_device(requested_resolved_device)
            fallback_context = MethodContext(
                config=deepcopy(context.config),
                emitter=context.emitter,
                cancellation=context.cancellation,
                seed=context.seed,
                channel=context.channel,
                channels=context.channels,
                time=context.time,
                times=context.times,
            )
            fallback_context.config["device"] = "cpu"
            output, metadata = _run_plane_on_device(image, fallback_context, plane_index, "cpu")
            metadata["device_fallback"] = {
                "from": requested_resolved_device,
                "to": "cpu",
                "reason": "accelerator_out_of_memory",
            }
            metadata.setdefault("warnings", []).append(warning)
            return output, metadata
        raise OutOfMemoryDenoisingError(
            f"method=noise2fast device={requested_resolved_device} phase=optimization: "
            "out of memory while running the full-plane paper algorithm; enable "
            "allow_cpu_fallback or use a device with more memory"
        ) from exc


def run(image: np.ndarray, context: MethodContext) -> MethodResult:
    if image.ndim == 2:
        output, metadata = _run_plane(image, context, 0)
        return MethodResult(output, metadata)
    output = np.empty_like(image, dtype=np.float32)
    planes = []
    for z in range(image.shape[0]):
        checkpoint(context.cancellation, "between_planes")
        context.emitter.status(
            "preparing",
            f"Preparing Noise2Fast plane {z + 1}/{image.shape[0]}",
            method=context.config["method"],
            z=z,
            z_total=image.shape[0],
        )
        output[z], info = _run_plane(image[z], context, z)
        planes.append(info)
    return MethodResult(output, {"planes": planes})
