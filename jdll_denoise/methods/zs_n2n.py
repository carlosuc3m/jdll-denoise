"""ZS-N2N's complete objective with explicit JDLL effort/architecture variants."""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..cancellation import checkpoint
from ..cleanup import cleanup_device
from ..errors import InputFormatError, OutOfMemoryDenoisingError
from ..presets import ZS_N2N_EFFORT_PRESETS_VERSION, method_preset
from ..tiling import _is_accelerator_oom, tiled_predict
from .base import MethodContext, MethodResult, optimization_event, seed_everything


class ZSN2NNetwork(nn.Module):
    """Three-convolution noise predictor; defaults retain the reference network."""

    def __init__(
        self,
        channels: int = 1,
        hidden_width: int = 48,
        middle_groups: int = 1,
        output_init: str = "native",
        output_init_std: float = 0.001,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, hidden_width, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                hidden_width,
                hidden_width,
                kernel_size=3,
                padding=1,
                groups=middle_groups,
                bias=True,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(hidden_width, channels, kernel_size=1, bias=True),
        )
        if output_init == "near_zero":
            head = self.layers[4]
            assert isinstance(head, nn.Conv2d) and head.bias is not None
            nn.init.normal_(head.weight, mean=0.0, std=output_init_std)
            nn.init.zeros_(head.bias)
        elif output_init != "native":
            raise ValueError("output_init must be native or near_zero")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Predict noise; subtraction is deliberately outside the network."""
        return self.layers(value)


def pair_downsampler(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the fixed channel-wise 2x2 stride-2 filters from the paper."""
    if value.ndim != 4:
        raise ValueError("ZS-N2N pair_downsampler expects a B,C,H,W tensor")
    channels = value.shape[1]
    kernels = value.new_tensor(
        [
            [[[0.0, 0.5], [0.5, 0.0]]],
            [[[0.5, 0.0], [0.0, 0.5]]],
        ]
    )
    first_filter = kernels[0].repeat(channels, 1, 1, 1)
    second_filter = kernels[1].repeat(channels, 1, 1, 1)
    first = F.conv2d(value, first_filter, stride=2, padding=0, groups=channels)
    second = F.conv2d(value, second_filter, stride=2, padding=0, groups=channels)
    return first, second


# Preserve the old helper name as a source-compatible alias while correcting
# its behavior to the canonical no-padding grouped convolution.
paired_views = pair_downsampler


def canonical_losses(
    model: nn.Module,
    noisy: torch.Tensor,
    batch_pairs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Complete objective; direct callers retain the unbatched reference path."""
    first, second = pair_downsampler(noisy)
    if batch_pairs:
        noise_first, noise_second = model(torch.cat((first, second), dim=0)).chunk(2, dim=0)
    else:
        noise_first, noise_second = model(first), model(second)
    prediction_first = first - noise_first
    prediction_second = second - noise_second
    full_prediction = noisy - model(noisy)
    full_first, full_second = pair_downsampler(full_prediction)
    residual = 0.5 * (F.mse_loss(first, prediction_second) + F.mse_loss(second, prediction_first))
    consistency = 0.5 * (
        F.mse_loss(prediction_first, full_first) + F.mse_loss(prediction_second, full_second)
    )
    return residual + consistency, residual, consistency


def canonical_inference(model: nn.Module, noisy: torch.Tensor) -> torch.Tensor:
    return torch.clamp(noisy - model(noisy), 0.0, 1.0)


def _as_bchw(image: np.ndarray, device: str) -> torch.Tensor:
    array = np.ascontiguousarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3:
        raise InputFormatError("method=zs_n2n phase=preparing: expected YX or CYX plane")
    return torch.from_numpy(array)[None].to(device=device, dtype=torch.float32)


def _fidelity(cfg: dict[str, Any], tiled: bool, effort: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    preset = method_preset("zs_n2n", effort)
    reference_network = (cfg["hidden_width"], cfg["middle_groups"], cfg["output_init"]) == (
        48,
        1,
        "native",
    )
    standard_variant = all(
        cfg[key] == preset[key] for key in ("hidden_width", "middle_groups", "output_init")
    ) and (cfg["output_init"] == "native" or cfg["output_init_std"] == preset["output_init_std"])
    if reference_network:
        status = (
            "reference_architecture_batched"
            if cfg["batch_pairs"]
            else "reference_architecture_unbatched"
        )
    elif standard_variant:
        status = "jdll_performance_variant"
    else:
        status = "noncanonical_extension"
    if not reference_network:
        warnings.append(
            f"JDLL ZS-N2N architecture variant: hidden_width={cfg['hidden_width']}, "
            f"middle_groups={cfg['middle_groups']}, output_init={cfg['output_init']}"
            + (
                f", output_init_std={cfg['output_init_std']}"
                if cfg["output_init"] == "near_zero"
                else ""
            )
            + "; full published objective retained, not the reference network"
        )
    custom_iterations = cfg["steps"] != preset["steps"]
    if custom_iterations:
        warnings.append(
            f"custom ZS-N2N iteration budget: {cfg['steps']} (preset {preset['steps']}); "
            "full published objective retained"
        )
    if float(cfg["learning_rate"]) != 1e-3:
        warnings.append("noncanonical ZS-N2N initial learning rate override")
    if int(cfg["scheduler_step_size"]) != 1000 or float(cfg["scheduler_gamma"]) != 0.5:
        warnings.append("noncanonical ZS-N2N StepLR override")
    if cfg.get("early_stopping", False):
        warnings.append("noncanonical ZS-N2N early stopping extension enabled")
    if cfg.get("restore_best_weights", False):
        warnings.append("noncanonical ZS-N2N best-weight restoration extension enabled")
    if tiled:
        warnings.append(
            "tiled inference is a JDLL memory extension; canonical training is unchanged"
        )
    if status == "noncanonical_extension" or any("noncanonical" in warning for warning in warnings):
        return "noncanonical_extension", warnings
    if custom_iterations:
        status += "_custom_iteration_budget"
    if tiled:
        status += "_with_tiled_inference"
    return status, warnings


def _run_plane_on_resolved_device(
    image: np.ndarray, context: MethodContext, plane_index: int
) -> tuple[np.ndarray, dict[str, Any]]:
    if min(image.shape[-2:]) < 2:
        raise InputFormatError(
            "method=zs_n2n phase=preparing: both spatial dimensions must be at least 2 pixels"
        )
    seed = context.seed + plane_index
    seed_everything(seed)
    device = context.config["device"]
    cfg = context.config["method_config"]
    channels = 1 if image.ndim == 2 else image.shape[0]
    context.emitter.status(
        "preparing",
        f"ZS-N2N {context.config['effort'].replace('_', '-').capitalize()}: "
        f"{cfg['hidden_width']} features, middle groups {cfg['middle_groups']}, "
        f"{cfg['steps']} updates, head {cfg['output_init']}"
        + (f" (std {cfg['output_init_std']})" if cfg["output_init"] == "near_zero" else "")
        + f", batch pairs {cfg['batch_pairs']}, Adam learning rate {cfg['learning_rate']}, "
        f"StepLR({cfg['scheduler_step_size']}, {cfg['scheduler_gamma']}), device {device}.",
        method="zs_n2n",
        channel=context.channel + 1,
        channels=context.channels,
        z=plane_index,
    )
    model = ZSN2NNetwork(
        channels,
        hidden_width=cfg["hidden_width"],
        middle_groups=cfg["middle_groups"],
        output_init=cfg["output_init"],
        output_init_std=cfg["output_init_std"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(cfg["scheduler_step_size"]),
        gamma=float(cfg["scheduler_gamma"]),
    )
    full = _as_bchw(image, device)
    steps = int(cfg["steps"])
    report_interval = max(1, int(cfg["report_interval"]))
    smooth_tensor: torch.Tensor | None = None
    best = float("inf")
    best_step = 0
    stale = 0
    best_state: dict[str, torch.Tensor] | None = None
    step = 0
    last_values = (float("nan"), float("nan"), float("nan"))
    optimization_started = time.perf_counter()
    try:
        model.train()
        for step in range(1, steps + 1):
            checkpoint(context.cancellation, "optimization")
            optimizer.zero_grad(set_to_none=True)
            total, residual, consistency = canonical_losses(
                model, full, batch_pairs=cfg["batch_pairs"]
            )
            total.backward()
            update_learning_rate = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            detached = total.detach()
            smooth_tensor = (
                detached if smooth_tensor is None else 0.95 * smooth_tensor + 0.05 * detached
            )
            should_report = step == 1 or step == steps or step % report_interval == 0
            track_every_update = cfg["early_stopping"] or cfg["restore_best_weights"]
            if should_report or track_every_update:
                last_values = (
                    float(total.detach().cpu()),
                    float(residual.detach().cpu()),
                    float(consistency.detach().cpu()),
                )
                smooth_value = float(smooth_tensor.cpu())
                if smooth_value < best - float(cfg.get("min_delta", 0.0)):
                    best, best_step, stale = smooth_value, step, 0
                    if cfg.get("restore_best_weights", False):
                        best_state = {
                            name: value.detach().cpu().clone()
                            for name, value in model.state_dict().items()
                        }
                else:
                    stale += 1
            stop = cfg["early_stopping"] and stale >= int(cfg.get("patience", 100))
            if should_report or stop:
                optimization_event(
                    context,
                    "ZS-N2N",
                    step,
                    steps,
                    last_values[0],
                    float(smooth_tensor.cpu()),
                    update_learning_rate,
                    time.perf_counter() - optimization_started,
                    best_step,
                    extra_metrics={
                        "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "loss_total": last_values[0],
                        "loss_residual": last_values[1],
                        "loss_consistency": last_values[2],
                    },
                )
            if stop:
                break
        if cfg.get("restore_best_weights", False) and best_state is not None:
            model.load_state_dict(best_state)
        optimization_seconds = time.perf_counter() - optimization_started
        checkpoint(context.cancellation, "inference")
        model.eval()
        prediction_device = str(next(model.parameters()).device)

        def predict(tile: np.ndarray, trained: nn.Module = model) -> np.ndarray:
            nonlocal prediction_device
            with torch.no_grad():
                tensor = _as_bchw(tile, device)
                prediction_device = str(tensor.device)
                result = canonical_inference(trained, tensor)[0].cpu().numpy()
                return result[0] if tile.ndim == 2 else result

        inference_started = time.perf_counter()
        tiled = bool(context.config["tiling"]["enabled"])
        if tiled:
            tile_config = context.config["tiling"]
            output = tiled_predict(
                image,
                predict,
                tile_size=tile_config["tile_size"],
                overlap=tile_config["overlap"],
                blend=tile_config["blend"],
                emitter=context.emitter,
                cancellation=context.cancellation,
            )
        else:
            output = predict(image)
        inference_seconds = time.perf_counter() - inference_started
        fidelity_status, fidelity_warnings = _fidelity(cfg, tiled, context.config["effort"])
        for warning in fidelity_warnings:
            context.emitter.warning("finalizing", warning, method="zs_n2n")
        return np.ascontiguousarray(output, dtype=np.float32), {
            "effort": context.config["effort"],
            "effort_presets_version": ZS_N2N_EFFORT_PRESETS_VERSION,
            "configured_iterations": steps,
            "optimization_steps_completed": step,
            "completed_iterations": step,
            "best_step": best_step,
            "final_smoothed_loss": (
                float(smooth_tensor.cpu()) if smooth_tensor is not None else float("nan")
            ),
            "final_losses": {
                "total": last_values[0],
                "residual": last_values[1],
                "consistency": last_values[2],
            },
            "optimizer": "Adam",
            "optimizer_settings": {
                "learning_rate": float(cfg["learning_rate"]),
                "betas": list(optimizer.defaults["betas"]),
                "eps": optimizer.defaults["eps"],
                "weight_decay": optimizer.defaults["weight_decay"],
                "amsgrad": optimizer.defaults["amsgrad"],
            },
            "initial_learning_rate": float(cfg["learning_rate"]),
            "scheduler": "StepLR",
            "scheduler_step_size": int(cfg["scheduler_step_size"]),
            "scheduler_gamma": float(cfg["scheduler_gamma"]),
            "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "hidden_width": int(cfg["hidden_width"]),
            "middle_groups": cfg["middle_groups"],
            "output_init": cfg["output_init"],
            "output_init_std": cfg["output_init_std"]
            if cfg["output_init"] == "near_zero"
            else None,
            "batch_pairs": cfg["batch_pairs"],
            "network_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "channel_mode": context.config["channel_mode"],
            "fidelity_status": fidelity_status,
            "fidelity_warnings": fidelity_warnings,
            "early_stopping": bool(cfg.get("early_stopping", False)),
            "selected_step": (best_step if cfg.get("restore_best_weights", False) else step),
            "inference_tiled": tiled,
            "model_device": str(next(model.parameters()).device),
            "training_input_device": str(full.device),
            "prediction_input_device": prediction_device,
            "device_used": device,
            "phase_timings_seconds": {
                "optimization": optimization_seconds,
                "inference": inference_seconds,
            },
        }
    except Exception as exc:
        if _is_accelerator_oom(exc):
            raise OutOfMemoryDenoisingError(
                f"method=zs_n2n device={device} phase=optimization/inference: out of memory; "
                "reduce image/tile size or enable CPU fallback"
            ) from exc
        raise
    finally:
        del scheduler, optimizer, model, full


def _run_plane(
    image: np.ndarray, context: MethodContext, plane_index: int
) -> tuple[np.ndarray, dict[str, Any]]:
    device = context.config["device"]
    try:
        return _run_plane_on_resolved_device(image, context, plane_index)
    except OutOfMemoryDenoisingError:
        if device == "cpu" or not context.config["allow_cpu_fallback"]:
            raise
        warning = (
            f"ZS-N2N accelerator OOM on {device}; explicitly falling back to CPU and "
            "restarting this plane"
        )
        context.emitter.warning("optimization", warning, method="zs_n2n")
        cleanup_device(device)
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
        output, metadata = _run_plane_on_resolved_device(image, fallback_context, plane_index)
        metadata["device_fallback"] = {
            "from": device,
            "to": "cpu",
            "reason": "accelerator_out_of_memory",
        }
        metadata.setdefault("warnings", []).append(warning)
        return output, metadata


def run(image: np.ndarray, context: MethodContext) -> MethodResult:
    native_channels = context.config["channel_mode"] == "native_color"
    if image.ndim == 2 or (native_channels and image.ndim == 3):
        output, metadata = _run_plane(image, context, 0)
        return MethodResult(output, metadata)
    if native_channels:
        outputs = np.empty_like(image, dtype=np.float32)
        plane_metadata = []
        for z in range(image.shape[1]):
            checkpoint(context.cancellation, "between_planes")
            outputs[:, z], info = _run_plane(image[:, z], context, z)
            plane_metadata.append(info)
        return MethodResult(outputs, {"planes": plane_metadata, "channel_mode": "native_color"})
    outputs = np.empty_like(image, dtype=np.float32)
    plane_metadata = []
    for z in range(image.shape[0]):
        checkpoint(context.cancellation, "between_planes")
        context.emitter.status(
            "preparing",
            f"Preparing ZS-N2N plane {z + 1}/{image.shape[0]}",
            method=context.config["method"],
            z=z,
            z_total=image.shape[0],
        )
        outputs[z], info = _run_plane(image[z], context, z)
        plane_metadata.append(info)
    return MethodResult(outputs, {"planes": plane_metadata, "channel_mode": "independent"})
