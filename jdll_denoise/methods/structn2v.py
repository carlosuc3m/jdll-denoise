"""Paper-faithful Structured Noise2Void (StructN2V) in PyTorch.

This module implements Broaddus et al., *Removing Structured Noise with
Self-Supervised Blind-Spot Networks* (ISBI 2020): an N2V active-pixel loss,
an extended blind mask adapted to the noise correlation, random patch
training, and the paper's distinct two-level 2D and anisotropic 3D U-Nets.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import nn
from torch.nn import functional as F

from ..cancellation import checkpoint
from ..errors import OutOfMemoryDenoisingError
from .base import MethodContext, MethodResult, seed_everything

STRUCTURES = ("local_isotropic", "horizontal", "vertical", "periodic_grid")


class _DoubleConv(nn.Module):
    def __init__(
        self, dimensions: int, in_channels: int, out_channels: int, kernel_size: tuple[int, ...]
    ):
        super().__init__()
        if dimensions == 2:
            kernel_2d = (kernel_size[0], kernel_size[1])
            padding_2d = (kernel_2d[0] // 2, kernel_2d[1] // 2)
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_2d, padding=padding_2d),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_2d, padding=padding_2d),
                nn.ReLU(inplace=True),
            )
        else:
            kernel_3d = (kernel_size[0], kernel_size[1], kernel_size[2])
            padding_3d = (
                kernel_3d[0] // 2,
                kernel_3d[1] // 2,
                kernel_3d[2] // 2,
            )
            self.layers = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_3d, padding=padding_3d),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_3d, padding=padding_3d),
                nn.ReLU(inplace=True),
            )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class StructN2VUNet(nn.Module):
    """The exact two-level U-Net family specified by the StructN2V paper."""

    def __init__(self, dimensions: int):
        super().__init__()
        if dimensions not in {2, 3}:
            raise ValueError("StructN2VUNet supports only 2D and 3D")
        self.dimensions = dimensions
        kernel = (3, 3) if dimensions == 2 else (3, 5, 5)
        pool_size = (2, 2) if dimensions == 2 else (1, 2, 2)
        pool = nn.MaxPool2d if dimensions == 2 else nn.MaxPool3d
        conv = nn.Conv2d if dimensions == 2 else nn.Conv3d
        self.pool_size = pool_size
        self.down_level_0 = _DoubleConv(dimensions, 1, 16, kernel)
        self.pool_0 = pool(pool_size)
        self.down_level_1 = _DoubleConv(dimensions, 16, 32, kernel)
        self.pool_1 = pool(pool_size)
        self.middle = _DoubleConv(dimensions, 32, 64, kernel)
        self.up_level_1 = _DoubleConv(dimensions, 64 + 32, 32, kernel)
        self.up_level_0 = _DoubleConv(dimensions, 32 + 16, 16, kernel)
        self.output = conv(16, 1, kernel_size=1)
        self.apply(self._initialize_like_csbdeep)

    @staticmethod
    def _initialize_like_csbdeep(module: nn.Module) -> None:
        # The Keras/CSBDeep implementation used Glorot-uniform kernels and
        # zero biases.  PyTorch's Conv defaults differ, so set them explicitly.
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _upsample(self, value: torch.Tensor, size: tuple[int, ...]) -> torch.Tensor:
        # Keras/CSBDeep U-Nets used by the paper use nearest-neighbor UpSampling.
        return F.interpolate(value, size=size, mode="nearest")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        level_0 = self.down_level_0(value)
        level_1 = self.down_level_1(self.pool_0(level_0))
        middle = self.middle(self.pool_1(level_1))
        up_1 = self.up_level_1(
            torch.cat((self._upsample(middle, level_1.shape[2:]), level_1), dim=1)
        )
        up_0 = self.up_level_0(
            torch.cat((self._upsample(up_1, level_0.shape[2:]), level_0), dim=1)
        )
        return self.output(up_0)


def noise_autocorrelation(noise: np.ndarray) -> np.ndarray:
    """N-dimensional normalized autocorrelation used by the authors' N2V code."""

    value = np.asarray(noise, dtype=np.float64)
    std = float(value.std())
    if std <= 1e-12:
        result = np.zeros(value.shape, dtype=np.float64)
        result[tuple(size // 2 for size in value.shape)] = 1.0
        return result
    transformed = np.fft.fftn((value - value.mean()) / std)
    correlation = np.fft.ifftn(np.abs(transformed) ** 2).real
    correlation /= correlation.flat[0]
    return np.fft.fftshift(correlation)


def _normalized_correlation(residual: np.ndarray, shift: tuple[int, ...]) -> float:
    slices_a: list[slice] = []
    slices_b: list[slice] = []
    for size, offset in zip(residual.shape, shift, strict=True):
        if abs(offset) >= size:
            return 0.0
        if offset >= 0:
            slices_a.append(slice(offset, size))
            slices_b.append(slice(0, size - offset))
        else:
            slices_a.append(slice(0, size + offset))
            slices_b.append(slice(-offset, size))
    first = residual[tuple(slices_a)]
    second = residual[tuple(slices_b)]
    denominator = float(np.sqrt(np.sum(first**2) * np.sum(second**2)))
    return 0.0 if denominator <= 1e-12 else float(abs(np.sum(first * second)) / denominator)


def select_noise_structure(image: np.ndarray, period: int) -> tuple[str, dict[str, float]]:
    """Rank mask families from a high-pass noise autocorrelation estimate.

    The paper recommends choosing the smallest rectangular support covering the
    noise autocorrelation.  When a separate dark/noise frame is unavailable,
    the JDLL array API obtains a deterministic proxy by high-pass filtering.
    Final ``auto`` selection additionally uses held-out masked prediction loss.
    """

    residual = np.asarray(image, dtype=np.float64) - gaussian_filter(
        np.asarray(image, dtype=np.float64), sigma=1.0
    )
    if residual.ndim == 3:
        residual_2d = residual.mean(axis=0)
    else:
        residual_2d = residual
    p = max(1, int(period))
    scores = {
        "local_isotropic": 0.5
        * (
            _normalized_correlation(residual_2d, (0, 1))
            + _normalized_correlation(residual_2d, (1, 0))
        ),
        "horizontal": _normalized_correlation(residual_2d, (0, 2)),
        "vertical": _normalized_correlation(residual_2d, (2, 0)),
        "periodic_grid": max(
            _normalized_correlation(residual_2d, (0, p)),
            _normalized_correlation(residual_2d, (p, 0)),
        ),
    }
    return max(scores, key=scores.__getitem__), scores


def make_structured_blind_mask(
    structure: str, dimensions: int, radius: int, period: int
) -> np.ndarray:
    """Return an odd-sized binary extended blind mask including its center."""

    if structure not in STRUCTURES:
        raise ValueError(f"unknown StructN2V mask structure: {structure}")
    radius = max(1, int(radius))
    if structure == "horizontal":
        horizontal_shape = (
            (1, 2 * radius + 1) if dimensions == 2 else (1, 1, 2 * radius + 1)
        )
        return np.ones(horizontal_shape, dtype=bool)
    if structure == "vertical":
        vertical_shape = (
            (2 * radius + 1, 1) if dimensions == 2 else (1, 2 * radius + 1, 1)
        )
        return np.ones(vertical_shape, dtype=bool)
    if structure == "local_isotropic":
        return np.ones((2 * radius + 1,) * dimensions, dtype=bool)

    extent = max(1, int(period))
    periodic_shape = (2 * extent + 1,) * dimensions
    mask = np.zeros(periodic_shape, dtype=bool)
    center = np.array(periodic_shape) // 2
    mask[tuple(center)] = True
    # Periodic readout artifacts are represented at +/- period in Y and X.
    for axis in range(max(0, dimensions - 2), dimensions):
        for sign in (-1, 1):
            point = center.copy()
            point[axis] += sign * extent
            mask[tuple(point)] = True
    return mask


def _active_coordinates(
    shape: tuple[int, ...], fraction: float, rng: np.random.Generator
) -> np.ndarray:
    count = max(1, int(np.prod(shape, dtype=np.int64) * fraction))
    flat = rng.choice(int(np.prod(shape)), size=count, replace=False)
    return np.stack(np.unravel_index(flat, shape), axis=1).astype(np.int64, copy=False)


def _mask_offsets(mask: np.ndarray) -> np.ndarray:
    center = np.array(mask.shape, dtype=np.int64) // 2
    offsets = np.argwhere(mask).astype(np.int64) - center
    return offsets[np.any(offsets != 0, axis=1)]


def structured_masks(
    shape: tuple[int, ...],
    *,
    structure: str,
    fraction: float,
    radius: int,
    period: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return active loss pixels and all hidden pixels, without border wraparound."""

    mask = make_structured_blind_mask(structure, len(shape), radius, period)
    coordinates = _active_coordinates(shape, fraction, rng)
    centers = np.zeros(shape, dtype=bool)
    centers[tuple(coordinates.T)] = True
    corrupted = centers.copy()
    offsets = _mask_offsets(mask)
    if offsets.size:
        expanded = coordinates[:, None, :] + offsets[None, :, :]
        expanded = expanded.reshape(-1, len(shape))
        valid = np.all((expanded >= 0) & (expanded < np.array(shape)), axis=1)
        expanded = expanded[valid]
        corrupted[tuple(expanded.T)] = True
    return centers, corrupted


def prepare_structn2v_batch(
    patches: np.ndarray,
    *,
    blind_mask: np.ndarray,
    active_fraction: float,
    neighborhood_radius: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the N2V active-pixel manipulator and StructN2V extended mask."""

    targets = np.ascontiguousarray(patches, dtype=np.float32)
    inputs = targets.copy()
    loss_mask = np.zeros(targets.shape, dtype=bool)
    spatial_shape = targets.shape[1:]
    dimensions = len(spatial_shape)
    offsets = _mask_offsets(blind_mask)
    radius = int(neighborhood_radius)

    for batch_index in range(targets.shape[0]):
        coordinates = _active_coordinates(spatial_shape, active_fraction, rng)
        loss_mask[(batch_index,) + tuple(coordinates.T)] = True

        # N2V's uniform-with-center-pixel manipulator samples a replacement
        # from the local (2r+1)^D neighborhood for every active pixel.
        lower = np.maximum(coordinates - radius, 0)
        upper = np.minimum(coordinates + radius + 1, np.array(spatial_shape))
        neighbors = rng.integers(lower, upper)
        inputs[(batch_index,) + tuple(coordinates.T)] = targets[
            (batch_index,) + tuple(neighbors.T)
        ]

        # StructN2V hides the remaining support of the blind mask using the
        # authors' uniform [-2, 2] replacement distribution.  Border support is
        # truncated instead of wrapped to the opposite image edge.
        if offsets.size:
            extended = coordinates[:, None, :] + offsets[None, :, :]
            extended = extended.reshape(-1, dimensions)
            valid = np.all(
                (extended >= 0) & (extended < np.array(spatial_shape, dtype=np.int64)), axis=1
            )
            extended = extended[valid]
            inputs[(batch_index,) + tuple(extended.T)] = rng.uniform(
                -2.0, 2.0, size=len(extended)
            ).astype(np.float32)
    return inputs, targets, loss_mask


@dataclass(frozen=True)
class _PatchSpec:
    start: tuple[int, ...]
    rotation: int
    flip: bool


def _pad_source_to_patch(source: np.ndarray, patch_shape: tuple[int, ...]) -> np.ndarray:
    padding: list[tuple[int, int]] = []
    for size, required in zip(source.shape, patch_shape, strict=True):
        missing = max(0, required - size)
        padding.append((missing // 2, missing - missing // 2))
    if not any(before or after for before, after in padding):
        return source
    if all(size > 1 for size in source.shape):
        return np.pad(source, padding, mode="reflect")
    return np.pad(source, padding, mode="edge")


def sample_patch_specs(
    source_shape: tuple[int, ...],
    patch_shape: tuple[int, ...],
    count: int,
    rng: np.random.Generator,
) -> list[_PatchSpec]:
    specs: list[_PatchSpec] = []
    for _ in range(count):
        start = tuple(
            int(rng.integers(0, size - patch + 1)) if size > patch else 0
            for size, patch in zip(source_shape, patch_shape, strict=True)
        )
        specs.append(_PatchSpec(start, int(rng.integers(0, 4)), bool(rng.integers(0, 2))))
    return specs


def _extract_patch(
    source: np.ndarray, patch_shape: tuple[int, ...], spec: _PatchSpec
) -> np.ndarray:
    slices = tuple(
        slice(start, start + size)
        for start, size in zip(spec.start, patch_shape, strict=True)
    )
    patch = source[slices]
    patch = np.rot90(patch, spec.rotation, axes=(-2, -1))
    if spec.flip:
        patch = np.flip(patch, axis=-1)
    return np.ascontiguousarray(patch, dtype=np.float32)


def _batch_from_specs(
    source: np.ndarray, patch_shape: tuple[int, ...], specs: Iterable[_PatchSpec]
) -> np.ndarray:
    return np.stack([_extract_patch(source, patch_shape, spec) for spec in specs])


def _tensor_batch(value: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value))[:, None].to(
        device=device, dtype=torch.float32
    )


def _masked_loss(
    model: nn.Module,
    inputs: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    device: str,
) -> torch.Tensor:
    input_tensor = _tensor_batch(inputs, device)
    target_tensor = _tensor_batch(targets, device)
    mask_tensor = torch.from_numpy(mask)[:, None].to(device=device)
    prediction = model(input_tensor)
    return F.mse_loss(prediction[mask_tensor], target_tensor[mask_tensor])


def _paper_patch_shape(image: np.ndarray, cfg: dict[str, Any]) -> tuple[int, ...]:
    dimensions = image.ndim
    configured = cfg["patch_size"]
    result: tuple[int, ...]
    if isinstance(configured, int):
        if dimensions == 2:
            result = (configured, configured)
        else:
            result = (min(32, image.shape[0]), configured, configured)
    else:
        result = tuple(int(value) for value in configured)
    if len(result) != dimensions:
        raise ValueError(f"StructN2V patch_size must have {dimensions} entries")
    # Two Y/X pooling operations require divisibility by four.
    result = result[:-2] + tuple(max(4, int(math.ceil(size / 4) * 4)) for size in result[-2:])
    return result


def _structure_tournament(
    source: np.ndarray,
    context: MethodContext,
    patch_shape: tuple[int, ...],
    autocorrelation_scores: dict[str, float],
) -> tuple[str, dict[str, float]]:
    """Select ``auto`` using deterministic held-out masked prediction loss."""

    cfg = context.config["method_config"]
    device = context.config["device"]
    dimensions = source.ndim
    selection_steps = int(cfg["selection_steps"])
    batch_size = int(cfg["batch_size"])
    rng_specs = np.random.default_rng(context.seed + 41)
    padded = _pad_source_to_patch(source, patch_shape)
    train_specs = sample_patch_specs(
        padded.shape, patch_shape, max(batch_size * selection_steps, batch_size), rng_specs
    )
    heldout_specs = sample_patch_specs(padded.shape, patch_shape, batch_size, rng_specs)
    heldout_patches = _batch_from_specs(padded, patch_shape, heldout_specs)
    losses: dict[str, float] = {}

    for candidate_index, structure in enumerate(STRUCTURES):
        checkpoint(context.cancellation, "selecting_structure")
        seed_everything(context.seed + 10_000)
        model = StructN2VUNet(dimensions).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
        blind_mask = make_structured_blind_mask(
            structure, dimensions, int(cfg["correlation_radius"]), int(cfg["period"])
        )
        candidate_rng = np.random.default_rng(context.seed + 1_000 + candidate_index)
        try:
            model.train()
            for step in range(selection_steps):
                checkpoint(context.cancellation, "selecting_structure")
                selected_specs = train_specs[step * batch_size : (step + 1) * batch_size]
                patches = _batch_from_specs(padded, patch_shape, selected_specs)
                inputs, targets, mask = prepare_structn2v_batch(
                    patches,
                    blind_mask=blind_mask,
                    active_fraction=float(cfg["mask_fraction"]),
                    neighborhood_radius=int(cfg["neighborhood_radius"]),
                    rng=candidate_rng,
                )
                optimizer.zero_grad(set_to_none=True)
                loss = _masked_loss(model, inputs, targets, mask, device)
                loss.backward()
                optimizer.step()

            heldout_rng = np.random.default_rng(context.seed + 20_000 + candidate_index)
            inputs, targets, mask = prepare_structn2v_batch(
                heldout_patches,
                blind_mask=blind_mask,
                active_fraction=float(cfg["mask_fraction"]),
                neighborhood_radius=int(cfg["neighborhood_radius"]),
                rng=heldout_rng,
            )
            model.eval()
            with torch.no_grad():
                losses[structure] = float(
                    _masked_loss(model, inputs, targets, mask, device).detach().cpu()
                )
        finally:
            del optimizer, model

    selected_structure = min(
        STRUCTURES,
        key=lambda name: (losses[name], -autocorrelation_scores.get(name, 0.0), name),
    )
    return selected_structure, losses


def _pad_for_unet(source: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = source.shape[-2:]
    pad_y = (-height) % 4
    pad_x = (-width) % 4
    if pad_y == 0 and pad_x == 0:
        return source, (height, width)
    mode = "reflect" if height > pad_y and width > pad_x and min(height, width) > 1 else "replicate"
    return F.pad(source, (0, pad_x, 0, pad_y), mode=mode), (height, width)


def _fit_and_predict(
    image: np.ndarray,
    context: MethodContext,
    structure: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    cfg = context.config["method_config"]
    device = context.config["device"]
    dimensions = image.ndim
    seed_everything(context.seed)
    rng = np.random.default_rng(context.seed)
    patch_shape = _paper_patch_shape(image, cfg)
    padded_source = _pad_source_to_patch(np.ascontiguousarray(image, dtype=np.float32), patch_shape)
    patch_count = int(cfg["patch_count"])
    batch_size = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])
    configured_steps = int(cfg["steps"])
    paper_steps = epochs * math.ceil(patch_count / batch_size)
    steps = configured_steps if configured_steps > 0 else paper_steps
    specs = sample_patch_specs(padded_source.shape, patch_shape, patch_count, rng)
    order = np.arange(patch_count)
    rng.shuffle(order)
    cursor = 0
    blind_mask = make_structured_blind_mask(
        structure, dimensions, int(cfg["correlation_radius"]), int(cfg["period"])
    )
    model = StructN2VUNet(dimensions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    report_interval = max(1, int(cfg["report_interval"]))
    smooth = float("nan")
    best = float("inf")
    best_step = 0
    last_loss = float("nan")
    started = time.perf_counter()

    try:
        model.train()
        for step in range(1, steps + 1):
            checkpoint(context.cancellation, "optimization")
            if cursor + batch_size > patch_count:
                rng.shuffle(order)
                cursor = 0
            selected_indices = order[cursor : cursor + batch_size]
            cursor += batch_size
            patches = _batch_from_specs(
                padded_source, patch_shape, (specs[int(index)] for index in selected_indices)
            )
            inputs, targets, mask = prepare_structn2v_batch(
                patches,
                blind_mask=blind_mask,
                active_fraction=float(cfg["mask_fraction"]),
                neighborhood_radius=int(cfg["neighborhood_radius"]),
                rng=rng,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = _masked_loss(model, inputs, targets, mask, device)
            loss.backward()
            optimizer.step()

            if step == 1 or step == steps or step % report_interval == 0:
                last_loss = float(loss.detach().cpu())
                smooth = last_loss if not np.isfinite(smooth) else 0.95 * smooth + 0.05 * last_loss
                if smooth < best:
                    best = smooth
                    best_step = step
                context.emitter.progress(
                    "optimization",
                    f"Training StructN2V - step {step}/{steps}",
                    step,
                    steps,
                    method="structn2v",
                    channel=context.channel + 1,
                    channels=context.channels,
                    metrics={
                        "loss": last_loss,
                        "smoothed_loss": smooth,
                        "learning_rate": float(cfg["learning_rate"]),
                        "elapsed_seconds": time.perf_counter() - started,
                        "best_step": best_step,
                    },
                )

        checkpoint(context.cancellation, "inference")
        full = torch.from_numpy(np.ascontiguousarray(image))[None, None].to(
            device=device, dtype=torch.float32
        )
        full_padded, original_yx = _pad_for_unet(full)
        model.eval()
        with torch.no_grad():
            output = model(full_padded)[0, 0, ..., : original_yx[0], : original_yx[1]]
            result = output.detach().cpu().numpy().astype(np.float32, copy=False)
        metadata = {
            "algorithm_fidelity": "StructN2V ISBI 2020",
            "architecture": "paper_2d_unet" if dimensions == 2 else "paper_anisotropic_3d_unet",
            "unet_levels": 2,
            "unet_first_level_features": 16,
            "unet_kernel_size": [3, 3] if dimensions == 2 else [3, 5, 5],
            "unet_pool_size": [2, 2] if dimensions == 2 else [1, 2, 2],
            "training_loss": "masked_mse_at_active_pixels_only",
            "optimizer": "Adam",
            "learning_rate": float(cfg["learning_rate"]),
            "noise_structure": structure,
            "blind_mask": blind_mask.astype(np.uint8).tolist(),
            "blind_mask_shape": list(blind_mask.shape),
            "active_pixel_fraction": float(cfg["mask_fraction"]),
            "active_pixel_manipulator": "uniform_with_center_pixel",
            "manipulator_neighborhood_radius": int(cfg["neighborhood_radius"]),
            "extended_mask_replacement_distribution": "uniform[-2,2]",
            "training_patch_shape": list(patch_shape),
            "training_patch_count": patch_count,
            "training_batch_size": batch_size,
            "training_epochs": epochs,
            "paper_training_steps": paper_steps,
            "optimization_steps_completed": steps,
            "configured_step_limit": configured_steps,
            "best_step": best_step,
            "final_loss": last_loss,
            "final_smoothed_loss": smooth,
            "model_device": str(next(model.parameters()).device),
            "training_input_device": device,
            "prediction_input_device": str(full.device),
            "device_used": device,
            "phase_timings_seconds": {"method_total": time.perf_counter() - started},
        }
        if configured_steps > 0 and configured_steps != paper_steps:
            metadata["warnings"] = [
                "StructN2V used an explicitly configured optimization-step override instead of "
                "the paper's complete epoch schedule"
            ]
        return np.ascontiguousarray(result), metadata
    finally:
        del optimizer, model


def run(image: np.ndarray, context: MethodContext) -> MethodResult:
    cfg = context.config["method_config"]
    structure = str(cfg["noise_structure"])
    autocorrelation_scores: dict[str, float] = {}
    heldout_losses: dict[str, float] = {}
    patch_shape = _paper_patch_shape(image, cfg)
    if structure == "auto":
        checkpoint(context.cancellation, "selecting_structure")
        _, autocorrelation_scores = select_noise_structure(image, int(cfg["period"]))
        structure, heldout_losses = _structure_tournament(
            image, context, patch_shape, autocorrelation_scores
        )
        context.emitter.status(
            "selecting_structure",
            f"Selected StructN2V noise structure: {structure}",
            method="structn2v",
            info={
                "autocorrelation_scores": autocorrelation_scores,
                "heldout_masked_prediction_losses": heldout_losses,
                "selected": structure,
            },
        )
    try:
        output, metadata = _fit_and_predict(image, context, structure)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        is_cuda_oom = isinstance(exc, torch.cuda.OutOfMemoryError)
        if not is_cuda_oom and "out of memory" not in str(exc).lower():
            raise
        raise OutOfMemoryDenoisingError(
            f"method=structn2v device={context.config['device']} phase=optimization/inference: "
            "out of memory while running the paper architecture and patch schedule"
        ) from exc
    metadata.update(
        {
            "structure_candidate_autocorrelation_scores": autocorrelation_scores,
            "structure_candidate_heldout_losses": heldout_losses,
            "structure_selection": (
                "heldout_masked_prediction_loss" if heldout_losses else "explicit"
            ),
        }
    )
    return MethodResult(output, metadata)
