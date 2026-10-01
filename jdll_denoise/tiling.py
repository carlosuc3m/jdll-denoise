"""Deterministic overlapping 2D tiled prediction and blending."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from .callbacks import EventEmitter
from .cancellation import CancellationLike, checkpoint


def tile_starts(length: int, size: int, overlap: int) -> list[int]:
    if length <= size:
        return [0]
    stride = size - overlap
    starts = list(range(0, max(1, length - size + 1), stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def _weight(shape: tuple[int, int], blend: str) -> np.ndarray:
    if blend == "linear":
        y = np.minimum(np.linspace(0.05, 1, shape[0] // 2 + 1), 1)
        y = np.concatenate((y, y[-2::-1]))[: shape[0]]
        x = np.minimum(np.linspace(0.05, 1, shape[1] // 2 + 1), 1)
        x = np.concatenate((x, x[-2::-1]))[: shape[1]]
    else:
        yy = np.linspace(-1, 1, shape[0])
        xx = np.linspace(-1, 1, shape[1])
        y = np.exp(-2.0 * yy * yy) + 1e-3
        x = np.exp(-2.0 * xx * xx) + 1e-3
    return np.outer(y, x).astype(np.float32)


def tiled_predict(
    image: np.ndarray,
    predict: Callable[[np.ndarray], np.ndarray],
    *,
    tile_size: int,
    overlap: int,
    blend: str = "gaussian",
    emitter: EventEmitter | None = None,
    cancellation: CancellationLike = None,
) -> np.ndarray:
    if image.ndim not in {2, 3}:
        raise ValueError("tiled_predict accepts a Y/X or C/Y/X plane")
    ys = tile_starts(image.shape[-2], tile_size, overlap)
    xs = tile_starts(image.shape[-1], tile_size, overlap)
    output = np.zeros(image.shape, dtype=np.float32)
    weights = np.zeros(image.shape[-2:], dtype=np.float32)
    maximum = len(ys) * len(xs)
    current = 0
    for y in ys:
        for x in xs:
            checkpoint(cancellation, "tiling")
            tile = image[..., y : y + tile_size, x : x + tile_size]
            prediction = np.asarray(predict(tile), dtype=np.float32)
            if prediction.shape != tile.shape:
                raise ValueError("tile predictor changed spatial shape")
            spatial_shape = (tile.shape[-2], tile.shape[-1])
            weight = _weight(spatial_shape, blend)
            broadcast_weight = weight.reshape((1,) * (image.ndim - 2) + weight.shape)
            output[..., y : y + spatial_shape[0], x : x + spatial_shape[1]] += (
                prediction * broadcast_weight
            )
            weights[y : y + spatial_shape[0], x : x + spatial_shape[1]] += weight
            current += 1
            if emitter:
                emitter.progress("tiling", f"Predicting tile {current}/{maximum}", current, maximum)
    checkpoint(cancellation, "blending")
    broadcast_weights = weights.reshape((1,) * (image.ndim - 2) + weights.shape)
    return np.ascontiguousarray(output / np.maximum(broadcast_weights, 1e-8), dtype=np.float32)


def _is_accelerator_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )


def batched_tiled_predict(
    image: np.ndarray,
    predict_batch: Callable[[np.ndarray], np.ndarray],
    *,
    tile_size: int,
    overlap: int,
    batch_size: int,
    blend: str = "gaussian",
    emitter: EventEmitter | None = None,
    cancellation: CancellationLike = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Predict 2D tiles in batches and halve a failing accelerator batch safely."""
    if image.ndim != 2:
        raise ValueError("batched_tiled_predict currently accepts one Y/X plane")
    ys = tile_starts(image.shape[0], tile_size, overlap)
    xs = tile_starts(image.shape[1], tile_size, overlap)
    coordinates = [(y, x) for y in ys for x in xs]
    output = np.zeros(image.shape, dtype=np.float32)
    weights = np.zeros(image.shape, dtype=np.float32)
    requested_batch_size = batch_size
    current = 0
    batch_reductions: list[dict[str, int]] = []
    while current < len(coordinates):
        checkpoint(cancellation, "tiled_prediction_batch")
        active_coordinates = coordinates[current : current + batch_size]
        tiles = np.stack(
            [image[y : y + tile_size, x : x + tile_size] for y, x in active_coordinates]
        )
        try:
            predictions = np.asarray(predict_batch(tiles), dtype=np.float32)
        except Exception as exc:
            if _is_accelerator_oom(exc) and batch_size > 1:
                previous = batch_size
                batch_size = max(1, batch_size // 2)
                batch_reductions.append({"from": previous, "to": batch_size})
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if emitter:
                    emitter.warning(
                        "tiling",
                        f"Inference tile batch reduced from {previous} to {batch_size} after OOM",
                    )
                continue
            raise
        if predictions.shape != tiles.shape:
            raise ValueError("batched tile predictor changed batch or spatial shape")
        for prediction, (y, x) in zip(predictions, active_coordinates, strict=True):
            checkpoint(cancellation, "blending")
            weight = _weight((prediction.shape[-2], prediction.shape[-1]), blend)
            output[y : y + prediction.shape[-2], x : x + prediction.shape[-1]] += (
                prediction * weight
            )
            weights[y : y + prediction.shape[-2], x : x + prediction.shape[-1]] += weight
            current += 1
        if emitter:
            emitter.progress(
                "tiling",
                f"Predicted tiles {current}/{len(coordinates)} in batches",
                current,
                len(coordinates),
                info={"tile_batch_size": batch_size},
            )
    checkpoint(cancellation, "blending")
    result = np.ascontiguousarray(output / np.maximum(weights, 1e-8), dtype=np.float32)
    return result, {
        "inference_tile_count": len(coordinates),
        "inference_tile_batch_size_requested": requested_batch_size,
        "inference_tile_batch_size": batch_size,
        "inference_tile_batch_reductions": batch_reductions,
    }
