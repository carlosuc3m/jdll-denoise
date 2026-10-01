"""Shared method interfaces, seeding, and optimization progress events."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..callbacks import EventEmitter
from ..cancellation import CancellationLike


@dataclass
class MethodContext:
    config: dict[str, Any]
    emitter: EventEmitter
    cancellation: CancellationLike
    seed: int
    channel: int = 0
    channels: int = 1
    time: int = 0
    times: int = 1


@dataclass
class MethodResult:
    data: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def optimization_event(
    context: MethodContext,
    method_label: str,
    step: int,
    steps: int,
    loss: float,
    smooth_loss: float,
    learning_rate: float,
    elapsed: float,
    best_step: int,
    extra_metrics: dict[str, float | int] | None = None,
) -> None:
    metrics: dict[str, float | int] = {
        "loss": loss,
        "smoothed_loss": smooth_loss,
        "learning_rate": learning_rate,
        "elapsed_seconds": elapsed,
        "best_step": best_step,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    context.emitter.progress(
        "optimization",
        f"Training {method_label} - step {step}/{steps}",
        step,
        steps,
        method=context.config["method"],
        channel=context.channel + 1,
        channels=context.channels,
        metrics=metrics,
    )
