"""Invertible per-channel scientific intensity normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Normalizer:
    offset: float
    scale: float
    source_min: float
    source_max: float
    low_percentile: float
    high_percentile: float
    clip: bool
    constant: bool

    @classmethod
    def fit(cls, data: np.ndarray, config: dict[str, Any]) -> Normalizer:
        source = np.asarray(data)
        source_min = float(source.min())
        source_max = float(source.max())
        low_p = float(config["low"])
        high_p = float(config["high"])
        low, high = np.percentile(source.astype(np.float64), [low_p, high_p])
        constant = bool(source_min == source_max or high <= low)
        return cls(
            offset=float(low if not constant else source_min),
            scale=float(high - low if not constant else 1.0),
            source_min=source_min,
            source_max=source_max,
            low_percentile=low_p,
            high_percentile=high_p,
            clip=bool(config["clip"]),
            constant=constant,
        )

    def normalize(self, data: np.ndarray) -> np.ndarray:
        result = (np.asarray(data, dtype=np.float32) - self.offset) / self.scale
        if self.clip:
            result = np.clip(result, 0.0, 1.0)
        return np.ascontiguousarray(result, dtype=np.float32)

    def invert(self, data: np.ndarray) -> np.ndarray:
        if self.constant:
            return np.full(np.asarray(data).shape, self.source_min, dtype=np.float32)
        result = np.asarray(data, dtype=np.float32) * self.scale + self.offset
        return np.ascontiguousarray(result, dtype=np.float32)

    def metadata(self) -> dict[str, Any]:
        return {
            "mode": "percentile",
            "offset": self.offset,
            "scale": self.scale,
            "source_min": self.source_min,
            "source_max": self.source_max,
            "low": self.low_percentile,
            "high": self.high_percentile,
            "clip": self.clip,
            "constant": self.constant,
        }
