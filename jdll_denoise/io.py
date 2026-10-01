"""Array validation and axis-safe job iteration."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .config import validate_axes
from .errors import InputFormatError


@dataclass(frozen=True)
class ArrayJob:
    index: tuple[int | slice, ...]
    data: np.ndarray
    channel: int
    channels: int
    time: int
    times: int


@dataclass(frozen=True)
class ArrayLayout:
    array: np.ndarray
    axes: str

    @classmethod
    def from_input(cls, image: object, axes: str) -> ArrayLayout:
        try:
            array = np.asarray(image)
        except Exception as exc:
            raise InputFormatError(
                f"phase=validating: input cannot be converted to an array: {exc}"
            ) from exc
        try:
            axes = validate_axes(axes, array.ndim)
        except Exception as exc:
            raise InputFormatError(f"phase=validating: {exc}") from exc
        if array.size == 0 or any(size == 0 for size in array.shape):
            raise InputFormatError(
                "phase=validating: empty arrays and zero-sized dimensions are unsupported"
            )
        if array.dtype.kind not in "biuf":
            raise InputFormatError(
                f"phase=validating: numeric arrays are required; got dtype={array.dtype}"
            )
        if not np.isfinite(array).all():
            raise InputFormatError(
                "phase=validating: input contains NaN or Inf; repair it before denoising"
            )
        return cls(array=array, axes=axes)

    def jobs(self) -> Iterator[ArrayJob]:
        c_axis = self.axes.find("c")
        t_axis = self.axes.find("t")
        channels = self.array.shape[c_axis] if c_axis >= 0 else 1
        times = self.array.shape[t_axis] if t_axis >= 0 else 1
        for time in range(times):
            for channel in range(channels):
                index: list[int | slice] = [slice(None)] * self.array.ndim
                if t_axis >= 0:
                    index[t_axis] = time
                if c_axis >= 0:
                    index[c_axis] = channel
                data = np.asarray(self.array[tuple(index)])
                remaining_axes = "".join(axis for axis in self.axes if axis not in {"t", "c"})
                target_axes = "zyx" if "z" in remaining_axes else "yx"
                permutation = tuple(remaining_axes.index(axis) for axis in target_axes)
                data = np.transpose(data, permutation)
                yield ArrayJob(tuple(index), data, channel, channels, time, times)

    def joint_channel_jobs(self) -> Iterator[ArrayJob]:
        """Yield time jobs with C retained for validated native-color methods."""
        c_axis = self.axes.find("c")
        if c_axis < 0:
            raise InputFormatError("channel_mode=native_color requires an explicit c axis")
        t_axis = self.axes.find("t")
        channels = self.array.shape[c_axis]
        times = self.array.shape[t_axis] if t_axis >= 0 else 1
        for time in range(times):
            index: list[int | slice] = [slice(None)] * self.array.ndim
            if t_axis >= 0:
                index[t_axis] = time
            data = np.asarray(self.array[tuple(index)])
            remaining_axes = self.axes.replace("t", "")
            target_axes = "czyx" if "z" in remaining_axes else "cyx"
            permutation = tuple(remaining_axes.index(axis) for axis in target_axes)
            yield ArrayJob(tuple(index), np.transpose(data, permutation), 0, channels, time, times)

    def put(self, output: np.ndarray, job: ArrayJob, value: np.ndarray) -> None:
        remaining_axes = "".join(axis for axis in self.axes if axis not in {"t", "c"})
        source_axes = "zyx" if "z" in remaining_axes else "yx"
        permutation = tuple(source_axes.index(axis) for axis in remaining_axes)
        output[job.index] = np.transpose(value, permutation)

    def put_joint_channels(self, output: np.ndarray, job: ArrayJob, value: np.ndarray) -> None:
        remaining_axes = self.axes.replace("t", "")
        source_axes = "czyx" if "z" in remaining_axes else "cyx"
        permutation = tuple(source_axes.index(axis) for axis in remaining_axes)
        output[job.index] = np.transpose(value, permutation)
