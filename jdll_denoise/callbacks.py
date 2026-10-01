"""Structured callback emission with default failure isolation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

Callback = Callable[[dict[str, Any]], None]
Callbacks = Callback | Iterable[Callback] | dict[str, Callback] | None


class EventEmitter:
    def __init__(self, callbacks: Callbacks):
        if callbacks is None:
            self._callbacks: list[Callback] = []
        elif callable(callbacks):
            self._callbacks = [callbacks]
        elif isinstance(callbacks, dict):
            self._callbacks = list(callbacks.values())
        else:
            self._callbacks = list(callbacks)
        if not all(callable(callback) for callback in self._callbacks):
            raise TypeError("every callback must be callable")
        self.failures: list[str] = []

    def emit(self, event_type: str, phase: str, message: str, **fields: Any) -> None:
        event: dict[str, Any] = {
            "type": event_type,
            "phase": phase,
            "method": fields.pop("method", None),
            "message": message,
            "current": fields.pop("current", 0),
            "maximum": fields.pop("maximum", 0),
            "channel": fields.pop("channel", 0),
            "channels": fields.pop("channels", 1),
            "z": fields.pop("z", 0),
            "z_total": fields.pop("z_total", 1),
            "metrics": fields.pop("metrics", {}),
            "info": fields.pop("info", {}),
        }
        event.update(fields)
        for callback in self._callbacks:
            try:
                callback(dict(event))
            except Exception as exc:  # callbacks cannot invalidate a completed calculation
                self.failures.append(f"callback {callback!r} failed: {exc}")

    def status(self, phase: str, message: str, **fields: Any) -> None:
        self.emit("status", phase, message, **fields)

    def progress(self, phase: str, message: str, current: int, maximum: int, **fields: Any) -> None:
        self.emit(
            "denoise_progress",
            phase,
            message,
            current=current,
            maximum=maximum,
            **fields,
        )

    def warning(self, phase: str, message: str, **fields: Any) -> None:
        self.emit("warning", phase, message, **fields)
