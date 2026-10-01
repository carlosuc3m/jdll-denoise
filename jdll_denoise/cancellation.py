"""Backend-neutral cooperative cancellation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from .errors import ConfigurationError, DenoisingCancelled


@runtime_checkable
class CancellationToken(Protocol):
    def is_cancelled(self) -> bool: ...


CancellationLike = CancellationToken | Callable[[], bool] | None


def is_cancelled(cancellation: CancellationLike) -> bool:
    if cancellation is None:
        return False
    try:
        if isinstance(cancellation, CancellationToken):
            return bool(cancellation.is_cancelled())
        if callable(cancellation):
            return bool(cancellation())
    except DenoisingCancelled:
        raise
    except Exception as exc:
        raise ConfigurationError(f"cancellation callback failed: {exc}") from exc
    raise ConfigurationError("cancellation must be callable or expose is_cancelled()")


def checkpoint(cancellation: CancellationLike, phase: str) -> None:
    if is_cancelled(cancellation):
        raise DenoisingCancelled(f"denoising cancelled during {phase}; no partial result returned")
