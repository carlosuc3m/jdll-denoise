"""Best-effort runtime cache cleanup."""

from __future__ import annotations

import gc

import torch


def cleanup_device(device: str | None = None) -> None:
    gc.collect()
    try:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if torch.cuda.is_initialized():
                torch.cuda.ipc_collect()
        elif device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    except Exception:
        # Cleanup must never hide the calculation's outcome.
        pass
