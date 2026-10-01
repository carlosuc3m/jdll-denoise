"""Method dispatch kept separate from the public orchestration layer."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from . import bm, noise2fast, structn2v, zs_n2n
from .base import MethodContext, MethodResult

Method = Callable[[np.ndarray, MethodContext], MethodResult]

METHODS: dict[str, Method] = {
    "bm3d_bm4d": bm.run,
    "zs_n2n": zs_n2n.run,
    "noise2fast": noise2fast.run,
    "structn2v": structn2v.run,
}
