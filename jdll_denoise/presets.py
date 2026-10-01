"""Single source of truth for effort presets."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

ZS_N2N_EFFORT_PRESETS_VERSION = 1

PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "bm3d_bm4d": {
        "quick": {"profile": "np", "stages": "hard_thresholding"},
        "balanced": {"profile": "np", "stages": "all"},
        "thorough": {"profile": "high", "stages": "all"},
    },
    "zs_n2n": {
        "quick": {
            "steps": 500,
            "learning_rate": 1e-3,
            "hidden_width": 8,
            "middle_groups": 1,
            "output_init": "near_zero",
            "output_init_std": 0.001,
            "batch_pairs": True,
            "scheduler_step_size": 1000,
            "scheduler_gamma": 0.5,
        },
        "balanced": {
            "steps": 1000,
            "learning_rate": 1e-3,
            "hidden_width": 16,
            "middle_groups": 2,
            "output_init": "near_zero",
            "output_init_std": 0.001,
            "batch_pairs": True,
            "scheduler_step_size": 1000,
            "scheduler_gamma": 0.5,
        },
        "balanced_high": {
            "steps": 2000,
            "learning_rate": 1e-3,
            "hidden_width": 24,
            "middle_groups": 1,
            "output_init": "native",
            "output_init_std": 0.001,
            "batch_pairs": True,
            "scheduler_step_size": 1000,
            "scheduler_gamma": 0.5,
        },
        "thorough": {
            "steps": 3000,
            "learning_rate": 1e-3,
            "hidden_width": 48,
            "middle_groups": 1,
            "output_init": "native",
            "output_init_std": 0.001,
            "batch_pairs": True,
            "scheduler_step_size": 1000,
            "scheduler_gamma": 0.5,
        },
    },
    "noise2fast": {
        "quick": {"hidden_width": 16, "patience": 50},
        "balanced": {"hidden_width": 24, "patience": 100},
        "thorough": {"hidden_width": 64, "patience": 100},
    },
    "structn2v": {
        "quick": {"architecture": "paper_unet"},
        "balanced": {"architecture": "paper_unet"},
        "thorough": {"architecture": "paper_unet"},
    },
}


def method_preset(method: str, effort: str) -> dict[str, Any]:
    return deepcopy(PRESETS[method][effort])
