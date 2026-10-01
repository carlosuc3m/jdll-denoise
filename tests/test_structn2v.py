from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from jdll_denoise import denoise
from jdll_denoise.config import resolve_config
from jdll_denoise.methods.structn2v import (
    StructN2VUNet,
    _masked_loss,
    make_structured_blind_mask,
    prepare_structn2v_batch,
    select_noise_structure,
    structured_masks,
)


def test_2d_network_matches_paper_architecture() -> None:
    model = StructN2VUNet(2)
    convolutions = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    assert [layer.out_channels for layer in convolutions] == [
        16,
        16,
        32,
        32,
        64,
        64,
        32,
        32,
        16,
        16,
        1,
    ]
    assert all(layer.kernel_size == (3, 3) for layer in convolutions[:-1])
    assert all(torch.count_nonzero(layer.bias) == 0 for layer in convolutions)
    pool_sizes = [
        module.kernel_size for module in model.modules() if isinstance(module, nn.MaxPool2d)
    ]
    assert pool_sizes == [
        (2, 2),
        (2, 2),
    ]
    assert model(torch.zeros(1, 1, 16, 20)).shape == (1, 1, 16, 20)


def test_3d_network_matches_paper_anisotropic_architecture() -> None:
    model = StructN2VUNet(3)
    convolutions = [module for module in model.modules() if isinstance(module, nn.Conv3d)]
    assert all(layer.kernel_size == (3, 5, 5) for layer in convolutions[:-1])
    assert all(torch.count_nonzero(layer.bias) == 0 for layer in convolutions)
    pool_sizes = [
        module.kernel_size for module in model.modules() if isinstance(module, nn.MaxPool3d)
    ]
    assert pool_sizes == [
        (1, 2, 2),
        (1, 2, 2),
    ]
    assert model(torch.zeros(1, 1, 3, 16, 20)).shape == (1, 1, 3, 16, 20)


@pytest.mark.parametrize(
    ("structure", "shape"),
    [
        ("horizontal", (1, 9)),
        ("vertical", (9, 1)),
        ("local_isotropic", (9, 9)),
        ("periodic_grid", (9, 9)),
    ],
)
def test_structured_mask_geometry(structure: str, shape: tuple[int, ...]) -> None:
    mask = make_structured_blind_mask(structure, 2, radius=4, period=4)
    assert mask.shape == shape
    assert mask[tuple(size // 2 for size in shape)]


def test_structured_mask_does_not_wrap_across_borders() -> None:
    found = False
    for seed in range(1000):
        centers, corrupted = structured_masks(
            (5, 5),
            structure="horizontal",
            fraction=0.04,
            radius=1,
            period=2,
            rng=np.random.default_rng(seed),
        )
        if centers[0, 0]:
            found = True
            assert corrupted[0, 0]
            assert corrupted[0, 1]
            assert not corrupted[0, 4]
            break
    assert found


def test_batch_preparation_keeps_targets_and_scores_only_active_pixels() -> None:
    patches = np.arange(64, dtype=np.float32).reshape(1, 8, 8) / 63
    inputs, targets, loss_mask = prepare_structn2v_batch(
        patches,
        blind_mask=make_structured_blind_mask("horizontal", 2, radius=2, period=4),
        active_fraction=0.02,
        neighborhood_radius=2,
        rng=np.random.default_rng(11),
    )
    np.testing.assert_array_equal(targets, patches)
    assert loss_mask.sum() == 1
    assert inputs.shape == targets.shape == loss_mask.shape
    assert np.any(inputs != targets)


def test_loss_ignores_every_unmasked_target_pixel() -> None:
    inputs = np.zeros((1, 4, 4), dtype=np.float32)
    targets = np.full((1, 4, 4), 100.0, dtype=np.float32)
    targets[0, 1, 2] = 2.0
    mask = np.zeros((1, 4, 4), dtype=bool)
    mask[0, 1, 2] = True
    loss = _masked_loss(nn.Identity(), inputs, targets, mask, "cpu")
    assert float(loss) == 4.0


def test_paper_schedules_resolve_for_2d_and_3d() -> None:
    two_d = resolve_config({"method": "structn2v", "axes": "yx"}, (512, 512))[
        "method_config"
    ]
    assert two_d["patch_size"] == [256, 256]
    assert two_d["patch_count"] == 1600
    assert two_d["batch_size"] == 4
    assert two_d["epochs"] == 600
    assert two_d["learning_rate"] == 2e-5
    assert two_d["mask_fraction"] == 0.02
    assert resolve_config({"method": "structn2v", "axes": "yx"}, (512, 512))[
        "normalization"
    ]["clip"] is False

    three_d = resolve_config(
        {"method": "structn2v", "axes": "zyx", "dimensions": "3d"}, (32, 512, 512)
    )["method_config"]
    assert three_d["patch_size"] == [32, 64, 64]
    assert three_d["patch_count"] == 1900
    assert three_d["batch_size"] == 10
    assert three_d["epochs"] == 100
    assert three_d["learning_rate"] == 4e-6


def test_auto_structure_ranking_is_deterministic() -> None:
    image = np.random.default_rng(7).random((32, 32), dtype=np.float32)
    assert select_noise_structure(image, 4) == select_noise_structure(image, 4)


def test_api_uses_masked_paper_model_and_reports_explicit_short_run() -> None:
    image = np.random.default_rng(3).random((8, 8), dtype=np.float32)
    config = {
        "method": "structn2v",
        "axes": "yx",
        "device": "cpu",
        "seed": 7,
        "method_config": {
            "noise_structure": "horizontal",
            "patch_size": [8, 8],
            "patch_count": 1,
            "batch_size": 1,
            "epochs": 1,
            "steps": 1,
            "report_interval": 1,
        },
    }
    result = denoise(image, config)
    job = result["metadata"]["jobs"][0]
    assert result["denoised"].shape == image.shape
    assert job["architecture"] == "paper_2d_unet"
    assert job["training_loss"] == "masked_mse_at_active_pixels_only"
    assert job["blind_mask_shape"] == [1, 9]
    assert job["active_pixel_fraction"] == 0.02
    assert job["optimization_steps_completed"] == 1


def test_auto_selection_runs_all_candidates_on_heldout_masked_loss() -> None:
    image = np.random.default_rng(23).random((8, 8), dtype=np.float32)
    config = {
        "method": "structn2v",
        "axes": "yx",
        "device": "cpu",
        "seed": 5,
        "method_config": {
            "noise_structure": "auto",
            "patch_size": [8, 8],
            "patch_count": 1,
            "batch_size": 1,
            "epochs": 1,
            "steps": 1,
            "selection_steps": 1,
            "report_interval": 1,
        },
    }
    job = denoise(image, config)["metadata"]["jobs"][0]
    assert job["structure_selection"] == "heldout_masked_prediction_loss"
    assert set(job["structure_candidate_heldout_losses"]) == {
        "local_isotropic",
        "horizontal",
        "vertical",
        "periodic_grid",
    }
