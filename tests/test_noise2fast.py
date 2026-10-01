from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from jdll_denoise import ConfigurationError, DenoisingCancelled, denoise
from jdll_denoise.config import resolve_config
from jdll_denoise.methods import noise2fast
from jdll_denoise.methods.noise2fast import Noise2FastNet, checkerboard_pairs


def _config(steps: int = 2) -> dict:
    return {
        "method": "noise2fast",
        "axes": "yx",
        "effort": "quick",
        "device": "cpu",
        "seed": 31,
        "tiling": {"enabled": False},
        "method_config": {
            "steps": steps,
            "report_interval": max(1, steps),
        },
    }


@pytest.mark.parametrize("width", [16, 24, 64])
def test_network_keeps_all_layers_and_receptive_field(width: int) -> None:
    model = Noise2FastNet(hidden_width=width)
    convolutions = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    assert len(convolutions) == 9
    assert convolutions[0].in_channels == 1
    assert all(layer.out_channels == width for layer in convolutions[:8])
    assert all(layer.in_channels == width for layer in convolutions[1:])
    assert all(layer.kernel_size == (3, 3) for layer in convolutions[:8])
    assert all(layer.padding == (1, 1) for layer in convolutions[:8])
    assert all(layer.stride == (1, 1) and layer.dilation == (1, 1) for layer in convolutions)
    assert sum(layer.kernel_size[0] - 1 for layer in convolutions) + 1 == 17
    assert (
        sum(parameter.numel() for parameter in model.parameters()) == 63 * width**2 + 18 * width + 1
    )
    assert convolutions[-1].kernel_size == (1, 1)
    assert convolutions[-1].out_channels == 1
    assert not any(isinstance(module, (nn.MaxPool2d, nn.BatchNorm2d)) for module in model.modules())
    output = model(torch.zeros(2, 1, 5, 7))
    assert output.shape == (2, 1, 5, 7)
    assert torch.all((0 <= output) & (output <= 1))


def test_direct_constructor_still_defaults_to_64_features() -> None:
    assert Noise2FastNet().blocks[0].conv1.out_channels == 64


def test_defaults_cannot_resolve_to_the_removed_patch_or_tiled_approximation() -> None:
    resolved = resolve_config({"method": "noise2fast", "axes": "yx"}, (1024, 1024))
    assert resolved["method_config"]["steps"] == 0
    assert resolved["method_config"]["patience"] == 100
    assert resolved["method_config"]["output_window"] == 100
    assert resolved["normalization"]["low"] == 0.0
    assert resolved["normalization"]["high"] == 100.0
    assert resolved["tiling"]["enabled"] is False
    with pytest.raises(ConfigurationError, match="unknown method_config"):
        resolve_config(
            {
                "method": "noise2fast",
                "axes": "yx",
                "method_config": {"training_patch_size": [256, 256]},
            },
            (1024, 1024),
        )
    with pytest.raises(ConfigurationError, match="paper fidelity"):
        resolve_config(
            {"method": "noise2fast", "axes": "yx", "tiling": {"enabled": True}},
            (1024, 1024),
        )


def test_four_directed_checkerboard_pairs_match_paper_equations() -> None:
    source = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    pairs = checkerboard_pairs(source)
    assert len(pairs) == 4
    up_even = torch.tensor([[[[0, 5, 2, 7], [8, 13, 10, 15]]]], dtype=torch.float32)
    up_odd = torch.tensor([[[[4, 1, 6, 3], [12, 9, 14, 11]]]], dtype=torch.float32)
    left_odd = torch.tensor([[[[1, 3], [4, 6], [9, 11], [12, 14]]]], dtype=torch.float32)
    left_even = torch.tensor([[[[0, 2], [5, 7], [8, 10], [13, 15]]]], dtype=torch.float32)
    expected = (
        (up_even, up_odd),
        (up_odd, up_even),
        (left_odd, left_even),
        (left_even, left_odd),
    )
    for actual, wanted in zip(pairs, expected, strict=True):
        assert torch.equal(actual[0], wanted[0])
        assert torch.equal(actual[1], wanted[1])


def test_checkerboard_training_crop_matches_reference_for_odd_shapes() -> None:
    pairs = checkerboard_pairs(torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5))
    assert pairs[0][0].shape == (1, 1, 2, 4)
    assert pairs[2][0].shape == (1, 1, 4, 2)
    used = torch.cat((pairs[0][0].flatten(), pairs[0][1].flatten())).tolist()
    assert 24.0 not in used


def test_api_reports_canonical_training_and_reconstruction() -> None:
    image = np.random.default_rng(4).random((8, 8), dtype=np.float32)
    result = denoise(image, _config())
    job = result["metadata"]["jobs"][0]
    assert result["denoised"].shape == image.shape
    assert result["denoised"].dtype == np.float32
    assert job["architecture"] == "8x3x3_conv_16_relu+1x1_sigmoid"
    assert job["hidden_width"] == 16
    assert job["validation_patience"] == 50
    assert job["training_pair_count"] == 4
    assert job["training_batch_size"] == 1
    assert job["training_loss"] == "binary_cross_entropy"
    assert job["learning_rate"] == 1e-3
    assert job["validation_metric"] == "negative_full_image_mse_to_noisy_input"
    assert job["output_average_count"] == 2
    assert job["stopped_by"] == "configured_step_limit"


def test_seeded_noise2fast_is_deterministic() -> None:
    image = np.random.default_rng(9).random((8, 8), dtype=np.float32)
    first = denoise(image, _config())["denoised"]
    second = denoise(image, _config())["denoised"]
    np.testing.assert_array_equal(first, second)


def test_validation_patience_stops_and_uses_paper_rolling_window() -> None:
    image = np.random.default_rng(3).random((4, 4), dtype=np.float32)
    config = _config(0)
    job = denoise(image, config)["metadata"]["jobs"][0]
    assert job["stopped_by"] == "validation_patience"
    assert job["epochs_without_validation_improvement"] == 50
    assert job["output_average_count"] == min(job["completed_steps"], 100)
    assert "warnings" not in job


def test_cancellation_is_checked_each_optimization_step() -> None:
    class Token:
        calls = 0

        def is_cancelled(self) -> bool:
            self.calls += 1
            return self.calls > 5

    with pytest.raises(DenoisingCancelled, match="optimization"):
        denoise(
            np.random.default_rng(12).random((8, 8), dtype=np.float32),
            _config(20),
            cancellation=Token(),
        )


def test_explicit_cpu_fallback_updates_device_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        "jdll_denoise.config.resolve_device", lambda *args, **kwargs: ("cuda", None)
    )

    def simulated_device_run(image, context, plane_index, device):
        if device == "cuda":
            raise noise2fast._DeviceOOM("simulated accelerator OOM")
        return image.copy(), {"device_used": "cpu", "phase_timings_seconds": {}}

    monkeypatch.setattr(noise2fast, "_run_plane_on_device", simulated_device_run)
    config = _config()
    config.update({"device": "cuda", "allow_cpu_fallback": True})
    result = denoise(np.random.default_rng(13).random((8, 8), dtype=np.float32), config)
    assert result["metadata"]["device_requested"] == "cuda"
    assert result["metadata"]["device_resolved"] == "cuda"
    assert result["metadata"]["device_used"] == "cpu"


@pytest.mark.parametrize(
    "effort,width,patience", [("quick", 16, 50), ("balanced", 24, 100), ("thorough", 64, 100)]
)
@pytest.mark.parametrize(
    "device",
    [
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
        ),
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS is unavailable"
            ),
        ),
    ],
)
def test_accelerator_execution_preserves_preset_and_devices(
    device, effort, width, patience
) -> None:
    image = np.random.default_rng(5).random((8, 8), dtype=np.float32)
    config = _config(1)
    config.update(device=device, effort=effort)
    result = denoise(image, config)
    job = result["metadata"]["jobs"][0]
    assert job["model_device"].startswith(device)
    assert job["training_input_device"].startswith(device)
    assert job["prediction_input_device"].startswith(device)
    assert job["hidden_width"] == width
    assert job["validation_patience"] == patience
