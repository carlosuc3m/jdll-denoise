from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from jdll_denoise import denoise, resolve_config
from jdll_denoise.methods.zs_n2n import (
    ZSN2NNetwork,
    canonical_inference,
    canonical_losses,
    pair_downsampler,
)


def _reference_pairs(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    channels = value.shape[1]
    first_kernel = torch.tensor(
        [[0.0, 0.5], [0.5, 0.0]], dtype=value.dtype, device=value.device
    ).reshape(1, 1, 2, 2)
    second_kernel = torch.tensor(
        [[0.5, 0.0], [0.0, 0.5]], dtype=value.dtype, device=value.device
    ).reshape(1, 1, 2, 2)
    return (
        F.conv2d(value, first_kernel.repeat(channels, 1, 1, 1), stride=2, groups=channels),
        F.conv2d(value, second_kernel.repeat(channels, 1, 1, 1), stride=2, groups=channels),
    )


def _reference_losses(
    model: nn.Module, noisy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    first, second = _reference_pairs(noisy)
    prediction_first = first - model(first)
    prediction_second = second - model(second)
    full = noisy - model(noisy)
    full_first, full_second = _reference_pairs(full)
    residual = 0.5 * (F.mse_loss(first, prediction_second) + F.mse_loss(second, prediction_first))
    consistency = 0.5 * (
        F.mse_loss(prediction_first, full_first) + F.mse_loss(prediction_second, full_second)
    )
    return residual + consistency, residual, consistency


def test_exact_reference_architecture_and_parameter_count() -> None:
    model = ZSN2NNetwork(channels=1)
    layers = list(model.layers)
    assert [type(layer) for layer in layers] == [
        nn.Conv2d,
        nn.LeakyReLU,
        nn.Conv2d,
        nn.LeakyReLU,
        nn.Conv2d,
    ]
    convolutions = [layer for layer in layers if isinstance(layer, nn.Conv2d)]
    assert [(layer.in_channels, layer.out_channels) for layer in convolutions] == [
        (1, 48),
        (48, 48),
        (48, 1),
    ]
    assert [layer.kernel_size for layer in convolutions] == [(3, 3), (3, 3), (1, 1)]
    assert all(layer.bias is not None for layer in convolutions)
    assert [layer.negative_slope for layer in layers if isinstance(layer, nn.LeakyReLU)] == [
        0.2,
        0.2,
    ]
    assert sum(parameter.numel() for parameter in model.parameters()) == 21_313


def test_standard_effort_widths_and_budgets() -> None:
    for effort, width, steps in [
        ("quick", 8, 500),
        ("balanced", 16, 1000),
        ("balanced_high", 24, 2000),
        ("thorough", 48, 3000),
    ]:
        resolved = resolve_config({"method": "zs_n2n", "effort": effort}, (16, 16))
        assert resolved["method_config"]["hidden_width"] == width
        assert resolved["method_config"]["steps"] == steps


@pytest.mark.parametrize("shape", [(1, 1, 6, 8), (1, 1, 5, 7)])
def test_pair_downsampler_matches_fixed_grouped_convolution(shape) -> None:
    source = torch.arange(np.prod(shape), dtype=torch.float32).square().reshape(shape)
    actual = pair_downsampler(source)
    expected = _reference_pairs(source)
    assert actual[0].shape[-2:] == (shape[-2] // 2, shape[-1] // 2)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_pair_downsampler_never_mixes_channels() -> None:
    source = torch.zeros((1, 3, 4, 4))
    source[:, 0] = 1
    source[:, 1] = 10
    source[:, 2] = 100
    first, second = pair_downsampler(source)
    for result in (first, second):
        torch.testing.assert_close(result[:, 0], torch.ones_like(result[:, 0]))
        torch.testing.assert_close(result[:, 1], torch.full_like(result[:, 1], 10))
        torch.testing.assert_close(result[:, 2], torch.full_like(result[:, 2], 100))


def test_all_canonical_loss_components_match_independent_reference() -> None:
    torch.manual_seed(12)
    model = ZSN2NNetwork(hidden_width=5)
    noisy = torch.rand((1, 1, 6, 8))
    actual = canonical_losses(model, noisy)
    expected = _reference_losses(model, noisy)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_value, expected_value, rtol=1e-7, atol=1e-8)
    assert actual[0] > actual[1]  # regression guard: consistency is present in total


@pytest.mark.parametrize("shape", [(1, 1, 6, 8), (1, 1, 5, 7), (2, 3, 6, 8), (2, 3, 5, 7)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("groups", [1, 2])
def test_batched_loss_gradients_and_update_match_independent_equations(shape, dtype, groups):
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(5489)
        reference = ZSN2NNetwork(shape[1], 4, middle_groups=groups).to(dtype=dtype)
        batched, unbatched = copy.deepcopy(reference), copy.deepcopy(reference)
        source = torch.rand(shape, dtype=dtype, requires_grad=True)
        reference_losses = _reference_losses(reference, source)
        reference_losses[0].backward()
        tolerance = (
            {"rtol": 1e-10, "atol": 1e-12}
            if dtype == torch.float64
            else {"rtol": 2e-5, "atol": 1e-7}
        )
        for model, batch_pairs in ((batched, True), (unbatched, False)):
            value = source.detach().clone().requires_grad_(True)
            calls = []
            model.register_forward_hook(
                lambda m, inputs, output, calls=calls: calls.append(inputs[0].shape)
            )
            losses = canonical_losses(model, value, batch_pairs=batch_pairs)
            for loss, expected in zip(losses, reference_losses, strict=True):
                torch.testing.assert_close(loss, expected, **tolerance)
            losses[0].backward()
            torch.testing.assert_close(value.grad, source.grad, **tolerance)
            for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
                torch.testing.assert_close(actual.grad, expected.grad, **tolerance)
            assert len(calls) == (2 if batch_pairs else 3)
            assert calls[0] == (
                shape[0] * (2 if batch_pairs else 1),
                shape[1],
                shape[2] // 2,
                shape[3] // 2,
            )
            assert calls[-1] == shape
        # One Adam update, not one update per view or per loss term.
        for model in (reference, batched, unbatched):
            torch.optim.Adam(model.parameters(), lr=0.001).step()
        for model in (batched, unbatched):
            for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
                torch.testing.assert_close(actual, expected, **tolerance)
    finally:
        torch.set_num_threads(original_threads)


def test_no_clamp_in_training_graph(monkeypatch):
    model = ZSN2NNetwork(hidden_width=4)
    with torch.no_grad():
        model.layers[4].bias.fill_(10)

    def forbidden(*args, **kwargs):
        raise AssertionError("clamp must only occur at final inference")

    monkeypatch.setattr(torch, "clamp", forbidden)
    total, residual, consistency = canonical_losses(model, torch.ones(1, 1, 4, 4), True)
    assert residual > 1  # Predictions outside [0, 1] were not clamped.
    total.backward()
    assert model.layers[4].bias.grad.abs().item() > 0


@pytest.mark.parametrize("component", [1, 2])
def test_each_loss_component_reaches_every_trainable_layer(component: int) -> None:
    torch.manual_seed(4)
    model = ZSN2NNetwork(hidden_width=4)
    noisy = torch.rand((1, 1, 6, 6))
    losses = canonical_losses(model, noisy)
    losses[component].backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) > 0


def test_one_adam_scheduler_step_matches_direct_transcription() -> None:
    torch.manual_seed(21)
    production = ZSN2NNetwork(hidden_width=4)
    reference = copy.deepcopy(production)
    noisy = torch.rand((1, 1, 6, 8))
    production_optimizer = torch.optim.Adam(production.parameters(), lr=1e-3)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=1e-3)
    production_scheduler = torch.optim.lr_scheduler.StepLR(
        production_optimizer, step_size=1000, gamma=0.5
    )
    reference_scheduler = torch.optim.lr_scheduler.StepLR(
        reference_optimizer, step_size=1000, gamma=0.5
    )

    production_optimizer.zero_grad(set_to_none=True)
    canonical_losses(production, noisy)[0].backward()
    production_optimizer.step()
    production_scheduler.step()

    reference_optimizer.zero_grad(set_to_none=True)
    _reference_losses(reference, noisy)[0].backward()
    reference_optimizer.step()
    reference_scheduler.step()

    for actual, expected in zip(production.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_step_lr_call_order() -> None:
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    used_rates = []
    for _ in range(3000):
        used_rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert used_rates == [0.001] * 1000 + [0.0005] * 1000 + [0.00025] * 1000


def test_canonical_inference_is_external_residual_and_clamp() -> None:
    torch.manual_seed(9)
    model = ZSN2NNetwork(hidden_width=3)
    noisy = torch.rand((1, 1, 5, 7))
    expected = torch.clamp(noisy - model(noisy), 0, 1)
    torch.testing.assert_close(canonical_inference(model, noisy), expected)


def test_api_reports_losses_scheduler_and_canonical_status() -> None:
    events: list[dict] = []
    image = np.random.default_rng(3).random((7, 9), dtype=np.float32)
    result = denoise(
        image,
        {
            "method": "zs_n2n",
            "axes": "yx",
            "effort": "quick",
            "method_config": {"steps": 1, "report_interval": 1},
        },
        callbacks=events.append,
    )
    optimization = [event for event in events if event["phase"] == "optimization"][-1]
    assert {"loss_total", "loss_residual", "loss_consistency"} <= set(optimization["metrics"])
    job = result["metadata"]["jobs"][0]
    assert result["denoised"].shape == image.shape
    assert job["fidelity_status"] == "jdll_performance_variant_custom_iteration_budget"
    assert job["optimizer"] == "Adam"
    assert job["scheduler"] == "StepLR"
    assert job["completed_iterations"] == 1


def test_native_rgb_joint_network_and_noncanonical_extension_metadata() -> None:
    image = np.random.default_rng(8).random((7, 9, 3), dtype=np.float32)
    result = denoise(
        image,
        {
            "method": "zs_n2n",
            "axes": "yxc",
            "channel_mode": "native_color",
            "tiling": {"enabled": True, "tile_size": 32, "overlap": 4},
            "method_config": {"steps": 1, "hidden_width": 4},
        },
    )
    assert result["denoised"].shape == image.shape
    assert result["metadata"]["channel_mode"] == "native_color"
    job = result["metadata"]["jobs"][0]
    # Omitted effort is now Balanced, whose middle convolution has two groups.
    assert job["network_parameter_count"] == 4 * 3 * 3 * 3 + 4 + 4 * 2 * 3 * 3 + 4 + 4 * 3 + 3
    assert job["fidelity_status"] == "noncanonical_extension"
    assert job["inference_tiled"] is True


def test_native_rgb_stack_and_time_axes_are_restored() -> None:
    image = np.random.default_rng(18).random((2, 3, 2, 6, 8), dtype=np.float32)
    result = denoise(
        image,
        {
            "method": "zs_n2n",
            "axes": "tczyx",
            "channel_mode": "native_color",
            "method_config": {"steps": 1, "hidden_width": 2},
        },
    )
    assert result["denoised"].shape == image.shape
    assert len(result["metadata"]["jobs"]) == 2
    assert all(len(job["planes"]) == 2 for job in result["metadata"]["jobs"])
    assert result["metadata"]["dimension_mode"] == "slice_wise_2d"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_zs_n2n_cuda_inputs_and_model_stay_on_cuda() -> None:
    image = np.random.default_rng(22).random((8, 8), dtype=np.float32)
    result = denoise(
        image,
        {
            "method": "zs_n2n",
            "axes": "yx",
            "device": "cuda",
            "method_config": {"steps": 1, "hidden_width": 4},
        },
    )
    job = result["metadata"]["jobs"][0]
    assert job["model_device"].startswith("cuda")
    assert job["training_input_device"].startswith("cuda")
    assert job["prediction_input_device"].startswith("cuda")
    assert result["metadata"]["device_used"] == "cuda"


def test_pinned_reference_regression_fixture() -> None:
    """Values generated by a standalone transcription pinned to torch 2.5 CPU."""

    class ReferenceNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(1, 48, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(48, 48, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(48, 1, 1),
            )

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.layers(value)

    noisy = torch.linspace(0, 1, 16).reshape(1, 1, 4, 4)
    expected_first = torch.tensor([1 / 6, 0.3, 0.7, 5 / 6]).reshape(1, 1, 2, 2)
    expected_initial = torch.tensor(
        [0.0019003646448254585, 0.0013812663964927197, 0.0005190982483327389]
    )
    expected_output = torch.tensor(
        [
            0.0,
            0.0428345874,
            0.1132772863,
            0.1722685695,
            0.2584914267,
            0.3017035127,
            0.3854316771,
            0.4257403910,
            0.4962853193,
            0.5635719299,
            0.6322836280,
            0.6892294884,
            0.7811433673,
            0.8625071049,
            0.9202211499,
            0.9669738412,
        ]
    ).reshape_as(noisy)

    torch.manual_seed(123)
    model = ReferenceNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    first, _ = _reference_pairs(noisy)
    torch.testing.assert_close(first, expected_first, atol=1e-7, rtol=0)
    initial = torch.stack(_reference_losses(model, noisy))
    torch.testing.assert_close(initial, expected_initial, atol=2e-8, rtol=0)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        _reference_losses(model, noisy)[0].backward()
        optimizer.step()
        scheduler.step()
    output = torch.clamp(noisy - model(noisy), 0, 1)
    torch.testing.assert_close(output, expected_output, atol=2e-6, rtol=0)
