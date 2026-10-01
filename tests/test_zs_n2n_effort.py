"""Version-1 ZS-N2N presets through the real resolver, network and public APIs."""

from __future__ import annotations

import importlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from jdll_denoise import (
    ConfigurationError,
    DenoisingCancelled,
    InputFormatError,
    capabilities,
    denoise,
    preview,
    resolve_config,
)
from jdll_denoise.errors import OutOfMemoryDenoisingError
from jdll_denoise.methods import zs_n2n
from jdll_denoise.normalization import Normalizer
from jdll_denoise.presets import PRESETS

CASES = [
    ("quick", 8, 1, 500, "near_zero", 673),
    ("balanced", 16, 2, 1000, "near_zero", 1345),
    ("balanced_high", 24, 1, 2000, "native", 5473),
    ("thorough", 48, 1, 3000, "native", 21313),
]


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def request(effort="balanced", **overrides):
    return {
        "method": "zs_n2n",
        "effort": effort,
        "axes": "yx",
        "device": "cpu",
        "seed": 5489,
        "method_config": overrides,
    }


@pytest.mark.parametrize("effort,width,groups,steps,initialization,parameters", CASES)
def test_presets_and_copies(effort, width, groups, steps, initialization, parameters):
    source = request(effort)
    original, table = deepcopy(source), deepcopy(PRESETS)
    resolved = resolve_config(source, (5, 7))
    assert resolved["effort_presets_version"] == 1
    assert resolved["method_config"] == {
        "hidden_width": width,
        "middle_groups": groups,
        "steps": steps,
        "output_init": initialization,
        "output_init_std": 0.001,
        "batch_pairs": True,
        "learning_rate": 0.001,
        "scheduler_step_size": 1000,
        "scheduler_gamma": 0.5,
        "report_interval": 10,
        "early_stopping": False,
        "restore_best_weights": False,
    }
    resolved["method_config"]["middle_groups"] = 99
    assert resolve_config(source, (5, 7))["method_config"]["middle_groups"] == groups
    assert source == original and PRESETS == table


def test_balanced_default_alias_and_capabilities_contract():
    resolved = resolve_config({"method": "zs_n2n"}, (5, 7))
    assert resolved["effort"] == "balanced" and resolved["method_config"]["hidden_width"] == 16
    assert resolve_config(request("balanced-high"), (5, 7))["effort"] == "balanced_high"
    cap = capabilities()["methods"]["zs_n2n"]
    assert cap["effort_presets_version"] == 1 and type(cap["effort_presets_version"]) is int
    assert cap["efforts"] == [case[0] for case in CASES]
    assert cap["available"] and cap["dimensions"] == ["2d"]
    assert cap["channel_modes"] == ["independent", "native_color"]
    cap["efforts"].clear()
    assert capabilities()["methods"]["zs_n2n"]["efforts"] == [case[0] for case in CASES]


@pytest.mark.parametrize("method", ["noise2fast", "bm3d_bm4d", "structn2v"])
@pytest.mark.parametrize("effort", ["balanced_high", "balanced-high"])
def test_balanced_high_rejected_for_other_methods(method, effort):
    with pytest.raises(ConfigurationError, match="effort must be one of"):
        resolve_config({"method": method, "effort": effort}, (5, 7))


@pytest.mark.parametrize("effort", [case[0] for case in CASES])
def test_explicit_overrides_win_without_mutation(effort):
    override = {
        "hidden_width": 12,
        "middle_groups": 3,
        "steps": 7,
        "output_init": "near_zero",
        "output_init_std": 0.002,
        "batch_pairs": False,
        "report_interval": 3,
        "learning_rate": 0.002,
        "scheduler_step_size": 2,
        "scheduler_gamma": 0.75,
    }
    source = request(effort, **override)
    original = deepcopy(source)
    actual = resolve_config(source, (5, 7))["method_config"]
    assert all(actual[key] == value for key, value in override.items())
    assert source == original
    alias = resolve_config(request(effort, features=12), (5, 7))
    assert alias["method_config"]["hidden_width"] == 12
    with pytest.raises(ConfigurationError, match="must agree"):
        resolve_config(request(effort, features=8, hidden_width=12), (5, 7))


@pytest.mark.parametrize(
    "key",
    [
        "hidden_width",
        "middle_groups",
        "steps",
        "scheduler_step_size",
        "report_interval",
        "patience",
    ],
)
@pytest.mark.parametrize("value", [True, False, 0, -1, 2.0, "2", None])
def test_invalid_integer_settings(key, value):
    with pytest.raises(ConfigurationError, match=rf"method_config\.{key}"):
        resolve_config(request(**{key: value}), (5, 7))


@pytest.mark.parametrize("key", ["batch_pairs", "early_stopping", "restore_best_weights"])
@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_boolean_settings_are_strict(key, value):
    with pytest.raises(ConfigurationError, match=rf"method_config\.{key}"):
        resolve_config(request(**{key: value}), (5, 7))


@pytest.mark.parametrize(
    "value", [True, False, 0, -0.1, "0.001", None, float("nan"), float("inf"), -float("inf")]
)
def test_invalid_head_standard_deviation(value):
    with pytest.raises(ConfigurationError, match="method_config.output_init_std"):
        resolve_config(request(output_init_std=value), (5, 7))


@pytest.mark.parametrize("value", [True, "zero", "small_output", "NATIVE", None, []])
def test_invalid_initialization(value):
    with pytest.raises(ConfigurationError, match="method_config.output_init"):
        resolve_config(request(output_init=value), (5, 7))


def test_nondivisible_width_rejected_without_adjusting_groups():
    with pytest.raises(ConfigurationError, match="middle_groups must divide hidden_width"):
        resolve_config(request(hidden_width=15), (5, 7))


@pytest.mark.parametrize(
    "overrides",
    [
        {"features": 2, "hidden_width": 2.0},
        {"features": True, "hidden_width": 1},
        {"features": 1, "hidden_width": True},
        {"output_init_std": 10**400},
    ],
)
def test_aliases_and_float_conversion_do_not_hide_invalid_values(overrides):
    with pytest.raises(ConfigurationError):
        resolve_config(request(**overrides), (5, 7))


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
@pytest.mark.parametrize("effort", [case[0] for case in CASES])
def test_device_and_size_do_not_change_presets(monkeypatch, device, effort):
    monkeypatch.setattr("jdll_denoise.config.resolve_device", lambda *a, **kw: (device, None))
    source = request(effort)
    source["device"] = device
    small = resolve_config(source, (5, 7))["method_config"]
    large = resolve_config(source, (2048, 2048))["method_config"]
    assert small == large == resolve_config(request(effort), (5, 7))["method_config"]


@pytest.mark.parametrize("effort,width,groups,steps,initialization,parameters", CASES)
@pytest.mark.parametrize("channels", [1, 3])
def test_exact_network_initialization_and_first_gradients(
    effort, width, groups, steps, initialization, parameters, channels
):
    torch.manual_seed(5489)
    native = zs_n2n.ZSN2NNetwork(channels, width, middle_groups=groups)
    torch.manual_seed(5489)
    model = zs_n2n.ZSN2NNetwork(channels, width, groups, initialization)
    assert [type(layer) for layer in model.layers] == [
        nn.Conv2d,
        nn.LeakyReLU,
        nn.Conv2d,
        nn.LeakyReLU,
        nn.Conv2d,
    ]
    assert [
        (
            model.layers[i].in_channels,
            model.layers[i].out_channels,
            model.layers[i].groups,
            model.layers[i].kernel_size,
            model.layers[i].padding,
        )
        for i in (0, 2, 4)
    ] == [
        (channels, width, 1, (3, 3), (1, 1)),
        (width, width, groups, (3, 3), (1, 1)),
        (width, channels, 1, (1, 1), (0, 0)),
    ]
    count = sum(p.numel() for p in model.parameters())
    assert count == width * (9 * channels + 1) + width * (9 * (width // groups) + 1) + channels * (
        width + 1
    )
    if channels == 1:
        assert count == parameters
    for index in (0, 2):
        torch.testing.assert_close(
            model.layers[index].weight, native.layers[index].weight, rtol=0, atol=0
        )
        torch.testing.assert_close(
            model.layers[index].bias, native.layers[index].bias, rtol=0, atol=0
        )
    for index in (1, 3):
        assert model.layers[index].negative_slope == 0.2 and model.layers[index].inplace
    if initialization == "near_zero":
        # Recreate the RNG sequence independently, including normal constructor draws.
        torch.manual_seed(5489)
        expected = zs_n2n.ZSN2NNetwork(channels, width, groups)
        nn.init.normal_(expected.layers[4].weight, mean=0.0, std=0.001)
        nn.init.zeros_(expected.layers[4].bias)
        torch.testing.assert_close(
            model.layers[4].weight, expected.layers[4].weight, rtol=0, atol=0
        )
        assert torch.count_nonzero(model.layers[4].weight) == model.layers[4].weight.numel()
        assert torch.count_nonzero(model.layers[4].bias) == 0
    else:
        torch.testing.assert_close(model.layers[4].weight, native.layers[4].weight, rtol=0, atol=0)
        torch.testing.assert_close(model.layers[4].bias, native.layers[4].bias, rtol=0, atol=0)
    value = torch.rand(2, channels, 5, 7)
    torch.testing.assert_close(model(value), model.layers(value), rtol=0, atol=0)  # No hidden skip.
    zs_n2n.canonical_losses(model, value, batch_pairs=True)[0].backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


@pytest.mark.parametrize("api", [denoise, preview])
@pytest.mark.parametrize("effort,width,groups,steps,initialization,parameters", CASES)
def test_public_api_constructs_presets_and_reports_actual_execution(
    monkeypatch, api, effort, width, groups, steps, initialization, parameters
):
    original = zs_n2n.ZSN2NNetwork.__init__
    original_step = torch.optim.Adam.step
    original_backward = torch.Tensor.backward
    models, calls, updates, backwards = [], [], [], []

    def observe(self, *args, **kwargs):
        original(self, *args, **kwargs)
        models.append(self)
        self.register_forward_hook(
            lambda m, inputs, out: calls.append((torch.is_grad_enabled(), tuple(inputs[0].shape)))
        )

    def optimizer_step(self, *args, **kwargs):
        updates.append(self.param_groups[0]["lr"])
        return original_step(self, *args, **kwargs)

    def backward(self, *args, **kwargs):
        backwards.append(True)
        return original_backward(self, *args, **kwargs)

    monkeypatch.setattr(zs_n2n.ZSN2NNetwork, "__init__", observe)
    monkeypatch.setattr(torch.optim.Adam, "step", optimizer_step)
    monkeypatch.setattr(torch.Tensor, "backward", backward)
    image = np.linspace(-20, 80, 35, dtype=np.float32).reshape(5, 7)
    config = request(effort, steps=3, report_interval=2)
    original_config, original_image = deepcopy(config), image.copy()
    threads = torch.get_num_threads(), torch.get_num_interop_threads()
    events = []
    result = api(image, config, callbacks=events.append)
    assert threads == (torch.get_num_threads(), torch.get_num_interop_threads())
    assert config == original_config
    np.testing.assert_array_equal(image, original_image)
    assert len(models) == 1 and models[0].layers[0].out_channels == width
    assert models[0].layers[2].groups == groups
    assert calls == [(True, (2, 1, 2, 3)), (True, (1, 1, 5, 7))] * 3 + [(False, (1, 1, 5, 7))]
    assert len(backwards) == len(updates) == 3
    metadata = result["metadata"]
    job = metadata["jobs"][0]
    assert (
        metadata["effort_presets_version"]
        == metadata["resolved_config"]["effort_presets_version"]
        == 1
    )
    assert job["effort_presets_version"] == 1 and job["effort"] == effort
    assert job["hidden_width"] == width and job["middle_groups"] == groups
    assert job["output_init"] == initialization and job["batch_pairs"] is True
    assert job["output_init_std"] == (0.001 if initialization == "near_zero" else None)
    assert job["network_parameter_count"] == parameters
    assert job["completed_iterations"] == job["selected_step"] == job["configured_iterations"] == 3
    assert job["optimizer"] == "Adam" and job["scheduler"] == "StepLR"
    assert job["optimizer_settings"] == {
        "learning_rate": 0.001,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0,
        "amsgrad": False,
    }
    assert job["scheduler_step_size"] == 1000 and job["scheduler_gamma"] == 0.5
    assert job["fidelity_status"].startswith(
        "reference_architecture_batched" if effort == "thorough" else "jdll_performance_variant"
    )
    progress = [event for event in events if event["phase"] == "optimization"]
    assert [event["current"] for event in progress] == [1, 2, 3]
    assert all(event["maximum"] == 3 for event in progress)
    for event in progress:
        metrics = event["metrics"]
        assert metrics["loss_total"] == pytest.approx(
            metrics["loss_residual"] + metrics["loss_consistency"]
        )
    initial = [event for event in events if f"{width} features" in event["message"]]
    assert len(initial) == 1
    assert f"middle groups {groups}" in initial[0]["message"]
    assert (
        "3 updates" in initial[0]["message"] and f"head {initialization}" in initial[0]["message"]
    )
    assert (
        next(
            event["info"]["resolved_config"]
            for event in events
            if event["type"] == "resolved_config"
        )
        == metadata["resolved_config"]
    )
    assert events[-1]["phase"] == "completed"
    assert np.isfinite(result["denoised"]).all() and result["denoised"].shape == image.shape
    normalizer = Normalizer.fit(image, metadata["resolved_config"]["normalization"])
    with torch.no_grad():
        final = zs_n2n.canonical_inference(
            models[0], torch.from_numpy(normalizer.normalize(image))[None, None]
        )
    np.testing.assert_array_equal(result["denoised"], normalizer.invert(final[0, 0].numpy()))
    json.dumps(metadata, allow_nan=False)
    json.dumps(events, allow_nan=False)


def test_advanced_execution_and_reporting_independence():
    image = np.linspace(0, 1, 35, dtype=np.float32).reshape(5, 7)
    results = []
    for interval in (1, 7):
        config = request(
            "quick",
            hidden_width=12,
            middle_groups=3,
            output_init_std=0.004,
            batch_pairs=False,
            steps=5,
            report_interval=interval,
        )
        results.append(denoise(image, config))
    np.testing.assert_array_equal(results[0]["denoised"], results[1]["denoised"])
    job = results[0]["metadata"]["jobs"][0]
    assert (
        job["hidden_width"],
        job["middle_groups"],
        job["output_init_std"],
        job["batch_pairs"],
    ) == (12, 3, 0.004, False)
    assert job["fidelity_status"] == "noncanonical_extension"
    assert job["selected_step"] == 5


def test_production_scheduler_uses_lr_before_decay(monkeypatch):
    used = []
    original = torch.optim.Adam.step

    def observe(self, *args, **kwargs):
        used.append(self.param_groups[0]["lr"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", observe)
    events = []
    denoise(
        np.arange(16, dtype=np.float32).reshape(4, 4),
        request("quick", steps=5, scheduler_step_size=2, report_interval=1),
        callbacks=events.append,
    )
    assert used == [0.001, 0.001, 0.0005, 0.0005, 0.00025]
    progress = [event for event in events if event["phase"] == "optimization"]
    assert [event["metrics"]["learning_rate"] for event in progress] == used
    assert progress[1]["metrics"]["next_learning_rate"] == 0.0005


def test_reference_unbatched_public_path_matches_independent_transcription():
    from test_zs_n2n import _reference_losses

    image = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    config = request("thorough", steps=2, batch_pairs=False)
    result = denoise(image, config)
    torch.manual_seed(config["seed"])
    reference = nn.Sequential(
        nn.Conv2d(1, 48, 3, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(48, 48, 3, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(48, 1, 1),
    )
    noisy = torch.from_numpy(image)[None, None]
    optimizer = torch.optim.Adam(reference.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        _reference_losses(reference, noisy)[0].backward()
        optimizer.step()
        scheduler.step()
    expected = torch.clamp(noisy - reference(noisy), 0, 1).detach()[0, 0].numpy()
    np.testing.assert_allclose(result["denoised"], expected, rtol=1e-6, atol=1e-7)
    assert result["metadata"]["jobs"][0]["fidelity_status"].startswith(
        "reference_architecture_unbatched"
    )


@pytest.mark.parametrize(
    "early,restore", [(False, False), (True, False), (False, True), (True, True)]
)
def test_advanced_stopping_and_best_selection_ignore_reporting_frequency(early, restore):
    image = np.arange(35, dtype=np.float32).reshape(5, 7)
    results = [
        denoise(
            image,
            request(
                "quick",
                steps=6,
                early_stopping=early,
                restore_best_weights=restore,
                patience=2,
                min_delta=1.0,
                report_interval=interval,
            ),
        )
        for interval in (1, 5)
    ]
    np.testing.assert_array_equal(results[0]["denoised"], results[1]["denoised"])
    for result in results:
        job = result["metadata"]["jobs"][0]
        assert job["completed_iterations"] == (3 if early else 6)
        assert job["selected_step"] == (1 if restore else job["completed_iterations"])


def test_minmax_defaults_explicit_override_constant_and_nonfinite_inputs():
    image = np.arange(35, dtype=np.float32).reshape(5, 7) * 10 - 25
    default = denoise(image, request(steps=1))
    norm = default["metadata"]["normalization"][0]
    assert (norm["low"], norm["high"], norm["offset"], norm["scale"]) == (0, 100, -25, 340)
    config = request(steps=1)
    config["normalization"] = {"low": 10, "high": 90, "clip": True}
    explicit = denoise(image, config)["metadata"]["normalization"][0]
    assert (explicit["low"], explicit["high"]) == (10, 90)
    assert explicit["offset"] == pytest.approx(9) and explicit["scale"] == pytest.approx(272)
    constant = np.full((5, 7), -7.5, dtype=np.float32)
    result = preview(constant, request())
    np.testing.assert_array_equal(result["denoised"], constant)
    assert result["metadata"]["effort_presets_version"] == 1
    for invalid in (float("nan"), float("inf")):
        with pytest.raises(InputFormatError, match="NaN or Inf"):
            denoise(np.full((5, 7), invalid), request(steps=1))


def test_other_methods_unchanged_since_this_spec():
    path = Path(__file__).parent / "fixtures" / "other_methods_before_zs_effort.json"
    for method, efforts in json.loads(path.read_text()).items():
        for effort, expected in efforts.items():
            assert (
                resolve_config({"method": method, "effort": effort, "axes": "yx"}, (16, 16))
                == expected
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
@pytest.mark.parametrize("effort", [case[0] for case in CASES])
def test_accelerator_public_execution(device, effort):
    config = request(effort, steps=1)
    config["device"] = device
    result = denoise(np.arange(35, dtype=np.float32).reshape(5, 7), config)
    job = result["metadata"]["jobs"][0]
    expected = PRESETS["zs_n2n"][effort]
    assert all(
        job[key] == expected[key]
        for key in ("hidden_width", "middle_groups", "batch_pairs", "output_init")
    )
    for key in ("model_device", "training_input_device", "prediction_input_device"):
        assert job[key].startswith(device)
    assert np.isfinite(result["denoised"]).all()


def test_cancellation_failure_cleanup_callback_isolation_and_cpu_fallback(monkeypatch):
    api_module = importlib.import_module("jdll_denoise.api")
    cleaned = []
    monkeypatch.setattr(api_module, "cleanup_device", cleaned.append)
    image = np.arange(35, dtype=np.float32).reshape(5, 7)
    cancelled, events = False, []

    def cancel(event):
        nonlocal cancelled
        events.append(event)
        if event["phase"] == "optimization":
            cancelled = True

    with pytest.raises(DenoisingCancelled):
        denoise(
            image,
            request(steps=4, report_interval=1),
            callbacks=cancel,
            cancellation=lambda: cancelled,
        )
    assert events[-1]["phase"] == "cancelled" and cleaned == ["cpu"]
    with monkeypatch.context() as patch:

        def fail(*args, **kwargs):
            raise RuntimeError("simulated loss failure")

        patch.setattr(zs_n2n, "canonical_losses", fail)
        with pytest.raises(RuntimeError, match="simulated loss failure"):
            denoise(image, request(steps=1), callbacks=events.append)
        assert events[-1]["phase"] == "failed" and cleaned == ["cpu", "cpu"]

    def broken(event):
        raise RuntimeError("bridge error")

    result = denoise(image, request(steps=1), callbacks=broken)
    assert any("bridge error" in warning for warning in result["metadata"]["warnings"])
    assert cleaned == ["cpu"] * 3
    original, seen = zs_n2n._run_plane_on_resolved_device, []

    def oom(image, context, plane):
        seen.append(deepcopy(context.config))
        if context.config["device"] == "cuda":
            raise OutOfMemoryDenoisingError("simulated OOM")
        return original(image, context, plane)

    monkeypatch.setattr("jdll_denoise.config.resolve_device", lambda *a, **kw: ("cuda", None))
    monkeypatch.setattr(zs_n2n, "_run_plane_on_resolved_device", oom)
    config = request("balanced_high", steps=1)
    config.update(device="cuda", allow_cpu_fallback=True)
    result = denoise(image, config)
    assert seen[0]["method_config"] == seen[1]["method_config"]
    assert result["metadata"]["device_used"] == "cpu"
    config["allow_cpu_fallback"] = False
    with pytest.raises(OutOfMemoryDenoisingError):
        denoise(image, config)


@pytest.mark.parametrize("mode", ["independent", "native_color"])
def test_stack_preview_parity_fresh_models_and_channel_axes(monkeypatch, mode):
    original, models = zs_n2n.ZSN2NNetwork.__init__, []

    def observe(self, *args, **kwargs):
        original(self, *args, **kwargs)
        models.append(self)

    monkeypatch.setattr(zs_n2n.ZSN2NNetwork, "__init__", observe)
    image = np.random.default_rng(4).random((2, 3, 3, 5, 7), dtype=np.float32)
    config = request("balanced", steps=1)
    config.update(axes="tczyx", channel_mode=mode)
    full, slab = denoise(image, config), preview(image, config)
    np.testing.assert_array_equal(slab["denoised"], full["denoised"][:, :, 1])
    assert full["metadata"]["resolved_config"] == slab["metadata"]["resolved_config"]
    assert (
        len(models)
        == len({id(model) for model in models})
        == (12 if mode == "native_color" else 36)
    )
    assert all(model.layers[2].groups == 2 for model in models)
    assert slab["metadata"]["output_axes"] == "tcyx"
    assert full["metadata"]["dimension_mode"] == "slice_wise_2d"
    assert slab["metadata"]["preview_region"] == {"z_start": 0, "z_end": 3, "center_z": 1}


def test_tiled_final_inference_keeps_single_full_plane_training_model(monkeypatch):
    original, models, calls = zs_n2n.ZSN2NNetwork.__init__, [], []

    def observe(self, *args, **kwargs):
        original(self, *args, **kwargs)
        models.append(self)
        self.register_forward_hook(
            lambda m, inputs, out: calls.append((torch.is_grad_enabled(), tuple(inputs[0].shape)))
        )

    monkeypatch.setattr(zs_n2n.ZSN2NNetwork, "__init__", observe)
    config = request("quick", steps=2)
    config["tiling"] = {"enabled": True, "tile_size": 32, "overlap": 4}
    image = np.random.default_rng(34).random((35, 39), dtype=np.float32)
    result = denoise(image, config)
    assert len(models) == 1
    assert calls[:4] == [(True, (2, 1, 17, 19)), (True, (1, 1, 35, 39))] * 2
    assert len(calls[4:]) > 1
    assert all(not grad and shape[-2:] == (32, 32) for grad, shape in calls[4:])
    assert result["denoised"].shape == image.shape and np.isfinite(result["denoised"]).all()
    assert result["metadata"]["jobs"][0]["inference_tiled"] is True
