"""Production effort presets and regression of the existing Noise2Fast loop."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch
from PIL import Image

from jdll_denoise import ConfigurationError, DenoisingCancelled, denoise, preview, resolve_config
from jdll_denoise.methods import noise2fast
from jdll_denoise.presets import PRESETS

PRESET_CASES = [("quick", 16, 50), ("balanced", 24, 100), ("thorough", 64, 100)]
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "options,expected",
    [
        (
            ["--efforts", "quick", "balanced", "thorough"],
            [("quick", 16, 50), ("balanced", 24, 100), ("thorough", 64, 100)],
        ),
        (["--widths", "32", "--patience", "75"], [("width_32", 32, 75)]),
    ],
)
def test_benchmark_cli_uses_public_api(tmp_path, options, expected):
    source = tmp_path / "input.png"
    Image.fromarray(np.arange(35, dtype=np.uint8).reshape(5, 7) * 7).save(source)
    output = tmp_path / "benchmark"
    script = Path(__file__).resolve().parents[1] / "benchmarks" / "noise2fast_width.py"
    process = subprocess.run(
        [
            sys.executable,
            str(script),
            "--image",
            str(source),
            "--device",
            "cpu",
            "--threads",
            "1",
            "--max-steps",
            "2",
            "--warmup-steps",
            "1",
            "--output",
            str(output),
            *options,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    configuration = json.loads((output / "config.json").read_text())
    assert configuration["api"] == "jdll_denoise.denoise"
    records = json.loads((output / "results.json").read_text())
    assert len(records) == len(expected)
    for record, (label, width, patience) in zip(records, expected, strict=True):
        assert (record["run"], record["width"], record["patience"]) == (label, width, patience)
        resolved = record["resolved_config"]["method_config"]
        assert (resolved["hidden_width"], resolved["patience"]) == (width, patience)
        assert resolved["steps"] == 2
        assert record["steps"] == 2
        assert record["stopped_by"] == "configured_step_limit"
        assert record["backend_metadata"]["output_average_window"] == 100
        assert record["backend_metadata"]["output_average_count"] == 2
        assert len((output / f"steps_{label}.csv").read_text().splitlines()) == 3
        pixels = tifffile.imread(output / f"denoised_{label}.tif")
        assert pixels.shape == (5, 7)
        assert np.isfinite(pixels).all()
    assert (output / "REPORT.md").is_file()
    assert (output / "comparison.png").is_file()


@pytest.fixture(autouse=True)
def single_cpu_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def request(effort="balanced", **overrides):
    return {
        "method": "noise2fast",
        "effort": effort,
        "axes": "yx",
        "device": "cpu",
        "seed": 5489,
        "method_config": overrides,
    }


@pytest.mark.parametrize("effort,width,patience", PRESET_CASES)
def test_preset_resolution_and_independent_copies(effort, width, patience):
    table_before = deepcopy(PRESETS)
    supplied = request(effort)
    before = deepcopy(supplied)
    resolved = resolve_config(supplied, (5, 7))
    cfg = resolved["method_config"]
    assert cfg == {
        "hidden_width": width,
        "patience": patience,
        "output_window": 100,
        "steps": 0,
        "report_interval": 10,
    }
    cfg["hidden_width"] = 99
    assert resolve_config(supplied, (5, 7))["method_config"]["hidden_width"] == width
    assert supplied == before
    assert PRESETS == table_before


def test_omitted_effort_is_balanced():
    resolved = resolve_config({"axes": "yx"}, (5, 7))
    assert resolved["effort"] == "balanced"
    assert resolved["method_config"]["hidden_width"] == 24
    assert resolved["method_config"]["patience"] == 100


@pytest.mark.parametrize("effort,_,__", PRESET_CASES)
def test_advanced_overrides_take_precedence(effort, _, __):
    config = request(effort, hidden_width=32, patience=75, steps=9, report_interval=3)
    before = deepcopy(config)
    cfg = resolve_config(config, (5, 7))["method_config"]
    assert cfg == {
        "hidden_width": 32,
        "patience": 75,
        "output_window": 100,
        "steps": 9,
        "report_interval": 3,
    }
    assert config == before
    assert (
        resolve_config(request(effort, hidden_width=7), (5, 7))["method_config"]["hidden_width"]
        == 7
    )


@pytest.mark.parametrize("key", ["hidden_width", "patience", "report_interval"])
@pytest.mark.parametrize("value", [True, False, 1.0, 1.5, "16", "auto", 0, -1, None, []])
def test_invalid_positive_integer_options_fail(key, value):
    with pytest.raises(ConfigurationError, match=rf"method_config\.{key}.*positive integer"):
        resolve_config(request(**{key: value}), (5, 7))


@pytest.mark.parametrize("value", [True, False, 1.0, "0", -1, None])
def test_step_cap_validation(value):
    with pytest.raises(ConfigurationError, match="method_config.steps.*non-negative integer"):
        resolve_config(request(steps=value), (5, 7))


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
@pytest.mark.parametrize("effort,width,patience", PRESET_CASES)
def test_resolution_does_not_tune_width_or_patience_for_device(
    monkeypatch, device, effort, width, patience
):
    monkeypatch.setattr("jdll_denoise.config.resolve_device", lambda *a, **kw: (device, None))
    supplied = request(effort)
    supplied["device"] = device
    cfg = resolve_config(supplied, (5, 7))["method_config"]
    assert (cfg["hidden_width"], cfg["patience"]) == (width, patience)


def test_full_width_matches_frozen_previous_cpu_loop():
    baseline = json.loads((FIXTURES / "noise2fast_64_before_effort.json").read_text())
    image = np.asarray(baseline["input"], dtype=np.float32)
    config = baseline["config"]
    config["method_config"].update(hidden_width=64, patience=100)
    result = denoise(image, config)
    np.testing.assert_allclose(result["denoised"], baseline["output"], rtol=1e-6, atol=1e-7)
    job = result["metadata"]["jobs"][0]
    for key in ("best_step", "completed_steps", "output_average_count"):
        assert job[key] == baseline[key]
    assert job["best_validation_negative_mse"] == pytest.approx(
        baseline["best_validation_negative_mse"], rel=1e-6, abs=1e-7
    )


@pytest.mark.parametrize("api", [denoise, preview])
@pytest.mark.parametrize("effort,width,patience", PRESET_CASES + [("quick", 32, 75)])
def test_public_api_constructs_actual_resolved_width(monkeypatch, api, effort, width, patience):
    # Observe real construction and forwards; do not replace the network.
    original = noise2fast.Noise2FastNet.__init__
    models = []
    forwards = []

    def observed_init(self, hidden_width=64):
        original(self, hidden_width)
        models.append(self)
        self.register_forward_hook(
            lambda model, inputs, output: forwards.append(
                (torch.is_grad_enabled(), tuple(inputs[0].shape))
            )
        )

    monkeypatch.setattr(noise2fast.Noise2FastNet, "__init__", observed_init)
    image = np.random.default_rng(19).random((5, 7), dtype=np.float32)
    cfg = request(effort, steps=3, report_interval=2)
    if width == 32:
        cfg["method_config"].update(hidden_width=width, patience=patience)
    events = []
    result = api(image, cfg, callbacks=events.append)
    assert len(models) == 1
    assert models[0].blocks[0].conv1.out_channels == width
    assert len(forwards) == 6
    assert forwards[::2] == [(True, forwards[index][1]) for index in range(0, 6, 2)]
    assert forwards[1::2] == [(False, (1, 1, 5, 7))] * 3
    assert np.isfinite(result["denoised"]).all()
    assert result["denoised"].shape == image.shape
    job = result["metadata"]["jobs"][0]
    assert job["hidden_width"] == width
    assert job["architecture"] == f"8x3x3_conv_{width}_relu+1x1_sigmoid"
    assert job["validation_patience"] == patience
    assert job["output_average_window"] == 100
    assert job["output_average_count"] == 3
    assert ("released" if width == 64 else "variant") in job["algorithm_fidelity"]
    resolved = result["metadata"]["resolved_config"]
    emitted = next(event for event in events if event["type"] == "resolved_config")
    assert emitted["info"]["resolved_config"] == resolved
    initialization = [event for event in events if f"{width} features" in event["message"]]
    assert len(initialization) == 1
    assert f"patience {patience} steps" in initialization[0]["message"]
    assert "averaging 100 predictions" in initialization[0]["message"]
    assert "device cpu" in initialization[0]["message"]
    assert [event["current"] for event in events if event["phase"] == "optimization"] == [1, 2, 3]
    assert events[-1]["phase"] == "completed"
    json.dumps(events, allow_nan=False)
    json.dumps(result["metadata"], allow_nan=False)


def controlled_validation(monkeypatch, scores):
    """Drive only network predictions/scores; exercise the production training loop."""
    state = {"training_calls": 0, "validation_calls": 0, "updates": 0}
    original_step = torch.optim.Adam.step

    def forward(self, value):
        if torch.is_grad_enabled():
            state["training_calls"] += 1
            return torch.sigmoid(self.output.bias).reshape(1, 1, 1, 1).expand_as(value)
        state["validation_calls"] += 1
        assert state["validation_calls"] == state["updates"]
        return torch.full_like(value, state["validation_calls"] / 1000)

    def mse(prediction, target):
        score = scores(state["validation_calls"])
        return torch.tensor(-score, device=prediction.device, dtype=prediction.dtype)

    def step(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs)
        state["updates"] += 1
        assert self.defaults["lr"] == 0.001
        assert self.defaults["betas"] == (0.9, 0.999)
        return result

    monkeypatch.setattr(noise2fast.Noise2FastNet, "forward", forward)
    monkeypatch.setattr(noise2fast.F, "mse_loss", mse)
    monkeypatch.setattr(torch.optim.Adam, "step", step)
    return state


@pytest.mark.parametrize("patience,last_improvement", [(50, 1), (100, 1), (50, 339), (100, 339)])
def test_exact_patience_boundary_and_averaging(monkeypatch, patience, last_improvement):
    state = controlled_validation(monkeypatch, lambda step: float(min(step, last_improvement)))
    image = np.linspace(0, 1, 35, dtype=np.float32).reshape(5, 7)
    events = []
    result = denoise(
        image, request("quick", patience=patience, report_interval=73), callbacks=events.append
    )
    expected_steps = last_improvement + patience
    expected_count = min(expected_steps, 100)
    expected_average = np.mean(
        np.arange(expected_steps - expected_count + 1, expected_steps + 1) / 1000
    )
    np.testing.assert_allclose(result["denoised"], expected_average, atol=1e-7)
    assert set(state.values()) == {expected_steps}
    job = result["metadata"]["jobs"][0]
    assert job["completed_steps"] == expected_steps
    assert job["best_step"] == last_improvement
    assert job["epochs_without_validation_improvement"] == patience
    assert job["output_average_count"] == expected_count
    assert job["output_average_window"] == 100
    assert job["stopped_by"] == "validation_patience"
    progress = [event for event in events if event["phase"] == "optimization"]
    assert all(event["maximum"] == 0 for event in progress)
    assert all(event["metrics"]["validation_patience"] == patience for event in progress)
    assert progress[-1]["current"] == expected_steps  # Even between report intervals.
    assert f"{patience}/{patience} steps" in progress[-1]["message"]
    assert events[-1]["phase"] == "completed"
    json.dumps(events, allow_nan=False)
    json.dumps(result["metadata"], allow_nan=False)


def test_later_strict_improvement_resets_counter(monkeypatch):
    controlled_validation(monkeypatch, lambda step: 1.0 if step <= 40 else 2.0)
    events = []
    image = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    result = preview(image, request("quick", report_interval=1), callbacks=events.append)
    progress = [event for event in events if event["phase"] == "optimization"]
    assert progress[39]["metrics"]["epochs_without_validation_improvement"] == 39
    assert progress[40]["metrics"]["epochs_without_validation_improvement"] == 0
    assert progress[40]["metrics"]["best_step"] == 41
    assert result["metadata"]["jobs"][0]["completed_steps"] == 91


@pytest.mark.parametrize(
    "patience,cap,expected,reason",
    [
        (50, 7, 7, "configured_step_limit"),
        (2, 9, 3, "validation_patience"),
        (2, 3, 3, "validation_patience"),
    ],
)
def test_explicit_cap_reports_termination_and_still_allows_patience(
    monkeypatch, patience, cap, expected, reason
):
    controlled_validation(monkeypatch, lambda step: 1.0)
    events = []
    result = denoise(
        np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4),
        request(steps=cap, patience=patience, report_interval=8),
        callbacks=events.append,
    )
    job = result["metadata"]["jobs"][0]
    assert job["stopped_by"] == reason
    assert job["completed_steps"] == expected
    progress = [event for event in events if event["phase"] == "optimization"]
    assert progress[-1]["current"] == expected
    assert all(event["maximum"] == cap for event in progress)
    if reason == "configured_step_limit":
        assert f"patience ({patience} steps" in job["warnings"][0]
        assert "100-epoch" not in job["warnings"][0]


def test_reporting_frequency_leaves_training_and_outputs_unchanged():
    image = np.random.default_rng(15).random((7, 9), dtype=np.float32)
    results = [
        denoise(image, request("quick", steps=6, report_interval=interval)) for interval in (1, 5)
    ]
    np.testing.assert_array_equal(results[0]["denoised"], results[1]["denoised"])
    assert (
        results[0]["metadata"]["jobs"][0]["best_step"]
        == results[1]["metadata"]["jobs"][0]["best_step"]
    )


def test_explicit_normalization_override_is_honored():
    image = np.arange(35, dtype=np.float32).reshape(5, 7)
    default = denoise(image, request(steps=1))["metadata"]["normalization"][0]
    cfg = request(steps=1)
    cfg["normalization"] = {"low": 10, "high": 90, "clip": True}
    explicit = denoise(image, cfg)["metadata"]["normalization"][0]
    assert (default["low"], default["high"], default["offset"], default["scale"]) == (0, 100, 0, 34)
    assert (explicit["low"], explicit["high"]) == (10, 90)
    assert explicit["offset"] == pytest.approx(3.4)
    assert explicit["scale"] == pytest.approx(27.2)


def test_other_methods_keep_previous_resolved_presets():
    baseline = json.loads((FIXTURES / "other_method_presets_before_effort.json").read_text())
    for method, efforts in baseline.items():
        if method == "zs_n2n":
            # Superseded by the separately agreed ZS-N2N effort contract.
            # Its old architecture/optimization path remains independently tested.
            continue
        for effort, previous in efforts.items():
            resolved = resolve_config({"method": method, "effort": effort, "axes": "yx"}, (16, 16))
            assert resolved["method_config"] == previous
            assert resolved["normalization"]["low"] == 0.1
            assert resolved["normalization"]["high"] == 99.9


def test_fallback_retains_width_and_patience(monkeypatch):
    monkeypatch.setattr("jdll_denoise.config.resolve_device", lambda *a, **kw: ("cuda", None))
    original = noise2fast._run_plane_on_device
    seen = []

    def simulate_cuda_oom(image, context, plane_index, device):
        seen.append((device, deepcopy(context.config["method_config"])))
        if device == "cuda":
            raise noise2fast._DeviceOOM("simulated accelerator OOM")
        return original(image, context, plane_index, device)

    monkeypatch.setattr(noise2fast, "_run_plane_on_device", simulate_cuda_oom)
    cfg = request("quick", hidden_width=32, patience=75, steps=1)
    cfg.update(device="cuda", allow_cpu_fallback=True)
    result = denoise(np.arange(35, dtype=np.float32).reshape(5, 7), cfg)
    assert seen[0][0] == "cuda" and seen[1][0] == "cpu"
    assert seen[0][1] == seen[1][1]
    job = result["metadata"]["jobs"][0]
    assert (job["hidden_width"], job["validation_patience"], job["device_used"]) == (32, 75, "cpu")


def test_noise2fast_callback_errors_and_cancellation_remain_isolated():
    def broken(event):
        raise RuntimeError("bridge unavailable")

    image = np.arange(35, dtype=np.float32).reshape(5, 7)
    result = denoise(image, request(steps=1), callbacks=broken)
    assert any("bridge unavailable" in warning for warning in result["metadata"]["warnings"])
    cancelled = False
    events = []

    def cancel_after_update(event):
        nonlocal cancelled
        events.append(event)
        if event["phase"] == "optimization":
            cancelled = True

    with pytest.raises(DenoisingCancelled, match="optimization"):
        denoise(
            image,
            request(steps=3, report_interval=1),
            callbacks=cancel_after_update,
            cancellation=lambda: cancelled,
        )
    assert events[-1]["phase"] == "cancelled"
    assert not any(event["phase"] == "completed" for event in events)


def test_preview_and_denoise_preserve_stack_channels_and_fresh_models(monkeypatch):
    original = noise2fast.Noise2FastNet.__init__
    models = []

    def observe(self, hidden_width=64):
        original(self, hidden_width)
        models.append(self)

    monkeypatch.setattr(noise2fast.Noise2FastNet, "__init__", observe)
    source = np.random.default_rng(4).random((2, 3, 5, 7), dtype=np.float32)
    cfg = request("quick", steps=1)
    cfg["axes"] = "czyx"
    full = denoise(source, cfg)
    slab = preview(source, cfg)
    assert len(models) == len({id(model) for model in models}) == 12
    np.testing.assert_array_equal(slab["denoised"], full["denoised"][:, 1])
    assert full["metadata"]["dimension_mode"] == "slice_wise_2d"
    assert slab["metadata"]["preview_region"] == {"z_start": 0, "z_end": 3, "center_z": 1}
    for job in full["metadata"]["jobs"]:
        assert len(job["planes"]) == 3
        assert all(
            plane["hidden_width"] == 16 and plane["validation_patience"] == 50
            for plane in job["planes"]
        )
