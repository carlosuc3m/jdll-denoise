"""Quality-only ablation of grouped ZS-N2N residuals and output initialization.

Use the production denoise API with process-local network substitution. The
canonical loss and optimizer stay unchanged. Reference images only score the
outputs; they never influence training or checkpoint selection.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import tifffile
import torch
from torch import nn

from zs_n2n_middle_layers import MiddleVariant, expected_parameters
from zs_n2n_width import denoise, mse, psnr, request, resolve_config, save_csv, save_json, zs_n2n
from jdll_denoise.normalization import Normalizer


VARIANTS = ("baseline", "residual", "small_output", "residual_small_output")


class InitializationVariant(MiddleVariant):
    def __init__(self, channels=1, hidden_width=16, *, variant, groups=2):
        super().__init__(channels, hidden_width, variant={1: "original", 2: "groups2", 4: "groups4"}[groups])
        self.residual = variant in ("residual", "residual_small_output")
        if variant in ("small_output", "residual_small_output"):
            nn.init.normal_(self.layers[4].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.layers[4].bias)

    def forward(self, value):
        if not self.residual:
            return self.layers(value)
        hidden = self.layers[1](self.layers[0](value))
        hidden = self.layers[3](hidden + self.layers[2](hidden))
        return self.layers[4](hidden)

    def fused(self):
        model = deepcopy(self)
        if model.residual:
            middle = model.layers[2]
            indices = torch.arange(middle.out_channels, device=middle.weight.device)
            with torch.no_grad():
                middle.weight[indices, indices % (middle.in_channels // middle.groups), 1, 1] += 1
            model.residual = False
        return model


def check_variants(width, groups=2):
    checks = []
    for shape in ((1, 1, 32, 34), (2, 3, 31, 33)):
        torch.manual_seed(123)
        value = torch.rand(shape, dtype=torch.float64)
        models = {}
        for name in VARIANTS:
            torch.manual_seed(5489)
            model = InitializationVariant(shape[1], width, variant=name, groups=groups).double()
            models[name] = model
            prediction = model(value)
            assert prediction.shape == value.shape and torch.isfinite(prediction).all()
            assert sum(p.numel() for p in model.parameters()) == expected_parameters(
                {1: "original", 2: "groups2", 4: "groups4"}[groups], shape[1], width)
            assert sum(isinstance(m, nn.LeakyReLU) for m in model.modules()) == 2
            loss = zs_n2n.canonical_losses(model, value)[0]
            loss.backward()
            assert torch.isfinite(loss)
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       and torch.count_nonzero(p.grad) > 0 for p in model.parameters())
            with torch.no_grad():
                fused_prediction = model.fused()(value)
            torch.testing.assert_close(prediction, fused_prediction, rtol=1e-12, atol=1e-12)
            checks.append({"variant": name, "shape": list(shape),
                           "initial_noise_rms": float(prediction.detach().square().mean().sqrt()),
                           "nonzero_finite_gradients": True,
                           "fusion_max_abs_difference": float((prediction - fused_prediction).detach().abs().max())})
        for name, model in models.items():
            # The ablation changes only the intended skip connection/head.
            for layer in (0, 2):
                assert torch.equal(model.layers[layer].weight, models["baseline"].layers[layer].weight)
                assert torch.equal(model.layers[layer].bias, models["baseline"].layers[layer].bias)
            if "small_output" in name:
                assert torch.count_nonzero(model.layers[4].bias) == 0
            else:
                assert torch.equal(model.layers[4].weight, models["baseline"].layers[4].weight)
    return checks


class Milestones:
    def __init__(self, normalizer, shape, milestones):
        self.normalizer = normalizer
        self.shape = (1, 1, *shape)
        self.milestones = set(milestones)
        self.full_calls = 0
        self.images = {}

    def __call__(self, model, inputs, predicted_noise):
        if tuple(inputs[0].shape) != self.shape:
            return
        # The next training forward sees weights after the preceding update.
        # Full calls 0..999 observe training; final inference supplies step 1000.
        completed = self.full_calls
        self.full_calls += 1
        if completed not in self.milestones:
            return
        with torch.no_grad():
            prediction = torch.clamp(inputs[0] - predicted_noise, 0, 1)[0, 0].cpu().numpy()
        self.images[completed] = self.normalizer.invert(prediction)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--milestones", nargs="+", type=int, default=[500, 700, 1000])
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--percentiles", nargs=2, type=float, default=[1, 99], metavar=("LOW", "HIGH"))
    parser.add_argument("--seed", type=int, default=5489)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    milestones = sorted(set(args.milestones))
    if min(milestones) < 1 or max(milestones) > 1000:
        parser.error("milestones must be between 1 and 1000; LR remains constant over this range")
    if args.width < 2 or args.width % 2:
        parser.error("width must be a positive multiple of two")
    if not 0 <= args.percentiles[0] < args.percentiles[1] <= 100:
        parser.error("percentiles must satisfy 0 <= LOW < HIGH <= 100")
    reference = tifffile.imread(args.input_dir / "reference.tif")
    noisy = tifffile.imread(args.input_dir / "noisy.tif")
    assert reference.ndim == 2 and reference.shape == noisy.shape
    assert np.isfinite(reference).all() and np.isfinite(noisy).all()
    assert 0 <= reference.min() <= reference.max() <= 1
    args.output.mkdir(parents=True, exist_ok=False)
    threads = {"intra_op": torch.get_num_threads(), "inter_op": torch.get_num_interop_threads()}
    print(f"Default CPU threads: {threads}; checking variants and residual fusion", flush=True)
    save_json(args.output / "architecture_checks.json", check_variants(args.width))
    config = request(args.width, milestones[-1], args.seed, "cpu")
    config["normalization"].update(low=args.percentiles[0], high=args.percentiles[1])
    normalizer = Normalizer.fit(noisy, config["normalization"])
    original_network, original_losses = zs_n2n.ZSN2NNetwork, zs_n2n.canonical_losses
    backend_hash = hashlib.sha256(Path(zs_n2n.__file__).read_bytes()).hexdigest()
    save_json(args.output / "config.json", {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "actual_threads": threads, "python": sys.version, "torch": torch.__version__,
        "shape_yx": list(noisy.shape), "resolved_config": resolve_config(config, noisy.shape),
        "normalization": normalizer.metadata(), "optimizer": "Adam", "weight_decay": 0,
        "learning_rate_used": 0.001, "groups": 2, "width": args.width, "pair_batching": False,
        "input_sha256": {name: hashlib.sha256((args.input_dir / name).read_bytes()).hexdigest()
                         for name in ("reference.tif", "noisy.tif")},
        "backend_sha256": backend_hash,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "network_script_sha256": hashlib.sha256(Path(sys.modules[MiddleVariant.__module__].__file__).read_bytes()).hexdigest(),
        "residual_note": "Only add the identity skip; conv2 keeps its native initialization, not a separately reduced branch initialization.",
        "output_init_note": "Only small_output variants use output weights Normal(0, 1e-3) and zero output bias.",
        "reference_note": "Original acquisition is not verified noise-free. Reference is only used for scoring, never training.",
        "milestone_note": "One trajectory per variant; snapshots reuse full-image forwards. LR is constant through update 1000. No timing, LR or beta trace is added.",
    })
    tifffile.imwrite(args.output / "reference.tif", reference)
    tifffile.imwrite(args.output / "noisy.tif", noisy)
    rows = []
    initial_rows = []
    for name in args.variants:
        print(f"Training {name}, {milestones[-1]} updates", flush=True)
        observer = Milestones(normalizer, noisy.shape, [0, *milestones])
        holder = {}
        training_progress = []

        def factory(*positional, **keyword):
            model = InitializationVariant(*positional, **keyword, variant=name)
            holder["model"] = model
            holder["hook"] = model.register_forward_hook(observer)
            return model

        def progress(event):
            if event["type"] != "denoise_progress" or event["phase"] != "optimization":
                return
            metrics = event["metrics"]
            training_progress.append({"step": event["current"],
                                      **{k: v for k, v in metrics.items() if k.startswith("loss")}})
            if event["current"] % 250 == 0 or event["current"] == milestones[-1]:
                print(f"{name}: {event['current']}/{milestones[-1]} updates", flush=True)

        with patch.object(zs_n2n, "ZSN2NNetwork", factory):
            result = denoise(noisy, config, callbacks=progress)
        holder["hook"].remove()
        assert zs_n2n.ZSN2NNetwork is original_network and zs_n2n.canonical_losses is original_losses
        job = result["metadata"]["jobs"][0]
        assert job["optimizer"] == "Adam"
        assert job["network_parameter_count"] == expected_parameters("groups2", 1, args.width)
        assert job["completed_iterations"] == job["selected_step"] == milestones[-1]
        assert job["hidden_width"] == args.width and result["metadata"]["device_used"] == "cpu"
        assert observer.full_calls == milestones[-1] + 1
        assert set(observer.images) == {0, *milestones}
        assert not any("callback " in warning for warning in result["metadata"]["warnings"])
        assert threads == {"intra_op": torch.get_num_threads(), "inter_op": torch.get_num_interop_threads()}
        np.testing.assert_array_equal(observer.images[milestones[-1]], result["denoised"])
        initial_rows.append({"variant": name, "reference_mse": mse(observer.images[0], reference)})
        for step in milestones:
            image = observer.images[step]
            assert image.shape == reference.shape and image.dtype == np.float32 and np.isfinite(image).all()
            error = mse(image, reference)
            row = {"variant": name, "width": args.width, "updates": step,
                   "reference_mse": error, "psnr_db": psnr(error),
                   "error_reduction_percent": 100 * (1 - error / mse(noisy, reference))}
            rows.append(row)
            path = args.output / f"{name}_{step}.tif"
            tifffile.imwrite(path, image)
            assert mse(tifffile.imread(path), reference) == error
            print(f"RESULT {name} step={step}: MSE={error:.9f}, PSNR={row['psnr_db']:.3f}dB", flush=True)
        fusion = {"applied": holder["model"].residual}
        if holder["model"].residual:
            tensor = torch.from_numpy(normalizer.normalize(noisy))[None, None]
            with torch.no_grad():
                original = holder["model"](tensor)
                fused = holder["model"].fused()(tensor)
            torch.testing.assert_close(original, fused, rtol=2e-5, atol=2e-6)
            fusion["trained_float32_max_noise_difference"] = float((original - fused).abs().max())
        result["metadata"]["benchmark_architecture"] = {"variant": name, "fusion_check": fusion}
        save_json(args.output / f"{name}_metadata.json", result["metadata"])
        save_csv(args.output / f"{name}_training_progress.csv", training_progress)
        save_json(args.output / "results.json", rows)
        save_csv(args.output / "results.csv", rows)
        save_json(args.output / "initial_predictions.json", initial_rows)
        del holder, result, observer
    assert hashlib.sha256(Path(zs_n2n.__file__).read_bytes()).hexdigest() == backend_hash
    save_json(args.output / "summary.json", {
        "noisy_mse": mse(noisy, reference), "noisy_psnr_db": psnr(mse(noisy, reference)),
        "results": rows,
        "best_at_each_milestone": {str(step): min((r for r in rows if r["updates"] == step),
                                                 key=lambda r: r["reference_mse"])
                                   for step in milestones},
    })


if __name__ == "__main__":
    main()
