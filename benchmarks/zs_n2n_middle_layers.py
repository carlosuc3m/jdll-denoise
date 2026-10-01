"""Benchmark lighter middle convolutions in ZS-N2N, without production edits.

The stem, output layer, two LeakyReLUs and original three-forward loss are
unchanged. Depthwise/pointwise and grouped middles are architectural variants,
not equivalent rewrites of the original model.
"""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import tifffile
import torch
from torch import nn

from zs_n2n_width import Progress, denoise, mse, psnr, request, resolve_config, save_csv, save_json, zs_n2n


VARIANTS = ("original", "depthwise", "groups2", "groups4")


class MiddleVariant(nn.Module):
    def __init__(self, channels=1, hidden_width=16, *, variant):
        super().__init__()
        w = hidden_width
        layers = [nn.Conv2d(channels, w, 3, padding=1), nn.LeakyReLU(0.2, inplace=True)]
        if variant == "depthwise":
            # No extra activation between the depthwise and pointwise layers.
            layers.extend([nn.Conv2d(w, w, 3, padding=1, groups=w), nn.Conv2d(w, w, 1)])
        else:
            groups = {"original": 1, "groups2": 2, "groups4": 4}[variant]
            layers.append(nn.Conv2d(w, w, 3, padding=1, groups=groups))
        layers.extend([nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(w, channels, 1)])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def expected_parameters(variant, channels, width):
    stem_and_head = width * (9 * channels + 1) + channels * (width + 1)
    if variant == "depthwise":
        return stem_and_head + width * 10 + width * (width + 1)
    groups = {"original": 1, "groups2": 2, "groups4": 4}[variant]
    return stem_and_head + width * (9 * (width // groups) + 1)


def check_architectures(factories, width):
    checks = []
    descriptions = {}
    for variant, factory in factories.items():
        for shape in ((1, 1, 32, 34), (2, 3, 31, 33)):
            torch.manual_seed(5489)
            model = factory(shape[1], width)
            x = torch.rand(shape, requires_grad=True)
            out = model(x)
            assert out.shape == x.shape and torch.isfinite(out).all()
            assert sum(isinstance(m, nn.LeakyReLU) for m in model.modules()) == 2
            parameters = sum(p.numel() for p in model.parameters())
            assert parameters == expected_parameters(variant, shape[1], width)
            losses = zs_n2n.canonical_losses(model, x)
            assert all(torch.isfinite(loss) for loss in losses)
            losses[0].backward()
            assert torch.isfinite(x.grad).all()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            checks.append({"variant": variant, "shape": list(shape), "parameters": parameters,
                           "total_loss": float(losses[0].detach()), "finite_gradients": True})
            if shape[1] == 1:
                descriptions[variant] = str(model)
    return checks, descriptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=5489)
    parser.add_argument("--warmup-steps", type=int, default=20)
    args = parser.parse_args()
    if min(args.width, args.steps, args.repeats, args.warmup_steps) < 1 or args.width % 4:
        parser.error("positive values required; width must be divisible by four")
    reference = tifffile.imread(args.input_dir / "reference.tif")
    noisy = tifffile.imread(args.input_dir / "noisy.tif")
    if reference.ndim != 2 or reference.shape != noisy.shape:
        raise ValueError("Expected matching grayscale images")
    if not np.isfinite(reference).all() or not np.isfinite(noisy).all():
        raise ValueError("Inputs must be finite")
    if reference.min() < 0 or reference.max() > 1:
        raise ValueError("PSNR expects reference intensities in [0,1]")
    args.output.mkdir(parents=True, exist_ok=False)
    original_network = zs_n2n.ZSN2NNetwork
    original_losses = zs_n2n.canonical_losses
    factories = {name: original_network if name == "original" else partial(MiddleVariant, variant=name)
                 for name in VARIANTS}
    threads = {"intra_op": torch.get_num_threads(), "inter_op": torch.get_num_interop_threads()}
    print(f"Default CPU threads: {threads}; checking architectures and gradients", flush=True)
    checks, descriptions = check_architectures(factories, args.width)
    save_json(args.output / "architecture_checks.json", checks)
    config = request(args.width, args.steps, args.seed, "cpu")
    configuration = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "actual_threads": threads, "python": sys.version, "torch": torch.__version__,
        "numpy": np.__version__, "shape_yx": list(noisy.shape),
        "input_sha256": {name: hashlib.sha256((args.input_dir / name).read_bytes()).hexdigest()
                         for name in ("reference.tif", "noisy.tif")},
        "backend_sha256": hashlib.sha256(Path(zs_n2n.__file__).read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "resolved_config": resolve_config(config, noisy.shape), "architectures": descriptions,
        "pair_batching": False,
        "run_order": [variant for repeat in range(args.repeats)
                      for variant in (VARIANTS if repeat % 2 == 0 else VARIANTS[::-1])],
        "reference_note": "Original acquisition, not verified noise-free; measures removal of added synthetic noise.",
        "normalization_note": "Noisy-image min-max as in previous benchmarks, not GUI percentile defaults.",
        "initialization_note": "Same seed and native PyTorch initializers; different architectures consume different random draws.",
        "timing_note": "Warmed public API calls including setup, training, prediction, callbacks and cleanup; excluding checks, warmup, scoring and file writes. CPU clocks and affinity unchanged.",
    }
    save_json(args.output / "config.json", configuration)
    tifffile.imwrite(args.output / "reference.tif", reference)
    tifffile.imwrite(args.output / "noisy.tif", noisy)
    records = []
    try:
        for index, variant in enumerate(configuration["run_order"]):
            repeat = index // len(VARIANTS) + 1
            print(f"Round {repeat}/{args.repeats}, {variant}; warmup", flush=True)
            zs_n2n.ZSN2NNetwork = factories[variant]
            denoise(noisy, request(args.width, args.warmup_steps, args.seed, "cpu"))
            progress = Progress(args.width, args.steps)
            started = time.perf_counter()
            result = denoise(noisy, config, callbacks=progress)
            elapsed = time.perf_counter() - started
            metadata = result["metadata"]
            job = metadata["jobs"][0]
            output = result["denoised"]
            assert metadata["device_used"] == "cpu"
            assert job["completed_iterations"] == job["selected_step"] == args.steps
            assert job["network_parameter_count"] == expected_parameters(variant, 1, args.width)
            assert progress.records[-1]["step"] == args.steps
            assert not any("callback " in warning for warning in metadata["warnings"])
            assert output.shape == reference.shape and np.isfinite(output).all()
            assert threads == {"intra_op": torch.get_num_threads(), "inter_op": torch.get_num_interop_threads()}
            assert zs_n2n.canonical_losses is original_losses
            metadata["benchmark_architecture"] = {"variant": variant, "layers": descriptions[variant]}
            error = mse(output, reference)
            row = {
                "variant": variant, "repeat": repeat, "run_order": index + 1,
                "width": args.width, "updates": args.steps, "parameters": job["network_parameter_count"],
                "total_seconds": elapsed,
                "optimization_seconds": job["phase_timings_seconds"]["optimization"],
                "mean_update_ms": job["phase_timings_seconds"]["optimization"] * 1000 / args.steps,
                "reference_mse": error, "psnr_db": psnr(error),
            }
            stem = f"round_{repeat}_{variant}"
            tifffile.imwrite(args.output / f"{stem}.tif", output)
            save_json(args.output / f"{stem}_metadata.json", metadata)
            save_csv(args.output / f"{stem}_progress.csv", progress.records)
            records.append(row)
            save_json(args.output / "results.json", records)
            save_csv(args.output / "results.csv", records)
            print(f"RESULT {variant}: {elapsed:.2f}s, {row['mean_update_ms']:.3f}ms/update, MSE={error:.9f}, PSNR={row['psnr_db']:.3f}dB", flush=True)
    finally:
        zs_n2n.ZSN2NNetwork = original_network
    summary = {}
    for variant in VARIANTS:
        rows = [r for r in records if r["variant"] == variant]
        summary[variant] = {"parameters": rows[0]["parameters"],
                            **{key: statistics.median(r[key] for r in rows)
                               for key in ("total_seconds", "mean_update_ms", "reference_mse", "psnr_db")}}
        summary[variant]["speedup_vs_original"] = summary["original"]["total_seconds"] / summary[variant]["total_seconds"]
    summary["noisy_mse"] = mse(noisy, reference)
    save_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
