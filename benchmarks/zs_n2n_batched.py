"""Compare separate and batched ZS-N2N pair forwards on CPU.

Only the benchmark process substitutes the loss implementation. Architecture,
loss weights, optimizer, preprocessing, callbacks and CPU thread defaults stay
unchanged. Each timing includes a complete warmed public denoise API call.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from functools import partial
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

import numpy as np
import tifffile
import torch
from torch.nn import functional as F

from zs_n2n_width import Progress, denoise, mse, psnr, request, resolve_config, save_csv, save_json, zs_n2n
from zs_n2n_initialization import InitializationVariant


def batched_losses(model, noisy):
    first, second = zs_n2n.pair_downsampler(noisy)
    noise_first, noise_second = model(torch.cat((first, second), dim=0)).chunk(2, dim=0)
    prediction_first = first - noise_first
    prediction_second = second - noise_second
    full_prediction = noisy - model(noisy)
    full_first, full_second = zs_n2n.pair_downsampler(full_prediction)
    residual = 0.5 * (F.mse_loss(first, prediction_second) + F.mse_loss(second, prediction_first))
    consistency = 0.5 * (
        F.mse_loss(prediction_first, full_first) + F.mse_loss(prediction_second, full_second)
    )
    return residual + consistency, residual, consistency


def check_equivalence(width):
    checks = []
    for shape in ((1, 1, 32, 34), (1, 1, 33, 35), (2, 3, 31, 33)):
        for dtype in (torch.float64, torch.float32):
            torch.manual_seed(5489)
            original = zs_n2n.ZSN2NNetwork(shape[1], width).to(dtype=dtype)
            batched = deepcopy(original)
            x = torch.rand(shape, dtype=dtype, requires_grad=True)
            y = x.detach().clone().requires_grad_(True)
            calls_original, calls_batched = [], []
            original.register_forward_hook(lambda model, inputs, output: calls_original.append(tuple(inputs[0].shape)))
            batched.register_forward_hook(lambda model, inputs, output: calls_batched.append(tuple(inputs[0].shape)))
            loss_original = zs_n2n.canonical_losses(original, x)
            loss_batched = batched_losses(batched, y)
            tolerance = {"rtol": 1e-10, "atol": 1e-12} if dtype == torch.float64 else {"rtol": 2e-5, "atol": 1e-7}
            for a, b in zip(loss_original, loss_batched):
                torch.testing.assert_close(a, b, **tolerance)
            loss_original[0].backward()
            loss_batched[0].backward()
            gradients = [(x.grad, y.grad)] + [(a.grad, b.grad) for a, b in zip(original.parameters(), batched.parameters())]
            for a, b in gradients:
                torch.testing.assert_close(a, b, **tolerance)
            assert len(calls_original) == 3 and len(calls_batched) == 2
            assert calls_batched[0] == (shape[0] * 2, shape[1], shape[2] // 2, shape[3] // 2)
            checks.append({
                "shape": list(shape), "dtype": str(dtype),
                "max_loss_difference": max(float((a-b).abs().detach()) for a, b in zip(loss_original, loss_batched)),
                "max_gradient_difference": max(float((a-b).abs().max()) for a, b in gradients),
                "original_forward_shapes": calls_original, "batched_forward_shapes": calls_batched,
                "tolerance": tolerance,
            })
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=5489)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--near-zero-init", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if min(args.width, args.steps, args.repeats, args.warmup_steps) < 1:
        parser.error("width, steps, repeats and warmup must be positive")
    reference = tifffile.imread(args.input_dir / "reference.tif")
    noisy = tifffile.imread(args.input_dir / "noisy.tif")
    if reference.ndim != 2 or reference.shape != noisy.shape:
        raise ValueError("Expected matching grayscale images")
    if not np.isfinite(reference).all() or not np.isfinite(noisy).all():
        raise ValueError("Inputs must be finite")
    if reference.min() < 0 or reference.max() > 1:
        raise ValueError("PSNR expects reference intensities in [0,1]")
    args.output.mkdir(parents=True, exist_ok=False)
    threads = {"intra_op": torch.get_num_threads(), "inter_op": torch.get_num_interop_threads()}
    print(f"Default CPU threads: {threads}; checking losses, gradients and forward counts", flush=True)
    original_network = zs_n2n.ZSN2NNetwork
    network_factory = (partial(InitializationVariant, variant="small_output", groups=1)
                       if args.near_zero_init else original_network)
    with patch.object(zs_n2n, "ZSN2NNetwork", network_factory):
        checks = check_equivalence(args.width)
    save_json(args.output / "equivalence.json", checks)
    config = request(args.width, args.steps, args.seed, "cpu")
    configuration = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "actual_threads": threads, "python": sys.version, "torch": torch.__version__,
        "numpy": np.__version__, "shape_yx": list(noisy.shape),
        "input_sha256": {name: hashlib.sha256((args.input_dir / name).read_bytes()).hexdigest()
                         for name in ("reference.tif", "noisy.tif")},
        "backend_sha256": hashlib.sha256(Path(zs_n2n.__file__).read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "resolved_config": resolve_config(config, noisy.shape),
        "groups": 1, "hidden_residual": False,
        "output_init": "Normal(0, 1e-3), bias=0" if args.near_zero_init else "native",
        "callbacks_enabled": not args.no_progress,
        "run_order": [variant for repeat in range(args.repeats)
                      for variant in (("original", "batched") if repeat % 2 == 0 else ("batched", "original"))],
        "reference_note": "Original acquisition, not verified noise-free; measures removal of added synthetic noise.",
        "normalization_note": "Noisy-image min-max as in previous benchmarks, not GUI percentile defaults.",
        "timing_note": "Warmed public API calls including setup, training, prediction, callbacks and cleanup; excluding correctness tests, warmup, scoring and file writes. No CPU clock or affinity changes.",
    }
    save_json(args.output / "config.json", configuration)
    tifffile.imwrite(args.output / "reference.tif", reference)
    tifffile.imwrite(args.output / "noisy.tif", noisy)
    original_losses = zs_n2n.canonical_losses
    records = []
    try:
        zs_n2n.ZSN2NNetwork = network_factory
        for index, variant in enumerate(configuration["run_order"]):
            repeat = index // 2 + 1
            print(f"Round {repeat}/{args.repeats}, {variant}; warmup", flush=True)
            zs_n2n.canonical_losses = original_losses if variant == "original" else batched_losses
            denoise(noisy, request(args.width, args.warmup_steps, args.seed, "cpu"))
            progress = Progress(args.width, args.steps)
            started = time.perf_counter()
            result = denoise(noisy, config, callbacks=None if args.no_progress else progress)
            elapsed = time.perf_counter() - started
            metadata = result["metadata"]
            job = metadata["jobs"][0]
            output = result["denoised"]
            assert metadata["device_used"] == "cpu"
            assert job["completed_iterations"] == job["selected_step"] == args.steps
            assert job["network_parameter_count"] == 9 * args.width**2 + 12 * args.width + 1
            if not args.no_progress:
                assert progress.records[-1]["step"] == args.steps
            assert not any("callback " in warning for warning in metadata["warnings"])
            assert output.shape == reference.shape and np.isfinite(output).all()
            assert threads == {"intra_op": torch.get_num_threads(), "inter_op": torch.get_num_interop_threads()}
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
            metadata["benchmark_architecture"] = {
                "width": args.width, "groups": 1, "hidden_residual": False,
                "output_init": configuration["output_init"], "pair_batching": variant == "batched",
            }
            tifffile.imwrite(args.output / f"{stem}.tif", output)
            save_json(args.output / f"{stem}_metadata.json", metadata)
            if progress.records:
                save_csv(args.output / f"{stem}_progress.csv", progress.records)
            records.append(row)
            save_json(args.output / "results.json", records)
            save_csv(args.output / "results.csv", records)
            print(f"RESULT {variant}: {elapsed:.2f}s, {row['mean_update_ms']:.3f}ms/update, MSE={error:.9f}, PSNR={row['psnr_db']:.3f}dB", flush=True)
    finally:
        zs_n2n.canonical_losses = original_losses
        zs_n2n.ZSN2NNetwork = original_network
    assert hashlib.sha256(Path(zs_n2n.__file__).read_bytes()).hexdigest() == configuration["backend_sha256"]
    summary = {}
    for variant in ("original", "batched"):
        rows = [r for r in records if r["variant"] == variant]
        summary[variant] = {key: statistics.median(r[key] for r in rows)
                            for key in ("total_seconds", "mean_update_ms", "reference_mse", "psnr_db")}
    summary["median_speedup"] = summary["original"]["total_seconds"] / summary["batched"]["total_seconds"]
    summary["noisy_mse"] = mse(noisy, reference)
    save_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
