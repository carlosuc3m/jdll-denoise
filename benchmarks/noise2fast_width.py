"""Compare Noise2Fast effort presets or advanced widths through denoise().

The source channel is a reference for added synthetic noise, not necessarily a
noise-free acquisition. Gaussian noise is not clipped. Training uses the same
min-max-normalized noisy image for every width; scoring reverses that transform.
The public API resolves all widths/patience settings, normalizes the noisy
input and restores its intensity scale. No constructor substitution is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jdll_denoise import denoise, resolve_config
from jdll_denoise.methods import noise2fast


class StepTimings:
    def __init__(self, width: int, report: bool):
        self.width = width
        self.report = report
        self.records = []
        self.previous_elapsed = 0.0
        self.previous_callback_seconds = 0.0

    def __call__(self, event):
        if event["type"] != "denoise_progress" or event["phase"] != "optimization":
            return
        started = time.perf_counter()
        current = event["current"]
        metrics = event["metrics"]
        elapsed = metrics["elapsed_seconds"]
        seconds = elapsed - self.previous_elapsed - self.previous_callback_seconds
        self.records.append(
            {
                "step": current,
                "step_seconds": seconds,
                "elapsed_seconds": elapsed,
                "training_bce": metrics["loss"],
                "validation_mse_to_normalized_noisy": -metrics["validation_negative_mse"],
                "best_step": metrics["best_step"],
                "stale_steps": metrics["epochs_without_validation_improvement"],
            }
        )
        if self.report and (current == 1 or current % 50 == 0):
            print(
                f"width={self.width} step={current} elapsed={elapsed:.1f}s "
                f"last_step={seconds:.3f}s best_step={metrics['best_step']} "
                f"stale={metrics['epochs_without_validation_improvement']}",
                flush=True,
            )
        self.previous_elapsed = elapsed
        self.previous_callback_seconds = time.perf_counter() - started


def run_model(image, config, args, maximum_steps=0, report=True):
    config = {
        **config,
        "method_config": {
            **config.get("method_config", {}),
            "steps": maximum_steps,
            "report_interval": 1,
        },
    }
    width = resolve_config(config, image.shape)["method_config"]["hidden_width"]
    timings = StepTimings(width, report)
    if args.device == "cuda":
        torch.cuda.synchronize()
    elif args.device == "mps":
        torch.mps.synchronize()
    started = time.perf_counter()
    result = denoise(image, config, callbacks=timings)
    if args.device == "cuda":
        torch.cuda.synchronize()
    elif args.device == "mps":
        torch.mps.synchronize()
    total = time.perf_counter() - started
    return result["denoised"], result["metadata"], timings.records, total


def mse(left, right):
    return float(np.mean((left.astype(np.float64) - right.astype(np.float64)) ** 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=25 / 255)
    parser.add_argument("--seed", type=int, default=5489)
    parser.add_argument("--noise-seed", type=int, default=20260908)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--efforts",
        nargs="+",
        choices=("quick", "balanced", "thorough"),
        help="Production presets; default: all three",
    )
    selection.add_argument(
        "--widths",
        type=int,
        nargs="+",
        help="Advanced widths on top of the default Balanced preset",
    )
    parser.add_argument(
        "--patience",
        type=int,
        help="Advanced override; omitted uses each resolved preset's patience",
    )
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0, help="0: patience only")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.patience is not None and args.patience < 1:
        parser.error("--patience must be at least 1")
    if args.widths is not None and any(width < 1 for width in args.widths):
        parser.error("--widths must contain positive integers")
    if args.max_steps < 0 or args.warmup_steps < 0:
        parser.error("step counts must be non-negative")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    args.output.mkdir(parents=True, exist_ok=False)
    with Image.open(args.image) as handle:
        source = np.asarray(handle)
    if source.ndim == 3:
        source = source[..., args.channel]
    if source.ndim != 2 or source.dtype != np.uint8:
        raise ValueError("This benchmark expects an 8-bit grayscale or RGB source image")
    clean = source.astype(np.float32) / 255
    noisy = clean + np.random.default_rng(args.noise_seed).normal(
        0, args.sigma, clean.shape
    ).astype(np.float32)
    offset = float(noisy.min())
    scale = float(noisy.max() - offset)
    runs = []
    selections = (
        args.widths
        if args.widths is not None
        else (args.efforts or ("quick", "balanced", "thorough"))
    )
    for selection in selections:
        method_config = {}
        if args.widths is not None:
            method_config["hidden_width"] = selection
        if args.patience is not None:
            method_config["patience"] = args.patience
        config = {
            "method": "noise2fast",
            "axes": "yx",
            "device": args.device,
            "seed": args.seed,
            "method_config": method_config,
        }
        if args.widths is None:
            config["effort"] = selection
        label = f"width_{selection}" if args.widths is not None else str(selection)
        runs.append((label, config))
    configuration = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "shape_yx": list(clean.shape),
        "image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        "backend_sha256": hashlib.sha256(Path(noise2fast.__file__).read_bytes()).hexdigest(),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 1,
        "api": "jdll_denoise.denoise",
        "output_window": 100,
        "requests": [config for _, config in runs],
        "normalization": {"offset": offset, "scale": scale, "noise_clipped": False},
        "timing_scope": (
            "Public denoise() call, including configuration, normalization, model "
            "initialization, final averaging, intensity restoration and cleanup; "
            "excludes imports, optional warmup, synthetic input preparation and file writes. "
            "Step timing includes optimization, full-image validation and prediction "
            "transfer, excluding preceding callback body."
        ),
        "reference_note": (
            "Original acquisition channel, not independently established noise-free "
            "ground truth. Measures removal of additional synthetic Gaussian noise."
        ),
        "background_note": "Concurrent workloads are not monitored or controlled by this script.",
    }
    (args.output / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    tifffile.imwrite(args.output / "reference.tif", clean)
    tifffile.imwrite(args.output / "noisy.tif", noisy)
    for _, config in runs:
        if args.warmup_steps:
            run_model(noisy, config, args, args.warmup_steps, report=False)
    print("Warmup complete; starting measured runs", flush=True)

    records = []
    display_images = [clean, noisy]
    noisy_mse = mse(noisy, clean)
    for label, config in runs:
        result, api_metadata, trace, total = run_model(noisy, config, args, args.max_steps)
        metadata = api_metadata["jobs"][0]
        width = metadata["hidden_width"]
        final_mse = mse(result, clean)
        durations = np.asarray([row["step_seconds"] for row in trace])
        record = {
            "run": label,
            "effort": api_metadata["effort"],
            "width": width,
            "patience": metadata["validation_patience"],
            "parameters": 63 * width * width + 18 * width + 1,
            "steps": metadata["completed_steps"],
            "best_step": metadata["best_step"],
            "stopped_by": metadata["stopped_by"],
            "total_seconds": total,
            "step_seconds_mean": float(durations.mean()),
            "step_seconds_median": float(np.median(durations)),
            "step_seconds_p10": float(np.percentile(durations, 10)),
            "step_seconds_p90": float(np.percentile(durations, 90)),
            "noisy_reference_mse": noisy_mse,
            "final_reference_mse": final_mse,
            "final_reference_psnr_db": float(-10 * np.log10(final_mse)),
            "mse_reduction_percent": 100 * (1 - final_mse / noisy_mse),
            "backend_metadata": metadata,
            "resolved_config": api_metadata["resolved_config"],
        }
        records.append(record)
        tifffile.imwrite(args.output / f"denoised_{label}.tif", result)
        display_images.append(result)
        with (args.output / f"steps_{label}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trace[0]))
            writer.writeheader()
            writer.writerows(trace)
        (args.output / "results.json").write_text(json.dumps(records, indent=2) + "\n")
        print(
            json.dumps({key: value for key, value in record.items() if key != "backend_metadata"}),
            flush=True,
        )

    # Display clipping is only for this side-by-side preview, never for scoring.
    Image.fromarray(
        np.round(np.clip(np.concatenate(display_images, axis=1), 0, 1) * 255).astype(np.uint8)
    ).save(args.output / "comparison.png")
    lines = [
        "# Noise2Fast Width Comparison",
        "",
        f"Source: `{args.image}`, channel {args.channel}, shape {list(clean.shape)}.",
        f"Device: {args.device}; CPU threads: {args.threads}; Gaussian sigma: {args.sigma:.8f}.",
        "Settings resolved by the public API; final output: "
        "last 100 available predictions averaged.",
        "Clean-reference MSE does not participate in training or stopping.",
        "",
        "| Run | Width | Patience | Steps | Total seconds | "
        "Mean seconds/step | Median seconds/step | Final MSE |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        lines.append(
            f"| {record['run']} | {record['width']} | {record['patience']} | "
            f"{record['steps']} | {record['total_seconds']:.3f} | "
            f"{record['step_seconds_mean']:.6f} | {record['step_seconds_median']:.6f} | "
            f"{record['final_reference_mse']:.9f} |"
        )
    lines.extend(
        [
            "",
            f"Noisy-reference MSE: {noisy_mse:.9f}.",
            "Preview order: reference, noisy, then denoised outputs in configured width order.",
            "One image/noise realization/initialization seed; concurrent workloads uncontrolled.",
            "See config.json for timing scope and reproducibility details.",
        ]
    )
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
