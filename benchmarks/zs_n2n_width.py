"""Benchmark ZS-N2N widths/update budgets through the production denoise API.

Reuse reference/noisy TIFFs from the Noise2Fast experiment. Each matrix cell
starts from the same model seed, preserving the original objective and LR
schedule. Min-max normalization matches that experiment, not the GUI's default
percentile clipping. The reference acquisition is not proven noise-free.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jdll_denoise import denoise, resolve_config
from jdll_denoise.methods import zs_n2n


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def gpu_status():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,temperature.gpu,power.draw,clocks.sm,"
         "clocks.mem,utilization.gpu,memory.used", "--format=csv"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def mse(left, right):
    return float(np.mean((left.astype(np.float64) - right.astype(np.float64)) ** 2))


def psnr(error):
    return float(-10 * np.log10(max(error, np.finfo(np.float64).tiny)))


def request(width, steps, seed, device="cuda"):
    return {
        "method": "zs_n2n", "axes": "yx", "dimensions": "2d",
        "effort": "thorough", "device": device, "seed": seed,
        "allow_cpu_fallback": False,
        "normalization": {"mode": "percentile", "low": 0, "high": 100,
                          "per_channel": True, "clip": True},
        "tiling": {"enabled": False},
        "method_config": {
            "hidden_width": width, "steps": steps, "learning_rate": 0.001,
            "scheduler_step_size": 1000, "scheduler_gamma": 0.5,
            "early_stopping": False, "restore_best_weights": False,
            "report_interval": 50,
        },
    }


class Progress:
    def __init__(self, width, steps):
        self.width, self.steps = width, steps
        self.records = []

    def __call__(self, event):
        if event["type"] != "denoise_progress" or event["phase"] != "optimization":
            return
        metrics = event["metrics"]
        self.records.append({"step": event["current"], **metrics})
        if event["current"] % 500 == 0 or event["current"] == self.steps:
            print(f"width={self.width} updates={event['current']}/{self.steps} "
                  f"optimization={metrics['elapsed_seconds']:.2f}s", flush=True)


def save_csv(path, records):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def make_comparison(output_dir, reference, noisy, widths, checkpoints):
    height, width = reference.shape
    header = 42
    canvas = Image.new("RGB", (width * len(checkpoints), (height + header) * (len(widths) + 1)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    def paste(array, column, row, label):
        x, y = column * width, row * (height + header)
        display = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
        canvas.paste(Image.fromarray(display).convert("RGB"), (x, y + header))
        draw.text((x + 8, y + 10), label, fill="black", font=font)

    paste(reference, 0, 0, "Reference acquisition")
    if len(checkpoints) > 1:
        paste(noisy, 1, 0, "Added Gaussian noise, sigma=25/255")
    for row, hidden_width in enumerate(widths, start=1):
        for column, steps in enumerate(checkpoints):
            image = tifffile.imread(output_dir / f"width_{hidden_width}_updates_{steps}.tif")
            paste(image, column, row, f"Width {hidden_width}, {steps} updates")
    canvas.save(output_dir / "comparison.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--widths", type=int, nargs="+", default=[48, 24, 16])
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[500, 1000, 2000, 3000])
    parser.add_argument("--seed", type=int, default=5489)
    parser.add_argument("--order-seed", type=int, default=20260915)
    thread_options = parser.add_mutually_exclusive_group()
    thread_options.add_argument("--threads", type=int, default=4)
    thread_options.add_argument("--default-threads", action="store_true",
                                help="Leave both PyTorch thread pools at their environment defaults")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--warmup-steps", type=int, default=20)
    args = parser.parse_args()
    if min(args.widths + args.checkpoints + [args.threads, args.warmup_steps]) < 1:
        parser.error("widths, checkpoints, threads and warmup must be positive")
    if len(set(args.widths)) != len(args.widths) or len(set(args.checkpoints)) != len(args.checkpoints):
        parser.error("widths and checkpoints must be unique")
    use_cuda = args.device == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; CPU fallback is disabled")
    if not args.default_threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
    actual_threads = {"intra_op": torch.get_num_threads(),
                      "inter_op": torch.get_num_interop_threads()}
    print(f"PyTorch threads: {actual_threads}; environment defaults={args.default_threads}", flush=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    reference = tifffile.imread(args.input_dir / "reference.tif")
    noisy = tifffile.imread(args.input_dir / "noisy.tif")
    if reference.ndim != 2 or reference.shape != noisy.shape:
        raise ValueError("Expected matching grayscale reference/noisy TIFFs")
    if not np.isfinite(reference).all() or not np.isfinite(noisy).all():
        raise ValueError("Inputs must be finite")
    if reference.min() < 0 or reference.max() > 1:
        raise ValueError("PSNR reporting expects reference intensities in [0,1]")
    args.output.mkdir(parents=True, exist_ok=False)
    order = [(width, steps) for width in args.widths for steps in args.checkpoints]
    random.Random(args.order_seed).shuffle(order)
    source_config = args.input_dir / "config.json"
    configuration = {
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "run_order": order, "python": sys.version, "torch": torch.__version__,
        "actual_threads": actual_threads,
        "numpy": np.__version__, "platform": platform.platform(),
        "device": torch.cuda.get_device_name() if use_cuda else platform.processor(),
        "shape_yx": list(reference.shape),
        "gpu_before": gpu_status() if use_cuda else None,
        "input_sha256": {name: hashlib.sha256((args.input_dir / name).read_bytes()).hexdigest()
                         for name in ("reference.tif", "noisy.tif")},
        "backend_sha256": hashlib.sha256(Path(zs_n2n.__file__).read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_config": json.loads(source_config.read_text()) if source_config.is_file() else None,
        "reference_note": "Original acquisition, not verified noise-free; measures removal of added synthetic noise.",
        "normalization_note": "Noisy-image min-max, shared across all runs and matching the prior Noise2Fast experiment; not GUI percentile defaults.",
        "timing_note": "Independent warmed public API calls. Includes setup, normalization, training, prediction, callbacks and cleanup; excludes imports, warmup, metrics and file writes. Mean update time is backend optimization time divided by updates, not CUDA-kernel-only time.",
        "precision": "float32; TF32 disabled; deterministic cuDNN; no AMP",
        "repetitions": 1,
    }
    save_json(args.output / "config.json", configuration)
    tifffile.imwrite(args.output / "reference.tif", reference)
    tifffile.imwrite(args.output / "noisy.tif", noisy)
    noisy_mse = mse(noisy, reference)
    print(f"Input {reference.shape}, noisy MSE={noisy_mse:.9f}, PSNR={psnr(noisy_mse):.3f} dB", flush=True)

    records = []
    for index, (width, steps) in enumerate(order, start=1):
        print(f"Run {index}/{len(order)}: width={width}, updates={steps}; warmup", flush=True)
        denoise(noisy, request(width, args.warmup_steps, args.seed, args.device))
        if use_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        config = request(width, steps, args.seed, args.device)
        progress = Progress(width, steps)
        started = time.perf_counter()
        result = denoise(noisy, config, callbacks=progress)
        if use_cuda:
            torch.cuda.synchronize()
        total_seconds = time.perf_counter() - started
        output, metadata = result["denoised"], result["metadata"]
        job = metadata["jobs"][0]
        assert metadata["device_used"] == args.device, metadata["device_used"]
        assert torch.get_num_threads() == actual_threads["intra_op"]
        assert torch.get_num_interop_threads() == actual_threads["inter_op"]
        assert job["completed_iterations"] == steps
        assert job["selected_step"] == steps
        assert output.shape == reference.shape and np.isfinite(output).all()
        assert progress.records[-1]["step"] == steps
        assert not any("callback " in warning for warning in metadata["warnings"])
        error = mse(output, reference)
        optimization = job["phase_timings_seconds"]["optimization"]
        row = {
            "width": width, "updates": steps, "run_order": index,
            "parameters": job["network_parameter_count"],
            "total_seconds": total_seconds, "optimization_seconds": optimization,
            "mean_update_ms": 1000 * optimization / steps,
            "inference_seconds": job["phase_timings_seconds"]["inference"],
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if use_cuda else None,
            "noisy_mse": noisy_mse, "reference_mse": error,
            "psnr_db": psnr(error), "psnr_gain_db": psnr(error) - psnr(noisy_mse),
            "mse_reduction_percent": 100 * (1 - error / noisy_mse),
        }
        stem = f"width_{width}_updates_{steps}"
        tifffile.imwrite(args.output / f"{stem}.tif", output)
        save_csv(args.output / f"{stem}_progress.csv", progress.records)
        save_json(args.output / f"{stem}_metadata.json", metadata)
        records.append(row)
        save_json(args.output / "results.json", records)
        save_csv(args.output / "results.csv", sorted(records, key=lambda r: (-r["width"], r["updates"])))
        peak = f"{row['peak_allocated_mib']:.1f}MiB" if use_cuda else "n/a (CPU)"
        print(f"RESULT width={width} updates={steps} total={total_seconds:.2f}s "
              f"mean_update={row['mean_update_ms']:.3f}ms MSE={error:.9f} "
              f"PSNR={row['psnr_db']:.3f}dB peak={peak}", flush=True)

    configuration["gpu_after"] = gpu_status() if use_cuda else None
    configuration["example_resolved_config"] = resolve_config(request(args.widths[0], args.checkpoints[-1], args.seed, args.device), noisy.shape)
    save_json(args.output / "config.json", configuration)
    make_comparison(args.output, reference, noisy, args.widths, sorted(args.checkpoints))
    print(f"Completed {len(records)} runs: {args.output}", flush=True)


if __name__ == "__main__":
    main()
