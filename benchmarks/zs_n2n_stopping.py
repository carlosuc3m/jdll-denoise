"""Measure reference MSE at every completed update, without changing training.

This is an oracle stopping study: the reference is never passed to the optimizer
or used for an actual stopping decision. It is not a deployable stopping rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import torch

from zs_n2n_width import Progress, denoise, mse, psnr, request, resolve_config, save_csv, save_json, zs_n2n
from jdll_denoise.normalization import Normalizer


class ReferenceCurve:
    def __init__(self, reference, normalizer, milestones):
        self.reference = reference
        self.normalizer = normalizer
        self.milestones = set(milestones)
        self.records = []
        self.images = {}
        self.best = None
        self.best_image = None
        self.observer_seconds = 0.0

    def __call__(self, model, inputs, predicted_noise):
        noisy = inputs[0]
        if tuple(noisy.shape) != (1, 1, *self.reference.shape):
            return
        started = time.perf_counter()
        # The training forward sees weights BEFORE the next update. The final
        # inference forward supplies the last point: updates 0 through N.
        updates = len(self.records)
        with torch.no_grad():
            normalized = torch.clamp(noisy - predicted_noise, 0.0, 1.0)[0, 0].cpu().numpy()
        output = self.normalizer.invert(normalized)
        error = mse(output, self.reference)
        row = {"updates": updates, "reference_mse": error, "psnr_db": psnr(error)}
        self.records.append(row)
        if self.best is None or error < self.best["reference_mse"]:
            self.best = row
            self.best_image = output.copy()
        if updates in self.milestones:
            self.images[updates] = output.copy()
        self.observer_seconds += time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=5489)
    parser.add_argument("--warmup-steps", type=int, default=20)
    args = parser.parse_args()
    if min(args.width, args.steps, args.warmup_steps) < 1:
        parser.error("width, steps and warmup must be positive")
    reference = tifffile.imread(args.input_dir / "reference.tif")
    noisy = tifffile.imread(args.input_dir / "noisy.tif")
    if reference.ndim != 2 or reference.shape != noisy.shape:
        raise ValueError("Expected matching grayscale images")
    if not np.isfinite(reference).all() or not np.isfinite(noisy).all():
        raise ValueError("Inputs must be finite")
    if reference.min() < 0 or reference.max() > 1:
        raise ValueError("PSNR assumes reference intensities in [0,1]")
    args.output.mkdir(parents=True, exist_ok=False)
    config = request(args.width, args.steps, args.seed, "cpu")
    threads = {"intra_op": torch.get_num_threads(), "inter_op": torch.get_num_interop_threads()}
    normalizer = Normalizer.fit(noisy, config["normalization"])
    milestones = sorted({n for n in (100, 250, 500, 750, 1000, 1500, 2000, 2500, args.steps)
                         if n <= args.steps})
    curve = ReferenceCurve(reference, normalizer, milestones)
    save_json(args.output / "config.json", {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "actual_threads": threads, "python": sys.version, "torch": torch.__version__,
        "input_sha256": {name: hashlib.sha256((args.input_dir / name).read_bytes()).hexdigest()
                         for name in ("reference.tif", "noisy.tif")},
        "backend_sha256": hashlib.sha256(Path(zs_n2n.__file__).read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "resolved_config": resolve_config(config, noisy.shape),
        "normalization": normalizer.metadata(),
        "reference_note": "Original acquisition, not verified noise-free; measures removal of added synthetic noise.",
        "timing_note": "Instrumented public API call, including every-update MSE observation. Not directly comparable to uninstrumented timings.",
    })
    print(f"Width={args.width}, updates={args.steps}, default CPU threads={threads}; warmup", flush=True)
    denoise(noisy, request(args.width, args.warmup_steps, args.seed, "cpu"))
    original_network = zs_n2n.ZSN2NNetwork

    def observed_network(*positional, **keyword):
        model = original_network(*positional, **keyword)
        model.register_forward_hook(curve)
        return model

    progress = Progress(args.width, args.steps)
    started = time.perf_counter()
    try:
        zs_n2n.ZSN2NNetwork = observed_network
        result = denoise(noisy, config, callbacks=progress)
    finally:
        zs_n2n.ZSN2NNetwork = original_network
    elapsed = time.perf_counter() - started
    job = result["metadata"]["jobs"][0]
    assert len(curve.records) == args.steps + 1
    assert job["completed_iterations"] == job["selected_step"] == args.steps
    assert result["metadata"]["device_used"] == "cpu"
    assert threads == {"intra_op": torch.get_num_threads(), "inter_op": torch.get_num_interop_threads()}
    assert not any("callback " in warning for warning in result["metadata"]["warnings"])
    np.testing.assert_array_equal(curve.images[args.steps], result["denoised"])
    best = curve.best
    within_one_percent = next(row for row in curve.records if row["reference_mse"] <= best["reference_mse"] * 1.01)
    summary = {
        "width": args.width, "maximum_updates": args.steps, "actual_threads": threads,
        "best_reference_mse": best,
        "best_within_1000_updates": min(curve.records[:1001], key=lambda r: r["reference_mse"]),
        "first_within_one_percent_of_best": within_one_percent,
        "milestones": [curve.records[n] for n in milestones],
        "minimum_smoothed_training_loss_checkpoint": curve.records[job["best_step"]],
        "instrumented_api_seconds": elapsed, "observer_seconds": curve.observer_seconds,
        "noisy_mse": mse(noisy, reference), "noisy_psnr_db": psnr(mse(noisy, reference)),
    }
    save_csv(args.output / "reference_curve.csv", curve.records)
    save_csv(args.output / "training_progress.csv", progress.records)
    save_json(args.output / "metadata.json", result["metadata"])
    save_json(args.output / "summary.json", summary)
    tifffile.imwrite(args.output / "reference.tif", reference)
    tifffile.imwrite(args.output / "noisy.tif", noisy)
    tifffile.imwrite(args.output / "best.tif", curve.best_image)
    for step, image in curve.images.items():
        tifffile.imwrite(args.output / f"updates_{step}.tif", image)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
