"""Time 500-update initialization variants through the public API."""

import argparse
import gc
import hashlib
import os
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import numpy as np
import tifffile
import torch

from zs_n2n_initialization import InitializationVariant, check_variants
from zs_n2n_middle_layers import expected_parameters
from zs_n2n_width import denoise, mse, psnr, request, save_csv, save_json, zs_n2n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--groups", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--verify-previous", action="store_true")
    args = parser.parse_args()
    if args.width < 1 or args.width % args.groups:
        parser.error("width must be positive and divisible by groups")
    args.output.mkdir(parents=True, exist_ok=False)
    reference = tifffile.imread(args.previous / "reference.tif")
    noisy = tifffile.imread(args.previous / "noisy.tif")
    config = request(args.width, 500, 5489, "cpu")
    config["normalization"].update(low=0, high=100)
    order = ["small_output", "residual_small_output", "residual_small_output", "small_output"]
    threads = [torch.get_num_threads(), torch.get_num_interop_threads()]
    backend_hash = hashlib.sha256(Path(zs_n2n.__file__).read_bytes()).hexdigest()
    save_json(args.output / "config.json", {
        "request": config, "order": order, "threads_intra_inter": threads,
        "groups": args.groups, "output_init_std": 0.001, "output_bias": 0,
        "residual_note": "Hidden skip only; the middle convolution keeps native initialization.",
        "torch": torch.__version__, "shape_yx": list(noisy.shape),
        "backend_sha256": backend_hash,
        "input_sha256": {name: hashlib.sha256((args.previous / name).read_bytes()).hexdigest()
                         for name in ("reference.tif", "noisy.tif")},
        "warmup_updates_per_variant": 10, "callbacks": False,
        "timing_note": "Warmed public API calls; excludes imports, file I/O and quality scoring. "
                       "Optimization time comes from the backend; no intermediate prediction hooks. "
                       "CPU thread settings, affinity and clocks are not changed.",
    })
    save_json(args.output / "architecture_checks.json", check_variants(args.width, args.groups))
    print(f"CPU threads intra/inter: {threads}; warming both variants", flush=True)
    original_network = zs_n2n.ZSN2NNetwork

    def run(variant, run_config):
        def factory(*args, **kwargs):
            return InitializationVariant(*args, **kwargs, variant=variant, groups=groups)

        with patch.object(zs_n2n, "ZSN2NNetwork", factory):
            return denoise(noisy, run_config)

    groups = args.groups
    parameter_count = expected_parameters({1: "original", 2: "groups2", 4: "groups4"}[groups], 1, args.width)
    for variant in order[:2]:
        warmup = request(args.width, 10, 5489, "cpu")
        run(variant, warmup)
    rows = []
    for index, variant in enumerate(order, 1):
        gc.collect()
        print(f"Run {index}/4: {variant}, 500 updates", flush=True)
        load_before = os.getloadavg()
        started = time.perf_counter()
        result = run(variant, config)
        total = time.perf_counter() - started
        job = result["metadata"]["jobs"][0]
        assert job["completed_iterations"] == job["selected_step"] == 500
        assert job["hidden_width"] == args.width and job["network_parameter_count"] == parameter_count
        assert job["optimizer"] == "Adam" and result["metadata"]["device_used"] == "cpu"
        assert threads == [torch.get_num_threads(), torch.get_num_interop_threads()]
        if args.verify_previous:
            np.testing.assert_array_equal(result["denoised"], tifffile.imread(args.previous / f"{variant}_500.tif"))
        output_path = args.output / f"{variant}_500.tif"
        if output_path.exists():
            np.testing.assert_array_equal(result["denoised"], tifffile.imread(output_path))
        else:
            tifffile.imwrite(output_path, result["denoised"])
        error = mse(result["denoised"], reference)
        optimization = job["phase_timings_seconds"]["optimization"]
        row = {"run": index, "variant": variant, "updates": 500,
               "width": args.width, "groups": groups, "parameters": parameter_count,
               "optimization_seconds": optimization, "mean_update_ms": optimization * 2,
               "api_seconds": total, "reference_mse": error, "psnr_db": psnr(error),
               "load_1m_before": load_before[0], "load_1m_after": os.getloadavg()[0]}
        rows.append(row)
        result["metadata"]["benchmark_architecture"] = {
            "variant": variant, "width": args.width, "groups": groups,
            "output_init_std": 0.001, "output_bias": 0,
        }
        save_json(args.output / f"run_{index}_metadata.json", result["metadata"])
        save_json(args.output / "results.json", rows)
        save_csv(args.output / "results.csv", rows)
        print(f"RESULT {variant}: optimization={optimization:.3f}s, "
              f"API={total:.3f}s, update={optimization * 2:.3f}ms, MSE={error:.9f}", flush=True)
        del result
    assert zs_n2n.ZSN2NNetwork is original_network
    assert hashlib.sha256(Path(zs_n2n.__file__).read_bytes()).hexdigest() == backend_hash
    summary = {}
    for variant in order[:2]:
        times = [r["optimization_seconds"] for r in rows if r["variant"] == variant]
        totals = [r["api_seconds"] for r in rows if r["variant"] == variant]
        summary[variant] = {"median_optimization_seconds": statistics.median(times),
                            "min_optimization_seconds": min(times), "max_optimization_seconds": max(times),
                            "median_api_seconds": statistics.median(totals),
                            "mean_update_ms": statistics.median(times) * 2}
    save_json(args.output / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
