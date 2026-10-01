"""Reproducible single-case benchmark with cold and warmed algorithm timings."""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc

PROCESS_STARTED = time.perf_counter()


def main() -> None:
    # Heavy imports deliberately occur after PROCESS_STARTED so their one-time
    # cost is visible separately from warmed optimization/inference phases.
    import numpy as np
    import torch

    from jdll_denoise import denoise

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="noise2fast")
    parser.add_argument("--effort", default="quick")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--kind", choices=("grayscale", "rgb", "stack"), default="grayscale")
    parser.add_argument("--stack-depth", type=int, default=4)
    parser.add_argument("--warm-runs", type=int, default=1)
    args = parser.parse_args()

    rng = np.random.default_rng(5489)
    if args.kind == "rgb":
        image = rng.random((3, args.size, args.size), dtype=np.float32)
        axes = "cyx"
    elif args.kind == "stack":
        image = rng.random((args.stack_depth, args.size, args.size), dtype=np.float32)
        axes = "zyx"
    else:
        image = rng.random((args.size, args.size), dtype=np.float32)
        axes = "yx"
    config = {
        "method": args.method,
        "effort": args.effort,
        "device": args.device,
        "axes": axes,
        "seed": 5489,
    }
    if args.steps is not None:
        config["method_config"] = {"steps": args.steps}
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tracemalloc.start()
    cold_started = time.perf_counter()
    cold_result = denoise(image, config)
    cold_elapsed = time.perf_counter() - cold_started
    process_start_to_cold_result = time.perf_counter() - PROCESS_STARTED
    warm_elapsed: list[float] = []
    warm_result = cold_result
    for _ in range(args.warm_runs):
        started = time.perf_counter()
        warm_result = denoise(image, config)
        warm_elapsed.append(time.perf_counter() - started)
    _, host_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    record = {
        "method": args.method,
        "effort": args.effort,
        "device": warm_result["metadata"]["device_used"],
        "shape": list(image.shape),
        "kind": args.kind,
        "cold_algorithm_end_to_end_seconds": cold_elapsed,
        "process_start_to_cold_result_seconds": process_start_to_cold_result,
        "warmed_algorithm_end_to_end_seconds": warm_elapsed,
        "peak_python_bytes": host_peak,
        "peak_cuda_bytes": (
            torch.cuda.max_memory_allocated()
            if args.device == "cuda" and torch.cuda.is_available()
            else None
        ),
        "torch_version": torch.__version__,
        "cold_jobs": cold_result["metadata"]["jobs"],
        "warmed_jobs": warm_result["metadata"]["jobs"],
        "cold_api_phase_timings_seconds": cold_result["metadata"]["phase_timings_seconds"],
        "warmed_api_phase_timings_seconds": warm_result["metadata"]["phase_timings_seconds"],
    }
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
