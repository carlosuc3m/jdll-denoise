"""Compare Adam and AdamW on configurable-width, two-group ZS-N2N variants.

Only this process overrides the backend's optimizer/network/scheduler constructors. The
public denoise API, canonical loss, default CPU threads and prior inputs stay
unchanged. Weight decay applies to every parameter, including biases.
"""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import tifffile
import torch

from zs_n2n_middle_layers import MiddleVariant, expected_parameters
from zs_n2n_width import Progress, denoise, mse, psnr, request, resolve_config, save_csv, save_json, zs_n2n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--optimizers", nargs="+", choices=("adam", "adamw"), default=["adam", "adamw"])
    parser.add_argument("--scheduler", choices=("step", "onecycle"), default="step")
    parser.add_argument("--record-schedule", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=5489)
    args = parser.parse_args()
    if min(args.steps, args.repeats, args.warmup_steps) < 1:
        parser.error("steps, repeats and warmup-steps must be positive")
    if args.width < 2 or args.width % 2:
        parser.error("width must be a positive multiple of two")
    if args.scheduler == "onecycle" and (args.steps <= 10 or args.warmup_steps > args.steps):
        parser.error("OneCycleLR requires steps > 10 and warmup-steps <= steps")
    reference = tifffile.imread(args.input_dir / "reference.tif")
    noisy = tifffile.imread(args.input_dir / "noisy.tif")
    assert reference.ndim == 2 and reference.shape == noisy.shape
    assert np.isfinite(reference).all() and np.isfinite(noisy).all()
    assert 0 <= reference.min() <= reference.max() <= 1
    args.output.mkdir(parents=True, exist_ok=False)
    original_network, original_losses = zs_n2n.ZSN2NNetwork, zs_n2n.canonical_losses
    backend_hash = hashlib.sha256(Path(zs_n2n.__file__).read_bytes()).hexdigest()
    threads = {"intra_op": torch.get_num_threads(), "inter_op": torch.get_num_interop_threads()}
    variants = {"adam": (torch.optim.Adam, 0.0), "adamw": (torch.optim.AdamW, 1e-4)}
    variants = {name: variants[name] for name in args.optimizers}
    order = [name for repeat in range(args.repeats)
             for name in (tuple(variants) if repeat % 2 == 0 else tuple(variants)[::-1])]
    config = request(args.width, args.steps, args.seed, "cpu")
    scheduler_config = ({"class": "OneCycleLR", "max_lr": 3e-3, "total_steps": args.steps,
                         "pct_start": 0.1, "anneal_strategy": "cos", "cycle_momentum": True,
                         "base_momentum": 0.85, "max_momentum": 0.95, "div_factor": 25.0,
                         "final_div_factor": 1e4, "three_phase": False}
                        if args.scheduler == "onecycle" else
                        {"class": "StepLR", "step_size": 1000, "gamma": 0.5})
    save_json(args.output / "config.json", {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "actual_threads": threads, "python": sys.version, "torch": torch.__version__,
        "numpy": np.__version__, "shape_yx": list(noisy.shape), "run_order": order,
        "input_sha256": {name: hashlib.sha256((args.input_dir / name).read_bytes()).hexdigest()
                         for name in ("reference.tif", "noisy.tif")},
        "backend_sha256": backend_hash,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "network_script_sha256": hashlib.sha256(Path(sys.modules[MiddleVariant.__module__].__file__).read_bytes()).hexdigest(),
        "resolved_config": resolve_config(config, noisy.shape), "architecture": "groups2",
        "width": args.width, "pair_batching": False, "decay_biases": True,
        "benchmark_scheduler_override": scheduler_config,
        "optimizers": {name: {"class": cls.__name__, "weight_decay": wd}
                       for name, (cls, wd) in variants.items()},
        "reference_note": "Original acquisition, not verified noise-free; measures error after added synthetic noise.",
        "normalization_note": "Noisy-image min-max, matching previous experiments rather than GUI defaults.",
        "timing_note": "Warmed public API call including callbacks and cleanup, excluding imports, warmup, scoring and file writes. Default CPU threads and clocks untouched.",
    })
    tifffile.imwrite(args.output / "reference.tif", reference)
    tifffile.imwrite(args.output / "noisy.tif", noisy)
    print(f"Default CPU threads: {threads}; order: {order}", flush=True)
    records, outputs = [], {}
    for index, name in enumerate(order):
        repeat = index // len(variants) + 1
        optimizer_class, decay = variants[name]
        optimizer_settings = []
        optimizer_traces = []
        scheduler_initial_lrs = []

        def optimizer_factory(parameters, *, lr):
            optimizer = optimizer_class(parameters, lr=lr, weight_decay=decay)
            assert type(optimizer) is optimizer_class
            assert len(optimizer.param_groups) == 1
            assert sum(p.numel() for p in optimizer.param_groups[0]["params"]) == expected_parameters("groups2", 1, args.width)
            optimizer_settings.append({"class": type(optimizer).__name__, **optimizer.defaults})
            if args.scheduler == "onecycle" and args.record_schedule:
                trace = []
                optimizer_traces.append(trace)

                def record_step(actual_optimizer, positional, keywords):
                    group = actual_optimizer.param_groups[0]
                    trace.append({"update": len(trace) + 1, "lr": group["lr"],
                                  "beta1": group["betas"][0], "beta2": group["betas"][1]})

                optimizer.register_step_pre_hook(record_step)
            return optimizer

        def scheduler_factory(optimizer, *, step_size, gamma):
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=3e-3, total_steps=args.steps,
                pct_start=0.1, anneal_strategy="cos",
            )
            scheduler_initial_lrs.append(optimizer.param_groups[0]["lr"])
            return scheduler

        # Shadow torch only in zs_n2n; never replace the shared torch.optim.Adam.
        torch_proxy = SimpleNamespace(**vars(torch))
        torch_proxy.optim = SimpleNamespace(**vars(torch.optim))
        torch_proxy.optim.Adam = optimizer_factory
        if args.scheduler == "onecycle":
            torch_proxy.optim.lr_scheduler = SimpleNamespace(**vars(torch.optim.lr_scheduler))
            torch_proxy.optim.lr_scheduler.StepLR = scheduler_factory
        with patch.object(zs_n2n, "torch", torch_proxy), patch.object(
            zs_n2n, "ZSN2NNetwork", partial(MiddleVariant, variant="groups2")
        ):
            print(f"Round {repeat}/{args.repeats}, {name}: warmup", flush=True)
            denoise(noisy, request(args.width, args.warmup_steps, args.seed, "cpu"))
            progress = Progress(args.width, args.steps)
            started = time.perf_counter()
            result = denoise(noisy, config, callbacks=progress)
            elapsed = time.perf_counter() - started
        assert zs_n2n.torch is torch and zs_n2n.ZSN2NNetwork is original_network
        assert zs_n2n.canonical_losses is original_losses
        assert len(optimizer_settings) == 2 and optimizer_settings[0] == optimizer_settings[1]
        assert optimizer_settings[1]["weight_decay"] == decay
        assert optimizer_settings[1]["lr"] == 0.001
        metadata, output = result["metadata"], result["denoised"]
        job = metadata["jobs"][0]
        assert job["completed_iterations"] == job["selected_step"] == args.steps
        assert job["network_parameter_count"] == expected_parameters("groups2", 1, args.width)
        assert metadata["device_used"] == "cpu"
        assert all(job[key] == "cpu" for key in ("model_device", "training_input_device", "prediction_input_device"))
        assert progress.records[-1]["step"] == args.steps
        assert not any("callback " in warning for warning in metadata["warnings"])
        assert output.shape == reference.shape and output.dtype == np.float32 and np.isfinite(output).all()
        assert threads == {"intra_op": torch.get_num_threads(), "inter_op": torch.get_num_interop_threads()}
        if name in outputs:
            assert np.array_equal(output, outputs[name]), "Repeated run changed output"
        outputs[name] = output
        # The backend hardcodes its optimizer name; correct benchmark metadata.
        job["optimizer"] = optimizer_class.__name__
        job["optimizer_defaults"] = optimizer_settings[1]
        if args.scheduler == "onecycle":
            assert len(scheduler_initial_lrs) == 2
            assert np.isclose(scheduler_initial_lrs[-1], 3e-3 / 25, rtol=1e-12, atol=0)
            job["initial_learning_rate"] = scheduler_initial_lrs[-1]
            if args.record_schedule:
                assert [len(trace) for trace in optimizer_traces] == [args.warmup_steps, args.steps]
                trace = optimizer_traces[-1]
                assert np.isclose(trace[0]["lr"], 3e-3 / 25, rtol=1e-12, atol=0)
                assert np.isclose(trace[-1]["lr"], 3e-3 / 25 / 1e4, rtol=1e-12, atol=0)
                assert all(np.isfinite(row["lr"]) and row["lr"] > 0 for row in trace)
                assert all(0.85 <= row["beta1"] <= 0.95 and row["beta2"] == 0.999 for row in trace)
                job["last_update_learning_rate"] = trace[-1]["lr"]
            else:
                assert not optimizer_traces
            job["scheduler"] = "OneCycleLR"
            job.pop("scheduler_step_size", None)
            job.pop("scheduler_gamma", None)
            job["scheduler_configuration"] = scheduler_config
        metadata["benchmark_override"] = {"architecture": "groups2", "optimizer": optimizer_settings[1],
                                          "scheduler": scheduler_config}
        error = mse(output, reference)
        row = {
            "optimizer": name, "scheduler": args.scheduler, "weight_decay": decay,
            "width": args.width, "parameters": job["network_parameter_count"],
            "repeat": repeat, "run_order": index + 1,
            "updates": args.steps, "total_seconds": elapsed,
            "mean_update_ms": job["phase_timings_seconds"]["optimization"] * 1000 / args.steps,
            "reference_mse": error, "psnr_db": psnr(error),
        }
        stem = f"round_{repeat}_{name}"
        tifffile.imwrite(args.output / f"{stem}.tif", output)
        assert mse(tifffile.imread(args.output / f"{stem}.tif"), reference) == error
        save_json(args.output / f"{stem}_metadata.json", metadata)
        save_csv(args.output / f"{stem}_progress.csv", progress.records)
        if args.scheduler == "onecycle" and args.record_schedule:
            save_csv(args.output / f"{stem}_lr.csv", optimizer_traces[-1])
        records.append(row)
        save_json(args.output / "results.json", records)
        save_csv(args.output / "results.csv", records)
        print(f"RESULT {name}: {elapsed:.2f}s, {row['mean_update_ms']:.3f}ms/update, MSE={error:.9f}, PSNR={row['psnr_db']:.3f}dB", flush=True)
    assert hashlib.sha256(Path(zs_n2n.__file__).read_bytes()).hexdigest() == backend_hash
    summary = {name: {key: statistics.median(r[key] for r in records if r["optimizer"] == name)
                      for key in ("total_seconds", "mean_update_ms", "reference_mse", "psnr_db")}
               for name in variants}
    summary["noisy_mse"] = mse(noisy, reference)
    if "adam" in summary and "adamw" in summary:
        summary["adamw_mse_change_percent"] = 100 * (summary["adamw"]["reference_mse"] / summary["adam"]["reference_mse"] - 1)
        summary["adamw_time_change_percent"] = 100 * (summary["adamw"]["total_seconds"] / summary["adam"]["total_seconds"] - 1)
    summary["repeat_outputs_identical"] = args.repeats > 1
    save_json(args.output / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
