# Public API

The stable imports are:

```python
from jdll_denoise import capabilities, denoise, preview, resolve_config
```

`resolve_config(config, input_shape=None)` validates strict common and
method-specific keys, applies effort presets, resolves device/dimensional
choices, and returns a deep copy containing no hardware-related `auto` values.
Unknown keys fail; future application data belongs under `extensions`.

`denoise(image, config, *, callbacks=None, cancellation=None)` returns:

```python
{
    "denoised": contiguous_float32_array,
    "metadata": {
        "method": "noise2fast",
        "effort": "balanced",
        "device_requested": "auto",
        "device_used": "cuda",
        "dimension_mode": "slice_wise_2d",
        "input_shape": [2, 16, 512, 512],
        "output_shape": [2, 16, 512, 512],
        "normalization": [...],
        "resolved_config": {...},
        "seed": 5489,
        "warnings": [],
    },
}
```

`preview` uses the same path. For a volume/slab it returns the center Z plane;
`output_axes`, `preview_region`, `cache_reusable_for_full_input`, and
`preview_is_representative_only` describe that reduction.

## Common configuration

```python
config = {
    "method": "noise2fast",
    "effort": "balanced",       # quick | balanced | thorough
    "device": "cpu",            # cpu | cuda | mps | auto
    "dimensions": "auto",       # auto | 2d | 2.5d | 3d
    "axes": "cyx",
    "seed": 5489,
    "allow_cpu_fallback": False,
    "normalization": {
        "mode": "percentile", "low": 0.0, "high": 100.0,
        "per_channel": True, "clip": True,
    },
    "tiling": {
        "enabled": "auto", "tile_size": "auto", "overlap": "auto",
        "blend": "gaussian", "batch_size": "auto",
    },
    "method_config": {},
    "extensions": {},
}
```

Those normalization values show the Noise2Fast and ZS-N2N defaults (min-max scaling
to `[0,1]`). StructN2V defaults to robust 0.1/99.9 percentiles without clipping,
so outliers can remain outside `[0,1]` as in the paper's robust normalization.

Noise2Fast resolves hidden feature width and validation patience from the
[effort presets](../README.md#effort-presets), identically on CPU/CUDA/MPS.
Omitted effort means Balanced (24 features, patience 100 optimizer steps).
Advanced overrides take precedence, without mutating the supplied config:

```python
config["method_config"] = {"hidden_width": 32, "patience": 75}
```

Both options accept any positive Python integer, rejecting booleans, floats,
strings, zero and negative values. `steps` is a non-negative integer (default
0); `report_interval` is a positive integer (default 10). Reporting frequency
never changes training or validation frequency. Width is the hidden feature
count, not the image size. Quick/Balanced are compact-width variants, not the
released 64-feature architecture; higher capacity does not guarantee lower error.

Noise2Fast always constructs the four directed checkerboard pairs from the
complete plane, uses batch size 1, BCE and Adam at 0.001, and validates on the
full noisy image after every update. A strictly improved negative MSE resets
the stale-step counter to zero. Training stops when that counter reaches the
resolved patience. This score is against noisy input, not independent clean
ground truth. Patience is not a total training budget or time limit.

The output window is fixed at 100 independently of patience: average the most
recent `min(completed_steps, 100)` validation predictions. `steps=0` means
patience-only stopping. A positive cap can stop earlier; a warning records when
it causes termination. Inference tiling and sampled-patch training are rejected.
Default normalization is per-channel min-max, but valid explicit normalization
overrides are honored. `denoise()` and `preview()` share these settings; a crop
preview trains its own model and need not equal a crop of a full-image run.

The resolved-config event and `metadata["resolved_config"]` include actual
width, patience, output window, cap and requested/resolved device. Per-plane
metadata records the actual `architecture`, `hidden_width`,
`validation_patience`, `output_average_window`, `output_average_count`,
`completed_steps`, `best_step`, stale counter, `stopped_by`, devices/fallback
and timings. `stopped_by` is `validation_patience` or `configured_step_limit`.
`algorithm_fidelity` distinguishes a width variant from the released
64-feature architecture, without claiming standalone-program bitwise equivalence.

StructN2V resolves `patch_size`, `patch_count`, `batch_size`, `epochs`,
`learning_rate`, `mask_fraction`, `neighborhood_radius`, and the extended blind
mask geometry. Its defaults are the complete paper schedules. `auto` compares
all bounded mask families with deterministic held-out masked-prediction loss,
using autocorrelation as a secondary tie-breaker. The final loss remains active
only at N2V center pixels, not over the entire extended blind mask.

The Python backend always returns the full denoised result. It does not accept
or apply display strength. JDLL applies `strength * denoised + (1-strength) *
original` and performs any integer quantization.

## ZS-N2N effort contract

`method="zs_n2n"` accepts `quick`, `balanced` (default), `balanced_high` and
`thorough`, using the [version-1 preset table](../README.md#effort-presets).
`balanced-high` is accepted as an alias, normalized to `balanced_high` in
resolved config/results. Other methods reject both spellings.

All presets use the complete symmetric residual plus full-resolution
consistency objective. `batch_pairs=True` concatenates the two small views
along the batch dimension for one forward, followed by the full-size forward;
there is one backward and one optimizer update on their sum. `False` retains
the original three-forward path. Losses/gradients agree within floating-point
tolerance; long trajectories need not be bitwise identical.

Advanced `method_config` overrides apply after preset resolution:

| Key | Validation / behavior |
| --- | --- |
| `hidden_width` | Positive integer; legacy `features` alias is retained (both must agree if supplied). |
| `middle_groups` | Positive integer dividing hidden width; only the second convolution is grouped. RGB channel count need not divide it. |
| `output_init` | `native` or `near_zero`; only the final convolution is reinitialized for near-zero. |
| `output_init_std` | Finite positive number, resolved as a float, default 0.001; applied only for near-zero. |
| `batch_pairs` | Strict boolean; default true in all presets. |
| `steps`, `scheduler_step_size`, `report_interval`, `patience` | Positive integers (not booleans); `report_interval` defaults to 10. |
| `learning_rate`, `scheduler_gamma` | Finite positive numbers; gamma must also be at most 1. |
| `early_stopping`, `restore_best_weights` | Strict booleans, both false by default. |
| `min_delta` | Finite non-negative number for the advanced stopping/best-weight criterion. |

The unbatched reference architecture/optimization path is explicitly available:

```python
reference_config = {
    "method": "zs_n2n", "effort": "thorough", "axes": "yx", "device": "cpu",
    "method_config": {
        "hidden_width": 48, "middle_groups": 1, "output_init": "native",
        "steps": 3000, "batch_pairs": False,
    },
}
```

Direct `ZSN2NNetwork()` also retains width 48, one group and native initialization;
direct `canonical_losses(model, noisy)` retains unbatched execution. The public
API passes its resolved parameters explicitly. There are no hidden skips,
normalization layers, pooling, dropout or extra convolutions in any preset.

Optimization is Adam (LR 0.001, betas 0.9/0.999, epsilon 1e-8, no weight decay
or AMSGrad) and StepLR(1000, 0.5), applied after each update. Updates 1–1000 use
0.001, 1001–2000 use 0.0005 and 2001–3000 use 0.00025. Default runs complete
exactly their resolved budget and use final weights. Reporting frequency does
not change updates/loss evaluation, including advanced stopping/restoration.

Default normalization is per-channel min-max (0/100 percentiles, clipping);
explicit overrides are honored. Final inference is `clamp(noisy-model(noisy),
0, 1)`, then inverse normalization. This wrapper is not claimed to reproduce
the paper's exact preprocessing. Independent biological channels, native RGB
(`channel_mode="native_color"`), plane-wise stacks/time points, and optional
tiled final inference retain their contracts. Training remains full-plane.
Preview uses the same budget, architecture and normalization, not a hidden
shorter run; crop/slab region semantics are unchanged.

Capabilities, resolved configuration, top-level result metadata and per-plane
metadata carry integer `effort_presets_version=1`. Per-plane metadata includes
`hidden_width`, `middle_groups`, `output_init`, applicable `output_init_std`
(`null` for native), `batch_pairs`, `network_parameter_count`,
`configured_iterations`, `completed_iterations`, `selected_step`, optimizer
settings, schedule, actual devices and timings. The resolved config always
retains the configured standard deviation, even when not applied. Normalization
parameters/inversion are recorded in the existing top-level normalization list.

Fidelity distinguishes `jdll_performance_variant`,
`reference_architecture_batched`, `reference_architecture_unbatched`, custom
iteration/tiled suffixes and `noncanonical_extension` for advanced model or
optimization changes. Quick/Balanced/Balanced-high are not advertised as the
unmodified reference network. No preset guarantees speed or lower error.

## Events

Every callback event includes `type`, `phase`, `method`, `message`, `current`,
`maximum`, `channel`, `channels`, `z`, `z_total`, `metrics`, and `info`.
Optimization events include current and smoothed loss, learning rate, elapsed
seconds, and best step. Callback failures are isolated and appended to metadata
warnings.

ZS-N2N's initialization status names the resolved effort, width, groups, head
initialization/std, batching, update count, Adam/StepLR settings and device.
Optimization `maximum` is the resolved update budget; `current` is completed
updates. Metrics expose total/residual/consistency losses. `learning_rate` is
the rate used by that update; `next_learning_rate` is the scheduler's rate for
the next update. No Appose classes are required.

Noise2Fast emits a readable initialization status with the resolved effort,
width, patience, averaging window, learning rate and device. During optimization,
`current` counts completed optimizer updates. `maximum=0` means an unknown
total for patience-only runs; do not divide by it or substitute patience as the
denominator. With an explicit cap, `maximum` is that cap, but patience can stop
earlier. Metrics include `validation_patience` and the compatibility field
`epochs_without_validation_improvement`, which counts optimizer **steps**, not
dataset epochs. Messages state the stale/patience counter in steps. The final
optimization state is emitted even between reporting intervals, followed by
the normal completion event (1/1).

## Errors

All user-facing failures derive from `DenoisingError`: `ConfigurationError`,
`InputFormatError`, `MethodUnavailableError`, `DeviceUnavailableError`,
`DenoisingCancelled`, `OutOfMemoryDenoisingError`, and
`ThirdPartyBackendError`.
