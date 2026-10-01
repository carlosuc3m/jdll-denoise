# Noise2Fast Performance Requirements

> **Historical performance proposal, superseded for the canonical method.**
> Sampled-patch training and tiled reconstruction change Noise2Fast's defining
> fixed four-image dataset and full-image validation/reconstruction. They were
> removed in the paper-fidelity implementation. The agreed
> [effort-preset requirements](README_NOISE2FAST_EFFORT_PRESETS_REQUIREMENTS.md)
> now authorize only width/patience variants: Quick 16/50, Balanced 24/100,
> Thorough 64/100. The existing full-plane loop and fixed 100-prediction window
> remain unchanged. Compact variants need not be numerically equivalent to
> width 64, but may not restore the bounded approximation proposed below.

## Problem

Noise2Fast `quick` mode is not currently bounded to an interactive workload. It
runs 400 optimization steps on a checkerboard view of the complete
native-resolution plane. Training cost therefore grows directly with image area;
tiling only applies to the final prediction. Large microscopy images can make
`quick` mode take minutes.

The current CUDA path also synchronizes with the host on every optimization step
through `float(loss.detach().cpu())`. Tiled prediction transfers and evaluates
one NumPy tile at a time, adding repeated host/device synchronization and copies.

JDLL also had an integration bug where disabling the acceleration checkbox while
a run was active made the worker request CPU. The Java integration must preserve
the selected device while controls are disabled and report both the requested and
resolved device. This is being fixed in JDLL, but the Python API must continue to
expose these values.

## Measured Baseline

Measurements used PyTorch 2.4.0+cu124 on an NVIDIA RTX 4060 Laptop GPU and the
current Noise2Fast implementation. Warmed 20-step optimization measurements were:

| Plane size | CPU time/step | CUDA time/step |
| --- | ---: | ---: |
| 128 x 128 | 0.0126 s | 0.0013 s |
| 256 x 256 | 0.028-0.031 s | 0.0015-0.0032 s |
| 512 x 512 | 0.113-0.123 s | 0.0054-0.0060 s |

For 1024 x 1024, ten optimization steps took 5.60 s on CPU and 0.46 s on
CUDA. At 400 steps this projects to approximately 224 s on CPU and 18 s on CUDA,
before imports, setup, final inference, transfers, and output handling. This is
not an acceptable definition of `quick` for arbitrary input sizes.

## Required Changes

### Bounded native-scale training

- Train Noise2Fast on sampled native-resolution patches when an image exceeds a
  configurable training patch size. Do not resize the source merely to reduce
  computation, because microscopy structures must retain their native scale.
- Keep full-plane training available for images that fit inside the configured
  patch dimensions.
- Make training patch size, batch size, number of sampled patches, sampling
  policy, steps, and seed configurable through the existing configuration
  resolution system.
- Define bounded defaults for `quick`, `balanced`, and `thorough`. Increasing the
  input dimensions must not silently increase the per-step training tensor beyond
  the resolved patch workload.
- Sample across the complete image reproducibly. Avoid repeatedly training on a
  single fixed crop, and document how borders and images smaller than a patch are
  handled.
- Re-evaluate the current 400/1200/2500 step presets using both runtime and output
  quality. Do not preserve these values solely for compatibility if fewer,
  better-sampled steps provide equivalent quality.

### Device execution

- Honor an explicit `cpu`, `cuda`, or `mps` request and retain the existing
  requested-device, resolved-device, and fallback-warning metadata.
- Add integration tests proving that the model, training input, and prediction
  input reside on the resolved CUDA or MPS device when available.
- Avoid synchronizing CUDA/MPS on every optimization step. Keep loss tracking on
  device where practical and transfer scalar values only at progress-report,
  early-stopping evaluation, or final-metadata intervals.
- Batch prediction tiles on accelerators where memory permits instead of doing a
  host-to-device transfer, model call, and device-to-host transfer for every tile.
  Tile batching must be configurable and degrade safely after an out-of-memory
  error.
- Preserve deterministic overlap blending and native output shape.

### Progress, cancellation, and metadata

- Progress callbacks must describe actual optimization steps and inference tiles
  or tile batches. Patch sampling must not make progress totals misleading.
- Cancellation must be checked during sampling, optimization, and tiled
  prediction, with bounded latency.
- Result metadata must include requested/resolved device, training patch size,
  training batch size, sampled-patch count, completed steps, inference tile size,
  inference tile batch size, and phase timings.
- Record cold end-to-end time separately from warmed optimization and inference
  time so environment/import overhead is not attributed to the algorithm.

### Memory and fallback behavior

- Select patch and tile batch defaults that fit common laptop GPUs and CPUs.
- On accelerator out-of-memory, first reduce the tile or patch batch size where
  semantically safe. If automatic CPU fallback is enabled, report it explicitly;
  never silently run on CPU.
- Release method-local tensors and accelerator caches according to the package's
  existing cleanup policy after success, cancellation, or failure.

## Validation

Add a reproducible benchmark matrix for 256, 512, 1024, and 2048 square planes
on CPU and, when available, CUDA and MPS. Cover grayscale, RGB/multichannel, and
Z-stack inputs; all effort presets; cold end-to-end runs; warmed optimization;
warmed inference; and peak host/device memory.

Performance changes must include quality regression tests against the current
full-plane implementation on synthetic noise and representative microscopy
images. Compare at least PSNR/SSIM where a clean reference exists and a stable
no-reference or residual statistic otherwise. A faster preset is acceptable only
when its documented quality remains suitable for that effort level.

Acceptance tests must also verify:

- Larger source images do not enlarge the resolved optimization tensor once
  patch training is active.
- CUDA is materially faster than CPU for sufficiently large inputs after warmup.
- Requested and actual devices agree unless an explicit, reported fallback occurs.
- Output shape, dtype, finite values, and deterministic seeded behavior remain
  unchanged.
- Strength blending remains a Java-side display/output operation and does not
  retrain or rerun Noise2Fast.
