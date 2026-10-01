# Benchmark report

> The bounded-patch results below are retained as historical measurements only.
> That implementation was removed because it did not reproduce Noise2Fast's
> paper. Current benchmarks must use the four full-plane checkerboard pairs,
> full-image validation after every update, and rolling-output reconstruction.

## Width/patience effort evidence (2026-09-08)

The agreed production presets are Quick 16/50, Balanced 24/100 and Thorough
64/100 (hidden width/patience in optimizer steps). These exploratory results
motivated that choice; they are historical observations, not CI assertions or
universal quality/runtime guarantees:

| Width | Patience | Steps | Total seconds | Mean seconds/step | Final reference MSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 100 | 409 | 452.05 | 1.10519 | 0.000343544 |
| 32 | 100 | 454 | 139.27 | 0.30671 | 0.000394405 |
| 24 | 100 | 439 | 92.14 | 0.20978 | 0.000357399 |
| 24 | 50 | 389 | 89.09 | 0.22891 | 0.000365788 |
| 16 | 50 | 423 | 46.91 | 0.11081 | 0.000420522 |

Intel Core i7-13700H CPU, four compute threads/one interop thread, PyTorch
2.4.0+cu121. One 383×512 acquisition channel, un-clipped added Gaussian noise
at sigma 25/255, noise seed 20260908 and model/pair seed 5489; three disposable
warmup steps. Training used min-max normalization and scoring used restored
intensities. The original acquisition is not verified noise-free ground truth.
Other user workloads were active. Full setup and preserved external artifact
locations are in the
[effort requirements](../README_NOISE2FAST_EFFORT_PRESETS_REQUIREMENTS.md#8-benchmark-evidence-and-reproduction).

The current `noise2fast_width.py` calls the production `denoise()` API for both
preset and advanced-width runs; it no longer replaces the constructor or calls
a private training loop. It records resolved configuration, actual per-plane
metadata, per-step CSVs, TIFF outputs and reference metrics. Public-API timings
include configuration, normalization and restoration, so their scope is not
identical to the historical private-loop timings above.

Run a short integration smoke check (not a quality benchmark) with:

```bash
/home/carlos/micromamba/envs/ofm/bin/python benchmarks/noise2fast_width.py \
  --image /home/carlos/Pictures/example/train/001_img.png \
  --efforts quick balanced thorough --device cpu --threads 4 \
  --max-steps 3 --warmup-steps 1 --output /tmp/noise2fast-presets-smoke
```

For an advanced override, use `--widths 32 --patience 75` instead of
`--efforts ...`. No selection defaults to all three production efforts.
Omit `--max-steps` (or set it to 0) only when intentionally running the complete
patience-based experiment; it has no known total duration. Do not launch large
matrices without coordination. The averaging window remains 100 for every run.

## Original full-plane baseline (2026-07-15)

Environment: Python 3.10, PyTorch 2.5.1+cu121, Linux; CUDA was available.
The input was a seeded 256×256 `float32` plane. A 20-step Noise2Fast Quick
configuration with 32 features measured:

| Device | 20 steps | seconds/step | projected Quick 400-step optimization |
| --- | ---: | ---: | ---: |
| CPU | 1.508 s | 0.0754 s | about 30.2 s |
| CUDA (warm) | 0.098 s | 0.0049 s | about 2.0 s |

Those projections motivated replacing area-growing full-plane training with a
bounded native-scale workload.

## Bounded workload (2026-07-16)

The historical bounded implementation used these now-removed presets:

| Effort | Steps | Patch | Batch | Coordinate pool |
| --- | ---: | ---: | ---: | ---: |
| Quick | 300 | 256×256 | 1 | 64 |
| Balanced | 800 | 256×256 | 2 | 128 |
| Thorough | 1,600 | 384×384 | 2 | 256 |

Increasing a source beyond the resolved patch does not increase its optimization
tensor. A seeded 64×64 synthetic phantom regression using 100 steps and a small
test network measured bounded 32×32-patch PSNR 22.71 dB versus 23.41 dB for the
former full-plane workload; SSIM was 0.610 versus 0.592. The noisy input PSNR was
21.46 dB. This supports sampled-patch quality at the tested scale while keeping
a tolerant regression gate. It does not replace representative microscopy
validation.

On the current PyTorch 2.5.1+cu121 environment, a 256×256 Quick workload with
20 steps measured:

| Device | Cold algorithm call | Warm algorithm call | Warm optimization | Warm inference |
| --- | ---: | ---: | ---: | ---: |
| CPU | 3.758 s | 0.716 s | 0.614 s | 0.0225 s |
| CUDA | 3.283 s | 0.095 s | 0.0341 s | 0.0007 s |

Cold method setup was about 3.0 s on both devices and is reported separately
from warmed phases. At this workload CUDA optimization was about 18× faster
than CPU after warmup. These measurements validate synchronization removal and
device residence; the full 256/512/1024/2048 matrix remains reproducible rather
than claimed complete on one machine.

The general runner now exercises the current algorithm, not the removed
historical implementation. Its explicit smoke-cap invocation is:

```bash
python benchmarks/benchmark.py --method noise2fast --effort quick --device cpu --size 256 --steps 20
```

Generate the required size/effort/device matrix (optionally using a bounded
smoke-step override) with:

```bash
python benchmarks/noise2fast_matrix.py --steps 20 --output matrix.json
```

Use `--stack-depth 32` for the required 32-plane volume rows.

Before a production JDLL bundle, archive all three efforts at 256, 512, 1024,
and 2048; three-channel inputs; representative Z stacks including
32×256×256; each available device; peak memory; and cancellation latency. Store
hardware identifiers with those results. Presets must only be changed together
with an updated report and monotonicity tests.
