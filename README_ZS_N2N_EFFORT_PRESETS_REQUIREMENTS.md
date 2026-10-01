# ZS-N2N Effort Presets and Batched Views

Status: Python implementation completed and CPU-verified, 2026-09-16.
The original benchmark-only variants are now implemented in the production
API. CUDA verification is blocked by GPU memory exhaustion on the development
machine; MPS is unavailable. The matching Java integration remains separate
in `/home/carlos/git/JDLL`.

## Scope and Precedence

Provide meaningful CPU-friendly ZS-N2N presets without removing any part of the
published loss. This document supersedes the restrictions in
`README_ZS_N2N_FIDELITY_REQUIREMENTS.md` that all efforts use width 48, native
initialization and three separate network forwards. Update conflicting docs
and tests when implementing this contract. The original reference network and
optimization path must remain available for regression testing.

Do not modify Noise2Fast, BM3D/BM4D or StructN2V. Do not add dependencies, Appose
imports, a second training loop, Java-specific backend classes, or automatic
device-dependent preset substitutions. Python owns preset resolution and
execution; Java forwards method/effort and ordinary callback dictionaries.

## 1. Agreed Presets

All requests use `method="zs_n2n"`. The canonical effort ID uses an underscore
in `balanced_high`; its GUI label is "Balanced-high".

| Effort ID | GUI label | Hidden width | Middle groups | Updates | Head initialization | Pair batching | Grayscale parameters |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| `quick` | Quick | 8 | 1 | 500 | near-zero | true | 673 |
| `balanced` | Balanced | 16 | 2 | 1000 | near-zero | true | 1345 |
| `balanced_high` | Balanced-high | 24 | 1 | 2000 | native | true | 5473 |
| `thorough` | Thorough | 48 | 1 | 3000 | native | true | 21313 |

Balanced-high retains the earlier Balanced candidate for testing. It is a
separate preset, not an alias for Balanced. Expose it only for ZS-N2N; reject
it for other methods. Keep `balanced` as the default when effort is omitted.
The optional spelling `balanced-high` may normalize to `balanced_high`, but
resolved configuration and capability reporting must use the canonical ID.

The same presets apply on CPU, CUDA and MPS. Do not silently change width,
groups, update count, initialization, precision or CPU thread settings based
on device, input size, preview mode or estimated duration. Quick describes an
effort budget, not a guaranteed runtime; larger presets do not guarantee
better quality on every image.

## 2. Configuration and Architecture

Extend the existing preset/configuration machinery, with a single table in
`jdll_denoise/presets.py`. Resolve the preset first and then apply explicit
advanced overrides without mutating the input dictionary or preset table.

Existing fields remain:

- `hidden_width`, `steps`, `learning_rate`, `scheduler_step_size`,
  `scheduler_gamma`, `report_interval` and existing advanced stopping controls.

Add these ZS-N2N `method_config` fields:

- `middle_groups`: positive integer dividing `hidden_width`; applies ONLY to
  the second convolution. The input/output image channel count does not need
  to be divisible by this value.
- `output_init`: `"native"` or `"near_zero"`.
- `output_init_std`: finite positive float, default `0.001`, used only with
  near-zero initialization. Expose and report the value rather than hiding it.
- `batch_pairs`: strict boolean, true in all four presets. False keeps the
  original three-forward execution for compatibility and regression tests.

Validate types and ranges consistently with the existing API. In particular,
reject boolean values masquerading as integers, zero/negative group counts,
non-divisible widths, invalid initialization names and nonfinite standard
deviations. Do not silently coerce an invalid architecture or ignore keys.

For C image channels and hidden width W, retain exactly:

```text
Conv2d(C, W, 3, padding=1, bias=True)
LeakyReLU(0.2, inplace=True)
Conv2d(W, W, 3, padding=1, groups=middle_groups, bias=True)
LeakyReLU(0.2, inplace=True)
Conv2d(W, C, 1, bias=True)
```

There is NO hidden residual connection in ANY preset. Do not add depthwise
layers, normalization, pooling, dropout or extra convolutions. Preserve the
existing direct-constructor defaults (48 channels, one group, native head)
for callers/tests that need the reference architecture. The public API must
explicitly construct the network from the resolved preset.

Near-zero means ONLY the last convolution's weights are initialized with
`Normal(mean=0, std=0.001)` and its bias with zero, after normal construction
and under the existing seeded RNG context. Keep the first two convolutions'
native initialization. This is NOT exactly-zero head weights: early layers
must receive gradients on the first update. Native means leaving the last
convolution at PyTorch's usual initialization too.

Parameter counts with biases are:
`W*(9*C+1) + W*(9*(W/groups)+1) + C*(W+1)`.

## 3. Batch the Small Views, Preserve the Loss

Keep the existing two fixed diagonal-average 2x2 filters, stride 2 and NO
padding. For noisy `[B,C,H,W]`, both views have shape
`[B,C,floor(H/2),floor(W/2)]`.

```python
first, second = pair_downsampler(noisy)
paired = torch.cat((first, second), dim=0)
noise_first, noise_second = model(paired).chunk(2, dim=0)
prediction_first = first - noise_first
prediction_second = second - noise_second
full_prediction = noisy - model(noisy)
full_first, full_second = pair_downsampler(full_prediction)

residual = 0.5 * (
    mse(first, prediction_second) + mse(second, prediction_first)
)
consistency = 0.5 * (
    mse(prediction_first, full_first) + mse(prediction_second, full_second)
)
loss = residual + consistency
```

This is TWO model forwards per update: one batch of small views, one full-size
image batch. It is still ONE backward and ONE optimizer step for their sum.
Do not concatenate views along channels, detach predictions, average the view
images before inference, remove consistency, or resize the full-resolution
pass to match the small views. This is execution batching, not gradient
accumulation or a changed loss. No clamping inside the training graph.

With `batch_pairs=false`, retain the original independent calls on first and
second. The loss and gradients must agree within floating-point tolerance;
long training trajectories need not be bitwise identical because batching can
change numerical reduction order. Do not introduce batch-dependent layers.

## 4. Optimization, Normalization and Orchestration

- Adam, LR `0.001`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay zero,
  AMSGrad false. No AdamW, OneCycle, AMP, gradient clipping or new optimizer.
- StepLR with `step_size=1000`, `gamma=0.5`, called AFTER each optimizer step.
  The first 1000 updates use LR 0.001; updates 1001-2000 use 0.0005;
  updates 2001-3000 use 0.00025. Quick has no effective LR decay.
- Use exactly the preset's update count unless cancelled/failed or explicitly
  overridden. Early stopping and best-weight restoration remain disabled by
  default. Final output uses the final weights, not a clean-reference oracle.
- Default ZS-N2N normalization: percentile mode, low=0, high=100, per-channel
  true, clip=true. This is min-max normalization of the supplied noisy image,
  matching the tested configurations. Do not retain an implicit 0.1/99.9 or
  1/99 clipping default for this method. Preserve explicit normalization
  overrides and leave other methods' defaults unchanged. Retain finite-input
  and constant-image handling. Do not claim this wrapper is the paper's exact
  preprocessing.
- Final denoising is `clamp(noisy - model(noisy), 0, 1)`, then inverse
  normalization. Preserve output axes, shape and physical intensity scale.
- Preserve independent biological-channel and native-color modes. In native
  color mode the network is C->C, with grouping only in the hidden layer.
- Preserve plane-wise 2D behavior for stacks/time points; these presets do
  not introduce true 3D ZS-N2N. Preserve full-plane training and optional tiled
  final inference. Do not train independent models per output tile or silently
  downscale the input for Quick.
- `preview()` must resolve the same effort and parameters as `denoise()`;
  keep its existing documented region-selection behavior, not a shorter hidden
  optimization budget. Strength blending remains Java's separate operation.
- Keep cancellation checks between updates/jobs/tiles, explicit device
  fallback policy, and cleanup on success, cancellation and exceptions.

## 5. Java/Python Contract

The existing public API stays unchanged:

```python
from jdll_denoise import denoise, preview, resolve_config, capabilities
result = denoise(image, config, callbacks=on_event)
# Preview uses preview(image, config, callbacks=on_event).
# result['denoised'] and result['metadata'] retain their current meanings.
```

Example request from Java:

```json
{
  "method": "zs_n2n",
  "effort": "balanced_high",
  "device": "cpu",
  "dimensions": "auto",
  "axes": "yx",
  "seed": 5489,
  "normalization": {
    "mode": "percentile", "low": 0, "high": 100,
    "per_channel": true, "clip": true
  },
  "tiling": {
    "enabled": "auto", "tile_size": "auto", "overlap": "auto",
    "blend": "gaussian"
  },
  "method_config": {},
  "allow_cpu_fallback": false
}
```

Java intentionally does NOT send copies of the architecture/budget table.
Those must be recovered from method/effort by the production resolver.

Once this entire contract is implemented and tested, extend the existing
capabilities entry (preserving its other fields) with:

```python
capabilities()['methods']['zs_n2n'] == {
    # existing availability/devices/dimensions/channel_modes fields ...
    'effort_presets_version': 1,
    'efforts': ['quick', 'balanced', 'balanced_high', 'thorough'],
}
```

`effort_presets_version` is an integer contract version, not the package version.
Advertise version 1 only when ALL FOUR presets, batching, initialization and
normalization rules work through the public APIs. Missing/older capability
data makes Java reject ZS-N2N with an update-required message instead of
silently executing the former width-48 Quick/Balanced presets. Java also
checks that the selected effort is advertised. Other methods remain usable.

The Java GUI shows the fourth radio option only for ZS-N2N. Switching from
Balanced-high to another method resets effort to Balanced. Preview caching
must be invalidated when effort changes; progress iteration maxima must use
the actual resolved update budget.

## 6. Metadata, Callbacks and Fidelity

Retain the existing plain-dictionary callbacks and result schema. No Appose
dependency. Continue reporting progress, all three loss components, device,
iteration count and elapsed time. Reporting frequency must not change the
number of optimizer steps or loss evaluations.

Result metadata/resolved configuration must accurately include effort,
hidden_width, middle_groups, output_init, applicable output_init_std,
batch_pairs, completed/selected updates, parameter count, optimizer settings,
schedule, normalization and actual device. Include the preset contract version
so logs/results can distinguish these runs from the former presets. An initial
preparing/optimization event should describe the resolved preset settings for
the Java log; never require Java to reconstruct them.

Quick/Balanced/Balanced-high are explicit JDLL performance variants, not the
unmodified published architecture. Thorough keeps the reference architecture,
native initialization, budget and objective, with equivalent batched execution
and the documented normalization wrapper. Update `_fidelity()` so the new
effort does not cause a KeyError and reports grouping/initialization/overrides
honestly. Do not label all width overrides as retaining the canonical network.
Keep a reference path (`48`, groups `1`, native, `3000`, batch_pairs=false).

## 7. Acceptance Tests

1. Resolver tests for all four method/effort pairs, omission default, explicit
   override precedence, copied dictionaries, invalid groups/types and rejection
   of Balanced-high for other methods. Test capabilities contract version/IDs.
2. Network tests: exact layers, group count, no hidden skip, grayscale parameter
   counts above, native-color channels, near-zero weights/zero head bias,
   unchanged stem/middle initialization at a fixed seed, finite/nonzero
   first-update gradients for all layers.
3. Compare batched/unbatched loss components and input/parameter gradients in
   float64 and float32 for even/odd H/W, B>1, C>1, groups 1 and 2. Count two
   network forwards versus three, and one optimizer step. Keep the independent
   reference equations in tests so a shared implementation bug cannot pass.
4. Scheduler boundary tests; final-weight selection; no training clamp; public
   preview/run preset parity. Test min-max default, explicit percentile override,
   normalization inversion and constant inputs. Other methods must not regress.
5. Lightweight public-API tests for each preset with explicit short step
   overrides, checking constructed architecture, actual steps, metadata and
   callbacks. Separately assert the unoverridden preset budgets. Exercise CPU;
   CUDA/MPS when available, without hardcoding device-specific timing limits.
6. Cancellation, failure cleanup, odd image sizes, channels, stack processing,
   and existing tiling contracts must retain their tests. No new dependencies.
7. Use the benchmark references below as numerical/performance evidence, not
   universal accuracy thresholds or timing assertions. Do not retrain every
   full preset in routine CI or write a report for each experiment.

## Existing Benchmark References

`benchmarks/zs_n2n_initialization.py` implements the tested head initialization
and hidden variants; use ONLY the near-zero/no-hidden-residual variant here.
`benchmarks/zs_n2n_batched.py` contains batched loss and equivalence checks.
Benchmark monkey-patching must not become the production architecture.

Raw data on the development machine:

- `/home/carlos/hack_git/jdll-denoise-benchmarks/zs-n2n-width8-near-zero-batched-500-20260916-cpu`
- `/home/carlos/hack_git/jdll-denoise-benchmarks/zs-n2n-init-p0-p100-20260916-cpu`
- `/home/carlos/hack_git/jdll-denoise-benchmarks/zs-n2n-width-20260915-cuda`

These used one 383x512 grayscale acquisition with added Gaussian noise, seed
5489 for training and min-max normalization. The acquisition is not verified
noise-free; reference MSE measures removal of the added noise, not universal
biological image quality. Quick with batching gave MSE about 0.000766885.
Timing varied, so do not advertise a universal speedup or quality ranking.

## Implementation and Verification Record (2026-09-16)

Implemented all four production presets, middle-only grouping, seeded
near-zero output initialization, batch-pair and reference execution paths,
min-max defaults and explicit overrides, version-1 capabilities/config/results,
actual-settings callbacks and fidelity reporting. The single existing training
loop retains the complete loss, Adam and StepLR; advanced stopping/restoration
now evaluates independently of callback frequency. No production CPU thread
or precision settings were changed.

Verification used `/home/carlos/micromamba/envs/ofm/bin/python`:

```bash
CUDA_VISIBLE_DEVICES='' /home/carlos/micromamba/envs/ofm/bin/python -m pytest -q --tb=short
/home/carlos/micromamba/envs/ofm/bin/python -m ruff check jdll_denoise tests
/home/carlos/micromamba/envs/ofm/bin/python -m mypy jdll_denoise
```

- Full CPU suite: **301 passed, 15 skipped** in 20.56 seconds. The skips were
  eight CUDA and seven MPS cases (including the existing Noise2Fast cases).
- CUDA-visible testing initially failed with CUDA out-of-memory, including
  PyTorch's accelerator graph-capture check during CPU Adam. `nvidia-smi`
  reported only 62 MiB free on the 8 GiB GPU. CUDA was hidden only in the CPU
  test process; no workloads were stopped or device defaults changed. CUDA
  execution checks remain present for rerunning when memory is available.
- MPS is not available on this machine.
- Lint passed; mypy passed for all 19 source files; formatting checks passed
  for all eight changed Python files.
- Tests cover independent-reference losses, input/parameter gradients in
  float32/float64 for even/odd shapes and multiple batches/channels/groups;
  first-update gradients; exact layer/parameter/init contracts; two-versus-three
  forwards and one backward/update; scheduler boundaries; retained reference
  optimization; public denoise/preview settings and metadata; normalization;
  independent/native-color stacks; tiled final inference without retraining;
  cancellation/failure cleanup, callback isolation and explicit CPU fallback.
- A pre-change resolver snapshot confirms Noise2Fast, BM3D/BM4D and StructN2V
  configurations are unchanged. Their existing tests also pass. Only the old
  test assertion that ZS-N2N presets remained unchanged was superseded.
- No full-preset benchmark matrix was run. Historical experiment scripts/data
  are preserved as evidence, not used as production monkey-patches.

Changed files:

- Production: `jdll_denoise/presets.py`, `jdll_denoise/config.py`,
  `jdll_denoise/capabilities.py`, `jdll_denoise/api.py`,
  `jdll_denoise/methods/zs_n2n.py`.
- Tests: `tests/test_zs_n2n.py`, `tests/test_zs_n2n_effort.py`,
  `tests/test_noise2fast_effort.py`,
  `tests/fixtures/other_methods_before_zs_effort.json`.
- Documentation: `README.md`, `docs/API.md`, `docs/JDLL_INTEGRATION.md`,
  `docs/IMPLEMENTATION_STATUS.md`, `CHANGELOG.md`,
  `README_IMPLEMENTATION_REQUIREMENTS.md`,
  `README_ZS_N2N_FIDELITY_REQUIREMENTS.md`, and this file.

Java follow-up: enforce the version/effort capability gate, expose
Balanced-high only for ZS-N2N (reset to Balanced on method change), invalidate
preview caches when effort changes, use actual progress budgets and min-max
normalization, and forward method/effort without copying the Python table.
No Java files, dependencies, commits or pushes were added by this implementation.
