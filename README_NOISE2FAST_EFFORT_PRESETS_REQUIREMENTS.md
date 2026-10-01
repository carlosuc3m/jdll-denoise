# Noise2Fast Effort Presets: Implementation Requirements

Status: agreed for implementation, 2026-09-08.
Python implementation completed and verified, 2026-09-09. Java follow-up
remains separate (section 6).

## 1. Purpose and Scope

Implement meaningful Noise2Fast effort presets in the production Python API,
using the network-width and validation-patience combinations tested with JDLL.
Before this implementation, the settings existed only in a standalone
benchmark; production presets all resolved to 64 features and patience 100.

This document supersedes earlier requirements that every Noise2Fast effort
must use 64 features, patience 100, or be numerically identical to that run.
In particular, update the conflicting Noise2Fast section of
`README_IMPLEMENTATION_REQUIREMENTS.md` and the relevant fidelity wording in
`README_NOISE2FAST_PERFORMANCE_REQUIREMENTS.md` when implementing this change.

This is a narrow exception for width and patience. It does NOT restore the
historical patch-sampled, tiled, low-fixed-step approximation described in the
older performance proposal. Do not change BM3D/BM4D, ZS-N2N or StructN2V.

Do not add dependencies, Appose imports, Java code, a new training framework,
or a separate implementation of the Noise2Fast training loop. JDLL is
responsible for forwarding ordinary Python callback dictionaries to Appose.

## 2. Agreed Presets

| Public effort identifier | GUI name | Hidden feature channels | Patience, steps |
| --- | --- | ---: | ---: |
| `quick` | Quick | 16 | 50 |
| `balanced` | Balanced | 24 | 100 |
| `thorough` | Thorough | 64 | 100 |

- Keep `balanced` as the default effort when omitted.
- Use exactly the same preset on CPU, CUDA and MPS. Device selection must not
  silently change architecture, patience, or output quality settings.
- 12 features was discussed but NOT selected or benchmarked. Do not use it for
  Quick. 32 features was tested but NOT selected for Balanced.
- Quick means lower computational effort, not a guaranteed time limit.
- Thorough means greater capacity and computational effort, not a guarantee of
  lower error on every image.

Put the mappings in `jdll_denoise/presets.py`, the single source of truth.
Do not duplicate the preset table in Java or in multiple Python modules.

## 3. Configuration and Model Construction

### Public configuration

Add `hidden_width` and `patience` to the allowed Noise2Fast `method_config`
keys in `jdll_denoise/config.py`. Use `hidden_width`, consistent with ZS-N2N.

Resolve the selected preset first, then apply explicitly supplied advanced
overrides. Both values must be positive integers; reject booleans, floats,
strings, zero and negative values with a useful `ConfigurationError`. Advanced
widths are not restricted to the three preset widths.

Do not subsequently overwrite the resolved patience with a hardcoded 100.
Retain existing `steps` and `report_interval` support and validation:

- `steps=0`, the default for every effort: stop using validation patience.
- Positive `steps`: an explicit advanced/debug upper bound, not a preset's
  normal stopping rule. Report clearly when this cap causes termination.
- Reporting frequency must not change training or validation frequency.

Example ordinary request, unchanged from what Java already sends:

```json
{
  "method": "noise2fast",
  "effort": "quick",
  "device": "cpu",
  "axes": "yx",
  "seed": 5489
}
```

Its resolved `method_config` must contain at least:

```json
{
  "hidden_width": 16,
  "patience": 50,
  "output_window": 100,
  "steps": 0
}
```

Include the existing resolved reporting interval as well. An advanced request
such as `"method_config": {"hidden_width": 32, "patience": 75}` overrides those
two values regardless of the effort label; metadata must reflect the actual
resolved values. Do not mutate the caller's dictionary or the preset table.

### Network

Parameterize the existing `Noise2FastBlock` and `Noise2FastNet` in
`jdll_denoise/methods/noise2fast.py`. Construct the network from the resolved
`hidden_width` in the actual production execution path.

Keep a default constructor width of 64 for existing direct Python callers of
`Noise2FastNet()`. This constructor default is distinct from the public API's
Balanced preset, which must explicitly construct a 24-feature network.

For a width F, retain exactly:

1. One 3x3 convolution from 1 input channel to F features, followed by ReLU.
2. Seven further 3x3 convolutions from F to F, each followed by ReLU.
3. One 1x1 convolution from F to 1, followed by sigmoid.

Keep the existing four two-convolution blocks, padding and stride. There must
be no pooling, U-Net, skip connection, batch normalization, or hidden resizing.
All widths retain a 17x17 theoretical receptive field on their input grid.

Expected parameter counts, including biases: `63*F*F + 18*F + 1`.

| Width | Parameters |
| ---: | ---: |
| 16 | 16,417 |
| 24 | 36,721 |
| 64 | 259,201 |

## 4. Preserve the Training and Reconstruction Algorithm

Apart from the resolved width and patience, retain the current implementation:

- A fresh network for each independent image/channel/plane, with the existing
  deterministic seed handling. Stacks remain plane-wise 2D, not true 3D.
- The four directed checkerboard pairs: two packing directions, each trained
  in both input/target directions. Randomly choose one pair per update.
- Batch size 1, Adam, learning rate 0.001, and binary cross-entropy on the
  normalized floating-point intensities. Keep the existing Adam defaults.
- One validation forward pass on the original full-resolution noisy plane
  after EVERY optimizer update, independent of callback reporting frequency.
- The same negative full-image MSE against the noisy input as the stopping
  score. It is not an independent clean-ground-truth quality metric.
- The existing odd-dimension handling: remove the last odd row/column only
  when building training pairs, not from the final image.
- The current output shape, float output contract, normalization/inversion,
  channel/axis handling and constant-image behavior.
- Cancellation checks each optimization step and between planes, cleanup on
  success/cancellation/failure, and the existing explicit fallback policy.

Do not introduce time limits, patch training, tiled reconstruction, image
downscaling, a different loss, a learning-rate schedule or altered validation
frequency as part of this work. Denoising-strength blending remains outside
this method; effort is not strength.

### Exact patience semantics

Let P be the resolved patience. On a strictly improved stopping score, update
the best score and reset the no-improvement counter to zero. Otherwise increment
the counter. Stop when it reaches P, not P+1. Preserve the current comparison
rule; do not introduce a minimum improvement threshold.

For example, if the last improvement is at step 339, and no earlier sequence
has already caused stopping:

- P=50 stops at step 389.
- P=100 stops at step 439.

P is NOT the total training budget. Improvements can keep resetting it.
Do not change these semantics to match the off-by-one condition in the authors'
standalone script; the tested JDLL loop already uses the above convention.

### Output averaging is independent of patience

Keep `output_window=100` for ALL presets, including Quick with patience 50.
Average the most recent `min(completed_steps, 100)` full-image predictions.
Do not average weights, use only the last prediction, return only the best
checkpoint prediction, or shorten the averaging window to match patience.
If fewer than 100 predictions exist, average those available without zero
padding the buffer or dividing their sum by 100.

In particular, changing patience changes the end of the averaging window;
the result need not equal the output of the longer run.

## 5. Callbacks and Metadata

Use the existing public callback and return-value contracts. Callbacks remain
plain Python functions receiving JSON-serializable dictionaries; no Appose
dependency is allowed.

### Resolved settings

The existing `resolved_config` event and final
`metadata["resolved_config"]` must include the actual `hidden_width`, `patience`,
`output_window`, step cap, requested/resolved device, and existing other fields.

Provide a readable initialization message for Noise2Fast, through the existing
resolved-config event or one status event, for example:

```text
Noise2Fast Balanced: 24 features, patience 100 steps, averaging 100 predictions, learning rate 0.001, device cpu.
```

Use resolved values in this message, including advanced overrides. Java will
display the information; do not require it to resolve the preset itself.

### Optimization progress

- Keep `current` equal to the number of completed optimizer updates.
- For patience-only runs (`steps=0`), use `maximum=0` to mean the total is
  unknown. Do not present `patience` or `step + remaining_patience` as a known
  total iteration count or a reliable completion percentage.
- With an explicit positive step cap, `maximum` may be that cap; the run can
  still stop sooner by patience.
- Include the configured patience in progress metrics as `validation_patience`.
- Preserve `epochs_without_validation_improvement` for compatibility, but
  document that this existing field counts optimizer STEPS, not dataset epochs.
  Use step terminology in user-facing messages.
- Retain existing loss, best-step, elapsed-time and stopping-score metrics.
  State the counter in the message, for example:
  `Noise2Fast step 250; no improvement for 12/50 steps.`
- Emit the final progress state even when stopping between reporting intervals.

Preserve the existing completion, failure, cancellation, channel/plane and
callback-failure-isolation behavior. Unknown optimization duration must not
prevent a completed operation from emitting its normal completion event.

### Final per-plane metadata

Keep existing keys and fill them from actual execution. At minimum report:

- `architecture`, with the actual feature count rather than hardcoded 64.
- `hidden_width`, `validation_patience`, and `output_average_window`.
- `output_average_count`, completed steps, best step, and final stale counter.
- `stopped_by`, distinguishing patience from the explicit debug step cap.
- Actual devices, fallback information and existing phase timings.

Remove hardcoded "100-epoch" messages and misleading fidelity labels. Quick
and Balanced are compact-width Noise2Fast variants preserving the training
principle, not the exact released 64-feature architecture. The 64-feature
network uses that released architecture; do not claim bitwise equivalence to
every detail of the original standalone reference program.

## 6. Normalization and Java Ownership

Preserve Noise2Fast's Python normalization defaults: per-channel 0th to 100th
percentiles, i.e. min-max scaling. Do not silently change those defaults to
0.1/99.9 or discard intensity tails. Continue honoring valid explicit
normalization overrides according to the existing API.

A separate Java issue has been identified: JDLL currently sends explicit
0.1/99.9 percentile clipping for every denoising method, overriding these
Noise2Fast defaults. The Python agent must NOT compensate by ignoring an
explicit caller configuration. This needs a Java-side correction.

Java follow-up, outside this repository's implementation scope:

- In `/home/carlos/git/JDLL/src/main/java/io/bioimage/modelrunner/model/special/denoise/DenoisingConfig.java`,
  let Noise2Fast use its backend normalization defaults.
- In `DenoisingPluginUI.java`, show the resolved width/patience and represent
  unknown optimization totals honestly. Continue forwarding effort and device.
- Keep the existing effort radio buttons and the Balanced default. Do not
  duplicate the Python preset mapping in the GUI.

The Python implementation must work through both `denoise()` and `preview()`.
Preserve the existing preview-region behavior; a crop preview need not produce
exactly the same result as training a new model on the entire image. Within
either path, use the same resolved effort parameters.

## 7. Tests and Acceptance Criteria

Update tests whose assumptions conflict with the new presets rather than
removing coverage. In particular, Quick must no longer be expected to use the
64-feature network or patience 100.

Required automated coverage:

1. All three preset mappings, omitted-effort Balanced behavior, independent
   configuration dictionaries, advanced override precedence and invalid values.
2. Layer structure, parameter count, output dimensions and output range for
   widths 16, 24 and 64; default direct constructor compatibility at width 64.
3. Regression of the full-width computation with explicit width 64/patience
   100, against the previous loop on the same seeded small CPU input. Use a
   short explicit cap for a fast numerical regression test; do not compare
   different devices bitwise.
4. Exact patience boundary/reset behavior with controlled validation scores,
   including a plateau and a later improvement that resets the counter.
5. Output averaging of 100 predictions when available and the correct smaller
   count when fewer exist, independent of whether patience is 50 or 100.
6. Public `denoise()` and `preview()` actually instantiate each resolved width
   and honor the selected/overridden patience, without the benchmark's
   constructor substitution. Short debug caps are acceptable for smoke tests.
7. Dynamic callback/metadata values, `maximum=0` for uncapped optimization,
   explicit-cap reporting, final progress emission and JSON serialization.
8. Existing checkerboard/odd-shape tests, finite outputs, constant images,
   independent channels/planes, cancellation and callback error isolation.
9. Available CUDA/MPS execution using the resolved model/input device, with
   hardware-dependent tests skipped explicitly when unavailable. Never require
   a GPU merely to pass the CPU suite or silently change presets on fallback.
10. Noise2Fast default min-max normalization and explicit override behavior;
    regression coverage ensuring other denoising methods are unchanged.

Retain deterministic seeded CPU regression coverage under the same execution
settings. No test should assume a particular width is always faster or more
accurate: hardware timing and training outcomes can vary.

Once the implementation is tested, update `README.md`, `docs/API.md`,
`docs/JDLL_INTEGRATION.md`, `docs/IMPLEMENTATION_STATUS.md` and the relevant
requirements/changelog entries where they describe the old behavior. Explain
the meaning of width, patience, unknown duration and the fixed averaging window.
Do not market the compact variants as identical to the full architecture.

## 8. Benchmark Evidence and Reproduction

These exploratory results motivated the presets. They are not universal quality
or runtime guarantees and are not numerical assertions for CI.

| Width | Patience | Completed steps | Total seconds | Mean seconds/step | Final reference MSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 100 | 409 | 452.05 | 1.10519 | 0.000343544 |
| 32 | 100 | 454 | 139.27 | 0.30671 | 0.000394405 |
| 24 | 100 | 439 | 92.14 | 0.20978 | 0.000357399 |
| 24 | 50 | 389 | 89.09 | 0.22891 | 0.000365788 |
| 16 | 50 | 423 | 46.91 | 0.11081 | 0.000420522 |

Setup:

- CPU: Intel Core i7-13700H, four PyTorch compute threads, one interop thread.
- Environment Python: `/home/carlos/.local/share/appose/biapy/.pixi/envs/linux-x86-64-cuda/bin/python`.
- PyTorch 2.4.0+cu121; computation explicitly on CPU.
- Source: `/home/carlos/Pictures/example/train/001_img.png`, green channel
  (channel index 1), 383x512 pixels, no image resizing.
- Source channel divided by 255; add un-clipped Gaussian noise with sigma
  25/255 and NumPy `default_rng` seed 20260908.
- Model/pair-selection seed 5489; normalize the noisy image by its min/max
  for training and invert that normalization before reference MSE scoring.
- Three warmup steps on a disposable model before the measured fresh run.
- Average the last 100 available predictions. Ground truth is never used for
  stopping or model selection.

The reference is the original acquisition channel, not independently verified
noise-free ground truth. Its MSE against the synthetically noisy image is
0.009583718. Only one image, one noise realization and one initialization seed
were used, and other user workloads were active. Timing varied between runs.

Reproduction script: `benchmarks/noise2fast_width.py`. The historical version
substituted the constructor and called the existing private training loop.
The implemented runner now calls public `denoise()` with `--efforts` (all three
by default), or advanced `--widths`/`--patience` overrides, without substitution.
It records actual resolved configuration and per-plane metadata. Historical
results remain preserved; see `benchmarks/REPORT.md` for invocation examples
and the distinction between historical and public-API timing scopes.

Recorded inputs, outputs, per-step CSVs, configurations and reports are under
`/home/carlos/hack_git/jdll-denoise-benchmarks/` in these directories:

- `noise2fast-width-20260908-cpu` (64 and 32, patience 100).
- `noise2fast-width24-20260908-cpu`.
- `noise2fast-width24-patience50-20260908-cpu`.
- `noise2fast-width16-patience50-20260908-cpu`.

Compare representative images and seeds as further validation, not to silently
change the agreed preset table. Do not alter machine-wide clocks, stop unrelated
processes, or run large benchmark matrices without coordinating with the user.

## 9. Completion Checklist

- [x] Production Quick, Balanced and Thorough resolve to 16/50, 24/100 and 64/100.
- [x] The public API uses those settings, not just the benchmark script.
- [x] The remaining training and 100-prediction averaging behavior is preserved.
- [x] Advanced width/patience overrides are validated, honored and reported.
- [x] Callbacks and metadata describe actual settings, steps and stopping reason.
- [x] Conflicting requirements, documentation and test expectations are updated.
- [x] CPU tests pass; unavailable accelerator tests are explicitly identified.
- [x] Java follow-up is reported separately; no Java or unrelated method changes.

Verification on 2026-09-09, using `/home/carlos/micromamba/envs/ofm/bin/python`:

- `-m pytest -q`: 158 passed, 3 skipped (MPS unavailable). CUDA execution checks
  passed for all three efforts. Includes short CPU public-API benchmark CLI
  tests for all presets and width 32/patience 75; no large matrix was run.
- `-m ruff check jdll_denoise tests benchmarks/noise2fast_width.py`: passed.
- `-m mypy jdll_denoise`: passed for all 19 source files.
- The explicit 64-feature/patience-100 run matches a frozen pre-change seeded
  CPU baseline within the test's numerical tolerance. Controlled scores verify
  the 389/439-step stopping examples, strict resets and output averaging.

Changed files:

- Production: `jdll_denoise/presets.py`, `jdll_denoise/config.py`,
  `jdll_denoise/methods/noise2fast.py`.
- Tests: `tests/test_noise2fast.py`, `tests/test_noise2fast_effort.py`,
  `tests/fixtures/noise2fast_64_before_effort.json`,
  `tests/fixtures/other_method_presets_before_effort.json`.
- Benchmarks: `benchmarks/noise2fast_width.py`, `benchmarks/REPORT.md`.
- Documentation: `README.md`, `docs/API.md`, `docs/JDLL_INTEGRATION.md`,
  `docs/IMPLEMENTATION_STATUS.md`, `CHANGELOG.md`,
  `README_IMPLEMENTATION_REQUIREMENTS.md`,
  `README_NOISE2FAST_PERFORMANCE_REQUIREMENTS.md`, and this requirements file.

Do not automatically commit or push. Report the changed files, verification
results and any remaining Java integration work to the user.
