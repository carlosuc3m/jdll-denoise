# jdll-denoise Implementation Requirements

Noise2Fast width/patience requirements below incorporate the agreed
[effort-preset exception](README_NOISE2FAST_EFFORT_PRESETS_REQUIREMENTS.md).
That document supersedes earlier fixed-64-feature/fixed-100-patience wording;
it does not authorize changes to other methods or other Noise2Fast behavior.
ZS-N2N now follows the separate
[version-1 effort/batching contract](README_ZS_N2N_EFFORT_PRESETS_REQUIREMENTS.md),
which supersedes fixed-width/native-only/three-forward restrictions in the
older fidelity document, but preserves its complete loss and reference path.

## 1. Purpose

`jdll-denoise` is the Python backend for a JDLL microscopy denoising plugin.
It must provide conservative classical denoising and zero-shot/self-supervised
neural denoising without clean target images or a pre-trained model.

The project must be independently installable, testable, and releasable.
JDLL will call it from an Appose-managed Python process, but this package must
not import or depend on Appose.

Project location:

```text
/home/carlos/hack_git/jdll-denoise
```

Package identity:

```text
distribution: jdll-denoise
module:       jdll_denoise
Python:       >=3.10
license:      Apache-2.0 for original project code
```

The implementation must be maintainable library code, not copied research
scripts wrapped in subprocess calls.

## 2. Product Decisions

The Java GUI exposes exactly these methods in this order:

1. `Conservative - BM3D/BM4D`
2. `Adaptive - ZS-N2N`
3. `Fast - Noise2Fast`
4. `Correlated - StructN2V`

The default is `Fast - Noise2Fast`.

Every method must work on CPU. GPU acceleration is available only when the
method and runtime support it. Classical BM3D/BM4D is CPU-only unless a real,
tested accelerated implementation is introduced later.

Effort has exactly three public values:

```text
quick
balanced
thorough
```

Denoising strength is not an algorithm parameter. The backend always returns
the fully denoised result. JDLL blends it with the original:

```python
displayed = strength * denoised + (1.0 - strength) * original
```

The strength range is `[0.0, 1.0]` and the default is `1.0`. Full strength is
the production default because it makes the selected method and configuration
the reproducible source of the output. An arbitrary lower default would conceal
part of the algorithm's effect.

## 3. Scope

Required in v1:

- Array-based 2D denoising.
- Multichannel microscopy images with preserved channel structure.
- Stack and volume processing according to the support matrix below.
- Time-series processing as independent C/Z volumes or C/Y/X frames.
- CPU execution for all methods.
- CUDA and MPS for neural methods where supported by PyTorch.
- Quick, Balanced, and Thorough presets.
- Structured callbacks for progress, status, warnings, and resolved settings.
- Cooperative cancellation at fine-grained safe points.
- Deterministic behavior when a seed is supplied.
- Cleanup of Python and device memory after success, cancellation, or failure.
- Same-shape output in the same physical intensity scale as the input.

Not required in v1:

- Supervised denoising.
- Persistent user-trained denoising models.
- A Java or Swing dependency.
- Appose-specific callbacks.
- File-system output naming or host metadata writing; JDLL owns those concerns.
- Custom StructN2V mask geometry in the GUI.
- True-3D adaptations of ZS-N2N or Noise2Fast that are not validated by their
  original methods.

## 4. Licensing Constraint

The official Tampere University `bm3d` and `bm4d` Python packages are limited to
non-commercial scientific/educational use. The implementation must:

- not copy or vendor their binaries or source into this Apache-2.0 project;
- isolate them behind an optional adapter;
- report their availability through `capabilities()`;
- retain their license notices when installed;
- document the restriction prominently;
- fail clearly when the adapter is selected but unavailable;
- allow JDLL packaging to exclude them if redistribution is not approved.

Before a public bundled release, maintainers must choose one of:

1. Obtain permission compatible with JDLL distribution.
2. Require users to install/accept the restricted package separately.
3. Select or implement a legally compatible BM3D/BM4D backend after review.

Do not silently replace BM3D with another algorithm while retaining the BM3D
name.

Primary references:

- BM3D/BM4D software: https://webpages.tuni.fi/foi/GCF-BM3D/index.html
- Noise2Fast: https://github.com/jason-lequyer/Noise2Fast
- ZS-N2N paper: https://openaccess.thecvf.com/content/CVPR2023/html/Mansour_Zero-Shot_Noise2Noise_Efficient_Image_Denoising_Without_Any_Data_CVPR_2023_paper.html
- Noise2Void/StructN2V implementation: https://github.com/juglab/n2v

## 5. Project Structure

Use this organization unless a clearly better equivalent is justified:

```text
jdll-denoise/
├── pyproject.toml
├── README.md
├── LICENSE
├── jdll_denoise/
│   ├── __init__.py
│   ├── api.py
│   ├── capabilities.py
│   ├── callbacks.py
│   ├── cancellation.py
│   ├── config.py
│   ├── devices.py
│   ├── errors.py
│   ├── io.py
│   ├── normalization.py
│   ├── presets.py
│   ├── tiling.py
│   ├── cleanup.py
│   ├── methods/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── bm.py
│   │   ├── zs_n2n.py
│   │   ├── noise2fast.py
│   │   └── structn2v.py
│   └── py.typed
└── tests/
    ├── test_api.py
    ├── test_bm.py
    ├── test_callbacks.py
    ├── test_cancellation.py
    ├── test_channels.py
    ├── test_config.py
    ├── test_dimensions.py
    ├── test_noise2fast.py
    ├── test_structn2v.py
    ├── test_tiling.py
    └── test_zs_n2n.py
```

Keep method adapters isolated. Shared normalization, tiling, device, callback,
and cancellation behavior must not be duplicated in each implementation.

## 6. Dependencies

Keep the base package light:

```text
numpy
torch
scipy
scikit-image
tifffile
imageio
jdll-unet
```

Use `jdll-unet` only where its model, tiling, scale, or inference utilities are
actually suitable. Do not force ZS-N2N or Noise2Fast into a segmentation UNet
when that would change the published method.

Optional dependency groups should isolate restricted or method-specific code:

```toml
[project.optional-dependencies]
bm = ["bm3d", "bm4d"]
test = ["pytest", "pytest-cov"]
dev = ["pytest", "pytest-cov", "ruff", "mypy"]
```

Do not depend on TensorFlow merely to reuse the old `n2v` package. Implement the
required StructN2V masking/training logic in PyTorch, using a `jdll-unet`
backbone when compatible. This keeps one neural runtime across CUDA, MPS, and
CPU.

## 7. Public API

The stable public API must be array-first:

```python
from jdll_denoise import capabilities, denoise, preview, resolve_config
```

Required signatures:

```python
def capabilities() -> dict: ...

def resolve_config(config: dict, input_shape: tuple[int, ...] | None = None) -> dict: ...

def preview(
    image,
    config: dict,
    *,
    callbacks=None,
    cancellation=None,
) -> dict: ...

def denoise(
    image,
    config: dict,
    *,
    callbacks=None,
    cancellation=None,
) -> dict: ...
```

`preview` and `denoise` must use the same method implementation. Preview may use
a bounded plane/slab and reduced effort explicitly supplied by JDLL, but it must
not use a different undocumented algorithm.

Required result schema:

```python
{
    "denoised": np.ndarray,       # float32, same shape/order as input
    "metadata": {
        "method": "noise2fast",
        "effort": "balanced",
        "device_requested": "cuda",
        "device_used": "cuda",
        "dimension_mode": "2d",
        "input_shape": [1, 512, 512],
        "output_shape": [1, 512, 512],
        "normalization": {...},
        "resolved_config": {...},
        "seed": 5489,
        "warnings": [],
    },
}
```

Never mutate the caller's input array or configuration dictionary.

## 8. Configuration Schema

Required common keys:

```json
{
  "method": "noise2fast",
  "effort": "balanced",
  "device": "cpu",
  "dimensions": "auto",
  "axes": "cyx",
  "seed": 5489,
  "normalization": {
    "mode": "percentile",
    "low": 0.0,
    "high": 100.0,
    "per_channel": true,
    "clip": true
  },
  "tiling": {
    "enabled": "auto",
    "tile_size": "auto",
    "overlap": "auto",
    "blend": "gaussian"
  },
  "method_config": {}
}
```

Accepted method identifiers:

```text
bm3d_bm4d
zs_n2n
noise2fast
structn2v
```

Accepted devices:

```text
cpu
cuda
mps
auto
```

Accepted dimension modes:

```text
auto
2d
2.5d
3d
```

Unknown keys must fail in strict mode and be retained under an explicit
`extensions` object if forward compatibility is needed. Invalid values must
produce actionable `ConfigurationError` messages.

`resolve_config` must return every automatic value resolved to a concrete value
and emit a `resolved_config` callback event before expensive work begins.

## 9. Input and Output Data Model

Canonical arrays are channels-first:

```text
2D:       C,Y,X
3D:       C,Z,Y,X
time 2D:  T,C,Y,X
time 3D:  T,C,Z,Y,X
```

The public API may accept explicit axes and normalize internally, but the result
must be returned in the same axis order and shape as the supplied input.

Requirements:

- Accept NumPy arrays and objects convertible with `np.asarray`.
- Validate finite values.
- Reject unsupported or ambiguous axes instead of guessing.
- Preserve every non-spatial axis.
- Return finite `float32` values in the original physical intensity scale.
- Record source dtype and intensity range in metadata.
- Do not quantize to source integer dtype in Python; Java performs final output
  conversion after blending.
- Empty dimensions and zero-sized arrays fail clearly.

## 10. Channel Semantics

For biological multichannel microscopy, channels normally represent distinct
signals. They must not be merged.

Required behavior:

- Denoise each biological channel independently unless a method has a validated
  joint-channel mode explicitly enabled.
- Preserve channel count and order.
- Estimate and invert normalization per channel.
- Restore each channel to its own original intensity scale.
- Use the same effort, seed policy, and output strength across channels.
- Report progress by channel.
- Never use one fluorescence channel to predict another by default.

Independent processing does not mean arbitrary color scaling. Normalization must
be exactly invertible apart from clipping, so RGB and composite images do not
acquire shifts merely because channels were processed separately.

For true RGB data, support `channel_mode="auto"`:

- use a validated native color mode when the selected implementation supports
  one, notably BM3D;
- otherwise process channels independently with reversible per-channel scaling;
- record the selected mode in metadata.

## 11. Dimensional Support Matrix

The implementation must report capabilities honestly:

| Method | 2D | 2.5D | True 3D | Required v1 behavior |
|---|---:|---:|---:|---|
| BM3D/BM4D | Yes | Not exposed | Yes via BM4D | BM3D for planes, BM4D for volumes |
| ZS-N2N | Yes | No | No | Process stacks plane-by-plane |
| Noise2Fast | Yes | Optional later | No | Process arbitrary stacks plane-by-plane |
| StructN2V | Yes | Yes | Yes | PyTorch blind-spot training with suitable UNet |

Definitions:

- `2d`: one C/Y/X plane is denoised.
- `2.5d`: neighboring Z planes provide context to predict the center plane; all
  planes are reconstructed by sliding the context window.
- `3d`: operations and model receptive fields are volumetric over Z/Y/X.

`dimensions="auto"` must choose:

- BM3D for 2D and BM4D for a true Z volume.
- Plane-wise 2D for ZS-N2N and Noise2Fast stacks.
- 2D StructN2V for a plane and true-3D StructN2V for a sufficiently deep
  volume, falling back to 2.5D when true-3D memory requirements are unsafe.

Never label plane-wise processing as true 3D. Metadata must report whether a
stack was processed as `slice_wise_2d`, `2.5d`, or `3d`.

For time series, process each time point independently by default. Do not use
future frames to denoise earlier frames unless a later temporal method is
explicitly implemented.

## 12. Preview Semantics

The backend receives the preview array selected by JDLL. JDLL owns host C/Z/T
selection and folder ordering.

Required backend behavior:

- A 2D preview denoises the complete supplied plane.
- A volumetric preview may receive a centered slab and must return its center
  plane plus optional slab metadata.
- BM4D and true-3D StructN2V previews should use a centered slab when available
  so the preview remains representative of the final dimensional method.
- ZS-N2N and Noise2Fast preview the selected 2D plane because final stack
  processing is also plane-wise.
- Preview must report whether it is directly reusable for Run.

Required metadata:

```json
{
  "cache_reusable_for_full_input": true,
  "preview_region": {"z_start": 0, "z_end": 1, "center_z": 0},
  "preview_is_representative_only": false
}
```

The backend does not apply the GUI strength blend. JDLL caches original and full
denoised pixels and performs instant blending without another backend call.

## 13. Method: BM3D/BM4D

Purpose: conservative classical denoising for stationary additive Gaussian or
correlated noise.

Requirements:

- Use BM3D for 2D planes.
- Use BM4D for true Z/Y/X volumes.
- Support grayscale and validated multichannel behavior.
- Estimate noise sigma automatically per channel when not supplied.
- Allow an advanced explicit `sigma` override.
- Use robust high-frequency/MAD estimation and document assumptions.
- Clamp clearly invalid estimates and emit a warning.
- Do not use CUDA/MPS unless the selected implementation genuinely supports it.
- Check cancellation between channels, volumes, and algorithm stages where the
  adapter permits it.

Initial effort mapping:

| Effort | Behavior |
|---|---|
| Quick | Hard-thresholding stage only or documented low-complexity profile |
| Balanced | Full two-stage default profile |
| Thorough | Full high-quality profile with conservative settings |

Use official profile APIs rather than undocumented internal constants. Record
the actual package version, profile, stages, and sigma in metadata.

If a third-party call cannot be interrupted internally, emit status before the
call and honor cancellation immediately afterward. Document that cancellation
latency for this method may be coarser.

## 14. Method: ZS-N2N

Implement CVPR 2023 Zero-Shot Noise2Noise's complete objective in PyTorch.
Reference architecture, loss equations and optimizer schedule are specified in
[`README_ZS_N2N_FIDELITY_REQUIREMENTS.md`](README_ZS_N2N_FIDELITY_REQUIREMENTS.md).
The superseding
[effort contract](README_ZS_N2N_EFFORT_PRESETS_REQUIREMENTS.md) authorizes the
explicit performance variants, near-zero initialization and equivalent pair
batching below. Keep the original reference path available for regression.

Requirements:

- Build the paired downsampled/noisy views described by the method.
- Retain three convolutions with LeakyReLU(0.2), grouping only the middle
  convolution. There is no hidden residual, pooling, normalization or dropout.
- Batch the two small views along B, retaining the separate full-resolution
  forward, both symmetric losses and one backward/Adam update on their sum.
- Retain Adam at 0.001 and StepLR(1000, 0.5), stepping after the optimizer.
- Train from the selected image only.
- Reinitialize for each independent image/plane.
- Support CPU, CUDA, and MPS.
- Preserve output shape with correct border handling.
- Complete the resolved update budget and select final weights by default.
  Early stopping and best-weight restoration remain explicit advanced options.
- Emit progress every configurable number of optimization steps.
- Check cancellation every optimization step.

Agreed version-1 presets (width / middle groups / updates / head initialization):

```text
quick:          8 / 1 /  500 / near_zero
balanced:      16 / 2 / 1000 / near_zero
balanced_high: 24 / 1 / 2000 / native
thorough:      48 / 1 / 3000 / native
```

Balanced remains the default; Balanced-high is distinct and ZS-N2N-only.
All presets use `batch_pairs=true`; `false` retains three separate forwards.
Near-zero means only the output weights use Normal(0, 0.001) and output bias
zero. Preserve direct constructor defaults: width 48/groups 1/native.
Expose validated advanced overrides; never mutate caller dictionaries or
silently change presets by device, image size or preview. Default normalization
is min-max (0/100 percentiles with clipping), honoring explicit overrides.
Capabilities must advertise `effort_presets_version=1` and all four canonical
IDs; results/configuration/callbacks report actual settings. Quick/Balanced/
Balanced-high are explicit JDLL variants, not the unmodified reference network.

Required metrics:

- current self-supervised loss;
- exponentially smoothed loss;
- learning rate;
- elapsed time;
- best step if early stopping is active.

## 15. Method: Noise2Fast

Implement the published checkerboard-downsampling method in maintainable
PyTorch code. The official project supports arbitrary extra dimensions by
applying its 2D method over planes; v1 must preserve that behavior without
claiming true volumetric context.

Requirements:

- Construct all four directed full-plane training pairs from the paper's
  squeeze-up and squeeze-left checkerboard downsampling equations.
- Use the paper/reference feed-forward CNN structure: four blocks of two 3x3
  convolutions with ReLU, followed by a 1x1 convolution and sigmoid. Resolve
  hidden width/patience from `presets.py`: Quick 16/50, Balanced 24/100 (default),
  Thorough 64/100. Compact widths are variants, not the released architecture.
- Allow validated positive-integer `hidden_width` and `patience` overrides;
  keep the direct `Noise2FastNet()` constructor default at 64.
- Use batch size 1, binary cross-entropy, and Adam at learning rate 0.001.
- After every update, predict the original full-sized noisy image and use its
  negative MSE to the noisy input as the validation score.
- Stop after the resolved patience in consecutive optimizer steps without a
  strictly improved validation score; reset the counter on each improvement.
  Return the mean of the last `min(completed_steps, 100)` predictions, with
  the 100-output window fixed independently of patience.
- Train a fresh model per independent image/plane.
- Preserve the published loss and reconstruction behavior.
- Support CPU, CUDA, and MPS.
- Reuse tested `jdll-unet` components only if they preserve Noise2Fast's model
  and training behavior.
- Process stacks/time series plane-by-plane with deterministic indexing.
- Check cancellation every optimization step and between planes.
- Release each plane's model before moving to the next unless reuse is shown to
  be methodologically correct.

Quick, Balanced, and Thorough retain the existing JDLL validation-patience
stopping rule with only the authorized width/patience changes. They may not
substitute fixed low step budgets, patch-sampled training, U-Nets, two-view
training, or tiled/blended reconstruction. A positive advanced `steps` value
is an explicit test/debug cap, with a warning when it causes stopping;
`steps=0` means patience-only stopping with an unknown total (`maximum=0`).
Report actual width, patience, averaging count/window and stopping reason in
callbacks/metadata; user-facing counters count steps, not dataset epochs.
Keep min-max normalization defaults while honoring valid explicit overrides.
Use the same settings on every device and through both denoise and preview.

Required metrics match ZS-N2N.

## 16. Method: StructN2V

Implement structured Noise2Void in PyTorch without adding TensorFlow. Reproduce
the paper architectures directly:

- 2D: two-level U-Net, two 3x3 convolutions per level, 2x2 max pooling,
  16 features at the first level, and doubling/halving features through the
  down/up paths.
- 3D/volumetric: two-level U-Net, two 3x5x5 (ZYX) convolutions per level,
  1x2x2 max pooling, and the same 16-feature doubling/halving rule.
- linear one-channel output for each independent biological channel.

The loss must be evaluated only at blind/structured mask locations. The model
must not learn an identity mapping through unmasked target pixels.

Noise structure options in v1:

```text
auto
local_isotropic
horizontal
vertical
periodic_grid
```

Do not expose Custom geometry in v1. Define an internal mask-generator protocol
so it can be added later without changing the main API.

Preset semantics:

- `local_isotropic`: compact symmetric exclusion neighborhood.
- `horizontal`: horizontally elongated exclusion mask.
- `vertical`: vertically elongated exclusion mask.
- `periodic_grid`: repeating offsets inferred or configured from autocorrelation.
- `auto`: inspect high-pass residual autocorrelation, test bounded candidates,
  and select using held-out masked prediction loss.

Auto selection must be deterministic for a supplied seed and must emit candidate
scores and the selected structure through callbacks/metadata.

The canonical schedules are those reported in the paper:

- 2D: 1600 random 256x256 patches, 600 epochs, batch size 4, Adam at 2e-5.
- 3D: 1900 random 32x64x64 patches, 100 epochs, batch size 10, Adam at 4e-6.
- N2V/StructN2V: randomly select 2% active pixels in every training patch.
- Replace each active center from its local N2V neighborhood, hide the remaining
  extended-mask support with random values, and evaluate MSE only at centers.

Effort presets may not shorten these schedules. A positive explicit `steps`
value is a reported test/debug override only; `steps=0` runs all epochs.

Required configurable values:

```json
{
  "noise_structure": "auto",
  "mask_fraction": "auto",
  "correlation_radius": "auto",
  "period": "auto",
  "architecture": "auto",
  "patch_size": "auto",
  "patch_count": "auto",
  "batch_size": "auto",
  "learning_rate": "auto",
  "epochs": "auto",
  "steps": "auto"
}
```

## 17. Preset Resolution

Presets must live in one typed/configured module, not scattered conditionals.

Each resolved preset must include:

- method;
- effort;
- dimension mode;
- device;
- optimization steps/stages;
- learning rate;
- architecture, where applicable;
- patch/tile size;
- overlap;
- normalization;
- seed;
- estimated memory;
- cancellation granularity.

Quick uses a lower work budget for methods with a tunable budget. Noise2Fast
uses the agreed width/patience effort presets while retaining the rest of its
loop; lower effort is not a runtime guarantee and greater capacity does not
guarantee lower error. StructN2V retains its complete fixed experimental
schedule at every effort level.

## 18. Normalization and Intensity Preservation

Neural methods require numerically stable normalization while scientific output
must remain in the source intensity domain.

Required flow per channel:

1. Record dtype, finite min/max, and configured percentiles.
2. Compute robust normalization parameters.
3. Normalize into the method's expected range.
4. Denoise in floating point.
5. Invert normalization exactly.
6. Return float32 in original intensity units.

Do not independently normalize tiles. Calculate normalization from the complete
plane/volume or a deterministic representative sample, then reuse it for every
tile.

Constant images must return unchanged without division by zero. NaN/Inf inputs
must fail clearly unless an explicit future repair policy is added.

## 19. Tiling and Large Inputs

Large images and volumes must not require full-device-memory inference.

Reuse `jdll-unet` tiling/blending utilities when suitable. Requirements:

- deterministic tile order;
- overlap sufficient for the model receptive field;
- Gaussian or equivalent seam-free blending;
- same-shape reconstruction;
- no per-tile normalization drift;
- progress and cancellation per tile;
- CPU fallback after device OOM only when explicitly allowed;
- automatic tile-size reduction on OOM before abandoning the requested device;
- metadata containing final tile size and any fallback.

Self-supervised training patch sampling and final tiled prediction are separate
concerns. Do not train each inference tile independently unless that is the
published method's required behavior.

BM3D/BM4D adapter tiling is disabled by default because naive tiling changes
block matching. If required for very large inputs, use large overlap and test
seams quantitatively before enabling it.

## 20. Device Behavior

`capabilities()` must inspect the runtime and return, at minimum:

```python
{
  "methods": {
    "bm3d_bm4d": {"available": True, "devices": ["cpu"], "dimensions": ["2d", "3d"]},
    "zs_n2n": {
        "available": True, "devices": ["cpu", "cuda"], "dimensions": ["2d"],
        "effort_presets_version": 1,
        "efforts": ["quick", "balanced", "balanced_high", "thorough"],
    },
    "noise2fast": {"available": True, "devices": ["cpu", "cuda"], "dimensions": ["2d"]},
    "structn2v": {"available": True, "devices": ["cpu", "cuda"], "dimensions": ["2d", "2.5d", "3d"]}
  },
  "cuda_available": True,
  "mps_available": False
}
```

Include MPS only when built and available. Never advertise acceleration merely
because an environment variant was selected.

If a method does not support acceleration, JDLL disables the checkbox. If a
stale request still asks for an unavailable device, the backend must either:

- fail with `DeviceUnavailableError`, or
- fall back to CPU only when `allow_cpu_fallback=true`, emitting a warning.

Record requested and actual devices.

## 21. Callbacks

Callbacks must be ordinary Python callables and must not know about Appose.
Accept either a single callable or a callback registry/list.

Every event is a dictionary containing:

```python
{
  "type": "denoise_progress",
  "phase": "optimization",
  "method": "noise2fast",
  "message": "Noise2Fast step 120; no improvement for 12/100 steps.",
  "current": 120,
  "maximum": 0,
  "channel": 1,
  "channels": 2,
  "z": 0,
  "z_total": 1,
  "metrics": {
    "loss": 0.0012,
    "validation_patience": 100,
    "epochs_without_validation_improvement": 12
  },
  "info": {}
}
```

Required phases:

```text
validating
resolving_config
estimating_noise
selecting_structure
preparing
optimization
inference
tiling
blending
finalizing
completed
cancelled
failed
```

Required event types:

```text
denoise_progress
resolved_config
warning
status
```

Callback failures must not corrupt computation. Decide and document whether they
are isolated with warnings or propagated in strict mode. Default should isolate
non-cancellation callback failures.

JDLL will translate these events to `task.update(...)`.

## 22. Cancellation

Define a backend-neutral cancellation protocol, for example:

```python
class CancellationToken(Protocol):
    def is_cancelled(self) -> bool: ...
```

Also accept a zero-argument callable returning bool.

Check cancellation:

- before validation and expensive setup;
- during every neural optimization step;
- between channels, Z planes, time points, and files;
- before and after third-party BM calls;
- before allocating large outputs;
- during tiled inference and blending;
- before returning/publishing a result.

Raise a dedicated `DenoisingCancelled` exception. Do not return partial arrays
as successful results.

## 23. Cleanup

Cleanup runs in `finally` after success, cancellation, and failure:

```python
del model, optimizer, temporary_tensors
gc.collect()
```

For CUDA:

```python
torch.cuda.empty_cache()
torch.cuda.ipc_collect()  # only when safe and CUDA initialized
```

For MPS:

```python
torch.mps.empty_cache()  # when available
```

Do not treat cache cleanup as correctness. Release references first. Cleanup
errors must not hide the original exception.

## 24. Error Model

Provide dedicated exceptions:

```text
DenoisingError
ConfigurationError
InputFormatError
MethodUnavailableError
DeviceUnavailableError
DenoisingCancelled
OutOfMemoryDenoisingError
ThirdPartyBackendError
```

Messages must include method, device, phase, and an actionable recommendation.
Do not expose only raw PyTorch or third-party stack traces to JDLL users.

## 25. Reproducibility Metadata

Return sufficient metadata for JDLL's sidecar JSON:

```json
{
  "software": "JDLL",
  "backend": "jdll-denoise",
  "backend_version": "0.1.0",
  "method": "noise2fast",
  "effort": "balanced",
  "denoising_strength": 1.0,
  "device_requested": "cuda",
  "device_used": "cuda",
  "dimensions": "slice_wise_2d",
  "channels": 2,
  "seed": 5489,
  "resolved_config": {},
  "third_party_versions": {},
  "warnings": []
}
```

Strength is supplied by JDLL when writing the final sidecar because blending is
performed there.

## 26. JDLL Integration Contract

The Java implementation will:

- install a Pixi environment containing this package;
- call `capabilities()` before enabling acceleration controls;
- pass arrays through Appose shared memory;
- pass explicit axes and dimensions;
- translate backend callbacks to Appose task updates;
- cache original and fully denoised preview arrays;
- perform real-time strength blending;
- preserve host metadata and create new host images;
- process folders deterministically;
- write temporary files and atomically rename outputs;
- write sidecar JSON metadata;
- own output collision naming.

Output rules in Java:

```text
single file:  image.tif -> image_denoised.tif
folder:       images/   -> images_denoised/
collision:    image_denoised-1.tif, images_denoised-1/, ...
```

The Python package must not write final source-adjacent files during normal
Appose array calls.

## 27. Performance Requirements

Benchmark and report:

- 256x256, 512x512, 1024x1024 2D grayscale;
- 3-channel 512x512;
- 32x256x256 volume;
- CPU, CUDA, and MPS where available;
- all effort levels;
- peak host and device memory;
- cancellation latency.

Minimum usability targets, measured on representative hardware and adjustable
only with documented evidence:

- Noise2Fast Quick 512x512 preview: practical interactive wait, ideally under
  10 seconds on a laptop GPU and under 60 seconds on a modern CPU. This is an
  aspiration, not a time limit: patience-only optimization has unknown duration.
- Progress callback latency: under 250 ms for neural methods.
- Cancellation response: under 500 ms for neural methods, excluding an
  uninterruptible third-party BM stage.
- Slider blending is outside Python and must remain interactive in Java.

Do not achieve targets by silently reducing output resolution.

## 28. Test Requirements

### Unit tests

- Config validation and preset monotonicity.
- Capabilities for CPU/CUDA/MPS combinations using mocks.
- Axis conversion and exact shape restoration.
- Constant, negative, float, uint8, uint16, and multichannel inputs.
- Per-channel intensity-scale restoration.
- Deterministic seeds.
- Callback schemas and ordering.
- Callback failure isolation.
- Cancellation in every long-running phase.
- Cleanup after success/failure/cancellation.
- Tiling reconstruction without seams.
- OOM tile reduction and CPU fallback policy.
- Every StructN2V mask preset.
- Auto structure selection determinism.
- 2D, slice-wise stack, 2.5D, and true-3D capability routing.
- Time-series axis preservation.

### Method tests

- BM3D/BM4D adapters preserve shape and improve a synthetic noisy phantom when
  optional dependencies are installed.
- ZS-N2N smoke training reduces its self-supervised loss and returns finite
  same-shape output.
- Noise2Fast checkerboard views and reconstruction match reference behavior.
- StructN2V loss is evaluated only at masked positions.
- Structured synthetic noise is reduced more effectively by an appropriate
  StructN2V preset than by an intentionally wrong orientation in a stable test.

### Integration tests

- Public `preview` and `denoise` for every method on CPU.
- CUDA/MPS tests guarded by availability.
- Appose-like callback forwarding without importing Appose.
- Shared-memory-compatible contiguous outputs.
- Cancel and immediately run again in the same Python process.
- Multiple sequential jobs do not retain models/device memory.

### Quality tests

Use synthetic clean phantoms with seeded Gaussian, Poisson-like, stripe, and
periodic noise. Check PSNR/SSIM improvement with tolerant method-appropriate
thresholds. Do not use exact pixel snapshots for stochastic neural methods.

## 29. CI and Code Quality

Required:

- Linux CPU tests on every push and pull request.
- Python 3.10, 3.11, and 3.12 where dependencies permit.
- Ruff linting.
- Mypy or equivalent type checking for public APIs.
- Coverage focused on configuration, dispatch, cancellation, and data handling.
- Optional BM tests skipped with a clear reason when restricted dependencies
  are absent.
- No downloaded datasets or model weights required for the normal test suite.

## 30. Acceptance Criteria

The backend is ready for JDLL integration when:

1. `pip install -e .` installs the base package cleanly.
2. `capabilities()` accurately reports method/device/dimension support.
3. Noise2Fast is the default resolved method.
4. Every method runs on CPU or reports a precise missing optional dependency.
5. Neural methods run on available CUDA/MPS devices.
6. Every output has the same shape/order as its input and is float32 in the
   original intensity scale.
7. Multichannel, stacks, volumes, and time series follow the declared support
   matrix without silent dimensional downgrades.
8. Preview and full run use the same algorithm and report cache reusability.
9. Structured progress reaches JDLL without an Appose dependency.
10. Cancellation stops neural work promptly and no partial result is returned.
11. Cleanup permits an immediate second job on another device.
12. Resolved configuration and reproducibility metadata are complete.
13. The full CPU test suite passes.
14. License restrictions for BM3D/BM4D are visible and respected.

## 31. Required Agent Deliverables

The implementing agent must deliver:

- Complete package source and tests.
- `pyproject.toml` with optional dependency groups.
- Public API documentation and examples.
- Method and dimensional capability table.
- Preset table with final benchmark-backed values.
- Licensing documentation for every third-party implementation.
- Benchmark report.
- Changelog entry.
- No Appose or Java dependency.
- A short JDLL integration example showing callback forwarding and array I/O.

Any deliberate deviation from this specification must be documented with the
reason, user-visible consequence, and migration path.
