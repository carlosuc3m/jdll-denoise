# ZS-N2N Algorithm Fidelity Requirements

The [ZS-N2N effort contract](README_ZS_N2N_EFFORT_PRESETS_REQUIREMENTS.md)
(2026-09-16) supersedes this document's earlier fixed-width, native-only,
three-forward restrictions for standard presets. Its explicit JDLL variants
retain the complete loss and reference path. This document still describes
the reference equations and regression requirements, not a mandate that all
presets have the reference architecture or initialization.

## Purpose

The implementation exposed as `method="zs_n2n"` must be a faithful,
maintainable PyTorch implementation of **Zero-Shot Noise2Noise: Efficient
Image Denoising Without Any Data** (Mansour and Heckel, CVPR 2023). It must not
be a method merely inspired by ZS-N2N while retaining the published name.

Primary references:

- Paper: <https://openaccess.thecvf.com/content/CVPR2023/html/Mansour_Zero-Shot_Noise2Noise_Efficient_Image_Denoising_Without_Any_Data_CVPR_2023_paper.html>
- Authors' executable Colab: <https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b>

The paper defines the objective. The authors' Colab is the implementation
reference for the reference architecture and optimization hyperparameters.
Authorized effort variants and advanced overrides must be documented honestly
in resolved configuration and result metadata; the native unbatched reference
path remains available explicitly.

## Historical approximation (removed before the effort contract)

The former `jdll_denoise/methods/zs_n2n.py` approximation had these problems
(the first six were removed by the fidelity implementation):

1. It optimizes one alternating view-to-view MSE per step. Published ZS-N2N
   optimizes the sum of a symmetric residual loss and a symmetric
   full-resolution consistency loss on every step.
2. It never evaluates the network on the full-resolution noisy image during
   optimization. The published consistency term requires this operation and
   is a critical regularizer against overfitting.
3. `LightweightDenoiser` has four `3x3` convolutions. The reference network has
   two `3x3` convolutions followed by one `1x1` convolution.
4. The current hidden activation uses LeakyReLU slope `0.1`; the reference uses
   `0.2`.
5. Current pair generation replicate-pads odd image dimensions. The reference
   uses fixed `2x2`, stride-2 grouped convolutions without padding, naturally
   discarding the unmatched final row or column from the training views.
6. The reference optimization uses Adam plus StepLR decay. The current
   implementation keeps a constant learning rate.
7. Width changes alter the published model capacity. The new effort contract
   now explicitly authorizes width/group/initialization changes, reported as
   JDLL performance variants rather than silently claiming the reference model.

The missing consistency loss is not a minor detail. The paper explicitly
identifies it as important for denoising performance and resistance to
single-image overfitting.

## Canonical pair downsampler

For input `y` with shape `B,C,H,W`, construct two fixed, channel-wise filters:

```text
k1 = [[0.0, 0.5],
      [0.5, 0.0]]

k2 = [[0.5, 0.0],
      [0.0, 0.5]]
```

Repeat each filter for all channels and apply grouped `conv2d` with `stride=2`
and no padding:

```python
d1 = conv2d(y, k1, stride=2, groups=channels)
d2 = conv2d(y, k2, stride=2, groups=channels)
```

Requirements:

- Filters are fixed and never trainable.
- Downsampling is channel-wise.
- Pair shapes are `B,C,floor(H/2),floor(W/2)`.
- Do not replicate, reflect or zero-pad before creating the training pair.
- Full-resolution final inference must still preserve the original `H,W`.

## Canonical network

The network `f_theta` predicts noise, not the denoised image. It must contain:

```text
Conv2d(C, 48, kernel_size=3, padding=1)
LeakyReLU(negative_slope=0.2, inplace=True)
Conv2d(48, 48, kernel_size=3, padding=1)
LeakyReLU(negative_slope=0.2, inplace=True)
Conv2d(48, C, kernel_size=1)
```

There are no pooling, normalization or additional convolution layers. For one
input channel this network has 21,313 trainable parameters with PyTorch's
default convolution biases.

The denoised estimate is always calculated outside the network:

```python
denoised = noisy - model(noisy)
```

Final normalized output is clamped to `[0, 1]`.

The reference hidden width is 48, with one middle group and native head
initialization. Direct construction retains these defaults. Production
presets instead use the widths/groups/initializations in the superseding
effort contract. Only the second convolution may be grouped, and near-zero
initialization changes only the last convolution's weights/bias. No preset
adds a hidden residual connection. Fidelity metadata distinguishes variants.

## Canonical loss

For noisy image `y`, let:

```python
d1, d2 = pair_downsampler(y)
p1 = d1 - model(d1)
p2 = d2 - model(d2)
full = y - model(y)
full_d1, full_d2 = pair_downsampler(full)
```

The residual loss is:

```python
loss_residual = 0.5 * (mse(d1, p2) + mse(d2, p1))
```

The consistency loss is:

```python
loss_consistency = 0.5 * (
    mse(p1, full_d1) + mse(p2, full_d2)
)
```

The optimized objective is exactly:

```python
loss_total = loss_residual + loss_consistency
```

Both directions and both loss components must be evaluated during every
optimization step. Alternating one direction between steps is not equivalent
and is not accepted. Do not detach any prediction used by these terms.

The equations above are the retained unbatched reference path. All version-1
presets evaluate `model(cat((d1, d2), dim=0)).chunk(2, dim=0)` for the two small
views, plus the unchanged full-size call. This equivalent execution needs two
forwards, one backward and one update; `batch_pairs=false` keeps three forwards.

Callbacks must report `loss_total`, `loss_residual`, and `loss_consistency`
separately, in addition to learning rate, step, total steps and elapsed time.

## Canonical optimization

The authors' reference configuration is:

```text
optimizer: Adam with PyTorch defaults
learning rate: 0.001
scheduler: StepLR(step_size=1000, gamma=0.5)
reference iterations: 3000
```

Call `scheduler.step()` once after every optimizer step. The resolved
configuration and metadata must record optimizer, initial learning rate,
scheduler type, scheduler step size, scheduler gamma and completed iterations.

The UI efforts now use the authoritative
[version-1 preset table](README_ZS_N2N_EFFORT_PRESETS_REQUIREMENTS.md#1-agreed-presets).
Quick/Balanced/Balanced-high are explicit performance variants with budgets
500/1000/2000; Thorough retains the reference architecture/native head and
3000-update budget. Every preset preserves the pairs, complete loss and
scheduler. Batching must not remove any loss term or detach a prediction.

Canonical mode uses the final optimization weights, matching the reference.
Early stopping and best-weight restoration are not part of published ZS-N2N
and must be disabled by default. If retained as advanced extensions, metadata
must identify the run as noncanonical and record the criterion and selected
step.

## Channels, stacks and tiling

The published network supports arbitrary channel counts jointly. Therefore:

- A true RGB/native-color mode must train one `C -> C` reference network over
  all color channels together.
- Independent biological-channel processing remains a valid JDLL orchestration
  mode; each job is then a faithful single-channel ZS-N2N run and metadata must
  say `channel_mode="independent"`.
- Z stacks are processed plane-by-plane because published ZS-N2N is 2D. Do not
  describe this as 2.5D or 3D ZS-N2N.
- Time points may be independent jobs.

The canonical reference path trains and infers on the complete plane. Tiled
inference is an optional JDLL memory extension, not part of the paper. It must
use overlap and blending, preserve output shape, and be clearly recorded in
metadata. It must not train a separate model per inference tile.

## Normalization and output restoration

The algorithm operates on finite floating-point values normalized to `[0,1]`.
JDLL retains its reversible per-channel percentile normalization wrapper,
defaulting to low=0/high=100 and clipping for ZS-N2N as of contract version 1.
Valid explicit overrides remain honored; this wrapper is not claimed to be
the exact paper preprocessing. Requirements:

- reference tests can bypass it and exercise the canonical `[0,1]` algorithm;
- the same normalized plane is used for pair creation, training and inference;
- denoised values are clamped before inverse normalization;
- original axes, shape and physical intensity scale are restored by the public
  API;
- normalization details are present in result metadata.

## Cancellation, devices and cleanup

CPU, CUDA and MPS support may be added without changing the algorithm.

- Check cancellation between optimization steps and inference tiles.
- Keep all model inputs and fixed filters on the resolved device.
- Do not silently fall back to CPU unless explicitly allowed by configuration.
- Release model, optimizer, scheduler and tensors in `finally` blocks.
- Run the existing device-specific cache cleanup after success, cancellation or
  failure.

These engineering additions do not authorize changes beyond the explicit effort
variants or alteration of the complete loss.

## Required tests

Implementation is incomplete until all tests below exist and pass.

### Structural tests

- Assert the exact three convolution layers, kernel sizes, channel dimensions,
  biases and LeakyReLU slope.
- Assert 21,313 parameters for `C=1`, hidden width 48.
- Assert each version-1 preset's width/groups/initialization/parameter count,
  plus unchanged direct constructor defaults and near-zero first-step gradients.

### Pair-downsampler tests

- Compare even-size outputs against the two explicit fixed-filter equations.
- For odd inputs, assert floor-sized outputs and exact equality with grouped
  stride-2 convolution without padding.
- Test multiple channels and verify that downsampling never mixes channels.

### Loss tests

- Implement a small, test-only reference loss directly from the equations
  above.
- Compare residual, consistency and total losses to the production
  implementation for fixed tensors and fixed model weights.
- Verify gradients from both loss components reach every trainable layer.
- Include a regression test that fails if the consistency term is removed.

### Optimization tests

- With fixed CPU seeds, compare one complete optimizer step against a direct
  transcription of the authors' Colab.
- Assert learning rates of `1e-3`, `5e-4`, and `2.5e-4` after 0, 1000, and 2000
  scheduler steps, respectively, accounting explicitly for PyTorch StepLR call
  order.
- Assert that canonical inference computes `clamp(y - model(y), 0, 1)`.

### Reference regression tests

- Vendor a tiny deterministic input fixture and expected intermediate pair,
  initial losses and short-run output generated from a pinned transcription of
  the authors' Colab.
- Compare CPU results within documented numerical tolerances.
- Keep the reference transcription in tests independent from production
  helpers so the same bug cannot make both paths pass.

### API tests

- Preserve shape and axes for even and odd 2D images.
- Cover single-channel, independent multichannel, native RGB, stack and time
  orchestration behavior.
- Verify progress payloads expose all three losses and the scheduled learning
  rate.
- Verify cancellation and cleanup on CPU, CUDA and MPS where available.
- Verify metadata distinguishes canonical runs from optional extensions such as
  early stopping, nonstandard width or tiled inference.

## Documentation and naming policy

After implementation:

- Remove the ZS-N2N approximation from
  `docs/IMPLEMENTATION_STATUS.md`; document only genuine remaining extensions.
- Cite the paper and authors' Colab in `README.md` and third-party/reference
  documentation.
- Record that this is an independent implementation, not authors' source code,
  unless code is directly reused under a compatible license.
- Do not claim newer methods such as LoTA-N2N are implemented under the
  `zs_n2n` name. They require separate method identifiers, references,
  configuration and fidelity tests.

Named algorithms must retain honest, documented fidelity claims. The new
effort contract explicitly authorizes ZS-N2N performance variants under
`method="zs_n2n"`; they preserve the full published objective but are not
advertised as the unmodified reference network. No broader approximation or
changes to other denoising methods are authorized by this exception.
