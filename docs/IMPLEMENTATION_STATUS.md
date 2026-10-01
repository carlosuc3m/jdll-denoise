# v0.1 implementation status and deliberate deviations

The stable array API, strict configuration, axis restoration, independent
channel/time jobs, normalization, callbacks, cancellation, cleanup, and all
denoising adapters are implemented and covered by CPU tests.

The self-supervised implementations and their explicitly authorized variants are:

- ZS-N2N implements the complete symmetric residual/consistency objective,
  fixed paired views, Adam/StepLR and external residual reconstruction. Its
  version-1 effort contract has Quick (width 8/groups 1/500 updates/near-zero
  head), Balanced (16/2/1000/near-zero, default), Balanced-high
  (24/1/2000/native) and Thorough (48/1/3000/native). All presets batch the small
  views while retaining the full-resolution forward and a single update on the
  complete loss. Only the middle convolution is grouped; no hidden skip exists.
  The first three presets are explicit JDLL performance variants. Thorough
  retains the reference architecture, native initialization and budget with
  equivalent batching. A width-48/groups-1/native/unbatched reference path
  remains available. Default min-max normalization is a documented wrapper,
  not a claim of identical paper preprocessing; explicit overrides are honored.
- Noise2Fast implements four directed full-plane checkerboard pairs, eight
  3×3 convolutions with ReLU, a 1×1 sigmoid head, BCE, Adam at 0.001 and
  per-update full-image validation. The authorized effort presets are Quick
  16 features/patience 50 steps, Balanced 24/100 (default), and Thorough 64/100,
  identical on every device. Quick/Balanced are compact-width variants;
  Thorough uses the released architecture, with no claim of bitwise equivalence
  to the standalone reference. Only width/patience differ from the prior JDLL
  loop. A strict improvement resets the counter, and stopping occurs at the
  resolved patience, not patience+1. Every preset averages the last 100 available
  predictions. The former U-Net, two-view, patch-sampled path remains removed.
- StructN2V implements N2V local-neighborhood active-pixel manipulation,
  extended structured blind masks with active-only MSE, random patch training,
  the paper's two-level 16-feature 2D U-Net, its anisotropic 3D U-Net, and the
  published 2D/3D patch, batch, epoch, learning-rate, and active-pixel settings.
  `auto` mask selection performs a deterministic held-out masked-loss tournament
  after autocorrelation analysis.

The remaining explicit limitations are outside those algorithm cores:

- BM RGB jobs use independent, reversibly normalized channels. Native-color
  mode remains disabled until the official packages have dedicated reference
  tests for that path.
- StructN2V paper schedules are intentionally expensive. A positive explicit
  `method_config.steps` value exists for tests and debugging, but produces a
  warning and is never described as a paper-complete run.
- Noise2Fast full-plane training and validation can require substantial memory.
  Accelerator OOM may restart on CPU only when `allow_cpu_fallback=true`;
  patching or tiled reconstruction is not silently substituted.
- Noise2Fast patience is not a time or total-step budget: optimization reports
  `maximum=0` without an explicit debug cap. Greater width is not guaranteed
  to improve every image. Positive advanced width/patience overrides are
  validated and reported; they never change the fixed 100-output window.
- The Java caller still needs to stop forcing 0.1/99.9 normalization for
  Noise2Fast and display resolved settings/unknown totals. See the separate
  [Java follow-up](JDLL_INTEGRATION.md#required-java-follow-up-not-implemented-in-this-python-repository).
- ZS-N2N's separate Java integration must gate on preset contract version 1
  and advertised efforts, show Balanced-high only for ZS-N2N, invalidate preview
  caches on effort changes and use min-max normalization defaults. See the
  [ZS-N2N bridge contract](JDLL_INTEGRATION.md#zs-n2n-version-1-contract).

The effort change preserves array shape/order, per-channel scale restoration,
callback/cancellation contracts, dimensional metadata, min-max defaults and
explicit normalization overrides. The ZS-N2N effort implementation does not
change Noise2Fast, BM3D/BM4D or StructN2V behavior.
