# Changelog

## Unreleased

- Implemented ZS-N2N effort contract version 1: Quick 8/1/500/near-zero,
  Balanced 16/2/1000/near-zero (default), Balanced-high 24/1/2000/native and
  Thorough 48/1/3000/native (width/groups/updates/head). Added batched pair
  execution without changing the complete loss, validated advanced controls,
  min-max defaults, versioned capabilities/results, actual-settings callbacks
  and honest fidelity metadata. Retained native/unbatched reference defaults.
  Advanced stopping/restoration no longer depends on reporting frequency.
  Added loss/gradient/reference, public API, cleanup and accelerator tests;
  documented the separate Java capability/UI/normalization integration.
- Added the agreed Noise2Fast effort presets: Quick 16 features/patience 50
  steps, Balanced 24/100 (default), Thorough 64/100, with identical settings on
  CPU/CUDA/MPS and validated advanced width/patience overrides. This is the
  authorized width/patience exception to the earlier full-width implementation.
  Retained the existing loop and fixed last-100-prediction average; callbacks
  and metadata now report actual settings, stale steps and unknown uncapped
  totals. Added CPU numerical/boundary regressions, public denoise/preview and
  accelerator checks, and a public-API benchmark runner. Documented separate
  Java normalization/UI follow-up; no Java or other-method behavior changes.
- Replaced the approximate ZS-N2N path with the exact three-layer noise
  predictor, fixed grouped pair downsampler, complete residual/consistency loss,
  Adam/StepLR optimization, joint RGB mode, and fidelity metadata/tests.
- Replaced the approximate Noise2Fast U-Net/two-view/patch path with the paper's
  eight-convolution 64-channel CNN, four directed full-plane checkerboard pairs,
  BCE, full-image validation stopping, and 100-output reconstruction average.
- Replaced the compact StructN2V models and rolled-pixel corruption with the
  paper's 2D and anisotropic 3D U-Nets, N2V neighborhood manipulation, extended
  blind masks, active-only loss, complete 2D/3D schedules, and held-out automatic
  mask selection.
- Added structural fidelity tests, explicit debug-cap warnings, device-residence
  metadata, and full-plane Noise2Fast CPU fallback after accelerator OOM.

## 0.1.0 - 2026-07-15

- Added the stable `capabilities`, `resolve_config`, `preview`, and `denoise`
  array API.
- Added optional license-isolated BM3D/BM4D adapters.
- Added PyTorch ZS-N2N, Noise2Fast, and 2D/2.5D/3D StructN2V implementations.
- Added axis-safe channel/time processing, per-channel normalization, tiled 2D
  inference, structured callbacks, cooperative cancellation, deterministic
  seeds, device selection, metadata, cleanup, and dedicated errors.
- Added CPU tests, integration documentation, third-party licensing notices,
  and a reproducible benchmark runner/report.
