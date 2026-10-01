# jdll-denoise

`jdll-denoise` is an array-first Python backend for classical and zero-shot /
self-supervised microscopy denoising in JDLL. It has no Java or Appose runtime
dependency. Inputs are NumPy-compatible arrays; outputs are contiguous `float32`
arrays in the input intensity scale.

The complete implementation contract is in
[`README_IMPLEMENTATION_REQUIREMENTS.md`](README_IMPLEMENTATION_REQUIREMENTS.md).
Noise2Fast width/patience presets follow the superseding
[effort requirements](README_NOISE2FAST_EFFORT_PRESETS_REQUIREMENTS.md).
ZS-N2N presets and batched views follow the superseding
[ZS-N2N effort contract](README_ZS_N2N_EFFORT_PRESETS_REQUIREMENTS.md).

## Methods and dimensional support

| Public method | 2D | Stack behavior | 2.5D | True 3D | Devices |
| --- | --- | --- | --- | --- | --- |
| `bm3d_bm4d` | BM3D | BM4D volume | — | BM4D | CPU |
| `zs_n2n` | complete ZS-N2N objective, explicit effort variants | independent planes | — | — | CPU/CUDA/MPS |
| `noise2fast` | four-pair Noise2Fast, configurable width | independent planes | — | — | CPU/CUDA/MPS |
| `structn2v` | paper 2D U-Net and anisotropic 3D U-Net | selectable | anisotropic 3D model | volumetric model | CPU/CUDA/MPS |

Noise2Fast is the default. Channels and time points are processed independently;
their count, order, and physical intensity scales are preserved. Runtime device
availability is always obtained from `capabilities()` rather than inferred from
the requested environment.

## Installation

```bash
python -m pip install -e .
python -m pip install -e ".[test]"  # tests
```

BM3D/BM4D are deliberately optional:

```bash
python -m pip install -e ".[bm]"
```

The official Tampere `bm3d` and `bm4d` packages are restricted to
non-commercial scientific/educational use. They are not copied or redistributed
by this Apache-2.0 project. Installing the extra means accepting and complying
with their separate terms. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## Minimal use

```python
import numpy as np
from jdll_denoise import capabilities, denoise

image = np.asarray(...)
print(capabilities())

result = denoise(
    image,
    {
        "method": "noise2fast",
        "effort": "balanced",
        "device": "auto",
        "dimensions": "auto",
        "axes": "cyx",
        "seed": 5489,
    },
)
pixels = result["denoised"]
metadata = result["metadata"]
```

Canonical axes are `CYX`, `CZYX`, `TCYX`, and `TCZYX`, though any explicit
permutation is restored on output. `Y` and `X` are mandatory. Four- and
five-dimensional arrays require an explicit `axes` value because channel, depth,
and time cannot be guessed safely.

`preview()` invokes the same method implementation. A supplied 2D plane returns
that plane. A supplied volumetric slab is denoised using the resolved volumetric
method and returns its center plane with `preview_region` metadata.

## Effort presets

| Method | Quick | Balanced | Thorough |
| --- | ---: | ---: | ---: |
| BM3D/BM4D | hard-threshold stage | complete `np` profile | complete `high` profile |
| Noise2Fast | 16 features, patience 50 steps | 24 features, patience 100 steps | 64 features, patience 100 steps |
| StructN2V | paper schedule | paper schedule | paper schedule |

ZS-N2N has four presets (contract version 1), with Balanced as the default:

| Effort ID | Hidden width | Middle groups | Updates | Head initialization | Grayscale parameters |
| --- | ---: | ---: | ---: | --- | ---: |
| `quick` | 8 | 1 | 500 | near-zero | 673 |
| `balanced` | 16 | 2 | 1,000 | near-zero | 1,345 |
| `balanced_high` (Balanced-high) | 24 | 1 | 2,000 | native | 5,473 |
| `thorough` | 48 | 1 | 3,000 | native | 21,313 |

Balanced-high is distinct from Balanced and is rejected for other methods.
All four presets batch the two small views in one network call, then make a
full-image call, preserving both symmetric loss terms and one Adam update.
Grouping applies only to the middle convolution; there is no hidden residual
connection. Near-zero initializes only the final weights as `Normal(0, 0.001)`
and its bias as zero; the first two layers retain native initialization.
Quick/Balanced/Balanced-high are explicit JDLL performance variants. Thorough
retains the reference architecture, initialization, budget and objective with
equivalent batched execution and the documented min-max normalization wrapper.

CPU/CUDA/MPS and preview use identical preset settings; production never tunes
thread counts or precision by effort. ZS-N2N defaults to per-channel min-max
normalization with clipping, honoring explicit percentile overrides. Final
weights are used, followed by external residual subtraction, clamping and
intensity restoration. No preset promises a runtime or quality ranking.
`capabilities()['methods']['zs_n2n']` advertises `effort_presets_version=1`
and the four canonical effort IDs. See [docs/API.md](docs/API.md#zs-n2n-effort-contract)
for advanced overrides and the retained unbatched reference path.

Noise2Fast defaults to Balanced. Width is the hidden feature count; Quick and
Balanced use compact-width variants, while Thorough uses the released
64-feature architecture. All devices use the same presets. They retain the
four full-plane checkerboard pairs, batch size 1, BCE, Adam at 0.001, and
full-image validation after every optimizer update. A strictly improved score
resets the counter; training stops after the selected number of consecutive
steps without improvement. Patience is not a total step budget, so uncapped
optimization reports an unknown total (`maximum=0`). Every preset averages the
last `min(completed_steps, 100)` predictions, including Quick with patience 50.

Quick means lower computational effort, not a time guarantee. Thorough means
greater capacity, not guaranteed lower error. Advanced `method_config` overrides
such as `{"hidden_width": 32, "patience": 75}` accept positive integers and are
reported as actually resolved. `steps=0` (default) uses patience stopping; a
positive `steps` is an explicit debug cap, with a warning if it causes stopping.
Noise2Fast defaults to per-channel min-max normalization and honors explicit
normalization overrides. Both `denoise()` and `preview()` use these settings.

StructN2V uses the paper's 2D schedule (1600 256² patches, batch 4, 600 epochs,
Adam at 2e-5) or 3D schedule (1900 32×64×64 patches, batch 10, 100 epochs,
Adam at 4e-6) at every effort level. Its positive explicit `method_config.steps`
is a debug/test cap and is always reported as a non-canonical override.

Noise2Fast's training principle and layer structure follow the
[paper](https://arxiv.org/abs/2108.10209) and the
[authors' implementation](https://github.com/jason-lequyer/Noise2Fast).
The authorized width/patience variants retain the existing JDLL loop; even the
64-feature path does not claim bitwise equivalence to the standalone program.
StructN2V follows the
[ISBI 2020 paper](https://uweschmidt.org/research/pubs/isbi20broaddus.pdf) and
the authors' [N2V implementation](https://github.com/juglab/n2v).

ZS-N2N follows Mansour and Heckel's complete symmetric residual and
full-resolution consistency objective and three-convolution noise-predictor
structure, with the explicit width/group/initialization variants above.
All presets retain Adam and `StepLR(1000, 0.5)`, stepping the scheduler after
each update. Primary references are the
[CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/html/Mansour_Zero-Shot_Noise2Noise_Efficient_Image_Denoising_Without_Any_Data_CVPR_2023_paper.html)
and the [authors' executable Colab](https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b).

## Callbacks, cancellation, and integration

Callbacks are ordinary callables receiving event dictionaries. Callback
exceptions are isolated by default and recorded in result warnings. Cancellation
accepts either a zero-argument callable or an object with `is_cancelled()` and
raises `DenoisingCancelled`; partial arrays are never returned as success.

See [docs/API.md](docs/API.md) for configuration/result details and
[docs/JDLL_INTEGRATION.md](docs/JDLL_INTEGRATION.md) for callback forwarding.

## Development

```bash
python -m pytest
python -m ruff check jdll_denoise tests
python -m mypy jdll_denoise
```

Known v0.1 limitations and migration paths are recorded in
[docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md).
