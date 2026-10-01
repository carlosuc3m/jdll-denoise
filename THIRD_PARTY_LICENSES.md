# Third-party licensing

Original source in this repository is Apache-2.0. Python dependencies retain
their own licenses and notices.

## Optional BM3D and BM4D

The adapters in `jdll_denoise.methods.bm` dynamically import the official
`bm3d` or `bm4d` package. No source, binary, model, or profile from those
packages is included here.

Tampere University distributes those packages for non-commercial
scientific/educational use. This is more restrictive than Apache-2.0. Before
installing `jdll-denoise[bm]`, users and distributors must review the official
terms at <https://webpages.tuni.fi/foi/GCF-BM3D/>. `capabilities()` reports the
packages independently, including versions and the restriction. An unavailable
adapter fails with installation and licensing guidance; it is never silently
replaced by a differently named filter.

Before a bundled JDLL release, maintainers must obtain compatible permission,
require separate user installation/acceptance, or replace the backend only
after a legally and scientifically reviewed implementation is available.

## Neural implementations

The ZS-N2N, Noise2Fast, and StructN2V code in this repository is an original,
maintainable PyTorch implementation informed by the cited papers/projects. It
does not vendor their research repositories:

- ZS-N2N paper: <https://openaccess.thecvf.com/content/CVPR2023/html/Mansour_Zero-Shot_Noise2Noise_Efficient_Image_Denoising_Without_Any_Data_CVPR_2023_paper.html>
- Authors' ZS-N2N executable Colab: <https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b>
- Noise2Fast reference: <https://github.com/jason-lequyer/Noise2Fast>
- Noise2Void reference: <https://github.com/juglab/n2v>

The ZS-N2N implementation is independently written from the published equations
and executable reference; authors' source code is not copied into this project.
