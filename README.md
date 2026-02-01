# PanoDreamer

> PanoDreamer: Optimization-Based Single Image to 360° 3D Scene With Diffusion  
> [Avinash Paliwal](http://avinashpaliwal.com/),
> [Xilong Zhou](https://xilongzhou.github.io/), 
> [Andrii Tsarov](https://www.linkedin.com/in/andrii-tsarov-b8a9bb13), 
> [Nima Khademi Kalantari](http://nkhademi.com/)

[![arXiv](https://img.shields.io/badge/arXiv-2412.04827-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2412.04827)
[![ACM](https://img.shields.io/badge/ACM-Paper-blue)](https://dl.acm.org/doi/full/10.1145/3757377.3763883)
[![Project Page](https://img.shields.io/badge/PanoDreamer-Website-blue?logo=googlechrome&logoColor=blue)](https://people.engr.tamu.edu/nimak/Papers/PanoDreamer/index.html)
[![Video](https://img.shields.io/badge/YouTube-Video-c4302b?logo=youtube&logoColor=red)](https://youtu.be/EyVfFCg4aF8)

<p align="center">
  <a href="">
    <img src="assets/banner.gif?raw=true" alt="demo" width="100%">
  </a>
</p>

## Overview

This repository implements **MultiConDiffusion** for image-to-panorama generation - an iterative inpainting approach that extends a single input image into a full panorama.

### Implementation Status

- [x] MultiConDiffusion panorama generation (perspective)
- [x] Cylindrical panorama generation
- [ ] Depth estimation
- [ ] 3D Gaussian Splatting (3DGS) scene creation

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Usage

### MultiConDiffusion (Perspective Panorama)
```bash
python multicondiffusion.py \
  --prompt_file examples/29_real_campus_3.txt \
  --input_image examples/29_real_campus_3.png \
  --output_dir output
```

### Cylindrical Panorama
```bash
python multicondiffusion_perspective.py \
  --prompt_file examples/29_real_campus_3.txt \
  --input_image examples/29_real_campus_3.png \
  --output_dir output
```

### Arguments
- `--prompt_file`: Text file with scene description
- `--input_image`: Input image (placed in center)
- `--steps`: Denoising steps per iteration (default: 50)
- `--iterations`: Number of refinement iterations (default: 15)
- `--H`, `--W`: Output dimensions (default: 512x2048)
- `--guidance`: Guidance scale (default: 7.5)
- `--seed`: Random seed (default: 0)
- `--debug`: Save debug visualizations

## Citation

```bibtex
@inproceedings{paliwal2024panodreamer,
    author = {Paliwal, Avinash and Zhou, Xilong and Tsarov, Andrii and Kalantari, Nima},
    title = {PanoDreamer: Optimization-Based Single Image to 360° 3D Scene With Diffusion},
    year = {2025},
    booktitle = {Proceedings of the SIGGRAPH Asia 2025 Conference Papers},
    articleno = {112},
    numpages = {10},
    doi = {10.1145/3757377.3763883},
    url = {https://doi.org/10.1145/3757377.3763883}
}
```
