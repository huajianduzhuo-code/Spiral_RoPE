# Spiral RoPE

Official implementation of **Spiral RoPE: Rotate Your Rotary Positional Embeddings in the 2D Plane**.

[[Paper]](https://arxiv.org/abs/XXXX.XXXXX) [[Hugging Face Models]](https://huggingface.co/haoyuliu00/Spiral_RoPE)

## Overview

This repository contains the official PyTorch implementation of Spiral RoPE, a novel positional encoding method for Vision Transformers that extends Rotary Position Embedding (RoPE) to 2D image domains.

## Implementations

| Directory | Description | Base Codebase |
|-----------|-------------|---------------|
| [RoPE_DeiT](./RoPE_DeiT) | Spiral RoPE for Vision Transformers (Image Classification) | [DeiT](https://github.com/facebookresearch/deit) |
| [rope_dit](./rope_dit) | Spiral RoPE for Diffusion Transformers (Image Generation) | [DiT](https://github.com/facebookresearch/DiT) |

## Pre-trained Models

All pre-trained checkpoints are available at 🤗 [Hugging Face](https://huggingface.co/haoyuliu00/Spiral_RoPE).

### DeiT + Spiral RoPE (Image Classification)

| Model | Resolution | Dataset | Checkpoint |
|-------|------------|---------|------------|
| DeiT-S + Spiral RoPE | 224×224 | ImageNet-1K | [Download](https://huggingface.co/haoyuliu00/Spiral_RoPE/blob/main/deit_small_patch16_LS_rms_rope/checkpoint.pth) |
| DeiT-B + Spiral RoPE | 224×224 | ImageNet-1K | [Download](https://huggingface.co/haoyuliu00/Spiral_RoPE/blob/main/deit_base_patch16_LS_rms_rope/checkpoint.pth) |
| DeiT-L + Spiral RoPE | 224×224 | ImageNet-1K | [Download](https://huggingface.co/haoyuliu00/Spiral_RoPE/blob/main/deit_large_patch16_LS_rms_rope/checkpoint.pth) |

### DiT + Spiral RoPE (Image Generation)

| Model | Training Steps | Dataset | Checkpoint |
|-------|----------------|---------|------------|
| DiT-S/2 + Spiral RoPE | 400K | ImageNet-1K | [Download](https://huggingface.co/haoyuliu00/Spiral_RoPE/blob/main/dit_s2/0400000.pt) |
| DiT-B/2 + Spiral RoPE | 400K | ImageNet-1K | [Download](https://huggingface.co/haoyuliu00/Spiral_RoPE/blob/main/dit_b2/0400000.pt) |
| DiT-L/2 + Spiral RoPE | 400K | ImageNet-1K | [Download](https://huggingface.co/haoyuliu00/Spiral_RoPE/blob/main/dit_l2/0400000.pt) |
| DiT-XL/2 + Spiral RoPE | 400K | ImageNet-1K | [Download](https://huggingface.co/haoyuliu00/Spiral_RoPE/blob/main/dit_xl2/0400000.pt) |
| DiT-XL/2 + Spiral RoPE | 7M | ImageNet-1K | [Download](https://huggingface.co/haoyuliu00/Spiral_RoPE/blob/main/dit_xl2/0700000.pth) |

## Quick Start

### Vision Transformer (Classification)

```bash
cd RoPE_DeiT
# Edit run_all.sh to set DATA_PATH
bash run_all.sh
```

See [RoPE_DeiT/README.md](./RoPE_DeiT/README.md) for detailed instructions.

### Diffusion Transformer (Generation)

```bash
cd rope_dit
# Edit run_all.sh to set DATA_PATH
bash run_all.sh
```

See [rope_dit/README.md](./rope_dit/README.md) for detailed instructions.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{spiral_rope,
  title={Spiral RoPE: Rotate Your Rotary Positional Embeddings in the 2D Plane},
  author={},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## Acknowledgements

This codebase is built upon:
- [DeiT](https://github.com/facebookresearch/deit) by Facebook Research
- [DiT](https://github.com/facebookresearch/DiT) by Facebook Research

## License

This project is released under the [MIT License](LICENSE).
