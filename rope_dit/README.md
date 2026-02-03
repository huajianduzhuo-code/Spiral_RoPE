# Spiral RoPE for Diffusion Transformers (DiT)

This repository implements **Spiral RoPE** for Diffusion Transformers, based on the [DiT (Scalable Diffusion Models with Transformers)](https://github.com/facebookresearch/DiT) codebase from Facebook Research.

## Pre-trained Models

Pre-trained checkpoints are available at 🤗 [Hugging Face](https://huggingface.co/haoyuliu00/Spiral_RoPE).

| Model | Training Steps | Dataset | Checkpoint |
|-------|----------------|---------|------------|
| DiT-S/2 + Spiral RoPE | 400K | ImageNet-1K | [Download](https://huggingface.co/haoyuliu00/Spiral_RoPE/blob/main/dit_s2/0400000.pt) |
| DiT-B/2 + Spiral RoPE | 400K | ImageNet-1K | [Download](https://huggingface.co/haoyuliu00/Spiral_RoPE/blob/main/dit_b2/0400000.pt) |
| DiT-L/2 + Spiral RoPE | 400K | ImageNet-1K | [Download](https://huggingface.co/haoyuliu00/Spiral_RoPE/blob/main/dit_l2/0400000.pt) |
| DiT-XL/2 + Spiral RoPE | 400K | ImageNet-1K | [Download](https://huggingface.co/haoyuliu00/Spiral_RoPE/blob/main/dit_xl2/0400000.pt) |
| DiT-XL/2 + Spiral RoPE | 7M | ImageNet-1K | [Download](https://huggingface.co/haoyuliu00/Spiral_RoPE/blob/main/dit_xl2/0700000.pth) |

## Quick Start

### 1. Set Data Path

Before running, edit `run_all.sh` to set your ImageNet path:

```bash
DATA_PATH="/path/to/ImageNet/train"  # Change this to your ImageNet training directory
```

### 2. Run Training Pipeline

Use `run_all.sh` to run the complete pipeline (Training → Sample Generation) in one command:

```bash
bash run_all.sh
```

### Command Line Options

```bash
bash run_all.sh [USE_ROPE] [ROTATE] [THETA] [FREQS_FOR] [MAX_FREQ] [TRUNCATE_PERCENTAGE] [MAX_FREQ_LANG] [MIN_FREQ_LANG] [SAME_THETA] [SEED] [MODEL]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `USE_ROPE` | True | Enable RoPE: True/False |
| `ROTATE` | 0 | Rotation parameter for RoPE (Note: `ROTATE` = $K/2$, where $K$ is the number of directions in our paper) |
| `THETA` | 10000 | Base frequency |
| `FREQS_FOR` | lang | Frequency type: pixel/lang |
| `MAX_FREQ` | 10 | Max frequency for pixel mode |
| `TRUNCATE_PERCENTAGE` | 0.0 | Percentage of frequencies to truncate |
| `MAX_FREQ_LANG` | 1.5 | Max frequency for lang mode |
| `MIN_FREQ_LANG` | 0.0 | Min frequency for lang mode |
| `SAME_THETA` | False | Whether different directions share the same frequencies (see below) |
| `SEED` | 0 | Random seed |
| `MODEL` | DiT-B/4 | Model architecture |

### Examples

```bash
# Run with standard RoPE
bash run_all.sh

# Run without RoPE (baseline)
bash run_all.sh False

# Custom RoPE with rotate=2
bash run_all.sh True 2

# Only change rotate=4, keep others default
bash run_all.sh - 4

# Custom seed
bash run_all.sh - - - - - - - - - 123
```

Use `-` to keep the default value for any parameter.

### About `SAME_THETA`

The `SAME_THETA` parameter controls whether different directions share the same rotation frequencies.

- **`SAME_THETA=False` (default, recommended)**: Each direction is assigned independent frequencies. This is the default and recommended setting.

- **`SAME_THETA=True`**: All directions share the same set of frequencies. This naive approach independently assigns $d/(2K)$ frequencies to each direction. However, this limits the number of distinct frequencies as $K$ increases, reducing the model's ability to encode positions at multiple scales.

## Pipeline Stages

The `run_all.sh` script runs two stages:

1. **Stage 1: Training** - Train the DiT model with Spiral RoPE on ImageNet (80 epochs)
2. **Stage 2: Sample Generation** - Generate 50K samples for FID evaluation

## Acknowledgements

This codebase is built upon [DiT](https://github.com/facebookresearch/DiT) by Facebook Research (William Peebles and Saining Xie).
