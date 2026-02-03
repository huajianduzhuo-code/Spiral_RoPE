# Spiral RoPE for Vision Transformers

This repository implements **Spiral RoPE** for Vision Transformers, based on the [DeiT (Data-efficient Image Transformers)](https://github.com/facebookresearch/deit) codebase from Facebook Research.

## Pre-trained Models

Pre-trained checkpoints are available at 🤗 [Hugging Face](https://huggingface.co/haoyuliu00/Spiral_RoPE).

| Model | Resolution | Dataset | Checkpoint |
|-------|------------|---------|------------|
| DeiT-S + Spiral RoPE | 224×224 | ImageNet-1K | [Download](https://huggingface.co/haoyuliu00/Spiral_RoPE/blob/main/deit_small_patch16_LS_rms_rope/checkpoint.pth) |
| DeiT-B + Spiral RoPE | 224×224 | ImageNet-1K | [Download](https://huggingface.co/haoyuliu00/Spiral_RoPE/blob/main/deit_base_patch16_LS_rms_rope/checkpoint.pth) |
| DeiT-L + Spiral RoPE | 224×224 | ImageNet-1K | [Download](https://huggingface.co/haoyuliu00/Spiral_RoPE/blob/main/deit_large_patch16_LS_rms_rope/checkpoint.pth) |

## Quick Start

### 1. Set Data Path

Before running, edit `run_all.sh` to set your ImageNet path:

```bash
DATA_PATH="/path/to/ImageNet"  # Change this to your ImageNet directory
```

### 2. Run Training Pipeline

Use `run_all.sh` to run the complete training pipeline (Training → Finetuning → Evaluation) in one command:

```bash
cd RoPE_DeiT
bash run_all.sh
```

### Command Line Options

```bash
bash run_all.sh [ROTATE] [TRUNCATE_PERCENTAGE] [MAX_FREQ_LANG] [MIN_FREQ_LANG] [SAME_THETA] [NPROC_PER_NODE] [EVAL_ONLY] [SKIP_TRAINING]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ROTATE` | 2 | Rotation parameter for RoPE (Note: `ROTATE` = $K/2$, where $K$ is the number of directions in our paper) |
| `TRUNCATE_PERCENTAGE` | 0.0 | Percentage of frequencies to truncate |
| `MAX_FREQ_LANG` | 1.5 | Max frequency for language RoPE |
| `MIN_FREQ_LANG` | 0.0 | Min frequency for language RoPE |
| `SAME_THETA` | False | Whether different directions share the same frequencies (see below) |
| `NPROC_PER_NODE` | 4 | Number of GPUs |
| `EVAL_ONLY` | False | Skip training/finetuning, only evaluate |
| `SKIP_TRAINING` | False | Skip training, only finetune + evaluate |

### Examples

```bash
# Run with all defaults (4 GPUs, rotate=2)
bash run_all.sh

# Use 8 GPUs with rotate=4
bash run_all.sh 4 - - - - 8

# Only run evaluation (skip training and finetuning)
bash run_all.sh - - - - - - True

# Only run finetuning + evaluation (skip training)
bash run_all.sh - - - - - - False True
```

Use `-` to keep the default value for any parameter.

### About `SAME_THETA`

The `SAME_THETA` parameter controls whether different directions share the same rotation frequencies.

- **`SAME_THETA=False` (default, recommended)**: Each direction is assigned independent frequencies. This is the default and recommended setting.

- **`SAME_THETA=True`**: All directions share the same set of frequencies. This naive approach independently assigns $d/(2K)$ frequencies to each direction. However, this limits the number of distinct frequencies as $K$ increases, reducing the model's ability to encode positions at multiple scales.

## Pipeline Stages

The `run_all.sh` script runs three stages:

1. **Stage 1: Training** - Pre-train the model with RoPE on ImageNet (300 epochs)
2. **Stage 2: Finetuning** - Finetune the pre-trained model (20 epochs)
3. **Stage 3: Evaluation** - Evaluate with multiple configurations (crop ratio, EMA, etc.)

## Acknowledgements

This codebase is built upon [DeiT](https://github.com/facebookresearch/deit) by Facebook Research.
