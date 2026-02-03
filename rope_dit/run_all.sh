#!/bin/bash

# =============================================================================
# DiT Complete Pipeline: Train → Sample
# =============================================================================
# This script integrates training, sample generation, and FID evaluation

set -e  # Exit on error

# =============================================================================
# COMMAND LINE USAGE
# =============================================================================
# Usage: bash run_all.sh [USE_ROPE] [ROTATE] [THETA] [FREQS_FOR] [MAX_FREQ] [TRUNCATE_PERCENTAGE] [MAX_FREQ_LANG] [MIN_FREQ_LANG] [SAME_THETA] [SEED] [MODEL]
# 
# Examples:
#   bash run_all.sh                           # Full pipeline with standard RoPE
#   bash run_all.sh False                     # Full pipeline without RoPE
#   bash run_all.sh True 2                    # Full pipeline with custom RoPE (rotate=2)
#   bash run_all.sh - 4 - - - - - - - 123     # rotate=4, seed=123

# =============================================================================
# HELP MESSAGE
# =============================================================================
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "DiT Complete Pipeline: Train → Sample → Evaluate"
    echo ""
    echo "Usage: bash run_all.sh [USE_ROPE] [ROTATE] [THETA] [FREQS_FOR] [MAX_FREQ] [TRUNCATE_PERCENTAGE] [MAX_FREQ_LANG] [MIN_FREQ_LANG] [SAME_THETA] [SEED] [MODEL]"
    echo ""
    echo "This script runs the complete pipeline:"
    echo "  1. Training (80 epochs, ~22 hours on 8×A100)"
    echo "  2. Sample generation (50K samples for FID)"
    echo ""
    echo "Parameters:"
    echo "  USE_ROPE            Use RoPE: True/False (default: True)"
    echo "  ROTATE              Rotation parameter (default: 0)"
    echo "  THETA               Base frequency (default: 10000)"
    echo "  FREQS_FOR           Frequency type: pixel/lang (default: lang)"
    echo "  MAX_FREQ            Max frequency for pixel mode (default: 10)"
    echo "  TRUNCATE_PERCENTAGE Truncation percentage (default: 0.0)"
    echo "  MAX_FREQ_LANG       Max frequency for lang mode (default: 1.0)"
    echo "  MIN_FREQ_LANG       Min frequency for lang mode (default: 0.0)"
    echo "  SAME_THETA          Use same theta: True/False (default: False)"
    echo "  SEED                Random seed (default: 0)"
    echo "  MODEL               Model architecture (default: DiT-B/4)"
    echo ""
    echo "Examples:"
    echo "  bash run_all.sh                      # Standard RoPE"
    echo "  bash run_all.sh False                # No RoPE"
    echo "  bash run_all.sh True 2               # Custom RoPE (rotate=2)"
    echo "  bash run_all.sh - 4                  # rotate=4"
    echo "  bash run_all.sh - - - - - - - - - 123 # seed=123"
    echo ""
    echo "Note: Use '-' to keep default value for any parameter"
    exit 0
fi

# =============================================================================
# CONFIGURATION
# =============================================================================
USE_ROPE=${1:-"True"}
ROPE_ROTATE=${2:-0}
ROPE_THETA=${3:-10000}
ROPE_FREQS_FOR=${4:-"lang"}
ROPE_MAX_FREQ=${5:-10}
ROPE_TRUNCATE_PERCENTAGE=${6:-0.0}
ROPE_MAX_FREQ_LANG=${7:-1.5}
ROPE_MIN_FREQ_LANG=${8:-0.0}
ROPE_SAME_THETA=${9:-"False"}
SEED=${10:-0}
MODEL=${11:-"DiT-B/4"}

# Handle "-" as "use default"
if [ "$1" = "-" ]; then USE_ROPE="True"; fi
if [ "$2" = "-" ]; then ROPE_ROTATE=0; fi
if [ "$3" = "-" ]; then ROPE_THETA=10000; fi
if [ "$4" = "-" ]; then ROPE_FREQS_FOR="lang"; fi
if [ "$5" = "-" ]; then ROPE_MAX_FREQ=10; fi
if [ "$6" = "-" ]; then ROPE_TRUNCATE_PERCENTAGE=0.0; fi
if [ "$7" = "-" ]; then ROPE_MAX_FREQ_LANG=1.0; fi
if [ "$8" = "-" ]; then ROPE_MIN_FREQ_LANG=0.0; fi
if [ "$9" = "-" ]; then ROPE_SAME_THETA="False"; fi
if [ "${10}" = "-" ]; then SEED=0; fi
if [ "${11}" = "-" ]; then MODEL="DiT-B/4"; fi

# =============================================================================
# PATHS AND SETTINGS
# =============================================================================
DATA_PATH="/path/to/ImageNet/train"
RESULTS_BASE="./results"
WANDB_PROJECT="DiT-ImageNet"

MODEL_PATH=$(echo ${MODEL} | tr '/' '-')

SEED_SUFFIX="_seed${SEED}"

if [ "${USE_ROPE}" = "True" ]; then
    RESULTS_DIR="${RESULTS_BASE}/${MODEL_PATH}/RoPE_rotate${ROPE_ROTATE}_theta${ROPE_THETA}_freqs${ROPE_FREQS_FOR}"
    
    if [ "${ROPE_FREQS_FOR}" = "pixel" ]; then
        RESULTS_DIR="${RESULTS_DIR}_maxfreq${ROPE_MAX_FREQ}"
    else
        RESULTS_DIR="${RESULTS_DIR}_maxfreqlang${ROPE_MAX_FREQ_LANG}_minfreqlang${ROPE_MIN_FREQ_LANG}"
    fi
    
    RESULTS_DIR="${RESULTS_DIR}_trunc${ROPE_TRUNCATE_PERCENTAGE}_sametheta${ROPE_SAME_THETA}${SEED_SUFFIX}"
else
    RESULTS_DIR="${RESULTS_BASE}/${MODEL_PATH}/NoRoPE${SEED_SUFFIX}"
fi

# Sample and evaluation directories
SAMPLE_DIR="${RESULTS_DIR}/samples"

# =============================================================================
# DISPLAY CONFIGURATION
# =============================================================================
echo "=========================================="
echo "DiT Complete Pipeline Configuration"
echo "=========================================="
echo "MODEL: ${MODEL}"
echo "USE_ROPE: ${USE_ROPE}"
if [ "${USE_ROPE}" = "True" ]; then
    echo "ROTATE: ${ROPE_ROTATE}"
    echo "THETA: ${ROPE_THETA}"
    echo "FREQS_FOR: ${ROPE_FREQS_FOR}"
    if [ "${ROPE_FREQS_FOR}" = "pixel" ]; then
        echo "MAX_FREQ (pixel): ${ROPE_MAX_FREQ}"
    fi
    if [ "${ROPE_FREQS_FOR}" = "lang" ]; then
        echo "MAX_FREQ_LANG: ${ROPE_MAX_FREQ_LANG}"
        echo "MIN_FREQ_LANG: ${ROPE_MIN_FREQ_LANG}"
    fi
    echo "TRUNCATE_PERCENTAGE: ${ROPE_TRUNCATE_PERCENTAGE}"
    echo "SAME_THETA: ${ROPE_SAME_THETA}"
fi
echo "SEED: ${SEED}"
echo ""
echo "Pipeline Stages:"
echo "  1. Training (80 epochs)"
echo "  2. Sample generation (50K samples)"
echo ""
echo "Paths:"
echo "  Results: ${RESULTS_DIR}"
echo "  Samples: ${SAMPLE_DIR}"
echo "=========================================="
echo ""

# =============================================================================
# STAGE 1: TRAINING
# =============================================================================
echo "=========================================="
echo "STAGE 1/2: TRAINING"
echo "=========================================="
echo ""

CMD="torchrun --nnodes=1 --nproc_per_node=8 train.py \
    --model ${MODEL} \
    --data-path ${DATA_PATH} \
    --num-classes 1000 \
    --image-size 256 \
    --epochs 80 \
    --global-batch-size 256 \
    --global-seed ${SEED} \
    --vae ema \
    --num-workers 4 \
    --log-every 100 \
    --ckpt-every 50000 \
    --results-dir ${RESULTS_DIR} \
    --use-wandb \
    --wandb-project ${WANDB_PROJECT}"

if [ "${USE_ROPE}" = "True" ]; then
    CMD="${CMD} --use-rope"
    CMD="${CMD} --rope-rotate ${ROPE_ROTATE}"
    CMD="${CMD} --rope-theta ${ROPE_THETA}"
    CMD="${CMD} --rope-freqs-for ${ROPE_FREQS_FOR}"
    CMD="${CMD} --rope-max-freq ${ROPE_MAX_FREQ}"
    CMD="${CMD} --rope-truncate-percentage ${ROPE_TRUNCATE_PERCENTAGE}"
    CMD="${CMD} --rope-max-freq-lang ${ROPE_MAX_FREQ_LANG}"
    CMD="${CMD} --rope-min-freq-lang ${ROPE_MIN_FREQ_LANG}"
    
    if [ "${ROPE_SAME_THETA}" = "True" ]; then
        CMD="${CMD} --rope-same-theta"
    fi
fi

echo "Starting training..."
echo ""
eval ${CMD}

if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi

echo ""
echo "✓ Training completed!"
echo ""

# Find the checkpoint directory
CHECKPOINT_DIR="${RESULTS_DIR}/checkpoints"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Error: Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

# Find the checkpoint with the highest step number
LATEST_CKPT=$(ls -1 "${CHECKPOINT_DIR}"/*.pt 2>/dev/null | sort -V | tail -1)

if [ -z "$LATEST_CKPT" ]; then
    echo "❌ Error: No checkpoints found in $CHECKPOINT_DIR"
    exit 1
fi

echo "Found checkpoint: $LATEST_CKPT"


# =============================================================================
# STAGE 2: SAMPLE GENERATION
# =============================================================================
echo "=========================================="
echo "STAGE 2/2: SAMPLE GENERATION"
echo "=========================================="
echo ""
echo "Generating 50K samples for FID evaluation..."
echo "Checkpoint: $LATEST_CKPT"
echo "Output directory: $SAMPLE_DIR"
echo ""

# Build sample generation command
SAMPLE_CMD="torchrun --nnodes=1 --nproc_per_node=8 sample_ddp.py \
    --model ${MODEL} \
    --ckpt ${LATEST_CKPT} \
    --image-size 256 \
    --num-classes 1000 \
    --cfg-scale 1.0 \
    --num-sampling-steps 250 \
    --num-fid-samples 50000 \
    --per-proc-batch-size 32 \
    --sample-dir ${SAMPLE_DIR} \
    --vae mse \
    --global-seed 100"

# Add RoPE parameters to sample_ddp.py command if model uses RoPE
if [ "${USE_ROPE}" = "True" ]; then
    SAMPLE_CMD="${SAMPLE_CMD} --use-rope"
    SAMPLE_CMD="${SAMPLE_CMD} --rope-rotate ${ROPE_ROTATE}"
    SAMPLE_CMD="${SAMPLE_CMD} --rope-theta ${ROPE_THETA}"
    SAMPLE_CMD="${SAMPLE_CMD} --rope-freqs-for ${ROPE_FREQS_FOR}"
    SAMPLE_CMD="${SAMPLE_CMD} --rope-max-freq ${ROPE_MAX_FREQ}"
    SAMPLE_CMD="${SAMPLE_CMD} --rope-truncate-percentage ${ROPE_TRUNCATE_PERCENTAGE}"
    SAMPLE_CMD="${SAMPLE_CMD} --rope-max-freq-lang ${ROPE_MAX_FREQ_LANG}"
    SAMPLE_CMD="${SAMPLE_CMD} --rope-min-freq-lang ${ROPE_MIN_FREQ_LANG}"
    
    if [ "${ROPE_SAME_THETA}" = "True" ]; then
        SAMPLE_CMD="${SAMPLE_CMD} --rope-same-theta"
    fi
fi

echo "Running sample generation..."
eval ${SAMPLE_CMD}

if [ $? -ne 0 ]; then
    echo "❌ Sample generation failed!"
    exit 1
fi

echo ""
echo "✓ Sample generation completed!"
echo ""

# Find the generated .npz file (automatically detect the latest one)
SAMPLE_NPZ=$(ls -1t "${SAMPLE_DIR}"/*.npz 2>/dev/null | head -1)

if [ -z "$SAMPLE_NPZ" ]; then
    echo "❌ Error: No .npz files found in $SAMPLE_DIR"
    exit 1
fi

echo "Found generated sample file: $(basename $SAMPLE_NPZ)"

echo "Sample file: $SAMPLE_NPZ"
FILE_SIZE=$(du -h "$SAMPLE_NPZ" | cut -f1)
echo "File size: $FILE_SIZE"
echo ""
