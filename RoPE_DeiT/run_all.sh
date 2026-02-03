#!/bin/bash

# =============================================================================
# COMMAND LINE USAGE
# =============================================================================
# Usage: bash run_all.sh [ROTATE] [TRUNCATE_PERCENTAGE] [MAX_FREQ_LANG] [MIN_FREQ_LANG] [SAME_THETA] [NPROC_PER_NODE] [EVAL_ONLY] [SKIP_TRAINING]
# 
# Examples:
#   bash run_all.sh                        # Use default values
#   bash run_all.sh 4                      # rotate=4, others default
#   bash run_all.sh 4 0.1                  # rotate=4, truncate_percentage=0.1, others default
#   bash run_all.sh 4 0.1 1.0              # rotate=4, truncate_percentage=0.1, max_freq_lang=1.0, others default
#   bash run_all.sh 4 0.1 1.0 0.01         # + min_freq_lang=0.01, others default
#   bash run_all.sh 4 0.1 1.0 0.01 True    # + same_theta=True, nproc default
#   bash run_all.sh 4 0.1 1.0 0.01 True 8  # All specified
#   bash run_all.sh 4 0.1 1.0 0.01 True 8 True # All specified + eval_only=True
#   bash run_all.sh 4 0.1 1.0 0.01 True 8 False True # + skip_training=True (finetune+eval only)
#   bash run_all.sh - - - 8                # Only specify nproc_per_node=8, others default
#   bash run_all.sh - - - - - True         # Only run evaluation (skip train/finetune)
#   bash run_all.sh - - - - - False True   # Only run finetune+evaluation (skip training)

# =============================================================================
# HELP MESSAGE
# =============================================================================
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "RoPE DeiT Unified Training Script"
    echo ""
    echo "Usage: bash run_all.sh [ROTATE] [TRUNCATE_PERCENTAGE] [MAX_FREQ_LANG] [MIN_FREQ_LANG] [SAME_THETA] [NPROC_PER_NODE] [EVAL_ONLY] [SKIP_TRAINING]"
    echo ""
    echo "Parameters:"
    echo "  ROTATE              Rotation parameter (default: 2)"
    echo "  TRUNCATE_PERCENTAGE Percentage of frequencies to truncate to zero (default: 0.0)"
    echo "  MAX_FREQ_LANG       Max frequency for language RoPE (default: 0.5)"
    echo "  MIN_FREQ_LANG       Min frequency for language RoPE (default: 0.0)"
    echo "  SAME_THETA          Same theta logic: True/False (default: False)"
    echo "  NPROC_PER_NODE      Number of GPUs (default: 4)"
    echo "  EVAL_ONLY           Skip training/finetuning, only run evaluation: True/False (default: False)"
    echo "  SKIP_TRAINING       Skip training, only run finetuning+evaluation: True/False (default: False)"
    echo ""
    echo "Examples:"
    echo "  bash run_all.sh                        # Use all defaults"
    echo "  bash run_all.sh 4                      # rotate=4, others default"
    echo "  bash run_all.sh 4 0.1                  # rotate=4, truncate_percentage=0.1"
    echo "  bash run_all.sh 4 0.1 1.0              # + max_freq_lang=1.0"
    echo "  bash run_all.sh 4 0.1 1.0 0.01         # + min_freq_lang=0.01"
    echo "  bash run_all.sh 4 0.1 1.0 0.01 True    # + same_theta=True"
    echo "  bash run_all.sh 4 0.1 1.0 0.01 True 8  # + nproc_per_node=8"
    echo "  bash run_all.sh 4 0.1 1.0 0.01 True 8 True # + eval_only=True"
    echo "  bash run_all.sh 4 0.1 1.0 0.01 True 8 False True # + skip_training=True"
    echo "  bash run_all.sh - - - 8                # Only change nproc_per_node=8"
    echo "  bash run_all.sh - - - - - True         # Only run evaluation"
    echo "  bash run_all.sh - - - - - False True   # Only run finetune+evaluation"
    echo ""
    echo "Note: Use '-' to keep default value for any parameter"
    exit 0
fi

# =============================================================================
# CONFIGURATION - Command line args or default values
# =============================================================================
ROTATE=${1:-2}
TRUNCATE_PERCENTAGE=${2:-0.0}
MAX_FREQ_LANG=${3:-1.5}
MIN_FREQ_LANG=${4:-0.0}
SAME_THETA=${5:-"False"}
NPROC_PER_NODE=${6:-4}
EVAL_ONLY=${7:-"False"}
SKIP_TRAINING=${8:-"False"}

# Handle "-" as "use default" for any parameter
if [ "$1" = "-" ]; then ROTATE=2; fi
if [ "$2" = "-" ]; then TRUNCATE_PERCENTAGE=0.0; fi
if [ "$3" = "-" ]; then MAX_FREQ_LANG=0.5; fi
if [ "$4" = "-" ]; then MIN_FREQ_LANG=0.0; fi
if [ "$5" = "-" ]; then SAME_THETA="False"; fi
if [ "$6" = "-" ]; then NPROC_PER_NODE=4; fi
if [ "$7" = "-" ]; then EVAL_ONLY="False"; fi
if [ "$8" = "-" ]; then SKIP_TRAINING="False"; fi

# =============================================================================
# CALCULATED SETTINGS (automatically adjusted based on NPROC_PER_NODE)
# =============================================================================
# Calculate accumulation iterations to maintain same effective batch size
# Training: target effective batch = 4096 (original: 4 GPUs × 256 batch × 4 accum_iter)
# Finetuning: target effective batch = 512 (original: 4 GPUs × 64 batch × 2 accum_iter)
TRAIN_ACCUM_ITER=$((4096 / (NPROC_PER_NODE * 256)))
FINETUNE_ACCUM_ITER=$((512 / (NPROC_PER_NODE * 64)))

# Ensure minimum accum_iter of 1
if [ $TRAIN_ACCUM_ITER -lt 1 ]; then TRAIN_ACCUM_ITER=1; fi
if [ $FINETUNE_ACCUM_ITER -lt 1 ]; then FINETUNE_ACCUM_ITER=1; fi

# =============================================================================
# PATHS AND SETTINGS (usually don't need to change)
# =============================================================================
MODEL="deit_base_patch16_LS_rms_rope"
DATA_PATH="/path/to/ImageNet"
OUTPUT_BASE="./outputs/RoPE_output_rotate${ROTATE}_trunc${TRUNCATE_PERCENTAGE}_maxfreq${MAX_FREQ_LANG}_minfreq${MIN_FREQ_LANG}_sametheta${SAME_THETA}"

# =============================================================================
# DISPLAY CONFIGURATION
# =============================================================================
echo "=========================================="
echo "RoPE DeiT Experiment Configuration"
echo "=========================================="
echo "ROTATE: ${ROTATE}"
echo "TRUNCATE_PERCENTAGE: ${TRUNCATE_PERCENTAGE}"
echo "MAX_FREQ_LANG: ${MAX_FREQ_LANG}"
echo "MIN_FREQ_LANG: ${MIN_FREQ_LANG}"
echo "SAME_THETA: ${SAME_THETA}"
echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "EVAL_ONLY: ${EVAL_ONLY}"
echo "SKIP_TRAINING: ${SKIP_TRAINING}"
echo ""
if [ "${EVAL_ONLY}" = "True" ]; then
    echo "Mode: EVALUATION ONLY (skipping training and finetuning)"
elif [ "${SKIP_TRAINING}" = "True" ]; then
    echo "Mode: FINETUNE + EVALUATION ONLY (skipping training)"
    echo "Calculated:"
    echo "  Finetuning accum_iter: ${FINETUNE_ACCUM_ITER} (effective batch: $((NPROC_PER_NODE * 64 * FINETUNE_ACCUM_ITER)))"
else
    echo "Mode: FULL PIPELINE (training + finetuning + evaluation)"
    echo "Calculated:"
    echo "  Training accum_iter: ${TRAIN_ACCUM_ITER} (effective batch: $((NPROC_PER_NODE * 256 * TRAIN_ACCUM_ITER)))"
    echo "  Finetuning accum_iter: ${FINETUNE_ACCUM_ITER} (effective batch: $((NPROC_PER_NODE * 64 * FINETUNE_ACCUM_ITER)))"
fi
echo ""
echo "Output directory: ${OUTPUT_BASE}"
echo "=========================================="

if [ "${EVAL_ONLY}" != "True" ] && [ "${SKIP_TRAINING}" != "True" ]; then
    # =============================================================================
    # STAGE 1: TRAINING
    # =============================================================================
    echo ""
    echo "=========================================="
    echo "STAGE 1: TRAINING (rotate=${ROTATE})"
    echo "Settings: ${NPROC_PER_NODE} GPUs, accum_iter=${TRAIN_ACCUM_ITER}"
    echo "Effective batch size: $((NPROC_PER_NODE * 256 * TRAIN_ACCUM_ITER))"
    echo "=========================================="

    torchrun --nproc_per_node ${NPROC_PER_NODE} --rdzv-backend=c10d --rdzv-endpoint=localhost:7558 main.py \
        --model ${MODEL} \
        --data-path ${DATA_PATH} \
        --output_dir ${OUTPUT_BASE}/base \
        --batch 256 --accum_iter ${TRAIN_ACCUM_ITER} --lr 2e-4 --weight-decay 0.3 --rotate ${ROTATE} --truncate-percentage ${TRUNCATE_PERCENTAGE} \
        --max-freq-lang ${MAX_FREQ_LANG} --min-freq-lang ${MIN_FREQ_LANG} $([ "${SAME_THETA}" = "True" ] && echo "--same-theta" || echo "--no-same-theta") \
        --beta1 0.95 --dist-eval --warmup-epochs 20 --model-ema-decay 0.9999 \
        --no-repeated-aug --ThreeAugment --drop-path 0.1 --epochs 300

    if [ $? -ne 0 ]; then
        echo "Training failed!"
        exit 1
    fi

    echo "Training completed!"
fi

if [ "${EVAL_ONLY}" != "True" ]; then
    # =============================================================================
    # STAGE 2: FINETUNING
    # =============================================================================
    echo "=========================================="
    echo "STAGE 2: FINETUNING (rotate=${ROTATE})"
    echo "Settings: ${NPROC_PER_NODE} GPUs, accum_iter=${FINETUNE_ACCUM_ITER}"
    echo "Effective batch size: $((NPROC_PER_NODE * 64 * FINETUNE_ACCUM_ITER))"
    echo "=========================================="

    torchrun --nproc_per_node ${NPROC_PER_NODE} --master_port 7559 main.py \
        --model ${MODEL} \
        --data-path ${DATA_PATH} \
        --output_dir ${OUTPUT_BASE}/base/ft \
        --finetune ${OUTPUT_BASE}/base/checkpoint.pth \
        --batch 64 --accum_iter ${FINETUNE_ACCUM_ITER} --lr 1e-5 --weight-decay 0.1 --unscale-lr --rotate ${ROTATE} --truncate-percentage ${TRUNCATE_PERCENTAGE} \
        --max-freq-lang ${MAX_FREQ_LANG} --min-freq-lang ${MIN_FREQ_LANG} $([ "${SAME_THETA}" = "True" ] && echo "--same-theta" || echo "--no-same-theta") \
        --reprob 0.0 --smoothing 0.1 --no-repeated-aug \
        --aa rand-m9-mstd0.5-inc1 --epochs 20 --drop-path 0.1 \
        --dist-eval --load_ema

    if [ $? -ne 0 ]; then
        echo "Finetuning failed!"
        exit 1
    fi

    echo "Finetuning completed!"
fi

# =============================================================================
# STAGE 3: EVALUATION
# =============================================================================
echo "=========================================="
echo "STAGE 3: EVALUATION (rotate=${ROTATE})"
echo "Settings: ${NPROC_PER_NODE} GPUs"
echo "=========================================="

# Eval 1: Best checkpoint
torchrun --nproc_per_node ${NPROC_PER_NODE} --rdzv-backend=c10d --rdzv-endpoint=localhost:7560 main.py \
    --model ${MODEL} \
    --data-path ${DATA_PATH} \
    --output_dir ${OUTPUT_BASE}/eval \
    --eval --batch 67 --dist-eval --disable_wandb --rotate ${ROTATE} \
    --max-freq-lang ${MAX_FREQ_LANG} --min-freq-lang ${MIN_FREQ_LANG} $([ "${SAME_THETA}" = "True" ] && echo "--same-theta" || echo "--no-same-theta") \
    --finetune ${OUTPUT_BASE}/base/ft/best_checkpoint.pth

# Eval 2: Best checkpoint with full crop
torchrun --nproc_per_node ${NPROC_PER_NODE} --rdzv-backend=c10d --rdzv-endpoint=localhost:7561 main.py \
    --model ${MODEL} \
    --data-path ${DATA_PATH} \
    --output_dir ${OUTPUT_BASE}/eval --eval-crop-ratio 1.0 \
    --eval --batch 67 --dist-eval --disable_wandb --rotate ${ROTATE} \
    --max-freq-lang ${MAX_FREQ_LANG} --min-freq-lang ${MIN_FREQ_LANG} $([ "${SAME_THETA}" = "True" ] && echo "--same-theta" || echo "--no-same-theta") \
    --finetune ${OUTPUT_BASE}/base/ft/best_checkpoint.pth

# Eval 3: Best checkpoint with EMA
torchrun --nproc_per_node ${NPROC_PER_NODE} --rdzv-backend=c10d --rdzv-endpoint=localhost:7562 main.py \
    --model ${MODEL} \
    --data-path ${DATA_PATH} \
    --output_dir ${OUTPUT_BASE}/eval \
    --eval --batch 67 --dist-eval --disable_wandb --rotate ${ROTATE} \
    --max-freq-lang ${MAX_FREQ_LANG} --min-freq-lang ${MIN_FREQ_LANG} $([ "${SAME_THETA}" = "True" ] && echo "--same-theta" || echo "--no-same-theta") \
    --finetune ${OUTPUT_BASE}/base/ft/best_checkpoint.pth --load_ema

# Eval 4: Best checkpoint with full crop and EMA
torchrun --nproc_per_node ${NPROC_PER_NODE} --rdzv-backend=c10d --rdzv-endpoint=localhost:7563 main.py \
    --model ${MODEL} \
    --data-path ${DATA_PATH} \
    --output_dir ${OUTPUT_BASE}/eval --eval-crop-ratio 1.0 \
    --eval --batch 67 --dist-eval --disable_wandb --rotate ${ROTATE} \
    --max-freq-lang ${MAX_FREQ_LANG} --min-freq-lang ${MIN_FREQ_LANG} $([ "${SAME_THETA}" = "True" ] && echo "--same-theta" || echo "--no-same-theta") \
    --finetune ${OUTPUT_BASE}/base/ft/best_checkpoint.pth --load_ema

# Eval 5: checkpoint
torchrun --nproc_per_node ${NPROC_PER_NODE} --rdzv-backend=c10d --rdzv-endpoint=localhost:7560 main.py \
    --model ${MODEL} \
    --data-path ${DATA_PATH} \
    --output_dir ${OUTPUT_BASE}/eval \
    --eval --batch 67 --dist-eval --disable_wandb --rotate ${ROTATE} \
    --max-freq-lang ${MAX_FREQ_LANG} --min-freq-lang ${MIN_FREQ_LANG} $([ "${SAME_THETA}" = "True" ] && echo "--same-theta" || echo "--no-same-theta") \
    --finetune ${OUTPUT_BASE}/base/ft/checkpoint.pth

# Eval 6: checkpoint with full crop
torchrun --nproc_per_node ${NPROC_PER_NODE} --rdzv-backend=c10d --rdzv-endpoint=localhost:7561 main.py \
    --model ${MODEL} \
    --data-path ${DATA_PATH} \
    --output_dir ${OUTPUT_BASE}/eval --eval-crop-ratio 1.0 \
    --eval --batch 67 --dist-eval --disable_wandb --rotate ${ROTATE} \
    --max-freq-lang ${MAX_FREQ_LANG} --min-freq-lang ${MIN_FREQ_LANG} $([ "${SAME_THETA}" = "True" ] && echo "--same-theta" || echo "--no-same-theta") \
    --finetune ${OUTPUT_BASE}/base/ft/checkpoint.pth

# Eval 7: checkpoint with EMA
torchrun --nproc_per_node ${NPROC_PER_NODE} --rdzv-backend=c10d --rdzv-endpoint=localhost:7562 main.py \
    --model ${MODEL} \
    --data-path ${DATA_PATH} \
    --output_dir ${OUTPUT_BASE}/eval \
    --eval --batch 67 --dist-eval --disable_wandb --rotate ${ROTATE} \
    --max-freq-lang ${MAX_FREQ_LANG} --min-freq-lang ${MIN_FREQ_LANG} $([ "${SAME_THETA}" = "True" ] && echo "--same-theta" || echo "--no-same-theta") \
    --finetune ${OUTPUT_BASE}/base/ft/checkpoint.pth --load_ema

# Eval 8: checkpoint with full crop and EMA
torchrun --nproc_per_node ${NPROC_PER_NODE} --rdzv-backend=c10d --rdzv-endpoint=localhost:7563 main.py \
    --model ${MODEL} \
    --data-path ${DATA_PATH} \
    --output_dir ${OUTPUT_BASE}/eval --eval-crop-ratio 1.0 \
    --eval --batch 67 --dist-eval --disable_wandb --rotate ${ROTATE} \
    --max-freq-lang ${MAX_FREQ_LANG} --min-freq-lang ${MIN_FREQ_LANG} $([ "${SAME_THETA}" = "True" ] && echo "--same-theta" || echo "--no-same-theta") \
    --finetune ${OUTPUT_BASE}/base/ft/checkpoint.pth --load_ema

echo "All evaluations completed!"

# =============================================================================
# SUMMARY
# =============================================================================
echo "=========================================="
if [ "${EVAL_ONLY}" = "True" ]; then
    echo "EVALUATION COMPLETED!"
elif [ "${SKIP_TRAINING}" = "True" ]; then
    echo "FINETUNE + EVALUATION COMPLETED!"
else
    echo "ALL STAGES COMPLETED!"
fi
echo "=========================================="
echo "Settings used:"
echo "  ROTATE: ${ROTATE}"
echo "  TRUNCATE_PERCENTAGE: ${TRUNCATE_PERCENTAGE}"
echo "  MAX_FREQ_LANG: ${MAX_FREQ_LANG}"
echo "  MIN_FREQ_LANG: ${MIN_FREQ_LANG}"
echo "  SAME_THETA: ${SAME_THETA}"
echo "  NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "  EVAL_ONLY: ${EVAL_ONLY}"
echo "  SKIP_TRAINING: ${SKIP_TRAINING}"
echo ""
if [ "${EVAL_ONLY}" != "True" ]; then
    echo "Calculated settings:"
    if [ "${SKIP_TRAINING}" != "True" ]; then
        echo "  Training accum_iter: ${TRAIN_ACCUM_ITER} (effective batch: $((NPROC_PER_NODE * 256 * TRAIN_ACCUM_ITER)))"
    fi
    echo "  Finetuning accum_iter: ${FINETUNE_ACCUM_ITER} (effective batch: $((NPROC_PER_NODE * 64 * FINETUNE_ACCUM_ITER)))"
    echo ""
fi
echo "Results saved in: ${OUTPUT_BASE}"
