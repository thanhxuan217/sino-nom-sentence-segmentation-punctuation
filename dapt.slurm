#!/bin/bash
#SBATCH --job-name=sikubert-dapt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/dapt_%j.out
#SBATCH --error=logs/dapt_%j.err

# ============================================================================
# Domain Adaptive Pretraining (DAPT) for SikuBERT
# Two-phase training: Phase 1 (128 tokens) → Phase 2 (512 tokens)
# ============================================================================

set -euo pipefail

# Source configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/dapt_config.slurm"

# Node constraint
#SBATCH --nodelist=${GPU_NODE}

# ============================================================================
# ENVIRONMENT
# ============================================================================
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "GPUs: ${NGPUS}"
echo "=========================================="

# Activate conda
source activate ${CONDA_ENV}
cd ${WORKING_DIR}

# Create output directories
mkdir -p "${PHASE1_OUTPUT}" "${PHASE2_OUTPUT}" "logs"

# ============================================================================
# PHASE 0: DATA PREPARATION (if not already done)
# ============================================================================
PHASE1_DATA="${DAPT_DATA_DIR}/phase1"
PHASE2_DATA="${DAPT_DATA_DIR}/phase2"

if [ ! -d "${PHASE1_DATA}/train" ] || [ -z "$(ls -A ${PHASE1_DATA}/train/ 2>/dev/null)" ]; then
    echo ""
    echo "=========================================="
    echo "PHASE 0a: Preparing Phase 1 data (max_length=${PHASE1_MAX_LENGTH})"
    echo "=========================================="
    python prepare_dapt_data.py \
        --input_path "${RAW_CORPUS_PATH}" \
        --input_format "${RAW_CORPUS_FORMAT}" \
        --output_dir "${PHASE1_DATA}" \
        --tokenizer_path "${TOKENIZER}" \
        --max_length ${PHASE1_MAX_LENGTH} \
        --shard_size ${SHARD_SIZE}
else
    echo "Phase 1 data already exists at ${PHASE1_DATA}, skipping preparation."
fi

if [ ! -d "${PHASE2_DATA}/train" ] || [ -z "$(ls -A ${PHASE2_DATA}/train/ 2>/dev/null)" ]; then
    echo ""
    echo "=========================================="
    echo "PHASE 0b: Preparing Phase 2 data (max_length=${PHASE2_MAX_LENGTH})"
    echo "=========================================="
    python prepare_dapt_data.py \
        --input_path "${RAW_CORPUS_PATH}" \
        --input_format "${RAW_CORPUS_FORMAT}" \
        --output_dir "${PHASE2_DATA}" \
        --tokenizer_path "${TOKENIZER}" \
        --max_length ${PHASE2_MAX_LENGTH} \
        --shard_size ${SHARD_SIZE}
else
    echo "Phase 2 data already exists at ${PHASE2_DATA}, skipping preparation."
fi

# ============================================================================
# PHASE 1: SHORT SEQUENCE WARMUP (max_length=128)
# ============================================================================
PHASE1_FINAL="${PHASE1_OUTPUT}/final_dapt_model"

if [ ! -d "${PHASE1_FINAL}" ]; then
    echo ""
    echo "=========================================="
    echo "PHASE 1: DAPT Warmup (max_length=${PHASE1_MAX_LENGTH})"
    echo "  Model:     ${MODEL_NAME}"
    echo "  LR:        ${PHASE1_LR}"
    echo "  Batch:     ${PHASE1_BATCH_SIZE} × ${NGPUS} × ${PHASE1_GRAD_ACCUM}"
    echo "  Epochs:    ${PHASE1_EPOCHS}"
    echo "=========================================="

    torchrun --nproc_per_node=${NGPUS} --master_port=${MASTER_PORT} \
        dapt_train.py \
        --model_name "${MODEL_NAME}" \
        --tokenizer_path "${TOKENIZER}" \
        --data_dir "${PHASE1_DATA}" \
        --output_dir "${PHASE1_OUTPUT}" \
        --max_length ${PHASE1_MAX_LENGTH} \
        --per_device_batch_size ${PHASE1_BATCH_SIZE} \
        --gradient_accumulation_steps ${PHASE1_GRAD_ACCUM} \
        --learning_rate ${PHASE1_LR} \
        --warmup_ratio ${WARMUP_RATIO} \
        --num_train_epochs ${PHASE1_EPOCHS} \
        --mlm_probability ${MLM_PROBABILITY} \
        --punct_mask_prob ${PUNCT_MASK_PROB} \
        --save_steps ${PHASE1_SAVE_STEPS} \
        --save_total_limit ${SAVE_TOTAL_LIMIT} \
        --logging_steps ${LOGGING_STEPS} \
        --num_workers ${NUM_WORKERS} \
        --seed ${SEED} \
        ${FP16} ${BF16}

    echo "Phase 1 complete. Model saved to: ${PHASE1_FINAL}"
else
    echo "Phase 1 already completed (${PHASE1_FINAL} exists), skipping."
fi

# ============================================================================
# PHASE 2: FULL SEQUENCE DAPT (max_length=512)
# ============================================================================
PHASE2_FINAL="${PHASE2_OUTPUT}/final_dapt_model"

echo ""
echo "=========================================="
echo "PHASE 2: DAPT Main (max_length=${PHASE2_MAX_LENGTH})"
echo "  Model:     ${PHASE1_FINAL}"
echo "  LR:        ${PHASE2_LR}"
echo "  Batch:     ${PHASE2_BATCH_SIZE} × ${NGPUS} × ${PHASE2_GRAD_ACCUM}"
echo "  Epochs:    ${PHASE2_EPOCHS}"
echo "=========================================="

torchrun --nproc_per_node=${NGPUS} --master_port=${MASTER_PORT} \
    dapt_train.py \
    --model_name "${PHASE1_FINAL}" \
    --tokenizer_path "${TOKENIZER}" \
    --data_dir "${PHASE2_DATA}" \
    --output_dir "${PHASE2_OUTPUT}" \
    --max_length ${PHASE2_MAX_LENGTH} \
    --per_device_batch_size ${PHASE2_BATCH_SIZE} \
    --gradient_accumulation_steps ${PHASE2_GRAD_ACCUM} \
    --learning_rate ${PHASE2_LR} \
    --warmup_ratio ${WARMUP_RATIO} \
    --num_train_epochs ${PHASE2_EPOCHS} \
    --mlm_probability ${MLM_PROBABILITY} \
    --punct_mask_prob ${PUNCT_MASK_PROB} \
    --save_steps ${PHASE2_SAVE_STEPS} \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --logging_steps ${LOGGING_STEPS} \
    --num_workers ${NUM_WORKERS} \
    --seed ${SEED} \
    ${FP16} ${BF16}

# ============================================================================
# DONE
# ============================================================================
echo ""
echo "=========================================="
echo "DAPT COMPLETE"
echo "=========================================="
echo "  Phase 1 model: ${PHASE1_FINAL}"
echo "  Phase 2 model: ${PHASE2_FINAL}"
echo ""
echo "  To finetune downstream, update config.slurm:"
echo "    export MODEL_NAME=\"${PHASE2_FINAL}\""
echo ""
echo "Job finished: $(date)"
echo "=========================================="
