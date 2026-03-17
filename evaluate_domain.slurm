#!/bin/bash
#SBATCH --job-name=sikubert_domain_eval       # Job name
#SBATCH --output=logs/domain_eval_%j.out      # Output file
#SBATCH --error=logs/domain_eval_%j.err       # Error file
#SBATCH --partition=batch                     # Partition name
#SBATCH --nodes=1                             # Number of nodes
#SBATCH --ntasks=1                            # Number of tasks
#SBATCH --cpus-per-task=16                    # CPUs per task
#SBATCH --gres=gpu:1                          # 1 GPU (stitching is sequential)
#SBATCH --mem=16G                             # Memory per node
#SBATCH --time=24:00:00                       # Time limit
#SBATCH --mail-type=BEGIN,END,FAIL            # Email notifications
#SBATCH --mail-user=xuanhuynh233@gmail.com    # Your email

set -euo pipefail

# ============================================================================
# LOAD CONFIGURATION
# ============================================================================
if [[ ! -f config.slurm ]]; then
    echo "ERROR: config.slurm not found in $(pwd)"
    exit 1
fi

source config.slurm

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Ensure directories exist
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

# ============================================================================
# SETUP ENVIRONMENT
# ============================================================================

echo "=========================================="
echo "Domain-Level Evaluation Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Number of GPUs: ${SLURM_GPUS_ON_NODE:-1}"
echo "Start Time: $(date)"
echo "=========================================="

# Activate conda environment
echo "Activating conda environment..."
source /media02/ddien02/datnd/miniconda3/etc/profile.d/conda.sh
conda activate /media02/ddien02/thanhxuan217/envs/sikubert

echo "Conda environment activated: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Set environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# ============================================================================
# PRINT CONFIGURATION
# ============================================================================

echo ""
echo "GPU Information:"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
else
    echo "nvidia-smi not found"
fi
echo ""

# ---  Domain eval overrides  ---
# Which split to evaluate (default: test)
EVAL_SPLIT="${EVAL_SPLIT:-test}"

# Overlap size (must match parquet creation)
EVAL_OVERLAP="${EVAL_OVERLAP:-128}"

# Max documents to process (empty = all)
EVAL_MAX_DOCS="${EVAL_MAX_DOCS:-}"

# Number of sample outputs per domain
EVAL_SAMPLE_PER_DOMAIN="${EVAL_SAMPLE_PER_DOMAIN:-3}"

echo ""
echo "=========================================="
echo "DOMAIN EVALUATION CONFIGURATION"
echo "=========================================="
echo "Task: $TASK"
echo "Split: $EVAL_SPLIT"
echo "Model: $MODEL_NAME"
echo "Model Path: $EVAL_MODEL_PATH"
echo "Data Dir: $DATA_DIR"
echo ""
echo "Overlap: $EVAL_OVERLAP"
echo "Batch Size: $BATCH_SIZE"
echo "Head Type: $HEAD_TYPE"
echo "Use QLoRA: ${USE_QLORA_FLAG:-disabled}"
echo "Max Docs: ${EVAL_MAX_DOCS:-all}"
echo "Sample per Domain: $EVAL_SAMPLE_PER_DOMAIN"
echo ""
echo "Output Dir: $OUTPUT_DIR"
echo "=========================================="
echo ""

# ============================================================================
# RUN DOMAIN EVALUATION (single GPU — stitching is sequential)
# ============================================================================

# Build tokenizer argument if TOKENIZER_NAME is set
if [[ -n "$TOKENIZER_NAME" ]]; then
    TOKENIZER_ARG="--tokenizer_name $TOKENIZER_NAME"
    echo "Using extended tokenizer: $TOKENIZER_NAME"
else
    TOKENIZER_ARG=""
    echo "Using default tokenizer from model: $MODEL_NAME"
fi

# Build max_docs argument if set
if [[ -n "$EVAL_MAX_DOCS" ]]; then
    MAX_DOCS_ARG="--max_docs $EVAL_MAX_DOCS"
else
    MAX_DOCS_ARG=""
fi

python evaluate_domain.py \
    --task $TASK \
    --split $EVAL_SPLIT \
    --model_path $EVAL_MODEL_PATH \
    --data_dir $DATA_DIR \
    --model_name $MODEL_NAME \
    --max_length $MAX_LENGTH \
    --dropout $DROPOUT \
    --cnn_kernel_sizes $CNN_KERNEL_SIZES \
    --cnn_num_filters $CNN_NUM_FILTERS \
    --head_type $HEAD_TYPE \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --overlap $EVAL_OVERLAP \
    --sample_per_domain $EVAL_SAMPLE_PER_DOMAIN \
    $FP16 \
    $PIN_MEMORY \
    $USE_QLORA_FLAG \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules $LORA_TARGET_MODULES \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR \
    $TOKENIZER_ARG \
    $MAX_DOCS_ARG

# ============================================================================
# COMPLETION
# ============================================================================

echo ""
echo "=========================================="
echo "Domain evaluation completed at: $(date)"
echo "=========================================="
