#!/bin/bash
#SBATCH --job-name=sikubert_infer            # Job name
#SBATCH --output=logs/infer_%j.out           # Output file
#SBATCH --error=logs/infer_%j.err            # Error file
#SBATCH --partition=batch                    # Partition name
#SBATCH --nodes=1                            # Number of nodes
#SBATCH --ntasks=1                           # Number of tasks
#SBATCH --cpus-per-task=16                   # CPUs per task
#SBATCH --gres=gpu:1                         # 1 GPU (stitching is sequential)
#SBATCH --mem=16G                            # Memory per node
#SBATCH --time=24:00:00                      # Time limit
#SBATCH --mail-type=BEGIN,END,FAIL           # Email notifications
#SBATCH --mail-user=xuanhuynh233@gmail.com   # Your email

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

# Ensure log directory exists
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

# ============================================================================
# SETUP ENVIRONMENT
# ============================================================================

echo "=========================================="
echo "Sliding Window Inference Job"
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

# ---  Inference-specific overrides  ---
# Which split to run inference on (default: test)
INFER_SPLIT="${INFER_SPLIT:-test}"

# Overlap size used during data chunking (must match the parquet creation)
INFER_OVERLAP="${INFER_OVERLAP:-128}"

echo ""
echo "=========================================="
echo "INFERENCE CONFIGURATION"
echo "=========================================="
echo "Task: $TASK"
echo "Split: $INFER_SPLIT"
echo "Model: $MODEL_NAME"
echo "Model Path: $EVAL_MODEL_PATH"
echo "Data Dir: $DATA_DIR"
echo ""
echo "Overlap: $INFER_OVERLAP"
echo "Batch Size: $BATCH_SIZE"
echo "Head Type: $HEAD_TYPE"
echo "Use QLoRA: ${USE_QLORA_FLAG:-disabled}"
echo ""
echo "Output Dir: $OUTPUT_DIR"
echo "=========================================="
echo ""

# ============================================================================
# RUN INFERENCE (single GPU — stitching is inherently sequential)
# ============================================================================

python infer_sliding_window.py \
    --task $TASK \
    --split $INFER_SPLIT \
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
    --overlap $INFER_OVERLAP \
    $FP16 \
    $PIN_MEMORY \
    $USE_QLORA_FLAG \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules $LORA_TARGET_MODULES \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR

# ============================================================================
# COMPLETION
# ============================================================================

echo ""
echo "=========================================="
echo "Inference completed at: $(date)"
echo "=========================================="
