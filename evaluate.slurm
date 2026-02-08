#!/bin/bash
#SBATCH --job-name=sikubert_eval           # Job name
#SBATCH --output=/media02/ddien02/thanhxuan217/main_src/logs/eval_%j.out  # Output file
#SBATCH --error=/media02/ddien02/thanhxuan217/main_src/logs/eval_%j.err   # Error file
#SBATCH --partition=batch                  # Partition name
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks=1                         # Number of tasks
#SBATCH --cpus-per-task=16                 # CPUs per task
#SBATCH --gres=gpu:2                       # 2 GPUs for parallel inference
#SBATCH --mem=64G                          # Memory per node
#SBATCH --time=8:00:00                     # Time limit
#SBATCH --mail-type=BEGIN,END,FAIL         # Email notifications
#SBATCH --mail-user=xuanhuynh233@gmail.com # Your email

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
mkdir -p "${LOG_DIR}"

# ============================================================================
# SETUP ENVIRONMENT
# ============================================================================

mkdir -p logs outputs

echo "=========================================="
echo "Multi-GPU Evaluation Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Number of GPUs: ${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-2}}"
echo "Start Time: $(date)"
echo "=========================================="

# Activate conda environment
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    source "/opt/conda/etc/profile.d/conda.sh"
fi

if command -v conda >/dev/null 2>&1; then
    conda activate "$CONDA_ENV"
else
    echo "WARNING: conda not found. Make sure your Python env is activated."
fi

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

echo ""
echo "=========================================="
echo "EVALUATION CONFIGURATION"
echo "=========================================="
echo "Task: $TASK"
echo "Model: $MODEL_NAME"
echo "Model Path: $EVAL_MODEL_PATH"
echo "Test Path: $TEST_PATH"
echo ""
echo "CNN Kernels: $CNN_KERNEL_SIZES"
echo "CNN Filters: $CNN_NUM_FILTERS"
echo ""
echo "Output Dir: $OUTPUT_DIR"
echo "=========================================="
echo ""

# ============================================================================
# RUN EVALUATION (Multi-GPU with torchrun)
# ============================================================================

# Number of GPUs
NGPUS=2

torchrun --nproc_per_node=$NGPUS evaluate.py \
    --task $TASK \
    --model_path $EVAL_MODEL_PATH \
    --test_path $TEST_PATH \
    --model_name $MODEL_NAME \
    --max_length $MAX_LENGTH \
    --dropout $DROPOUT \
    --cnn_kernel_sizes $CNN_KERNEL_SIZES \
    --cnn_num_filters $CNN_NUM_FILTERS \
    --head_type $HEAD_TYPE \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    $FP16 \
    $PIN_MEMORY \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR

# ============================================================================
# COMPLETION
# ============================================================================

echo ""
echo "=========================================="
echo "Evaluation completed at: $(date)"
echo "=========================================="
