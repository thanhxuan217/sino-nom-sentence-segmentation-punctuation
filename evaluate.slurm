#!/bin/bash
#SBATCH --job-name=sikubert_eval           # Job name
#SBATCH --output=logs/eval_%j.out  # Output file
#SBATCH --error=logs/eval_%j.err   # Error file
#SBATCH --partition=batch                  # Partition name
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks=1                         # Number of tasks
#SBATCH --cpus-per-task=16                 # CPUs per task
#SBATCH --gres=gpu:2                       # 2 GPUs for parallel inference
#SBATCH --mem=16G                          # Memory per node
#SBATCH --time=10:00:00                     # Time limit
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

echo ""
echo "=========================================="
echo "EVALUATION CONFIGURATION"
echo "=========================================="
echo "Task: $TASK"
echo "Test Split: ${TEST_SPLIT:-test}"
echo "Model: $MODEL_NAME"
echo "Model Path: $EVAL_MODEL_PATH"
echo "Data Dir: $DATA_DIR"
echo ""
echo "CNN Kernels: $CNN_KERNEL_SIZES"
echo "CNN Filters: $CNN_NUM_FILTERS"
echo "Head Type: $HEAD_TYPE"
echo ""
echo "Use QLoRA: ${USE_QLORA_FLAG:-disabled}"
echo "LoRA Rank: $LORA_R"
echo "LoRA Alpha: $LORA_ALPHA"
echo "LoRA Dropout: $LORA_DROPOUT"
echo "LoRA Target Modules: $LORA_TARGET_MODULES"
echo ""
echo "Output Dir: $OUTPUT_DIR"
echo "=========================================="
echo ""

# ============================================================================
# RUN EVALUATION (Multi-GPU with torchrun)
# ============================================================================

# Number of GPUs
NGPUS=2

torchrun --nproc_per_node=$NGPUS --master_port=$MASTER_PORT evaluate.py \
    --task $TASK \
    --test_split ${TEST_SPLIT:-test} \
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
echo "Evaluation completed at: $(date)"
echo "=========================================="
