#!/bin/bash
#SBATCH --job-name=sikubert_cnn           # Job name
#SBATCH --output=logs/slurm_%j.out  # Output file (absolute path)
#SBATCH --error=logs/slurm_%j.err   # Error file (absolute path)
#SBATCH --partition=batch                 # Partition name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=16                # CPUs per task
#SBATCH --gres=gpu:2                      # Number of GPUs (change as needed)
#SBATCH --mem=32G                         # Memory per node (max 64GB per server rules)
#SBATCH --time=24:00:00                   # Time limit (max 48h per server rules)
#SBATCH --mail-type=BEGIN,END,FAIL        # Email notifications
#SBATCH --mail-user=xuanhuynh233@gmail.com # Your email

set -euo pipefail

# ============================================================================
# LOAD CONFIGURATION
# ============================================================================
# Source the configuration file for parameters
if [[ ! -f config.slurm ]]; then
    echo "ERROR: config.slurm not found in $(pwd)"
    exit 1
fi

source config.slurm

# Set output/error files using absolute paths from config
# Note: SLURM doesn't support variables in #SBATCH directives, so we set them here
# and redirect manually if needed, or rely on SLURM_SUBMIT_DIR
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# ============================================================================
# SETUP ENVIRONMENT
# ============================================================================

# Ensure required directories exist
mkdir -p logs outputs models

echo "=========================================="
echo "Multi-GPU Training Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Number of GPUs: ${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}"
echo "Start Time: $(date)"
echo "=========================================="

# Load required modules (adjust based on your cluster)
# module load python/3.11
# module load cuda/11.8
# module load cudnn/8.6

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

# Print GPU information
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
echo "TRAINING CONFIGURATION"
echo "=========================================="
echo "Task: $TASK"
echo "Model: $MODEL_NAME"
echo "Data Directory: $DATA_DIR"
echo ""
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Epochs: $NUM_EPOCHS"
echo "Warmup Ratio: $WARMUP_RATIO"
echo "Weight Decay: $WEIGHT_DECAY"
echo "Dropout: $DROPOUT"
echo ""
echo "CNN Kernels: $CNN_KERNEL_SIZES"
echo "CNN Filters: $CNN_NUM_FILTERS"
echo "Head Type: $HEAD_TYPE"
echo ""
echo "Use QLoRA: ${USE_QLORA_FLAG:-False}"
echo "LoRA Rank: $LORA_R"
echo "LoRA Alpha: $LORA_ALPHA"
echo "LoRA Dropout: $LORA_DROPOUT"
echo "LoRA Target Modules: $LORA_TARGET_MODULES"
echo "Use Streaming: ${USE_STREAMING:-False}"
echo ""
echo "Output Dir: $OUTPUT_DIR"
echo "Model Save Dir: $MODEL_SAVE_DIR"
echo "=========================================="
echo ""

# ============================================================================
# RUN TRAINING
# ============================================================================

# Number of GPUs - hardcoded to 2 (max allowed per server rules)
NGPUS=2

# Build checkpoint argument if path is set
if [[ -n "$CHECKPOINT_PATH" && -d "$CHECKPOINT_PATH" ]]; then
    CHECKPOINT_ARG="--resume_from_checkpoint $CHECKPOINT_PATH"
    echo "Resuming from checkpoint: $CHECKPOINT_PATH"
else
    CHECKPOINT_ARG=""
    echo "No checkpoint specified or checkpoint not found: $CHECKPOINT_PATH"
fi

torchrun --nproc_per_node=$NGPUS train.py \
    --task $TASK \
    --model_name $MODEL_NAME \
    --max_length $MAX_LENGTH \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --warmup_ratio $WARMUP_RATIO \
    --weight_decay $WEIGHT_DECAY \
    --dropout $DROPOUT \
    --max_grad_norm $MAX_GRAD_NORM \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --seed $SEED \
    --num_workers $NUM_WORKERS \
    --data_dir $DATA_DIR \
    --max_steps $MAX_STEPS \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOGGING_STEPS \
    --max_eval_samples $MAX_EVAL_SAMPLES \
    $FP16 \
    $BF16 \
    $PIN_MEMORY \
    $PERSISTENT_WORKERS \
    --cnn_kernel_sizes $CNN_KERNEL_SIZES \
    --cnn_num_filters $CNN_NUM_FILTERS \
    --head_type $HEAD_TYPE \
    $USE_QLORA_FLAG \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules $LORA_TARGET_MODULES \
    --output_dir $OUTPUT_DIR \
    --model_save_dir $MODEL_SAVE_DIR \
    --log_dir $LOG_DIR \
    $CHECKPOINT_ARG

# ============================================================================
# COMPLETION
# ============================================================================

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
