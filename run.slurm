#!/bin/bash
#SBATCH --job-name=sikubert_cnn           # Job name
#SBATCH --output=logs/slurm_%j.out        # Output file (%j = job ID)
#SBATCH --error=logs/slurm_%j.err         # Error file
#SBATCH --partition=gpu                   # Partition name (change as needed)
#SBATCH --nodelist=gpu01                  # Specific GPU node (must match where data is stored on /raid!)
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=16                # CPUs per task
#SBATCH --gres=gpu:2                      # Number of GPUs (change as needed)
#SBATCH --mem=64G                         # Memory per node (max 64GB per server rules)
#SBATCH --time=48:00:00                   # Time limit (max 48h per server rules)
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

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

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

# Activate conda environment (using prefix path from config)
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
echo "Train Path: $TRAIN_PATH"
echo "Val Path: $VAL_PATH"
echo "Test Path: $TEST_PATH"
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
echo ""
echo "Output Dir: $OUTPUT_DIR"
echo "Model Save Dir: $MODEL_SAVE_DIR"
echo "=========================================="
echo ""

# ============================================================================
# RUN TRAINING
# ============================================================================

# Get number of GPUs from SLURM
NGPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}

torchrun --nproc_per_node=$NGPUS train.py \
    --task $TASK \
    --train_path $TRAIN_PATH \
    --val_path $VAL_PATH \
    --test_path $TEST_PATH \
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
    $FP16 \
    $PIN_MEMORY \
    $PERSISTENT_WORKERS \
    --cnn_kernel_sizes $CNN_KERNEL_SIZES \
    --cnn_num_filters $CNN_NUM_FILTERS \
    --output_dir $OUTPUT_DIR \
    --model_save_dir $MODEL_SAVE_DIR \
    --log_dir $LOG_DIR

# ============================================================================
# COMPLETION
# ============================================================================

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
