#!/bin/bash
SBATCH --job-name=sikubert_cnn          # Job name
SBATCH --output=logs/slurm_%j.out       # Output file (%j = job ID)
SBATCH --error=logs/slurm_%j.err        # Error file
SBATCH --partition=gpu                  # Partition name (change as needed)
SBATCH --nodes=1                        # Number of nodes
SBATCH --ntasks=1                       # Number of tasks
SBATCH --cpus-per-task=16               # CPUs per task
SBATCH --gres=gpu:4                     # Number of GPUs (change as needed)
SBATCH --mem=128G                        # Memory per node
SBATCH --time=72:00:00                  # Time limit (HH:MM:SS)
SBATCH --mail-type=BEGIN,END,FAIL       # Email notifications
SBATCH --mail-user=xuanhuynh233@gmail.com  # Your email

# ============================================================================
# LOAD CONFIGURATION
# ============================================================================
# Source the configuration file for parameters
source config.sh

# ============================================================================
# SETUP ENVIRONMENT
# ============================================================================

echo "=========================================="
echo "Multi-GPU Training Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
echo "Start Time: $(date)"
echo "=========================================="

# Load required modules (adjust based on your cluster)
# module load python/3.9
# module load cuda/11.8
# module load cudnn/8.6

conda create -n sikubert python=3.10
conda activate sikubert

# Set environment variables
export PYTHONUNBUFFERED=1
export MASTER_PORT=29500
export MASTER_ADDR=$(hostname)

# Create necessary directories
mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR
mkdir -p $MODEL_SAVE_DIR

# ============================================================================
# PRINT CONFIGURATION
# ============================================================================

# Print GPU information
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
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

torchrun \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --master_port=$MASTER_PORT \
    train.py \
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
    --seed $SEED \
    --cnn_kernel_sizes $CNN_KERNEL_SIZES \
    --cnn_num_filters $CNN_NUM_FILTERS \
    --output_dir $OUTPUT_DIR \
    --model_save_dir $MODEL_SAVE_DIR

# ============================================================================
# COMPLETION
# ============================================================================

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
