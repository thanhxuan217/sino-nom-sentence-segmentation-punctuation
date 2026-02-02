# Configuration file for SikuBERT Training
# This file can be sourced in the SLURM script for easier parameter management

# ============================================================================
# DATA PATHS
# ============================================================================
# IMPORTANT: Change these to your actual data locations
export TRAIN_PATH="../data/segmentation_train.json"
export VAL_PATH="../data/segmentation_val.json"
export TEST_PATH="../data/segmentation_test.json"

# ============================================================================
# TASK CONFIGURATION
# ============================================================================
# Options: "segmentation" or "punctuation"
export TASK="segmentation"

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
export MODEL_NAME="SIKU-BERT/sikubert"
export MAX_LENGTH=256

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
export BATCH_SIZE=128
export LEARNING_RATE=2e-5
export NUM_EPOCHS=5
export WARMUP_RATIO=0.1
export WEIGHT_DECAY=0.01
export DROPOUT=0.1
export SEED=42
export MAX_GRAD_NORM=1.0
export GRADIENT_ACCUMULATION_STEPS=1
export EARLY_STOPPING_PATIENCE=3

# ============================================================================
# CNN CONFIGURATION
# ============================================================================
export CNN_KERNEL_SIZES="3 5 7"
export CNN_NUM_FILTERS=256

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================
export OUTPUT_DIR="outputs"
export MODEL_SAVE_DIR="models"
export LOG_DIR="logs"

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================
# Uncomment and modify if needed
# export TRANSFORMERS_CACHE="/path/to/cache/transformers"
# export HF_HOME="/path/to/cache/huggingface"
# export PYTHONUNBUFFERED=1

# ============================================================================
# CLUSTER-SPECIFIC SETTINGS (for reference)
# ============================================================================
# These are typically set in the SLURM script header
# export SLURM_PARTITION="gpu"
# export SLURM_NODES=1
# export SLURM_CPUS_PER_TASK=4
# export SLURM_GPUS=1
# export SLURM_MEM="32G"
# export SLURM_TIME="24:00:00"
