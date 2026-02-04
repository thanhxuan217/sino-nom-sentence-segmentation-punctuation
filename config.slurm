# Configuration file for SikuBERT Training
# This file can be sourced in the SLURM script for easier parameter management

# ============================================================================
# DATA PATHS
# ============================================================================
# IMPORTANT: Change these to your actual data locations
export TRAIN_PATH="data/segmentation_train.jsonl"
export VAL_PATH="data/segmentation_val.jsonl"
export TEST_PATH="data/segmentation_test.jsonl"

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
# DISTRIBUTED TRAINING (Multi-GPU)
# ============================================================================
export NGPUS=2
export MASTER_PORT=29500

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
export BATCH_SIZE=64
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
# DATALOADER CONFIGURATION (Multiprocessing on Slurm)
# ============================================================================
# Number of DataLoader workers (adjust based on CPU cores allocated)
export NUM_WORKERS=4
# Enable FP16 mixed precision training (recommended for GPU)
export FP16="--fp16"
# Pin memory for faster GPU transfer
export PIN_MEMORY="--pin_memory"
# Persistent workers - keeps workers alive between epochs (only works when num_workers > 0)
export PERSISTENT_WORKERS="--persistent_workers"

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
