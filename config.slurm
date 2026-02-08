# Configuration file for SikuBERT Training
# This file can be sourced in the SLURM script for easier parameter management

# ============================================================================
# SERVER RULES REMINDER (IMPORTANT!)
# ============================================================================
# - Max 2 jobs per group at a time
# - Max resources per job: 16 CPUs, 64GB RAM, 2 GPUs
# - Max runtime: 48 hours (jobs will be killed after this)
# - Save checkpoints regularly as jobs may be interrupted
# - Data storage: /media02/ddien02 (for source code & environments)
# - GPU node /raid can be used for training data (node-specific)
# ============================================================================

# ============================================================================
# WORKING DIRECTORY & GPU NODE
# ============================================================================
export WORKING_DIR="/media02/ddien02/thanhxuan217/main_src"
export CONDA_ENV="${WORKING_DIR}/envs/sikubert"

# GPU Node Configuration (for large datasets stored on /raid)
# IMPORTANT: Change GPU_NODE to match the node where your data is stored!
# Run 'sinfo -N' to see available nodes
export GPU_NODE="gpu04"  # Available: gpu01, gpu02, gpu03, gpu04
export DATA_DIR="/raid/${USER}/data"  # Node-local fast storage for large datasets
# export DATA_DIR="${WORKING_DIR}/data"  # Local storage for small datasets (slower)

# ============================================================================
# DATA PATHS
# ============================================================================
# Data is stored on /raid of the specific GPU node for faster I/O
# Make sure to copy your data to the GPU node first!
export TRAIN_PATH="${DATA_DIR}/train/segmentation_sliding_part_0.jsonl"
export VAL_PATH="${DATA_DIR}/val/segmentation_sliding_val.jsonl"
export TEST_PATH="${DATA_DIR}/test/segmentation_sliding_test.jsonl"

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
# HEAD TYPE CONFIGURATION
# ============================================================================
# Options: "softmax" (FC only), "crf" (BERT+CRF), "cnn" (BERT+CNN)
export HEAD_TYPE="cnn"

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================
export OUTPUT_DIR="${WORKING_DIR}/outputs"
export MODEL_SAVE_DIR="${WORKING_DIR}/models"
export LOG_DIR="${WORKING_DIR}/logs"

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
# Path to the trained model for evaluation
export EVAL_MODEL_PATH="${MODEL_SAVE_DIR}/best_${TASK}_model_${HEAD_TYPE}.pt"

# ============================================================================
# MULTI-PHASE TRAINING (Resume from checkpoint)
# ============================================================================
# Set to empty string "" if starting fresh, or path to checkpoint to resume
# Example: export CHECKPOINT_PATH="${MODEL_SAVE_DIR}/best_segmentation_model_cnn.pt"
export CHECKPOINT_PATH=""

