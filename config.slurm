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
# export DATA_DIR="/raid/${USER}/data"  # Node-local fast storage for large datasets
export DATA_DIR="${WORKING_DIR}/data"  # Local storage for small datasets (slower)

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
export MASTER_PORT=29501

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
# Dataset Statistics (calculated from dataset):
# - Total samples: 5,556,224
# - Steps per epoch: 43,408 (with BATCH_SIZE=64 per GPU, NGPUS=2)
# - Max steps for 5 epochs: 217,040

export BATCH_SIZE=64
export LEARNING_RATE=2e-5
export NUM_EPOCHS=-1  # Using MAX_STEPS instead
export WARMUP_RATIO=0.1
export WEIGHT_DECAY=0.01
export DROPOUT=0.1
export SEED=42
export MAX_GRAD_NORM=1.0
export GRADIENT_ACCUMULATION_STEPS=1
export EARLY_STOPPING_PATIENCE=5

# Training steps (based on 5 epochs of dataset)
export MAX_STEPS=217040  # 5 epochs 43,408 steps/epoch
export EVAL_STEPS=4341   # ~10% of one epoch (evaluate ~10 times per epoch)
export SAVE_STEPS=8682   # ~20% of one epoch (save ~5 times per epoch)
export LOGGING_STEPS=50 # Log every 100 steps
export MAX_EVAL_SAMPLES=6000  # Reduced to prevent OOM during evaluation

# ============================================================================
# QLORA CONFIGURATION
# ============================================================================
# Enable QLoRA (set to "--use_qlora" to enable, or "" to disable if default is False)
export USE_QLORA_FLAG="--use_qlora"
export LORA_R=16
export LORA_ALPHA=32
export LORA_DROPOUT=0.1
# Space-separated list of target modules (e.g., "query key value" or "q_proj v_proj")
export LORA_TARGET_MODULES="query key value"

# ============================================================================
# DATALOADER CONFIGURATION (Multiprocessing on Slurm)
# ============================================================================
# Number of DataLoader workers (adjust based on CPU cores allocated)
# NOTE: Set to 0 to avoid OOM issues with streaming datasets and limited shards
export NUM_WORKERS=0
# Enable FP16 mixed precision training (recommended for GPU)
export FP16="--fp16"
# Enable BF16 mixed precision training (alternative to FP16, requires Ampere+ GPU)
# export BF16="--bf16"
export BF16=""
# Pin memory for faster GPU transfer
export PIN_MEMORY="--pin_memory"
# Persistent workers - keeps workers alive between epochs (only works when num_workers > 0)
# NOTE: Disabled when NUM_WORKERS=0
export PERSISTENT_WORKERS=""

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

# Path to the trained model checkpoint for evaluation
# For Trainer checkpoints: point to the checkpoint DIRECTORY (e.g., outputs/checkpoint-XXXX)
# For legacy .pt files: point to the .pt file directly
# UPDATE THIS to your actual best checkpoint path!
export EVAL_MODEL_PATH="${OUTPUT_DIR}/checkpoint-best"

# ============================================================================
# MULTI-PHASE TRAINING (Resume from checkpoint)
# ============================================================================
# Set to empty string "" if starting fresh, or path to checkpoint to resume
# Example: export CHECKPOINT_PATH="${MODEL_SAVE_DIR}/best_segmentation_model_cnn.pt"
export CHECKPOINT_PATH=""
