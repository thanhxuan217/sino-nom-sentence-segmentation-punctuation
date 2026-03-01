# ============================================================================
# DAPT Configuration for SikuBERT
# Source this file in dapt.slurm
# ============================================================================

# ============================================================================
# PATHS
# ============================================================================
export WORKING_DIR="/media02/ddien02/thanhxuan217/main_src"
export CONDA_ENV="${WORKING_DIR}/envs/sikubert"
export GPU_NODE="gpu04"

# Original pretrained model
export MODEL_NAME="SIKU-BERT/sikubert"
# Extended tokenizer
export TOKENIZER="${WORKING_DIR}/sikubert_extended_vocab"

# ============================================================================
# RAW CORPUS (for data preparation - run prepare_dapt_data.py first)
# ============================================================================
# Set this to your raw corpus location
# For txt format: directory containing .txt files
# For jsonl format: path to .jsonl file
export RAW_CORPUS_PATH="/path/to/your/raw/corpus"
export RAW_CORPUS_FORMAT="txt"  # "txt" or "jsonl"

# ============================================================================
# DAPT DATA (output of prepare_dapt_data.py)
# ============================================================================
export DAPT_DATA_DIR="${WORKING_DIR}/dapt_data"

# ============================================================================
# GPU CONFIGURATION
# ============================================================================
export NGPUS=2
export MASTER_PORT=29502  # Different from finetune port (29501)

# ============================================================================
# PHASE 1: Short sequences (warmup)
# ============================================================================
export PHASE1_MAX_LENGTH=128
export PHASE1_BATCH_SIZE=64
export PHASE1_GRAD_ACCUM=4
# Effective batch: 64 * 2 * 4 = 512
export PHASE1_LR=5e-5
export PHASE1_EPOCHS=1
export PHASE1_SAVE_STEPS=2000
export PHASE1_OUTPUT="${WORKING_DIR}/dapt_outputs/phase1"

# ============================================================================
# PHASE 2: Full sequences (main DAPT)
# ============================================================================
export PHASE2_MAX_LENGTH=512
export PHASE2_BATCH_SIZE=32
export PHASE2_GRAD_ACCUM=8
# Effective batch: 32 * 2 * 8 = 512
export PHASE2_LR=2e-5
export PHASE2_EPOCHS=1
export PHASE2_SAVE_STEPS=1000
export PHASE2_OUTPUT="${WORKING_DIR}/dapt_outputs/phase2"

# ============================================================================
# COMMON TRAINING PARAMETERS
# ============================================================================
export WARMUP_RATIO=0.06
export WEIGHT_DECAY=0.01
export MAX_GRAD_NORM=1.0
export MLM_PROBABILITY=0.15
export PUNCT_MASK_PROB=0.30
export SEED=42
export LOGGING_STEPS=100
export SAVE_TOTAL_LIMIT=5
export NUM_WORKERS=4

# Mixed precision
export FP16="--fp16"
export BF16=""

# ============================================================================
# DATA PREPARATION PARAMETERS
# ============================================================================
export SHARD_SIZE=100000  # Sequences per parquet shard
