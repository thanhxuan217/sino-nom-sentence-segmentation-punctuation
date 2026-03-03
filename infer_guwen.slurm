#!/bin/bash
#SBATCH --job-name=guwen_infer               # Job name
#SBATCH --output=logs/guwen_infer_%j.out     # Output file
#SBATCH --error=logs/guwen_infer_%j.err      # Error file
#SBATCH --partition=batch                    # Partition name
#SBATCH --nodes=1                            # Number of nodes
#SBATCH --ntasks=1                           # Number of tasks
#SBATCH --cpus-per-task=16                   # CPUs per task
#SBATCH --gres=gpu:1                         # 1 GPU (stitching is sequential)
#SBATCH --mem=16G                            # Memory per node
#SBATCH --time=24:00:00                      # Time limit
#SBATCH --mail-type=BEGIN,END,FAIL           # Email notifications
#SBATCH --mail-user=xuanhuynh233@gmail.com   # Your email

set -euo pipefail

# ============================================================================
# LOAD CONFIGURATION (reuse existing config.slurm for paths)
# ============================================================================
if [[ ! -f config.slurm ]]; then
    echo "ERROR: config.slurm not found in $(pwd)"
    exit 1
fi

source config.slurm

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Ensure log directory exists
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

# ============================================================================
# SETUP ENVIRONMENT
# ============================================================================

echo "=========================================="
echo "Guwen Baseline Inference Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Number of GPUs: ${SLURM_GPUS_ON_NODE:-1}"
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
# GUWEN-SPECIFIC CONFIGURATION
# ============================================================================

# Which split to run inference on (default: test)
INFER_SPLIT="${INFER_SPLIT:-test}"

# Overlap size used during data chunking (must match the parquet creation)
INFER_OVERLAP="${INFER_OVERLAP:-128}"

# Max length for guwen models (max_position_embeddings=514, minus 2 for [CLS]+[SEP])
GUWEN_MAX_LENGTH="${GUWEN_MAX_LENGTH:-510}"

# ============================================================================
# GPU INFO
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
echo "GUWEN INFERENCE CONFIGURATION"
echo "=========================================="
echo "Task: $TASK"
echo "Split: $INFER_SPLIT"
echo "Model: (auto-selected based on task)"
echo "Max Length: $GUWEN_MAX_LENGTH"
echo "Data Dir: $DATA_DIR"
echo ""
echo "Overlap: $INFER_OVERLAP"
echo "Batch Size: $BATCH_SIZE"
echo ""
echo "Output Dir: $OUTPUT_DIR"
echo "=========================================="
echo ""

# ============================================================================
# RUN INFERENCE (single GPU — stitching is inherently sequential)
# ============================================================================

python infer_guwen_seg.py \
    --task $TASK \
    --data_dir $DATA_DIR \
    --split $INFER_SPLIT \
    --max_length $GUWEN_MAX_LENGTH \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --overlap $INFER_OVERLAP \
    $FP16 \
    $PIN_MEMORY \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR

# ============================================================================
# COMPLETION
# ============================================================================

echo ""
echo "=========================================="
echo "Guwen inference ($TASK) completed at: $(date)"
echo "=========================================="
