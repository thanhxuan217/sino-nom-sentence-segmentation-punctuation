# SikuBERT / GuwenBERT Fine-tuning with QLoRA & CNN for SLURM

This repository contains code to fine-tune SikuBERT, GuwenBERT, SikuRoberta, and other classical Chinese language models with QLoRA and CNN layers for token classification tasks (sentence segmentation and punctuation).

## 🚀 Quick Start

### 1. Run Setup Script (Recommended)

This will create required folders (`logs`, `outputs`, `models`, `data`) and check SLURM/CUDA availability:

```bash
bash setup.slurm
```

### 2. Setup Environment

#### For SLURM Server (Recommended)

> ⚠️ **Server Rules:**
> - Max 2 jobs per group at a time
> - Max resources per job: 16 CPUs, 64GB RAM, 2 GPUs
> - Max runtime: 48 hours
> - Store source code & env in `/media02/ddien02/<username>/`

**Step 1: Create your personal working directory:**
```bash
# Replace 'thanhxuan217' with your username
export WORKING_DIR="/media02/ddien02/thanhxuan217/main_src"
mkdir -p ${WORKING_DIR}/{envs,outputs,models,logs,data}
```

**Step 2: Create Conda environment with prefix:**
```bash
# Create conda environment in your working directory
conda create --prefix ${WORKING_DIR}/envs/sikubert python=3.11 -y

# Activate using the full path
conda activate ${WORKING_DIR}/envs/sikubert

# Install dependencies
pip install -r requirements.txt
```

**Step 3: Copy source code:**
```bash
# Copy source code to your working directory
cp -r ./* ${WORKING_DIR}/
```

**Step 4: Configure your node in config files (Optional):**
If you need specific node configuration updates, modify `config.slurm`.

#### For Local Development

```bash
# Option 1: Using pip
pip install -r requirements.txt

# Option 2: Using conda (standard)
conda create -n sikubert python=3.11
conda activate sikubert
pip install -r requirements.txt
```

### 3. Prepare Your Data

Your data **must be in Parquet format** to support streaming and memory efficiency. The dataset should be partitioned into `train`, `val`, and `test` subdirectories.

**Directory Structure:**
```
data/
├── train/
│   ├── part-00000.parquet
│   └── ...
├── val/
│   ├── part-00000.parquet
│   └── ...
└── test/
    ├── part-00000.parquet
    └── ...
```

**Parquet Schema:**
Each row should contain:
- `text`: The input text (string or list of characters).
- `labels`: List of labels corresponding to each character/token.

**Example Structure:**

| text | labels |
|------|--------|
| "天地玄黃..." | ["B", "M", "M", "E", ...] |
| "日月盈昃..." | ["B", "M", "M", "E", ...] |

### 4. Configure Your Training

**Option A: Edit `config.slurm`**

Modify the `config.slurm` file with your settings:

```bash
# Edit data paths (Data is stored in data/ folder)
export DATA_DIR="${WORKING_DIR}/data"

# Change task if needed
export TASK="segmentation"  # or "punctuation"

# Model Selection
# Options: SIKU-BERT/sikubert, ethanyt/guwenbert-base, hfl/chinese-roberta-wwm-ext, etc.
export MODEL_NAME="SIKU-BERT/sikubert"

# QLoRA Settings
export USE_QLORA_FLAG="--use_qlora"
export LORA_R=16

# Adjust hyperparameters
export BATCH_SIZE=64
export LEARNING_RATE=2e-5
export NUM_EPOCHS=5
```

### 5. Submit to SLURM

```bash
# Navigate to your working directory first (if not already there)
cd /media02/ddien02/thanhxuan217/main_src

# Submit the job
sbatch run.slurm
```

## 🧪 Running Evaluation

You can evaluate the trained model on either the **validation** or **test** set.

### Option 1: Using SLURM (Recommended)

1.  **Configure `config.slurm`**:
    Edit the `TEST_SPLIT` variable to choose the dataset split:
    ```bash
    # Set to "val" for validation set, or "test" for test set
    export TEST_SPLIT="test"
    
    # Ensure EVAL_MODEL_PATH points to your trained model checkpoint
    export EVAL_MODEL_PATH="${MODEL_SAVE_DIR}/final_segmentation_model_cnn"
    ```

2.  **Submit the evaluation job**:
    ```bash
    sbatch evaluate.slurm
    ```

### Option 2: Running Locally

```bash
python evaluate.py \
    --task segmentation \
    --test_split test \
    --model_path models/final_segmentation_model_cnn \
    --data_dir data \
    --head_type cnn \
    --batch_size 32 \
    --output_dir outputs
```

> **Note:** Use `--test_split val` to evaluate on the validation set.

### 6. Monitor Your Job

```bash
# Check job status
squeue -u $USER

# View output in real-time
tail -f logs/slurm_<JOB_ID>.out

# View errors
tail -f logs/slurm_<JOB_ID>.err

# Cancel a job
scancel <JOB_ID>
```

## ⚙️ Configuration Options

### SLURM Resources

Edit the `#SBATCH` directives in the SLURM script:

```bash
#SBATCH --partition=gpu          # GPU partition name
#SBATCH --gres=gpu:1             # Number of GPUs (1, 2, 4, etc.)
#SBATCH --mem=32G                # Memory (16G, 32G, 64G, etc.)
#SBATCH --time=24:00:00          # Time limit
#SBATCH --cpus-per-task=4        # Number of CPU cores
```

### Training Hyperparameters

Key parameters you can adjust:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--task` | Task type (segmentation/punctuation) | segmentation |
| `--model_name` | Pretrained model (SikuBERT, GuwenBERT, etc.) | SIKU-BERT/sikubert |
| `--use_qlora` | Enable QLoRA fine-tuning | True |
| `--lora_r` | LoRA rank | 16 |
| `--batch_size` | Training batch size | 64 |
| `--learning_rate` | Learning rate | 2e-5 |
| `--num_epochs` | Number of training epochs | 5 |
| `--head_type` | Classification head type | cnn |

### Classification Head Types

The model supports 3 classification head architectures:

| Head Type | Description | Architecture |
|-----------|-------------|-------------|
| `softmax` | Fully Connected only | BERT → Dropout → Linear → Softmax |
| `crf` | BERT + CRF | BERT → Dropout → Linear → CRF |
| `cnn` | BERT + CNN (default) | BERT → Dropout → CNN → Dropout → Linear |

### Model Architecture

The model consists of:
1. **Backbone**: SikuBERT, GuwenBERT, SikuRoberta, etc. (supports QLoRA 4-bit loading)
2. **Optional CNN layer** with configurable kernel sizes (when using `--head_type cnn`)
3. **Optional CRF layer** for sequence labeling (when using `--head_type crf`)
4. **Classification head** for token-level predictions

Example architectures:
```
# softmax (FC only)
BERT(QLoRA) → Dropout → Linear Classification

# crf
BERT(QLoRA) → Dropout → Linear → CRF

# cnn (default)
BERT(QLoRA) → Dropout → Multi-Kernel CNN → Dropout → Linear Classification
```

## 📊 Output Files

After training completes, you'll find:

```
outputs/
├── train_segmentation.log           # Training log
└── segmentation_results.json        # Test set results

models/
└── best_segmentation_model_cnn.pt   # Best model checkpoint

logs/
├── slurm_<JOB_ID>.out              # SLURM stdout
└── slurm_<JOB_ID>.err              # SLURM stderr
```

### Results JSON Format

```json
{
  "task": "segmentation",
  "test_metrics": {
    "loss": 0.1234,
    "precision": 0.9567,
    "recall": 0.9523,
    "f1": 0.9545
  },
  "config": {
    "batch_size": 64,
    "learning_rate": 2e-5,
    ...
  }
}
```

## 🔧 Advanced Usage

### Running Locally (without SLURM)

```bash
python train.py \
    --task segmentation \
    --data_dir data \
    --head_type cnn \
    --batch_size 32 \
    --num_epochs 5 \
    --output_dir outputs \
    --model_save_dir models
```

#### Examples with Different Head Types

```bash
# Train with Softmax (FC only)
python train.py --task segmentation --head_type softmax ...

# Train with CRF
python train.py --task segmentation --head_type crf ...

# Train with CNN (default)
python train.py --task segmentation --head_type cnn ...
```

### Hyperparameter Sweep

You can modify the script to run hyperparameter searches:

```bash
for lr in 1e-5 2e-5 5e-5; do
    for bs in 32 64 128; do
        sbatch --export=LEARNING_RATE=$lr,BATCH_SIZE=$bs run.slurm
    done
done
```

## 🎯 Task-Specific Notes

### Segmentation Task

- **Labels**: B (Beginning), M (Middle), E (End), S (Single)
- **Purpose**: Split classical Chinese text into sentences
- **Evaluation**: All labels are evaluated

### Punctuation Task

- **Labels**: O (no punctuation), ，。：、；？！
- **Purpose**: Add punctuation to unpunctuated text
- **Evaluation**: O tokens are ignored in metrics

## 🐛 Troubleshooting

### Out of Memory (OOM)

1. Reduce batch size: `--batch_size 32` or `--batch_size 16`
2. Reduce max sequence length: `--max_length 128`
3. Request more GPU memory in SLURM: `#SBATCH --mem=64G`

### CUDA Out of Memory

```bash
# Enable gradient accumulation
--gradient_accumulation_steps 2
```

### Slow Training

1. Increase batch size if memory allows
2. Use multiple GPUs: `#SBATCH --gres=gpu:2`
3. Check data loading isn't a bottleneck

### Model Not Loading

- Ensure the model name is correct: `SIKU-BERT/sikubert` or `ethanyt/guwenbert-base`
- Check internet connection for downloading pretrained weights
- Set cache directory: `export TRANSFORMERS_CACHE=/path/to/cache`

### QLoRA / Bitsandbytes Issues

- Ensure `bitsandbytes` is installed and CUDA is available.
- If running on CPU, disable QLoRA (`--use_qlora False`) as 4-bit loading requires GPU.

### Job Not Starting

```bash
# Check queue
squeue -u $USER

# Check job details
scontrol show job <JOB_ID>

# Check cluster status
sinfo
```

## 📈 Monitoring and Logging

### Real-time Monitoring

```bash
# Watch SLURM output
watch -n 5 tail -20 logs/slurm_<JOB_ID>.out

# Monitor GPU usage (if you have access to the node)
nvidia-smi -l 1
```

### Training Metrics

The training log includes:
- Loss per epoch
- Validation metrics (precision, recall, F1)
- Early stopping information
- Best model checkpoint notification

Example log output:
```
======================================================================
TRAINING START
======================================================================

Epoch 1/5
Training: 100%|██████████| 1000/1000 [05:23<00:00, 3.09it/s, loss=0.234]
Train Loss: 0.2340
Val Loss: 0.1520
Val Precision: 0.9234
Val Recall: 0.9187
Val F1: 0.9210
✓ New best F1: 0.9210 - Model saved!
```

## 🔍 Model Evaluation

The script automatically evaluates on the test set after training and saves:

1. **Overall metrics**: Precision, Recall, F1
2. **Per-label metrics**: Performance for each label class
3. **Configuration**: All hyperparameters used

## 📝 Citation

If you use SikuBERT in your research, please cite:

```bibtex
@article{sikubert,
  title={SIKU-BERT: A Domain-Specific Language Model for Classical Chinese},
  author={hxuan},
  journal={...},
  year={2026}
}
```

## 📧 Contact

For questions or issues, please open an issue or contact the maintainers.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
