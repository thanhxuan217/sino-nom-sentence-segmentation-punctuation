# SikuBERT Fine-tuning with CNN for SLURM

This repository contains code to fine-tune SikuBERT with CNN layers for token classification tasks (sentence segmentation and punctuation) on a SLURM cluster.


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
> - Store large datasets on `/raid/` of GPU nodes

**Step 1: Create your personal working directory:**
```bash
# Replace 'thanhxuan217' with your username
export WORKING_DIR="/media02/ddien02/thanhxuan217/main_src"
mkdir -p ${WORKING_DIR}/{envs,outputs,models,logs}
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

**Step 4: Setup large dataset on GPU node's /raid:**

> 💡 **Why /raid?** The `/raid` partition on each GPU node has several TB of fast local storage. 
> This is ideal for large datasets, but the data is **only accessible from that specific node**.

```bash
# First, check available GPU nodes
sinfo -N

# SSH to the specific GPU node (or use srun)
srun --nodelist=gpu01 --pty bash

# Create your data directory on the node's /raid
mkdir -p /raid/${USER}/data

# Copy your large dataset (run this ON the GPU node or use scp)
cp /path/to/your/data/*.jsonl /raid/${USER}/data/

# Verify the data is there
ls -la /raid/${USER}/data/

# Exit the node
exit
```

**Step 5: Configure your node in config files:**
```bash
# Edit config.slurm - set the GPU node name
export GPU_NODE="gpu01"  # Must match where you stored the data!

# Edit run.slurm - set the same node
#SBATCH --nodelist=gpu01
```

> ⚠️ **IMPORTANT:** The `GPU_NODE` in `config.slurm` and `--nodelist` in `run.slurm` must 
> match the node where you copied your data, otherwise the job won't find the files!

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

Your data should be in JSON format with the following structure:

```json
[
  {
    "text": "天地玄黃宇宙洪荒",
    "labels": ["B", "M", "M", "E", "B", "M", "M", "E"]
  },
  {
    "text": "日月盈昃辰宿列張",
    "labels": ["B", "M", "M", "E", "B", "M", "M", "E"]
  }
]
```

### 4. Configure Your Training

**Option A: Edit `config.slurm`**

Modify the `config.slurm` file with your settings:

```bash
# Edit data paths
export TRAIN_PATH="/path/to/your/segmentation_train.json"
export VAL_PATH="/path/to/your/segmentation_val.json"
export TEST_PATH="/path/to/your/segmentation_test.json"

# Change task if needed
export TASK="segmentation"  # or "punctuation"

# Adjust hyperparameters
export BATCH_SIZE=64
export LEARNING_RATE=2e-5
export NUM_EPOCHS=5
```

### 5. Submit to SLURM

```bash
# Navigate to your working directory first
cd /media02/ddien02/thanhxuan217/main_src

# Submit the job
sbatch run.slurm
```

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
| `--batch_size` | Training batch size | 64 |
| `--learning_rate` | Learning rate | 2e-5 |
| `--num_epochs` | Number of training epochs | 5 |
| `--warmup_ratio` | Warmup ratio for learning rate | 0.1 |
| `--weight_decay` | Weight decay for AdamW | 0.01 |
| `--dropout` | Dropout rate | 0.1 |
| `--cnn_kernel_sizes` | CNN kernel sizes (for cnn head) | 3 5 7 |
| `--cnn_num_filters` | Number of CNN filters (for cnn head) | 256 |
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
1. **SikuBERT** backbone (pretrained)
2. **Optional CNN layer** with configurable kernel sizes (when using `--head_type cnn`)
3. **Optional CRF layer** for sequence labeling (when using `--head_type crf`)
4. **Classification head** for token-level predictions

Example architectures:
```
# softmax (FC only)
SikuBERT → Dropout → Linear Classification

# crf
SikuBERT → Dropout → Linear → CRF

# cnn (default)
SikuBERT → Dropout → Multi-Kernel CNN → Dropout → Linear Classification
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
    --train_path data/segmentation_train.json \
    --val_path data/segmentation_val.json \
    --test_path data/segmentation_test.json \
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

- Ensure the model name is correct: `SIKU-BERT/sikubert`
- Check internet connection for downloading pretrained weights
- Set cache directory: `export TRANSFORMERS_CACHE=/path/to/cache`

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
