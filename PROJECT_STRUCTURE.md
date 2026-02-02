# SikuBERT SLURM Training - Complete Project Structure

## 📁 Directory Tree

```
sikubert-slurm-training/
│
├── 📄 train.py                          # Main training script
├── 📄 requirements.txt                  # Python dependencies
├── 📄 README.md                         # Documentation
│
├── 🔧 Configuration Files
│   ├── config.sh                        # Centralized configuration
│   ├── run_slurm.sh                     # SLURM script (standalone)
│   ├── run_slurm_with_config.sh         # SLURM script (uses config.sh)
│   └── run_slurm_multigpu.sh            # Multi-GPU SLURM script
│
├── 🛠️ Utility Scripts
│   ├── setup.sh                         # Initial setup script
│   └── slurm_helper.sh                  # Job management helper
│
├── 📂 data/                             # Your training data
│   ├── segmentation_train.json          # Training set
│   ├── segmentation_val.json            # Validation set
│   ├── segmentation_test.json           # Test set
│   ├── punctuation_train.json           # (Alternative task)
│   ├── punctuation_val.json
│   └── punctuation_test.json
│
├── 📂 models/                           # Saved model checkpoints
│   ├── best_segmentation_model_cnn.pt   # Best segmentation model
│   └── best_punctuation_model_cnn.pt    # Best punctuation model
│
├── 📂 outputs/                          # Training outputs
│   ├── train_segmentation.log           # Training log
│   ├── segmentation_results.json        # Test results
│   └── test_pred.json                   # Detailed predictions
│
└── 📂 logs/                             # SLURM job logs
    ├── slurm_12345.out                  # Job stdout
    └── slurm_12345.err                  # Job stderr
```

---

## 📊 Data File Structures

### 1. Input Data Format (JSON)

**File**: `data/segmentation_train.json`

```json
[
  {
    "text": "天地玄黃宇宙洪荒",
    "labels": ["B", "M", "M", "E", "B", "M", "M", "E"]
  },
  {
    "text": "日月盈昃辰宿列張",
    "labels": ["B", "M", "M", "E", "B", "M", "M", "E"]
  },
  {
    "text": "寒來暑往秋收冬藏",
    "labels": ["B", "M", "M", "E", "B", "M", "M", "E"]
  }
]
```

**Schema**:
- `text` (string): Raw Classical Chinese text without spaces or punctuation
- `labels` (array of strings): Character-level labels
  - For **segmentation**: `B` (Begin), `M` (Middle), `E` (End), `S` (Single)
  - For **punctuation**: `O` (no punctuation), `，`, `。`, `：`, `、`, `；`, `？`, `！`

**Requirements**:
- `len(text) == len(labels)` for each sample
- All characters must have corresponding labels
- UTF-8 encoding

---

### 2. Configuration File Structure

**File**: `config.sh`

```bash
# Data paths - YOU MUST MODIFY THESE
export TRAIN_PATH="/path/to/data/segmentation_train.json"
export VAL_PATH="/path/to/data/segmentation_val.json"
export TEST_PATH="/path/to/data/segmentation_test.json"

# Task selection
export TASK="segmentation"  # or "punctuation"

# Model configuration
export MODEL_NAME="SIKU-BERT/sikubert"
export MAX_LENGTH=256

# Training hyperparameters
export BATCH_SIZE=64
export LEARNING_RATE=2e-5
export NUM_EPOCHS=5
export WARMUP_RATIO=0.1
export WEIGHT_DECAY=0.01
export DROPOUT=0.1
export SEED=42

# CNN architecture
export CNN_KERNEL_SIZES="3 5 7"
export CNN_NUM_FILTERS=256

# Output directories
export OUTPUT_DIR="outputs"
export MODEL_SAVE_DIR="models"
export LOG_DIR="logs"
```

---

### 3. Model Output Structure

**File**: `models/best_segmentation_model_cnn.pt`

```
PyTorch State Dict (.pt file)
├── bert.embeddings.word_embeddings.weight     [vocab_size, 768]
├── bert.embeddings.position_embeddings.weight [512, 768]
├── bert.encoder.layer.0.attention...          [various shapes]
├── ...
├── extra_layer.convs.0.weight                 [256, 768, 3]
├── extra_layer.convs.1.weight                 [256, 768, 5]
├── extra_layer.convs.2.weight                 [256, 768, 7]
└── classifier.weight                          [num_labels, 768]
```

**Size**: ~400-500 MB (depends on configuration)

---

### 4. Training Results Structure

**File**: `outputs/segmentation_results.json`

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
    "task": "segmentation",
    "train_path": "/path/to/segmentation_train.json",
    "val_path": "/path/to/segmentation_val.json",
    "test_path": "/path/to/segmentation_test.json",
    "model_name": "SIKU-BERT/sikubert",
    "max_length": 256,
    "batch_size": 64,
    "learning_rate": 2e-5,
    "num_epochs": 5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "dropout": 0.1,
    "seed": 42,
    "cnn_kernel_sizes": [3, 5, 7],
    "cnn_num_filters": 256,
    "output_dir": "outputs",
    "model_save_dir": "models"
  }
}
```

---

### 5. Training Log Structure

**File**: `outputs/train_segmentation.log`

```
2024-02-02 10:00:00 | ======================================================================
2024-02-02 10:00:00 | TRAINING CONFIGURATION
2024-02-02 10:00:00 | ======================================================================
2024-02-02 10:00:00 | task: segmentation
2024-02-02 10:00:00 | train_path: /path/to/segmentation_train.json
2024-02-02 10:00:00 | batch_size: 64
2024-02-02 10:00:00 | learning_rate: 2e-05
2024-02-02 10:00:00 | ======================================================================
2024-02-02 10:00:01 | 
2024-02-02 10:00:01 | ✓ Device: cuda
2024-02-02 10:00:01 |   GPU: NVIDIA A100-SXM4-40GB
2024-02-02 10:00:02 | 
2024-02-02 10:00:02 | ✓ Task: segmentation
2024-02-02 10:00:02 |   Labels: ['B', 'M', 'E', 'S']
2024-02-02 10:00:02 |   Num labels: 4
2024-02-02 10:00:03 | 
2024-02-02 10:00:03 | ✓ Loading tokenizer...
2024-02-02 10:00:04 | ✓ Loading data...
2024-02-02 10:00:04 |   Train samples: 10000
2024-02-02 10:00:04 |   Val samples: 2000
2024-02-02 10:00:05 | ✓ Creating dataloaders...
2024-02-02 10:00:06 | ✓ Creating model...
2024-02-02 10:00:06 |   Total parameters: 103,456,789
2024-02-02 10:00:07 | 
2024-02-02 10:00:07 | ======================================================================
2024-02-02 10:00:07 | TRAINING START
2024-02-02 10:00:07 | ======================================================================
2024-02-02 10:00:07 | 
2024-02-02 10:00:07 | Epoch 1/5
2024-02-02 10:05:30 | Train Loss: 0.2340
2024-02-02 10:06:15 | Val Loss: 0.1520
2024-02-02 10:06:15 | Val Precision: 0.9234
2024-02-02 10:06:15 | Val Recall: 0.9187
2024-02-02 10:06:15 | Val F1: 0.9210
2024-02-02 10:06:15 | ✓ New best F1: 0.9210 - Model saved!
...
```

---

### 6. SLURM Log Structure

**File**: `logs/slurm_12345.out`

```
==========================================
Job ID: 12345
Job Name: sikubert_cnn
Node: gpu-node-01
Start Time: Fri Feb 02 10:00:00 2024
==========================================

==========================================
TRAINING CONFIGURATION
==========================================
Task: segmentation
Model: SIKU-BERT/sikubert
Batch Size: 64
Learning Rate: 2e-05
Epochs: 5
CNN Kernels: 3 5 7
CNN Filters: 256
==========================================

[Training output from train.py...]

==========================================
Job completed at: Fri Feb 02 12:30:00 2024
==========================================
```

**File**: `logs/slurm_12345.err`

```
[Error messages, warnings, or empty if no errors]
```

---

## 🔄 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Data                              │
│  (segmentation_train.json, val.json, test.json)             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              ClassicalChineseDataset                        │
│  • Tokenizes text character-by-character                    │
│  • Aligns labels with tokens                                │
│  • Handles padding and truncation                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   DataLoader                                │
│  • Batches data                                             │
│  • Shuffles training data                                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         SikuBERTForTokenClassification                      │
│                                                             │
│  ┌──────────────┐                                           │
│  │  SikuBERT    │  Pretrained BERT for Classical Chinese   │
│  │  Encoder     │  Output: [batch, seq_len, 768]           │
│  └──────┬───────┘                                           │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                           │
│  │   Dropout    │                                           │
│  └──────┬───────┘                                           │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                           │
│  │ Multi-Kernel │  CNN with kernels [3, 5, 7]             │
│  │     CNN      │  Output: [batch, seq_len, 768]           │
│  └──────┬───────┘                                           │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                           │
│  │   Dropout    │                                           │
│  └──────┬───────┘                                           │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                           │
│  │   Linear     │  Classification head                     │
│  │ Classifier   │  Output: [batch, seq_len, num_labels]    │
│  └──────────────┘                                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                Training Loop                                │
│  • Forward pass                                             │
│  • Compute CrossEntropyLoss                                 │
│  • Backward pass                                            │
│  • Optimizer step (AdamW)                                   │
│  • Learning rate scheduling                                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Validation & Early Stopping                    │
│  • Evaluate on validation set                               │
│  • Calculate precision, recall, F1                          │
│  • Save best model                                          │
│  • Stop if no improvement for N epochs                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Test Evaluation                            │
│  • Load best model                                          │
│  • Evaluate on test set (never seen before)                 │
│  • Calculate final metrics                                  │
│  • Save results to JSON                                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     Outputs                                 │
│  • models/best_segmentation_model_cnn.pt                    │
│  • outputs/train_segmentation.log                           │
│  • outputs/segmentation_results.json                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Task-Specific Label Structures

### Segmentation Task

**Labels**: `['B', 'M', 'E', 'S']`

```
Text:   天 地 玄 黃 宇 宙 洪 荒
Labels: B  M  M  E  B  M  M  E

Meaning:
- B (Begin):  First character of a sentence
- M (Middle): Middle character of a sentence
- E (End):    Last character of a sentence
- S (Single): Single-character sentence
```

**Example Data**:
```json
{
  "text": "天地玄黃宇宙洪荒",
  "labels": ["B", "M", "M", "E", "B", "M", "M", "E"]
}
```

**Prediction Output**:
```
Segmented: 天地玄黃 | 宇宙洪荒
```

---

### Punctuation Task

**Labels**: `['O', '，', '。', '：', '、', '；', '？', '！']`

```
Text:   天 地 玄 黃 宇 宙 洪 荒
Labels: O  O  O  ，  O  O  O  。

Meaning:
- O:  No punctuation after this character
- ，: Comma
- 。: Period
- ：: Colon
- 、: Enumeration comma
- ；: Semicolon
- ？: Question mark
- ！: Exclamation mark
```

**Example Data**:
```json
{
  "text": "天地玄黃宇宙洪荒",
  "labels": ["O", "O", "O", "，", "O", "O", "O", "。"]
}
```

**Prediction Output**:
```
Punctuated: 天地玄黃，宇宙洪荒。
```

---

## 📦 Model Architecture Details

```
SikuBERTForTokenClassification(
  (bert): AutoModel(
    vocab_size: 21128
    hidden_size: 768
    num_hidden_layers: 12
    num_attention_heads: 12
    parameters: ~102M
  )
  
  (dropout): Dropout(p=0.1)
  
  (extra_layer): MultiKernelCNN(
    (convs): ModuleList(
      (0): Conv1d(768, 256, kernel_size=3, padding=1)
      (1): Conv1d(768, 256, kernel_size=5, padding=2)
      (2): Conv1d(768, 256, kernel_size=7, padding=3)
    )
    output_size: 768 (256 * 3 kernels)
    parameters: ~1.5M
  )
  
  (classifier): Linear(
    in_features: 768
    out_features: 4  (for segmentation) or 8 (for punctuation)
    parameters: ~3K
  )
)

Total Parameters: ~103.5M
Trainable Parameters: ~103.5M
```

---

## 🔢 Batch Processing Example

**Input Batch Shape**:
```python
{
  'input_ids': torch.Size([64, 256]),      # [batch_size, max_length]
  'attention_mask': torch.Size([64, 256]), # [batch_size, max_length]
  'labels': torch.Size([64, 256])          # [batch_size, max_length]
}
```

**Model Forward Pass**:
```python
Input: [64, 256] token IDs
  ↓
BERT Encoder: [64, 256, 768] hidden states
  ↓
Dropout: [64, 256, 768]
  ↓
Multi-Kernel CNN: [64, 256, 768]
  ↓
Dropout: [64, 256, 768]
  ↓
Linear Classifier: [64, 256, 4] logits
  ↓
Output: Predictions + Loss
```

---

## 💾 Storage Requirements

| Component | Size | Notes |
|-----------|------|-------|
| Raw Data | ~10-100 MB | JSON files (depends on dataset size) |
| Model Checkpoint | ~400-500 MB | PyTorch state dict |
| Training Logs | ~1-10 MB | Text logs per experiment |
| SLURM Logs | ~1-5 MB | Per job |
| Cache (HuggingFace) | ~400 MB | Downloaded pretrained models |
| **Total** | **~1-2 GB** | Per experiment |

---

## 🚀 Execution Flow

```
1. User edits config.sh
   └─> Sets data paths, hyperparameters
   
2. User submits: sbatch run_slurm_with_config.sh
   └─> SLURM schedules job on GPU node
   
3. Job starts on compute node
   ├─> Loads environment (modules, virtualenv)
   ├─> Sources config.sh
   ├─> Creates directories (logs/, outputs/, models/)
   └─> Executes: python train.py [args...]
   
4. train.py execution
   ├─> Setup logging
   ├─> Load tokenizer
   ├─> Load and preprocess data
   ├─> Create dataloaders
   ├─> Initialize model
   ├─> Training loop
   │   ├─> Train epoch
   │   ├─> Validate
   │   ├─> Save best model
   │   └─> Early stopping check
   ├─> Load best model
   ├─> Test evaluation
   └─> Save results
   
5. Job completes
   ├─> Outputs saved to outputs/
   ├─> Model saved to models/
   └─> Logs saved to logs/
   
6. User checks results
   └─> ./slurm_helper.sh logs <job_id>
```

---

## 📝 File Naming Conventions

```
Training Logs:    train_{task}.log
Model Files:      best_{task}_model_cnn.pt
Result Files:     {task}_results.json
SLURM Logs:       slurm_{job_id}.out/err
Prediction Files: test_pred.json
```

---

## ✅ Data Validation Checklist

Before training, ensure:

- [ ] Data files exist at specified paths
- [ ] JSON format is valid
- [ ] `len(text) == len(labels)` for all samples
- [ ] Labels match task configuration
- [ ] Files are UTF-8 encoded
- [ ] Train/val/test splits are separate
- [ ] No data leakage between splits
- [ ] Reasonable dataset size (>1000 samples recommended)

---

This structure provides a complete overview of how data flows through the system and where everything is stored.
