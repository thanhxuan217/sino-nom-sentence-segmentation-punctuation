# Sino-Nom Sentence Segmentation & Punctuation API

This directory contains the FastAPI service for serving the trained Sino-Nom models (segmentation and punctuation).

## 📥 1. Download Pretrained and Fine-tuned Models

Before running the API, you need to set up the model checkpoints. You can download the required models from the provided Google Drive link:

**[Download Models from Google Drive](https://drive.google.com/drive/folders/1DnTpxkqu5hQDQ9uDYrwWrjO-QX7CnbVT?usp=sharing)**

Once downloaded, extract the contents and place them in the correct directories as instructed below.

### Setup Pretrained Backbone (SikuBERT)

The API loads the base transformer model directly from a local folder instead of Hugging Face. Place the pretrained `sikubert` files into the `pretrained/` directory at the project root:

```
pretrained/sikubert/
├── config.json
├── pytorch_model.bin
├── tokenizer.json
├── tokenizer_config.json
└── vocab.txt
```

### Setup Fine-tuned Checkpoints

Place your fine-tuned model checkpoints (PyTorch `.pt` files containing adapter weights & CNN head) into the `models/` directory at the project root:

```
models/
├── final_segmentation_model_cnn
└── final_punctuation_model_cnn
```

*Note: The exact paths can be configured via environment variables if desired.*

---

## 🚀 2. Running Locally (Without Docker)

### Install Dependencies

Make sure you are in the project root directory, then install the required packages:

```bash
# Optional: Create and activate virtual environment
conda create -n sinonom python=3.11
conda activate sinonom

# Install requirements
pip install -r requirements.txt
```

### Start the API

**Production** — run with multiple workers to handle concurrent requests (each worker loads its own model, increasing RAM usage):

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 16
```

> ⚠️ `--reload` is **incompatible** with `--workers`. Use `--reload` only for development (single worker).

**Development** (auto-reload, 1 worker):
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🐳 3. Running with Docker

Docker is the easiest way to run the API without installing Python dependencies locally.

### Build the Image

Run this from the **project root**:

```bash
docker build -t sinonom-api .
```

### Run the Container

You must mount the `models/` and `pretrained/` directories into the container so the API can access the weights:

```bash
docker run -d -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/pretrained:/app/pretrained \
  --name sinonom-api-container \
  sinonom-api
```

*(On Windows PowerShell, use `${PWD}` instead of `$(pwd)`.)*

---

## ⚙️ 4. Configuration (Environment Variables)

The API is fully configurable. Defaults are set up to work out-of-the-box if you follow the folder structure above.

| Variable | Default Value | Description |
|---|---|---|
| `MODEL_NAME` | `pretrained/sikubert` | Path to the base backbone. |
| `SEG_MODEL_PATH` | `models/final_segmentation_model_cnn` | Path to segmentation checkpoint. |
| `PUNCT_MODEL_PATH` | `models/final_punctuation_model_cnn` | Path to punctuation checkpoint. |
| `MAX_LENGTH` | `256` | Max sequence length for inference. |
| `OVERLAP` | `128` | Sliding window overlap tokens. |
| `HEAD_TYPE` | `cnn` | Type of classification head used. |
| `USE_QLORA` | `1` | Set to `1` to enable QLoRA inference. |
| `WORKERS` | `1` | Number of Uvicorn workers. |

You can pass these into Docker using `-e`:
```bash
docker run -p 8000:8000 -e WORKERS=8 -v $(pwd)/models:/app/models -v $(pwd)/pretrained:/app/pretrained sinonom-api
```

---

## 📡 5. API Endpoints Usage

The API provides Swagger UI at `http://localhost:8000/docs` where you can naturally test the endpoints.

### Health Check

```bash
curl http://localhost:8000/health
```

### Segmentation Task (`POST /segment`)

**Process plain text:**
```bash
curl -X POST "http://localhost:8000/segment" \
     -F "text=天地玄黃宇宙洪荒"
```

**Process a file (.txt, .docx):**
```bash
curl -X POST "http://localhost:8000/segment" \
     -F "file=@/path/to/document.docx"
```

### Punctuation Task (`POST /punctuate`)

**Process plain text:**
```bash
curl -X POST "http://localhost:8000/punctuate" \
     -F "text=天地玄黃宇宙洪荒"
```

**Process a file (.txt, .docx):**
```bash
curl -X POST "http://localhost:8000/punctuate" \
     -F "file=@/path/to/document.docx"
```
