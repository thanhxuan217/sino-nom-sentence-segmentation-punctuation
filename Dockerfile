# =============================================================================
# Sino-Nom Sentence Segmentation & Punctuation — API Server
# =============================================================================
# Build:
#   docker build -t sinonom-api .
#
# Run (CPU):
#   docker run -p 8000:8000 \
#     -v /path/to/models:/app/models \
#     sinonom-api
#
# Run (GPU — requires nvidia-container-toolkit):
#   docker run --gpus all -p 8000:8000 \
#     -v /path/to/models:/app/models \
#     sinonom-api
# =============================================================================

FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .

# Install PyTorch CPU (lighter image) + project dependencies + FastAPI
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision && \
    pip install --no-cache-dir \
    transformers tokenizers datasets scikit-learn tqdm \
    accelerate numpy pytorch-crf peft safetensors \
    fastapi uvicorn[standard] python-multipart pypdf python-docx

# Copy source code
COPY src/ ./src/
COPY api/ ./api/

# Copy model checkpoints (if baked into the image).
# If you prefer to mount models at runtime, comment this out
# and use: docker run -v /path/to/models:/app/models ...
# COPY models/ ./models/

# Expose API port
EXPOSE 8000

# Environment variables (can be overridden at runtime)
ENV MODEL_NAME="SIKU-BERT/sikubert" \
    TOKENIZER_NAME="" \
    MAX_LENGTH="256" \
    SEG_MODEL_PATH="models/final_segmentation_model_cnn" \
    PUNCT_MODEL_PATH="models/final_punctuation_model_cnn" \
    HEAD_TYPE="cnn" \
    CNN_NUM_FILTERS="256" \
    DROPOUT="0.1" \
    USE_QLORA="1"

# Run the API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
