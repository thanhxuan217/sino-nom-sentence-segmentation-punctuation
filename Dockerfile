# =============================================================================
# Sino-Nom Sentence Segmentation & Punctuation — API Server (CPU Only)
# =============================================================================
# Build:
#   docker build -t sinonom-api .
# Run:
#   docker run -d -p 8000:8000 \
#     -v $(pwd)/models:/app/models \
#     -v $(pwd)/pretrained:/app/pretrained \
#     -e WORKERS=8 \ # Descrease workers if RAM is not enough
#     --name sinonom-api-container \
#     sinonom-api
#
#   *(On Windows PowerShell, use ${PWD} instead of $(pwd).)*
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements-cpu.txt .

# Install all dependencies (PyTorch CPU + project packages)
RUN pip install --no-cache-dir -r requirements-cpu.txt

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
ENV MODEL_NAME="pretrained/sikubert" \
    TOKENIZER_NAME="" \
    MAX_LENGTH="256" \
    SEG_MODEL_PATH="models/final_segmentation_model_cnn" \
    PUNCT_MODEL_PATH="models/final_punctuation_model_cnn" \
    HEAD_TYPE="cnn" \
    CNN_NUM_FILTERS="256" \
    DROPOUT="0.1" \
    USE_QLORA="1" \
    WORKERS="8"

# Run the API server with configurable workers
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers ${WORKERS}"]
