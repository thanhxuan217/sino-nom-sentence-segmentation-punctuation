#!/usr/bin/env python3
"""
FastAPI application for Sino-Nom Sentence Segmentation & Punctuation.

Endpoints:
    POST /segment   — Sentence segmentation
    POST /punctuate — Sentence punctuation
    GET  /health    — Health check
"""

import logging
import os
import traceback
import unicodedata
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from api.inference import ModelManager
from api.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION (via environment variables with sensible defaults)
# ============================================================================

MODEL_NAME = os.getenv("MODEL_NAME", "pretrained/sikubert")
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", None)
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))
OVERLAP = int(os.getenv("OVERLAP", "128"))

SEG_MODEL_PATH = os.getenv(
    "SEG_MODEL_PATH", "models/final_segmentation_model_cnn"
)
PUNCT_MODEL_PATH = os.getenv(
    "PUNCT_MODEL_PATH", "models/final_punctuation_model_cnn"
)

HEAD_TYPE = os.getenv("HEAD_TYPE", "cnn")
CNN_NUM_FILTERS = int(os.getenv("CNN_NUM_FILTERS", "256"))
DROPOUT = float(os.getenv("DROPOUT", "0.1"))

USE_QLORA = os.getenv("USE_QLORA", "1") == "1"
LORA_R = int(os.getenv("LORA_R", "16"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.1"))
LORA_TARGET_MODULES = os.getenv("LORA_TARGET_MODULES", "query,key,value").split(",")

MAX_INPUT_TEXT_LENGTH = 2048


# ============================================================================
# MODEL MANAGERS (module-level singletons, initialised at startup)
# ============================================================================

seg_manager = ModelManager(
    task_name="segmentation",
    model_path=SEG_MODEL_PATH,
    model_name=MODEL_NAME,
    tokenizer_name=TOKENIZER_NAME,
    max_length=MAX_LENGTH,
    overlap=OVERLAP,
    head_type=HEAD_TYPE,
    cnn_num_filters=CNN_NUM_FILTERS,
    dropout=DROPOUT,
    use_qlora=USE_QLORA,
    lora_r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    lora_target_modules=LORA_TARGET_MODULES,
)

punct_manager = ModelManager(
    task_name="punctuation",
    model_path=PUNCT_MODEL_PATH,
    model_name=MODEL_NAME,
    tokenizer_name=TOKENIZER_NAME,
    max_length=MAX_LENGTH,
    overlap=OVERLAP,
    head_type=HEAD_TYPE,
    cnn_num_filters=CNN_NUM_FILTERS,
    dropout=DROPOUT,
    use_qlora=USE_QLORA,
    lora_r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    lora_target_modules=LORA_TARGET_MODULES,
)


# ============================================================================
# LIFESPAN (load models at startup)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models when the application starts up."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger.info("Starting model loading …")

    try:
        seg_manager.load()
    except Exception:
        logger.error(f"Failed to load segmentation model:\n{traceback.format_exc()}")

    try:
        punct_manager.load()
    except Exception:
        logger.error(f"Failed to load punctuation model:\n{traceback.format_exc()}")

    yield  # Application is running

    logger.info("Shutting down.")


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Sino-Nom Sentence Segmentation & Punctuation API",
    description=(
        "API for classical Chinese (Sino-Nom) text processing.\n\n"
        "Two endpoints are provided:\n"
        "- **POST /segment** — splits raw text into words/phrases\n"
        "- **POST /punctuate** — inserts punctuation marks into raw text"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error. Please try again later."},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"message": "Vui lòng cung cấp trường 'text' trong request body."},
    )

# ============================================================================
# UTILITIES
# ============================================================================

def preprocess_input_text(text: str) -> str:
    """Remove all punctuation marks and whitespace (including newlines) before inference.
    
    Strips every Unicode punctuation character (category 'P*') and all whitespace,
    keeping only CJK ideographs, letters, and digits.
    """
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith("P") and not ch.isspace()).strip()

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
async def health():
    """Returns the service health status and whether models are loaded."""
    return HealthResponse(
        status="ok",
        segmentation_model_loaded=seg_manager.is_loaded,
        punctuation_model_loaded=punct_manager.is_loaded,
    )


@app.post(
    "/segment",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"},
    },
    summary="Sentence segmentation",
    tags=["Inference"],
)
async def segment(
    text: str = Body(..., embed=True, description="Đoạn văn bản cần phân đoạn"),
):
    """Segment raw classical Chinese text into words / phrases.

    Returns the text with ` | ` separators inserted between segments,
    along with per-character BMES labels.
    """
    if not text.strip():
        return JSONResponse(
            status_code=400,
            content={"message": "Vui lòng cung cấp văn bản 'text'."}
        )

    if len(text) > MAX_INPUT_TEXT_LENGTH:
        return JSONResponse(
            status_code=400,
            content={"message": f"Văn bản quá dài (tối đa {MAX_INPUT_TEXT_LENGTH} ký tự)."}
        )

    if not seg_manager.is_loaded:
        return JSONResponse(
            status_code=503,
            content={"message": "Segmentation model is not loaded. Please try again later."}
        )

    input_text = preprocess_input_text(text.strip())
    if not input_text:
        return JSONResponse(
            status_code=400,
            content={"message": "Văn bản sau khi xử lý rỗng. Vui lòng cung cấp văn bản hợp lệ."}
        )

    try:
        result_text, labels = seg_manager.predict(input_text)
    except Exception as e:
        logger.error(f"Segmentation inference failed: {e}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"message": "Inference failed due to an internal error."}
        )

    return PredictionResponse(
        result=result_text,
        labels=labels
    )


@app.post(
    "/punctuate",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"},
    },
    summary="Sentence punctuation",
    tags=["Inference"],
)
async def punctuate(
    text: str = Body(..., embed=True, description="Đoạn văn bản cần thêm dấu câu"),
):
    """Insert punctuation marks into raw classical Chinese text.

    Returns the text with punctuation (，。：、；？！) inserted,
    along with per-character predicted labels.
    """
    if not text.strip():
        return JSONResponse(
            status_code=400,
            content={"message": "Vui lòng cung cấp văn bản 'text'."}
        )

    if len(text) > MAX_INPUT_TEXT_LENGTH:
        return JSONResponse(
            status_code=400,
            content={"message": f"Văn bản quá dài (tối đa {MAX_INPUT_TEXT_LENGTH} ký tự)."}
        )

    if not punct_manager.is_loaded:
        return JSONResponse(
            status_code=503,
            content={"message": "Punctuation model is not loaded. Please try again later."}
        )

    input_text = preprocess_input_text(text.strip())
    if not input_text:
        return JSONResponse(
            status_code=400,
            content={"message": "Văn bản sau khi xử lý rỗng. Vui lòng cung cấp văn bản hợp lệ."}
        )

    try:
        result_text, labels = punct_manager.predict(input_text)
    except Exception as e:
        logger.error(f"Punctuation inference failed: {e}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"message": "Inference failed due to an internal error."}
        )

    return PredictionResponse(
        result=result_text,
        labels=labels
    )
