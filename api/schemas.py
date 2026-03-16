#!/usr/bin/env python3
"""
Pydantic schemas for the Sino-Nom Sentence Segmentation & Punctuation API.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class TextPredictionResponse(BaseModel):
    """Successful prediction response for text input."""

    status: str = Field(default="success", description="Status of the request", examples=["success"])
    input_type: str = Field(default="text", description="Type of input", examples=["text"])
    result: str = Field(
        ...,
        description="Text with predicted segmentation separators or punctuation inserted.",
    )
    labels: List[str] = Field(
        ...,
        description="Per-character predicted labels.",
    )


class FilePredictionResponse(BaseModel):
    """Successful prediction response for file input."""

    status: str = Field(default="success", description="Status of the request", examples=["success"])
    input_type: str = Field(default="file", description="Type of input", examples=["file"])
    filename: str = Field(
        ...,
        description="Name of the uploaded file.",
    )
    result: str = Field(
        ...,
        description="Text with predicted segmentation separators or punctuation inserted.",
    )


class ErrorResponse(BaseModel):
    """Error response body."""

    message: str = Field(..., description="Error message.")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status.", examples=["ok"])
    segmentation_model_loaded: bool = Field(
        ..., description="Whether the segmentation model is loaded."
    )
    punctuation_model_loaded: bool = Field(
        ..., description="Whether the punctuation model is loaded."
    )
