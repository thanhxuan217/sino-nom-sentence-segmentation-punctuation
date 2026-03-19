#!/usr/bin/env python3
"""
Pydantic schemas for the Sino-Nom Sentence Segmentation & Punctuation API.
"""

from typing import List

from pydantic import BaseModel, Field


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class PredictionResponse(BaseModel):
    """Successful prediction response."""

    result: str = Field(
        ...,
        description="Text with predicted segmentation separators or punctuation inserted.",
    )
    labels: List[str] = Field(
        ...,
        description="Per-character predicted labels.",
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
