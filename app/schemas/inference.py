"""
Pydantic schemas for Inference entity.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field, ConfigDict


class InferenceRequest(BaseModel):
    """Schema for inference request."""
    
    state: List[float] = Field(..., min_length=1)
    model_version_id: Optional[str] = Field(None, description="Specific model version to use. If not provided, uses active model.")
    metadata: Optional[Dict[str, Any]] = Field(default=None)
    
    @field_validator("state")
    @classmethod
    def validate_state(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("state cannot be empty")
        return v


class InferenceResponse(BaseModel):
    """Schema for inference response."""
    
    action: int
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    model_version_id: str
    model_version: int
    inference_time_ms: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class InferenceBatchRequest(BaseModel):
    """Schema for batch inference request."""
    
    states: List[List[float]] = Field(..., min_length=1, max_length=100)
    model_version_id: Optional[str] = None


class InferenceBatchResponse(BaseModel):
    """Schema for batch inference response."""
    
    actions: List[int]
    confidences: Optional[List[float]] = None
    model_version_id: str
    inference_time_ms: float
    timestamp: datetime
