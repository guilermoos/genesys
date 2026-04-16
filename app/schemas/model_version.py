"""
Pydantic schemas for ModelVersion entity.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field, ConfigDict


class ModelVersionCreate(BaseModel):
    """Schema for creating a new model version (internal use)."""
    
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=500)


class ModelVersionActivate(BaseModel):
    """Schema for activating a model version."""
    
    pass  # No body needed, just the POST action


class ModelVersionResponse(BaseModel):
    """Schema for model version response."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: str
    project_id: str
    job_id: Optional[str] = None
    version: int
    name: str
    description: Optional[str] = None
    
    # Paths
    artifact_path: str
    config_path: Optional[str] = None
    
    # Model metadata
    state_size: int
    action_size: int
    template: str
    hyperparameters: Dict[str, Any]
    
    # Performance metrics
    avg_reward: Optional[float] = None
    total_episodes: Optional[int] = None
    training_duration_seconds: Optional[float] = None
    
    # Status
    is_active: bool
    file_size_bytes: Optional[int] = None
    
    created_at: datetime
    updated_at: datetime


class ModelVersionListResponse(BaseModel):
    """Schema for list of model versions response."""
    
    items: List[ModelVersionResponse]
    total: int
    page: int
    page_size: int


class ModelDownloadResponse(BaseModel):
    """Schema for model download response."""
    
    download_url: str
    expires_at: datetime
    file_size_bytes: int
