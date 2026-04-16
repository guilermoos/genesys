"""
Pydantic schemas for Project entity.
"""

from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field, ConfigDict


class ProjectBase(BaseModel):
    """Base project schema with common fields."""
    
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)


class ProjectCreate(ProjectBase):
    """Schema for creating a new project."""
    
    template_default: str = Field(..., min_length=1, max_length=100)


class ProjectUpdate(BaseModel):
    """Schema for updating a project."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    status: Optional[str] = Field(None, pattern="^(active|archived)$")


class ProjectResponse(ProjectBase):
    """Schema for project response."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: str
    user_id: str
    template_default: str
    status: str
    max_models: int
    created_at: datetime
    updated_at: datetime
    job_count: int = 0
    model_count: int = 0


class ProjectListResponse(BaseModel):
    """Schema for list of projects response."""
    
    items: List[ProjectResponse]
    total: int
    page: int
    page_size: int
