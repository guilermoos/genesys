"""
Pydantic schemas for request/response validation.
"""

from app.schemas.user import UserCreate, UserResponse, UserLogin
from app.schemas.project import ProjectCreate, ProjectResponse, ProjectUpdate
from app.schemas.training_job import (
    TrainingJobCreate,
    TrainingJobResponse,
    TrainingJobStatus,
    TrainingConfig,
)
from app.schemas.model_version import (
    ModelVersionResponse,
    ModelVersionActivate,
    ModelVersionCreate,
)
from app.schemas.inference import InferenceRequest, InferenceResponse

__all__ = [
    "UserCreate",
    "UserResponse",
    "UserLogin",
    "ProjectCreate",
    "ProjectResponse",
    "ProjectUpdate",
    "TrainingJobCreate",
    "TrainingJobResponse",
    "TrainingJobStatus",
    "TrainingConfig",
    "ModelVersionResponse",
    "ModelVersionActivate",
    "ModelVersionCreate",
    "InferenceRequest",
    "InferenceResponse",
]
