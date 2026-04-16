"""
Pydantic schemas for TrainingJob entity.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field, ConfigDict, field_validator


class TrainingConfig(BaseModel):
    """Schema for training configuration."""
    
    # Environment config
    state_size: int = Field(..., ge=1, le=10000)
    action_space: List[int] = Field(..., min_length=1)
    
    # Training hyperparameters
    episodes: int = Field(default=1000, ge=10, le=1000000)
    max_steps: int = Field(default=500, ge=10, le=10000)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    learning_rate: float = Field(default=0.001, ge=0.000001, le=1.0)
    epsilon_start: float = Field(default=1.0, ge=0.0, le=1.0)
    epsilon_end: float = Field(default=0.01, ge=0.0, le=1.0)
    epsilon_decay: float = Field(default=0.995, ge=0.0, le=1.0)
    batch_size: int = Field(default=64, ge=8, le=512)
    memory_size: int = Field(default=10000, ge=1000, le=1000000)
    target_update_freq: int = Field(default=100, ge=1, le=10000)
    
    # Resource limits
    max_workers: int = Field(default=1, ge=1, le=8)
    use_gpu: bool = Field(default=False)
    
    # Template-specific config
    env_config: Optional[Dict[str, Any]] = Field(default=None)
    reward_config: Optional[Dict[str, Any]] = Field(default=None)
    
    @field_validator("action_space")
    @classmethod
    def validate_action_space(cls, v: List[int]) -> List[int]:
        if len(v) != len(set(v)):
            raise ValueError("action_space must contain unique values")
        return sorted(v)


class TrainingJobCreate(BaseModel):
    """Schema for creating a new training job."""
    
    template: str = Field(..., min_length=1, max_length=100)
    config: TrainingConfig
    name: Optional[str] = Field(None, max_length=255)


class TrainingJobStatus(BaseModel):
    """Schema for training job status."""
    
    status: str  # queued, running, completed, failed, cancelled
    progress: Optional[float] = Field(None, ge=0.0, le=100.0)
    current_episode: Optional[int] = None
    current_step: Optional[int] = None
    message: Optional[str] = None


class TrainingMetrics(BaseModel):
    """Schema for training metrics."""
    
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    losses: List[float] = []
    epsilon_history: List[float] = []
    avg_reward_100: Optional[float] = None
    best_reward: Optional[float] = None


class TrainingJobResponse(BaseModel):
    """Schema for training job response."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: str
    project_id: str
    status: str
    template: str
    config: Dict[str, Any]
    name: Optional[str] = None
    
    # Timing
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    
    # Metrics summary
    total_episodes: Optional[int] = None
    total_steps: Optional[int] = None
    avg_reward: Optional[float] = None
    final_loss: Optional[float] = None
    training_duration_seconds: Optional[float] = None
    metrics_summary: Optional[Dict[str, Any]] = None
    
    # Error info
    error_message: Optional[str] = None
    
    created_at: datetime
    updated_at: datetime


class TrainingJobListResponse(BaseModel):
    """Schema for list of training jobs response."""
    
    items: List[TrainingJobResponse]
    total: int
    page: int
    page_size: int


class TrainingLogsResponse(BaseModel):
    """Schema for training logs response."""
    
    job_id: str
    logs: List[str]
    total_lines: int
