"""
ModelVersion model for tracking trained model artifacts.
"""

from typing import Optional, Dict, Any, List

from sqlalchemy import String, ForeignKey, Float, Integer, JSON, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class ModelVersion(Base, TimestampMixin):
    """Model version entity for tracking trained model artifacts."""
    
    __tablename__ = "model_versions"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    job_id: Mapped[str] = mapped_column(
        ForeignKey("training_jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    
    # Version info
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    
    # Artifact paths
    artifact_path: Mapped[str] = mapped_column(String(500), nullable=False)
    config_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    
    # Model metadata
    state_size: Mapped[int] = mapped_column(Integer, nullable=False)
    action_size: Mapped[int] = mapped_column(Integer, nullable=False)
    template: Mapped[str] = mapped_column(String(100), nullable=False)
    hyperparameters: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    
    # Performance metrics
    avg_reward: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_episodes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    training_duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="model_versions")
    training_job: Mapped[Optional["TrainingJob"]] = relationship(
        "TrainingJob",
        back_populates="model_version",
    )
    inference_logs: Mapped[List["InferenceLog"]] = relationship(
        "InferenceLog",
        back_populates="model_version",
        lazy="selectin",
    )
    
    def __repr__(self) -> str:
        return f"<ModelVersion(id={self.id}, version={self.version}, active={self.is_active})>"
