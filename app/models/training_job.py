"""
TrainingJob model for tracking asynchronous training jobs.
"""

from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import String, ForeignKey, Text, DateTime, Float, Integer, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class TrainingJob(Base, TimestampMixin):
    """Training job entity for tracking asynchronous training execution."""
    
    __tablename__ = "training_jobs"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        default="queued",
        nullable=False,
        index=True,
    )
    template: Mapped[str] = mapped_column(String(100), nullable=False)
    config: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    
    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Metrics summary
    total_episodes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_steps: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    avg_reward: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    final_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    training_duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Additional metadata
    metrics_summary: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    
    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="training_jobs")
    model_version: Mapped[Optional["ModelVersion"]] = relationship(
        "ModelVersion",
        back_populates="training_job",
        uselist=False,
    )
    
    def __repr__(self) -> str:
        return f"<TrainingJob(id={self.id}, status={self.status}, template={self.template})>"
    
    @property
    def is_active(self) -> bool:
        """Check if job is still active (queued or running)."""
        return self.status in ("queued", "running")
    
    @property
    def is_completed(self) -> bool:
        """Check if job completed successfully."""
        return self.status == "completed"
    
    @property
    def is_failed(self) -> bool:
        """Check if job failed."""
        return self.status == "failed"
