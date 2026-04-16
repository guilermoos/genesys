"""
Project model for organizing training jobs and models.
"""

from typing import List, Optional

from sqlalchemy import String, ForeignKey, Text, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class Project(Base, TimestampMixin):
    """Project entity for organizing training jobs and models."""
    
    __tablename__ = "projects"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    template_default: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="active", nullable=False)
    max_models: Mapped[int] = mapped_column(Integer, default=10, nullable=False)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="projects")
    training_jobs: Mapped[List["TrainingJob"]] = relationship(
        "TrainingJob",
        back_populates="project",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    model_versions: Mapped[List["ModelVersion"]] = relationship(
        "ModelVersion",
        back_populates="project",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    inference_logs: Mapped[List["InferenceLog"]] = relationship(
        "InferenceLog",
        back_populates="project",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    
    def __repr__(self) -> str:
        return f"<Project(id={self.id}, name={self.name}, template={self.template_default})>"
