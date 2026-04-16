"""
InferenceLog model for tracking inference requests.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List

from sqlalchemy import String, ForeignKey, DateTime, Float, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class InferenceLog(Base):
    """Inference log entity for tracking prediction requests."""
    
    __tablename__ = "inference_logs"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_version_id: Mapped[str] = mapped_column(
        ForeignKey("model_versions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    
    # Request/Response data
    input_state: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    output_action: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Performance
    inference_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    
    # Additional metadata
    metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="inference_logs")
    model_version: Mapped[Optional["ModelVersion"]] = relationship(
        "ModelVersion",
        back_populates="inference_logs",
    )
    
    def __repr__(self) -> str:
        return f"<InferenceLog(id={self.id}, action={self.output_action}, time_ms={self.inference_time_ms})>"
