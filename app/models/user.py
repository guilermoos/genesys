"""
User model for authentication and project ownership.
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import String, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class User(Base, TimestampMixin):
    """User entity for authentication and project ownership."""
    
    __tablename__ = "users"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    api_key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    projects: Mapped[List["Project"]] = relationship(
        "Project",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email})>"
