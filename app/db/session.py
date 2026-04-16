"""
Database session management.
"""

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.models.base import Base
from app.utils.config import get_settings


# Create engine
settings = get_settings()
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
    pool_pre_ping=True,
    echo=settings.DEBUG,
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def init_db() -> None:
    """Initialize database by creating all tables."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Get database session.
    
    Yields:
        SQLAlchemy session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
