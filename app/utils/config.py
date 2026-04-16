"""
Configuration management for Genesys platform.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    APP_NAME: str = "Genesys"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    
    # API
    API_V1_PREFIX: str = "/v1"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 1 day
    API_KEY_LENGTH: int = 32
    
    # Database
    DATABASE_URL: str = "sqlite:///./storage/genesys.db"
    
    # Storage
    STORAGE_PATH: str = "./storage"
    MODELS_PATH: str = "./storage/models"
    LOGS_PATH: str = "./storage/logs"
    
    # Redis (for Celery)
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # Training
    MAX_CONCURRENT_JOBS: int = 2
    DEFAULT_EPISODES: int = 1000
    MAX_EPISODES: int = 100000
    
    # Model Management
    MAX_MODELS_PER_PROJECT: int = 10
    MODEL_VERSIONING_ENABLED: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        Path(self.STORAGE_PATH).mkdir(parents=True, exist_ok=True)
        Path(self.MODELS_PATH).mkdir(parents=True, exist_ok=True)
        Path(self.LOGS_PATH).mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings
