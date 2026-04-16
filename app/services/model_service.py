"""
Model service for managing model versions.
"""

from typing import List, Optional, Tuple
from datetime import datetime
import os

from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.models.model_version import ModelVersion
from app.models.project import Project
from app.models.training_job import TrainingJob
from app.utils.id_generator import generate_uuid
from app.utils.config import get_settings


class ModelService:
    """Service for model version management."""
    
    @staticmethod
    def create_model_version(
        db: Session,
        project_id: str,
        job_id: str,
        name: str,
        artifact_path: str,
        config: dict,
        metrics: dict,
        description: Optional[str] = None,
    ) -> ModelVersion:
        """
        Create a new model version.
        
        Args:
            db: Database session
            project_id: Project ID
            job_id: Training job ID
            name: Model name
            artifact_path: Path to model artifact
            config: Model configuration
            metrics: Training metrics
            description: Optional description
            
        Returns:
            Created model version
        """
        # Get next version number
        latest_version = (
            db.query(ModelVersion)
            .filter(ModelVersion.project_id == project_id)
            .order_by(desc(ModelVersion.version))
            .first()
        )
        
        version_number = 1 if not latest_version else latest_version.version + 1
        
        # Get file size
        file_size = os.path.getsize(artifact_path) if os.path.exists(artifact_path) else None
        
        model = ModelVersion(
            id=generate_uuid(),
            project_id=project_id,
            job_id=job_id,
            version=version_number,
            name=name,
            description=description,
            artifact_path=artifact_path,
            state_size=config.get("state_size"),
            action_size=config.get("action_size"),
            template=config.get("template"),
            hyperparameters=config.get("hyperparameters", {}),
            avg_reward=metrics.get("avg_reward"),
            total_episodes=metrics.get("total_episodes"),
            training_duration_seconds=metrics.get("training_duration_seconds"),
            is_active=False,
            file_size_bytes=file_size,
        )
        
        db.add(model)
        db.commit()
        db.refresh(model)
        
        return model
    
    @staticmethod
    def get_model(
        db: Session,
        model_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[ModelVersion]:
        """
        Get model version by ID.
        
        Args:
            db: Database session
            model_id: Model ID
            user_id: Optional user ID for access control
            
        Returns:
            Model version if found
        """
        query = db.query(ModelVersion).filter(ModelVersion.id == model_id)
        
        if user_id:
            query = query.join(Project).filter(Project.user_id == user_id)
        
        return query.first()
    
    @staticmethod
    def list_models(
        db: Session,
        project_id: str,
        user_id: str,
        skip: int = 0,
        limit: int = 100,
    ) -> Tuple[List[ModelVersion], int]:
        """
        List model versions for a project.
        
        Args:
            db: Database session
            project_id: Project ID
            user_id: User ID for access control
            skip: Number of models to skip
            limit: Maximum number of models
            
        Returns:
            Tuple of (models list, total count)
        """
        # Verify project access
        project = db.query(Project).filter(
            Project.id == project_id,
            Project.user_id == user_id,
        ).first()
        
        if not project:
            return [], 0
        
        query = db.query(ModelVersion).filter(ModelVersion.project_id == project_id)
        
        total = query.count()
        models = (
            query.order_by(desc(ModelVersion.version))
            .offset(skip)
            .limit(limit)
            .all()
        )
        
        return models, total
    
    @staticmethod
    def activate_model(
        db: Session,
        model_id: str,
        user_id: str,
    ) -> Optional[ModelVersion]:
        """
        Activate a model version for inference.
        
        Args:
            db: Database session
            model_id: Model ID
            user_id: User ID for access control
            
        Returns:
            Activated model
        """
        model = ModelService.get_model(db, model_id, user_id)
        
        if not model:
            return None
        
        # Deactivate other models in the same project
        db.query(ModelVersion).filter(
            ModelVersion.project_id == model.project_id,
            ModelVersion.is_active == True,
        ).update({"is_active": False})
        
        # Activate this model
        model.is_active = True
        model.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(model)
        
        return model
    
    @staticmethod
    def get_active_model(
        db: Session,
        project_id: str,
        user_id: str,
    ) -> Optional[ModelVersion]:
        """
        Get active model for a project.
        
        Args:
            db: Database session
            project_id: Project ID
            user_id: User ID for access control
            
        Returns:
            Active model if exists
        """
        # Verify project access
        project = db.query(Project).filter(
            Project.id == project_id,
            Project.user_id == user_id,
        ).first()
        
        if not project:
            return None
        
        return (
            db.query(ModelVersion)
            .filter(
                ModelVersion.project_id == project_id,
                ModelVersion.is_active == True,
            )
            .first()
        )
    
    @staticmethod
    def delete_model(
        db: Session,
        model_id: str,
        user_id: str,
    ) -> bool:
        """
        Delete a model version.
        
        Args:
            db: Database session
            model_id: Model ID
            user_id: User ID for access control
            
        Returns:
            True if deleted
        """
        model = ModelService.get_model(db, model_id, user_id)
        
        if not model:
            return False
        
        # Delete artifact file
        if os.path.exists(model.artifact_path):
            os.remove(model.artifact_path)
        
        db.delete(model)
        db.commit()
        
        return True
    
    @staticmethod
    def get_model_download_url(
        db: Session,
        model_id: str,
        user_id: str,
    ) -> Optional[str]:
        """
        Get download URL for a model.
        
        Args:
            db: Database session
            model_id: Model ID
            user_id: User ID for access control
            
        Returns:
            Download URL if model exists
        """
        model = ModelService.get_model(db, model_id, user_id)
        
        if not model:
            return None
        
        # For local files, return the path
        # In production, this would generate a signed URL
        return model.artifact_path
