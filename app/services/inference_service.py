"""
Inference service for running predictions with trained models.
"""

from typing import Optional, List
from datetime import datetime
import time

from sqlalchemy.orm import Session
import numpy as np
import torch

from app.models.model_version import ModelVersion
from app.models.project import Project
from app.models.inference_log import InferenceLog
from app.rl.agent import DQNAgent
from app.services.model_service import ModelService
from app.utils.id_generator import generate_uuid


class InferenceService:
    """Service for model inference."""
    
    # Cache for loaded models
    _model_cache: dict = {}
    
    @staticmethod
    def predict(
        db: Session,
        project_id: str,
        user_id: str,
        state: List[float],
        model_version_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Run inference on a state.
        
        Args:
            db: Database session
            project_id: Project ID
            user_id: User ID for access control
            state: Input state vector
            model_version_id: Specific model version (uses active if not provided)
            metadata: Optional metadata to log
            
        Returns:
            Prediction result with action and metadata
            
        Raises:
            ValueError: If model not found or inference fails
        """
        start_time = time.time()
        
        # Get model
        if model_version_id:
            model = ModelService.get_model(db, model_version_id, user_id)
            if not model or model.project_id != project_id:
                raise ValueError("Model not found or does not belong to project")
        else:
            model = ModelService.get_active_model(db, project_id, user_id)
            if not model:
                raise ValueError("No active model found for project")
        
        # Load agent
        agent = InferenceService._load_agent(model)
        
        # Validate state size
        if len(state) != model.state_size:
            raise ValueError(
                f"Invalid state size. Expected {model.state_size}, got {len(state)}"
            )
        
        # Run inference
        state_array = np.array(state, dtype=np.float32)
        
        with torch.no_grad():
            q_values = agent.get_q_values(state_array)
            action = int(np.argmax(q_values))
            confidence = float(
                (q_values[action] - q_values.min()) / 
                (q_values.max() - q_values.min() + 1e-8)
            )
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Log inference
        log = InferenceLog(
            id=generate_uuid(),
            project_id=project_id,
            model_version_id=model.id,
            input_state={"state": state},
            output_action=action,
            confidence=confidence,
            inference_time_ms=inference_time_ms,
            timestamp=datetime.utcnow(),
            extra_metadata=metadata,
        )
        
        db.add(log)
        db.commit()
        
        return {
            "action": action,
            "confidence": confidence,
            "model_version_id": model.id,
            "model_version": model.version,
            "inference_time_ms": inference_time_ms,
            "timestamp": log.timestamp,
            "q_values": q_values.tolist(),
        }
    
    @staticmethod
    def predict_batch(
        db: Session,
        project_id: str,
        user_id: str,
        states: List[List[float]],
        model_version_id: Optional[str] = None,
    ) -> dict:
        """
        Run batch inference.
        
        Args:
            db: Database session
            project_id: Project ID
            user_id: User ID for access control
            states: List of input state vectors
            model_version_id: Specific model version
            
        Returns:
            Batch prediction results
        """
        start_time = time.time()
        
        # Get model
        if model_version_id:
            model = ModelService.get_model(db, model_version_id, user_id)
            if not model or model.project_id != project_id:
                raise ValueError("Model not found")
        else:
            model = ModelService.get_active_model(db, project_id, user_id)
            if not model:
                raise ValueError("No active model found")
        
        # Load agent
        agent = InferenceService._load_agent(model)
        
        # Run batch inference
        states_array = np.array(states, dtype=np.float32)
        
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states_array).to(agent.device)
            q_values = agent.q_network(states_tensor)
            actions = q_values.argmax(dim=1).cpu().numpy()
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        return {
            "actions": actions.tolist(),
            "model_version_id": model.id,
            "inference_time_ms": inference_time_ms,
            "timestamp": datetime.utcnow(),
        }
    
    @staticmethod
    def _load_agent(model: ModelVersion) -> DQNAgent:
        """
        Load DQN agent from model version.
        
        Args:
            model: Model version
            
        Returns:
            Loaded DQN agent
        """
        cache_key = model.id
        
        # Check cache
        if cache_key in InferenceService._model_cache:
            return InferenceService._model_cache[cache_key]
        
        # Create agent
        hyperparams = model.hyperparameters
        
        agent = DQNAgent(
            state_size=model.state_size,
            action_size=model.action_size,
            hidden_layers=hyperparams.get("hidden_layers", [128, 128]),
            learning_rate=hyperparams.get("learning_rate", 0.001),
            gamma=hyperparams.get("gamma", 0.99),
        )
        
        # Load weights
        agent.load(model.artifact_path)
        
        # Cache model
        InferenceService._model_cache[cache_key] = agent
        
        return agent
    
    @staticmethod
    def clear_cache() -> None:
        """Clear model cache."""
        InferenceService._model_cache.clear()
    
    @staticmethod
    def get_inference_stats(
        db: Session,
        project_id: str,
        user_id: str,
        hours: int = 24,
    ) -> dict:
        """
        Get inference statistics for a project.
        
        Args:
            db: Database session
            project_id: Project ID
            user_id: User ID for access control
            hours: Time window in hours
            
        Returns:
            Inference statistics
        """
        # Verify project access
        project = db.query(Project).filter(
            Project.id == project_id,
            Project.user_id == user_id,
        ).first()
        
        if not project:
            return {}
        
        from sqlalchemy import func
        
        # Get stats
        stats = (
            db.query(
                func.count(InferenceLog.id).label("total_requests"),
                func.avg(InferenceLog.inference_time_ms).label("avg_time_ms"),
            )
            .filter(
                InferenceLog.project_id == project_id,
                InferenceLog.timestamp >= datetime.utcnow() - __import__("datetime").timedelta(hours=hours),
            )
            .first()
        )
        
        return {
            "total_requests": stats.total_requests or 0,
            "avg_inference_time_ms": round(stats.avg_time_ms, 2) if stats.avg_time_ms else 0,
            "time_window_hours": hours,
        }
