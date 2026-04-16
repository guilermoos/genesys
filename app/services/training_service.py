"""
Training service for managing training jobs.
"""

from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.models.training_job import TrainingJob
from app.models.project import Project
from app.schemas.training_job import TrainingJobCreate, TrainingConfig
from app.utils.id_generator import generate_uuid
from app.templates.base import TemplateRegistry
from app.workers.training_tasks import run_training_job


class TrainingService:
    """Service for training job management."""
    
    @staticmethod
    def create_training_job(
        db: Session,
        project_id: str,
        user_id: str,
        job_data: TrainingJobCreate,
    ) -> TrainingJob:
        """
        Create and queue a new training job.
        
        Args:
            db: Database session
            project_id: Project ID
            user_id: User ID for access control
            job_data: Training job creation data
            
        Returns:
            Created training job
            
        Raises:
            ValueError: If project not found or template invalid
        """
        # Verify project exists and belongs to user
        project = db.query(Project).filter(
            Project.id == project_id,
            Project.user_id == user_id,
        ).first()
        
        if not project:
            raise ValueError("Project not found")
        
        # Validate template
        try:
            TemplateRegistry.get(job_data.template)
        except ValueError as e:
            raise ValueError(f"Invalid template: {e}")
        
        # Create job
        job = TrainingJob(
            id=generate_uuid(),
            project_id=project_id,
            status="queued",
            template=job_data.template,
            config=job_data.config.model_dump(),
        )
        
        db.add(job)
        db.commit()
        db.refresh(job)
        
        # Queue the job with Celery
        task = run_training_job.delay(job.id)
        
        # Store Celery task ID
        job.celery_task_id = task.id
        db.commit()
        
        return job
    
    @staticmethod
    def get_job(
        db: Session,
        job_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[TrainingJob]:
        """
        Get training job by ID.
        
        Args:
            db: Database session
            job_id: Job ID
            user_id: Optional user ID for access control
            
        Returns:
            Training job if found and accessible
        """
        query = db.query(TrainingJob).filter(TrainingJob.id == job_id)
        
        if user_id:
            query = query.join(Project).filter(Project.user_id == user_id)
        
        return query.first()
    
    @staticmethod
    def list_jobs(
        db: Session,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> Tuple[List[TrainingJob], int]:
        """
        List training jobs.
        
        Args:
            db: Database session
            project_id: Optional project filter
            user_id: Optional user filter
            status: Optional status filter
            skip: Number of jobs to skip
            limit: Maximum number of jobs
            
        Returns:
            Tuple of (jobs list, total count)
        """
        query = db.query(TrainingJob)
        
        if project_id:
            query = query.filter(TrainingJob.project_id == project_id)
        
        if user_id:
            query = query.join(Project).filter(Project.user_id == user_id)
        
        if status:
            query = query.filter(TrainingJob.status == status)
        
        total = query.count()
        jobs = (
            query.order_by(desc(TrainingJob.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )
        
        return jobs, total
    
    @staticmethod
    def cancel_job(
        db: Session,
        job_id: str,
        user_id: str,
    ) -> Optional[TrainingJob]:
        """
        Cancel a training job.
        
        Args:
            db: Database session
            job_id: Job ID
            user_id: User ID for access control
            
        Returns:
            Updated job if cancelled
        """
        job = TrainingService.get_job(db, job_id, user_id)
        
        if not job:
            return None
        
        if job.status not in ("queued", "running"):
            raise ValueError(f"Cannot cancel job with status: {job.status}")
        
        # Revoke Celery task if queued or running
        if job.celery_task_id:
            from celery import current_app
            current_app.control.revoke(job.celery_task_id, terminate=True)
        
        job.status = "cancelled"
        job.finished_at = datetime.utcnow()
        
        db.commit()
        db.refresh(job)
        
        return job
    
    @staticmethod
    def get_job_logs(
        db: Session,
        job_id: str,
        user_id: str,
        lines: int = 100,
    ) -> Optional[List[str]]:
        """
        Get training job logs.
        
        Args:
            db: Database session
            job_id: Job ID
            user_id: User ID for access control
            lines: Number of log lines to retrieve
            
        Returns:
            List of log lines if job found
        """
        job = TrainingService.get_job(db, job_id, user_id)
        
        if not job:
            return None
        
        # Read log file if exists
        log_path = f"./storage/logs/job_{job_id}.log"
        
        try:
            with open(log_path, "r") as f:
                all_lines = f.readlines()
                return all_lines[-lines:] if len(all_lines) > lines else all_lines
        except FileNotFoundError:
            return []
    
    @staticmethod
    def update_job_status(
        db: Session,
        job_id: str,
        status: str,
        metrics: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> Optional[TrainingJob]:
        """
        Update job status (used by workers).
        
        Args:
            db: Database session
            job_id: Job ID
            status: New status
            metrics: Optional metrics to update
            error_message: Optional error message
            
        Returns:
            Updated job
        """
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        
        if not job:
            return None
        
        job.status = status
        
        if status == "running" and not job.started_at:
            job.started_at = datetime.utcnow()
        
        if status in ("completed", "failed", "cancelled"):
            job.finished_at = datetime.utcnow()
        
        if metrics:
            job.total_episodes = metrics.get("total_episodes")
            job.total_steps = metrics.get("total_steps")
            job.avg_reward = metrics.get("avg_reward")
            job.final_loss = metrics.get("final_loss")
            job.training_duration_seconds = metrics.get("training_duration_seconds")
            job.metrics_summary = metrics
        
        if error_message:
            job.error_message = error_message
        
        db.commit()
        db.refresh(job)
        
        return job
    
    @staticmethod
    def get_job_metrics(
        db: Session,
        job_id: str,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed metrics for a training job.
        
        Args:
            db: Database session
            job_id: Job ID
            user_id: User ID for access control
            
        Returns:
            Job metrics if found
        """
        job = TrainingService.get_job(db, job_id, user_id)
        
        if not job:
            return None
        
        return {
            "job_id": job.id,
            "status": job.status,
            "template": job.template,
            "total_episodes": job.total_episodes,
            "total_steps": job.total_steps,
            "avg_reward": job.avg_reward,
            "final_loss": job.final_loss,
            "training_duration_seconds": job.training_duration_seconds,
            "metrics_summary": job.metrics_summary,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
        }
