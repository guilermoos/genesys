"""
Training job routes.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_user
from app.schemas.training_job import (
    TrainingJobCreate,
    TrainingJobResponse,
    TrainingJobListResponse,
    TrainingLogsResponse,
)
from app.services.training_service import TrainingService
from app.models.user import User

router = APIRouter()


@router.post("/projects/{project_id}/train", response_model=TrainingJobResponse, status_code=status.HTTP_201_CREATED)
def create_training_job(
    project_id: str,
    job_data: TrainingJobCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Create and queue a new training job.
    
    Args:
        project_id: Project ID
        job_data: Training job configuration
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Created training job
    """
    try:
        job = TrainingService.create_training_job(
            db,
            project_id,
            current_user.id,
            job_data,
        )
        return job
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("", response_model=TrainingJobListResponse)
def list_training_jobs(
    project_id: str = Query(None),
    status: str = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    List training jobs.
    
    Args:
        project_id: Filter by project
        status: Filter by status
        skip: Number of jobs to skip
        limit: Maximum number of jobs
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        List of training jobs
    """
    jobs, total = TrainingService.list_jobs(
        db,
        project_id=project_id,
        user_id=current_user.id,
        status=status,
        skip=skip,
        limit=limit,
    )
    
    return TrainingJobListResponse(
        items=[TrainingJobResponse.model_validate(j) for j in jobs],
        total=total,
        page=skip // limit + 1 if limit > 0 else 1,
        page_size=limit,
    )


@router.get("/{job_id}", response_model=TrainingJobResponse)
def get_training_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get training job by ID.
    
    Args:
        job_id: Job ID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Training job if found
    """
    job = TrainingService.get_job(db, job_id, current_user.id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training job not found",
        )
    
    return job


@router.get("/{job_id}/logs", response_model=TrainingLogsResponse)
def get_training_logs(
    job_id: str,
    lines: int = Query(100, ge=1, le=10000),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get training job logs.
    
    Args:
        job_id: Job ID
        lines: Number of log lines to retrieve
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Training job logs
    """
    logs = TrainingService.get_job_logs(db, job_id, current_user.id, lines)
    
    if logs is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training job not found",
        )
    
    return TrainingLogsResponse(
        job_id=job_id,
        logs=logs,
        total_lines=len(logs),
    )


@router.get("/{job_id}/metrics")
def get_training_metrics(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get training job metrics.
    
    Args:
        job_id: Job ID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Training metrics
    """
    metrics = TrainingService.get_job_metrics(db, job_id, current_user.id)
    
    if not metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training job not found",
        )
    
    return metrics


@router.post("/{job_id}/cancel")
def cancel_training_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Cancel a training job.
    
    Args:
        job_id: Job ID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Cancelled job
    """
    try:
        job = TrainingService.cancel_job(db, job_id, current_user.id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Training job not found",
            )
        
        return {"message": "Job cancelled successfully", "job": job}
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
