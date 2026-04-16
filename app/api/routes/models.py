"""
Model routes.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_user
from app.schemas.model_version import (
    ModelVersionResponse,
    ModelVersionListResponse,
)
from app.services.model_service import ModelService
from app.models.user import User

router = APIRouter()


@router.get("/projects/{project_id}/models", response_model=ModelVersionListResponse)
def list_models(
    project_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    List model versions for a project.
    
    Args:
        project_id: Project ID
        skip: Number of models to skip
        limit: Maximum number of models
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        List of model versions
    """
    models, total = ModelService.list_models(
        db,
        project_id,
        current_user.id,
        skip=skip,
        limit=limit,
    )
    
    return ModelVersionListResponse(
        items=[ModelVersionResponse.model_validate(m) for m in models],
        total=total,
        page=skip // limit + 1 if limit > 0 else 1,
        page_size=limit,
    )


@router.get("/{model_id}", response_model=ModelVersionResponse)
def get_model(
    model_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get model version by ID.
    
    Args:
        model_id: Model ID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Model version if found
    """
    model = ModelService.get_model(db, model_id, current_user.id)
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )
    
    return model


@router.post("/{model_id}/activate")
def activate_model(
    model_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Activate a model version for inference.
    
    Args:
        model_id: Model ID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Activated model
    """
    model = ModelService.activate_model(db, model_id, current_user.id)
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )
    
    return {
        "message": "Model activated successfully",
        "model": ModelVersionResponse.model_validate(model),
    }


@router.get("/projects/{project_id}/models/active")
def get_active_model(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get active model for a project.
    
    Args:
        project_id: Project ID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Active model if exists
    """
    model = ModelService.get_active_model(db, project_id, current_user.id)
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active model found for this project",
        )
    
    return ModelVersionResponse.model_validate(model)


@router.get("/{model_id}/download")
def download_model(
    model_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Download model artifact.
    
    Args:
        model_id: Model ID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Model file
    """
    model = ModelService.get_model(db, model_id, current_user.id)
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )
    
    import os
    if not os.path.exists(model.artifact_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model file not found",
        )
    
    return FileResponse(
        model.artifact_path,
        filename=f"model_{model.version}.pt",
        media_type="application/octet-stream",
    )


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model(
    model_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Delete a model version.
    
    Args:
        model_id: Model ID
        current_user: Current authenticated user
        db: Database session
    """
    deleted = ModelService.delete_model(db, model_id, current_user.id)
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )
