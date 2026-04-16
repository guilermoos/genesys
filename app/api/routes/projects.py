"""
Project routes.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_user
from app.schemas.project import (
    ProjectCreate,
    ProjectResponse,
    ProjectUpdate,
    ProjectListResponse,
)
from app.services.project_service import ProjectService
from app.models.user import User

router = APIRouter()


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
def create_project(
    project_data: ProjectCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Create a new project.
    
    Args:
        project_data: Project creation data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Created project
    """
    try:
        project = ProjectService.create_project(
            db,
            current_user.id,
            project_data,
        )
        return project
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("", response_model=ProjectListResponse)
def list_projects(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: str = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    List projects for current user.
    
    Args:
        skip: Number of projects to skip
        limit: Maximum number of projects
        status: Filter by status
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        List of projects
    """
    projects, total = ProjectService.list_projects(
        db,
        current_user.id,
        skip=skip,
        limit=limit,
        status=status,
    )
    
    return ProjectListResponse(
        items=[ProjectResponse.model_validate(p) for p in projects],
        total=total,
        page=skip // limit + 1 if limit > 0 else 1,
        page_size=limit,
    )


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get project by ID.
    
    Args:
        project_id: Project ID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Project if found
    """
    project = ProjectService.get_project(
        db,
        project_id,
        current_user.id,
    )
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )
    
    return project


@router.patch("/{project_id}", response_model=ProjectResponse)
def update_project(
    project_id: str,
    project_data: ProjectUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Update a project.
    
    Args:
        project_id: Project ID
        project_data: Update data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Updated project
    """
    project = ProjectService.update_project(
        db,
        project_id,
        current_user.id,
        project_data,
    )
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )
    
    return project


@router.post("/{project_id}/archive")
def archive_project(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Archive a project.
    
    Args:
        project_id: Project ID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Archived project
    """
    project = ProjectService.archive_project(
        db,
        project_id,
        current_user.id,
    )
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )
    
    return {"message": "Project archived successfully", "project": project}


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Delete a project.
    
    Args:
        project_id: Project ID
        current_user: Current authenticated user
        db: Database session
    """
    deleted = ProjectService.delete_project(
        db,
        project_id,
        current_user.id,
    )
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )


@router.get("/{project_id}/stats")
def get_project_stats(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get project statistics.
    
    Args:
        project_id: Project ID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Project statistics
    """
    stats = ProjectService.get_project_stats(
        db,
        project_id,
        current_user.id,
    )
    
    if not stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )
    
    return stats
