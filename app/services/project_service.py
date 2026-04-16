"""
Project service for managing projects.
"""

from typing import List, Optional, Tuple
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.models.project import Project
from app.models.user import User
from app.schemas.project import ProjectCreate, ProjectUpdate
from app.utils.id_generator import generate_uuid
from app.templates.base import TemplateRegistry


class ProjectService:
    """Service for project management operations."""
    
    @staticmethod
    def create_project(
        db: Session,
        user_id: str,
        project_data: ProjectCreate,
    ) -> Project:
        """
        Create a new project.
        
        Args:
            db: Database session
            user_id: Owner user ID
            project_data: Project creation data
            
        Returns:
            Created project
            
        Raises:
            ValueError: If template is invalid
        """
        # Validate template
        if not TemplateRegistry.list_templates():
            raise ValueError("No templates available")
        
        # Create project
        project = Project(
            id=generate_uuid(),
            user_id=user_id,
            name=project_data.name,
            description=project_data.description,
            template_default=project_data.template_default,
            status="active",
        )
        
        db.add(project)
        db.commit()
        db.refresh(project)
        
        return project
    
    @staticmethod
    def get_project(
        db: Session,
        project_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[Project]:
        """
        Get project by ID.
        
        Args:
            db: Database session
            project_id: Project ID
            user_id: Optional user ID for access control
            
        Returns:
            Project if found and accessible
        """
        query = db.query(Project).filter(Project.id == project_id)
        
        if user_id:
            query = query.filter(Project.user_id == user_id)
        
        return query.first()
    
    @staticmethod
    def list_projects(
        db: Session,
        user_id: str,
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None,
    ) -> Tuple[List[Project], int]:
        """
        List projects for a user.
        
        Args:
            db: Database session
            user_id: User ID
            skip: Number of projects to skip
            limit: Maximum number of projects
            status: Optional status filter
            
        Returns:
            Tuple of (projects list, total count)
        """
        query = db.query(Project).filter(Project.user_id == user_id)
        
        if status:
            query = query.filter(Project.status == status)
        
        total = query.count()
        projects = (
            query.order_by(desc(Project.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )
        
        return projects, total
    
    @staticmethod
    def update_project(
        db: Session,
        project_id: str,
        user_id: str,
        project_data: ProjectUpdate,
    ) -> Optional[Project]:
        """
        Update a project.
        
        Args:
            db: Database session
            project_id: Project ID
            user_id: User ID for access control
            project_data: Update data
            
        Returns:
            Updated project if found
        """
        project = ProjectService.get_project(db, project_id, user_id)
        
        if not project:
            return None
        
        update_data = project_data.model_dump(exclude_unset=True)
        
        for field, value in update_data.items():
            setattr(project, field, value)
        
        project.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(project)
        
        return project
    
    @staticmethod
    def archive_project(
        db: Session,
        project_id: str,
        user_id: str,
    ) -> Optional[Project]:
        """
        Archive a project.
        
        Args:
            db: Database session
            project_id: Project ID
            user_id: User ID for access control
            
        Returns:
            Archived project if found
        """
        return ProjectService.update_project(
            db,
            project_id,
            user_id,
            ProjectUpdate(status="archived"),
        )
    
    @staticmethod
    def delete_project(
        db: Session,
        project_id: str,
        user_id: str,
    ) -> bool:
        """
        Delete a project.
        
        Args:
            db: Database session
            project_id: Project ID
            user_id: User ID for access control
            
        Returns:
            True if deleted
        """
        project = ProjectService.get_project(db, project_id, user_id)
        
        if not project:
            return False
        
        db.delete(project)
        db.commit()
        
        return True
    
    @staticmethod
    def get_project_stats(
        db: Session,
        project_id: str,
        user_id: str,
    ) -> Optional[dict]:
        """
        Get project statistics.
        
        Args:
            db: Database session
            project_id: Project ID
            user_id: User ID for access control
            
        Returns:
            Project statistics
        """
        project = ProjectService.get_project(db, project_id, user_id)
        
        if not project:
            return None
        
        return {
            "project_id": project.id,
            "name": project.name,
            "template": project.template_default,
            "status": project.status,
            "job_count": len(project.training_jobs),
            "model_count": len(project.model_versions),
            "active_model": next(
                (m.id for m in project.model_versions if m.is_active),
                None
            ),
            "created_at": project.created_at,
            "updated_at": project.updated_at,
        }
