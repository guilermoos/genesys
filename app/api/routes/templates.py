"""
Template routes.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status

from app.templates.base import TemplateRegistry
from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter()


@router.get("")
def list_templates(
    current_user: User = Depends(get_current_user),
):
    """
    List all available templates.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        List of available templates
    """
    templates = TemplateRegistry.get_all_templates_info()
    
    return {
        "templates": templates,
        "count": len(templates),
    }


@router.get("/{template_name}")
def get_template(
    template_name: str,
    current_user: User = Depends(get_current_user),
):
    """
    Get detailed information about a template.
    
    Args:
        template_name: Template name
        current_user: Current authenticated user
        
    Returns:
        Template details
    """
    try:
        template_info = TemplateRegistry.get_template_info(template_name)
        return template_info
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
