"""
API routes.
"""

from fastapi import APIRouter

from app.api.routes import auth, projects, training, models, inference, templates

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(projects.router, prefix="/projects", tags=["Projects"])
api_router.include_router(training.router, prefix="/jobs", tags=["Training Jobs"])
api_router.include_router(models.router, prefix="/models", tags=["Models"])
api_router.include_router(inference.router, prefix="/inference", tags=["Inference"])
api_router.include_router(templates.router, prefix="/templates", tags=["Templates"])
