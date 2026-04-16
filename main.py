"""
Genesys - Plataforma SaaS de Deep Q-Learning API-First

Main application entry point.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import api_router
from app.db.session import init_db
from app.utils.config import get_settings
from app.templates import *  # Register templates


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    settings = get_settings()
    
    # Initialize database
    init_db()
    
    print(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} started")
    print(f"📊 Database: {settings.DATABASE_URL}")
    print(f"🤖 Templates: {__import__('app.templates.base').TemplateRegistry.list_templates()}")
    
    yield
    
    # Shutdown
    print(f"👋 {settings.APP_NAME} shutting down")


def create_application() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    settings = get_settings()
    
    app = FastAPI(
        title=settings.APP_NAME,
        description="Plataforma SaaS de Deep Q-Learning com templates e inferência via API",
        version=settings.APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)
    
    @app.get("/")
    def root():
        """Root endpoint."""
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs",
        }
    
    @app.get("/health")
    def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "version": settings.APP_VERSION,
        }
    
    return app


# Create application instance
app = create_application()

if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )
