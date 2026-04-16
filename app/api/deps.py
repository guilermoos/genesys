"""
API dependencies for authentication and database access.
"""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.user import User
from app.services.user_service import UserService

# Security scheme
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """
    Get current user from API key or JWT token.
    
    Args:
        credentials: Authorization credentials
        db: Database session
        
    Returns:
        Current user
        
    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    
    # Try API key first
    user = UserService.get_user_by_api_key(db, token)
    
    if user:
        return user
    
    # Try JWT token
    from app.utils.security import verify_token
    
    payload = verify_token(token)
    
    if payload and "sub" in payload:
        user = UserService.get_user_by_id(db, payload["sub"])
        if user:
            return user
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """
    Get current user if authenticated, None otherwise.
    
    Args:
        credentials: Authorization credentials
        db: Database session
        
    Returns:
        Current user or None
    """
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials, db)
    except HTTPException:
        return None
