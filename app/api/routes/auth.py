"""
Authentication routes.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_user
from app.schemas.user import UserCreate, UserResponse, UserLogin
from app.services.user_service import UserService
from app.models.user import User

router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(
    user_data: UserCreate,
    db: Session = Depends(get_db),
):
    """
    Register a new user.
    
    Args:
        user_data: User registration data
        db: Database session
        
    Returns:
        Created user
    """
    try:
        user = UserService.create_user(db, user_data)
        return user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/login")
def login(
    login_data: UserLogin,
    db: Session = Depends(get_db),
):
    """
    Login with email and password.
    
    Args:
        login_data: Login credentials
        db: Database session
        
    Returns:
        Access token and user info
    """
    user = UserService.authenticate_user(db, login_data)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    
    token = UserService.create_access_token(user)
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": UserResponse.model_validate(user),
    }


@router.get("/me", response_model=UserResponse)
def get_current_user_info(
    current_user: User = Depends(get_current_user),
):
    """
    Get current user information.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User information
    """
    return current_user


@router.post("/regenerate-api-key")
def regenerate_api_key(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Generate a new API key.
    
    Args:
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        New API key
    """
    new_key = UserService.regenerate_api_key(db, current_user.id)
    
    return {
        "api_key": new_key,
        "message": "Store this API key safely. It will not be shown again.",
    }
