"""
User service for authentication and user management.
"""

from typing import Optional
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import or_

from app.models.user import User
from app.schemas.user import UserCreate, UserLogin
from app.utils.security import (
    hash_password,
    verify_password,
    generate_api_key,
    create_access_token,
)
from app.utils.id_generator import generate_uuid


class UserService:
    """Service for user management and authentication."""
    
    @staticmethod
    def create_user(db: Session, user_data: UserCreate) -> User:
        """
        Create a new user.
        
        Args:
            db: Database session
            user_data: User creation data
            
        Returns:
            Created user
            
        Raises:
            ValueError: If email already exists
        """
        # Check if email exists
        existing = db.query(User).filter(User.email == user_data.email).first()
        if existing:
            raise ValueError("Email already registered")
        
        # Create user
        user = User(
            id=generate_uuid(),
            name=user_data.name,
            email=user_data.email,
            hashed_password=hash_password(user_data.password),
            api_key=generate_api_key(),
        )
        
        db.add(user)
        db.commit()
        db.refresh(user)
        
        return user
    
    @staticmethod
    def authenticate_user(
        db: Session,
        login_data: UserLogin,
    ) -> Optional[User]:
        """
        Authenticate user with email and password.
        
        Args:
            db: Database session
            login_data: Login credentials
            
        Returns:
            User if authenticated, None otherwise
        """
        user = db.query(User).filter(User.email == login_data.email).first()
        
        if not user:
            return None
        
        if not verify_password(login_data.password, user.hashed_password):
            return None
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()
        
        return user
    
    @staticmethod
    def get_user_by_api_key(db: Session, api_key: str) -> Optional[User]:
        """
        Get user by API key.
        
        Args:
            db: Database session
            api_key: API key
            
        Returns:
            User if found
        """
        return db.query(User).filter(
            User.api_key == api_key,
            User.is_active == True,
        ).first()
    
    @staticmethod
    def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
        """
        Get user by ID.
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            User if found
        """
        return db.query(User).filter(User.id == user_id).first()
    
    @staticmethod
    def regenerate_api_key(db: Session, user_id: str) -> Optional[str]:
        """
        Generate new API key for user.
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            New API key if user found
        """
        user = UserService.get_user_by_id(db, user_id)
        
        if not user:
            return None
        
        user.api_key = generate_api_key()
        db.commit()
        
        return user.api_key
    
    @staticmethod
    def create_access_token(user: User) -> str:
        """
        Create JWT access token for user.
        
        Args:
            user: User to create token for
            
        Returns:
            JWT token
        """
        return create_access_token(
            data={"sub": user.id, "email": user.email}
        )
