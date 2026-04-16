"""
Pydantic schemas for User entity.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, ConfigDict


class UserBase(BaseModel):
    """Base user schema with common fields."""
    
    name: str = Field(..., min_length=1, max_length=255)
    email: EmailStr


class UserCreate(UserBase):
    """Schema for creating a new user."""
    
    password: str = Field(..., min_length=8, max_length=100)


class UserLogin(BaseModel):
    """Schema for user login."""
    
    email: EmailStr
    password: str


class UserResponse(UserBase):
    """Schema for user response."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: str
    api_key: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None


class UserInDB(UserBase):
    """Schema for user stored in database (internal use)."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: str
    hashed_password: str
    api_key: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
