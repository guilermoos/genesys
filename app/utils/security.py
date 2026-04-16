"""
Security utilities for authentication and authorization.
"""

from datetime import datetime, timedelta
from typing import Optional
import secrets
import hashlib
import hmac

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.utils.config import get_settings


# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def generate_api_key(length: int = 32) -> str:
    """
    Generate a secure random API key.
    
    Args:
        length: Length of the API key
        
    Returns:
        Secure random API key
    """
    return secrets.token_urlsafe(length)


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password
        
    Returns:
        True if password matches
    """
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create JWT access token.
    
    Args:
        data: Data to encode in token
        expires_delta: Optional expiration time
        
    Returns:
        JWT token string
    """
    settings = get_settings()
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm="HS256",
    )
    
    return encoded_jwt


def verify_token(token: str) -> Optional[dict]:
    """
    Verify and decode JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token data or None if invalid
    """
    settings = get_settings()
    
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=["HS256"],
        )
        return payload
    except JWTError:
        return None


def verify_api_key(provided_key: str, stored_key: str) -> bool:
    """
    Verify API key using constant-time comparison.
    
    Args:
        provided_key: API key provided in request
        stored_key: Stored API key
        
    Returns:
        True if keys match
    """
    return hmac.compare_digest(provided_key, stored_key)
