"""
ID generation utilities.
"""

import uuid
from typing import Optional


def generate_uuid() -> str:
    """
    Generate a UUID4 string.
    
    Returns:
        UUID string without dashes
    """
    return str(uuid.uuid4())


def generate_id(prefix: Optional[str] = None) -> str:
    """
    Generate a unique ID with optional prefix.
    
    Args:
        prefix: Optional prefix for the ID
        
    Returns:
        Unique ID string
    """
    uid = generate_uuid()
    if prefix:
        return f"{prefix}_{uid}"
    return uid


def generate_short_id(length: int = 12) -> str:
    """
    Generate a short random ID.
    
    Args:
        length: Length of the ID
        
    Returns:
        Short random ID
    """
    import secrets
    import string
    
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))
