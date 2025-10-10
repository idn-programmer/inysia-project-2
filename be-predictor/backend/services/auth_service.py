from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import models as orm

settings = get_settings()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt


def verify_token(token: str) -> Optional[int]:
    """Verify JWT token and return user ID."""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        user_id: int = payload.get("sub")
        if user_id is None:
            return None
        return int(user_id)
    except JWTError:
        return None


def authenticate_user(db: Session, username: str, password: str) -> Optional[orm.User]:
    """Authenticate a user with username and password."""
    user = db.query(orm.User).filter(orm.User.username == username).first()
    if not user:
        return None
    # Simple password comparison (no hashing as requested)
    if password != user.password:
        return None
    return user


def get_user_by_id(db: Session, user_id: int) -> Optional[orm.User]:
    """Get user by ID."""
    return db.query(orm.User).filter(orm.User.id == user_id).first()
