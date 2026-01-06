"""
Authentication router for FPL SaaS Backend

This module provides authentication endpoints using PostgreSQL database.
"""

from datetime import timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from backend.models import LoginRequest, LoginResponse
from src.database import get_db
from src.models import User
from src.auth import verify_password, create_access_token, decode_access_token
from src.config import settings
from src.logger import logger


router = APIRouter()

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


@router.post("/login", response_model=LoginResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    Authenticate user and return JWT token.

    Args:
        form_data: OAuth2 form data with username and password
        db: Database session

    Returns:
        LoginResponse with token and user data

    Raises:
        HTTPException: If authentication fails
    """
    logger.info(f"Login attempt for user: {form_data.username}")

    try:
        # Query user from database
        user = db.query(User).filter(User.username == form_data.username).first()

        if not user:
            logger.warning(f"User {form_data.username} not found")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Verify password
        if not verify_password(form_data.password, user.hashed_password):
            logger.warning(f"Invalid password for user: {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Create access token
        access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
        access_token = create_access_token(
            data={"sub": user.username, "user_id": user.id, "role": user.role},
            expires_delta=access_token_expires
        )

        logger.info(f"Login successful for user: {user.username} (ID: {user.id})")

        return LoginResponse(
            success=True,
            message=f"Welcome back, {user.username}!",
            token=access_token,
            user={
                "id": user.id,
                "username": user.username,
                "role": user.role
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error for user {form_data.username}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service temporarily unavailable"
        )


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """
    Dependency to get current authenticated user from JWT token.

    Args:
        token: JWT access token
        db: Database session

    Returns:
        User object

    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_access_token(token)

        if payload is None:
            raise credentials_exception

        username: str = payload.get("sub")
        user_id: int = payload.get("user_id")

        if username is None:
            raise credentials_exception

        # Query user from database
        user = db.query(User).filter(User.id == user_id).first()

        if user is None:
            raise credentials_exception

        return user

    except Exception as e:
        logger.error(f"Token validation error: {e}")
        raise credentials_exception


def get_current_active_user(current_user: User = Depends(get_current_user)):
    """
    Dependency to get current active user.

    Args:
        current_user: User object from get_current_user

    Returns:
        User object

    Raises:
        HTTPException: If user is inactive (placeholder for future use)
    """
    # For now, all users are considered active
    # In the future, you might check user.is_active or similar
    return current_user
