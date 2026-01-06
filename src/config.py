"""
Configuration Module for FPL SaaS Application

This module manages application configuration using pydantic-settings.
Handles environment variables and provides default values.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Supports loading from .env files and environment variables.
    """

    # Database Configuration
    database_url: str = Field(
        default="postgresql://user:password@localhost/fpl_db",
        description="PostgreSQL database connection URL"
    )

    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )

    # Application Configuration
    app_name: str = Field(
        default="FPL SaaS",
        description="Application name"
    )

    app_version: str = Field(
        default="1.0.0",
        description="Application version"
    )

    # Optional: Secret key for JWT or other security features
    secret_key: Optional[str] = Field(
        default=None,
        description="Secret key for cryptographic operations"
    )

    # Authentication Configuration
    algorithm: str = Field(
        default="HS256",
        description="JWT algorithm for token signing"
    )

    access_token_expire_minutes: int = Field(
        default=30,
        description="JWT access token expiration time in minutes"
    )

    class Config:
        """
        Pydantic configuration for settings.

        Enables loading from environment variables and .env files.
        """
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
