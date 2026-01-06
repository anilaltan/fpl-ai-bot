"""
SQLAlchemy Models for FPL SaaS Application

This module defines the database models using SQLAlchemy ORM.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, func
from src.database import Base


class User(Base):
    """
    User model for authentication and user management.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="user", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Player(Base):
    """
    Player model for FPL player data.
    Standardized field names across the entire application.
    """
    __tablename__ = "players"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)  # Player name (from web_name)
    team = Column(String, nullable=False)  # Team name (from teams API)
    position = Column(String, nullable=False)  # Position: GKP, DEF, MID, FWD
    price = Column(Float, nullable=False)  # Price in millions
    xp = Column(Float, nullable=False, default=0.0)  # Expected points (from ep_next)
    photo = Column(String, nullable=True)  # Player photo code (from photo field)
    chance_of_playing = Column(Integer, nullable=False, default=100)  # Chance of playing (0-100)


class Team(Base):
    """
    Team model for FPL teams.
    """
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(Integer, unique=True, nullable=False)
    name = Column(String, nullable=False)
    short_name = Column(String, nullable=False)
