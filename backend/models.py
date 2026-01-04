"""
FastAPI Pydantic Models for FPL SaaS Backend

This module defines request and response schemas for the FastAPI endpoints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel


# Authentication Models
class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    success: bool
    message: str
    token: Optional[str] = None
    user: Optional[Dict[str, Any]] = None


# Player Data Models
class PlayerData(BaseModel):
    id: int
    web_name: str
    team_name: str
    position: str
    price: float
    form: Optional[float] = None
    total_points: Optional[int] = None
    minutes: Optional[int] = None
    goals_scored: Optional[int] = None
    assists: Optional[int] = None
    clean_sheets: Optional[int] = None
    saves: Optional[int] = None
    expected_goals_per_90: Optional[float] = None
    expected_assists_per_90: Optional[float] = None
    threat: Optional[float] = None
    creativity: Optional[float] = None
    influence: Optional[float] = None
    ict_index: Optional[float] = None
    # Additional calculated fields
    gw19_xP: Optional[float] = None
    long_term_xP: Optional[float] = None
    risk_adjusted_xP: Optional[float] = None
    availability_score: Optional[float] = None


class PlayersResponse(BaseModel):
    success: bool
    message: str
    players: List[PlayerData]


# Dream Team Optimization Models
class DreamTeamRequest(BaseModel):
    budget: Optional[float] = None  # Optional custom budget, defaults to config value


class DreamTeamPlayer(BaseModel):
    id: int
    web_name: str
    team_name: str
    position: str
    price: float
    gw19_xP: float
    risk_adjusted_xP: float
    is_captain: bool = False


class DreamTeamResponse(BaseModel):
    success: bool
    message: str
    squad: List[DreamTeamPlayer]
    total_cost: float
    total_points: float
    captain_name: Optional[str] = None


# User Team Models
class UserTeamResponse(BaseModel):
    success: bool
    message: str
    team_id: Optional[str] = None
    players: List[str]  # List of web_names


# Error Response Model
class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    error_code: Optional[str] = None


# Token-based Authentication (for future use)
class TokenData(BaseModel):
    username: str
    user_id: int
    fpl_id: Optional[str] = None
