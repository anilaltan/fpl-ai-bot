"""
Players API Router for FPL SaaS Application

This module provides simplified player data endpoints for the frontend Players page.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Request, status
from pydantic import BaseModel
import pandas as pd

# Initialize router
router = APIRouter()

logger = logging.getLogger(__name__)


# Simplified Player Model for frontend
class SimplePlayerData(BaseModel):
    id: int
    name: str
    team_name: str
    position: str
    price: float
    predicted_xP: float
    photo: Optional[str] = None
    chance_of_playing: Optional[int] = None
    form: Optional[float] = None
    status: Optional[str] = None


class PlayersListResponse(BaseModel):
    success: bool
    message: str
    players: List[SimplePlayerData]


@router.get("/players", response_model=PlayersListResponse)
async def get_players(request: Request):
    """
    Get all players data in simplified format for the frontend Players page.

    Returns:
        List of players with basic info: id, name, team_name, position, price, predicted_xP, form, status
    """
    try:
        logger.info("Fetching players data from cache...")

        # Check if data is cached in app.state
        if not hasattr(request.app.state, 'fpl_data') or request.app.state.fpl_data is None:
            logger.warning("FPL data not cached in app.state")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Player data is not available. Please try again later."
            )

        # Get cached data
        data = request.app.state.fpl_data
        players_data = data.get('players_df', [])

        if not players_data:
            logger.warning("No players data in cache")
            return PlayersListResponse(
                success=False,
                message="No player data available",
                players=[]
            )

        # Convert to simplified player objects
        players = []
        for player in players_data:
            try:
                player_obj = SimplePlayerData(
                    id=int(player.get('id', 0)),
                    name=str(player.get('name', '')),
                    team_name=str(player.get('team', 'Unknown')),  # Already team name string
                    position=str(player.get('position', '')),
                    price=float(player.get('price', 0.0)),
                    predicted_xP=float(player.get('xp', 0.0)),
                    photo=player.get('photo'),
                    chance_of_playing=player.get('chance_of_playing')
                )
                players.append(player_obj)

            except Exception as e:
                logger.warning(f"Error processing player {player.get('id', 'unknown')}: {e}")
                continue

        logger.info(f"Successfully retrieved {len(players)} players")

        return PlayersListResponse(
            success=True,
            message=f"Retrieved {len(players)} players",
            players=players
        )

    except Exception as e:
        logger.error(f"Error in get_players endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve players data"
        )
