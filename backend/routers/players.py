"""
Players API Router for FPL SaaS Application

This module provides simplified player data endpoints for the frontend Players page.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import pandas as pd

# Import existing components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data_loader import DataLoader

# Initialize router and components
router = APIRouter()
data_loader = DataLoader()

logger = logging.getLogger(__name__)


# Simplified Player Model for frontend
class SimplePlayerData(BaseModel):
    id: int
    name: str
    team_name: str
    position: str
    price: float
    predicted_xP: float
    form: Optional[float] = None
    status: Optional[str] = None


class PlayersListResponse(BaseModel):
    success: bool
    message: str
    players: List[SimplePlayerData]


def load_players_data() -> pd.DataFrame:
    """
    Load and cache player data from DataLoader.
    Returns simplified DataFrame with required fields.
    """
    try:
        # Use the global cache from data_loader if available
        if hasattr(data_loader, '_data_cache') and data_loader._data_cache:
            df = data_loader._data_cache.get('players_df')
            if df is not None and not df.empty:
                return df

        # Otherwise load fresh data
        df_understat, df_fpl, df_fixtures = data_loader.fetch_all_data()
        df_merged = data_loader.merge_data(df_understat, df_fpl)
        df_metrics = data_loader.enrich_data(df_merged)

        # Store in cache for future use
        data_loader._data_cache = {
            'players_df': df_metrics,
            'last_updated': pd.Timestamp.now()
        }

        return df_metrics

    except Exception as e:
        logger.error(f"Error loading players data: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to load player data"
        )


@router.get("/players", response_model=PlayersListResponse)
async def get_players():
    """
    Get all players data in simplified format for the frontend Players page.

    Returns:
        List of players with basic info: id, name, team_name, position, price, predicted_xP, form, status
    """
    try:
        logger.info("Fetching players data for frontend...")

        # Load player data
        df_players = load_players_data()

        # Convert to simplified player objects
        players = []
        for _, row in df_players.iterrows():
            try:
                # Map position abbreviations to full names
                position_map = {
                    'GK': 'Goalkeeper',
                    'DEF': 'Defender',
                    'MID': 'Midfielder',
                    'FWD': 'Forward'
                }

                player = SimplePlayerData(
                    id=int(row.get('id', 0)),
                    name=str(row.get('web_name', '')),
                    team_name=str(row.get('team_name', '')),
                    position=position_map.get(str(row.get('position', '')), str(row.get('position', ''))),
                    price=float(row.get('price', 0.0)),
                    predicted_xP=float(row.get('predicted_xP', row.get('ep_next', 0.0))),
                    form=float(row.get('form', 0.0)) if pd.notna(row.get('form')) else None,
                    status=str(row.get('status', 'a')) if pd.notna(row.get('status')) else 'a'  # 'a' = available
                )
                players.append(player)

            except Exception as e:
                logger.warning(f"Error processing player {row.get('web_name', 'unknown')}: {e}")
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
