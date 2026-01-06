"""
Dream Team API Router for FPL SaaS Application

This module provides dream team optimization endpoints using linear programming.
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel

# Import from src/
from src.solver import solve_dream_team_with_retry
from src.logger import logger


router = APIRouter()

logger = logging.getLogger(__name__)


# Pydantic models for API responses
class DreamTeamPlayer(BaseModel):
    id: int
    name: str
    team_name: str
    position: str
    price: float
    predicted_xP: float
    is_starting: bool = False
    is_captain: bool = False
    is_vice_captain: bool = False
    form: Optional[float] = None
    status: Optional[str] = None


class DreamTeamResponse(BaseModel):
    success: bool
    message: str
    squad: List[DreamTeamPlayer]
    total_cost: float
    total_xP: float
    budget_used_percent: float
    position_breakdown: Dict[str, int]
    team_breakdown: Dict[str, int]


@router.get("/dream-team", response_model=DreamTeamResponse)
async def get_dream_team(
    request: Request,
    budget: float = Query(100.0, description="Maximum budget in millions (£)", ge=50.0, le=200.0)
):
    """
    Calculate and return the optimal FPL dream team using linear programming.

    This endpoint solves the fantasy football optimization problem to select
    the best 15 players that maximize predicted points while respecting all
    FPL constraints (budget, positions, team limits).

    Args:
        budget: Maximum total cost allowed (default: £100M)

    Returns:
        Dream team with 15 players, total cost, total xP, and breakdowns
    """
    try:
        logger.info(f"🎯 Dream team request received - Budget: £{budget}M")

        # Check if FPL data is available in app state
        if not hasattr(request.app.state, 'fpl_data') or request.app.state.fpl_data is None:
            logger.warning("FPL data not available in app.state")
            raise HTTPException(
                status_code=503,
                detail="Player data is not available. Please try again later."
            )

        # Get cached player data
        data = request.app.state.fpl_data
        players_data = data.get('players_df', [])

        if not players_data:
            logger.warning("No players data in cache")
            raise HTTPException(
                status_code=503,
                detail="No player data available for optimization."
            )

        # Players data is already a list of dicts, just validate and prepare for solver
        players_list = []
        for player in players_data:
            try:
                # Use standardized field names directly
                player_id = player.get('id')
                name = player.get('name', '').strip()
                team = player.get('team')  # String team name
                position = player.get('position')  # String position (GKP, DEF, MID, FWD)
                price = player.get('price', 0.0)
                xp = player.get('xp', 0.0)

                # Skip invalid players
                if not player_id or not name or not team or not position or price <= 0:
                    continue

                # Prepare player dict for solver
                player_dict = {
                    'id': int(player_id),
                    'name': name,
                    'team_name': team,  # Already team name string
                    'position': position,  # Already position string (GKP, DEF, etc.)
                    'price': float(price),
                    'xp': float(xp)  # Standardized field name
                }

                players_list.append(player_dict)

            except Exception as player_error:
                logger.warning(f"Error processing player {player.get('id', 'unknown')} for optimization: {player_error}")
                continue

        if len(players_list) < 15:
            logger.warning(f"Not enough valid players for optimization: {len(players_list)} < 15")
            raise HTTPException(
                status_code=422,
                detail=f"Insufficient player data for optimization. Only {len(players_list)} valid players available."
            )

        logger.info(f"🚀 Starting optimization with {len(players_list)} players, budget £{budget}M")

        # Solve the optimization problem with retry mechanism
        result = solve_dream_team_with_retry(players_list, budget)

        if result["status"] == "failed" or not result["players"]:
            logger.warning("All optimization attempts failed - no valid solution found")
            raise HTTPException(
                status_code=422,
                detail="Unable to find a valid dream team solution with current constraints."
            )

        selected_players = result["players"]
        solution_status = result["status"]

        # Calculate summary statistics
        total_cost = sum(player['price'] for player in selected_players)
        total_xp = sum(player['xp'] for player in selected_players)
        budget_used_percent = (total_cost / budget) * 100

        # Position breakdown
        position_breakdown = {}
        for player in selected_players:
            pos = player.get('position', 'Unknown')
            position_breakdown[pos] = position_breakdown.get(pos, 0) + 1

        # Team breakdown
        team_breakdown = {}
        for player in selected_players:
            team = player.get('team_name', 'Unknown')
            team_breakdown[team] = team_breakdown.get(team, 0) + 1

        # Convert to Pydantic models
        squad = [
            DreamTeamPlayer(
                id=player['id'],
                name=player['name'],
                team_name=player['team_name'],
                position=player['position'],
                price=player['price'],
                predicted_xP=player['xp'],
                is_starting=player.get('is_starting', False),
                is_captain=player.get('is_captain', False),
                is_vice_captain=player.get('is_vice_captain', False),
                form=player.get('form'),
                status=player.get('status')
            )
            for player in selected_players
        ]

        logger.info("✅ Dream team calculated successfully!")
        logger.info(f"   Total cost: £{total_cost:.1f}M ({budget_used_percent:.1f}% of budget)")
        logger.info(f"   Total predicted xP: {total_xp:.1f}")
        logger.info(f"   Position breakdown: {position_breakdown}")

        status_message = "Dream team optimized for £{:.0f}M budget".format(budget)
        if solution_status == "fallback":
            status_message += " (Fallback: Top players by XP)"
        elif solution_status == "optimal":
            status_message += " (Optimal solution)"

        return DreamTeamResponse(
            success=True,
            message=status_message,
            squad=squad,
            total_cost=round(total_cost, 2),
            total_xP=round(total_xp, 2),
            budget_used_percent=round(budget_used_percent, 1),
            position_breakdown=position_breakdown,
            team_breakdown=team_breakdown
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Dream team calculation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during dream team optimization."
        )
