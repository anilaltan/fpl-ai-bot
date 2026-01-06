"""
FastAPI Backend for FPL SaaS Application

This module provides REST API endpoints for Fantasy Premier League optimization,
integrating with existing optimization algorithms from the src/ directory.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer

# Import our Pydantic models
from backend.models import (
    LoginRequest, LoginResponse, PlayersResponse, ErrorResponse, PlayerData
)

# Import routers
from backend.routers.players import router as players_router
from backend.routers.auth import router as auth_router
from backend.routers.dream_team import router as dream_team_router

# Import from src/
from src.database import engine, Base, SessionLocal
from src.data_loader import get_fpl_data
from src.models import Player
from src.logger import logger

# Initialize FastAPI app
app = FastAPI(
    title="FPL SaaS API",
    description="FastAPI backend for Fantasy Premier League optimization and analysis",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc
)

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow ALL origins for debugging - will be restricted in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router, prefix="/auth", tags=["authentication"])
app.include_router(players_router)
app.include_router(dream_team_router, tags=["optimization"])

# Security scheme (simplified for now - in production use JWT)
security = HTTPBearer(auto_error=False)


# Import get_current_user from auth router
from backend.routers.auth import get_current_user


@app.on_event("startup")
async def startup_event():
    """
    Initialize database tables and load FPL data at startup.
    Cache data in app.state for fast access.
    """
    try:
        logger.info("🚀 Starting FPL SaaS API...")

        # Ensure database tables exist
        logger.info("📦 Creating database tables if they don't exist...")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables ready")

        # Load and cache FPL data in app.state
        logger.info("📊 Loading and caching FPL data...")
        try:
            data = load_fpl_data()

            # Sync data to database (now expects list, not DataFrame)
            logger.info("💾 Syncing FPL data to database...")
            players_list = data.get('players_df', [])
            if isinstance(players_list, list):
                sync_fpl_data_to_database(players_list)
            else:
                logger.warning("Players data is not a list, skipping database sync")

            app.state.fpl_data = data
            logger.info(f"✅ FPL data cached in app.state.fpl_data - {len(data['players_df'])} players loaded")
        except Exception as data_error:
            logger.error(f"❌ Failed to load and cache FPL data: {data_error}")
            app.state.fpl_data = None
            # Don't crash the app, just log the error

    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        # Don't crash the app, just log the error

# Global data cache (simplified for now)
_data_cache: Optional[Dict[str, Any]] = None


# get_current_user is now imported from auth router


def load_fpl_data(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Load FPL data using simplified get_fpl_data function.
    Returns data in the format expected by the application.
    """
    global _data_cache

    try:
        logger.info("Loading FPL data from API...")

        # Fetch fresh data from FPL API
        players_list = get_fpl_data()

        if not players_list:
            logger.warning("No players data received from API")
            return {"players_df": [], "fixtures_df": [], "last_updated": pd.Timestamp.now()}

        logger.info(f"Loaded {len(players_list)} players from FPL API")

        # Store list directly (NO DataFrame conversion for safety)
        _data_cache = {
            "players_df": players_list,  # Now contains the list directly
            "fixtures_df": [],  # Empty fixtures list for consistency
            "last_updated": pd.Timestamp.now()
        }

        return _data_cache

    except Exception as e:
        logger.error(f"Error loading FPL data: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to load FPL data"
        )


def sync_fpl_data_to_database(players_list: list) -> None:
    """
    Sync FPL player data to PostgreSQL database.
    Performs UPSERT operations - updates existing records, inserts new ones.

    Args:
        players_list: List of player dictionaries from FPL API
    """
    db = SessionLocal()
    try:
        synced_count = 0
        updated_count = 0
        inserted_count = 0

        logger.info(f"Starting database sync for {len(players_list)} players...")

        for player_data in players_list:
            try:
                # Validate player data
                player_id = player_data.get('id')
                if not player_id:
                    continue  # Skip invalid records

                # Prepare player data using standardized field names
                # player_data already has the correct field names from data_loader
                clean_player_data = {
                    'id': int(player_data['id']),
                    'name': str(player_data['name']),
                    'team': str(player_data['team']),  # Now team name string
                    'position': str(player_data['position']),  # Now position string (GKP, DEF, etc.)
                    'price': float(player_data['price']),
                    'xp': float(player_data['xp']),
                    'photo': str(player_data.get('photo', '')),
                    'chance_of_playing': int(player_data.get('chance_of_playing', 100))
                }

                # Check if player already exists
                existing_player = db.query(Player).filter(Player.id == player_id).first()

                if existing_player:
                    # Update existing player
                    for key, value in clean_player_data.items():
                        setattr(existing_player, key, value)
                    updated_count += 1
                    logger.debug(f"Updated player: {clean_player_data['name']} (ID: {player_id})")
                else:
                    # Insert new player
                    new_player = Player(**clean_player_data)
                    db.add(new_player)
                    inserted_count += 1
                    logger.debug(f"Inserted new player: {clean_player_data['name']} (ID: {player_id})")

                synced_count += 1

                # Commit each successful operation to avoid transaction locks
                db.commit()

            except Exception as player_error:
                logger.warning(f"Error processing player {player_data.get('id', 'unknown')}: {player_error}")
                # CRITICAL: Rollback on error to prevent transaction abortion
                db.rollback()
                continue  # Continue with next player

        logger.info(f"✅ Database sync completed: {synced_count} players processed")
        logger.info(f"   📊 Updated: {updated_count} players")
        logger.info(f"   ➕ Inserted: {inserted_count} players")

    except Exception as e:
        logger.error(f"❌ Database sync failed: {e}")
        db.rollback()  # Rollback any pending transaction
        raise
    finally:
        db.close()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "FPL SaaS API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": pd.Timestamp.now().isoformat()}


# Authentication endpoints are now handled by auth router


@app.get("/players", response_model=PlayersResponse)
async def get_players():
    """Get all players data."""
    try:
        # Load FPL data
        data = load_fpl_data()
        df_players = data["players_df"]

        # Convert DataFrame to PlayerData objects (simplified)
        players = []
        for _, row in df_players.head(50).iterrows():  # Limit to first 50 for demo
            try:
                player = PlayerData(
                    id=int(row.get('id', 0)),
                    web_name=str(row.get('web_name', '')),
                    team_name=str(row.get('team_name', '')),
                    position=str(row.get('position', '')),
                    price=float(row.get('price', 0.0)),
                )
                players.append(player)
            except Exception as e:
                logger.warning(f"Error processing player {row.get('web_name', 'unknown')}: {e}")
                continue

        return PlayersResponse(
            success=True,
            message=f"Retrieved {len(players)} players",
            players=players
        )

    except Exception as e:
        logger.error(f"Error retrieving players: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve player data"
        )


# @app.post("/optimize/dream-team", response_model=DreamTeamResponse)
# async def optimize_dream_team(
#     request: DreamTeamRequest,
#     req: Request,  # FastAPI Request object to access headers
#     user: Optional[Dict[str, Any]] = Depends(get_current_user)
# ):
    """Optimize dream team for authenticated user."""
    auth_header = req.headers.get('authorization', 'None')
    print("🚀 [BACKEND] Dream Team endpoint called!")
    print(f"👤 [BACKEND] User: {user.get('username', 'unknown') if user else 'None (Not Authenticated)'}")
    print(f"📦 [BACKEND] Request body: {request.dict()}")
    print(f"🔗 [BACKEND] Authorization header: {'Present' if auth_header != 'None' else 'Missing'}")
    if auth_header != 'None':
        print(f"🔗 [BACKEND] Auth header value: {auth_header[:20]}...")

    if not user:
        print("[DREAM TEAM] ERROR: No user provided (authentication failed)")
        # #region agent log
        try:
            with open('/root/fpl-test/.cursor/debug.log', 'a') as f:
                log_entry = {
                    "id": f"log_{int(time.time()*1000)}_auth_fail",
                    "timestamp": int(time.time()*1000),
                    "location": "backend/main.py:369",
                    "message": "Authentication failed - no user",
                    "data": {},
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A"
                }
                f.write(json.dumps(log_entry) + '\n')
        except Exception as log_err:
            pass
        # #endregion
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    print(f"[DREAM TEAM] Starting optimization for user {user.get('username')}")
    try:
        # Load fresh FPL data (like debug_optimizer.py - no cache)
        print("[DREAM TEAM] Loading fresh data without cache...")
        df_understat, df_fpl, df_fixtures = data_loader.fetch_all_data()

        # Merge and enrich data like debug_optimizer.py
        df_players = data_loader.merge_data(df_understat, df_fpl)

        # Calculate metrics using Optimizer
        df_players = optimizer.calculate_metrics(df_players, df_fixtures)

        print(f"[DREAM TEAM] Fresh data loaded - Players: {len(df_players)}, Fixtures: {len(df_fixtures)}")


        # #region agent log
        try:
            with open('/root/fpl-test/.cursor/debug.log', 'a') as f:
                log_entry = {
                    "id": f"log_{int(time.time()*1000)}_data_loaded",
                    "timestamp": int(time.time()*1000),
                    "location": "backend/main.py:356",
                    "message": "Data loaded successfully",
                    "data": {
                        "players_count": len(df_players),
                        "fixtures_count": len(df_fixtures),
                        "players_columns": df_players.columns.tolist()[:10],
                        "has_gw19_xP": 'gw19_xP' in df_players.columns,
                        "has_risk_adjusted_xP": 'risk_adjusted_xP' in df_players.columns
                    },
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A"
                }
                f.write(json.dumps(log_entry) + '\n')
        except Exception as log_err:
            pass
        # #endregion

        # Skip stale data check since we always load fresh data for optimization

        # Use real optimization instead of mock data
        print(f"[DREAM TEAM] Running real optimization for user {user.get('username')}")
        print(f"[DREAM TEAM] Available players: {len(df_players)}")

        # Check if we have players data
        if df_players.empty:
            print("[DREAM TEAM] ERROR: No players available")
            # #region agent log
            try:
                with open('/root/fpl-test/.cursor/debug.log', 'a') as f:
                    log_entry = {
                        "id": f"log_{int(time.time()*1000)}_no_players",
                        "timestamp": int(time.time()*1000),
                        "location": "backend/main.py:376",
                        "message": "No players available in dataset",
                        "data": {"players_count": len(df_players)},
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "A"
                    }
                    f.write(json.dumps(log_entry) + '\n')
            except Exception as log_err:
                pass
            # #endregion
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No player data available"
            )

        # Run real optimization - FAIL FAST MODE
        dream_team_df = optimizer.solve_dream_team(df_players, budget=100.0)
        print(f"[DREAM TEAM] Optimization completed with {len(dream_team_df)} players")

        if dream_team_df.empty:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Dream team optimization returned empty result - no valid solution found"
            )

        # Convert to response format
        squad = []
        captain_name = None
        total_cost = 0.0
        total_points = 0.0

        for _, player in dream_team_df.iterrows():
            try:
                player_data = DreamTeamPlayer(
                    id=int(player.get('id', 0)),
                    web_name=str(player.get('web_name', '')),
                    team_name=str(player.get('team_name', '')),
                    position=str(player.get('position', '')),
                    price=float(player.get('price', 0.0)),
                    gw19_xP=float(player.get('gw19_xP', 0.0)),
                    risk_adjusted_xP=float(player.get('risk_adjusted_xP', 0.0)),
                    is_captain=bool(player.get('is_captain', False))
                )
                squad.append(player_data)

                if player.get('is_captain'):
                    captain_name = player.get('web_name', '')

                total_cost += player.get('price', 0.0)
                total_points += player.get('risk_adjusted_xP', 0.0)
            except Exception as player_error:
                print(f"[DREAM TEAM] ERROR processing player {player.get('web_name', 'unknown')}: {player_error}")
                # #region agent log
                try:
                    with open('/root/fpl-test/.cursor/debug.log', 'a') as f:
                        log_entry = {
                            "id": f"log_{int(time.time()*1000)}_player_error",
                            "timestamp": int(time.time()*1000),
                            "location": "backend/main.py:408",
                            "message": "Error processing player",
                            "data": {
                                "player_name": player.get('web_name', 'unknown'),
                                "error": str(player_error),
                                "has_gw19_xP": 'gw19_xP' in player.index,
                                "has_risk_adjusted_xP": 'risk_adjusted_xP' in player.index,
                                "gw19_xP_value": player.get('gw19_xP'),
                                "risk_adjusted_xP_value": player.get('risk_adjusted_xP')
                            },
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "A"
                        }
                        f.write(json.dumps(log_entry) + '\n')
                except Exception as log_err:
                    pass
                # #endregion
                raise player_error

        # #region agent log
        try:
            with open('/root/fpl-test/.cursor/debug.log', 'a') as f:
                log_entry = {
                    "id": f"log_{int(time.time()*1000)}_response_created",
                    "timestamp": int(time.time()*1000),
                    "location": "backend/main.py:427",
                    "message": "Dream team response created successfully",
                    "data": {
                        "squad_size": len(squad),
                        "total_cost": total_cost,
                        "total_points": total_points,
                        "captain_name": captain_name
                    },
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A"
                }
                f.write(json.dumps(log_entry) + '\n')
        except Exception as log_err:
            pass
        # #endregion

        return DreamTeamResponse(
            success=True,
            message=f"Dream team optimized for {user['username']}",
            squad=squad,
            total_cost=round(total_cost, 2),
            total_points=round(total_points, 2),
            captain_name=captain_name
        )

    except Exception as e:
        # Hata detaylarını tam traceback ile logla
        error_traceback = traceback.format_exc()
        logger.error(f"Error optimizing dream team: {e}")
        logger.error(f"Full traceback:\n{error_traceback}")

        print(f"🚨 [DREAM TEAM ERROR] Exception occurred:")
        print(f"   Error: {e}")
        print(f"   Type: {type(e).__name__}")
        print(f"   Traceback:\n{error_traceback}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to optimize dream team"
        )


# @app.get("/user/team", response_model=UserTeamResponse)
# async def get_user_team(user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    """Get user's FPL team."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    try:
        fpl_id = user.get('fpl_id')
        if not fpl_id:
            return UserTeamResponse(
                success=False,
                message="FPL ID not set for this user. Please update your profile.",
                team_id=None,
                players=[]
            )

        # Load player data for ID to name mapping
        data = load_fpl_data()
        df_players = data["players_df"]

        # Fetch user's team using existing DataLoader method
        team_players = data_loader.fetch_user_team(fpl_id, df_players)

        if not team_players:
            return UserTeamResponse(
                success=False,
                message="Could not retrieve team data. Please check your FPL ID.",
                team_id=fpl_id,
                players=[]
            )

        return UserTeamResponse(
            success=True,
            message=f"Retrieved team for user {user['username']}",
            team_id=fpl_id,
            players=team_players
        )

    except Exception as e:
        logger.error(f"Error retrieving user team: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user team data"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="46.224.178.180", port=8000, reload=True)
