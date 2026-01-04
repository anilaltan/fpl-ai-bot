"""
FastAPI Backend for FPL SaaS Application

This module provides REST API endpoints for Fantasy Premier League optimization,
integrating with existing optimization algorithms from the src/ directory.
"""

import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Import our Pydantic models
from .models import (
    LoginRequest, LoginResponse, PlayersResponse, DreamTeamRequest,
    DreamTeamResponse, UserTeamResponse, ErrorResponse, PlayerData, DreamTeamPlayer
)

# Import routers
from .routers.players import router as players_router

# Import existing logic from src/
sys.path.append(str(Path(__file__).parent.parent))
from src.database import DatabaseManager
from src.data_loader import DataLoader
from src.model import FPLModel
from src.optimizer import Optimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
app.include_router(players_router)

# Initialize components
db_manager = DatabaseManager()
data_loader = DataLoader()
optimizer = Optimizer()

# Security scheme (simplified for now - in production use JWT)
security = HTTPBearer(auto_error=False)


@app.on_event("startup")
async def startup_event():
    """
    Load FPL data at startup to ensure fresh data is available.
    """
    try:
        logger.info("🚀 Starting FPL SaaS API - Loading initial data...")
        global _data_cache
        _data_cache = load_fpl_data()
        logger.info("✅ Initial data load completed")
    except Exception as e:
        logger.error(f"❌ Failed to load initial data: {e}")
        # Don't crash the app, just log the error

# Global data cache (in production, use Redis or similar)
_data_cache: Optional[Dict[str, Any]] = None

# Data freshness threshold (24 hours)
DATA_FRESHNESS_HOURS = 24

# Force cache reset on startup for fresh data
print("🔄 [CACHE] Resetting global data cache for fresh FPL data...")
_data_cache = None


def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[Dict[str, Any]]:
    """
    Dependency to get current authenticated user.
    For now, using simple token-based auth (simplified approach).
    """
    if not credentials:
        return None

    # In this simplified version, we're not implementing full JWT yet
    # Just checking if token exists (placeholder for future JWT implementation)
    token = credentials.credentials

    # For now, return a mock user - in production, decode JWT and validate
    # This is just a placeholder until proper JWT auth is implemented
    return {"username": "mock_user", "user_id": 1, "fpl_id": "123456"}


def load_fpl_data(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Load FPL data using existing DataLoader.
    Cached to avoid repeated API calls, but refreshes if stale (24+ hours).
    """
    global _data_cache

    # Check if data is stale (older than 24 hours)
    if _data_cache is not None and not force_refresh:
        last_updated = _data_cache.get('last_updated')
        if last_updated:
            hours_old = (pd.Timestamp.now() - last_updated).total_seconds() / 3600
            if hours_old < DATA_FRESHNESS_HOURS:
                print(f"[DATA] Using cached data ({hours_old:.1f} hours old)")
                return _data_cache
            else:
                print(f"[DATA] Data is stale ({hours_old:.1f} hours old), refreshing...")
                print("[DATA] Data is stale, refreshing...")

    # Force refresh or no cache available
    if force_refresh:
        print("[DATA] Force refresh requested, fetching fresh data...")

    try:
        logger.info("Loading FPL data...")

        # Fetch data using existing DataLoader
        df_understat, df_fpl, df_fixtures = data_loader.fetch_all_data()

        print(f"[DATA] FPL data fetched - Players: {len(df_fpl)}, Fixtures: {len(df_fixtures)}")
        print(f"[DATA] FPL columns: {df_fpl.columns.tolist()[:10]}...")  # First 10 columns
        print(f"[DATA] Fixture columns: {df_fixtures.columns.tolist()}")

        # Merge and enrich data
        df_merged = data_loader.merge_data(df_understat, df_fpl)

        # Add fixture information (data already enriched in merge_data)

        # Calculate metrics using Optimizer
        df_metrics = optimizer.calculate_metrics(df_merged, df_fixtures)

        # Add xP predictions using FPL data
        print("[DATA] Adding xP predictions using FPL data...")

        # Primary: Use FPL's ep_next (expected points next)
        if 'ep_next' in df_metrics.columns:
            df_metrics['predicted_xP_per_90'] = pd.to_numeric(df_metrics['ep_next'], errors='coerce')
            print(f"[DATA] Using ep_next as predicted_xP_per_90 - sample: {df_metrics['ep_next'].head(3).tolist()}")

        # Secondary fallback: Use form (recent performance)
        if df_metrics['predicted_xP_per_90'].isna().all() or (df_metrics['predicted_xP_per_90'] == 0).all():
            if 'form' in df_metrics.columns:
                df_metrics['predicted_xP_per_90'] = pd.to_numeric(df_metrics['form'], errors='coerce')
                print("[DATA] Using form as predicted_xP_per_90 (secondary fallback)")
            else:
                print("[DATA] WARNING: No form column available")

        # Tertiary fallback: Use xG + xA combination (if available from Understat merge)
        if df_metrics['predicted_xP_per_90'].isna().all() or (df_metrics['predicted_xP_per_90'] == 0).all():
            if 'xG_per_90' in df_metrics.columns and 'xA_per_90' in df_metrics.columns:
                df_metrics['predicted_xP_per_90'] = (
                    pd.to_numeric(df_metrics['xG_per_90'], errors='coerce').fillna(0) +
                    pd.to_numeric(df_metrics['xA_per_90'], errors='coerce').fillna(0) * 2.0
                )
                print("[DATA] Using xG+xA combination as predicted_xP_per_90 (tertiary fallback)")
            else:
                print("[DATA] WARNING: No xG/xA columns available")

        # Final fallback: Default value for any remaining NaN values
        df_metrics['predicted_xP_per_90'] = df_metrics['predicted_xP_per_90'].fillna(2.5)

        print(f"[DATA] Final predicted_xP_per_90 stats:")
        print(f"[DATA] - Mean: {df_metrics['predicted_xP_per_90'].mean():.2f}")
        print(f"[DATA] - Non-zero values: {(df_metrics['predicted_xP_per_90'] > 0).sum()}")
        print(f"[DATA] - Sample values: {df_metrics['predicted_xP_per_90'].head(5).tolist()}")

        df_predictions = df_metrics
        print(f"[DATA] xP predictions added. Shape: {df_predictions.shape}")

        # Filter out invalid players (those without proper names)
        valid_players = df_predictions[
            df_predictions['web_name'].notna() &
            (df_predictions['web_name'].str.len() > 0) &
            ~df_predictions['web_name'].str.contains('Unknown', na=False, case=False)
        ].copy()

        if len(valid_players) != len(df_predictions):
            print(f"[DATA] Filtered out {len(df_predictions) - len(valid_players)} invalid players")
            df_predictions = valid_players

        print(f"[DATA] Final dataset: {len(df_predictions)} players")
        print(f"[DATA] Sample players: {df_predictions[['web_name', 'team_name', 'position', 'predicted_xP_per_90']].head(5).to_dict('records')}")

        _data_cache = {
            "players_df": df_predictions,
            "fixtures_df": df_fixtures,
            "last_updated": pd.Timestamp.now()
        }

        logger.info(f"Loaded data for {len(df_metrics)} players")
        return _data_cache

    except Exception as e:
        logger.error(f"Error loading FPL data: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to load FPL data"
        )


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


# Authentication endpoints will be implemented in the next step
@app.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Authenticate user and return token."""
    print(f"🔐 [AUTH] Login attempt for user: {request.username}")
    try:
        # Verify user credentials using existing DatabaseManager
        user_data = db_manager.verify_user(request.username, request.password)

        if user_data:
            print(f"✅ [AUTH] Login successful for user: {user_data['username']} (ID: {user_data['id']})")
            # For now, return a simple token (in production, use JWT)
            # This is a simplified approach - in production implement proper JWT
            token = f"mock_token_{user_data['id']}_{request.username}"

            return LoginResponse(
                success=True,
                message=f"Welcome back, {user_data['username']}!",
                token=token,
                user={
                    "id": user_data["id"],
                    "username": user_data["username"],
                    "email": user_data["email"],
                    "fpl_id": user_data.get("fpl_id"),
                    "subscription_plan": user_data.get("subscription_plan", "free")
                }
            )
        else:
            print(f"❌ [AUTH] Login failed for user: {request.username} - Invalid credentials")
            return LoginResponse(
                success=False,
                message="Invalid username or password",
                token=None,
                user=None
            )

    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service temporarily unavailable"
        )


@app.get("/players", response_model=PlayersResponse)
async def get_players():
    """Get all players data."""
    try:
        # Load FPL data (with staleness check)
        data = load_fpl_data()
        df_players = data["players_df"]

        # Convert DataFrame to PlayerData objects
        players = []
        for _, row in df_players.iterrows():
            try:
                player = PlayerData(
                    id=int(row.get('id', 0)),
                    web_name=str(row.get('web_name', '')),
                    team_name=str(row.get('team_name', '')),
                    position=str(row.get('position', '')),
                    price=float(row.get('price', 0.0)),
                    form=float(row.get('form', 0.0)) if pd.notna(row.get('form')) else None,
                    total_points=int(row.get('total_points', 0)) if pd.notna(row.get('total_points')) else None,
                    minutes=int(row.get('minutes', 0)) if pd.notna(row.get('minutes')) else None,
                    goals_scored=int(row.get('goals_scored', 0)) if pd.notna(row.get('goals_scored')) else None,
                    assists=int(row.get('assists', 0)) if pd.notna(row.get('assists')) else None,
                    clean_sheets=int(row.get('clean_sheets', 0)) if pd.notna(row.get('clean_sheets')) else None,
                    saves=int(row.get('saves', 0)) if pd.notna(row.get('saves')) else None,
                    expected_goals_per_90=float(row.get('expected_goals_per_90', 0.0)) if pd.notna(row.get('expected_goals_per_90')) else None,
                    expected_assists_per_90=float(row.get('expected_assists_per_90', 0.0)) if pd.notna(row.get('expected_assists_per_90')) else None,
                    threat=float(row.get('threat', 0.0)) if pd.notna(row.get('threat')) else None,
                    creativity=float(row.get('creativity', 0.0)) if pd.notna(row.get('creativity')) else None,
                    influence=float(row.get('influence', 0.0)) if pd.notna(row.get('influence')) else None,
                    ict_index=float(row.get('ict_index', 0.0)) if pd.notna(row.get('ict_index')) else None,
                    gw19_xP=float(row.get('gw19_xP', 0.0)) if pd.notna(row.get('gw19_xP')) else None,
                    long_term_xP=float(row.get('long_term_xP', 0.0)) if pd.notna(row.get('long_term_xP')) else None,
                    risk_adjusted_xP=float(row.get('risk_adjusted_xP', 0.0)) if pd.notna(row.get('risk_adjusted_xP')) else None,
                    availability_score=float(row.get('availability_score', 0.0)) if pd.notna(row.get('availability_score')) else None,
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


@app.post("/optimize/dream-team", response_model=DreamTeamResponse)
async def optimize_dream_team(
    request: DreamTeamRequest,
    req: Request,  # FastAPI Request object to access headers
    user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
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


@app.get("/user/team", response_model=UserTeamResponse)
async def get_user_team(user: Optional[Dict[str, Any]] = Depends(get_current_user)):
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
