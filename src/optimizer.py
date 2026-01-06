"""
FPL Optimizer Module

This module contains the Optimizer class responsible for calculating player metrics,
fixture difficulty ratings, and optimizing squad selection for Fantasy Premier League.
"""

import ast
from typing import Any, Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

# Import centralized logger
from src.logger import logger

from src.news_radar import NewsRadar

# Configure logging
logger = logging.getLogger(__name__)


class Optimizer:
    """
    FPL squad optimizer for calculating player projections and building optimal teams.
    
    This class handles fixture difficulty calculations, player metric calculations,
    and squad optimization following SOLID principles.
    """
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize the optimizer with configuration.
        
        Args:
            config_path: Optional path to config.yaml file. If None, uses default path.
        """
        self.config = self._load_config(config_path)
        self.team_name_mapping = self.config['optimizer']['team_name_mapping']
        rotation_cfg = self.config['optimizer'].get('rotation_risk', {})
        self.risk_threshold: float = float(rotation_cfg.get('minutes_var_threshold', 400.0))
        self.risk_coefficient: float = float(rotation_cfg.get('risk_coefficient', 0.85))
        self.team_penalty_multiplier: float = float(rotation_cfg.get('team_penalty_multiplier', 0.9))
        self.risk_teams = set(rotation_cfg.get('risk_teams', []))
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Optional path to config file.
            
        Returns:
            Dictionary containing optimizer configuration.
            
        Raises:
            FileNotFoundError: If config file cannot be found.
            yaml.YAMLError: If config file is invalid YAML.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config.yaml'
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in config file: {e}")
            raise
    
    def calculate_team_defensive_strength(
        self,
        df_fixtures: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate defensive strength (xGA) for each team.

        Args:
            df_fixtures: DataFrame containing fixture data with xG information.

        Returns:
            Dictionary mapping team names to their average xGA per match.
        """
        try:
            played = df_fixtures[df_fixtures['isResult'] == True].copy()

            if played.empty:
                logger.warning("No played fixtures found for defensive strength calculation")
                return {}

            # Dynamic column detection for FPL API compatibility - expanded to handle abbreviated columns
            home_col = ('home_team' if 'home_team' in played.columns else
                       'team_h' if 'team_h' in played.columns else
                       'h' if 'h' in played.columns else None)
            away_col = ('away_team' if 'away_team' in played.columns else
                       'team_a' if 'team_a' in played.columns else
                       'a' if 'a' in played.columns else None)

            if not home_col or not away_col:
                logger.error(f"Could not find home/away team columns. Available columns: {played.columns.tolist()}")
                return {}

            # Extract team names from columns (handle both string and dict formats)
            def extract_team_names(series):
                """Extract team names from a series that may contain strings or dicts"""
                teams = []
                for val in series.dropna():
                    if isinstance(val, dict):
                        # Extract team name from dict format: {"id": "87", "title": "Liverpool", "short_title": "LIV"}
                        team_name = val.get('title') or val.get('name') or str(val)
                        teams.append(team_name)
                    else:
                        # Handle string format
                        teams.append(str(val))
                return teams

            home_teams = extract_team_names(played[home_col])
            away_teams = extract_team_names(played[away_col])
            all_teams = set(home_teams + away_teams)

            team_xga: Dict[str, float] = {}
            default_xga = self.config['optimizer']['default_team_xga']

            for team in all_teams:
                if not isinstance(team, str):
                    continue

                # Helper function to check if a dict column contains the team
                def matches_team(val, target_team):
                    if isinstance(val, dict):
                        return (val.get('title') == target_team or
                               val.get('name') == target_team or
                               str(val) == target_team)
                    else:
                        return str(val) == target_team

                home_matches = played[played[home_col].apply(lambda x: matches_team(x, team))]
                away_matches = played[played[away_col].apply(lambda x: matches_team(x, team))]
                
                # Extract xGA from xG column (handle both dict and separate column formats)
                def extract_xga(row, is_home_game):
                    """Extract expected goals against from a fixture row"""
                    if isinstance(row.get('xG'), dict):
                        # Dictionary format: {"h": "2.33", "a": "1.57"}
                        if is_home_game:
                            # Team playing home concedes goals from away team
                            return float(row['xG'].get('a', 0))
                        else:
                            # Team playing away concedes goals from home team
                            return float(row['xG'].get('h', 0))
                    else:
                        # Fallback to separate columns if they exist
                        if is_home_game and 'xG_a' in row.index:
                            return float(row.get('xG_a', 0))
                        elif not is_home_game and 'xG_h' in row.index:
                            return float(row.get('xG_h', 0))
                        else:
                            return 0.0

                home_xga = home_matches.apply(lambda row: extract_xga(row, True), axis=1).sum()
                away_xga = away_matches.apply(lambda row: extract_xga(row, False), axis=1).sum()
                total_xga = home_xga + away_xga
                matches = len(home_matches) + len(away_matches)
                
                team_xga[team] = (
                    total_xga / matches if matches > 0 else default_xga
                )
            
            logger.info(f"Calculated defensive strength for {len(team_xga)} teams")
            return team_xga
            
        except Exception as e:
            logger.error(
                f"Error calculating team defensive strength: {e}",
                exc_info=True
            )
            return {}
    
    def calculate_fdr_map(
        self, 
        team_xga: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate Fixture Difficulty Rating (FDR) map from team xGA.
        
        Args:
            team_xga: Dictionary mapping team names to average xGA.
            
        Returns:
            Dictionary mapping team names to FDR multipliers.
        """
        try:
            if not team_xga:
                return {}
            
            avg_xga = np.mean(list(team_xga.values()))
            fdr_map = {team: xga / avg_xga for team, xga in team_xga.items()}
            
            return fdr_map
            
        except Exception as e:
            logger.error(f"Error calculating FDR map: {e}", exc_info=True)
            return {}
    
    def map_team_name(self, team_name: str) -> str:
        """
        Map FPL team name to Understat team name.
        
        Args:
            team_name: FPL team name.
            
        Returns:
            Mapped team name (or original if no mapping exists).
        """
        return self.team_name_mapping.get(team_name, team_name)
    
    def calculate_fixture_strength(
        self,
        team_name: str,
        df_fixtures: pd.DataFrame,
        fdr_map: Dict[str, float],
        weeks: int
    ) -> float:
        """
        Calculate fixture strength for a team over specified weeks.

        Eğer fixture verisi yoksa default score döndürür.

        Args:
            team_name: Team name in FPL format.
            df_fixtures: DataFrame containing fixture data (may be None/empty).
            fdr_map: Dictionary mapping opponent teams to FDR multipliers.
            weeks: Number of weeks to project.

        Returns:
            Total fixture strength score.
        """
        try:
            # Check if fixtures are available
            if df_fixtures is None or df_fixtures.empty or fdr_map is None or not fdr_map:
                default_score = self.config['optimizer']['default_fdr_score']
                return weeks * default_score

            mapped_team = self.map_team_name(team_name)

            # Check if required columns exist
            required_cols = ['home_team', 'away_team', 'isResult']
            if not all(col in df_fixtures.columns for col in required_cols):
                default_score = self.config['optimizer']['default_fdr_score']
                return weeks * default_score

            upcoming = df_fixtures[
                ((df_fixtures['home_team'] == mapped_team) |
                 (df_fixtures['away_team'] == mapped_team)) &
                (df_fixtures['isResult'] == False)
            ].sort_values('datetime').head(weeks)

            if upcoming.empty:
                default_score = self.config['optimizer']['default_fdr_score']
                return weeks * default_score

            home_mult = self.config['optimizer']['home_advantage_multiplier']
            away_mult = self.config['optimizer']['away_disadvantage_multiplier']

            score = 0.0
            for _, row in upcoming.iterrows():
                is_home = row['home_team'] == mapped_team
                opponent = row['away_team'] if is_home else row['home_team']

                fdr_multiplier = fdr_map.get(opponent, 1.0)
                site_multiplier = home_mult if is_home else away_mult

                score += fdr_multiplier * site_multiplier

            return score

        except Exception as e:
            logger.error(
                f"Error calculating fixture strength for {team_name}: {e}",
                exc_info=True
            )
            default_score = self.config['optimizer']['default_fdr_score']
            return weeks * default_score
    
    def calculate_base_projection(
        self,
        df_players: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate base expected points projection for players.

        Args:
            df_players: DataFrame with player data including predicted_xP.

        Returns:
            DataFrame with added base_xP column.
        """
        try:
            df = df_players.copy()

            games_per_season = self.config['optimizer']['games_per_season']
            max_minutes = self.config['optimizer']['max_minutes_per_game']

            # Use predicted_xP from FPL API (ep_next mapped)
            if 'predicted_xP' not in df.columns:
                logger.warning("predicted_xP column missing, using fallback")
                df['predicted_xP'] = 2.5  # Default FPL expected points

            # Convert predicted_xP to per-90-minute rate (simplified)
            df['predicted_xP_per_90'] = pd.to_numeric(df['predicted_xP'], errors='coerce').fillna(2.5)

            df['avg_mins'] = (df['minutes'] / games_per_season).clip(upper=max_minutes)
            df['base_xP'] = df['predicted_xP_per_90'] * (df['avg_mins'] / max_minutes)

            return df

        except Exception as e:
            logger.error(f"Error calculating base projection: {e}", exc_info=True)
            raise

    # -------------------- Risk Management -------------------- #
    def calculate_minutes_risk(self, df_players: pd.DataFrame) -> pd.Series:
        """
        Son 5 maçtaki oynanan dakikaların varyansını hesaplar.
        Dakika listesi bulunamazsa 0 döner (risk yok sayılır).
        """
        candidate_cols = [
            'recent_minutes', 'minutes_history', 'minutes_last_five',
            'past_minutes', 'gw_minutes'
        ]

        def _safe_list(val):
            if isinstance(val, list):
                return val
            if isinstance(val, str):
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    return None
            return None

        risks: List[float] = []
        for _, row in df_players.iterrows():
            minutes_list = []
            for col in candidate_cols:
                if col in row.index:
                    lst = _safe_list(row[col])
                    if lst:
                        minutes_list = [float(x) for x in lst if pd.notna(x)]
                        break
            if minutes_list:
                last5 = minutes_list[-5:] if len(minutes_list) >= 5 else minutes_list
                risks.append(float(np.var(last5)) if len(last5) >= 2 else 0.0)
            else:
                risks.append(0.0)

        return pd.Series(risks, index=df_players.index, name='minutes_risk')
    
    def calculate_metrics(
        self,
        df_players: pd.DataFrame,
        df_fixtures: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Calculate player metrics including projections.

        Eğer fixture verisi yoksa, sadece predicted_xP ve price bazlı basit optimizasyon yapar.

        Args:
            df_players: DataFrame with player data.
            df_fixtures: DataFrame with fixture data (optional).

        Returns:
            DataFrame with added projection columns.
        """
        logger.info("Calculating player metrics and projections")

        try:
            # Check if fixtures are available
            fixtures_available = df_fixtures is not None and not df_fixtures.empty

            if fixtures_available:
                # Full calculation with fixtures
                team_xga = self.calculate_team_defensive_strength(df_fixtures)
                fdr_map = self.calculate_fdr_map(team_xga)
                logger.info("Using fixture-based FDR calculations")
            else:
                # Fallback: no fixture data, use simplified approach
                fdr_map = {}
                logger.info("No fixture data available, using simplified FDR (all teams = 1.0)")

            # Step 3: Filter players by minimum minutes
            min_minutes = self.config['optimizer']['min_minutes_for_optimization']
            df = df_players[df_players['minutes'] >= min_minutes].copy()

            if df.empty:
                logger.warning("No players meet minimum minutes requirement")
                return df

            # Step 4: Calculate base projection
            df = self.calculate_base_projection(df)

            # Step 4.1: Risk yönetimi (dakika varyansı) - check if column exists
            if 'minutes_last_five' in df.columns:
                df['minutes_risk'] = self.calculate_minutes_risk(df)
            else:
                df['minutes_risk'] = 0.0  # No risk data available

            def _risk_multiplier(row: pd.Series) -> float:
                multiplier = 1.0

                # Team penalty applies regardless of individual minutes risk
                if 'team_name' in row.index and row['team_name'] in self.risk_teams:
                    multiplier *= self.team_penalty_multiplier

                # Individual variance penalty - only if risk data available
                if 'minutes_risk' in row.index and row['minutes_risk'] > self.risk_threshold:
                    multiplier *= self.risk_coefficient

                return multiplier

            df['risk_multiplier'] = df.apply(_risk_multiplier, axis=1)

            # Step 5: Calculate projections
            if fixtures_available:
                # Full ensemble with fixtures
                short_term_weeks = self.config['optimizer']['short_term_weeks']
                df['gw19_strength'] = df['team_name'].apply(
                    lambda x: self.calculate_fixture_strength(
                        x, df_fixtures, fdr_map, short_term_weeks
                    )
                )

                # Calculate individual expert scores
                base_xp_risked = df['base_xP'] * df['risk_multiplier']
                df['technical_score'] = base_xp_risked * df['gw19_strength']
                df['market_score'] = self._calculate_market_score(df)
                df['tactical_score'] = self._calculate_tactical_score(df)

                # Ensemble: Weighted voting
                df['gw19_xP'] = (
                    df['technical_score'] * 0.50 +
                    df['market_score'] * 0.30 +
                    df['tactical_score'] * 0.20
                )
            else:
                # Simplified: just use base_xP with risk adjustment
                df['gw19_xP'] = df['base_xP'] * df['risk_multiplier']
                logger.info("Using simplified projection (no fixtures)")

            # Risk yönetimi entegrasyonu - check if NewsRadar data available
            try:
                radar = NewsRadar()
                df['availability_score'] = df.apply(radar.calculate_availability_score, axis=1)
                df['risk_adjusted_xP'] = df['gw19_xP'] * df['availability_score']
            except Exception:
                df['availability_score'] = 1.0  # Assume all available
                df['risk_adjusted_xP'] = df['gw19_xP']
                logger.warning("NewsRadar availability scoring failed, using defaults")

            # Step 6: Calculate long-term projection
            if fixtures_available:
                long_term_weeks = self.config['optimizer']['long_term_weeks']
                df['gw5_strength'] = df['team_name'].apply(
                    lambda x: self.calculate_fixture_strength(
                        x, df_fixtures, fdr_map, long_term_weeks
                    )
                )
                df['long_term_xP'] = df['base_xP'] * df['gw5_strength'] * df['risk_multiplier']
            else:
                # Simplified long-term: assume same as short-term
                df['long_term_xP'] = df['base_xP'] * df['risk_multiplier']

            df['final_5gw_xP'] = df['long_term_xP'] * df['availability_score']

            logger.info(f"Metrics calculated for {len(df)} players (fixtures: {fixtures_available})")

            return df

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}", exc_info=True)
            raise

    def _calculate_market_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate market score based on betting odds (Model B: The Bookie).

        Args:
            df: DataFrame with player data including odds features.

        Returns:
            Series of market scores.
        """
        try:
            # Base points from odds probabilities
            goal_prob = df.get('implied_goal_prob', pd.Series(0.0, index=df.index))
            cs_prob = df.get('implied_cs_prob', pd.Series(0.0, index=df.index))

            # Position-based expected points from odds
            position_multipliers = {
                'GK': 4.0,   # Clean sheet focused
                'DEF': 3.0,  # Clean sheet + occasional goals
                'MID': 2.5,  # Goals + assists
                'FWD': 3.5   # Primary goal scorers
            }

            if 'position' in df.columns:
                pos_multiplier = df['position'].map(position_multipliers).fillna(2.5)
            else:
                pos_multiplier = pd.Series(2.5, index=df.index)

            # Market score = probability-weighted expected points
            market_score = (goal_prob * pos_multiplier * 0.7) + (cs_prob * pos_multiplier * 0.3)

            return market_score

        except Exception as e:
            logger.warning(f"Market score calculation failed: {e}")
            return pd.Series(0.0, index=df.index)

    def _calculate_tactical_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate tactical score based on matchup advantage and set piece threat.

        Args:
            df: DataFrame with player data including tactical features.

        Returns:
            Series of tactical scores.
        """
        try:
            # Get tactical features
            matchup = df.get('matchup_advantage', pd.Series(1.0, index=df.index))
            set_piece = df.get('set_piece_threat', pd.Series(0.0, index=df.index))

            # Tactical score = matchup advantage + set piece contribution
            tactical_score = matchup + (set_piece * 0.5)  # Set pieces contribute 50% weight

            # Normalize to expected points scale (roughly 0-10 range)
            tactical_score = tactical_score * 2.0  # Scale up for meaningful contribution

            return tactical_score

        except Exception as e:
            logger.warning(f"Tactical score calculation failed: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _create_initial_squad(
        self,
        pool: pd.DataFrame,
        target_metric: str
    ) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int], float]:
        """
        Create initial squad by selecting top players within constraints.
        
        Args:
            pool: DataFrame of available players.
            target_metric: Column name to optimize for.
            
        Returns:
            Tuple of (squad_dataframe, position_counts, team_counts, total_cost).
        """
        squad: List[pd.Series] = []
        selected_names: set = set()
        
        current_cost = 0.0
        pos_limits = self.config['optimizer']['position_limits']
        pos_counts: Dict[str, int] = {
            pos: 0 for pos in pos_limits.keys()
        }
        team_counts: Dict[str, int] = {
            team: 0 for team in pool['team_name'].unique()
        }
        squad_size = self.config['optimizer']['squad_size']
        max_players_per_team = self.config['optimizer']['max_players_per_team']
        
        for _, player in pool.iterrows():
            if len(squad) >= squad_size:
                break
            
            pos = player['position']
            team = player['team_name']
            name = player['web_name']
            
            if name not in selected_names:
                if (pos_counts.get(pos, 0) < pos_limits.get(pos, 0) and
                    team_counts.get(team, 0) < max_players_per_team):
                    
                    squad.append(player)
                    selected_names.add(name)
                    pos_counts[pos] = pos_counts.get(pos, 0) + 1
                    team_counts[team] = team_counts.get(team, 0) + 1
                    current_cost += player['price']
        
        squad_df = pd.DataFrame(squad)
        return squad_df, pos_counts, team_counts, current_cost
    
    def _find_best_swap(
        self,
        squad_df: pd.DataFrame,
        pool: pd.DataFrame,
        team_counts: Dict[str, int],
        target_metric: str
    ) -> Optional[Tuple[pd.Index, pd.Series]]:
        """
        Find the best player swap to reduce cost while minimizing point loss.
        
        Args:
            squad_df: Current squad DataFrame.
            pool: Available players pool.
            team_counts: Current team player counts.
            target_metric: Column name to optimize for.
            
        Returns:
            Tuple of (outgoing_player_index, incoming_player_series) or None.
        """
        best_swap = None
        min_loss_ratio = float('inf')
        
        current_squad_names = squad_df['web_name'].tolist()
        candidates = pool[~pool['web_name'].isin(current_squad_names)]
        max_players_per_team = self.config['optimizer']['max_players_per_team']
        
        for idx, p_out in squad_df.iterrows():
            avail = candidates[
                (candidates['position'] == p_out['position']) &
                (candidates['price'] < p_out['price'])
            ]
            
            for _, p_in in avail.iterrows():
                team_ok = (
                    p_in['team_name'] == p_out['team_name'] or
                    team_counts.get(p_in['team_name'], 0) < max_players_per_team
                )
                
                if team_ok:
                    price_diff = p_out['price'] - p_in['price']
                    xp_diff = p_out[target_metric] - p_in[target_metric]
                    
                    if price_diff > 0:
                        ratio = xp_diff / price_diff
                        if ratio < min_loss_ratio:
                            min_loss_ratio = ratio
                            best_swap = (idx, p_in)
        
        return best_swap
    
    def solve_dream_team(
        self,
        df: pd.DataFrame,
        target_metric: str = 'risk_adjusted_xP',  # Risk-aware optimization default
        budget: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Solve for optimal dream team within budget constraints.
        
        Uses an Integer Linear Program (ILP) to maximize:
            total_points(15 players) + captain_points
        which reflects FPL captaincy (2x points for one player) without hard-locking
        any specific player into the squad.
        
        Args:
            df: DataFrame with player data and projections.
            target_metric: Column name to optimize for (default: 'gw19_xP').
            budget: Maximum budget. If None, uses config value.
            
        Returns:
            DataFrame containing optimal squad.
        """
        logger.info(f"Solving dream team with target metric: {target_metric}")
        
        try:
            if budget is None:
                budget = self.config['optimizer']['budget']
            
            # Work on a copy to avoid mutating upstream data
            work_df = df.copy()

            # BULLETPROOF DATA SANITIZATION - Optimizasyondan önce kritik sütunları temizle
            logger.info("Applying data sanitization before optimization...")

            # Kritik sütunlardaki NaN veya Sonsuz değerleri temizle
            work_df['price'] = work_df['price'].fillna(100.0)  # Fiyat yoksa çok pahalı yap ki seçmesin
            work_df['predicted_xP'] = work_df['predicted_xP'].fillna(0.0)

            # Diğer optimizasyon için gerekli sütunları da temizle
            if target_metric in work_df.columns:
                work_df[target_metric] = pd.to_numeric(work_df[target_metric], errors='coerce').fillna(0.0)

            # Veri tiplerini logla (Debug için)
            logger.debug("Data Types Check:")
            logger.debug(f"   - price: {work_df['price'].dtype} (range: {work_df['price'].min():.1f}-{work_df['price'].max():.1f})")
            logger.debug(f"   - predicted_xP: {work_df['predicted_xP'].dtype} (range: {work_df['predicted_xP'].min():.2f}-{work_df['predicted_xP'].max():.2f})")
            if target_metric in work_df.columns:
                logger.debug(f"   - {target_metric}: {work_df[target_metric].dtype} (range: {work_df[target_metric].min():.2f}-{work_df[target_metric].max():.2f})")

            # NaN veya infinite değer kontrolü
            nan_check = work_df[['price', 'predicted_xP', target_metric]].isna().sum()
            if nan_check.sum() > 0:
                logger.warning(f"Found NaN values: {nan_check.to_dict()}")

            # Mandatory injury filter: drop players unlikely to feature
            if 'chance_of_playing_next_round' in work_df.columns:
                work_df['chance_of_playing_next_round'] = pd.to_numeric(
                    work_df['chance_of_playing_next_round'], errors='coerce'
                )
                before_filter = len(work_df)
                # Requirement: drop if chance < 75 (unknown/NaN is kept)
                work_df = work_df[
                    work_df['chance_of_playing_next_round'].isna()
                    | (work_df['chance_of_playing_next_round'] >= 75.0)
                ].copy()
                dropped = before_filter - len(work_df)
                if dropped > 0:
                    logger.info(
                        "Injury filter removed %d players with <75%% chance to play",
                        dropped
                    )

            # Additional risk filter: drop players with very low availability (< 0.25)
            if 'availability_score' in work_df.columns:
                before_risk_filter = len(work_df)
                work_df = work_df[
                    work_df['availability_score'].isna()
                    | (work_df['availability_score'] >= 0.25)  # Çok düşük riskli oyuncuları çıkar
                ].copy()
                risk_dropped = before_risk_filter - len(work_df)
                if risk_dropped > 0:
                    logger.info(
                        "Risk filter removed %d players with availability_score < 0.25",
                        risk_dropped
                    )
            else:
                logger.warning("availability_score column not found, risk filter skipped")
            
            if target_metric not in work_df.columns:
                logger.warning(
                    "Target metric %s not found in dataframe columns", target_metric
                )
                return pd.DataFrame()
            
            # Ensure numeric types for optimization columns
            required_cols = ['web_name', 'team_name', 'position', 'price', target_metric]
            missing = [c for c in required_cols if c not in work_df.columns]
            if missing:
                logger.warning("Missing required columns for optimization: %s", missing)
                return pd.DataFrame()

            work_df[target_metric] = pd.to_numeric(work_df[target_metric], errors='coerce')
            work_df['price'] = pd.to_numeric(work_df['price'], errors='coerce')
            work_df = work_df.dropna(subset=['web_name', 'team_name', 'position', 'price', target_metric]).copy()

            # Deduplicate the pool (keep best projected row per player+team)
            pool = (
                work_df.sort_values(target_metric, ascending=False)
                .drop_duplicates(subset=['web_name', 'team_name'], keep='first')
                .reset_index(drop=True)
            )

            if pool.empty:
                logger.warning("No players available for squad optimization after filtering")
                return pd.DataFrame()

            # ---- ILP: selection + captain ----
            try:
                from pulp import (
                    LpProblem,
                    LpMaximize,
                    LpVariable,
                    LpBinary,
                    lpSum,
                    PULP_CBC_CMD,
                    LpStatus,
                    value,
                )
            except Exception as import_exc:
                logger.error("PuLP import failed; cannot run ILP optimizer: %s", import_exc, exc_info=True)
                raise

            squad_size = int(self.config['optimizer']['squad_size'])
            pos_limits: Dict[str, int] = dict(self.config['optimizer']['position_limits'])
            max_players_per_team = int(self.config['optimizer']['max_players_per_team'])

            # Variables
            x = {
                i: LpVariable(f"x_{i}", lowBound=0, upBound=1, cat=LpBinary)
                for i in range(len(pool))
            }
            c = {
                i: LpVariable(f"c_{i}", lowBound=0, upBound=1, cat=LpBinary)
                for i in range(len(pool))
            }

            prob = LpProblem("fpl_dream_team", LpMaximize)

            points = pool[target_metric].astype(float).tolist()
            prices = pool['price'].astype(float).tolist()

            # Objective:
            #   maximize total_points(15) + captain_points
            # plus a soft "premium captaincy potential" bias (no hard locks):
            #   players with price >= 12.0 get a mild upweight in the objective.
            premium_price_threshold = 12.0
            premium_bias = 1.2
            obj_points = [
                (points[i] * premium_bias) if (prices[i] >= premium_price_threshold) else points[i]
                for i in range(len(points))
            ]

            prob += lpSum(obj_points[i] * x[i] for i in x) + lpSum(obj_points[i] * c[i] for i in c)

            # Squad size
            prob += lpSum(x[i] for i in x) == squad_size

            # Position constraints - KATI FPL KURALI: Pozisyon limitleri
            logger.info("Applying position constraints:")
            for pos, lim in pos_limits.items():
                idxs = [i for i in range(len(pool)) if str(pool.loc[i, 'position']) == str(pos)]
                player_count = len(idxs)
                prob += lpSum(x[i] for i in idxs) == int(lim)
                logger.info(f"   Position '{pos}': {player_count} players available, required = {lim}")

            # Team constraints - KATI FPL KURALI: Max 3 oyuncu aynı takımdan
            logger.info(f"Applying team constraints: max {max_players_per_team} players per team")
            team_constraint_count = 0
            for team in pool['team_name'].unique().tolist():
                idxs = [i for i in range(len(pool)) if pool.loc[i, 'team_name'] == team]
                player_count = len(idxs)
                if player_count > 0:  # Sadece o takımda oyuncu varsa kısıtlama ekle
                    prob += lpSum(x[i] for i in idxs) <= max_players_per_team
                    team_constraint_count += 1
                    logger.info(f"   Team '{team}': {player_count} players available, constraint <= {max_players_per_team}")
            logger.info(f"Added {team_constraint_count} team constraints")

            # Budget
            prob += lpSum(prices[i] * x[i] for i in x) <= float(budget)

            # Captain: exactly one, and must be selected
            prob += lpSum(c[i] for i in c) == 1
            for i in range(len(pool)):
                prob += c[i] <= x[i]

            # Solve
            solver = PULP_CBC_CMD(msg=False, timeLimit=20)
            prob.solve(solver)

            status_str = LpStatus.get(prob.status, str(prob.status))
            if status_str not in ("Optimal", "Feasible"):
                logger.warning("ILP did not find a feasible solution (status=%s)", status_str)
                return pd.DataFrame()

            selected_idxs = [i for i in range(len(pool)) if (value(x[i]) or 0) > 0.5]
            captain_idxs = [i for i in range(len(pool)) if (value(c[i]) or 0) > 0.5]

            squad_df = pool.loc[selected_idxs].copy()
            squad_df['is_captain'] = False
            if captain_idxs:
                squad_df.loc[squad_df.index.isin(captain_idxs), 'is_captain'] = True

            total_cost = float(squad_df['price'].sum()) if not squad_df.empty else 0.0
            total_points = float(squad_df[target_metric].sum()) if not squad_df.empty else 0.0
            captain_points = 0.0
            captain_name = None
            if captain_idxs:
                cap_row = pool.loc[captain_idxs[0]]
                captain_points = float(cap_row[target_metric])
                captain_name = str(cap_row.get('web_name', ''))

            # Takım dağılımını logla - kısıtlama kontrolü için
            if not squad_df.empty:
                team_distribution = squad_df['team_name'].value_counts().sort_values(ascending=False)
                max_from_team = team_distribution.max()
                logger.info("Team distribution in squad:")
                for team, count in team_distribution.items():
                    status = "VIOLATION!" if count > max_players_per_team else "OK"
                    logger.info(f"   {team}: {count} players {status}")
                logger.info(f"Max players from any team: {max_from_team} (limit: {max_players_per_team})")

            logger.info(
                "Dream team solved (ILP): %d players, cost=%.2f, base_points=%.2f, captain=%s (+%.2f)",
                len(squad_df),
                total_cost,
                total_points,
                captain_name,
                captain_points,
            )

            return squad_df.reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Error solving dream team: {e}", exc_info=True)
            raise
    
    def suggest_transfer(
        self,
        current_team_df: pd.DataFrame,
        all_players_df: pd.DataFrame,
        bank_balance: float,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Suggest the best transfer options by comparing current team with available players.

        Args:
            current_team_df: DataFrame of current team players.
            all_players_df: DataFrame of all available players.
            bank_balance: Current bank balance.
            top_n: Number of best suggestions to return.

        Returns:
            List of dictionaries with 'out', 'in', 'gain', and 'remaining_bank' keys.
            Empty list if no beneficial transfers found.
        """
        logger.info(f"Searching for top {top_n} transfer opportunities")

        try:
            # Remove current team from pool
            pool = all_players_df[
                ~all_players_df['web_name'].isin(current_team_df['web_name'])
            ].copy()

            if pool.empty:
                logger.warning("No players available for transfer")
                return []

            suggestions = []

            # Try selling each player in current team
            for idx, p_out in current_team_df.iterrows():
                # Budget = sold player price + bank balance
                available_budget = p_out['price'] + bank_balance

                # Filter candidates (same position, within budget)
                candidates = pool[
                    (pool['position'] == p_out['position']) &
                    (pool['price'] <= available_budget)
                ]

                if candidates.empty:
                    continue

                # Use long_term_xP or final_5gw_xP
                metric_col = (
                    'final_5gw_xP' if 'final_5gw_xP' in candidates.columns
                    else 'long_term_xP'
                )

                # Remove NaN values
                candidates = candidates.dropna(subset=[metric_col])
                if candidates.empty:
                    continue

                # Find best replacement
                best_in = candidates.sort_values(metric_col, ascending=False).iloc[0]

                # Calculate gain
                p_out_score = p_out.get(metric_col, 0.0)
                gain = best_in[metric_col] - p_out_score

                # Calculate remaining bank after transfer
                remaining_bank = available_budget - best_in['price']

                # Only add if gain > 0
                if gain > 0:
                    suggestions.append({
                        'out': p_out,
                        'in': best_in,
                        'gain': gain,
                        'remaining_bank': remaining_bank
                    })

            # Sort by gain (descending) and take top_n
            suggestions.sort(key=lambda x: x['gain'], reverse=True)
            top_suggestions = suggestions[:top_n]

            if top_suggestions:
                logger.info(f"Found {len(top_suggestions)} beneficial transfer suggestions")
                for i, s in enumerate(top_suggestions, 1):
                    logger.info(
                        f"#{i}: {s['out']['web_name']} -> {s['in']['web_name']}, "
                        f"gain: {s['gain']:.2f}, bank: {s['remaining_bank']:.1f}"
                    )
            else:
                logger.info("No beneficial transfers found")

            return top_suggestions

        except Exception as e:
            logger.error(f"Error suggesting transfers: {e}", exc_info=True)
            return []

    def analyze_chips(
        self,
        current_team_df: pd.DataFrame,
        all_players_df: pd.DataFrame,
        gameweek: int = 19
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze FPL chip strategies and provide recommendations.

        Args:
            current_team_df: DataFrame of current team players.
            all_players_df: DataFrame of all available players.
            gameweek: Current gameweek (default: 19 for GW19 analysis).

        Returns:
            Dictionary with chip analysis results.
        """
        logger.info(f"Analyzing chip strategies for GW{gameweek}")

        results = {
            'WC': {'status': 'Unknown', 'reason': '', 'score': 0.0},
            'TC': {'status': 'Unknown', 'reason': '', 'score': 0.0},
            'BB': {'status': 'Unknown', 'reason': '', 'score': 0.0}
        }

        try:
            # Determine GW column name
            gw_col = f'gw{gameweek}_xP'
            if gw_col not in all_players_df.columns:
                gw_col = 'gw19_xP'  # fallback

            # TRIPLE CAPTAIN ANALYSIS
            if gw_col in current_team_df.columns:
                # Find player with highest gw_xP
                tc_candidates = current_team_df.dropna(subset=[gw_col])
                if not tc_candidates.empty:
                    best_player = tc_candidates.loc[tc_candidates[gw_col].idxmax()]
                    tc_score = float(best_player.get(gw_col, 0.0) or 0.0)

                    if tc_score > 11.0:
                        results['TC'] = {
                            'status': 'Öneriliyor',
                            'reason': f"{best_player['web_name']} xP'si {tc_score:.1f}, yüksek potansiyel!",
                            'score': tc_score
                        }
                    elif tc_score > 9.0:
                        results['TC'] = {
                            'status': 'Düşünülebilir',
                            'reason': f"{best_player['web_name']} xP'si {tc_score:.1f}, orta seviye potansiyel.",
                            'score': tc_score
                        }
                    else:
                        results['TC'] = {
                            'status': 'Bekle',
                            'reason': f"En yüksek xP {tc_score:.1f}, henüz erken.",
                            'score': tc_score
                        }

            # BENCH BOOST ANALYSIS
            # Identify bench players (conservative: lowest 4 xP players)
            if gw_col in current_team_df.columns:
                sorted_team = current_team_df.dropna(subset=[gw_col]).sort_values(gw_col, ascending=True)
                bench_players = sorted_team.head(4)  # Assume lowest 4 are bench
                bb_score = float(bench_players[gw_col].sum() or 0.0)

                if bb_score > 18.0:
                    results['BB'] = {
                        'status': 'Öneriliyor',
                        'reason': f"Yedek toplam xP {bb_score:.1f}, güçlü yedekler!",
                        'score': bb_score
                    }
                elif bb_score > 15.0:
                    results['BB'] = {
                        'status': 'Düşünülebilir',
                        'reason': f"Yedek toplam xP {bb_score:.1f}, orta seviye yedekler.",
                        'score': bb_score
                    }
                else:
                    results['BB'] = {
                        'status': 'Bekle',
                        'reason': f"Yedek toplam xP {bb_score:.1f}, henüz erken.",
                        'score': bb_score
                    }

            # WILDCARD ANALYSIS
            try:
                # Current team total xP
                current_team_xp = float(current_team_df[gw_col].sum() or 0.0)

                # Solve for optimal team
                optimal_team = self.solve_dream_team(
                    all_players_df,
                    target_metric=gw_col,
                    budget=100.0  # Standard FPL budget
                )

                if not optimal_team.empty and gw_col in optimal_team.columns:
                    optimal_xp = float(optimal_team[gw_col].sum() or 0.0)
                    opportunity_cost = optimal_xp - current_team_xp

                    if opportunity_cost > 15.0:
                        results['WC'] = {
                            'status': 'Öneriliyor',
                            'reason': f"Kadro verimsiz! Opt: {optimal_xp:.1f}, Mevcut: {current_team_xp:.1f}, Fark: {opportunity_cost:.1f}",
                            'score': opportunity_cost
                        }
                    elif opportunity_cost > 10.0:
                        results['WC'] = {
                            'status': 'İzlemede Kal',
                            'reason': f"Potansiyel var. Opt: {optimal_xp:.1f}, Mevcut: {current_team_xp:.1f}, Fark: {opportunity_cost:.1f}",
                            'score': opportunity_cost
                        }
                    else:
                        results['WC'] = {
                            'status': 'Gerek Yok',
                            'reason': f"Kadro yeterli. Opt: {optimal_xp:.1f}, Mevcut: {current_team_xp:.1f}, Fark: {opportunity_cost:.1f}",
                            'score': opportunity_cost
                        }
                else:
                    results['WC'] = {
                        'status': 'Veri Yok',
                        'reason': 'Optimal kadro hesaplanamadı.',
                        'score': 0.0
                    }

            except Exception as e:
                logger.warning(f"Wildcard analysis failed: {e}")
                results['WC'] = {
                    'status': 'Hata',
                    'reason': f'Analiz başarısız: {str(e)}',
                    'score': 0.0
                }

            logger.info(f"Chip analysis completed: WC={results['WC']['status']}, TC={results['TC']['status']}, BB={results['BB']['status']}")

        except Exception as e:
            logger.error(f"Error analyzing chips: {e}", exc_info=True)

        return results
