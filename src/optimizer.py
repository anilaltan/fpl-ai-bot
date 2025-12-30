"""
FPL Optimizer Module

This module contains the Optimizer class responsible for calculating player metrics,
fixture difficulty ratings, and optimizing squad selection for Fantasy Premier League.
"""

import logging
import ast
from typing import Any, Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

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
            
            team_xga: Dict[str, float] = {}
            all_teams = (
                set(played['home_team'].dropna().unique()) | 
                set(played['away_team'].dropna().unique())
            )
            
            default_xga = self.config['optimizer']['default_team_xga']
            
            for team in all_teams:
                if not isinstance(team, str):
                    continue
                
                home_matches = played[played['home_team'] == team]
                away_matches = played[played['away_team'] == team]
                
                total_xga = (
                    home_matches['xG_a'].sum() + 
                    away_matches['xG_h'].sum()
                )
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
        
        Args:
            team_name: Team name in FPL format.
            df_fixtures: DataFrame containing fixture data.
            fdr_map: Dictionary mapping opponent teams to FDR multipliers.
            weeks: Number of weeks to project.
            
        Returns:
            Total fixture strength score.
        """
        try:
            mapped_team = self.map_team_name(team_name)
            
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
            df_players: DataFrame with player data including predicted_xP_per_90.
            
        Returns:
            DataFrame with added base_xP column.
        """
        try:
            df = df_players.copy()
            
            games_per_season = self.config['optimizer']['games_per_season']
            max_minutes = self.config['optimizer']['max_minutes_per_game']
            
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
        df_fixtures: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate player metrics including short-term and long-term projections.
        
        This method orchestrates the calculation of fixture difficulty, base projections,
        and gameweek-specific expected points.
        
        Args:
            df_players: DataFrame with player data.
            df_fixtures: DataFrame with fixture data.
            
        Returns:
            DataFrame with added projection columns.
        """
        logger.info("Calculating player metrics and projections")
        
        try:
            # Step 1: Calculate team defensive strength
            team_xga = self.calculate_team_defensive_strength(df_fixtures)
            
            # Step 2: Calculate FDR map
            fdr_map = self.calculate_fdr_map(team_xga)
            
            # Step 3: Filter players by minimum minutes
            min_minutes = self.config['optimizer']['min_minutes_for_optimization']
            df = df_players[df_players['minutes'] >= min_minutes].copy()
            
            if df.empty:
                logger.warning("No players meet minimum minutes requirement")
                return df
            
            # Step 4: Calculate base projection
            df = self.calculate_base_projection(df)

            # Step 4.1: Risk yönetimi (dakika varyansı)
            df['minutes_risk'] = self.calculate_minutes_risk(df)

            def _risk_multiplier(row: pd.Series) -> float:
                multiplier = 1.0

                # Team penalty applies regardless of individual minutes risk
                if row['team_name'] in self.risk_teams:
                    multiplier *= self.team_penalty_multiplier

                # Individual variance penalty
                if row['minutes_risk'] > self.risk_threshold:
                    multiplier *= self.risk_coefficient

                return multiplier

            df['risk_multiplier'] = df.apply(_risk_multiplier, axis=1)
            
            # Step 5: Calculate short-term projection (GW19)
            short_term_weeks = self.config['optimizer']['short_term_weeks']
            df['gw19_strength'] = df['team_name'].apply(
                lambda x: self.calculate_fixture_strength(
                    x, df_fixtures, fdr_map, short_term_weeks
                )
            )
            df['gw19_xP'] = df['base_xP'] * df['gw19_strength'] * df['risk_multiplier']
            
            # Step 6: Calculate long-term projection (5 weeks)
            long_term_weeks = self.config['optimizer']['long_term_weeks']
            df['gw5_strength'] = df['team_name'].apply(
                lambda x: self.calculate_fixture_strength(
                    x, df_fixtures, fdr_map, long_term_weeks
                )
            )
            df['long_term_xP'] = df['base_xP'] * df['gw5_strength'] * df['risk_multiplier']
            df['final_5gw_xP'] = df['long_term_xP']
            
            logger.info(f"Metrics calculated for {len(df)} players")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}", exc_info=True)
            raise
    
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
        target_metric: str = 'gw19_xP', 
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

            # Position constraints
            for pos, lim in pos_limits.items():
                idxs = [i for i in range(len(pool)) if str(pool.loc[i, 'position']) == str(pos)]
                prob += lpSum(x[i] for i in idxs) == int(lim)

            # Team constraints
            for team in pool['team_name'].unique().tolist():
                idxs = [i for i in range(len(pool)) if pool.loc[i, 'team_name'] == team]
                prob += lpSum(x[i] for i in idxs) <= max_players_per_team

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
        bank_balance: float
    ) -> Optional[Dict[str, Any]]:
        """
        Suggest the best transfer by comparing current team with available players.
        
        Args:
            current_team_df: DataFrame of current team players.
            all_players_df: DataFrame of all available players.
            bank_balance: Current bank balance.
            
        Returns:
            Dictionary with 'out', 'in', and 'gain' keys, or None if no transfer found.
        """
        logger.info("Searching for transfer opportunities")
        
        try:
            # Remove current team from pool
            pool = all_players_df[
                ~all_players_df['web_name'].isin(current_team_df['web_name'])
            ].copy()
            
            if pool.empty:
                logger.warning("No players available for transfer")
                return None
            
            best_transfer = None
            max_gain = 0.0
            
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
                
                if gain > max_gain:
                    max_gain = gain
                    best_transfer = {
                        'out': p_out,
                        'in': best_in,
                        'gain': gain
                    }
            
            if best_transfer:
                logger.info(
                    f"Transfer suggestion found: "
                    f"{best_transfer['out']['web_name']} -> "
                    f"{best_transfer['in']['web_name']}, gain: {max_gain:.2f}"
                )
            else:
                logger.info("No beneficial transfer found")
            
            return best_transfer
            
        except Exception as e:
            logger.error(f"Error suggesting transfer: {e}", exc_info=True)
            return None
