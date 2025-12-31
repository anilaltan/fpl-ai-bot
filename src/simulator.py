"""
Monte Carlo Simulation Engine for FPL Risk Analysis

This module provides probabilistic modeling of Fantasy Premier League team performance
through Monte Carlo simulations, converting deterministic xP predictions into
probability distributions.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Container for Monte Carlo simulation results."""
    team_total_points: np.ndarray  # Shape: (n_simulations,)
    player_contributions: Dict[str, np.ndarray]  # player_name -> points array
    captain_points: np.ndarray  # Captain's doubled points
    vice_captain_points: np.ndarray  # Vice captain's doubled points (when captain injured)
    risk_metrics: Dict[str, float]  # VaR, mean, ceiling, etc.


@dataclass
class VolatilityProfile:
    """Volatility settings for different player positions."""
    goalkeeper: float = 1.0
    defender: float = 1.5
    midfielder: float = 2.5
    forward: float = 3.5
    explosive_midfielder_multiplier: float = 1.4  # For Salah, Saka, etc.


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for FPL team performance modeling.

    This class simulates probabilistic outcomes for Fantasy Premier League teams,
    accounting for player volatility, fixture difficulty, and captaincy decisions.
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        n_gameweeks: int = 5,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the Monte Carlo engine.

        Args:
            n_simulations: Number of simulation runs (default: 1000)
            n_gameweeks: Number of gameweeks to simulate (default: 5)
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.n_gameweeks = n_gameweeks
        self.volatility_profile = VolatilityProfile()

        if random_seed is not None:
            np.random.seed(random_seed)

        # Explosive midfielders (high-variance players)
        self.explosive_players = {
            'mohamed-salah', 'bukayo-saka', 'son-heung-min', 'kevin-de-bruyne',
            'phil-foden', 'jack-grealish', 'marcus-rashford', 'gabriel-martinelli',
            'martin-odegaard', 'bruno-fernandes', 'james-maddison', 'jarrod-bowen'
        }

        logger.info(f"Monte Carlo Engine initialized: {n_simulations} simulations, {n_gameweeks} GWs")

    def simulate_team_performance(
        self,
        team_df: pd.DataFrame,
        fixtures_df: Optional[pd.DataFrame] = None,
        captain_name: Optional[str] = None,
        vice_captain_name: Optional[str] = None
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation for a complete FPL team.

        Args:
            team_df: DataFrame with team players (must have 'web_name', 'gw19_xP', 'position')
            fixtures_df: Optional fixture data for difficulty adjustment
            captain_name: Name of captain player
            vice_captain_name: Name of vice-captain player

        Returns:
            SimulationResult with probabilistic outcomes
        """
        try:
            logger.info(f"Starting simulation for {len(team_df)} players")

            # Initialize results storage
            team_totals = np.zeros(self.n_simulations)
            player_contributions = {}
            captain_points = np.zeros(self.n_simulations)
            vice_captain_points = np.zeros(self.n_simulations)

            # Simulate each player
            for _, player in team_df.iterrows():
                player_name = str(player.get('web_name', 'Unknown'))
                base_xp = float(player.get('gw19_xP', 0.0) or 0.0)
                position = str(player.get('position', 'MID')).upper()

                # Calculate player-specific volatility
                volatility = self._calculate_player_volatility(player_name, position)

                # Simulate player performance across gameweeks
                player_points = self._simulate_player_performance(
                    base_xp, volatility, self.n_gameweeks
                )

                player_contributions[player_name] = player_points

                # Add to team total (regular player points)
                team_totals += player_points

            # Handle captaincy (2x multiplier)
            if captain_name and captain_name in player_contributions:
                captain_base = player_contributions[captain_name].copy()
                captain_points = captain_base * 2  # Captain gets 2x points
                # Remove captain's base points and add doubled points
                team_totals = team_totals - captain_base + captain_points
            elif vice_captain_name and vice_captain_name in player_contributions:
                # Auto-sub: Vice captain becomes captain if captain fails
                vice_base = player_contributions[vice_captain_name].copy()
                vice_captain_points = vice_base * 2
                team_totals = team_totals - vice_base + vice_captain_points
                logger.warning(f"Captain {captain_name} not found, using vice-captain {vice_captain_name}")

            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(team_totals)

            result = SimulationResult(
                team_total_points=team_totals,
                player_contributions=player_contributions,
                captain_points=captain_points,
                vice_captain_points=vice_captain_points,
                risk_metrics=risk_metrics
            )

            logger.info(f"Simulation completed. Mean total: {risk_metrics['mean']:.1f}, VaR 95%: {risk_metrics['var_95']:.1f}")
            return result

        except Exception as e:
            logger.error(f"Simulation failed: {e}", exc_info=True)
            raise

    def _calculate_player_volatility(self, player_name: str, position: str) -> float:
        """
        Calculate position and player-specific volatility.

        Args:
            player_name: Player's web_name
            position: Player position (GK, DEF, MID, FWD)

        Returns:
            Volatility (standard deviation) for this player
        """
        # Base volatility by position
        base_volatility = {
            'GK': self.volatility_profile.goalkeeper,
            'DEF': self.volatility_profile.defender,
            'MID': self.volatility_profile.midfielder,
            'FWD': self.volatility_profile.forward
        }.get(position, self.volatility_profile.midfielder)

        # Explosive midfielder multiplier
        player_key = player_name.lower().replace(' ', '-')
        if position == 'MID' and player_key in self.explosive_players:
            base_volatility *= self.volatility_profile.explosive_midfielder_multiplier

        return base_volatility

    def _simulate_player_performance(
        self,
        base_xp: float,
        volatility: float,
        n_gameweeks: int
    ) -> np.ndarray:
        """
        Simulate a player's performance across multiple gameweeks.

        Args:
            base_xp: Base expected points per gameweek
            volatility: Standard deviation of performance
            n_gameweeks: Number of gameweeks to simulate

        Returns:
            Array of total points across n_gameweeks (shape: n_simulations,)
        """
        # Generate noise for each simulation and gameweek
        noise = np.random.normal(
            loc=0.0,
            scale=volatility,
            size=(self.n_simulations, n_gameweeks)
        )

        # Calculate points per gameweek (base + noise), ensure non-negative
        gw_points = np.maximum(0, base_xp + noise)

        # Sum across gameweeks for total points
        total_points = np.sum(gw_points, axis=1)

        return total_points

    def _calculate_risk_metrics(self, team_totals: np.ndarray) -> Dict[str, float]:
        """
        Calculate risk metrics from simulation results.

        Args:
            team_totals: Array of simulated team totals

        Returns:
            Dictionary of risk metrics
        """
        metrics = {}

        # Basic statistics
        metrics['mean'] = float(np.mean(team_totals))
        metrics['median'] = float(np.median(team_totals))
        metrics['std'] = float(np.std(team_totals))

        # Percentiles for risk assessment
        metrics['p5'] = float(np.percentile(team_totals, 5))      # Bottom 5% (floor)
        metrics['p95'] = float(np.percentile(team_totals, 95))    # Top 5% (ceiling)
        metrics['p10'] = float(np.percentile(team_totals, 10))    # Conservative estimate
        metrics['p90'] = float(np.percentile(team_totals, 90))    # Optimistic estimate

        # Value at Risk (VaR) - losses beyond this point are rare
        # VaR 95% = 5th percentile (95% confidence that you'll score at least this)
        metrics['var_95'] = float(np.percentile(team_totals, 5))
        metrics['var_99'] = float(np.percentile(team_totals, 1))

        # Conditional Value at Risk (CVaR) - expected loss beyond VaR
        var_95_value = metrics['var_95']
        tail_losses = team_totals[team_totals <= var_95_value]
        metrics['cvar_95'] = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_95_value

        # Probability of "mega weeks" (>15 points per GW equivalent)
        mega_threshold = 15.0 * self.n_gameweeks  # Scale for multiple GWs
        metrics['prob_mega_week'] = float(np.mean(team_totals > mega_threshold))

        # Consistency metrics
        metrics['consistency_score'] = metrics['p10'] / metrics['mean'] if metrics['mean'] > 0 else 0

        return metrics

    def compare_players(
        self,
        player1_data: Dict,
        player2_data: Dict,
        n_gameweeks: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Compare two players probabilistically for captaincy decisions.

        Args:
            player1_data: {'name': str, 'xp': float, 'position': str}
            player2_data: {'name': str, 'xp': float, 'position': str}
            n_gameweeks: Number of gameweeks to simulate

        Returns:
            Dictionary with simulation results for both players
        """
        results = {}

        for player_data in [player1_data, player2_data]:
            name = player_data['name']
            xp = player_data['xp']
            position = player_data['position']

            volatility = self._calculate_player_volatility(name, position)
            simulated_points = self._simulate_player_performance(xp, volatility, n_gameweeks)

            results[name] = simulated_points

        return results

    def analyze_captaincy_duel(
        self,
        player1_simulations: np.ndarray,
        player2_simulations: np.ndarray,
        player1_name: str,
        player2_name: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze captaincy duel between two players.

        Args:
            player1_simulations: Simulated points for player 1
            player2_simulations: Simulated points for player 2
            player1_name: Name of player 1
            player2_name: Name of player 2

        Returns:
            Detailed comparison metrics
        """
        # Since captain gets 2x points, compare doubled performances
        p1_captaincy = player1_simulations * 2
        p2_captaincy = player2_simulations * 2

        analysis = {
            player1_name: {},
            player2_name: {},
            'comparison': {}
        }

        # Individual metrics
        for name, points in [(player1_name, p1_captaincy), (player2_name, p2_captaincy)]:
            analysis[name] = {
                'mean_captaincy_points': float(np.mean(points)),
                'median_captaincy_points': float(np.median(points)),
                'prob_double_blank': float(np.mean(points == 0)),  # 0 points as captain = double blank
                'prob_mega_haul': float(np.mean(points > 30)),     # >15 points as captain = mega haul
                'var_95': float(np.percentile(points, 5)),
                'ceiling': float(np.percentile(points, 95))
            }

        # Comparison metrics
        p1_wins = np.sum(p1_captaincy > p2_captaincy)
        p2_wins = np.sum(p2_captaincy > p1_captaincy)
        total_sims = len(p1_captaincy)

        analysis['comparison'] = {
            f'{player1_name}_win_rate': p1_wins / total_sims,
            f'{player2_name}_win_rate': p2_wins / total_sims,
            'tie_rate': (total_sims - p1_wins - p2_wins) / total_sims,
            'avg_point_difference': float(np.mean(p1_captaincy - p2_captaincy)),
            'p1_volatility': float(np.std(p1_captaincy)),
            'p2_volatility': float(np.std(p2_captaincy))
        }

        return analysis


# Convenience functions for external use
def simulate_team(
    team_df: pd.DataFrame,
    n_simulations: int = 1000,
    captain_name: Optional[str] = None,
    vice_captain_name: Optional[str] = None
) -> SimulationResult:
    """
    Convenience function to run team simulation.

    Args:
        team_df: Team DataFrame
        n_simulations: Number of simulations
        captain_name: Captain player name
        vice_captain_name: Vice-captain player name

    Returns:
        SimulationResult object
    """
    engine = MonteCarloEngine(n_simulations=n_simulations)
    return engine.simulate_team_performance(
        team_df=team_df,
        captain_name=captain_name,
        vice_captain_name=vice_captain_name
    )


def compare_captaincy_options(
    player1: Dict,
    player2: Dict,
    n_simulations: int = 1000
) -> Dict:
    """
    Compare two players for captaincy decision.

    Args:
        player1: {'name': str, 'xp': float, 'position': str}
        player2: {'name': str, 'xp': float, 'position': str}
        n_simulations: Number of simulations

    Returns:
        Comparison analysis
    """
    engine = MonteCarloEngine(n_simulations=n_simulations)
    simulations = engine.compare_players(player1, player2)
    return engine.analyze_captaincy_duel(
        simulations[player1['name']],
        simulations[player2['name']],
        player1['name'],
        player2['name']
    )
