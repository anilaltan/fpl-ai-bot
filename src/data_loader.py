from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
import numpy as np
import ast
import difflib
import logging
import time
import yaml
from pathlib import Path
from datetime import datetime

from src.understat_adapter import UnderstatAdapter

# Logging yapılandırması
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('error.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize DataLoader with configuration-driven constants.
        """
        self.config = self._load_config(config_path)
        cfg = self.config.get('data_loader', {})
        self.understat_url: str = cfg.get('understat_url')
        self.understat_league: str = cfg.get('understat_league', 'EPL')
        self.fpl_url: str = cfg.get('fpl_url')
        self.fpl_fixtures_url: str = cfg.get('fpl_fixtures_url')
        user_agent = cfg.get('user_agent', 'Mozilla/5.0')
        self.headers: Dict[str, str] = {
            'User-Agent': user_agent,
            'Accept': 'application/json,text/plain,*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Origin': 'https://fantasy.premierleague.com',
            'Referer': 'https://fantasy.premierleague.com/'
        }
        self.max_retries: int = int(cfg.get('max_retries', 3))
        self.retry_backoff_base: int = int(cfg.get('retry_backoff_base', 2))
        self.understat_seasons: List[int] = [
            int(s) for s in cfg.get('understat_seasons', []) if str(s).isdigit()
        ]
        self.request_timeout: int = int(cfg.get('request_timeout', 15))

        # Cached matchup / conceded stats (computed from element-summary history)
        self._teams_cache: Optional[Dict[int, str]] = None
        self.team_conceded_avg_by_pos: Dict[str, Dict[str, float]] = {}
        self.league_avg_pts_by_pos: Dict[str, float] = {}
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load YAML configuration.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config.yaml'
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError as exc:
            logger.error("Config file not found: %s", config_path)
            raise exc
        except yaml.YAMLError as exc:
            logger.error("Invalid YAML in config file: %s", exc)
            raise exc

    def _force_numeric(
        self,
        df: pd.DataFrame,
        columns: List[str],
        fill_value: float = 0.0
    ) -> pd.DataFrame:
        """
        Belirtilen sütunları zorla float'a çevirir; hatalı değerleri NaN -> fill_value yapar.
        """
        df = df.copy()
        cols = [c for c in columns if c in df.columns]
        if not cols:
            return df

        df[cols] = (
            df[cols]
            .apply(pd.to_numeric, errors='coerce')
            .astype(float)
            .fillna(fill_value)
        )
        return df

    def _get_understat_season_candidates(self, season: Optional[int] = None) -> List[int]:
        """
        Build a prioritized list of Understat seasons to try when fetching player data.
        """
        candidates: List[int] = []

        if season:
            candidates.append(int(season))

        if getattr(self, "understat_seasons", None):
            candidates.extend(self.understat_seasons)

        try:
            tail = str(self.understat_url).rstrip('/').split('/')[-1]
            if tail.isdigit():
                candidates.append(int(tail))
        except Exception:
            # Derivation failure is non-critical; fall back to defaults
            pass

        current_year = datetime.utcnow().year
        candidates.extend([current_year, current_year - 1])

        # Deduplicate while preserving order
        uniq: List[int] = []
        seen = set()
        for c in candidates:
            if c > 0 and c not in seen:
                uniq.append(c)
                seen.add(c)
        return uniq
    
    def _retry_request(self, func: Callable[..., requests.Response], *args: Any, **kwargs: Any) -> requests.Response:
        """
        Exponential backoff ile API isteği yapar.
        Config'teki max_retries ve retry_backoff_base kullanılır.
        """
        last_exception: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                kwargs.setdefault('timeout', self.request_timeout)
                response = func(*args, **kwargs)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as exc:
                last_exception = exc
                wait_time = self.retry_backoff_base ** attempt
                if attempt < self.max_retries - 1:
                    logger.warning(
                        "API isteği başarısız (Deneme %s/%s). %s sn bekleniyor... Hata: %s",
                        attempt + 1,
                        self.max_retries,
                        wait_time,
                        str(exc)
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        "API isteği %s denemeden sonra başarısız oldu. Son hata: %s",
                        self.max_retries,
                        str(exc)
                    )
            except Exception as exc:  # pragma: no cover - unforeseen errors
                last_exception = exc
                logger.error("Beklenmeyen hata: %s", str(exc))
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_backoff_base ** attempt
                    time.sleep(wait_time)
        
        if last_exception:
            raise last_exception
        raise RuntimeError("Retry loop exited without response or exception.")

    def _get_fpl_teams_dict(self) -> Dict[int, str]:
        """
        Fetch and cache FPL teams dict: {team_id: team_name}.
        """
        if isinstance(getattr(self, "_teams_cache", None), dict) and self._teams_cache:
            return self._teams_cache
        try:
            r_fpl = self._retry_request(requests.get, self.fpl_url, headers=self.headers)
            data_fpl = r_fpl.json() or {}
            teams = {
                int(t["id"]): str(t["name"])
                for t in (data_fpl.get("teams") or [])
                if isinstance(t, dict) and "id" in t and "name" in t
            }
            self._teams_cache = teams
            return teams
        except Exception:
            self._teams_cache = {}
            return {}

    def _fetch_understat_players(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Understat verisini içeri adapte edilmiş lightweight client ile çeker.
        Önce JSON endpoint, sonra HTML fallback. İlk başarılı sezonda durur.
        """
        adapter = UnderstatAdapter(
            max_retries=self.max_retries,
            backoff_base=self.retry_backoff_base,
            timeout=self.request_timeout,
            user_agent=self.headers.get('User-Agent', 'Mozilla/5.0')
        )

        seasons = self._get_understat_season_candidates()

        for season_candidate in seasons:
            players, fixtures = adapter.fetch_league(self.understat_league, season_candidate)
            if players:
                df_players = pd.DataFrame(players)
                df_players['season'] = season_candidate
                df_fixtures = pd.DataFrame(fixtures)
                logger.info("Understat verisi çekildi (sezon: %s)", season_candidate)
                return df_players, df_fixtures

        return pd.DataFrame(), pd.DataFrame()

    def fetch_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Understat ve FPL statik verilerini çeker.
        """
        logger.info("🌍 Veriler çekiliyor...")

        # Önce adapte edilmiş client ile dene, sonra legacy endpoint'e düş
        df_understat, df_fixtures = self._fetch_understat_players()

        if df_understat.empty:
            try:
                r_us = self._retry_request(
                    requests.get,
                    self.understat_url,
                    headers={'X-Requested-With': 'XMLHttpRequest'}
                )
                data_us = r_us.json()
                df_understat = pd.DataFrame(data_us.get('players', []))
                df_fixtures = pd.DataFrame(data_us.get('dates', []))
            except Exception as exc:
                logger.error("Understat API Hatası: %s", str(exc), exc_info=True)
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Numerik kolonları normalize et
        if not df_understat.empty:
            cols = ['games', 'time', 'goals', 'xG', 'xA', 'shots', 'key_passes', 'xGChain', 'xGBuildup']
            for c in cols:
                if c in df_understat.columns:
                    df_understat[c] = pd.to_numeric(df_understat[c], errors='coerce')

        try:
            r_fpl = self._retry_request(requests.get, self.fpl_url, headers=self.headers)
            data_fpl = r_fpl.json()
            df_fpl = pd.DataFrame(data_fpl.get('elements', []))
            
            teams = {t['id']: t['name'] for t in data_fpl.get('teams', [])}
            element_types = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            
            df_fpl['team_name'] = df_fpl['team'].map(teams)
            df_fpl['position'] = df_fpl['element_type'].map(element_types)
            df_fpl['price'] = df_fpl['now_cost'] / 10.0
            
        except Exception as exc:
            logger.error("FPL API Hatası: %s", str(exc), exc_info=True)
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        return df_understat, df_fpl, df_fixtures

    def merge_data(self, df_understat: pd.DataFrame, df_fpl: pd.DataFrame) -> pd.DataFrame:
        """
        FPL ve Understat verilerini isim benzerliği ile birleştirir.
        """
        if df_understat.empty or df_fpl.empty:
            return pd.DataFrame()

        logger.info("🤝 Veriler birleştiriliyor...")
        matched_rows: List[Dict[str, Any]] = []
        team_map = {
            'Arsenal': 'Arsenal', 'Aston Villa': 'Aston Villa', 'Bournemouth': 'Bournemouth',
            'Brentford': 'Brentford', 'Brighton': 'Brighton', 'Burnley': 'Burnley',
            'Chelsea': 'Chelsea', 'Crystal Palace': 'Crystal Palace', 'Everton': 'Everton',
            'Fulham': 'Fulham', 'Leeds': 'Leeds', 'Liverpool': 'Liverpool',
            'Man City': 'Manchester City', 'Man Utd': 'Manchester United', 'Newcastle': 'Newcastle United',
            "Nott'm Forest": 'Nottingham Forest', 'Spurs': 'Tottenham', 'Sunderland': 'Sunderland',
            'West Ham': 'West Ham', 'Wolves': 'Wolverhampton Wanderers'
        }

        for idx, fpl_row in df_fpl.iterrows():
            fpl_name = fpl_row.get('web_name', '')
            full_name = fpl_row.get('first_name', '') + " " + fpl_row.get('second_name', '')
            fpl_team = fpl_row.get('team_name', '')
            us_target = team_map.get(fpl_team, fpl_team)
            
            candidates = df_understat[df_understat['team_title'] == us_target]
            
            best_score = 0
            match_data = None
            
            if not candidates.empty:
                for _, us_row in candidates.iterrows():
                    score = difflib.SequenceMatcher(None, full_name.lower(), us_row['player_name'].lower()).ratio() * 100
                    if score > best_score:
                        best_score = score
                        match_data = us_row.to_dict()
            
            res = fpl_row.to_dict()
            if best_score > 60 and match_data:
                for k, v in match_data.items():
                    res[f'us_{k}'] = v
                # Understat oyuncu ID'sini sakla (form momentum için gerekli)
                if 'id' in match_data:
                    res['us_player_id'] = match_data['id']
            matched_rows.append(res)
            
        df_merged = pd.DataFrame(matched_rows)

        # Sakatlık ve risk yönetimi kolonlarının mevcut olup olmadığını kontrol et
        availability_cols = ['chance_of_playing_next_round', 'chance_of_playing_this_round', 'news', 'status']
        missing_availability_cols = [col for col in availability_cols if col not in df_merged.columns]
        if missing_availability_cols:
            logger.warning(f"FPL API'den eksik availability kolonları: {missing_availability_cols}")
        else:
            logger.info("✅ Availability kolonları FPL verisinde mevcut")

        df_merged['us_xG'] = pd.to_numeric(df_merged.get('us_xG', 0), errors='coerce').fillna(0)
        df_merged['us_xA'] = pd.to_numeric(df_merged.get('us_xA', 0), errors='coerce').fillna(0)
        df_merged['us_games'] = pd.to_numeric(df_merged.get('us_games', 0), errors='coerce').fillna(0)
        df_merged['minutes'] = pd.to_numeric(df_merged['minutes'], errors='coerce').fillna(0)
        df_merged['xG_per_90'] = np.where(df_merged['minutes']>0, (df_merged['us_xG']/df_merged['minutes'])*90, 0)
        df_merged['xA_per_90'] = np.where(df_merged['minutes']>0, (df_merged['us_xA']/df_merged['minutes'])*90, 0)
        
        # Form momentum hesaplama
        df_merged = self.calculate_form_momentum(df_merged)

        # Sayısal kolonları object/string kalmasın diye float'a zorla
        numeric_cols = [
            'price', 'form', 'ict_index', 'influence', 'creativity', 'threat',
            'value_form', 'goals_scored', 'assists', 'clean_sheets', 'saves',
            'goals_conceded', 'goals_conceded_per_90', 'expected_goals_conceded_per_90',
            'expected_goals_per_90', 'expected_assists_per_90',
            'expected_goal_involvements_per_90', 'xG_per_90', 'xA_per_90',
            'us_xGChain', 'us_xGBuildup', 'us_shots', 'us_key_passes', 'us_xG',
            'us_xA', 'us_games', 'minutes', 'total_points', 'saves_per_90',
            'clean_sheets_per_90',
            # Sakatlık ve risk yönetimi için gerekli kolonlar
            'chance_of_playing_next_round', 'chance_of_playing_this_round'
        ]
        df_merged = self._force_numeric(df_merged, numeric_cols, fill_value=0.0)

        return df_merged

    def enrich_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature engineering enrichment step(s).

        Adds:
        - set_piece_threat: Universal set piece / dead-ball threat proxy.
          Uses aerial threat (ICT threat per 90) + set-piece taker bonuses.
        - matchup_advantage: positional matchup bonus based on opponent conceding points
          to that position vs league average. Defaults to 1.0 when unavailable.
        """
        if df is None or df.empty:
            return df

        df = df.copy()

        # Initialize columns
        df["set_piece_threat"] = 0.0
        df["matchup_advantage"] = 1.0

        # Helpers
        def _series_or_zeros(col: str) -> pd.Series:
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
            return pd.Series(0.0, index=df.index, dtype=float)

        minutes = _series_or_zeros("minutes")
        threat = _series_or_zeros("threat")

        # Universal: Aerial threat via ICT threat per 90
        base_threat_p90 = np.where(minutes > 0, (threat / minutes) * 90.0, 0.0)
        base_threat_p90 = pd.Series(base_threat_p90, index=df.index, dtype=float)

        # Determine defender mask for weighting
        if "position" in df.columns:
            pos = df["position"].astype(str).str.upper()
            is_def = pos.eq("DEF")
        elif "element_type" in df.columns:
            et = pd.to_numeric(df["element_type"], errors="coerce").fillna(-1).astype(int)
            is_def = et.eq(2)  # FPL: 2=DEF
        else:
            is_def = pd.Series(False, index=df.index)

        aerial_component = base_threat_p90 * np.where(is_def, 0.2, 0.1)
        aerial_component = pd.Series(aerial_component, index=df.index, dtype=float)

        # Set-piece taker bonus (order columns can be float/string)
        corners_order = pd.to_numeric(
            df["corners_and_indirect_freekicks_order"], errors="coerce"
        ) if "corners_and_indirect_freekicks_order" in df.columns else pd.Series(np.nan, index=df.index)
        direct_fk_order = pd.to_numeric(
            df["direct_freekicks_order"], errors="coerce"
        ) if "direct_freekicks_order" in df.columns else pd.Series(np.nan, index=df.index)
        pens_order = pd.to_numeric(
            df["penalties_order"], errors="coerce"
        ) if "penalties_order" in df.columns else pd.Series(np.nan, index=df.index)

        bonus = pd.Series(0.0, index=df.index, dtype=float)

        # Penalties: 1 -> +3.0, 2 -> +1.0
        bonus += np.where(np.isclose(pens_order, 1.0, equal_nan=False), 3.0, 0.0)
        bonus += np.where(np.isclose(pens_order, 2.0, equal_nan=False), 1.0, 0.0)

        # Direct free kicks: 1 -> +1.5, 2 -> +0.5
        bonus += np.where(np.isclose(direct_fk_order, 1.0, equal_nan=False), 1.5, 0.0)
        bonus += np.where(np.isclose(direct_fk_order, 2.0, equal_nan=False), 0.5, 0.0)

        # Corners/indirect: 1 -> +1.0, 2 -> +0.25
        bonus += np.where(np.isclose(corners_order, 1.0, equal_nan=False), 1.0, 0.0)
        bonus += np.where(np.isclose(corners_order, 2.0, equal_nan=False), 0.25, 0.0)

        set_piece_threat = (aerial_component + bonus).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df["set_piece_threat"] = pd.to_numeric(set_piece_threat, errors="coerce").fillna(0.0).astype(float)

        # Debug print: Top 10 players by set_piece_threat
        try:
            name_cols = [c for c in ["web_name", "team_name", "position"] if c in df.columns]
            debug_df = df[name_cols].copy() if name_cols else df.copy()
            debug_df["set_piece_threat"] = df["set_piece_threat"].astype(float)
            debug_df = debug_df.sort_values("set_piece_threat", ascending=False).head(10)
            print("\n[DEBUG] Top 10 players by set_piece_threat:")
            print(debug_df.to_string(index=False, float_format=lambda x: f"{float(x):.4f}"))
        except Exception as exc:
            logger.debug("set_piece_threat debug print failed: %s", exc)

        # Matchup advantage (requires opponent_team and position)
        try:
            if "opponent_team" in df.columns and "position" in df.columns:
                league_avg = dict(getattr(self, "league_avg_pts_by_pos", {}) or {})
                conceded_avg = dict(getattr(self, "team_conceded_avg_by_pos", {}) or {})

                pos_series = df["position"].astype(str).str.upper()
                opp_series = df["opponent_team"].astype(str)

                def _baseline(pos: str) -> float:
                    v = float(league_avg.get(pos, 0.0) or 0.0)
                    return v if v > 0 else 0.0

                matchup_vals: List[float] = []
                for pos, opp in zip(pos_series.tolist(), opp_series.tolist()):
                    base = _baseline(pos)
                    if not opp or opp == "Unknown" or base <= 0:
                        matchup_vals.append(1.0)
                        continue
                    conceded = float((conceded_avg.get(opp, {}) or {}).get(pos, 0.0) or 0.0)
                    if conceded <= 0:
                        matchup_vals.append(1.0)
                        continue
                    matchup_vals.append(conceded / base if base > 0 else 1.0)

                df["matchup_advantage"] = (
                    pd.Series(matchup_vals, index=df.index, dtype=float)
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(1.0)
                    .astype(float)
                )

                # Debug print: Top 5 players by matchup_advantage
                debug_cols = [c for c in ["web_name", "team_name", "position", "opponent_team", "matchup_advantage"] if c in df.columns]
                if debug_cols:
                    top5 = df.sort_values("matchup_advantage", ascending=False).head(5)[debug_cols]
                    print("\n[DEBUG] Top 5 players by matchup_advantage:")
                    print(top5.to_string(index=False, float_format=lambda x: f"{float(x):.4f}"))
        except Exception as exc:
            logger.debug("matchup_advantage computation failed: %s", exc)

        # Model B: The Bookie - Betting Odds Integration
        try:
            logger.info("🎲 Adding betting odds features (Model B: The Bookie)")

            # Initialize odds columns
            df["implied_goal_prob"] = 0.0
            df["implied_cs_prob"] = 0.0

            # Check if real odds exist in dataset
            has_real_goal_odds = "odds_goal" in df.columns
            has_real_cs_odds = "odds_cs" in df.columns

            if has_real_goal_odds or has_real_cs_odds:
                logger.info("📊 Using real betting odds from dataset")
                if has_real_goal_odds:
                    df["implied_goal_prob"] = pd.to_numeric(df["odds_goal"], errors="coerce").fillna(0.0)
                if has_real_cs_odds:
                    df["implied_cs_prob"] = pd.to_numeric(df["odds_cs"], errors="coerce").fillna(0.0)
            else:
                logger.info("🎯 Generating synthetic odds (no real odds found)")

                # Synthetic odds generation
                # implied_goal_prob = (1 / FDR) * (Player Form) * 0.5
                # implied_cs_prob = (1 / Opponent FDR) * (Team Defense Strength)

                # Get opponent FDR (we'll use a simplified approach)
                # For now, use opponent_team to get a basic difficulty rating
                base_goal_prob = 0.15  # Base probability for scoring
                base_cs_prob = 0.35    # Base probability for clean sheet

                # Player form factors
                form = _series_or_zeros("form") / 10.0  # Normalize form (0-10 scale)
                minutes_factor = np.where(minutes > 0, np.minimum(minutes / 1800, 1.0), 0.0)  # Minutes played factor

                # Position-based adjustments
                if "position" in df.columns:
                    pos_factor = np.where(
                        df["position"].str.upper() == "FWD", 1.5,  # Forwards more likely to score
                        np.where(df["position"].str.upper() == "MID", 1.2,  # Mids moderate
                        np.where(df["position"].str.upper() == "DEF", 0.8, 1.0))  # Defs less likely
                    )
                else:
                    pos_factor = pd.Series(1.0, index=df.index)

                # Calculate implied goal probability
                player_form_factor = (form * 0.3) + (minutes_factor * 0.4) + (pos_factor * 0.3)
                df["implied_goal_prob"] = base_goal_prob * player_form_factor

                # Calculate implied clean sheet probability
                # Use a simple team-based approach (would be better with actual FDR data)
                if "opponent_difficulty" in df.columns:
                    opp_difficulty = pd.to_numeric(df["opponent_difficulty"], errors="coerce").fillna(3.0)
                    # Lower FDR (easier opponent) = higher CS probability
                    cs_factor = 1.0 / opp_difficulty
                else:
                    cs_factor = pd.Series(1.0, index=df.index)

                team_defense_factor = np.where(
                    df["position"].str.upper() == "GK", 2.0,  # Goalkeepers boost CS prob
                    np.where(df["position"].str.upper() == "DEF", 1.5, 1.0)  # Defenders help CS
                ) if "position" in df.columns else 1.0

                df["implied_cs_prob"] = base_cs_prob * cs_factor * team_defense_factor

            # Normalize probabilities to 0-1 range
            df["implied_goal_prob"] = np.clip(df["implied_goal_prob"], 0.0, 1.0)
            df["implied_cs_prob"] = np.clip(df["implied_cs_prob"], 0.0, 1.0)

            logger.info(f"✅ Odds features added for {len(df)} players")

        except Exception as exc:
            logger.warning("Betting odds feature generation failed: %s", exc)
            # Set defaults
            df["implied_goal_prob"] = 0.15
            df["implied_cs_prob"] = 0.35

        return df

    def process_fixtures(self, df_fixtures: pd.DataFrame) -> pd.DataFrame:
        """
        Understat fikstür verisini xG ve takım adları ile zenginleştirir.
        """
        if df_fixtures.empty:
            return df_fixtures
        
        logger.info("🛠️ Fikstürler işleniyor...")
        def parse_val(val: Any, key: str) -> Optional[Any]:
            if isinstance(val, dict):
                return val.get(key)
            if isinstance(val, str):
                try: 
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, dict):
                        return parsed.get(key)
                except Exception:
                    return None
            return None

        df_fixtures['xG_h'] = df_fixtures['xG'].apply(lambda x: float(parse_val(x, 'h') or 0))
        df_fixtures['xG_a'] = df_fixtures['xG'].apply(lambda x: float(parse_val(x, 'a') or 0))
        
        if 'h' in df_fixtures.columns:
            df_fixtures['home_team'] = df_fixtures['h'].apply(lambda x: parse_val(x, 'title'))
        if 'a' in df_fixtures.columns:
            df_fixtures['away_team'] = df_fixtures['a'].apply(lambda x: parse_val(x, 'title'))
        
        return df_fixtures

    def get_next_gw(self) -> Dict[str, Any]:
        """Sıradaki Gameweek bilgisini çeker"""
        try:
            r = self._retry_request(requests.get, self.fpl_url, headers=self.headers)
            data = r.json()
            next_event = next((e for e in data['events'] if e['is_next']), None)
            if next_event:
                return {
                    'id': next_event['id'],
                    'name': next_event['name'],
                    'deadline': next_event['deadline_time']
                }
            return {'id': 0, 'name': 'Unknown GW', 'deadline': '-'}
        except Exception as exc:
            error_msg = f"get_next_gw API Hatası: {str(exc)}"
            logger.error(error_msg, exc_info=True)
            return {'id': 0, 'name': 'Unknown GW', 'deadline': '-'}
    
    def fetch_fpl_fixtures(self) -> List[Dict[str, Any]]:
        """
        FPL API'sinden fixtures verilerini çeker.
        Retry mekanizması ile güvenli API çağrısı yapar.
        """
        logger.info("📅 FPL fixtures verileri çekiliyor...")
        try:
            r = self._retry_request(requests.get, self.fpl_fixtures_url, headers=self.headers)
            fixtures_data = r.json()
            return fixtures_data
        except Exception as exc:
            error_msg = f"FPL Fixtures API Hatası: {str(exc)}"
            logger.error(error_msg, exc_info=True)
            return []
    
    def fetch_player_history_batch(
        self,
        df_players: pd.DataFrame,
        max_workers: int = 16
    ) -> pd.DataFrame:
        """
        FPL element-summary endpoint'inden son 5 GW'deki dakika bilgilerini
        paralel olarak çeker ve minutes_last_five sütunu olarak ekler.
        """
        if df_players.empty or 'id' not in df_players.columns:
            return df_players

        df_players = df_players.copy()
        player_ids = [int(pid) for pid in df_players['id'].dropna().unique()]
        if not player_ids:
            df_players['minutes_last_five'] = [[] for _ in range(len(df_players))]
            return df_players

        # Position map for conceded stats aggregation (opponent weakness by position)
        pid_to_pos: Dict[int, str] = {}
        if "position" in df_players.columns:
            try:
                for pid, pos in zip(df_players["id"].tolist(), df_players["position"].tolist()):
                    if pd.notna(pid) and pd.notna(pos):
                        pid_to_pos[int(pid)] = str(pos).upper()
            except Exception:
                pid_to_pos = {}

        # Team id -> name mapping for opponent_team ids inside element-summary history
        teams_dict = self._get_fpl_teams_dict()

        base_url = str(self.fpl_url or "").strip()
        if 'bootstrap-static' in base_url:
            base_url = base_url.split('bootstrap-static')[0]
        if not base_url.endswith('/'):
            base_url += '/'

        def _fetch_minutes(pid: int) -> Tuple[int, List[int], Dict[str, Dict[str, Dict[str, float]]]]:
            url = f"{base_url}element-summary/{pid}/"
            try:
                response = self._retry_request(
                    requests.get,
                    url,
                    headers=self.headers,
                    timeout=self.request_timeout
                )
                payload = response.json() or {}
                history = payload.get('history') or []
                last_five = [int(item.get('minutes', 0) or 0) for item in history[-5:]]
                # Build per-player contributions to "opponent conceded points by position"
                # Structure: {opponent_team_name: {position: {'pts': sum, 'matches': count}}}
                pos = pid_to_pos.get(int(pid), "UNK")
                contrib: Dict[str, Dict[str, Dict[str, float]]] = {}
                for item in history:
                    try:
                        mins = float(item.get("minutes", 0) or 0)
                        if mins <= 0:
                            continue
                        opp_id = int(item.get("opponent_team", 0) or 0)
                        if opp_id <= 0:
                            continue
                        opp_name = teams_dict.get(opp_id)
                        if not opp_name:
                            continue
                        pts = float(item.get("total_points", 0) or 0)
                        bucket = contrib.setdefault(opp_name, {}).setdefault(pos, {"pts": 0.0, "matches": 0.0})
                        bucket["pts"] += pts
                        bucket["matches"] += 1.0
                    except Exception:
                        continue
                return pid, last_five, contrib
            except Exception as exc:
                logger.warning("Dakika geçmişi çekilemedi (id: %s): %s", pid, str(exc))
                return pid, [], {}

        worker_count = min(max_workers, max(1, len(player_ids)))
        results: Dict[int, List[int]] = {}
        conceded_totals: Dict[str, Dict[str, Dict[str, float]]] = {}
        league_totals: Dict[str, Dict[str, float]] = {}

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {executor.submit(_fetch_minutes, pid): pid for pid in player_ids}
            for future in as_completed(future_map):
                pid, minutes_list, contrib = future.result()
                results[pid] = minutes_list

                # Aggregate to global totals
                for opp_team, pos_map in (contrib or {}).items():
                    for pos, stats in (pos_map or {}).items():
                        tot = conceded_totals.setdefault(opp_team, {}).setdefault(pos, {"pts": 0.0, "matches": 0.0})
                        tot["pts"] += float(stats.get("pts", 0.0) or 0.0)
                        tot["matches"] += float(stats.get("matches", 0.0) or 0.0)

                        lt = league_totals.setdefault(pos, {"pts": 0.0, "matches": 0.0})
                        lt["pts"] += float(stats.get("pts", 0.0) or 0.0)
                        lt["matches"] += float(stats.get("matches", 0.0) or 0.0)

        df_players['minutes_last_five'] = [
            results.get(int(pid), []) if pd.notna(pid) else []
            for pid in df_players['id']
        ]

        # Cache conceded averages and league averages for matchup feature engineering
        league_avg: Dict[str, float] = {}
        for pos, st in league_totals.items():
            m = float(st.get("matches", 0.0) or 0.0)
            league_avg[pos] = (float(st.get("pts", 0.0) or 0.0) / m) if m > 0 else 0.0

        conceded_avg: Dict[str, Dict[str, float]] = {}
        for opp_team, pos_map in conceded_totals.items():
            conceded_avg[opp_team] = {}
            for pos, st in pos_map.items():
                m = float(st.get("matches", 0.0) or 0.0)
                conceded_avg[opp_team][pos] = (float(st.get("pts", 0.0) or 0.0) / m) if m > 0 else 0.0

        self.team_conceded_avg_by_pos = conceded_avg
        self.league_avg_pts_by_pos = league_avg

        return df_players
    
    def get_fpl_data(self, df_players: pd.DataFrame) -> pd.DataFrame:
        """
        Oyuncu verilerine fixture bilgilerini ekler.
        Her oyuncunun bir sonraki maçını bulur ve şu sütunları ekler:
        - is_home: (Boolean) Eğer maç iç sahadaysa 1, değilse 0
        - opponent_difficulty: (Int) Rakip takımın zorluk derecesi (FDR)
        - opponent_team: (String) Rakip takımın adı
        
        Args:
            df_players: Oyuncu verilerini içeren DataFrame (team sütunu içermeli)
        
        Returns:
            DataFrame: Fixture bilgileriyle zenginleştirilmiş oyuncu verileri
        """
        if df_players.empty:
            return df_players
        
        logger.info("🔍 Oyuncuların bir sonraki maçları bulunuyor...")
        
        fixtures_data = self.fetch_fpl_fixtures()
        if not fixtures_data:
            logger.warning("Fixtures verisi çekilemedi, varsayılan değerler kullanılıyor.")
            df_players['is_home'] = 0
            df_players['opponent_difficulty'] = 3
            df_players['opponent_team'] = 'Unknown'
            return df_players
        
        # Bootstrap-static'ten takım bilgilerini ve gameweek bilgisini çek
        try:
            r_static = self._retry_request(requests.get, self.fpl_url, headers=self.headers)
            static_data = r_static.json()
            teams_dict = {t['id']: t['name'] for t in static_data['teams']}
            
            next_event = next((e for e in static_data['events'] if e.get('is_next', False)), None)
            if not next_event:
                next_event = next((e for e in static_data['events'] if not e.get('finished', True)), None)
            
            next_gw_id = next_event['id'] if next_event else None
        except Exception as exc:
            logger.warning(f"Bootstrap-static verisi çekilemedi: {str(exc)}")
            teams_dict = {}
            next_gw_id = None
        
        # Fixtures'ı DataFrame'e çevir
        df_fixtures = pd.DataFrame(fixtures_data)
        
        if df_fixtures.empty:
            logger.warning("Fixtures DataFrame boş, varsayılan değerler kullanılıyor.")
            df_players['is_home'] = 0
            df_players['opponent_difficulty'] = 3
            df_players['opponent_team'] = 'Unknown'
            return df_players
        
        # Bir sonraki gameweek'e ait fixtures'ı filtrele
        if next_gw_id is not None:
            df_next_fixtures = df_fixtures[df_fixtures['event'] == next_gw_id].copy()
        else:
            # Eğer next_gw_id yoksa, finished=False olan fixtures'ları al
            if 'finished' in df_fixtures.columns:
                df_next_fixtures = df_fixtures[df_fixtures['finished'] == False].copy()
            else:
                df_next_fixtures = pd.DataFrame()
            
            if df_next_fixtures.empty:
                # Hiçbiri yoksa, event sütununa göre en küçük event'i al
                if 'event' in df_fixtures.columns:
                    min_event = df_fixtures['event'].min()
                    df_next_fixtures = df_fixtures[df_fixtures['event'] == min_event].copy()
        
        if df_next_fixtures.empty:
            logger.warning("Bir sonraki gameweek için fixture bulunamadı, varsayılan değerler kullanılıyor.")
            df_players['is_home'] = 0
            df_players['opponent_difficulty'] = 3
            df_players['opponent_team'] = 'Unknown'
            return df_players
        
        # Her oyuncu için bir sonraki maçı bul
        is_home_list = []
        opponent_difficulty_list = []
        opponent_team_list = []
        
        for idx, player_row in df_players.iterrows():
            player_team_id = player_row.get('team')
            
            if pd.isna(player_team_id) or player_team_id is None:
                # Takım bilgisi yoksa varsayılan değerler
                is_home_list.append(0)
                opponent_difficulty_list.append(3)
                opponent_team_list.append('Unknown')
                continue
            
            # Bu takımın bir sonraki maçını bul
            player_fixture = None
            
            # Ev sahibi olarak oynayacak mı?
            home_fixtures = df_next_fixtures[df_next_fixtures['team_h'] == player_team_id]
            if not home_fixtures.empty:
                player_fixture = home_fixtures.iloc[0]
                is_home = 1
                opponent_team_id = player_fixture.get('team_a')
                opponent_difficulty = player_fixture.get('team_h_difficulty', 3)
            else:
                # Deplasman olarak oynayacak mı?
                away_fixtures = df_next_fixtures[df_next_fixtures['team_a'] == player_team_id]
                if not away_fixtures.empty:
                    player_fixture = away_fixtures.iloc[0]
                    is_home = 0
                    opponent_team_id = player_fixture.get('team_h')
                    opponent_difficulty = player_fixture.get('team_a_difficulty', 3)
                else:
                    # Maç bulunamadı
                    is_home = 0
                    opponent_team_id = None
                    opponent_difficulty = 3
            
            # Rakip takım adını bul
            if opponent_team_id is not None and opponent_team_id in teams_dict:
                opponent_team_name = teams_dict[opponent_team_id]
            else:
                opponent_team_name = 'Unknown'
            
            is_home_list.append(is_home)
            opponent_difficulty_list.append(int(opponent_difficulty) if not pd.isna(opponent_difficulty) else 3)
            opponent_team_list.append(opponent_team_name)
        
        # Sütunları ekle
        df_players['is_home'] = is_home_list
        df_players['opponent_difficulty'] = opponent_difficulty_list
        df_players['opponent_team'] = opponent_team_list

        # Feature engineering that depends on opponent_team (e.g., matchup_advantage)
        df_players = self.enrich_data(df_players)
        
        print(f"✅ {len(df_players)} oyuncu için fixture bilgileri eklendi.")
        
        return df_players
 
    def fetch_player_match_history(self, player_id: int, season: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Understat API'den oyuncu bazlı maç geçmişini çeker.
        Sezon bilgisi dinamik olarak belirlenir ve mevcut değilse önceki sezonlara düşer.
        """
        season_candidates = self._get_understat_season_candidates(season)
        headers = {'X-Requested-With': 'XMLHttpRequest'}

        for season_candidate in season_candidates:
            player_api_url = f"https://understat.com/getPlayerData/?player_id={player_id}&season={season_candidate}"
            last_exception: Optional[Exception] = None

            for attempt in range(self.max_retries):
                try:
                    response = requests.get(player_api_url, headers=headers, timeout=15)

                    if response.status_code == 404:
                        logger.info(
                            "Understat oyuncu verisi bulunamadı (id: %s, sezon: %s). Alternatif sezon deneniyor.",
                            player_id,
                            season_candidate
                        )
                        break  # Bir sonraki sezon adayına geç

                    response.raise_for_status()
                    data = response.json()

                    if isinstance(data, dict) and 'matches' in data:
                        return data['matches']
                    if isinstance(data, list) and data:
                        return data

                    logger.debug(
                        "Oyuncu maç geçmişi beklenen formatta değil (id: %s, sezon: %s). Sonraki sezona geçiliyor.",
                        player_id,
                        season_candidate
                    )
                    break  # Bu sezon için başka deneme yapma, sonraki sezona geç
                except requests.exceptions.RequestException as exc:
                    last_exception = exc
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_backoff_base ** attempt
                        logger.debug(
                            "Oyuncu verisi çekilemedi (id: %s, sezon: %s, deneme: %s/%s): %s",
                            player_id,
                            season_candidate,
                            attempt + 1,
                            self.max_retries,
                            str(exc)
                        )
                        time.sleep(wait_time)
                        continue
                    logger.warning(
                        "Oyuncu maç geçmişi çekilemedi (id: %s, sezon: %s). Hata: %s",
                        player_id,
                        season_candidate,
                        str(exc)
                    )

            if last_exception is None:
                # 404 veya beklenmedik format nedeniyle sonraki sezon denenecek
                continue

        logger.debug(
            "Oyuncu maç geçmişi bulunamadı (player_id: %s, sezon adayları: %s)",
            player_id,
            season_candidates
        )
        return []
    
    def calculate_form_momentum(self, df_merged: pd.DataFrame) -> pd.DataFrame:
        """
        Son 3 maçtaki xG ortalaması ile son 10 maçtaki xG ortalamasını karşılaştırarak
        form_momentum sütunu oluşturur.
        Gerçek maç bazlı veriler için Understat API'den oyuncu bazlı maç geçmişi çekilir.
        """
        logger.info("📊 Form momentum hesaplanıyor...")
        
        form_momentum_values = []
        player_match_cache = {}  # API çağrılarını azaltmak için cache
        
        for idx, row in df_merged.iterrows():
            try:
                # Understat oyuncu ID'sini al
                us_player_id = row.get('us_player_id') or row.get('us_id')
                
                # Eğer oyuncu ID yoksa, fallback hesaplama yap
                if not us_player_id or pd.isna(us_player_id):
                    # Fallback: Mevcut verilerden tahmin
                    us_xG = pd.to_numeric(row.get('us_xG', 0), errors='coerce') or 0
                    games = pd.to_numeric(row.get('us_games', 0), errors='coerce') or 0
                    
                    if games < 3:
                        form_momentum_values.append(0.0)
                        continue
                    
                    avg_xG_per_game = us_xG / games if games > 0 else 0
                    if games >= 10:
                        # Basit tahmin: son 3 maç için küçük bir varyasyon
                        last_3_avg = avg_xG_per_game * 1.05
                        last_10_avg = avg_xG_per_game
                        momentum = last_3_avg - last_10_avg
                    else:
                        momentum = 0.0
                    
                    form_momentum_values.append(round(momentum, 3))
                    continue
                
                # Cache'den kontrol et
                if us_player_id not in player_match_cache:
                    # API'den maç geçmişini çek
                    matches = self.fetch_player_match_history(int(us_player_id))
                    player_match_cache[us_player_id] = matches
                else:
                    matches = player_match_cache[us_player_id]
                
                if not matches or len(matches) < 3:
                    # Yeterli maç verisi yoksa fallback kullan
                    us_xG = pd.to_numeric(row.get('us_xG', 0), errors='coerce') or 0
                    games = pd.to_numeric(row.get('us_games', 0), errors='coerce') or 0
                    
                    if games < 3:
                        form_momentum_values.append(0.0)
                        continue
                    
                    avg_xG_per_game = us_xG / games if games > 0 else 0
                    if games >= 10:
                        last_3_avg = avg_xG_per_game * 1.05
                        last_10_avg = avg_xG_per_game
                        momentum = last_3_avg - last_10_avg
                    else:
                        momentum = 0.0
                    
                    form_momentum_values.append(round(momentum, 3))
                    continue
                
                # Maç bazlı xG verilerini işle
                match_xG_list = []
                for match in matches:
                    # xG değerini al (farklı formatlar olabilir)
                    xG = 0
                    if isinstance(match, dict):
                        xG = pd.to_numeric(match.get('xG', match.get('xG_value', 0)), errors='coerce') or 0
                    match_xG_list.append(xG)
                
                # En son maçlardan başlayarak sırala (zaten sıralı olabilir)
                match_xG_list = [x for x in match_xG_list if x > 0]  # Sadece pozitif değerleri al
                
                if len(match_xG_list) < 3:
                    form_momentum_values.append(0.0)
                    continue
                
                # Son 3 maç ortalaması
                last_3_matches = match_xG_list[-3:] if len(match_xG_list) >= 3 else match_xG_list
                last_3_avg = np.mean(last_3_matches) if last_3_matches else 0
                
                # Son 10 maç ortalaması (eğer 10 maçtan fazla varsa)
                if len(match_xG_list) >= 10:
                    last_10_matches = match_xG_list[-10:]
                    last_10_avg = np.mean(last_10_matches)
                else:
                    # 10 maçtan azsa, tüm maçların ortalamasını kullan
                    last_10_avg = np.mean(match_xG_list)
                
                # Momentum hesapla: son 3 maç ortalaması - son 10 maç ortalaması
                # Pozitif değer = form artışı, negatif değer = form düşüşü
                momentum = last_3_avg - last_10_avg
                
                form_momentum_values.append(round(momentum, 3))
                
                # API rate limiting için küçük bir bekleme
                if idx % 10 == 0:
                    time.sleep(0.1)
                
            except Exception as exc:
                logger.warning(f"Form momentum hesaplanamadı (index: {idx}): {str(exc)}")
                form_momentum_values.append(0.0)
        
        df_merged['form_momentum'] = form_momentum_values
        return df_merged

    def fetch_user_team(self, team_id: str, df_players: pd.DataFrame) -> List[str]:
        """
        Fetch user's team from previous gameweek and return player web_names.

        Args:
            team_id: FPL team ID
            df_players: DataFrame with player data to map IDs to names

        Returns:
            List of web_names for the user's team
        """
        logger.info("👤 Kullanıcı takımı çekiliyor: %s", team_id)
        clean_team_id = str(team_id).strip()
        if not clean_team_id.isdigit():
            logger.error("Geçersiz takım ID formatı: %s", team_id)
            return []

        try:
            # Get current event from bootstrap-static
            r_static = self._retry_request(
                requests.get,
                self.fpl_url,
                headers=self.headers,
                timeout=15
            )
            data_static = r_static.json()
        except Exception as exc:
            logger.error("bootstrap-static alınamadı: %s", str(exc), exc_info=True)
            return []

        # Find current event
        events = data_static.get('events', []) if isinstance(data_static, dict) else []
        current_event = None
        for event in events:
            if event.get('is_current'):
                current_event = event.get('id')
                break
        if not current_event:
            for event in events:
                if event.get('is_next'):
                    current_event = event.get('id')
                    break
        if not current_event:
            logger.error("Current event bulunamadı")
            return []

        # Use previous GW (current_event - 1)
        target_gw = current_event - 1
        if target_gw < 1:
            logger.error("Geçerli bir önceki GW bulunamadı (current_event: %s)", current_event)
            return []

        try:
            # Fetch picks from previous GW
            picks_url = f"https://fantasy.premierleague.com/api/entry/{clean_team_id}/event/{target_gw}/picks/"
            r_picks = self._retry_request(
                requests.get,
                picks_url,
                headers=self.headers,
                timeout=15
            )

            data_picks = r_picks.json()
            picks = data_picks.get('picks') or []
            if not picks:
                logger.warning(
                    "Picks listesi boş döndü (team_id: %s, gw: %s). Önceki GW'de takım olmayabilir.",
                    clean_team_id,
                    target_gw
                )
                return []

            # Get player IDs
            player_ids = [int(p['element']) for p in picks if 'element' in p]

            # Map IDs to web_names using df_players
            id_to_name = dict(zip(df_players['id'], df_players['web_name']))
            player_names = []
            for pid in player_ids:
                name = id_to_name.get(pid)
                if name:
                    player_names.append(name)
                else:
                    logger.warning("Oyuncu ID bulunamadı: %s", pid)

            logger.info("✅ %d oyuncu çekildi (Team ID: %s, GW: %s)", len(player_names), clean_team_id, target_gw)
            return player_names

        except Exception as exc:
            error_msg = f"fetch_user_team API Hatası (team_id: {clean_team_id}, gw: {target_gw}): {str(exc)}"
            logger.error(error_msg, exc_info=True)
            return []
