from typing import Any, Callable, Dict, List, Optional, Tuple
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

    def fetch_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Understat ve FPL statik verilerini çeker.
        """
        logger.info("🌍 Veriler çekiliyor...")
        try:
            r_us = self._retry_request(
                requests.get, 
                self.understat_url, 
                headers={'X-Requested-With': 'XMLHttpRequest'}
            )
            data_us = r_us.json()
            df_understat = pd.DataFrame(data_us.get('players', []))
            df_fixtures = pd.DataFrame(data_us.get('dates', []))
            
            cols = ['games', 'time', 'goals', 'xG', 'xA', 'shots', 'key_passes', 'xGChain', 'xGBuildup']
            for c in cols:
                if c in df_understat.columns:
                    df_understat[c] = pd.to_numeric(df_understat[c], errors='coerce')
        except Exception as exc:
            logger.error("Understat API Hatası: %s", str(exc), exc_info=True)
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

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
        df_merged['us_xG'] = pd.to_numeric(df_merged.get('us_xG', 0), errors='coerce').fillna(0)
        df_merged['us_xA'] = pd.to_numeric(df_merged.get('us_xA', 0), errors='coerce').fillna(0)
        df_merged['us_games'] = pd.to_numeric(df_merged.get('us_games', 0), errors='coerce').fillna(0)
        df_merged['minutes'] = pd.to_numeric(df_merged['minutes'], errors='coerce').fillna(0)
        df_merged['xG_per_90'] = np.where(df_merged['minutes']>0, (df_merged['us_xG']/df_merged['minutes'])*90, 0)
        df_merged['xA_per_90'] = np.where(df_merged['minutes']>0, (df_merged['us_xA']/df_merged['minutes'])*90, 0)
        
        # Form momentum hesaplama
        df_merged = self.calculate_form_momentum(df_merged)
        
        return df_merged

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

    def fetch_user_team(self, team_id: str) -> Tuple[Optional[List[int]], float]:
        logger.info("👤 Kullanıcı takımı çekiliyor: %s", team_id)
        clean_team_id = str(team_id).strip()
        if not clean_team_id.isdigit():
            logger.error("Geçersiz takım ID formatı: %s", team_id)
            return None, 0.0

        try:
            r_static = self._retry_request(
                requests.get,
                self.fpl_url,
                headers=self.headers,
                timeout=15
            )
            data_static = r_static.json()
        except Exception as exc:
            logger.error("bootstrap-static alınamadı: %s", str(exc), exc_info=True)
            data_static = {}

        try:
            entry_url = f"https://fantasy.premierleague.com/api/entry/{clean_team_id}/"
            r_entry = self._retry_request(
                requests.get,
                entry_url,
                headers=self.headers,
                timeout=15
            )
            entry_data = r_entry.json()
        except Exception as exc:
            error_msg = f"entry API Hatası (team_id: {clean_team_id}): {str(exc)}"
            logger.error(error_msg, exc_info=True)
            return None, 0.0

        events = data_static.get('events', []) if isinstance(data_static, dict) else []
        current_gw_obj = None
        gw_id = entry_data.get('current_event') or entry_data.get('last_deadline_event')
        if not gw_id and events:
            current_gw_obj = next((x for x in events if x.get('is_current')), None)
            if not current_gw_obj:
                current_gw_obj = next((x for x in events if x.get('is_next')), None)
            if not current_gw_obj:
                current_gw_obj = next((x for x in reversed(events) if x.get('finished') or x.get('is_previous')), None)
            if current_gw_obj:
                gw_id = current_gw_obj.get('id')

        if not gw_id:
            logger.error("Aktif gameweek belirlenemedi (team_id: %s)", clean_team_id)
            return None, 0.0
        
        try:
            picks_url = f"https://fantasy.premierleague.com/api/entry/{clean_team_id}/event/{gw_id}/picks/"
            r_picks = self._retry_request(
                requests.get,
                picks_url,
                headers=self.headers,
                timeout=15
            )
            
            data_picks = r_picks.json()
            picks = data_picks.get('picks') or []
            if not picks:
                logger.error(
                    "Picks listesi boş döndü (team_id: %s, gw: %s). Response: %s",
                    clean_team_id,
                    gw_id,
                    data_picks
                )
                return None, 0.0

            player_ids = [int(p['element']) for p in picks if 'element' in p]
            bank_raw = (data_picks.get('entry_history') or {}).get('bank', 0)
            bank = float(bank_raw) / 10.0 if bank_raw is not None else 0.0
            return player_ids, bank
        except Exception as exc:
            error_msg = f"fetch_user_team API Hatası (team_id: {clean_team_id}): {str(exc)}"
            logger.error(error_msg, exc_info=True)
            return None, 0.0
