import pandas as pd
import requests
import numpy as np
import ast
import difflib
import logging
import time

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
    def __init__(self):
        self.understat_url = "https://understat.com/getLeagueData/EPL/2025"
        self.fpl_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        self.fpl_fixtures_url = "https://fantasy.premierleague.com/api/fixtures/"
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        self.max_retries = 3
    
    def _retry_request(self, func, *args, **kwargs):
        """
        Exponential backoff ile API isteği yapar.
        3 kez deneme yapar: 1s, 2s, 4s bekleme süreleri ile.
        """
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = func(*args, **kwargs)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                last_exception = e
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"API isteği başarısız (Deneme {attempt + 1}/{self.max_retries}). "
                        f"{wait_time} saniye bekleniyor... Hata: {str(e)}"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"API isteği {self.max_retries} denemeden sonra başarısız oldu. "
                        f"Son hata: {str(e)}"
                    )
            except Exception as e:
                last_exception = e
                logger.error(f"Beklenmeyen hata: {str(e)}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
        
        raise last_exception

    def fetch_all_data(self):
        print("🌍 Veriler çekiliyor...")
        try:
            r_us = self._retry_request(
                requests.get, 
                self.understat_url, 
                headers={'X-Requested-With': 'XMLHttpRequest'}
            )
            data_us = r_us.json()
            df_understat = pd.DataFrame(data_us['players'])
            df_fixtures = pd.DataFrame(data_us['dates'])
            
            cols = ['games', 'time', 'goals', 'xG', 'xA', 'shots', 'key_passes', 'xGChain', 'xGBuildup']
            for c in cols:
                if c in df_understat.columns:
                    df_understat[c] = pd.to_numeric(df_understat[c], errors='coerce')
        except Exception as e:
            error_msg = f"Understat API Hatası: {str(e)}"
            print(f"❌ {error_msg}")
            logger.error(error_msg, exc_info=True)
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        try:
            r_fpl = self._retry_request(requests.get, self.fpl_url)
            data_fpl = r_fpl.json()
            df_fpl = pd.DataFrame(data_fpl['elements'])
            
            teams = {t['id']: t['name'] for t in data_fpl['teams']}
            element_types = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            
            df_fpl['team_name'] = df_fpl['team'].map(teams)
            df_fpl['position'] = df_fpl['element_type'].map(element_types)
            df_fpl['price'] = df_fpl['now_cost'] / 10.0
            
        except Exception as e:
            error_msg = f"FPL API Hatası: {str(e)}"
            print(f"❌ {error_msg}")
            logger.error(error_msg, exc_info=True)
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        return df_understat, df_fpl, df_fixtures

    def merge_data(self, df_understat, df_fpl):
        if df_understat.empty or df_fpl.empty:
            return pd.DataFrame()

        print("🤝 Veriler birleştiriliyor...")
        matched_rows = []
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

    def process_fixtures(self, df_fixtures):
        if df_fixtures.empty:
            return df_fixtures
        
        print("🛠️ Fikstürler işleniyor...")
        def parse_val(val, key):
            if isinstance(val, dict): return val.get(key)
            if isinstance(val, str):
                try: 
                    d = ast.literal_eval(val)
                    if isinstance(d, dict): return d.get(key)
                except: pass
            return None

        df_fixtures['xG_h'] = df_fixtures['xG'].apply(lambda x: float(parse_val(x, 'h') or 0))
        df_fixtures['xG_a'] = df_fixtures['xG'].apply(lambda x: float(parse_val(x, 'a') or 0))
        
        if 'h' in df_fixtures.columns:
            df_fixtures['home_team'] = df_fixtures['h'].apply(lambda x: parse_val(x, 'title'))
        if 'a' in df_fixtures.columns:
            df_fixtures['away_team'] = df_fixtures['a'].apply(lambda x: parse_val(x, 'title'))
        
        return df_fixtures

    def get_next_gw(self):
        """Sıradaki Gameweek bilgisini çeker"""
        try:
            r = self._retry_request(requests.get, self.fpl_url)
            data = r.json()
            next_event = next((e for e in data['events'] if e['is_next']), None)
            if next_event:
                return {
                    'id': next_event['id'],
                    'name': next_event['name'],
                    'deadline': next_event['deadline_time']
                }
            return {'id': 0, 'name': 'Unknown GW', 'deadline': '-'}
        except Exception as e:
            error_msg = f"get_next_gw API Hatası: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {'id': 0, 'name': 'Unknown GW', 'deadline': '-'}
    
    def fetch_fpl_fixtures(self):
        """
        FPL API'sinden fixtures verilerini çeker.
        Retry mekanizması ile güvenli API çağrısı yapar.
        """
        print("📅 FPL fixtures verileri çekiliyor...")
        try:
            r = self._retry_request(requests.get, self.fpl_fixtures_url, headers=self.headers)
            fixtures_data = r.json()
            return fixtures_data
        except Exception as e:
            error_msg = f"FPL Fixtures API Hatası: {str(e)}"
            print(f"❌ {error_msg}")
            logger.error(error_msg, exc_info=True)
            return []
    
    def get_fpl_data(self, df_players):
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
        
        print("🔍 Oyuncuların bir sonraki maçları bulunuyor...")
        
        # FPL fixtures verilerini çek
        fixtures_data = self.fetch_fpl_fixtures()
        if not fixtures_data:
            print("⚠️ Fixtures verisi çekilemedi, varsayılan değerler kullanılıyor.")
            df_players['is_home'] = 0
            df_players['opponent_difficulty'] = 3
            df_players['opponent_team'] = 'Unknown'
            return df_players
        
        # Bootstrap-static'ten takım bilgilerini ve gameweek bilgisini çek
        try:
            r_static = self._retry_request(requests.get, self.fpl_url)
            static_data = r_static.json()
            teams_dict = {t['id']: t['name'] for t in static_data['teams']}
            
            # Bir sonraki gameweek'i bul
            next_event = next((e for e in static_data['events'] if e.get('is_next', False)), None)
            if not next_event:
                # Eğer is_next yoksa, finished=False olan ilk event'i bul
                next_event = next((e for e in static_data['events'] if not e.get('finished', True)), None)
            
            next_gw_id = next_event['id'] if next_event else None
        except Exception as e:
            logger.warning(f"Bootstrap-static verisi çekilemedi: {str(e)}")
            teams_dict = {}
            next_gw_id = None
        
        # Fixtures'ı DataFrame'e çevir
        df_fixtures = pd.DataFrame(fixtures_data)
        
        if df_fixtures.empty:
            print("⚠️ Fixtures DataFrame boş, varsayılan değerler kullanılıyor.")
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
            print("⚠️ Bir sonraki gameweek için fixture bulunamadı, varsayılan değerler kullanılıyor.")
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

    def fetch_player_match_history(self, player_id, season=2025):
        """
        Understat API'den oyuncu bazlı maç geçmişini çeker.
        player_id: Understat oyuncu ID'si
        season: Sezon yılı (varsayılan: 2025)
        """
        try:
            player_api_url = f"https://understat.com/getPlayerData/?player_id={player_id}&season={season}"
            r = self._retry_request(
                requests.get, 
                player_api_url,
                headers={'X-Requested-With': 'XMLHttpRequest'}
            )
            
            data = r.json()
            # API'den gelen veri muhtemelen maç bazlı xG verilerini içerir
            if isinstance(data, dict) and 'matches' in data:
                return data['matches']
            elif isinstance(data, list):
                return data
            else:
                logger.warning(f"Oyuncu maç geçmişi formatı beklenmedik (player_id: {player_id})")
                return []
        except Exception as e:
            logger.debug(f"Oyuncu maç geçmişi çekilemedi (player_id: {player_id}): {str(e)}")
            return []
    
    def calculate_form_momentum(self, df_merged):
        """
        Son 3 maçtaki xG ortalaması ile son 10 maçtaki xG ortalamasını karşılaştırarak
        form_momentum sütunu oluşturur.
        Gerçek maç bazlı veriler için Understat API'den oyuncu bazlı maç geçmişi çekilir.
        """
        print("📊 Form momentum hesaplanıyor...")
        
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
                
            except Exception as e:
                logger.warning(f"Form momentum hesaplanamadı (index: {idx}): {str(e)}")
                form_momentum_values.append(0.0)
        
        df_merged['form_momentum'] = form_momentum_values
        return df_merged

    def fetch_user_team(self, team_id):
        print(f"👤 Kullanıcı takımı çekiliyor: {team_id}")
        try:
            r_static = self._retry_request(requests.get, self.fpl_url)
            data_static = r_static.json()
            current_gw_obj = next((x for x in data_static['events'] if x.get('is_current', False)), None)
            if not current_gw_obj:
                current_gw_obj = next((x for x in reversed(data_static['events']) if x.get('finished', False)), None)
            if not current_gw_obj:
                return None, 0
            
            gw_id = current_gw_obj['id']
            picks_url = f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw_id}/picks/"
            r_picks = self._retry_request(requests.get, picks_url)
            
            data_picks = r_picks.json()
            player_ids = [p['element'] for p in data_picks['picks']]
            bank = data_picks['entry_history']['bank'] / 10.0
            return player_ids, bank
        except Exception as e:
            error_msg = f"fetch_user_team API Hatası (team_id: {team_id}): {str(e)}"
            print(f"❌ {error_msg}")
            logger.error(error_msg, exc_info=True)
            return None, 0
