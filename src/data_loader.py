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
            matched_rows.append(res)
            
        df_merged = pd.DataFrame(matched_rows)
        df_merged['us_xG'] = df_merged.get('us_xG', 0).fillna(0)
        df_merged['us_xA'] = df_merged.get('us_xA', 0).fillna(0)
        df_merged['minutes'] = pd.to_numeric(df_merged['minutes'], errors='coerce').fillna(0)
        df_merged['xG_per_90'] = np.where(df_merged['minutes']>0, (df_merged['us_xG']/df_merged['minutes'])*90, 0)
        df_merged['xA_per_90'] = np.where(df_merged['minutes']>0, (df_merged['us_xA']/df_merged['minutes'])*90, 0)
        
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
