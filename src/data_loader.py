import pandas as pd
import requests
import numpy as np
import ast
from thefuzz import process, fuzz
import difflib

class DataLoader:
    def __init__(self):
        self.understat_url = "https://understat.com/getLeagueData/EPL/2025"
        self.fpl_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        self.headers = {'User-Agent': 'Mozilla/5.0'}

    def fetch_all_data(self):
        print("🌍 Veriler çekiliyor...")
        
        # 1. Understat
        r_us = requests.get(self.understat_url, headers={'X-Requested-With': 'XMLHttpRequest'})
        data_us = r_us.json()
        df_understat = pd.DataFrame(data_us['players'])
        df_fixtures = pd.DataFrame(data_us['dates'])
        
        # Sayısal dönüşümler
        cols = ['games', 'time', 'goals', 'xG', 'xA', 'shots', 'key_passes', 'xGChain', 'xGBuildup']
        for c in cols:
            if c in df_understat.columns:
                df_understat[c] = pd.to_numeric(df_understat[c], errors='coerce')

        # 2. FPL
        r_fpl = requests.get(self.fpl_url)
        data_fpl = r_fpl.json()
        df_fpl = pd.DataFrame(data_fpl['elements'])
        
        # Takım ve Pozisyon isimlerini eşle
        teams = {t['id']: t['name'] for t in data_fpl['teams']}
        element_types = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        
        df_fpl['team_name'] = df_fpl['team'].map(teams)
        df_fpl['position'] = df_fpl['element_type'].map(element_types)
        
        return df_understat, df_fpl, df_fixtures

    def merge_data(self, df_understat, df_fpl):
        print("🤝 Veriler birleştiriliyor (Fuzzy Matching)...")
        # İsim eşleştirme mantığı (Senin kodun)
        matched_rows = []
        
        # Takım İsim Haritası
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
            fpl_name = fpl_row['web_name'] # web_name bazen daha temizdir
            full_name = fpl_row['first_name'] + " " + fpl_row['second_name']
            fpl_team = fpl_row['team_name']
            us_target = team_map.get(fpl_team, fpl_team)
            
            # Sadece o takımdakileri filtrele
            candidates = df_understat[df_understat['team_title'] == us_target]
            
            best_score = 0
            match_data = None
            
            if not candidates.empty:
                # Difflib ile eşleşme
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
        
        # Feature Engineering
        df_merged['us_xG'] = df_merged['us_xG'].fillna(0)
        df_merged['us_xA'] = df_merged['us_xA'].fillna(0)
        df_merged['minutes'] = pd.to_numeric(df_merged['minutes'], errors='coerce').fillna(0)
        
        # Per 90 Stats
        df_merged['xG_per_90'] = np.where(df_merged['minutes']>0, (df_merged['us_xG']/df_merged['minutes'])*90, 0)
        df_merged['xA_per_90'] = np.where(df_merged['minutes']>0, (df_merged['us_xA']/df_merged['minutes'])*90, 0)
        
        return df_merged

    def process_fixtures(self, df_fixtures):
        # xG ve Takım isimlerini parse etme (Hata düzeltilmiş versiyon)
        print("🛠️ Fikstürler işleniyor...")
        
        def parse_val(val, key):
            if isinstance(val, dict): return val.get(key)
            if isinstance(val, str):
                try: 
                    d = ast.literal_eval(val)
                    return d.get(key)
                except: pass
            return None

        # xG Parsing
        df_fixtures['xG_h'] = df_fixtures['xG'].apply(lambda x: float(parse_val(x, 'h') or 0))
        df_fixtures['xG_a'] = df_fixtures['xG'].apply(lambda x: float(parse_val(x, 'a') or 0))
        
        # Team Name Parsing
        df_fixtures['home_team'] = df_fixtures['h'].apply(lambda x: parse_val(x, 'title'))
        df_fixtures['away_team'] = df_fixtures['a'].apply(lambda x: parse_val(x, 'title'))
        
        return df_fixtures