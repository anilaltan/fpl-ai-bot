import pandas as pd
import numpy as np

class Optimizer:
    def calculate_5gw_projection(self, df_players, df_fixtures):
        print("📅 5 Haftalık projeksiyon hesaplanıyor...")
        
        # 1. Takım Savunma Gücü (xGA)
        played = df_fixtures[df_fixtures['isResult'] == True]
        team_xga = {}
        all_teams = set(played['home_team'].unique()) | set(played['away_team'].unique())
        
        for t in all_teams:
            home = played[played['home_team'] == t]
            away = played[played['away_team'] == t]
            total_xga = home['xG_a'].sum() + away['xG_h'].sum()
            matches = len(home) + len(away)
            team_xga[t] = total_xga / matches if matches > 0 else 1.5
            
        avg_xga = np.mean(list(team_xga.values()))
        fdr_map = {t: x/avg_xga for t, x in team_xga.items()}
        
        # 2. Takım Eşleştirme (FPL -> Understat)
        fpl_to_us = {
            'Man Utd': 'Manchester United', 'Man City': 'Manchester City', 'Spurs': 'Tottenham',
            'Newcastle': 'Newcastle United', 'Wolves': 'Wolverhampton Wanderers', 
            "Nott'm Forest": 'Nottingham Forest', 'Sheffield Utd': 'Sheffield United'
        }
        
        # 3. 5 Haftalık Zorluk Katsayısı
        def get_strength(team_name):
            mapped = fpl_to_us.get(team_name, team_name)
            upcoming = df_fixtures[
                ((df_fixtures['home_team'] == mapped) | (df_fixtures['away_team'] == mapped)) &
                (df_fixtures['isResult'] == False)
            ].sort_values('datetime').head(5)
            
            if upcoming.empty: return 5.0
            
            score = 0
            for _, row in upcoming.iterrows():
                is_home = row['home_team'] == mapped
                opp = row['away_team'] if is_home else row['home_team']
                mult = fdr_map.get(opp, 1.0)
                site = 1.1 if is_home else 0.9
                score += mult * site
            return score

        # 4. Hesaplama
        df_players = df_players[df_players['minutes'] >= 500].copy()
        
        # Weighted xP (Dakika düzeltmesi)
        df_players['avg_mins'] = (df_players['minutes'] / 18).clip(upper=90)
        df_players['weighted_xP'] = df_players['predicted_xP_per_90'] * (df_players['avg_mins'] / 90)
        
        # Fikstür Çarpanı
        df_players['fixture_strength'] = df_players['team_name'].apply(get_strength)
        df_players['final_5gw_xP'] = df_players['weighted_xP'] * df_players['fixture_strength']
        
        return df_players