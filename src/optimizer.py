import pandas as pd
import numpy as np

class Optimizer:
    def calculate_metrics(self, df_players, df_fixtures):
        print("📅 5 Haftalık projeksiyon hesaplanıyor...")
        
        # 1. Takım Savunma Gücü (xGA)
        played = df_fixtures[df_fixtures['isResult'] == True]
        team_xga = {}
        all_teams = set(played['home_team'].dropna().unique()) | set(played['away_team'].dropna().unique())
        
        for t in all_teams:
            if not isinstance(t, str): continue
            home = played[played['home_team'] == t]
            away = played[played['away_team'] == t]
            total_xga = home['xG_a'].sum() + away['xG_h'].sum()
            matches = len(home) + len(away)
            team_xga[t] = total_xga / matches if matches > 0 else 1.5
            
        avg_xga = np.mean(list(team_xga.values())) if team_xga else 1.5
        fdr_map = {t: x/avg_xga for t, x in team_xga.items()}
        
        # 2. Takım Eşleştirme
        fpl_to_us = {
            'Man Utd': 'Manchester United', 'Man City': 'Manchester City', 'Spurs': 'Tottenham',
            'Newcastle': 'Newcastle United', 'Wolves': 'Wolverhampton Wanderers', 
            "Nott'm Forest": 'Nottingham Forest', 'Sheffield Utd': 'Sheffield United', 'Luton': 'Luton Town'
        }
        
        # 3. Fikstür Zorluğu
        def get_strength(team_name, weeks=5):
            mapped = fpl_to_us.get(team_name, team_name)
            upcoming = df_fixtures[
                ((df_fixtures['home_team'] == mapped) | (df_fixtures['away_team'] == mapped)) &
                (df_fixtures['isResult'] == False)
            ].sort_values('datetime').head(weeks)
            
            if upcoming.empty: return weeks * 1.0
            
            score = 0
            for _, row in upcoming.iterrows():
                is_home = row['home_team'] == mapped
                opp = row['away_team'] if is_home else row['home_team']
                mult = fdr_map.get(opp, 1.0)
                site = 1.1 if is_home else 0.9
                score += mult * site
            return score

        # 4. Veri Hazırlığı
        df = df_players[df_players['minutes'] >= 500].copy()
        
        # Weighted Base xP
        df['avg_mins'] = (df['minutes'] / 18).clip(upper=90)
        df['base_xP'] = df['predicted_xP_per_90'] * (df['avg_mins'] / 90)
        
        # GW19 Puanı
        df['gw19_strength'] = df['team_name'].apply(lambda x: get_strength(x, weeks=1))
        df['gw19_xP'] = df['base_xP'] * df['gw19_strength']
        
        # 5 Haftalık Puan
        df['gw5_strength'] = df['team_name'].apply(lambda x: get_strength(x, weeks=5))
        df['long_term_xP'] = df['base_xP'] * df['gw5_strength']
        # app.py uyumluluğu için final_5gw_xP sütununu da oluşturuyoruz
        df['final_5gw_xP'] = df['long_term_xP']
        
        return df

    def solve_dream_team(self, df, target_metric='gw19_xP', budget=100.0):
        print(f"🧩 Kadro kuruluyor... Hedef: {target_metric}")
        
        pool = df.drop_duplicates(subset=['web_name', 'team_name']).sort_values(target_metric, ascending=False).copy()
        
        squad = []
        selected_names = set()
        
        current_cost = 0
        pos_limits = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        pos_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        team_counts = {t: 0 for t in pool['team_name'].unique()}
        
        for _, p in pool.iterrows():
            if len(squad) < 15:
                pos = p['position']
                team = p['team_name']
                name = p['web_name']
                
                if name not in selected_names:
                    if pos_counts.get(pos, 0) < pos_limits.get(pos, 0) and team_counts.get(team, 0) < 3:
                        squad.append(p)
                        selected_names.add(name)
                        pos_counts[pos] = pos_counts.get(pos, 0) + 1
                        team_counts[team] = team_counts.get(team, 0) + 1
                        current_cost += p['price']
        
        squad_df = pd.DataFrame(squad)
        
        iter_count = 0
        while current_cost > budget and iter_count < 1000:
            best_swap = None
            min_loss_ratio = float('inf')
            
            current_squad_names = squad_df['web_name'].tolist()
            candidates = pool[~pool['web_name'].isin(current_squad_names)]
            
            for idx, p_out in squad_df.iterrows():
                avail = candidates[
                    (candidates['position'] == p_out['position']) & 
                    (candidates['price'] < p_out['price'])
                ]
                
                for _, p_in in avail.iterrows():
                    team_ok = (p_in['team_name'] == p_out['team_name']) or (team_counts.get(p_in['team_name'], 0) < 3)
                    
                    if team_ok:
                        price_diff = p_out['price'] - p_in['price']
                        xp_diff = p_out[target_metric] - p_in[target_metric]
                        
                        if price_diff > 0:
                            ratio = xp_diff / price_diff
                            if ratio < min_loss_ratio:
                                min_loss_ratio = ratio
                                best_swap = (idx, p_in)
            
            if best_swap:
                idx_out, p_in = best_swap
                p_out = squad_df.loc[idx_out]
                
                current_cost = current_cost - p_out['price'] + p_in['price']
                team_counts[p_out['team_name']] -= 1
                team_counts[p_in['team_name']] = team_counts.get(p_in['team_name'], 0) + 1
                
                squad_df = squad_df.drop(idx_out)
                squad_df = pd.concat([squad_df, p_in.to_frame().T], ignore_index=True)
            else:
                break
            
            iter_count += 1
            
        return squad_df

    def suggest_transfer(self, current_team_df, all_players_df, bank_balance):
        print("🔄 Transfer fırsatları aranıyor...")
        
        # Ana havuzdan mevcut takımı çıkar
        pool = all_players_df[~all_players_df['web_name'].isin(current_team_df['web_name'])].copy()
        
        best_transfer = None
        max_gain = 0
        
        # Takımdaki her oyuncuyu satmayı dene
        for idx, p_out in current_team_df.iterrows():
            # Bütçe = Satılan Oyuncu Fiyatı + Bankadaki Para
            budget = p_out['price'] + bank_balance
            
            # Adayları Filtrele (Aynı pozisyon, Bütçe yetiyor)
            candidates = pool[
                (pool['position'] == p_out['position']) & 
                (pool['price'] <= budget)
            ]
            
            if candidates.empty: continue
            
            # En iyi adayı bul (5 haftalık puana göre)
            # Eğer final_5gw_xP yoksa, long_term_xP kullan (yukarıda eşitledik)
            metric_col = 'final_5gw_xP' if 'final_5gw_xP' in candidates.columns else 'long_term_xP'
            
            # NaN değerleri temizle
            candidates = candidates.dropna(subset=[metric_col])
            if candidates.empty: continue

            best_in = candidates.sort_values(metric_col, ascending=False).iloc[0]
            
            # Puan Kazancı (Gain)
            # p_out içinde metric_col olmayabilir (eğer dışarıdan geldiyse), kontrol et
            p_out_score = p_out.get(metric_col, 0)
            
            gain = best_in[metric_col] - p_out_score
            
            if gain > max_gain:
                max_gain = gain
                best_transfer = {
                    'out': p_out,
                    'in': best_in,
                    'gain': gain
                }
                
        return best_transfer
