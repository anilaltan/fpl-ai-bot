# Project technical system dump preparation
_Exported on 12/30/2025 at 03:26:17 GMT+3 from Cursor (2.2.44)_

---

**User**

@Codebase
(Model: GPT-5.1 Codex Max)

Ben bu projenin teknik detaylarını yeni bir AI Mimar (Gemini Advanced) ile paylaşacağım. Ona projenin **tam mevcut durumunu** aktaracak bir "System Dump" (Sistem Dökümü) hazırla.

Aşağıdaki başlıkları içeren, kopyalanabilir tek bir Markdown çıktısı ver:

1.  **FILE TREE:** Projenin dosya ağacı yapısı (derinlik 2 seviye yeterli).
2.  **CONFIG:** `config.yaml` dosyasının **tam içeriği**. (Ayarlarımızı görmesi şart).
3.  **DATA SCHEMA:** `data/all_players.csv` dosyasının **sadece sütun isimleri (headers)** ve ilk 2 satır örneği. (Modelin hangi veriyi kullandığını anlaması için).
4.  **METADATA:** `data/feature_list.json` dosyasının içeriği (Varsa).
5.  **CORE LOGIC:**
    * `src/model.py`: Özellikle `train_model` ve `predict_points` fonksiyonlarının tam kodu.
    * `src/optimizer.py`: Özellikle `calculate_minutes_risk` ve optimizasyon mantığının olduğu kısımlar.
    * `src/data_loader.py`: Sadece `get_fpl_data` fonksiyonunun başlığı ve veri işleme (feature engineering) yapılan kısımları.

**ÖNEMLİ:** Yorum yapma, sadece kod ve veri içeriğini yapılandırılmış şekilde ver. Çıktı, bir başka LLM'in okuyup projeyi 3 saniyede anlayabileceği formatta olsun.

---

**Cursor**

## System Dump

1. FILE TREE
- app.py
- auth_debug.py
- config.yaml
- data/
  - all_players.csv
  - dream_team_long.csv
  - dream_team_short.csv
  - metadata.json
  - model_metrics.json
  - model_validation.csv
- diagnose_auth.py
- error.log
- generate_password.py
- LICENSE
- README.md
- requirements.txt
- src/
  - __init__.py
  - __pycache__/
  - data_loader.py
  - model.py
  - optimizer.py
  - understat_adapter.py
- updater.py

2. CONFIG (`config.yaml`)
```yaml
credentials:
  usernames:
    admin:
      email: admin@fplaibot.com
      name: Admin User
      password: $2b$12$V6S4pXIVEZWDQFH6Dy80uOvYh04IQnaAXScQgxL2djg0WeFrxj4Na # Password: admin123
      role: admin  # Full access to everything
    
    premium_user:
      email: premium@example.com
      name: Premium User
      password: $2b$12$WH0KrnWFWOewnG38adqpkey6LYvDeRxDJGqmXg4FZJI0xoPxQa.66 # Password: premium123
      role: premium  # Access to all features including Transfer Wizard and Model Lab
    
    testuser:
      email: test@example.com
      name: Test User
      password: $2b$12$O9mu24e1m/RxlheycJNOE.LPBwUnjhAVEl8qGYLVW2CkojMo.4RcW # Password: test123
      role: free  # Limited access - no Transfer Wizard or Model Lab

cookie:
  expiry_days: 30
  key: fpl_ai_bot_secret_key_change_this  # IMPORTANT: Change this to a random string for production
  name: fpl_ai_bot_cookie

# NOTES:
# ------
# 1. The passwords above are hashed using bcrypt
# 2. To generate new hashed passwords, use the generate_password.py script
# 3. Never commit real passwords to version control
# 4. Change the cookie key to a random string before deploying
#
# USER ROLES:
# -----------
# - admin: Full access to everything (for you and trusted admins)
# - premium: Paid users - access to Transfer Wizard, Model Lab, and all features
# - free: Free tier users - access to GW Dream Team, Long Term Predictions, Player Pool only
#
# TO ADD NEW USERS:
# -----------------
# 1. Run: python generate_password.py
# 2. Enter the desired password when prompted
# 3. Copy the hashed password
# 4. Add a new user entry under credentials.usernames following the format above
# 5. Set their role to either 'free', 'premium', or 'admin'
#
# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
model:
  # Gradient Boosting Regressor hyperparameters
  n_estimators: 200
  learning_rate: 0.05
  max_depth: 4
  random_state: 42
  
  # Training data filtering
  min_minutes_for_training: 400  # Minimum minutes played to include in training
  
  # Validation split
  test_size: 0.2  # 20% of data for validation
  
  # Feature engineering
  minutes_per_game: 90  # Standard minutes per game
  
  # Model features
  features:
    - 'xG_per_90'
    - 'xA_per_90'
    - 'us_xGChain'
    - 'us_xGBuildup'
    - 'us_shots'
    - 'us_key_passes'
    - 'pos_DEF'
    - 'pos_FWD'
    - 'pos_GK'
    - 'pos_MID'

# ============================================================================
# DATA LOADER CONFIGURATION
# ============================================================================
data_loader:
  understat_url: "https://understat.com/getLeagueData/EPL/2025"
  understat_league: "EPL"
  understat_seasons:
    - 2025
    - 2024
  fpl_url: "https://fantasy.premierleague.com/api/bootstrap-static/"
  fpl_fixtures_url: "https://fantasy.premierleague.com/api/fixtures/"
  user_agent: "Mozilla/5.0"
  max_retries: 3
  retry_backoff_base: 2  # exponential backoff base (2^attempt)
  request_timeout: 15   # seconds

# ============================================================================
# OPTIMIZER CONFIGURATION
# ============================================================================
optimizer:
  # Fixture difficulty calculation
  default_team_xga: 1.5  # Default xGA when team data unavailable
  default_fdr_score: 1.0  # Default FDR score
  home_advantage_multiplier: 1.1  # Home team advantage multiplier
  away_disadvantage_multiplier: 0.9  # Away team disadvantage multiplier
  
  # Projection periods
  short_term_weeks: 1  # GW19 projection
  long_term_weeks: 5  # 5-week projection
  
  # Player filtering
  min_minutes_for_optimization: 500  # Minimum minutes to include in optimization
  
  # Minutes calculation
  games_per_season: 18  # Number of games to normalize minutes
  max_minutes_per_game: 90  # Maximum minutes per game
  
  # Squad building constraints
  budget: 100.0  # Maximum squad budget
  squad_size: 15  # Total squad size
  position_limits:
    GK: 2
    DEF: 5
    MID: 5
    FWD: 3
  max_players_per_team: 3  # Maximum players from same team
  
  # Optimization algorithm
  max_iterations: 1000  # Maximum iterations for budget optimization

  # Rotation / Minutes Risk Management
  rotation_risk:
    minutes_var_threshold: 400.0   # variance threshold to apply risk
    risk_coefficient: 0.85         # multiplier when risk is high
    team_penalty_multiplier: 0.9   # additional multiplier for risky teams
    risk_teams:
      - "Man City"
      - "Chelsea"
  
  # Team name mapping (FPL to Understat)
  team_name_mapping:
    'Man Utd': 'Manchester United'
    'Man City': 'Manchester City'
    'Spurs': 'Tottenham'
    'Newcastle': 'Newcastle United'
    'Wolves': 'Wolverhampton Wanderers'
    "Nott'm Forest": 'Nottingham Forest'
    'Sheffield Utd': 'Sheffield United'
    'Luton': 'Luton Town'
```

3. DATA SCHEMA (`data/all_players.csv`)
- Headers: `can_transact,can_select,chance_of_playing_next_round,chance_of_playing_this_round,code,cost_change_event,cost_change_event_fall,cost_change_start,cost_change_start_fall,dreamteam_count,element_type,ep_next,ep_this,event_points,first_name,form,id,in_dreamteam,news,news_added,now_cost,photo,points_per_game,removed,second_name,selected_by_percent,special,squad_number,status,team,team_code,total_points,transfers_in,transfers_in_event,transfers_out,transfers_out_event,value_form,value_season,web_name,region,team_join_date,birth_date,has_temporary_code,opta_code,minutes,goals_scored,assists,clean_sheets,goals_conceded,own_goals,penalties_saved,penalties_missed,yellow_cards,red_cards,saves,bonus,bps,influence,creativity,threat,ict_index,clearances_blocks_interceptions,recoveries,tackles,defensive_contribution,starts,expected_goals,expected_assists,expected_goal_involvements,expected_goals_conceded,corners_and_indirect_freekicks_order,corners_and_indirect_freekicks_text,direct_freekicks_order,direct_freekicks_text,penalties_order,penalties_text,scout_risks,influence_rank,influence_rank_type,creativity_rank,creativity_rank_type,threat_rank,threat_rank_type,ict_index_rank,ict_index_rank_type,expected_goals_per_90,saves_per_90,expected_assists_per_90,expected_goal_involvements_per_90,expected_goals_conceded_per_90,goals_conceded_per_90,now_cost_rank,now_cost_rank_type,form_rank,form_rank_type,points_per_game_rank,points_per_game_rank_type,selected_rank,selected_rank_type,starts_per_90,clean_sheets_per_90,defensive_contribution_per_90,team_name,price,us_id,us_player_name,us_games,us_time,us_goals,us_xG,us_assists,us_xA,us_shots,us_key_passes,us_yellow_cards,us_red_cards,us_position,us_team_title,us_npg,us_npxG,us_xGChain,us_xGBuildup,us_season,us_player_id,xG_per_90,xA_per_90,form_momentum,is_home,opponent_difficulty,opponent_team,volatility_index,pos_DEF,pos_FWD,pos_GK,pos_MID,position_orig,predicted_xP_per_90,predicted_xP_def,predicted_xP_off,upside_potential,position,avg_mins,base_xP,minutes_risk,risk_factor,gw19_strength,gw19_xP,gw5_strength,long_term_xP,final_5gw_xP`
- Row 1: `True,True,,,154561,0,0,5,-5,1,1,4.7,4.7,2,David,3.7,1,False,,,60,154561.jpg,4.3,False,Raya Martín,35.6,False,,a,1,3,77,3222242,19075,1333224,52471,0.6,12.8,Raya,200.0,2024-07-04,1995-09-15,False,p154561,1620.0,0.0,0.0,9.0,11.0,0,0,0,1,0,29.0,4,314,288.6,20.3,0.0,31.0,13,147,0,0,18,0.00,0.04,0.04,10.97,,,,,,,[],101,16,347,3,766,88,248,15,0.0,1.61,0.0,0.0,0.61,0.61,85,1,82,9,58,3,6,1,1.0,0.5,0.0,Arsenal,6.0,9676,David Raya,18.0,1620.0,0.0,0.0,0,0.1079027578234672,0.0,2.0,1,0,GK,Arsenal,0,0,3.296872779726982,3.2473727725446224,2025.0,9676,0.0,0.005994597656859288,0.0,1,3,Aston Villa,0.0,0.0,0.0,1.0,0.0,GK,4.273470878601074,4.273470878601074,,,GK,90.0,4.273470878601074,0.0,1.0,1.1184970561575502,4.779864597290321,5.142753661807314,21.977408009552594,21.977408009552594`
- Row 2: `True,True,100.0,25.0,226597,0,0,2,-2,3,2,1.2,0.3,1,Gabriel,0.2,5,False,,2025-11-17T12:00:07.436586Z,62,226597.jpg,6.8,False,dos Santos Magalhães,14.0,False,,a,1,3,82,4077473,187695,4888020,11122,0.0,13.2,Gabriel,30.0,2020-09-01,1997-12-19,False,p226597,1009.0,1.0,2.0,7.0,5.0,0,0,0,1,0,0.0,13,281,318.6,62.8,92.0,47.2,89,16,12,101,11,0.84,0.69,1.53,6.49,,,,,,,[],81,37,245,71,185,41,155,51,0.07,0.0,0.06,0.13,0.58,0.45,75,3,412,151,2,1,25,8,0.98,0.62,9.01,Arsenal,6.2,,,0.0,,,0.0,,0.0,0.0,0.0,,,,,,,0.0,0.0,,,0.0,0.0,0.0,1,3,Aston Villa,0.0,1.0,0.0,0.0,0.0,DEF,7.314554214477539,7.314554214477539,,,DEF,56.05555555555556,4.555793334819652,0.0,1.0,1.1184970561575502,5.09564143345797,5.142753661807314,23.42932285508112,23.42932285508112`

4. METADATA
- `data/feature_list.json`: not found

5. CORE LOGIC
- `src/model.py` – train_model: not present
- `src/model.py` – predict_points:
```python
    def predict_points(
        self,
        model: XGBRegressor,
        df_segment: pd.DataFrame,
        feature_list: List[str],
        segment_name: str
    ) -> np.ndarray:
        """
        Tahmin öncesi veri ve sonuç istatistiklerini loglar.
        Pre-flight kontrol, dtype zorlaması ve scaler doğrulaması içerir.
        """
        feature_payload = self._load_feature_list()
        loaded_features = feature_payload.get(segment_name) or feature_list

        print(f"Model Beklentisi: {loaded_features[:5]}...")
        print(f"Gelen Veri Sütunları: {list(df_segment.columns[:5])}...")

        missing_features = [f for f in loaded_features if f not in df_segment.columns]
        extra_features = [c for c in df_segment.columns if c not in loaded_features]
        if missing_features or extra_features:
            logger.debug(
                "[%s] Feature alignment - missing: %s | extra: %s",
                segment_name,
                missing_features,
                extra_features
            )

        aligned_df = df_segment.reindex(columns=loaded_features, fill_value=0)
        aligned_df = (
            aligned_df.apply(pd.to_numeric, errors='coerce')
            .fillna(0.0)
            .astype(float)
        )
        input_data = aligned_df[loaded_features]

        print(f"[DEBUG][{segment_name}] Input head:\n{input_data.head()}")
        print(f"[DEBUG][{segment_name}] Input describe():\n{input_data.describe(include='all')}")

        scaler = getattr(self, "scaler", None)
        if scaler is not None:
            try:
                sample_scaled = scaler.transform(input_data.iloc[: min(len(input_data), 5)])
                print(
                    f"[DEBUG][{segment_name}] Scaler sample min/max: "
                    f"min={sample_scaled.min():.4f}, max={sample_scaled.max():.4f}"
                )
            except Exception as exc:
                print(f"[DEBUG][{segment_name}] Scaler kontrolü hatası: {exc}")
        else:
            print(f"[DEBUG][{segment_name}] Scaler tanımlı değil, kontrol atlandı.")

        predictions = model.predict(input_data)

        print(f"[DEBUG][{segment_name}] Prediction describe():\n{pd.Series(predictions).describe()}")
        print(f"[DEBUG][{segment_name}] İlk 5 tahmin: {predictions[:5]}")

        if hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=loaded_features)
            print(
                f"[DEBUG][{segment_name}] Feature importances (top 20):\n"
                f"{importances.sort_values(ascending=False).head(20)}"
            )
            if (importances == 0).all():
                print(
                    f"[DEBUG][{segment_name}] WARNING: Feature importances are all zero. "
                    "Model eğitimi başarısız olabilir."
                )

        return predictions
```

- `src/optimizer.py` – calculate_minutes_risk:
```python
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
                if col in row.index and pd.notna(row[col]):
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
```

- `src/optimizer.py` – optimization logic (initial squad, swap, solver):
```python
    def _create_initial_squad(
        self,
        pool: pd.DataFrame,
        target_metric: str
    ) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int], float]:
        """
        Create initial squad by selecting top players within constraints.
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
        """
        logger.info(f"Solving dream team with target metric: {target_metric}")
        
        try:
            if budget is None:
                budget = self.config['optimizer']['budget']
            
            pool = df.drop_duplicates(
                subset=['web_name', 'team_name']
            ).sort_values(target_metric, ascending=False).copy()
            
            if pool.empty:
                logger.warning("No players available for squad building")
                return pd.DataFrame()
            
            squad_df, pos_counts, team_counts, current_cost = self._create_initial_squad(
                pool, target_metric
            )
            
            if squad_df.empty:
                logger.warning("Could not create initial squad")
                return pd.DataFrame()
            
            max_iterations = self.config['optimizer']['max_iterations']
            iter_count = 0
            
            while current_cost > budget and iter_count < max_iterations:
                best_swap = self._find_best_swap(
                    squad_df, pool, team_counts, target_metric
                )
                
                if best_swap:
                    idx_out, p_in = best_swap
                    p_out = squad_df.loc[idx_out]
                    
                    current_cost = current_cost - p_out['price'] + p_in['price']
                    team_counts[p_out['team_name']] -= 1
                    team_counts[p_in['team_name']] = (
                        team_counts.get(p_in['team_name'], 0) + 1
                    )
                    
                    squad_df = squad_df.drop(idx_out)
                    squad_df = pd.concat(
                        [squad_df, p_in.to_frame().T], 
                        ignore_index=True
                    )
                else:
                    break
                
                iter_count += 1
            
            logger.info(
                f"Dream team solved: {len(squad_df)} players, "
                f"cost: {current_cost:.2f}, iterations: {iter_count}"
            )
            
            return squad_df
            
        except Exception as e:
            logger.error(f"Error solving dream team: {e}", exc_info=True)
            raise
```

- `src/data_loader.py` – get_fpl_data:
```python
    def get_fpl_data(self, df_players: pd.DataFrame) -> pd.DataFrame:
        """
        Oyuncu verilerine fixture bilgilerini ekler.
        Her oyuncunun bir sonraki maçını bulur ve şu sütunları ekler:
        - is_home: (Boolean) Eğer maç iç sahadaysa 1, değilse 0
        - opponent_difficulty: (Int) Rakip takımın zorluk derecesi (FDR)
        - opponent_team: (String) Rakip takımın adı
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
        
        df_fixtures = pd.DataFrame(fixtures_data)
        
        if df_fixtures.empty:
            logger.warning("Fixtures DataFrame boş, varsayılan değerler kullanılıyor.")
            df_players['is_home'] = 0
            df_players['opponent_difficulty'] = 3
            df_players['opponent_team'] = 'Unknown'
            return df_players
        
        if next_gw_id is not None:
            df_next_fixtures = df_fixtures[df_fixtures['event'] == next_gw_id].copy()
        else:
            if 'finished' in df_fixtures.columns:
                df_next_fixtures = df_fixtures[df_fixtures['finished'] == False].copy()
            else:
                df_next_fixtures = pd.DataFrame()
            
            if df_next_fixtures.empty:
                if 'event' in df_fixtures.columns:
                    min_event = df_fixtures['event'].min()
                    df_next_fixtures = df_fixtures[df_fixtures['event'] == min_event].copy()
        
        if df_next_fixtures.empty:
            logger.warning("Bir sonraki gameweek için fixture bulunamadı, varsayılan değerler kullanılıyor.")
            df_players['is_home'] = 0
            df_players['opponent_difficulty'] = 3
            df_players['opponent_team'] = 'Unknown'
            return df_players
        
        is_home_list = []
        opponent_difficulty_list = []
        opponent_team_list = []
        
        for idx, player_row in df_players.iterrows():
            player_team_id = player_row.get('team')
            
            if pd.isna(player_team_id) or player_team_id is None:
                is_home_list.append(0)
                opponent_difficulty_list.append(3)
                opponent_team_list.append('Unknown')
                continue
            
            player_fixture = None
            
            home_fixtures = df_next_fixtures[df_next_fixtures['team_h'] == player_team_id]
            if not home_fixtures.empty:
                player_fixture = home_fixtures.iloc[0]
                is_home = 1
                opponent_team_id = player_fixture.get('team_a')
                opponent_difficulty = player_fixture.get('team_h_difficulty', 3)
            else:
                away_fixtures = df_next_fixtures[df_next_fixtures['team_a'] == player_team_id]
                if not away_fixtures.empty:
                    player_fixture = away_fixtures.iloc[0]
                    is_home = 0
                    opponent_team_id = player_fixture.get('team_h')
                    opponent_difficulty = player_fixture.get('team_a_difficulty', 3)
                else:
                    is_home = 0
                    opponent_team_id = None
                    opponent_difficulty = 3
            
            if opponent_team_id is not None and opponent_team_id in teams_dict:
                opponent_team_name = teams_dict[opponent_team_id]
            else:
                opponent_team_name = 'Unknown'
            
            is_home_list.append(is_home)
            opponent_difficulty_list.append(int(opponent_difficulty) if not pd.isna(opponent_difficulty) else 3)
            opponent_team_list.append(opponent_team_name)
        
        df_players['is_home'] = is_home_list
        df_players['opponent_difficulty'] = opponent_difficulty_list
        df_players['opponent_team'] = opponent_team_list
        
        print(f"✅ {len(df_players)} oyuncu için fixture bilgileri eklendi.")
        
        return df_players
```

```yaml
credentials:
  usernames:
    admin:
      email: admin@fplaibot.com
      name: Admin User
      password: $2b$12$V6S4pXIVEZWDQFH6Dy80uOvYh04IQnaAXScQgxL2djg0WeFrxj4Na # Password: admin123
      role: admin  # Full access to everything
    
    premium_user:
      email: premium@example.com
      name: Premium User
      password: $2b$12$WH0KrnWFWOewnG38adqpkey6LYvDeRxDJGqmXg4FZJI0xoPxQa.66 # Password: premium123
      role: premium  # Access to all features including Transfer Wizard and Model Lab
    
    testuser:
      email: test@example.com
      name: Test User
      password: $2b$12$O9mu24e1m/RxlheycJNOE.LPBwUnjhAVEl8qGYLVW2CkojMo.4RcW # Password: test123
      role: free  # Limited access - no Transfer Wizard or Model Lab

cookie:
  expiry_days: 30
  key: fpl_ai_bot_secret_key_change_this  # IMPORTANT: Change this to a random string for production
  name: fpl_ai_bot_cookie

# NOTES:
# ------
# 1. The passwords above are hashed using bcrypt
# 2. To generate new hashed passwords, use the generate_password.py script
# 3. Never commit real passwords to version control
# 4. Change the cookie key to a random string before deploying
#
# USER ROLES:
# -----------
# - admin: Full access to everything (for you and trusted admins)
# - premium: Paid users - access to Transfer Wizard, Model Lab, and all features
# - free: Free tier users - access to GW Dream Team, Long Term Predictions, Player Pool only
#
# TO ADD NEW USERS:
# -----------------
# 1. Run: python generate_password.py
# 2. Enter the desired password when prompted
# 3. Copy the hashed password
# 4. Add a new user entry under credentials.usernames following the format above
# 5. Set their role to either 'free', 'premium', or 'admin'
#
# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
model:
  # Gradient Boosting Regressor hyperparameters
  n_estimators: 200
  learning_rate: 0.05
  max_depth: 4
  random_state: 42
  
  # Training data filtering
  min_minutes_for_training: 400  # Minimum minutes played to include in training
  
  # Validation split
  test_size: 0.2  # 20% of data for validation
  
  # Feature engineering
  minutes_per_game: 90  # Standard minutes per game
  
  # Model features
  features:
    - 'xG_per_90'
    - 'xA_per_90'
    - 'us_xGChain'
    - 'us_xGBuildup'
    - 'us_shots'
    - 'us_key_passes'
    - 'pos_DEF'
    - 'pos_FWD'
    - 'pos_GK'
    - 'pos_MID'

# ============================================================================
# DATA LOADER CONFIGURATION
# ============================================================================
data_loader:
  understat_url: "https://understat.com/getLeagueData/EPL/2025"
  understat_league: "EPL"
  understat_seasons:
    - 2025
    - 2024
  fpl_url: "https://fantasy.premierleague.com/api/bootstrap-static/"
  fpl_fixtures_url: "https://fantasy.premierleague.com/api/fixtures/"
  user_agent: "Mozilla/5.0"
  max_retries: 3
  retry_backoff_base: 2  # exponential backoff base (2^attempt)
  request_timeout: 15   # seconds

# ============================================================================
# OPTIMIZER CONFIGURATION
# ============================================================================
optimizer:
  # Fixture difficulty calculation
  default_team_xga: 1.5  # Default xGA when team data unavailable
  default_fdr_score: 1.0  # Default FDR score
  home_advantage_multiplier: 1.1  # Home team advantage multiplier
  away_disadvantage_multiplier: 0.9  # Away team disadvantage multiplier
  
  # Projection periods
  short_term_weeks: 1  # GW19 projection
  long_term_weeks: 5  # 5-week projection
  
  # Player filtering
  min_minutes_for_optimization: 500  # Minimum minutes to include in optimization
  
  # Minutes calculation
  games_per_season: 18  # Number of games to normalize minutes
  max_minutes_per_game: 90  # Maximum minutes per game
  
  # Squad building constraints
  budget: 100.0  # Maximum squad budget
  squad_size: 15  # Total squad size
  position_limits:
    GK: 2
    DEF: 5
    MID: 5
    FWD: 3
  max_players_per_team: 3  # Maximum players from same team
  
  # Optimization algorithm
  max_iterations: 1000  # Maximum iterations for budget optimization

  # Rotation / Minutes Risk Management
  rotation_risk:
    minutes_var_threshold: 400.0   # variance threshold to apply risk
    risk_coefficient: 0.85         # multiplier when risk is high
    team_penalty_multiplier: 0.9   # additional multiplier for risky teams
    risk_teams:
      - "Man City"
      - "Chelsea"
  
  # Team name mapping (FPL to Understat)
  team_name_mapping:
    'Man Utd': 'Manchester United'
    'Man City': 'Manchester City'
    'Spurs': 'Tottenham'
    'Newcastle': 'Newcastle United'
    'Wolves': 'Wolverhampton Wanderers'
    "Nott'm Forest": 'Nottingham Forest'
    'Sheffield Utd': 'Sheffield United'
    'Luton': 'Luton Town'
```

```python
    def predict_points(
        self,
        model: XGBRegressor,
        df_segment: pd.DataFrame,
        feature_list: List[str],
        segment_name: str
    ) -> np.ndarray:
        """
        Tahmin öncesi veri ve sonuç istatistiklerini loglar.
        Pre-flight kontrol, dtype zorlaması ve scaler doğrulaması içerir.
        """
        feature_payload = self._load_feature_list()
        loaded_features = feature_payload.get(segment_name) or feature_list

        print(f"Model Beklentisi: {loaded_features[:5]}...")
        print(f"Gelen Veri Sütunları: {list(df_segment.columns[:5])}...")

        missing_features = [f for f in loaded_features if f not in df_segment.columns]
        extra_features = [c for c in df_segment.columns if c not in loaded_features]
        if missing_features or extra_features:
            logger.debug(
                "[%s] Feature alignment - missing: %s | extra: %s",
                segment_name,
                missing_features,
                extra_features
            )

        aligned_df = df_segment.reindex(columns=loaded_features, fill_value=0)
        aligned_df = (
            aligned_df.apply(pd.to_numeric, errors='coerce')
            .fillna(0.0)
            .astype(float)
        )
        input_data = aligned_df[loaded_features]

        print(f"[DEBUG][{segment_name}] Input head:\n{input_data.head()}")
        print(f"[DEBUG][{segment_name}] Input describe():\n{input_data.describe(include='all')}")

        scaler = getattr(self, "scaler", None)
        if scaler is not None:
            try:
                sample_scaled = scaler.transform(input_data.iloc[: min(len(input_data), 5)])
                print(
                    f"[DEBUG][{segment_name}] Scaler sample min/max: "
                    f"min={sample_scaled.min():.4f}, max={sample_scaled.max():.4f}"
                )
            except Exception as exc:
                print(f"[DEBUG][{segment_name}] Scaler kontrolü hatası: {exc}")
        else:
            print(f"[DEBUG][{segment_name}] Scaler tanımlı değil, kontrol atlandı.")

        predictions = model.predict(input_data)

        print(f"[DEBUG][{segment_name}] Prediction describe():\n{pd.Series(predictions).describe()}")
        print(f"[DEBUG][{segment_name}] İlk 5 tahmin: {predictions[:5]}")

        if hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=loaded_features)
            print(
                f"[DEBUG][{segment_name}] Feature importances (top 20):\n"
                f"{importances.sort_values(ascending=False).head(20)}"
            )
            if (importances == 0).all():
                print(
                    f"[DEBUG][{segment_name}] WARNING: Feature importances are all zero. "
                    "Model eğitimi başarısız olabilir."
                )

        return predictions
```

```python
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
                if col in row.index and pd.notna(row[col]):
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
```

```python
    def _create_initial_squad(
        self,
        pool: pd.DataFrame,
        target_metric: str
    ) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int], float]:
        """
        Create initial squad by selecting top players within constraints.
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
        """
        logger.info(f"Solving dream team with target metric: {target_metric}")
        
        try:
            if budget is None:
                budget = self.config['optimizer']['budget']
            
            pool = df.drop_duplicates(
                subset=['web_name', 'team_name']
            ).sort_values(target_metric, ascending=False).copy()
            
            if pool.empty:
                logger.warning("No players available for squad building")
                return pd.DataFrame()
            
            squad_df, pos_counts, team_counts, current_cost = self._create_initial_squad(
                pool, target_metric
            )
            
            if squad_df.empty:
                logger.warning("Could not create initial squad")
                return pd.DataFrame()
            
            max_iterations = self.config['optimizer']['max_iterations']
            iter_count = 0
            
            while current_cost > budget and iter_count < max_iterations:
                best_swap = self._find_best_swap(
                    squad_df, pool, team_counts, target_metric
                )
                
                if best_swap:
                    idx_out, p_in = best_swap
                    p_out = squad_df.loc[idx_out]
                    
                    current_cost = current_cost - p_out['price'] + p_in['price']
                    team_counts[p_out['team_name']] -= 1
                    team_counts[p_in['team_name']] = (
                        team_counts.get(p_in['team_name'], 0) + 1
                    )
                    
                    squad_df = squad_df.drop(idx_out)
                    squad_df = pd.concat(
                        [squad_df, p_in.to_frame().T], 
                        ignore_index=True
                    )
                else:
                    break
                
                iter_count += 1
            
            logger.info(
                f"Dream team solved: {len(squad_df)} players, "
                f"cost: {current_cost:.2f}, iterations: {iter_count}"
            )
            
            return squad_df
            
        except Exception as e:
            logger.error(f"Error solving dream team: {e}", exc_info=True)
            raise
```

```python
    def get_fpl_data(self, df_players: pd.DataFrame) -> pd.DataFrame:
        """
        Oyuncu verilerine fixture bilgilerini ekler.
        Her oyuncunun bir sonraki maçını bulur ve şu sütunları ekler:
        - is_home: (Boolean) Eğer maç iç sahadaysa 1, değilse 0
        - opponent_difficulty: (Int) Rakip takımın zorluk derecesi (FDR)
        - opponent_team: (String) Rakip takımın adı
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
        
        df_fixtures = pd.DataFrame(fixtures_data)
        
        if df_fixtures.empty:
            logger.warning("Fixtures DataFrame boş, varsayılan değerler kullanılıyor.")
            df_players['is_home'] = 0
            df_players['opponent_difficulty'] = 3
            df_players['opponent_team'] = 'Unknown'
            return df_players
        
        if next_gw_id is not None:
            df_next_fixtures = df_fixtures[df_fixtures['event'] == next_gw_id].copy()
        else:
            if 'finished' in df_fixtures.columns:
                df_next_fixtures = df_fixtures[df_fixtures['finished'] == False].copy()
            else:
                df_next_fixtures = pd.DataFrame()
            
            if df_next_fixtures.empty:
                if 'event' in df_fixtures.columns:
                    min_event = df_fixtures['event'].min()
                    df_next_fixtures = df_fixtures[df_fixtures['event'] == min_event].copy()
        
        if df_next_fixtures.empty:
            logger.warning("Bir sonraki gameweek için fixture bulunamadı, varsayılan değerler kullanılıyor.")
            df_players['is_home'] = 0
            df_players['opponent_difficulty'] = 3
            df_players['opponent_team'] = 'Unknown'
            return df_players
        
        is_home_list = []
        opponent_difficulty_list = []
        opponent_team_list = []
        
        for idx, player_row in df_players.iterrows():
            player_team_id = player_row.get('team')
            
            if pd.isna(player_team_id) or player_team_id is None:
                is_home_list.append(0)
                opponent_difficulty_list.append(3)
                opponent_team_list.append('Unknown')
                continue
            
            player_fixture = None
            
            home_fixtures = df_next_fixtures[df_next_fixtures['team_h'] == player_team_id]
            if not home_fixtures.empty:
                player_fixture = home_fixtures.iloc[0]
                is_home = 1
                opponent_team_id = player_fixture.get('team_a')
                opponent_difficulty = player_fixture.get('team_h_difficulty', 3)
            else:
                away_fixtures = df_next_fixtures[df_next_fixtures['team_a'] == player_team_id]
                if not away_fixtures.empty:
                    player_fixture = away_fixtures.iloc[0]
                    is_home = 0
                    opponent_team_id = player_fixture.get('team_h')
                    opponent_difficulty = player_fixture.get('team_a_difficulty', 3)
                else:
                    is_home = 0
                    opponent_team_id = None
                    opponent_difficulty = 3
            
            if opponent_team_id is not None and opponent_team_id in teams_dict:
                opponent_team_name = teams_dict[opponent_team_id]
            else:
                opponent_team_name = 'Unknown'
            
            is_home_list.append(is_home)
            opponent_difficulty_list.append(int(opponent_difficulty) if not pd.isna(opponent_difficulty) else 3)
            opponent_team_list.append(opponent_team_name)
        
        df_players['is_home'] = is_home_list
        df_players['opponent_difficulty'] = opponent_difficulty_list
        df_players['opponent_team'] = opponent_team_list
        
        print(f"✅ {len(df_players)} oyuncu için fixture bilgileri eklendi.")
        
        return df_players
```

