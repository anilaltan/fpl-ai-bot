import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

class FPLModel:
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        self.features = [
            'xG_per_90', 'xA_per_90', 'us_xGChain', 'us_xGBuildup', 
            'us_shots', 'us_key_passes', 'pos_DEF', 'pos_FWD', 'pos_GK', 'pos_MID'
        ]

    def train_and_predict(self, df):
        print("🤖 Model eğitiliyor...")
        
        # One-Hot Encoding
        df = pd.get_dummies(df, columns=['position'], prefix='pos')
        for col in self.features:
            if col not in df.columns: df[col] = 0
            
        df[self.features] = df[self.features].fillna(0)
        
        # Eğitim Seti (>400 dk)
        train_data = df[df['minutes'] > 400].copy()
        train_data['pts_per_90'] = (train_data['total_points'] / train_data['minutes']) * 90
        train_data = train_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['pts_per_90'])
        
        X = train_data[self.features]
        y = train_data['pts_per_90']
        
        self.model.fit(X, y)
        
        # Tahmin
        print("🔮 Tahminler üretiliyor...")
        df['predicted_xP_per_90'] = self.model.predict(df[self.features])
        
        return df