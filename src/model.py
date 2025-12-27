import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class FPLModel:
    def __init__(self):
        # Parametreleri biraz daha güçlendirdik
        self.model = GradientBoostingRegressor(
            n_estimators=200, 
            learning_rate=0.05, 
            max_depth=4, 
            random_state=42
        )
        self.features = [
            'xG_per_90', 'xA_per_90', 'us_xGChain', 'us_xGBuildup', 
            'us_shots', 'us_key_passes', 'pos_DEF', 'pos_FWD', 'pos_GK', 'pos_MID'
        ]

    def train_and_predict(self, df):
        print("🤖 Model eğitiliyor ve test ediliyor...")
        
        original_df = df.copy()
        
        # One-Hot Encoding
        df_encoded = pd.get_dummies(df, columns=['position'], prefix='pos')
        for col in self.features:
            if col not in df_encoded.columns: df_encoded[col] = 0
        df_encoded[self.features] = df_encoded[self.features].fillna(0)
        
        # Eğitim Verisi Hazırlığı (>400 dk oynayanlar)
        train_data = df_encoded[df_encoded['minutes'] > 400].copy()
        train_data['pts_per_90'] = (train_data['total_points'] / train_data['minutes']) * 90
        train_data = train_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['pts_per_90'])
        
        X = train_data[self.features]
        y = train_data['pts_per_90']
        
        # --- VALIDATION SPLIT (Test Mekanizması) ---
        # Verinin %20'sini test için ayırıyoruz
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Modeli Eğit (Sadece %80 ile)
        self.model.fit(X_train, y_train)
        
        # Test Performansını Ölç (%20 üzerinde)
        y_pred_test = self.model.predict(X_test)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'mae': mean_absolute_error(y_test, y_pred_test),
            'r2': r2_score(y_test, y_pred_test),
            'feature_importance': dict(zip(self.features, self.model.feature_importances_))
        }
        
        # Test verilerini detaylı analiz için sakla
        validation_df = X_test.copy()
        validation_df['Actual_Points'] = y_test
        validation_df['Predicted_Points'] = y_pred_test
        validation_df['Error'] = validation_df['Actual_Points'] - validation_df['Predicted_Points']
        
        # İsimleri geri getirmek için index kullan
        validation_df = validation_df.join(original_df[['web_name', 'team_name', 'position', 'price']], how='left')
        
        # --- TÜM VERİ İÇİN TAHMİN ---
        # Şimdi modeli tüm veriyle tekrar eğitmiyoruz, mevcut eğitilmiş modelle herkese tahmin yapıyoruz
        print("🔮 Tüm oyuncular için tahminler üretiliyor...")
        df_encoded['predicted_xP_per_90'] = self.model.predict(df_encoded[self.features])
        
        # Orijinal sütunları geri getir
        df_encoded['position'] = original_df['position']
        
        return df_encoded, metrics, validation_df
