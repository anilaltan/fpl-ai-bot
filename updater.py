from src.data_loader import DataLoader
from src.model import FPLModel
from src.optimizer import Optimizer
import pandas as pd
import os

def main():
    print("🚀 Güncelleme Başlatılıyor...")
    
    # Adım 1: Veri Yükle
    loader = DataLoader()
    df_us, df_fpl, df_fixtures = loader.fetch_all_data()
    df_merged = loader.merge_data(df_us, df_fpl)
    df_fixtures = loader.process_fixtures(df_fixtures)
    
    # Adım 2: Modelleme
    model = FPLModel()
    df_predictions = model.train_and_predict(df_merged)
    
    # Adım 3: Optimizasyon ve 5GW
    opt = Optimizer()
    final_df = opt.calculate_5gw_projection(df_predictions, df_fixtures)
    
    # Adım 4: Kaydet
    if not os.path.exists('data'):
        os.makedirs('data')
        
    final_df.to_csv('data/final_predictions.csv', index=False)
    print("✅ BİTTİ! 'data/final_predictions.csv' kaydedildi.")

if __name__ == "__main__":
    main()