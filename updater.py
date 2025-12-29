from src.data_loader import DataLoader
from src.model import FPLModel
from src.optimizer import Optimizer
import pandas as pd
import json
import os
import logging

logger = logging.getLogger(__name__)

def main() -> None:
    logger.info("🚀 Güncelleme Başlatılıyor...")
    
    # 1. Veri Yükle
    loader = DataLoader()
    
    # DINAMIK GW BILGISI
    gw_info = loader.get_next_gw()
    logger.info("📅 Hedef Hafta: %s (GW%s)", gw_info['name'], gw_info['id'])
    
    df_us, df_fpl, df_fixtures = loader.fetch_all_data()
    df_merged = loader.merge_data(df_us, df_fpl)
    df_fixtures = loader.process_fixtures(df_fixtures)
    
    # Fixture bilgilerini ekle (is_home, opponent_difficulty, opponent_team)
    df_merged = loader.get_fpl_data(df_merged)
    
    # 2. Modelleme
    model = FPLModel()
    df_predictions, metrics, df_validation = model.train_and_predict(df_merged)
    
    # 3. Optimizasyon
    opt = Optimizer()
    df_with_metrics = opt.calculate_metrics(df_predictions, df_fixtures)
    
    # 4. Kayıt Klasörü
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # --- METADATA KAYDI (Siteye GW bilgisini aktarmak için) ---
    with open('data/metadata.json', 'w') as f:
        json.dump(gw_info, f)
        
    # A. Ana Veriler
    df_with_metrics.to_csv('data/all_players.csv', index=False)
    
    # B. Model Test Verileri
    df_validation.to_csv('data/model_validation.csv', index=False)
    with open('data/model_metrics.json', 'w') as f:
        json.dump(metrics, f)
    
    # C. Dream Teams (Genel İsimlendirme)
    # Kısa Vade (Short Term) - Dinamik GW ismi yerine sabit dosya ismi kullanıyoruz ki app.py bozulmasın
    dt_short = opt.solve_dream_team(df_with_metrics, target_metric='gw19_xP', budget=100.0)
    dt_short.to_csv('data/dream_team_short.csv', index=False)
    
    # Uzun Vade (Long Term)
    dt_long = opt.solve_dream_team(df_with_metrics, target_metric='long_term_xP', budget=100.0)
    dt_long.to_csv('data/dream_team_long.csv', index=False)
    
    logger.info("✅ BİTTİ! %s verileri güncellendi.", gw_info['name'])

if __name__ == "__main__":
    main()
