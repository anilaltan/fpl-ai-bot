from pathlib import Path
from typing import Optional, Union, Any
import json
import logging
import pandas as pd
import numpy as np
from src.data_loader import DataLoader
from src.model import FPLModel
from src.optimizer import Optimizer

logger = logging.getLogger(__name__)

def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj

def main(data_dir: Optional[Union[str, Path]] = None) -> Path:
    logger.info("🚀 Güncelleme Başlatılıyor...")
    target_dir = Path(data_dir) if data_dir else Path(__file__).parent / 'data'
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Veri Yükle
    loader = DataLoader()
    
    # DINAMIK GW BILGISI
    gw_info = loader.get_next_gw()
    logger.info("📅 Hedef Hafta: %s (GW%s)", gw_info['name'], gw_info['id'])
    
    df_us, df_fpl, df_fixtures = loader.fetch_all_data()
    df_merged = loader.merge_data(df_us, df_fpl)
    df_merged = loader.fetch_player_history_batch(df_merged)
    df_fixtures = loader.process_fixtures(df_fixtures)
    
    # Fixture bilgilerini ekle (is_home, opponent_difficulty, opponent_team)
    df_merged = loader.get_fpl_data(df_merged)
    
    # 2. Modelleme
    model = FPLModel()
    df_predictions, metrics, df_validation = model.train_and_predict(df_merged)
    
    # 3. Optimizasyon
    opt = Optimizer()
    df_with_metrics = opt.calculate_metrics(df_predictions, df_fixtures)
    
    # --- METADATA KAYDI (Siteye GW bilgisini aktarmak için) ---
    with open(target_dir / 'metadata.json', 'w') as f:
        json.dump(gw_info, f)
        
    # A. Ana Veriler
    df_with_metrics.to_csv(target_dir / 'all_players.csv', index=False)
    
    # B. Model Test Verileri
    df_validation.to_csv(target_dir / 'model_validation.csv', index=False)
    metrics_serializable = _to_serializable(metrics)
    try:
        with open(target_dir / 'model_metrics.json', 'w') as f:
            json.dump(metrics_serializable, f)
    except Exception as exc:
        raise
    
    # C. Dream Teams (Genel İsimlendirme)
    # Kısa Vade (Short Term) - Dinamik GW ismi yerine sabit dosya ismi kullanıyoruz ki app.py bozulmasın
    dt_short = opt.solve_dream_team(df_with_metrics, target_metric='gw19_xP', budget=100.0)
    dt_short.to_csv(target_dir / 'dream_team_short.csv', index=False)
    
    # Uzun Vade (Long Term)
    dt_long = opt.solve_dream_team(df_with_metrics, target_metric='long_term_xP', budget=100.0)
    dt_long.to_csv(target_dir / 'dream_team_long.csv', index=False)
    
    logger.info("✅ BİTTİ! %s verileri güncellendi.", gw_info['name'])
    return target_dir

if __name__ == "__main__":
    main()
