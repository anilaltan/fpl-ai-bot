import sys
from pathlib import Path

# Add project root to Python path to enable src/ imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.data_loader import DataLoader
from src.optimizer import Optimizer

print("--- 1. VERİ YÜKLENİYOR ---")
try:
    data_loader = DataLoader()
    optimizer = Optimizer()

    # Use full data pipeline like backend does
    df_understat, df_fpl, df_fixtures = data_loader.fetch_all_data()

    # Merge data
    df = data_loader.merge_data(df_understat, df_fpl)

    # Calculate metrics
    df = optimizer.calculate_metrics(df, df_fixtures)

    print(f"✅ Veri Yüklendi. Toplam Oyuncu: {len(df)}")

    # Veri setinden Arsenal örnekleri görelim
    arsenal_players = df[df['team'].astype(str).str.contains("Arsenal", case=False, na=False)]
    print(f"\n🔍 Arsenal Oyuncuları ({len(arsenal_players)} adet):")
    if not arsenal_players.empty:
        display_cols = ['name', 'team', 'position', 'price']
        available_cols = [col for col in display_cols if col in arsenal_players.columns]
        print(arsenal_players[available_cols].head(3))
    else:
        print("Arsenal oyuncuları bulunamadı")

    print("\n--- 2. OPTİMİZASYON BAŞLIYOR ---")
    optimizer = Optimizer()
    squad = optimizer.solve_dream_team(df, budget=100.0)
    
    print("\n--- 3. SONUÇLAR ---")
    if not squad.empty:
        print("✅ Kadro Kuruldu!")
        print(squad[['name', 'team', 'position']])
        
        # Takım sayımlarını kontrol et
        team_counts = squad['team'].value_counts()
        print("\n📊 TAKIM DAĞILIMI (Kritik Nokta):")
        print(team_counts)
        
        if any(team_counts > 3):
            print("\n❌ HATA: Bir takımdan 3'ten fazla oyuncu var!")
        else:
            print("\n✅ BAŞARILI: Takım limitleri korunmuş.")
    else:
        print("❌ Kadro kurulamadı (Empty Result).")

except Exception as e:
    print(f"\n❌ KRİTİK HATA: {e}")
    import traceback
    traceback.print_exc()