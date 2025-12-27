import streamlit as st
import pandas as pd
import math
import json
from src.data_loader import DataLoader
from src.optimizer import Optimizer

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="FPL AI Pro 🤖", layout="wide")

# --- VERİ YÜKLEME ---
@st.cache_data
def load_files():
    try:
        all_players = pd.read_csv('data/all_players.csv')
        dt_short = pd.read_csv('data/dream_team_short.csv')
        dt_long = pd.read_csv('data/dream_team_long.csv')
        df_validation = pd.read_csv('data/model_validation.csv')
        with open('data/model_metrics.json', 'r') as f:
            metrics = json.load(f)
        with open('data/metadata.json', 'r') as f:
            meta = json.load(f)
        return all_players, dt_short, dt_long, df_validation, metrics, meta
    except:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {'name': 'GW?', 'id': 0, 'deadline': '-'}

df_all, df_short, df_long, df_val, metrics, meta = load_files()

current_gw_label = meta.get('name', 'Next GW')
next_5_gw_label = f"GW{meta.get('id', 0)} - GW{meta.get('id', 0)+4}"

st.title(f"⚽ FPL Yapay Zeka Kahini: {current_gw_label}")
st.caption(f"📅 Deadline: {meta.get('deadline', '-').replace('T', ' ')[:16]}")

# --- YAN PANEL ---
st.sidebar.header("👤 Kullanıcı Paneli")
user_id = st.sidebar.text_input("FPL Team ID (Örn: 123456)")

if not df_all.empty:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔄 Transfer Sihirbazı", 
        f"🚀 {current_gw_label} Dream Team", 
        f"🔮 Uzun Vade (5GW)", 
        "📊 Oyuncu Havuzu", 
        "🧪 Model Lab"
    ])
    
    # --- TAB 1: TRANSFER SİHİRBAZI ---
    with tab1:
        st.header("Kişisel Kadro Analizi")
        if not user_id:
            st.info("👈 Devam etmek için soldaki menüden FPL Team ID'nizi girin.")
        else:
            if st.button("Takımımı Analiz Et"):
                with st.spinner("Takım verileri FPL API'den çekiliyor..."):
                    loader = DataLoader()
                    player_ids, bank = loader.fetch_user_team(user_id)
                    
                    # HATA ÖNLEME: my_team'i her zaman tanımla
                    my_team = pd.DataFrame() 
                    
                    if player_ids:
                        # CSV'deki id sütunu ile FPL'den gelen id'leri eşle
                        my_team = df_all[df_all['id'].isin(player_ids)].copy()
                        
                        if not my_team.empty:
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Kadro Puan Beklentisi (5GW)", f"{my_team['final_5gw_xP'].sum():.1f}")
                            c2.metric("Bankadaki Para", f"£{bank}")
                            c3.metric("Kadro Değeri", f"£{my_team['price'].sum():.1f}")
                            
                            st.subheader("Mevcut Kadronuz")
                            st.dataframe(my_team[['web_name', 'position', 'team_name', 'price', 'final_5gw_xP']], use_container_width=True, hide_index=True)
                            
                            # Transfer Önerisi
                            st.divider()
                            st.subheader("🤖 Yapay Zeka Transfer Önerisi")
                            opt = Optimizer()
                            suggestion = opt.suggest_transfer(my_team, df_all, bank)
                            
                            if suggestion:
                                col_out, col_arrow, col_in = st.columns([1, 0.2, 1])
                                with col_out:
                                    st.error(f"SAT: {suggestion['out']['web_name']}")
                                    st.caption(f"Fiyat: £{suggestion['out']['price']} | xP: {suggestion['out']['final_5gw_xP']:.1f}")
                                with col_arrow:
                                    st.markdown("<h2 style='text-align: center;'>➡️</h2>", unsafe_allow_html=True)
                                with col_in:
                                    st.success(f"AL: {suggestion['in']['web_name']}")
                                    st.caption(f"Fiyat: £{suggestion['in']['price']} | xP: {suggestion['in']['final_5gw_xP']:.1f}")
                                st.info(f"📈 Bu hamle 5 haftada toplam **+{suggestion['gain']:.1f} puan** fark yaratabilir.")
                            else:
                                st.warning("Mevcut bütçenizle daha iyi bir transfer seçeneği bulunamadı.")
                        else:
                            st.error("Kadronuzdaki oyuncular veritabanımızda bulunamadı. Lütfen updater.py'yi çalıştırın.")
                    else:
                        st.error("Takım verileri çekilemedi. Team ID'nin doğruluğundan emin olun.")

    # --- DİĞER TABLAR (Kısa ve öz halleri) ---
    with tab2:
        st.header(f"🚀 {current_gw_label} En İyi 15")
        st.dataframe(df_short[['position', 'web_name', 'team_name', 'price', 'gw19_xP']], use_container_width=True, hide_index=True)

    with tab3:
        st.header(f"🔮 {next_5_gw_label} Projeksiyonu")
        st.dataframe(df_long[['position', 'web_name', 'team_name', 'price', 'long_term_xP']], use_container_width=True, hide_index=True)

    with tab4:
        st.header("🔍 Oyuncu Filtreleme")
        st.dataframe(df_all[['web_name', 'team_name', 'position', 'price', 'final_5gw_xP', 'chance_of_playing_next_round']], use_container_width=True)

    with tab5:
        st.header("📊 Model Başarımı")
        if metrics:
            m1, m2, m3 = st.columns(3)
            m1.metric("R² Başarı Oranı", f"{metrics.get('r2', 0):.3f}")
            m2.metric("RMSE (Hata Payı)", f"{metrics.get('rmse', 0):.2f}")
            m3.metric("MAE (Ort. Hata)", f"{metrics.get('mae', 0):.2f}")
            st.scatter_chart(df_val, x='Actual_Points', y='Predicted_Points', color='position')

else:
    st.error("⚠️ Veri dosyaları eksik! Lütfen sunucuda 'python3 updater.py' komutunu çalıştırın.")
