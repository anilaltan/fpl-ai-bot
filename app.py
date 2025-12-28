import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import pandas as pd
import json
from pathlib import Path
from src.data_loader import DataLoader
from src.optimizer import Optimizer

# ============================================================================
# SAYFA AYARLARI
# ============================================================================
st.set_page_config(
    page_title="FPL AI Pro 🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# AUTHENTICATION (GİRİŞ SİSTEMİ)
# ============================================================================
@st.cache_resource
def load_auth_config():
    """config.yaml dosyasını yükler"""
    config_path = Path(__file__).parent / 'config.yaml'
    if not config_path.exists():
        st.error("❌ config.yaml bulunamadı! Lütfen oluşturun.")
        st.stop()
    with open(config_path) as file:
        config = yaml.load(file, Loader=SafeLoader)
    return config

config = load_auth_config()

try:
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )
except Exception as e:
    st.error('❌ Giriş sistemi başlatılamadı.')
    st.stop()

# ============================================================================
# VERİ YÜKLEME
# ============================================================================
@st.cache_data
def load_files():
    try:
        all_players = pd.read_csv('data/all_players.csv')
        
        # Hata önleyici okuma (Dosyalar henüz oluşmadıysa boş dön)
        try: dt_short = pd.read_csv('data/dream_team_short.csv')
        except: dt_short = pd.DataFrame()
        
        try: dt_long = pd.read_csv('data/dream_team_long.csv')
        except: dt_long = pd.DataFrame()
        
        try: df_validation = pd.read_csv('data/model_validation.csv')
        except: df_validation = pd.DataFrame()
        
        with open('data/model_metrics.json', 'r') as f: metrics = json.load(f)
        with open('data/metadata.json', 'r') as f: meta = json.load(f)
        
        return all_players, dt_short, dt_long, df_validation, metrics, meta
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {'name': 'GW?', 'id': 0, 'deadline': '-'}

df_all, df_short, df_long, df_val, metrics, meta = load_files()

# ============================================================================
# YARDIMCI FONKSİYONLAR
# ============================================================================
def get_user_role(username):
    try:
        user_data = config['credentials']['usernames'].get(username, {})
        # Rol 'roles' listesi içinde mi yoksa 'role' stringi mi kontrol et
        roles = user_data.get('roles', [])
        if isinstance(roles, list) and 'premium' in roles: return 'premium'
        if isinstance(roles, list) and 'admin' in roles: return 'admin'
        return user_data.get('role', 'free')
    except:
        return 'free'

def is_premium_user(username):
    role = get_user_role(username)
    return role in ['premium', 'admin']

def display_locked_feature(feature_name):
    st.warning(f"🔒 **{feature_name} Sadece Premium Üyeler İçindir**")
    st.info("Bu özelliği kullanmak için lütfen hesabınızı yükseltin.")

# ============================================================================
# ANA UYGULAMA
# ============================================================================
def main():
    # --- GİRİŞ KUTUSU ---
    login_result = authenticator.login(location='main')
    
    if isinstance(login_result, (list, tuple)) and len(login_result) == 3:
        name, authentication_status, username = login_result
    else:
        status = st.session_state.get('authentication_status')
        if status is True:
            name = st.session_state.get('name')
            username = st.session_state.get('username')
            authentication_status = True
        elif status is False:
            authentication_status = False
        else:
            authentication_status = None

    if authentication_status is False:
        st.error('❌ Kullanıcı adı veya şifre hatalı')
        st.stop()
    elif authentication_status is None:
        st.warning('👋 Lütfen giriş yapınız')
        st.stop()

    # --- GİRİŞ BAŞARILI ---
    if authentication_status:
        role = get_user_role(username)
        premium = is_premium_user(username)
        
        # Sidebar
        with st.sidebar:
            st.success(f"Hoşgeldin, {name}!")
            st.markdown(f"**Üyelik:** {role.upper()}")
            authenticator.logout(location='sidebar')
            st.divider()
            st.markdown("### 📊 Menü")
            st.markdown("- 🎯 Transfer Sihirbazı (Pro)")
            st.markdown("- 🚀 Dream Team")
            st.markdown("- 🔮 Uzun Vade")

        # Başlıklar
        st.title("⚽ FPL AI Pro")
        current_gw = meta.get('name', 'Next GW')
        st.caption(f"📅 Şu Anki Dönem: {current_gw}")

        # Sekmeler
        tabs = st.tabs([
            "🎯 Transfer Sihirbazı",
            "🚀 GW Dream Team",
            "🔮 Uzun Vade",
            "📊 Oyuncu Havuzu",
            "🧪 Model Lab"
        ])

        # --- TAB 1: TRANSFER SİHİRBAZI (PRO) ---
        with tabs[0]:
            st.header("🎯 Kişisel Transfer Sihirbazı")
            if not premium:
                display_locked_feature("Transfer Sihirbazı")
            else:
                user_id = st.text_input("FPL Team ID (Örn: 123456)")
                if user_id and st.button("Takımımı Analiz Et"):
                    with st.spinner("Analiz yapılıyor..."):
                        loader = DataLoader()
                        player_ids, bank = loader.fetch_user_team(user_id)
                        
                        if player_ids and not df_all.empty:
                            if 'id' in df_all.columns:
                                my_team = df_all[df_all['id'].isin(player_ids)].copy()
                                if not my_team.empty:
                                    st.success(f"Takım Yüklendi! Banka: £{bank}")
                                    
                                    # Takım Tablosu
                                    display_team = my_team[['web_name', 'team_name', 'position', 'price', 'final_5gw_xP']].copy()
                                    display_team.columns = ['Oyuncu', 'Takım', 'Poz', 'Fiyat', '5H Puan Beklentisi']
                                    st.dataframe(display_team, use_container_width=True)
                                    
                                    st.divider()
                                    st.subheader("🤖 Yapay Zeka Önerisi")
                                    
                                    opt = Optimizer()
                                    suggestion = opt.suggest_transfer(my_team, df_all, bank)
                                    
                                    if suggestion:
                                        c1, c2, c3 = st.columns([1, 0.2, 1])
                                        c1.error(f"SAT: {suggestion['out']['web_name']}")
                                        c2.markdown("➡️")
                                        c3.success(f"AL: {suggestion['in']['web_name']}")
                                        st.info(f"📈 Beklenen Kazanç: +{suggestion['gain']:.1f} Puan")
                                    else:
                                        st.warning("Mevcut bütçe ile daha iyi bir transfer bulunamadı.")
                                else:
                                    st.error("Takım oyuncuları veritabanında bulunamadı.")
                            else:
                                st.error("Veritabanında ID sütunu eksik.")
                        else:
                            st.error("Takım ID hatalı veya veri çekilemedi.")

        # --- TAB 2: GW DREAM TEAM ---
        with tabs[1]:
            st.header(f"🚀 {current_gw} En İyiler")
            if not df_short.empty:
                # Sütun isimlerini eşle: web_name -> Name
                display_df = df_short[['position', 'web_name', 'team_name', 'price', 'gw19_xP']].copy()
                display_df.columns = ['Poz', 'Oyuncu', 'Takım', 'Fiyat', 'Puan Beklentisi']
                st.dataframe(display_df, use_container_width=True)
                
                # Kart Görünümü (İsteğe bağlı, hata veren kısım burasıydı)
                st.divider()
                st.subheader("İlk 11 Kadrosu")
                cols = st.columns(4)
                for idx, row in df_short.iterrows():
                    # Modulo ile sütunlara dağıt
                    with cols[idx % 4]:
                        st.markdown(f"**{row['web_name']}**")
                        st.caption(f"{row['team_name']} | {row['position']}")
                        st.markdown(f"⭐ {row['gw19_xP']:.1f}")
            else:
                st.info("Kısa vadeli veriler henüz oluşmadı.")

        # --- TAB 3: UZUN VADE ---
        with tabs[2]:
            st.header("🔮 5 Haftalık Projeksiyon")
            if not df_long.empty:
                display_df = df_long[['position', 'web_name', 'team_name', 'price', 'long_term_xP']].copy()
                display_df.columns = ['Poz', 'Oyuncu', 'Takım', 'Fiyat', '5H Puan']
                st.dataframe(display_df, use_container_width=True)
            else:
                st.info("Uzun vadeli veriler henüz oluşmadı.")

        # --- TAB 4: OYUNCU HAVUZU ---
        with tabs[3]:
            st.header("📊 Oyuncu Havuzu")
            if not df_all.empty:
                display_df = df_all[['web_name', 'team_name', 'position', 'price', 'final_5gw_xP']].copy()
                display_df.columns = ['Oyuncu', 'Takım', 'Poz', 'Fiyat', '5H Puan']
                st.dataframe(display_df, use_container_width=True)

        # --- TAB 5: MODEL LAB (PRO) ---
        with tabs[4]:
            st.header("🧪 Model Laboratuvarı")
            if not premium:
                display_locked_feature("Model Lab")
            else:
                if metrics:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("R2 Skoru", f"{metrics.get('r2', 0):.3f}")
                    c2.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
                    c3.metric("MAE", f"{metrics.get('mae', 0):.2f}")
                    
                    if not df_val.empty:
                        st.subheader("Tahmin vs Gerçek")
                        # Model.py çıktısı Büyük Harf kullanıyor: Actual_Points, Predicted_Points
                        chart_data = df_val[['Actual_Points', 'Predicted_Points']].copy()
                        st.scatter_chart(chart_data, x='Predicted_Points', y='Actual_Points')
                else:
                    st.info("Model metrikleri bulunamadı.")

if __name__ == "__main__":
    main()