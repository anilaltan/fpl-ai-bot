import streamlit as st
import pandas as pd
import json
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from src.data_loader import DataLoader
from src.optimizer import Optimizer

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="FPL AI Pro 🤖", layout="wide")

# --- AUTHENTICATION (Giriş Sistemi) ---
try:
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
except FileNotFoundError:
    st.error("⚠️ 'config.yaml' dosyası bulunamadı. Lütfen oluşturun.")
    st.stop()

# Authenticator Kurulumu
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# --- SIDEBAR & GİRİŞ ---
with st.sidebar:
    st.header("🔐 Kullanıcı Girişi")
    # Login kutusu
    name, authentication_status, username = authenticator.login('main')
    
    if authentication_status:
        st.success(f"Hoşgeldin, {name}!")
        
        # Rol Kontrolü (Free vs Premium)
        user_roles = config['credentials']['usernames'][username].get('roles', [])
        # config.yaml yapısına göre role ya string ya da liste olabilir, kontrol ediyoruz:
        if isinstance(user_roles, list):
            is_premium = 'premium' in user_roles
        else:
            is_premium = user_roles == 'premium'
        
        if is_premium:
            st.markdown("🌟 **PREMIUM ÜYE**")
        else:
            st.markdown("👤 **Standart Üye**")
            
        authenticator.logout('Çıkış Yap', 'sidebar')
        
    elif authentication_status is False:
        st.error('Kullanıcı adı veya şifre hatalı')
    elif authentication_status is None:
        st.info('Lütfen giriş yapınız')

# --- ANA UYGULAMA ---
st.title("⚽ FPL Yapay Zeka Kahini")

# --- VERİ YÜKLEME ---
@st.cache_data
def load_files():
    try:
        all_players = pd.read_csv('data/all_players.csv')
        # Dosyalar yoksa hata vermemesi için kontrol
        try: dt_short = pd.read_csv('data/dream_team_short.csv')
        except: dt_short = pd.DataFrame()
        
        try: dt_long = pd.read_csv('data/dream_team_long.csv')
        except: dt_long = pd.DataFrame()
        
        try: df_validation = pd.read_csv('data/model_validation.csv')
        except: df_validation = pd.DataFrame()
        
        with open('data/model_metrics.json', 'r') as f:
            metrics = json.load(f)
        with open('data/metadata.json', 'r') as f:
            meta = json.load(f)
        return all_players, dt_short, dt_long, df_validation, metrics, meta
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {'name': 'GW?', 'id': 0, 'deadline': '-'}

df_all, df_short, df_long, df_val, metrics, meta = load_files()
current_gw_label = meta.get('name', 'Next GW')

# Veri kontrolü
if df_all.empty:
    st.warning("⚠️ Veriler yüklenemedi. Lütfen sunucuda `python3 updater.py` komutunu çalıştırarak verileri oluşturun.")
    st.stop()

# --- SEKMELER (Giriş Durumuna Göre) ---
# Temel sekmeler
tab_titles = [
    "🏆 GW Dream Team", 
    "🔮 Uzun Vade", 
    "📊 Oyuncu Havuzu"
]

# Premium sekmeler
if authentication_status:
    tab_titles.insert(0, "🔄 Transfer Sihirbazı (PRO)")
    tab_titles.append("🧪 Model Lab (PRO)")
else:
    tab_titles.insert(0, "🔒 Transfer Sihirbazı")
    tab_titles.append("🔒 Model Lab")

tabs = st.tabs(tab_titles)

# --- TAB: TRANSFER SİHİRBAZI (Index 0) ---
with tabs[0]:
    if authentication_status: # Giriş yapmış mı?
        st.header("Kişisel Kadro Analizi")
        user_id = st.text_input("FPL Team ID (Örn: 123456)")
        
        if user_id and st.button("Takımımı Analiz Et"):
            # DÜZELTME BURADA: Sınıf yapısını doğru kullanıyoruz
            with st.spinner("Takım verileri çekiliyor..."):
                loader = DataLoader()
                player_ids, bank = loader.fetch_user_team(user_id)
                
                my_team = pd.DataFrame()
                if player_ids:
                    if 'id' in df_all.columns:
                        my_team = df_all[df_all['id'].isin(player_ids)].copy()
                    
                    if not my_team.empty:
                        st.success(f"Takım Bulundu! Banka: £{bank}")
                        st.dataframe(my_team[['web_name', 'position', 'price', 'final_5gw_xP']], use_container_width=True)
                        
                        st.divider()
                        st.subheader("🤖 Yapay Zeka Transfer Önerisi")
                        
                        # DÜZELTME BURADA: Optimizer sınıfını kullanıyoruz
                        opt = Optimizer()
                        suggestion = opt.suggest_transfer(my_team, df_all, bank)
                        
                        if suggestion:
                            c1, c2, c3 = st.columns([1,0.2,1])
                            c1.error(f"SAT: {suggestion['out']['web_name']}")
                            c2.markdown("<h2 style='text-align: center;'>➡️</h2>", unsafe_allow_html=True)
                            c3.success(f"AL: {suggestion['in']['web_name']}")
                            st.info(f"📈 Beklenen Kazanç: +{suggestion['gain']:.1f} Puan")
                        else:
                            st.warning("Mevcut bütçe ile daha iyi bir transfer önerisi bulunamadı.")
                    else:
                        st.error("Oyuncular veritabanında bulunamadı. Lütfen verileri güncelleyin.")
                else:
                    st.error("Takım ID hatalı veya bu hafta için kadro kurulamamış.")
    else:
        # GİRİŞ YAPMAMIŞSA
        st.warning("⚠️ Bu özellik sadece üyeler içindir.")
        st.info("Lütfen soldaki panelden giriş yapınız. (Test hesabı: testuser / şifre: 123)")

# --- TAB: GW DREAM TEAM (Index 1) ---
with tabs[1]:
    st.header(f"🚀 {current_gw_label} En İyiler")
    if not df_short.empty:
        st.dataframe(df_short[['position', 'web_name', 'team_name', 'price', 'gw19_xP']], use_container_width=True)
    else:
        st.info("Kısa vadeli veriler henüz oluşmadı.")

# --- TAB: UZUN VADE (Index 2) ---
with tabs[2]:
    st.header("🔮 5 Haftalık Projeksiyon")
    if not df_long.empty:
        st.dataframe(df_long[['position', 'web_name', 'team_name', 'price', 'long_term_xP']], use_container_width=True)
    else:
        st.info("Uzun vadeli veriler henüz oluşmadı.")

# --- TAB: OYUNCU HAVUZU (Index 3) ---
with tabs[3]:
    st.header("📊 Tüm Oyuncular")
    st.dataframe(df_all[['web_name', 'team_name', 'position', 'price', 'final_5gw_xP']], use_container_width=True)

# --- TAB: MODEL LAB (Index 4) ---
with tabs[4]:
    if authentication_status:
        # Sadece PREMIUM üyelere özel
        if is_premium:
            st.header("🧪 Model Laboratuvarı")
            if metrics:
                c1, c2 = st.columns(2)
                c1.metric("R2 Skoru", f"{metrics.get('r2', 0):.3f}")
                c2.metric("Hata Payı (RMSE)", f"{metrics.get('rmse', 0):.2f}")
                if not df_val.empty:
                    st.scatter_chart(df_val, x='Actual_Points', y='Predicted_Points')
            else:
                st.info("Model metrikleri henüz hesaplanmadı.")
        else:
            st.warning("Bu alan sadece **PREMIUM** üyeler içindir.")
            st.info("Mevcut Paketiniz: Free Plan")
    else:
        st.error("Lütfen giriş yapınız.")