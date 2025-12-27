import streamlit as st
import pandas as pd

st.set_page_config(page_title="FPL AI 🔮", layout="wide")

st.title("⚽ FPL Yapay Zeka Kahini")
st.caption("Understat xG verileri ve XGBoost kullanılarak hesaplanmıştır.")

@st.cache_data
def get_data():
    try:
        return pd.read_csv('data/final_predictions.csv')
    except:
        return pd.DataFrame()

df = get_data()

if not df.empty:
    col1, col2, col3 = st.columns(3)
    team_filter = col1.multiselect("Takım", df['team_name'].unique())
    pos_filter = col2.multiselect("Pozisyon", df['position'].unique())
    price_filter = col3.slider("Max Fiyat", 4.0, 15.0, 15.0)
    
    filtered = df.copy()
    if team_filter: filtered = filtered[filtered['team_name'].isin(team_filter)]
    if pos_filter: filtered = filtered[filtered['position'].isin(pos_filter)]
    filtered = filtered[filtered['price'] <= price_filter]
    
    st.subheader("🏆 Önümüzdeki 5 Hafta İçin En İyiler")
    
    st.dataframe(
        filtered.sort_values('final_5gw_xP', ascending=False)[
            ['web_name', 'team_name', 'position', 'price', 'final_5gw_xP', 'chance_of_playing_next_round']
        ],
        column_config={
            "final_5gw_xP": st.column_config.NumberColumn("5 Haftalık Puan", format="%.2f 🔥"),
            "chance_of_playing_next_round": st.column_config.ProgressColumn("Oynama %", min_value=0, max_value=100)
        },
        use_container_width=True,
        height=800
    )
else:
    st.error("Veri bulunamadı. Lütfen updater.py'yi çalıştırın.")