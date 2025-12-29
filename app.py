import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

from src.data_loader import DataLoader
from src.optimizer import Optimizer

logger = logging.getLogger(__name__)

# ============================================================================
# FUTURISTIC CUSTOM CSS - INSPIRED BY REFERENCE DESIGN
# ============================================================================
def load_custom_css() -> None:
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
    
    /* Main background - Dark theme */
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1729 100%);
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Sidebar - Dark panel */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 2px solid rgba(0, 255, 159, 0.1);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Futuristic card styling */
    .player-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%);
        border: 2px solid rgba(0, 255, 159, 0.3);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 255, 159, 0.15),
                    inset 0 1px 0 rgba(255, 255, 255, 0.05);
        margin: 10px 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .player-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, 
            transparent 0%, 
            rgba(0, 255, 159, 0.8) 50%, 
            transparent 100%);
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .player-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 48px rgba(0, 255, 159, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border-color: rgba(0, 255, 159, 0.6);
    }
    
    .player-card:hover::before {
        opacity: 1;
    }
    
    /* Pitch container - 3D style */
    .pitch-container {
        background: linear-gradient(135deg, #0a192f 0%, #1a2332 100%);
        border-radius: 20px;
        padding: 40px;
        margin: 30px 0;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5),
                    inset 0 1px 0 rgba(255, 255, 255, 0.05);
        border: 2px solid rgba(0, 255, 159, 0.2);
        position: relative;
        perspective: 1000px;
    }
    
    .pitch-container::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, 
            rgba(0, 255, 159, 0.3) 0%, 
            rgba(0, 191, 255, 0.3) 50%, 
            rgba(0, 255, 159, 0.3) 100%);
        border-radius: 20px;
        z-index: -1;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .pitch-container:hover::before {
        opacity: 1;
    }
    
    /* Transfer comparison cards - Neon style */
    .transfer-out {
        background: linear-gradient(135deg, rgba(255, 59, 92, 0.15) 0%, rgba(255, 107, 107, 0.1) 100%);
        border: 2px solid rgba(255, 59, 92, 0.5);
        border-radius: 16px;
        padding: 25px;
        color: #ffffff;
        box-shadow: 0 8px 32px rgba(255, 59, 92, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.05);
        position: relative;
        overflow: hidden;
    }
    
    .transfer-out::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 59, 92, 0.2), 
            transparent);
        transition: left 0.5s;
    }
    
    .transfer-out:hover::after {
        left: 100%;
    }
    
    .transfer-in {
        background: linear-gradient(135deg, rgba(0, 255, 159, 0.15) 0%, rgba(0, 255, 159, 0.1) 100%);
        border: 2px solid rgba(0, 255, 159, 0.5);
        border-radius: 16px;
        padding: 25px;
        color: #ffffff;
        box-shadow: 0 8px 32px rgba(0, 255, 159, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.05);
        position: relative;
        overflow: hidden;
    }
    
    .transfer-in::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(0, 255, 159, 0.2), 
            transparent);
        transition: left 0.5s;
    }
    
    .transfer-in:hover::after {
        left: 100%;
    }
    
    /* Position badges - Neon glow */
    .position-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        margin: 5px 0;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-family: 'Orbitron', sans-serif;
        box-shadow: 0 0 20px currentColor;
    }
    
    .pos-GKP { 
        background: rgba(255, 193, 7, 0.2);
        color: #ffd43b;
        border: 1px solid #ffd43b;
    }
    .pos-DEF { 
        background: rgba(0, 255, 159, 0.2);
        color: #00ff9f;
        border: 1px solid #00ff9f;
    }
    .pos-MID { 
        background: rgba(0, 191, 255, 0.2);
        color: #00bfff;
        border: 1px solid #00bfff;
    }
    .pos-FWD { 
        background: rgba(255, 59, 92, 0.2);
        color: #ff3b5c;
        border: 1px solid #ff3b5c;
    }
    
    /* Metric container - Glassmorphism */
    .metric-container {
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 255, 159, 0.2);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.3s;
    }
    
    .metric-container:hover {
        border-color: rgba(0, 255, 159, 0.5);
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(0, 255, 159, 0.2);
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }
    
    .metric-value {
        color: #00ff9f;
        font-size: 32px;
        font-weight: 900;
        font-family: 'Orbitron', sans-serif;
        text-shadow: 0 0 20px rgba(0, 255, 159, 0.5);
    }
    
    /* Headers - Futuristic style */
    h1, h2, h3 {
        color: #ffffff;
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        letter-spacing: 2px;
        text-shadow: 0 0 30px rgba(0, 255, 159, 0.5);
    }
    
    h1 {
        font-size: 48px;
        background: linear-gradient(135deg, #00ff9f 0%, #00bfff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Button styling - Neon glow */
    .stButton>button {
        background: linear-gradient(135deg, rgba(0, 255, 159, 0.2) 0%, rgba(0, 191, 255, 0.2) 100%);
        color: #00ff9f;
        border: 2px solid #00ff9f;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 700;
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: all 0.3s;
        box-shadow: 0 0 20px rgba(0, 255, 159, 0.3);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, rgba(0, 255, 159, 0.3) 0%, rgba(0, 191, 255, 0.3) 100%);
        transform: translateY(-3px);
        box-shadow: 0 0 40px rgba(0, 255, 159, 0.6);
        border-color: #00bfff;
        color: #00bfff;
    }
    
    /* Tab styling - Modern */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(15, 23, 42, 0.5);
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
        padding: 12px 24px;
        color: #94a3b8;
        border: 1px solid transparent;
        font-weight: 600;
        font-family: 'Rajdhani', sans-serif;
        transition: all 0.3s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(30, 41, 59, 0.8);
        color: #00ff9f;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 255, 159, 0.2) 0%, rgba(0, 191, 255, 0.2) 100%);
        border-color: #00ff9f;
        color: #00ff9f;
        box-shadow: 0 0 20px rgba(0, 255, 159, 0.3);
    }
    
    /* Data tables - Dark theme */
    .dataframe {
        background: rgba(15, 23, 42, 0.7) !important;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(0, 255, 159, 0.2);
    }
    
    /* Text inputs - Neon style */
    .stTextInput>div>div>input {
        background: rgba(15, 23, 42, 0.7);
        border: 2px solid rgba(0, 255, 159, 0.3);
        border-radius: 8px;
        color: #ffffff;
        font-family: 'Rajdhani', sans-serif;
        padding: 12px;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #00ff9f;
        box-shadow: 0 0 20px rgba(0, 255, 159, 0.4);
    }
    
    /* Sidebar items */
    .sidebar-nav {
        background: rgba(30, 41, 59, 0.5);
        border-left: 3px solid transparent;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 8px;
        color: #94a3b8;
        transition: all 0.3s;
    }
    
    .sidebar-nav:hover {
        background: rgba(30, 41, 59, 0.8);
        border-left-color: #00ff9f;
        color: #00ff9f;
        transform: translateX(5px);
    }
    
    /* Welcome panel */
    .welcome-panel {
        background: linear-gradient(135deg, rgba(0, 255, 159, 0.1) 0%, rgba(0, 191, 255, 0.1) 100%);
        border: 2px solid rgba(0, 255, 159, 0.3);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0, 255, 159, 0.2);
    }
    
    /* Stats card on the side */
    .stats-card {
        background: rgba(15, 23, 42, 0.9);
        border: 2px solid rgba(0, 255, 159, 0.3);
        border-radius: 12px;
        padding: 16px;
        margin: 10px 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    
    /* Glowing text effect */
    .glow-text {
        color: #00ff9f;
        text-shadow: 0 0 10px rgba(0, 255, 159, 0.8),
                     0 0 20px rgba(0, 255, 159, 0.6),
                     0 0 30px rgba(0, 255, 159, 0.4);
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
    }
    
    /* Locked feature - Premium style */
    .locked-feature {
        background: linear-gradient(135deg, rgba(138, 43, 226, 0.15) 0%, rgba(75, 0, 130, 0.1) 100%);
        border: 2px solid rgba(138, 43, 226, 0.5);
        border-radius: 16px;
        padding: 40px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(138, 43, 226, 0.3);
    }
    
    /* Animations */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(0, 255, 159, 0.3);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 255, 159, 0.5);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(0, 255, 159, 0.2);
        border-radius: 8px;
        color: #00ff9f;
        font-weight: 600;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(30, 41, 59, 0.8);
        border-color: rgba(0, 255, 159, 0.4);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: rgba(0, 255, 159, 0.1);
        border-left: 4px solid #00ff9f;
        color: #00ff9f;
    }
    
    .stError {
        background: rgba(255, 59, 92, 0.1);
        border-left: 4px solid #ff3b5c;
        color: #ff3b5c;
    }
    
    .stInfo {
        background: rgba(0, 191, 255, 0.1);
        border-left: 4px solid #00bfff;
        color: #00bfff;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS FOR UI COMPONENTS
# ============================================================================
def render_player_card(player: pd.Series, show_comparison: bool = False) -> str:
    """Render a futuristic player card with neon glow"""
    position_class = f"pos-{player.get('position', 'MID')}"
    
    expected_points = player.get('final_5gw_xP', player.get('gw19_xP', player.get('long_term_xP', 0)))
    
    card_html = f"""
    <div class="player-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 style="margin: 0; color: #ffffff; font-size: 20px;">{player.get('web_name', 'Unknown')}</h3>
                <p style="margin: 8px 0; color: #94a3b8; font-size: 14px;">{player.get('team_name', 'N/A')}</p>
                <span class="position-badge {position_class}">{player.get('position', 'N/A')}</span>
            </div>
            <div style="text-align: right;">
                <div class="glow-text" style="font-size: 36px; margin: 0;">{expected_points:.1f}</div>
                <p style="margin: 8px 0 0 0; color: #00bfff; font-weight: 700; font-size: 18px;">£{player.get('price', 0):.1f}m</p>
            </div>
        </div>
    </div>
    """
    return card_html

def render_transfer_comparison(out_player: pd.Series, in_player: pd.Series, gain: float) -> None:
    """Render transfer comparison with neon-styled cards"""
    col1, col2, col3 = st.columns([5, 1, 5])
    
    with col1:
        st.markdown(f"""
        <div class="transfer-out">
            <h3 style="margin: 0 0 20px 0; font-family: 'Orbitron', sans-serif;">❌ TRANSFER OUT</h3>
            <h2 style="margin: 10px 0; font-size: 28px;">{out_player.get('web_name', 'Unknown')}</h2>
            <p style="margin: 8px 0; opacity: 0.9; font-size: 16px;">{out_player.get('team_name', 'N/A')} • {out_player.get('position', 'N/A')}</p>
            <div style="margin-top: 24px; display: flex; justify-content: space-between;">
                <div>
                    <p style="margin: 5px 0; font-size: 12px; opacity: 0.7; text-transform: uppercase; letter-spacing: 1px;">Price</p>
                    <p style="margin: 0; font-size: 28px; font-weight: 900; font-family: 'Orbitron', sans-serif;">£{out_player.get('price', 0):.1f}m</p>
                </div>
                <div>
                    <p style="margin: 5px 0; font-size: 12px; opacity: 0.7; text-transform: uppercase; letter-spacing: 1px;">xPoints</p>
                    <p style="margin: 0; font-size: 28px; font-weight: 900; font-family: 'Orbitron', sans-serif;">{out_player.get('final_5gw_xP', 0):.1f}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding-top: 80px;'>
            <div style='font-size: 48px; color: #00ff9f; text-shadow: 0 0 20px rgba(0, 255, 159, 0.8);'>➡️</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="transfer-in">
            <h3 style="margin: 0 0 20px 0; font-family: 'Orbitron', sans-serif;">✅ TRANSFER IN</h3>
            <h2 style="margin: 10px 0; font-size: 28px;">{in_player.get('web_name', 'Unknown')}</h2>
            <p style="margin: 8px 0; opacity: 0.9; font-size: 16px;">{in_player.get('team_name', 'N/A')} • {in_player.get('position', 'N/A')}</p>
            <div style="margin-top: 24px; display: flex; justify-content: space-between;">
                <div>
                    <p style="margin: 5px 0; font-size: 12px; opacity: 0.7; text-transform: uppercase; letter-spacing: 1px;">Price</p>
                    <p style="margin: 0; font-size: 28px; font-weight: 900; font-family: 'Orbitron', sans-serif;">£{in_player.get('price', 0):.1f}m</p>
                </div>
                <div>
                    <p style="margin: 5px 0; font-size: 12px; opacity: 0.7; text-transform: uppercase; letter-spacing: 1px;">xPoints</p>
                    <p style="margin: 0; font-size: 28px; font-weight: 900; font-family: 'Orbitron', sans-serif;">{in_player.get('final_5gw_xP', 0):.1f}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(0, 255, 159, 0.2) 0%, rgba(0, 191, 255, 0.2) 100%); 
                border: 2px solid rgba(0, 255, 159, 0.5);
                border-radius: 16px; padding: 24px; text-align: center; 
                margin-top: 24px; box-shadow: 0 8px 32px rgba(0, 255, 159, 0.3);">
        <h3 style="margin: 0; color: #00ff9f; font-family: 'Orbitron', sans-serif; 
                   text-shadow: 0 0 20px rgba(0, 255, 159, 0.8);">
            📈 EXPECTED GAIN: <span style="font-size: 32px;">+{gain:.1f}</span> POINTS
        </h3>
    </div>
    """, unsafe_allow_html=True)

def render_pitch_view(df_team: pd.DataFrame) -> None:
    """Render team in a futuristic 3D pitch formation"""
    st.markdown("""
    <div class="pitch-container">
        <h3 style="text-align: center; margin-bottom: 30px; color: #00ff9f; font-family: 'Orbitron', sans-serif; 
                   font-size: 24px; text-transform: uppercase; letter-spacing: 3px;">
            ⚽ STARTING XI FORMATION
        </h3>
    """, unsafe_allow_html=True)
    
    positions = ['GKP', 'DEF', 'MID', 'FWD']
    position_names = {
        'GKP': '🥅 GOALKEEPER', 
        'DEF': '🛡️ DEFENDERS', 
        'MID': '⚡ MIDFIELDERS', 
        'FWD': '⚔️ FORWARDS'
    }
    
    for pos in positions:
        pos_players = df_team[df_team['position'] == pos]
        if not pos_players.empty:
            st.markdown(f"""
            <div style="margin: 20px 0;">
                <h4 style="color: #00bfff; font-family: 'Orbitron', sans-serif; 
                           text-align: center; text-transform: uppercase; letter-spacing: 2px;">
                    {position_names.get(pos, pos)}
                </h4>
            </div>
            """, unsafe_allow_html=True)
            
            cols = st.columns(len(pos_players))
            for idx, (_, player) in enumerate(pos_players.iterrows()):
                with cols[idx]:
                    st.markdown(render_player_card(player), unsafe_allow_html=True)
            
            st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def render_metric_card(label: str, value: Any, delta: Any = None, delta_color: str = "normal") -> None:
    """Render a futuristic metric card with glassmorphism"""
    delta_html = ""
    if delta:
        color = "#00ff9f" if delta_color == "normal" else "#ff3b5c"
        delta_html = f'<p style="margin: 8px 0 0 0; color: {color}; font-weight: 700; font-size: 14px;">{delta}</p>'
    
    st.markdown(f"""
    <div class="metric-container">
        <p class="metric-label">{label}</p>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="FPL AI Pro 🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "FPL AI Pro - AI-Powered Fantasy Premier League Analytics"
    }
)

load_custom_css()

# ============================================================================
# AUTHENTICATION
# ============================================================================
@st.cache_resource
def load_auth_config() -> Dict[str, Any]:
    """Load config.yaml file"""
    config_path = Path(__file__).parent / 'config.yaml'
    if not config_path.exists():
        st.error("❌ config.yaml not found! Please create it.")
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
    st.error('❌ Authentication system failed to initialize.')
    st.stop()

# ============================================================================
# DATA LOADING
# ============================================================================
DATA_DIR = Path(__file__).parent / 'data'
REQUIRED_DATA_FILES = [
    'all_players.csv',
    'dream_team_short.csv',
    'dream_team_long.csv',
    'model_validation.csv',
    'model_metrics.json',
    'metadata.json'
]

def ensure_data_files() -> Tuple[bool, str]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    missing = [f for f in REQUIRED_DATA_FILES if not (DATA_DIR / f).exists()]
    
    if missing:
        try:
            from updater import main as refresh_data
            refresh_data(data_dir=DATA_DIR)
        except Exception as exc:
            logger.error("Automatic data refresh failed: %s", str(exc), exc_info=True)
            return False, str(exc)
    
    remaining = [f for f in REQUIRED_DATA_FILES if not (DATA_DIR / f).exists()]
    if remaining:
        return False, f"Missing files after refresh: {', '.join(remaining)}"
    
    return True, ""

@st.cache_data
def load_files() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    ready, error_msg = ensure_data_files()
    if not ready:
        st.error("❌ Veri dosyaları bulunamadı ve otomatik güncelleme başarısız oldu. Lütfen sunucuda `python updater.py` komutunu çalıştırın.")
        logger.error("Data files missing: %s", error_msg)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {'name': 'GW?', 'id': 0, 'deadline': '-'}
    
    try:
        all_players = pd.read_csv(DATA_DIR / 'all_players.csv')
        
        try: dt_short = pd.read_csv(DATA_DIR / 'dream_team_short.csv')
        except: dt_short = pd.DataFrame()
        
        try: dt_long = pd.read_csv(DATA_DIR / 'dream_team_long.csv')
        except: dt_long = pd.DataFrame()
        
        try: df_validation = pd.read_csv(DATA_DIR / 'model_validation.csv')
        except: df_validation = pd.DataFrame()
        
        with open(DATA_DIR / 'model_metrics.json', 'r') as f: metrics = json.load(f)
        with open(DATA_DIR / 'metadata.json', 'r') as f: meta = json.load(f)
        
        return all_players, dt_short, dt_long, df_validation, metrics, meta
    except Exception as exc:
        logger.error("Data load failed: %s", str(exc), exc_info=True)
        st.error("❌ Veri dosyaları yüklenemedi. Lütfen tekrar deneyin veya `python updater.py` çalıştırın.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {'name': 'GW?', 'id': 0, 'deadline': '-'}

df_all, df_short, df_long, df_val, metrics, meta = load_files()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_user_role(username: str) -> str:
    try:
        user_data = config['credentials']['usernames'].get(username, {})
        roles = user_data.get('roles', [])
        if isinstance(roles, list) and 'premium' in roles: return 'premium'
        if isinstance(roles, list) and 'admin' in roles: return 'admin'
        return user_data.get('role', 'free')
    except:
        return 'free'

def is_premium_user(username: str) -> bool:
    role = get_user_role(username)
    return role in ['premium', 'admin']

def display_locked_feature(feature_name: str) -> None:
    st.markdown(f"""
    <div class="locked-feature">
        <div style="font-size: 64px; margin-bottom: 20px;">🔒</div>
        <h2 style="margin: 0 0 15px 0; color: #a855f7; font-family: 'Orbitron', sans-serif;">
            {feature_name}
        </h2>
        <p style="margin: 0; color: #cbd5e1; font-size: 18px; line-height: 1.6;">
            This premium feature is available exclusively for <span style="color: #a855f7; font-weight: 700;">PRO</span> members.<br>
            Unlock advanced AI-powered insights and take your FPL game to the next level!
        </p>
        <div style="margin-top: 24px;">
            <span style="background: linear-gradient(135deg, #a855f7 0%, #6366f1 100%);
                        padding: 12px 32px; border-radius: 8px; color: white; font-weight: 700;
                        text-transform: uppercase; letter-spacing: 2px; font-family: 'Orbitron', sans-serif;">
                ⭐ UPGRADE NOW
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main() -> None:
    # --- LOGIN ---
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
        st.error('❌ Username or password is incorrect')
        st.stop()
    elif authentication_status is None:
        st.warning('👋 Please login to continue')
        st.stop()

    # --- AUTHENTICATED ---
    if authentication_status:
        role = get_user_role(username)
        premium = is_premium_user(username)
        
        # Sidebar
        with st.sidebar:
            st.markdown(f"""
            <div class="welcome-panel">
                <div style="text-align: center;">
                    <div style="font-size: 64px; margin-bottom: 12px;">🤖</div>
                    <h3 style="margin: 0; color: #94a3b8; font-size: 14px; text-transform: uppercase; letter-spacing: 2px;">
                        Welcome back
                    </h3>
                    <h2 class="glow-text" style="margin: 8px 0; font-size: 28px;">
                        {name}
                    </h2>
                    <div style="margin-top: 12px;">
                        <span style="background: {'linear-gradient(135deg, #a855f7 0%, #6366f1 100%)' if premium else 'rgba(148, 163, 184, 0.2)'}; 
                                     padding: 8px 20px; border-radius: 20px; 
                                     color: white; font-weight: 700;
                                     text-transform: uppercase; letter-spacing: 1px;
                                     font-size: 12px; font-family: 'Orbitron', sans-serif;">
                            {role.upper()} {'⭐' if premium else ''}
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            authenticator.logout(location='sidebar')
            
            st.divider()
            
            st.markdown("""
            <div style="margin: 20px 0;">
                <h3 style="color: #00ff9f; font-family: 'Orbitron', sans-serif; 
                           font-size: 16px; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 16px;">
                    📊 NAVIGATION
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            nav_items = [
                ("🎯", "Transfer Wizard", "AI-powered suggestions"),
                ("🚀", "Dream Team", "Best XI for this GW"),
                ("🔮", "Long Term", "5-week projections"),
                ("📊", "Player Pool", "Full database"),
                ("🧪", "Model Lab", "Performance metrics")
            ]
            
            for icon, title, desc in nav_items:
                st.markdown(f"""
                <div class="sidebar-nav">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span style="font-size: 24px;">{icon}</span>
                        <div>
                            <div style="font-weight: 700; font-size: 14px;">{title}</div>
                            <div style="font-size: 11px; opacity: 0.7;">{desc}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            if not premium:
                st.divider()
                st.markdown("""
                <div style="background: rgba(168, 85, 247, 0.1); border-radius: 12px; 
                            padding: 20px; border: 1px solid rgba(168, 85, 247, 0.3);">
                    <div style="text-align: center;">
                        <div style="font-size: 48px; margin-bottom: 12px;">⭐</div>
                        <p style="margin: 0; color: #a855f7; font-weight: 700; font-size: 16px;">
                            Upgrade to Premium
                        </p>
                        <p style="margin: 8px 0 0 0; font-size: 12px; color: #cbd5e1;">
                            Unlock Transfer Wizard & Model Lab
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Header
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); backdrop-filter: blur(10px);
                    border: 2px solid rgba(0, 255, 159, 0.3); border-radius: 20px; 
                    padding: 40px; box-shadow: 0 8px 32px rgba(0, 255, 159, 0.2); 
                    margin-bottom: 40px; text-align: center;">
            <h1 style="margin: 0 0 12px 0;">⚽ FPL AI PRO</h1>
            <p style="margin: 0; color: #94a3b8; font-size: 18px; font-weight: 600;">
                <span style="color: #00ff9f;">●</span> Powered by Advanced Machine Learning
                <span style="margin: 0 16px;">|</span>
                <span style="color: #00bfff;">{meta.get('name', 'Next GW')}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Tabs
        tabs = st.tabs([
            "🎯 TRANSFER WIZARD",
            "🚀 DREAM TEAM",
            "🔮 LONG TERM",
            "📊 PLAYER POOL",
            "🧪 MODEL LAB"
        ])

        # --- TAB 1: TRANSFER WIZARD ---
        with tabs[0]:
            st.markdown("""
            <h2 style="color: #00ff9f; font-family: 'Orbitron', sans-serif; 
                       text-align: center; margin-bottom: 30px; font-size: 32px;">
                🎯 AI TRANSFER WIZARD
            </h2>
            """, unsafe_allow_html=True)
            
            if not premium:
                display_locked_feature("Transfer Wizard")
            else:
                st.markdown("""
                <div style="background: rgba(30, 41, 59, 0.5); border-radius: 12px; 
                            padding: 24px; margin-bottom: 30px; border: 1px solid rgba(0, 255, 159, 0.2);">
                    <p style="margin: 0; color: #cbd5e1; text-align: center; font-size: 16px;">
                        Enter your FPL Team ID to get personalized transfer recommendations powered by our AI engine.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    user_id_input = st.text_input(
                        "FPL Team ID",
                        placeholder="Enter your FPL Team ID (e.g., 123456)",
                        label_visibility="collapsed"
                    )
                with col2:
                    analyze_btn = st.button("🔍 ANALYZE", use_container_width=True)
                
                if analyze_btn:
                    team_id = user_id_input.strip()
                    
                    if not team_id:
                        st.error("❌ Please enter your FPL Team ID.")
                    elif not team_id.isdigit():
                        st.error("❌ Team ID should contain digits only.")
                    elif df_all.empty:
                        st.error("❌ Player database is empty. Please refresh data files.")
                    else:
                        with st.spinner("🤖 AI is analyzing your team..."):
                            loader = DataLoader()
                            player_ids, bank = loader.fetch_user_team(team_id)
                            
                            if player_ids and not df_all.empty:
                                if 'id' in df_all.columns:
                                    my_team = df_all[df_all['id'].isin(player_ids)].copy()
                                    if not my_team.empty:
                                        st.success(f"✅ Team loaded successfully!")
                                        
                                        # Display bank
                                        st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            render_metric_card("TEAM VALUE", f"£{my_team['price'].sum():.1f}m")
                                        with col2:
                                            render_metric_card("BANK BALANCE", f"£{bank}m")
                                        with col3:
                                            render_metric_card("SQUAD SIZE", f"{len(my_team)}")
                                        
                                        st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
                                        
                                        # Team table
                                        with st.expander("📋 VIEW FULL SQUAD", expanded=True):
                                            display_team = my_team[['web_name', 'team_name', 'position', 'price', 'final_5gw_xP']].copy()
                                            display_team = display_team.sort_values('final_5gw_xP', ascending=False)
                                            display_team.columns = ['Player', 'Club', 'Position', 'Price (£m)', '5GW xPoints']
                                            st.dataframe(
                                                display_team,
                                                use_container_width=True,
                                                hide_index=True
                                            )
                                        
                                        st.divider()
                                        st.markdown("""
                                        <h2 style="color: #00ff9f; font-family: 'Orbitron', sans-serif; 
                                                   text-align: center; margin: 40px 0; font-size: 28px;">
                                            🤖 AI RECOMMENDATION
                                        </h2>
                                        """, unsafe_allow_html=True)
                                        
                                        opt = Optimizer()
                                        suggestion = opt.suggest_transfer(my_team, df_all, bank)
                                        
                                        if suggestion:
                                            render_transfer_comparison(
                                                suggestion['out'],
                                                suggestion['in'],
                                                suggestion['gain']
                                            )
                                        else:
                                            st.markdown("""
                                            <div style="background: linear-gradient(135deg, rgba(0, 191, 255, 0.15) 0%, rgba(0, 255, 159, 0.15) 100%); 
                                                        border: 2px solid rgba(0, 255, 159, 0.5);
                                                        border-radius: 16px; padding: 40px; text-align: center;
                                                        box-shadow: 0 8px 32px rgba(0, 255, 159, 0.2);">
                                                <div style="font-size: 64px; margin-bottom: 20px;">✨</div>
                                                <h3 style="margin: 0; color: #00ff9f; font-family: 'Orbitron', sans-serif; font-size: 28px;">
                                                    YOUR TEAM IS OPTIMIZED!
                                                </h3>
                                                <p style="margin: 16px 0 0 0; color: #cbd5e1; font-size: 16px;">
                                                    No better transfers found within your current budget.
                                                </p>
                                            </div>
                                            """, unsafe_allow_html=True)
                                    else:
                                        st.error("❌ Could not find team players in database")
                                else:
                                    st.error("❌ Database schema error: missing 'id' column")
                            else:
                                st.markdown("""
                                <div style="background: rgba(255, 59, 92, 0.1); border-radius: 12px; 
                                            padding: 30px; border-left: 4px solid #ff3b5c;">
                                    <h4 style="margin: 0 0 12px 0; color: #ff3b5c; font-family: 'Orbitron', sans-serif;">
                                        ❌ TEAM NOT FOUND
                                    </h4>
                                    <p style="margin: 0; color: #cbd5e1; line-height: 1.6;">
                                        Please check your Team ID and try again. You can find your Team ID 
                                        in the FPL website URL when viewing your team.
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)

        # --- TAB 2: DREAM TEAM ---
        with tabs[1]:
            st.markdown(f"""
            <h2 style="color: #00ff9f; font-family: 'Orbitron', sans-serif; 
                       text-align: center; margin-bottom: 30px; font-size: 32px;">
                🚀 {meta.get('name', 'GW')} BEST XI
            </h2>
            """, unsafe_allow_html=True)
            
            if not df_short.empty:
                # Summary metrics
                st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    render_metric_card("TOTAL COST", f"£{df_short['price'].sum():.1f}m")
                with col2:
                    render_metric_card("AVG PRICE", f"£{df_short['price'].mean():.1f}m")
                with col3:
                    render_metric_card("EXPECTED PTS", f"{df_short['gw19_xP'].sum():.1f}")
                with col4:
                    render_metric_card("PLAYERS", f"{len(df_short)}")
                
                st.markdown("<div style='margin: 50px 0;'></div>", unsafe_allow_html=True)
                
                # Pitch view
                render_pitch_view(df_short)
                
                # Detailed table
                st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
                with st.expander("📊 DETAILED STATISTICS"):
                    display_df = df_short[['position', 'web_name', 'team_name', 'price', 'gw19_xP']].copy()
                    display_df = display_df.sort_values('gw19_xP', ascending=False)
                    display_df.columns = ['Position', 'Player', 'Club', 'Price (£m)', 'xPoints']
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("📊 Short-term predictions are being generated. Check back soon!")

        # --- TAB 3: LONG TERM ---
        with tabs[2]:
            st.markdown("""
            <h2 style="color: #00ff9f; font-family: 'Orbitron', sans-serif; 
                       text-align: center; margin-bottom: 30px; font-size: 32px;">
                🔮 5-WEEK PROJECTION
            </h2>
            """, unsafe_allow_html=True)
            
            if not df_long.empty:
                # Summary
                st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    render_metric_card("TEAM COST", f"£{df_long['price'].sum():.1f}m")
                with col2:
                    render_metric_card("5-WEEK POINTS", f"{df_long['long_term_xP'].sum():.1f}")
                with col3:
                    render_metric_card("BEST PICK", df_long.nlargest(1, 'long_term_xP')['web_name'].iloc[0])
                
                st.markdown("<div style='margin: 50px 0;'></div>", unsafe_allow_html=True)
                
                # Player cards
                st.markdown("""
                <h3 style="text-align: center; color: #00bfff; font-family: 'Orbitron', sans-serif; 
                           margin-bottom: 30px; text-transform: uppercase; letter-spacing: 2px;">
                    🌟 TOP LONG-TERM PICKS
                </h3>
                """, unsafe_allow_html=True)
                
                cols = st.columns(3)
                for idx, (_, player) in enumerate(df_long.head(6).iterrows()):
                    with cols[idx % 3]:
                        st.markdown(render_player_card(player), unsafe_allow_html=True)
                
                # Full table
                st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
                with st.expander("📊 FULL RANKINGS"):
                    display_df = df_long[['position', 'web_name', 'team_name', 'price', 'long_term_xP']].copy()
                    display_df = display_df.sort_values('long_term_xP', ascending=False)
                    display_df.columns = ['Position', 'Player', 'Club', 'Price (£m)', '5-Week Points']
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("📊 Long-term projections are being calculated. Check back soon!")

        # --- TAB 4: PLAYER POOL ---
        with tabs[3]:
            st.markdown("""
            <h2 style="color: #00ff9f; font-family: 'Orbitron', sans-serif; 
                       text-align: center; margin-bottom: 30px; font-size: 32px;">
                📊 COMPLETE PLAYER DATABASE
            </h2>
            """, unsafe_allow_html=True)
            
            if not df_all.empty:
                # Filters
                st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    position_filter = st.multiselect(
                        "FILTER BY POSITION",
                        options=['GKP', 'DEF', 'MID', 'FWD'],
                        default=['GKP', 'DEF', 'MID', 'FWD']
                    )
                with col2:
                    price_range = st.slider(
                        "PRICE RANGE (£m)",
                        float(df_all['price'].min()),
                        float(df_all['price'].max()),
                        (float(df_all['price'].min()), float(df_all['price'].max()))
                    )
                with col3:
                    search = st.text_input("🔍 SEARCH PLAYER", "")
                
                # Apply filters
                filtered_df = df_all.copy()
                if position_filter:
                    filtered_df = filtered_df[filtered_df['position'].isin(position_filter)]
                filtered_df = filtered_df[
                    (filtered_df['price'] >= price_range[0]) & 
                    (filtered_df['price'] <= price_range[1])
                ]
                if search:
                    filtered_df = filtered_df[
                        filtered_df['web_name'].str.contains(search, case=False, na=False)
                    ]
                
                st.markdown(f"""
                <div style="text-align: center; margin: 20px 0;">
                    <span style="color: #00ff9f; font-size: 18px; font-weight: 700; 
                                 font-family: 'Orbitron', sans-serif;">
                        SHOWING {len(filtered_df)} PLAYERS
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                # Display table
                display_df = filtered_df[['web_name', 'team_name', 'position', 'price', 'final_5gw_xP']].copy()
                display_df = display_df.sort_values('final_5gw_xP', ascending=False)
                display_df.columns = ['Player', 'Club', 'Position', 'Price (£m)', '5GW xPoints']
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    height=600
                )

        # --- TAB 5: MODEL LAB ---
        with tabs[4]:
            st.markdown("""
            <h2 style="color: #00ff9f; font-family: 'Orbitron', sans-serif; 
                       text-align: center; margin-bottom: 30px; font-size: 32px;">
                🧪 MODEL PERFORMANCE LAB
            </h2>
            """, unsafe_allow_html=True)
            
            if not premium:
                display_locked_feature("Model Lab")
            else:
                if metrics:
                    st.markdown("""
                    <h3 style="text-align: center; color: #00bfff; font-family: 'Orbitron', sans-serif; 
                               margin: 30px 0; text-transform: uppercase; letter-spacing: 2px;">
                        📈 MODEL METRICS
                    </h3>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        render_metric_card("R² SCORE", f"{metrics.get('r2', 0):.3f}")
                    with col2:
                        render_metric_card("RMSE", f"{metrics.get('rmse', 0):.2f}")
                    with col3:
                        render_metric_card("MAE", f"{metrics.get('mae', 0):.2f}")
                    with col4:
                        render_metric_card("ACCURACY", f"{metrics.get('r2', 0)*100:.1f}%")
                    
                    st.markdown("<div style='margin: 60px 0;'></div>", unsafe_allow_html=True)
                    
                    if not df_val.empty:
                        st.markdown("""
                        <h3 style="text-align: center; color: #00bfff; font-family: 'Orbitron', sans-serif; 
                                   margin: 30px 0; text-transform: uppercase; letter-spacing: 2px;">
                            🎯 PREDICTION QUALITY
                        </h3>
                        """, unsafe_allow_html=True)
                        
                        # Chart
                        chart_data = df_val[['Actual_Points', 'Predicted_Points']].copy()
                        st.scatter_chart(
                            chart_data,
                            x='Predicted_Points',
                            y='Actual_Points',
                            use_container_width=True,
                            height=400
                        )
                        
                        # Details
                        with st.expander("📊 VALIDATION DETAILS"):
                            st.dataframe(df_val, use_container_width=True)
                else:
                    st.info("📊 Model metrics are being calculated.")

if __name__ == "__main__":
    main()