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
# CUSTOM CSS FOR PREMIUM LOOK
# ============================================================================
def load_custom_css():
    st.markdown("""
    <style>
    /* Main background and colors */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e2e 0%, #2d2d44 100%);
    }
    
    /* Card styling */
    .player-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        transition: transform 0.2s;
    }
    
    .player-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Transfer comparison cards */
    .transfer-out {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        border-radius: 12px;
        padding: 25px;
        color: white;
        box-shadow: 0 4px 6px rgba(255, 107, 107, 0.3);
    }
    
    .transfer-in {
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
        border-radius: 12px;
        padding: 25px;
        color: white;
        box-shadow: 0 4px 6px rgba(81, 207, 102, 0.3);
    }
    
    /* Position badges */
    .position-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        margin: 5px 0;
    }
    
    .pos-GKP { background: #ffd43b; color: #1e1e2e; }
    .pos-DEF { background: #51cf66; color: white; }
    .pos-MID { background: #339af0; color: white; }
    .pos-FWD { background: #ff6b6b; color: white; }
    
    /* Metric styling */
    .metric-container {
        background: white;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1e1e2e;
        font-weight: 700;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #1e1e2e;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS FOR UI COMPONENTS
# ============================================================================
def render_player_card(player, show_comparison=False):
    """Render a beautiful player card with stats"""
    position_class = f"pos-{player.get('position', 'MID')}"
    
    card_html = f"""
    <div class="player-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 style="margin: 0; color: #1e1e2e;">{player.get('web_name', 'Unknown')}</h3>
                <p style="margin: 5px 0; color: #666;">{player.get('team_name', 'N/A')}</p>
                <span class="position-badge {position_class}">{player.get('position', 'N/A')}</span>
            </div>
            <div style="text-align: right;">
                <h2 style="margin: 0; color: #667eea;">⭐ {player.get('final_5gw_xP', 0):.1f}</h2>
                <p style="margin: 5px 0; color: #666; font-weight: 600;">£{player.get('price', 0):.1f}m</p>
            </div>
        </div>
    </div>
    """
    return card_html

def render_transfer_comparison(out_player, in_player, gain):
    """Render transfer comparison with side-by-side cards"""
    col1, col2, col3 = st.columns([5, 1, 5])
    
    with col1:
        st.markdown(f"""
        <div class="transfer-out">
            <h3 style="margin: 0 0 15px 0;">❌ TRANSFER OUT</h3>
            <h2 style="margin: 10px 0;">{out_player.get('web_name', 'Unknown')}</h2>
            <p style="margin: 5px 0; opacity: 0.9;">{out_player.get('team_name', 'N/A')} • {out_player.get('position', 'N/A')}</p>
            <div style="margin-top: 20px; display: flex; justify-content: space-between;">
                <div>
                    <p style="margin: 5px 0; font-size: 14px; opacity: 0.8;">Price</p>
                    <p style="margin: 0; font-size: 24px; font-weight: bold;">£{out_player.get('price', 0):.1f}m</p>
                </div>
                <div>
                    <p style="margin: 5px 0; font-size: 14px; opacity: 0.8;">Expected Points</p>
                    <p style="margin: 0; font-size: 24px; font-weight: bold;">{out_player.get('final_5gw_xP', 0):.1f}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div style='text-align: center; padding-top: 80px;'><h1>➡️</h1></div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="transfer-in">
            <h3 style="margin: 0 0 15px 0;">✅ TRANSFER IN</h3>
            <h2 style="margin: 10px 0;">{in_player.get('web_name', 'Unknown')}</h2>
            <p style="margin: 5px 0; opacity: 0.9;">{in_player.get('team_name', 'N/A')} • {in_player.get('position', 'N/A')}</p>
            <div style="margin-top: 20px; display: flex; justify-content: space-between;">
                <div>
                    <p style="margin: 5px 0; font-size: 14px; opacity: 0.8;">Price</p>
                    <p style="margin: 0; font-size: 24px; font-weight: bold;">£{in_player.get('price', 0):.1f}m</p>
                </div>
                <div>
                    <p style="margin: 5px 0; font-size: 14px; opacity: 0.8;">Expected Points</p>
                    <p style="margin: 0; font-size: 24px; font-weight: bold;">{in_player.get('final_5gw_xP', 0):.1f}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #ffd43b 0%, #fab005 100%); 
                border-radius: 12px; padding: 20px; text-align: center; 
                margin-top: 20px; box-shadow: 0 4px 6px rgba(255, 212, 59, 0.3);">
        <h3 style="margin: 0; color: #1e1e2e;">📈 Expected Gain: +{gain:.1f} Points</h3>
    </div>
    """, unsafe_allow_html=True)

def render_pitch_view(df_team):
    """Render team in a pitch-like formation"""
    st.markdown("### ⚽ Starting XI Formation")
    
    # Group by position
    positions = ['GKP', 'DEF', 'MID', 'FWD']
    position_names = {'GKP': 'Goalkeeper', 'DEF': 'Defenders', 'MID': 'Midfielders', 'FWD': 'Forwards'}
    
    for pos in positions:
        pos_players = df_team[df_team['position'] == pos]
        if not pos_players.empty:
            st.markdown(f"#### {position_names.get(pos, pos)}")
            cols = st.columns(len(pos_players))
            for idx, (_, player) in enumerate(pos_players.iterrows()):
                with cols[idx]:
                    st.markdown(render_player_card(player), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

def render_metric_card(label, value, delta=None, delta_color="normal"):
    """Render a custom metric card"""
    st.markdown(f"""
    <div class="metric-container">
        <p style="margin: 0; color: #666; font-size: 14px;">{label}</p>
        <h2 style="margin: 10px 0; color: #1e1e2e;">{value}</h2>
        {f'<p style="margin: 0; color: {"#51cf66" if delta_color == "normal" else "#ff6b6b"};">{delta}</p>' if delta else ''}
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
        'About': "FPL AI Pro - Your Ultimate Fantasy Premier League Assistant"
    }
)

load_custom_css()

# ============================================================================
# AUTHENTICATION
# ============================================================================
@st.cache_resource
def load_auth_config():
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
@st.cache_data
def load_files():
    try:
        all_players = pd.read_csv('data/all_players.csv')
        
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
# UTILITY FUNCTIONS
# ============================================================================
def get_user_role(username):
    try:
        user_data = config['credentials']['usernames'].get(username, {})
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
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #ffd43b 0%, #fab005 100%); 
                border-radius: 12px; padding: 30px; text-align: center;
                box-shadow: 0 4px 6px rgba(255, 212, 59, 0.3);">
        <h2 style="margin: 0 0 15px 0; color: #1e1e2e;">🔒 {feature_name}</h2>
        <p style="margin: 0; color: #495057; font-size: 16px;">
            This premium feature is available exclusively for Pro members.<br>
            Upgrade your account to unlock advanced AI-powered insights!
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
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
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        border-radius: 12px; padding: 20px; margin-bottom: 20px;">
                <h3 style="margin: 0; color: white;">👋 Welcome back!</h3>
                <h2 style="margin: 5px 0; color: white;">{name}</h2>
                <span style="background: rgba(255,255,255,0.2); padding: 5px 15px; 
                             border-radius: 20px; color: white; font-weight: bold;">
                    {role.upper()} {'⭐' if premium else ''}
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            authenticator.logout(location='sidebar')
            
            st.divider()
            
            st.markdown("### 📊 Navigation")
            st.markdown("""
            - 🎯 **Transfer Wizard** - AI-powered suggestions
            - 🚀 **Dream Team** - Best XI for this GW
            - 🔮 **Long Term** - 5-week projections
            - 📊 **Player Pool** - Full database
            - 🧪 **Model Lab** - Performance metrics
            """)
            
            if not premium:
                st.divider()
                st.markdown("""
                <div style="background: rgba(255, 212, 59, 0.1); border-radius: 8px; 
                            padding: 15px; border-left: 4px solid #ffd43b;">
                    <p style="margin: 0; color: #fab005; font-weight: bold;">
                        ⭐ Upgrade to Premium
                    </p>
                    <p style="margin: 5px 0 0 0; font-size: 12px; color: #ced4da;">
                        Unlock Transfer Wizard & Model Lab
                    </p>
                </div>
                """, unsafe_allow_html=True)

        # Header
        st.markdown(f"""
        <div style="background: white; border-radius: 12px; padding: 30px; 
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 30px;">
            <h1 style="margin: 0; color: #1e1e2e;">⚽ FPL AI Pro</h1>
            <p style="margin: 10px 0 0 0; color: #666; font-size: 18px;">
                Powered by Advanced Machine Learning | {meta.get('name', 'Next GW')}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Tabs
        tabs = st.tabs([
            "🎯 Transfer Wizard",
            "🚀 Dream Team",
            "🔮 Long Term",
            "📊 Player Pool",
            "🧪 Model Lab"
        ])

        # --- TAB 1: TRANSFER WIZARD ---
        with tabs[0]:
            st.markdown("## 🎯 AI Transfer Wizard")
            
            if not premium:
                display_locked_feature("Transfer Wizard")
            else:
                with st.container():
                    st.markdown("""
                    <div style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
                        <p style="margin: 0; color: #666;">
                            Enter your FPL Team ID to get personalized transfer recommendations powered by our AI engine.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        user_id = st.text_input("FPL Team ID", placeholder="e.g., 123456")
                    with col2:
                        st.markdown("<br>", unsafe_allow_html=True)
                        analyze_btn = st.button("🔍 Analyze Team", use_container_width=True)
                    
                    if user_id and analyze_btn:
                        with st.spinner("🤖 AI is analyzing your team..."):
                            loader = DataLoader()
                            player_ids, bank = loader.fetch_user_team(user_id)
                            
                            if player_ids and not df_all.empty:
                                if 'id' in df_all.columns:
                                    my_team = df_all[df_all['id'].isin(player_ids)].copy()
                                    if not my_team.empty:
                                        st.success(f"✅ Team loaded successfully!")
                                        
                                        # Display bank
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            render_metric_card("Team Value", f"£{my_team['price'].sum():.1f}m")
                                        with col2:
                                            render_metric_card("Bank Balance", f"£{bank}m")
                                        with col3:
                                            render_metric_card("Total Squad", f"{len(my_team)} players")
                                        
                                        st.markdown("<br>", unsafe_allow_html=True)
                                        
                                        # Team table with enhanced styling
                                        with st.expander("📋 View Full Squad", expanded=True):
                                            display_team = my_team[['web_name', 'team_name', 'position', 'price', 'final_5gw_xP']].copy()
                                            display_team.columns = ['Player', 'Club', 'Position', 'Price (£m)', '5GW Expected Points']
                                            display_team = display_team.sort_values('final_5gw_xP', ascending=False)
                                            st.dataframe(
                                                display_team,
                                                use_container_width=True,
                                                hide_index=True
                                            )
                                        
                                        st.divider()
                                        st.markdown("## 🤖 AI Recommendation")
                                        
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
                                            <div style="background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); 
                                                        border-radius: 12px; padding: 30px; text-align: center;">
                                                <h3 style="margin: 0; color: white;">✨ Your Team is Optimized!</h3>
                                                <p style="margin: 10px 0 0 0; color: white; opacity: 0.9;">
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
                                <div style="background: #ffe3e3; border-radius: 12px; padding: 20px; 
                                            border-left: 4px solid #ff6b6b;">
                                    <h4 style="margin: 0 0 10px 0; color: #c92a2a;">❌ Team Not Found</h4>
                                    <p style="margin: 0; color: #495057;">
                                        Please check your Team ID and try again. You can find your Team ID 
                                        in the FPL website URL when viewing your team.
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)

        # --- TAB 2: DREAM TEAM ---
        with tabs[1]:
            st.markdown(f"## 🚀 {meta.get('name', 'GW')} Best XI")
            
            if not df_short.empty:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    render_metric_card("Total Cost", f"£{df_short['price'].sum():.1f}m")
                with col2:
                    render_metric_card("Avg Player Price", f"£{df_short['price'].mean():.1f}m")
                with col3:
                    render_metric_card("Expected Points", f"{df_short['gw19_xP'].sum():.1f}")
                with col4:
                    render_metric_card("Players", f"{len(df_short)}")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Pitch view
                render_pitch_view(df_short)
                
                # Detailed table
                with st.expander("📊 Detailed Statistics"):
                    display_df = df_short[['position', 'web_name', 'team_name', 'price', 'gw19_xP']].copy()
                    display_df.columns = ['Position', 'Player', 'Club', 'Price (£m)', 'Expected Points']
                    display_df = display_df.sort_values('Expected Points', ascending=False)
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("📊 Short-term predictions are being generated. Check back soon!")

        # --- TAB 3: LONG TERM ---
        with tabs[2]:
            st.markdown("## 🔮 5-Week Projection")
            
            if not df_long.empty:
                # Summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    render_metric_card("Team Cost", f"£{df_long['price'].sum():.1f}m")
                with col2:
                    render_metric_card("5-Week Points", f"{df_long['long_term_xP'].sum():.1f}")
                with col3:
                    render_metric_card("Best Pick", df_long.nlargest(1, 'long_term_xP')['web_name'].iloc[0])
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Player cards
                st.markdown("### 🌟 Top Long-Term Picks")
                cols = st.columns(3)
                for idx, (_, player) in enumerate(df_long.head(6).iterrows()):
                    with cols[idx % 3]:
                        st.markdown(render_player_card(player), unsafe_allow_html=True)
                
                # Full table
                with st.expander("📊 Full Rankings"):
                    display_df = df_long[['position', 'web_name', 'team_name', 'price', 'long_term_xP']].copy()
                    display_df.columns = ['Position', 'Player', 'Club', 'Price (£m)', '5-Week Points']
                    display_df = display_df.sort_values('5-Week Points', ascending=False)
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("📊 Long-term projections are being calculated. Check back soon!")

        # --- TAB 4: PLAYER POOL ---
        with tabs[3]:
            st.markdown("## 📊 Complete Player Database")
            
            if not df_all.empty:
                # Filters
                with st.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        position_filter = st.multiselect(
                            "Filter by Position",
                            options=['GKP', 'DEF', 'MID', 'FWD'],
                            default=['GKP', 'DEF', 'MID', 'FWD']
                        )
                    with col2:
                        price_range = st.slider(
                            "Price Range (£m)",
                            float(df_all['price'].min()),
                            float(df_all['price'].max()),
                            (float(df_all['price'].min()), float(df_all['price'].max()))
                        )
                    with col3:
                        search = st.text_input("🔍 Search Player", "")
                
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
                
                st.markdown(f"**Showing {len(filtered_df)} players**")
                
                display_df = filtered_df[['web_name', 'team_name', 'position', 'price', 'final_5gw_xP']].copy()
                display_df.columns = ['Player', 'Club', 'Position', 'Price (£m)', '5GW Expected Points']
                display_df = display_df.sort_values('5GW Expected Points', ascending=False)
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    height=600
                )

        # --- TAB 5: MODEL LAB ---
        with tabs[4]:
            st.markdown("## 🧪 Model Performance Lab")
            
            if not premium:
                display_locked_feature("Model Lab")
            else:
                if metrics:
                    st.markdown("### 📈 Model Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        render_metric_card("R² Score", f"{metrics.get('r2', 0):.3f}")
                    with col2:
                        render_metric_card("RMSE", f"{metrics.get('rmse', 0):.2f}")
                    with col3:
                        render_metric_card("MAE", f"{metrics.get('mae', 0):.2f}")
                    with col4:
                        render_metric_card("Accuracy", f"{metrics.get('r2', 0)*100:.1f}%")
                    
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    
                    if not df_val.empty:
                        st.markdown("### 🎯 Prediction Quality")
                        
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
                        with st.expander("📊 Validation Details"):
                            st.dataframe(df_val, use_container_width=True)
                else:
                    st.info("📊 Model metrics are being calculated.")

if __name__ == "__main__":
    main()