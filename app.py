import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import pandas as pd
import math
import json
from pathlib import Path
from src.data_loader import DataLoader
from src.optimizer import Optimizer

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="FPL AI Pro 🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# AUTHENTICATION SETUP
# ============================================================================
@st.cache_resource
def load_auth_config():
    """Load authentication configuration from config.yaml"""
    config_path = Path(__file__).parent / 'config.yaml'
    
    if not config_path.exists():
        st.error("❌ config.yaml file not found!")
        st.stop()
    
    with open(config_path) as file:
        config = yaml.load(file, Loader=SafeLoader)
    
    return config

# Load config
config = load_auth_config()


# Create authenticator with defensive diagnostics
def _mask(s: str, keep: int = 4):
    if not isinstance(s, str):
        return str(type(s))
    if len(s) <= keep:
        return '*' * len(s)
    return s[:keep] + '...' 

try:
    creds = config.get('credentials')
    cookie_cfg = config.get('cookie', {})

    if not creds or 'usernames' not in creds:
        st.error('❌ config.yaml missing `credentials.usernames`.')
        st.stop()

    # Show a minimal, non-sensitive summary for troubleshooting
    st.debug = getattr(st, 'debug', lambda *a, **k: None)
    st.debug('Auth config usernames:', list(creds.get('usernames', {}).keys()))

    authenticator = stauth.Authenticate(
        creds,
        cookie_cfg.get('name'),
        cookie_cfg.get('key'),
        cookie_cfg.get('expiry_days')
    )
except Exception as e:
    st.error('❌ Failed to initialize authenticator. See diagnostics below.')
    st.exception(e)
    # Helpful config summary (non-sensitive)
    try:
        usernames = list(config.get('credentials', {}).get('usernames', {}).keys())
        cookie_name = config.get('cookie', {}).get('name', 'N/A')
        cookie_key = _mask(config.get('cookie', {}).get('key', ''))
        expiry = config.get('cookie', {}).get('expiry_days', 'N/A')

        st.info('Config summary:')
        st.write({'usernames': usernames, 'cookie_name': cookie_name, 'cookie_key': cookie_key, 'expiry_days': expiry})
    except Exception:
        pass
    st.stop()

# ============================================================================
# DATA LOADING (EXISTING FUNCTION - PRESERVED)
# ============================================================================
@st.cache_data
def load_files():
    """Load all data files (CSV/JSON) - Original function preserved"""
    try:
        # Load player data
        all_players = pd.read_csv('data/all_players.csv')
        
        # Load dream teams
        dt_short = pd.read_csv('data/dream_team_short.csv')
        dt_long = pd.read_csv('data/dream_team_long.csv')
        
        # Load validation data
        df_validation = pd.read_csv('data/validation_results.csv')
        
        # Load metrics
        with open('data/model_metrics.json', 'r') as f:
            metrics = json.load(f)
        
        # Load metadata
        with open('data/metadata.json', 'r') as f:
            meta = json.load(f)
        
        return all_players, dt_short, dt_long, df_validation, metrics, meta
    
    except FileNotFoundError as e:
        st.error(f"❌ Data file not found: {e}")
        st.warning("⚠️ Please run `python updater.py` first to generate data files.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

# ============================================================================
# HELPER FUNCTIONS FOR AUTHENTICATION
# ============================================================================
def get_user_role(username):
    """Get the role of the logged-in user"""
    try:
        return config['credentials'].get('usernames', {}).get(username, {}).get('role', 'free')
    except Exception:
        return 'free'

def is_premium_user(username):
    """Check if user has premium access"""
    role = get_user_role(username)
    return role in ['premium', 'admin']

def display_locked_feature(feature_name="This feature"):
    """Display message for locked features"""
    st.warning(f"🔒 **{feature_name} is Premium Only**")
    st.markdown("""
    Upgrade to Premium to unlock:
    - 🎯 Personal Transfer Wizard
    - 🔬 Model Performance Analytics
    - 📊 Advanced Statistics
    - ⚡ Priority Support
    
    Contact us to upgrade your account!
    """)

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # ========================================================================
    # AUTHENTICATION
    # ========================================================================
    login_result = authenticator.login(location='main')

    if login_result is None:
        st.error('❌ Authentication failed: login returned no data. Check config.yaml and streamlit_authenticator settings.')
        st.stop()

    # If login_result is not the expected 3-tuple, show diagnostics
    if not (isinstance(login_result, (list, tuple)) and len(login_result) == 3):
        st.error('❌ Authentication returned an unexpected value (not a 3-tuple). Showing diagnostics below.')
        try:
            st.write({'type': str(type(login_result)), 'repr': repr(login_result)})
        except Exception as e:
            st.write('Failed to serialize login_result:', str(e))
        st.stop()

    name, authentication_status, username = login_result

    # Handle authentication states
    if authentication_status is False:
        st.error('❌ Username/password is incorrect')
        st.stop()

    if authentication_status is None:
        st.warning('👋 Please enter your username and password')
        st.stop()
    
    # ========================================================================
    # USER IS AUTHENTICATED - LOAD DATA
    # ========================================================================
    if authentication_status:
        # Check premium status
        is_premium = is_premium_user(username)
        user_role = get_user_role(username)
        
        # Sidebar
        with st.sidebar:
            st.markdown(f"### Welcome, {name}!")
            
            # Display role badge
            if user_role == 'admin':
                st.markdown("**Status:** 👑 Admin")
            elif user_role == 'premium':
                st.markdown("**Status:** 🌟 Premium")
            else:
                st.markdown("**Status:** 🆓 Free")
            
            st.divider()
            
            # Logout button
            authenticator.logout(location='sidebar')
            
            st.divider()
            
            # App info
            st.markdown("""
            ### FPL AI Pro
            AI-powered Fantasy Premier League assistant
            
            **Free Features:**
            - ✅ GW Dream Team
            - ✅ Long Term Predictions
            - ✅ Player Pool
            
            **Premium Features:**
            - 🌟 Transfer Wizard
            - 🌟 Model Lab
            """)
        
        # Load data files
        df_all, df_short, df_long, df_val, metrics, meta = load_files()
        
        # Initialize classes
        loader = DataLoader()
        opt = Optimizer()
        
        # ====================================================================
        # MAIN CONTENT
        # ====================================================================
        st.title("⚽ FPL AI Pro")
        st.markdown("### Your AI-Powered Fantasy Premier League Assistant")
        
        # Current gameweek info
        current_gw = meta.get('current_gameweek', 'N/A')
        last_updated = meta.get('last_updated', 'N/A')
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📅 **Current Gameweek:** {current_gw}")
        with col2:
            st.info(f"🔄 **Last Updated:** {last_updated}")
        
        st.divider()
        
        # ====================================================================
        # TABS
        # ====================================================================
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Transfer Wizard",
            "⭐ GW Dream Team",
            "📈 Long Term Predictions",
            "🏊 Player Pool",
            "🔬 Model Lab"
        ])
        
        # ====================================================================
        # TAB 1: TRANSFER WIZARD (AUTH REQUIRED)
        # ====================================================================
        with tab1:
            st.header("🎯 Transfer Wizard")
            
            if not authentication_status:
                display_locked_feature("Transfer Wizard")
            else:
                st.markdown("Get AI-powered transfer recommendations for your team.")
                
                # Team ID input
                team_id = st.number_input(
                    "Enter your FPL Team ID",
                    min_value=1,
                    max_value=10000000,
                    value=None,
                    help="Find your Team ID in the FPL website URL"
                )
                
                if team_id:
                    with st.spinner("🔄 Analyzing your team..."):
                        try:
                            # Fetch user team using DataLoader
                            player_ids, bank = loader.fetch_user_team(team_id)
                            
                            if player_ids:
                                st.success(f"✅ Team loaded! Bank: £{bank}m")
                                
                                # Get current team details
                                current_team = df_all[df_all['id'].isin(player_ids)].copy()
                                
                                # Display current team
                                st.subheader("📋 Your Current Team")
                                
                                display_team = current_team[['name', 'team', 'position', 'price', 'predicted_points']].copy()
                                display_team.columns = ['Player', 'Team', 'Pos', 'Price (£m)', 'Predicted Pts']
                                display_team = display_team.sort_values('Pos')
                                
                                st.dataframe(display_team, use_container_width=True, hide_index=True)
                                
                                # Calculate team metrics
                                total_value = current_team['price'].sum()
                                expected_points = current_team['predicted_points'].sum()
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Total Team Value", f"£{total_value:.1f}m")
                                with col2:
                                    st.metric("Expected Points", f"{expected_points:.1f}")
                                
                                st.divider()
                                
                                # Get transfer suggestion using Optimizer
                                st.subheader("🔄 Transfer Recommendation")
                                
                                with st.spinner("🤖 AI analyzing best transfers..."):
                                    best_transfer = opt.suggest_transfer(current_team, df_all, bank)
                                    
                                    if best_transfer:
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.markdown("#### 🔴 Transfer OUT")
                                            out_player = best_transfer['player_out']
                                            st.markdown(f"**{out_player['name']}**")
                                            st.caption(f"{out_player['team']} • {out_player['position']} • £{out_player['price']}m")
                                            st.caption(f"Predicted: {out_player['predicted_points']:.1f} pts")
                                        
                                        with col2:
                                            st.markdown("#### 🟢 Transfer IN")
                                            in_player = best_transfer['player_in']
                                            st.markdown(f"**{in_player['name']}**")
                                            st.caption(f"{in_player['team']} • {in_player['position']} • £{in_player['price']}m")
                                            st.caption(f"Predicted: {in_player['predicted_points']:.1f} pts")
                                        
                                        # Show improvement
                                        improvement = best_transfer.get('improvement', 0)
                                        st.metric(
                                            "Expected Improvement",
                                            f"+{improvement:.1f} pts",
                                            help="Expected additional points from this transfer"
                                        )
                                        
                                        # Show reasoning
                                        if 'reason' in best_transfer:
                                            st.info(f"💡 **Reasoning:** {best_transfer['reason']}")
                                    else:
                                        st.success("✅ Your team is optimal! No transfers needed.")
                            else:
                                st.error("❌ Could not fetch team. Please check your Team ID.")
                        
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                            st.info("Make sure your Team ID is correct and the FPL API is accessible.")
        
        # ====================================================================
        # TAB 2: GW DREAM TEAM (FREE)
        # ====================================================================
        with tab2:
            st.header("⭐ GW Dream Team")
            st.markdown("Optimized squad for the next gameweek.")
            
            st.info(f"📅 Predictions for Gameweek {current_gw}")
            
            if df_short is not None and not df_short.empty:
                # Calculate formation
                formation = df_short.groupby('position').size()
                gkp = formation.get('GKP', 0)
                def_ = formation.get('DEF', 0)
                mid = formation.get('MID', 0)
                fwd = formation.get('FWD', 0)
                
                st.subheader(f"Formation: {gkp}-{def_}-{mid}-{fwd}")
                
                # Display team by position
                positions = ['GKP', 'DEF', 'MID', 'FWD']
                
                for pos in positions:
                    st.subheader(f"📍 {pos}")
                    pos_players = df_short[df_short['position'] == pos]
                    
                    if not pos_players.empty:
                        for idx, player in pos_players.iterrows():
                            col1, col2, col3 = st.columns([3, 1, 1])
                            
                            with col1:
                                st.markdown(f"**{player['name']}** ({player['team']})")
                            with col2:
                                st.markdown(f"£{player['price']:.1f}m")
                            with col3:
                                st.markdown(f"🎯 {player['predicted_points']:.1f} pts")
                
                # Team totals
                st.divider()
                
                total_cost = df_short['price'].sum()
                total_points = df_short['predicted_points'].sum()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Team Value", f"£{total_cost:.1f}m")
                
                with col2:
                    st.metric("Expected Points", f"{total_points:.1f}")
            else:
                st.warning("⚠️ Dream team data not available. Please run updater.py")
        
        # ====================================================================
        # TAB 3: LONG TERM PREDICTIONS (FREE)
        # ====================================================================
        with tab3:
            st.header("📈 Long Term Predictions")
            st.markdown("Optimized squad for the next 5 gameweeks.")
            
            st.info(f"📅 Predictions for GW{current_gw} to GW{current_gw+4}")
            
            if df_long is not None and not df_long.empty:
                # Key metrics
                total_cost = df_long['price'].sum()
                total_points = df_long['predicted_points_total'].sum()
                avg_per_gw = total_points / 5
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Team Value", f"£{total_cost:.1f}m")
                
                with col2:
                    st.metric("Expected Points (5 GW)", f"{total_points:.1f}")
                
                with col3:
                    st.metric("Average per GW", f"{avg_per_gw:.1f}")
                
                st.divider()
                
                # Display squad
                st.subheader("📋 Optimized Squad")
                
                display_df = df_long[['name', 'team', 'position', 'price', 'predicted_points_total']].copy()
                display_df.columns = ['Player', 'Team', 'Position', 'Price (£m)', 'Expected Points (5 GW)']
                display_df = display_df.sort_values(['Position', 'Expected Points (5 GW)'], ascending=[True, False])
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Top performers
                st.subheader("🌟 Top 5 Long-Term Picks")
                
                top_5 = df_long.nlargest(5, 'predicted_points_total')[['name', 'team', 'position', 'price', 'predicted_points_total']]
                
                for idx, player in top_5.iterrows():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.markdown(f"**{player['name']}** ({player['team']}) • {player['position']}")
                    with col2:
                        st.markdown(f"£{player['price']:.1f}m")
                    with col3:
                        st.markdown(f"🎯 {player['predicted_points_total']:.1f} pts")
            else:
                st.warning("⚠️ Long-term data not available. Please run updater.py")
        
        # ====================================================================
        # TAB 4: PLAYER POOL (FREE)
        # ====================================================================
        with tab4:
            st.header("🏊 Player Pool")
            st.markdown("Explore all players with AI predictions.")
            
            if df_all is not None and not df_all.empty:
                # Filters
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    position_filter = st.multiselect(
                        "Position",
                        options=['GKP', 'DEF', 'MID', 'FWD'],
                        default=['GKP', 'DEF', 'MID', 'FWD']
                    )
                
                with col2:
                    price_range = st.slider(
                        "Price Range (£m)",
                        min_value=float(df_all['price'].min()),
                        max_value=float(df_all['price'].max()),
                        value=(float(df_all['price'].min()), float(df_all['price'].max()))
                    )
                
                with col3:
                    sort_by = st.selectbox(
                        "Sort By",
                        options=['Predicted Points', 'Price', 'Value (Points per £m)', 'Name']
                    )
                
                # Filter data
                filtered = df_all[
                    (df_all['position'].isin(position_filter)) &
                    (df_all['price'] >= price_range[0]) &
                    (df_all['price'] <= price_range[1])
                ].copy()
                
                # Calculate value
                filtered['value'] = filtered['predicted_points'] / filtered['price']
                
                # Sort
                sort_map = {
                    'Predicted Points': ('predicted_points', False),
                    'Price': ('price', False),
                    'Value (Points per £m)': ('value', False),
                    'Name': ('name', True)
                }
                sort_col, ascending = sort_map[sort_by]
                filtered = filtered.sort_values(sort_col, ascending=ascending)
                
                # Display
                st.subheader(f"📊 {len(filtered)} Players Found")
                
                # Create display dataframe
                display_df = filtered[['name', 'team', 'position', 'price', 'predicted_points', 'value']].copy()
                display_df.columns = ['Player', 'Team', 'Position', 'Price (£m)', 'Predicted Points', 'Value (Pts/£m)']
                display_df['Value (Pts/£m)'] = display_df['Value (Pts/£m)'].round(2)
                display_df['Predicted Points'] = display_df['Predicted Points'].round(1)
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Top performers by position
                st.subheader("🌟 Top 5 by Position")
                
                cols = st.columns(4)
                positions = ['GKP', 'DEF', 'MID', 'FWD']
                
                for idx, pos in enumerate(positions):
                    with cols[idx]:
                        st.markdown(f"**{pos}**")
                        top_pos = filtered[filtered['position'] == pos].nlargest(5, 'predicted_points')
                        for _, player in top_pos.iterrows():
                            st.markdown(f"• {player['name']} ({player['predicted_points']:.1f})")
            else:
                st.warning("⚠️ Player data not available. Please run updater.py")
        
        # ====================================================================
        # TAB 5: MODEL LAB (PREMIUM ONLY)
        # ====================================================================
        with tab5:
            st.header("🔬 Model Lab")
            
            if not is_premium:
                display_locked_feature("Model Lab")
            else:
                st.markdown("Analyze model performance and feature importance.")
                
                if metrics:
                    # Performance metrics
                    st.subheader("📊 Model Performance")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        r2 = metrics.get('r2_score', 0)
                        st.metric(
                            "R² Score",
                            f"{r2:.3f}",
                            help="Coefficient of determination (higher is better)"
                        )
                    
                    with col2:
                        rmse = metrics.get('rmse', 0)
                        st.metric(
                            "RMSE",
                            f"{rmse:.2f}",
                            help="Root Mean Squared Error (lower is better)"
                        )
                    
                    with col3:
                        mae = metrics.get('mae', 0)
                        st.metric(
                            "MAE",
                            f"{mae:.2f}",
                            help="Mean Absolute Error (lower is better)"
                        )
                    
                    st.divider()
                    
                    # Feature importance
                    if 'feature_importance' in metrics:
                        st.subheader("📈 Feature Importance")
                        st.markdown("Which statistics influence player points the most?")
                        
                        importance_df = pd.DataFrame(metrics['feature_importance'])
                        importance_df = importance_df.sort_values('importance', ascending=True).tail(15)
                        
                        st.bar_chart(importance_df.set_index('feature')['importance'])
                    
                    # Validation results
                    if df_val is not None and not df_val.empty:
                        st.subheader("🎯 Prediction Accuracy")
                        
                        # Calculate accuracy metrics
                        actual_mean = df_val['actual_points'].mean()
                        predicted_mean = df_val['predicted_points'].mean()
                        error = abs(actual_mean - predicted_mean)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Actual Avg", f"{actual_mean:.2f}")
                        with col2:
                            st.metric("Predicted Avg", f"{predicted_mean:.2f}")
                        with col3:
                            st.metric("Difference", f"{error:.2f}")
                        
                        # Scatter plot data
                        st.markdown("#### Predicted vs Actual Points")
                        
                        chart_data = df_val[['predicted_points', 'actual_points']].copy()
                        chart_data.columns = ['Predicted', 'Actual']
                        
                        st.scatter_chart(chart_data, x='Predicted', y='Actual')
                    
                    # Training info
                    st.divider()
                    st.subheader("📅 Training Information")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Last Updated:** {metrics.get('last_updated', 'N/A')}")
                        st.markdown(f"**Training Samples:** {metrics.get('n_samples', 'N/A'):,}")
                    
                    with col2:
                        st.markdown(f"**Model Type:** {metrics.get('model_type', 'XGBoost')}")
                        st.markdown(f"**Features Used:** {metrics.get('n_features', 'N/A')}")
                else:
                    st.warning("⚠️ Model metrics not available. Please run updater.py")
        
        # ====================================================================
        # FOOTER
        # ====================================================================
        st.divider()
        st.markdown("""
        <div style='text-align: center; color: #666;'>
            Made with ❤️ by FPL AI Pro | Powered by XGBoost & Understat xG
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()