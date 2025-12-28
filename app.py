import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import pandas as pd
import pickle
from pathlib import Path

# Import your existing modules
from src.data_loader import load_data, get_current_gameweek
from src.optimizer import optimize_team, get_transfer_suggestions
from src.model import predict_points

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="FPL AI Bot - SaaS Edition",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# AUTHENTICATION SETUP
# ============================================================================
@st.cache_resource
def load_config():
    """Load authentication configuration from config.yaml"""
    config_path = Path(__file__).parent / 'config.yaml'
    
    if not config_path.exists():
        st.error("❌ config.yaml file not found! Please create it first.")
        st.stop()
    
    with open(config_path) as file:
        config = yaml.load(file, Loader=SafeLoader)
    
    return config

# Load configuration
config = load_config()

# Create authenticator object
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# ============================================================================
# AUTHENTICATION LOGIC
# ============================================================================
def display_login_page():
    """Display the login interface"""
    st.title("⚽ FPL AI Bot - Premium Edition")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ### Welcome to FPL AI Bot
        
        Your AI-powered Fantasy Premier League assistant using:
        - 🧠 XGBoost Machine Learning
        - 📊 Understat xG Data
        - 🎯 Advanced Optimization Algorithms
        
        Please login to continue.
        """)
        
        # Login widget
        name, authentication_status, username = authenticator.login('Login', 'main')
    
    return name, authentication_status, username

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_user_role(username):
    """Get the role of the logged-in user"""
    return config['credentials']['usernames'][username].get('role', 'free')

def check_premium_access(username):
    """Check if user has premium access"""
    role = get_user_role(username)
    return role in ['premium', 'admin']

def display_upgrade_message():
    """Display upgrade message for free users"""
    st.warning("🔒 **Premium Feature**")
    st.markdown("""
    This feature is available for Premium users only.
    
    **Upgrade to Premium to unlock:**
    - 🎯 Transfer Wizard with AI Recommendations
    - 🔬 Model Lab with Performance Analytics
    - 📈 Advanced Statistics and Insights
    - ⚡ Priority Updates and Support
    
    Contact us to upgrade your account!
    """)

def display_user_badge(username):
    """Display user badge based on role"""
    role = get_user_role(username)
    
    if role == 'admin':
        return "👑 Admin"
    elif role == 'premium':
        return "🌟 Premium"
    else:
        return "🆓 Free"

# ============================================================================
# SIDEBAR
# ============================================================================
def setup_sidebar(name, username):
    """Setup sidebar with user info and controls"""
    with st.sidebar:
        # User info
        st.markdown(f"### Welcome, {name}!")
        badge = display_user_badge(username)
        st.markdown(f"**Status:** {badge}")
        
        st.divider()
        
        # Logout button
        authenticator.logout('Logout', 'sidebar')
        
        st.divider()
        
        # App info
        st.markdown("""
        ### About
        FPL AI Bot uses machine learning to predict player performance 
        and optimize your Fantasy Premier League team.
        
        ### Features
        - ✅ GW Dream Team
        - ✅ Long Term Predictions
        - ✅ Player Pool Analysis
        """)
        
        if check_premium_access(username):
            st.markdown("""
            - 🌟 Transfer Wizard
            - 🌟 Model Lab
            """)
        else:
            st.markdown("""
            - 🔒 Transfer Wizard (Premium)
            - 🔒 Model Lab (Premium)
            """)

# ============================================================================
# TAB CONTENT FUNCTIONS
# ============================================================================

def transfer_wizard_tab(username):
    """Transfer Wizard - Premium Feature"""
    if not check_premium_access(username):
        display_upgrade_message()
        return
    
    st.header("🎯 Transfer Wizard")
    st.markdown("Get AI-powered transfer recommendations for your team.")
    
    # Team ID input
    team_id = st.number_input(
        "Enter your FPL Team ID",
        min_value=1,
        value=None,
        help="You can find your Team ID in the FPL website URL"
    )
    
    if team_id:
        with st.spinner("🔄 Analyzing your team..."):
            try:
                # Load data
                data = load_data()
                current_gw = get_current_gameweek()
                
                # Get transfer suggestions (using your existing function)
                suggestions = get_transfer_suggestions(team_id, data, current_gw)
                
                if suggestions:
                    st.success(f"✅ Analysis complete for GW{current_gw}")
                    
                    # Display current team
                    st.subheader("📋 Your Current Team")
                    if 'current_team' in suggestions:
                        st.dataframe(suggestions['current_team'], use_container_width=True)
                    
                    # Display transfer recommendations
                    st.subheader("🔄 Recommended Transfers")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 🔴 Players OUT")
                        if 'transfers_out' in suggestions:
                            for player in suggestions['transfers_out']:
                                st.markdown(f"- {player['name']} ({player['team']}) - £{player['price']}m")
                    
                    with col2:
                        st.markdown("#### 🟢 Players IN")
                        if 'transfers_in' in suggestions:
                            for player in suggestions['transfers_in']:
                                st.markdown(f"- {player['name']} ({player['team']}) - £{player['price']}m")
                                st.caption(f"Predicted Points: {player['predicted_points']:.1f}")
                    
                    # Expected impact
                    if 'expected_impact' in suggestions:
                        st.metric(
                            "Expected Points Improvement",
                            f"+{suggestions['expected_impact']:.1f} pts",
                            help="Expected additional points from these transfers"
                        )
                
            except Exception as e:
                st.error(f"❌ Error analyzing team: {str(e)}")
                st.info("Please check your Team ID and try again.")

def gw_dream_team_tab():
    """GW Dream Team - Free Feature"""
    st.header("⭐ GW Dream Team")
    st.markdown("Optimized squad for the next gameweek.")
    
    with st.spinner("🔄 Building dream team..."):
        try:
            # Load data
            data = load_data()
            current_gw = get_current_gameweek()
            
            st.info(f"📅 Predictions for Gameweek {current_gw}")
            
            # Get optimized team (using your existing function)
            dream_team = optimize_team(data, horizon=1)  # 1 GW ahead
            
            if dream_team is not None:
                # Display formation
                st.subheader("Formation")
                st.markdown(f"**{dream_team['formation']}**")
                
                # Display team by position
                positions = ['GKP', 'DEF', 'MID', 'FWD']
                
                for pos in positions:
                    st.subheader(f"📍 {pos}")
                    pos_players = dream_team['players'][dream_team['players']['position'] == pos]
                    
                    for _, player in pos_players.iterrows():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{player['name']}** ({player['team']})")
                        with col2:
                            st.markdown(f"£{player['price']}m")
                        with col3:
                            st.markdown(f"🎯 {player['predicted_points']:.1f} pts")
                
                # Total team value and expected points
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Team Value", f"£{dream_team['total_cost']}m")
                
                with col2:
                    st.metric("Expected Points", f"{dream_team['expected_points']:.1f}")
                
        except Exception as e:
            st.error(f"❌ Error building dream team: {str(e)}")

def long_term_predictions_tab():
    """Long Term Predictions - Free Feature"""
    st.header("📈 Long Term Predictions")
    st.markdown("Optimized squad for the next 5 gameweeks.")
    
    with st.spinner("🔄 Analyzing long-term fixtures..."):
        try:
            # Load data
            data = load_data()
            current_gw = get_current_gameweek()
            
            st.info(f"📅 Predictions for GW{current_gw} to GW{current_gw+4}")
            
            # Get optimized team for 5 GWs (using your existing function)
            long_term_team = optimize_team(data, horizon=5)  # 5 GW ahead
            
            if long_term_team is not None:
                # Display key metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Team Value", f"£{long_term_team['total_cost']}m")
                
                with col2:
                    st.metric("Expected Points (5 GW)", f"{long_term_team['expected_points']:.1f}")
                
                with col3:
                    st.metric("Average per GW", f"{long_term_team['expected_points']/5:.1f}")
                
                # Display team
                st.subheader("📋 Optimized Squad")
                
                # Create a display dataframe
                display_df = long_term_team['players'][['name', 'team', 'position', 'price', 'predicted_points_total']].copy()
                display_df.columns = ['Player', 'Team', 'Position', 'Price (£m)', 'Expected Points (5 GW)']
                display_df = display_df.sort_values(['Position', 'Expected Points (5 GW)'], ascending=[True, False])
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Fixture difficulty chart
                st.subheader("📊 Fixture Difficulty (Next 5 GWs)")
                if 'fixture_difficulty' in long_term_team:
                    st.bar_chart(long_term_team['fixture_difficulty'])
                
        except Exception as e:
            st.error(f"❌ Error generating long-term predictions: {str(e)}")

def player_pool_tab():
    """Player Pool - Free Feature"""
    st.header("🏊 Player Pool")
    st.markdown("Explore all players with AI predictions.")
    
    try:
        # Load data
        data = load_data()
        current_gw = get_current_gameweek()
        
        st.info(f"📅 Data for Gameweek {current_gw}")
        
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
                min_value=float(data['price'].min()),
                max_value=float(data['price'].max()),
                value=(float(data['price'].min()), float(data['price'].max()))
            )
        
        with col3:
            sort_by = st.selectbox(
                "Sort By",
                options=['Predicted Points', 'Price', 'Value (Points per £m)']
            )
        
        # Filter data
        filtered_data = data[
            (data['position'].isin(position_filter)) &
            (data['price'] >= price_range[0]) &
            (data['price'] <= price_range[1])
        ].copy()
        
        # Calculate value metric
        filtered_data['value'] = filtered_data['predicted_points'] / filtered_data['price']
        
        # Sort
        if sort_by == 'Predicted Points':
            filtered_data = filtered_data.sort_values('predicted_points', ascending=False)
        elif sort_by == 'Price':
            filtered_data = filtered_data.sort_values('price', ascending=False)
        else:
            filtered_data = filtered_data.sort_values('value', ascending=False)
        
        # Display
        st.subheader(f"📊 {len(filtered_data)} Players Found")
        
        # Create display dataframe
        display_df = filtered_data[['name', 'team', 'position', 'price', 'predicted_points', 'value']].copy()
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
                top_pos = filtered_data[filtered_data['position'] == pos].head(5)
                for _, player in top_pos.iterrows():
                    st.markdown(f"• {player['name']} ({player['predicted_points']:.1f})")
        
    except Exception as e:
        st.error(f"❌ Error loading player pool: {str(e)}")

def model_lab_tab(username):
    """Model Lab - Premium Feature"""
    if not check_premium_access(username):
        display_upgrade_message()
        return
    
    st.header("🔬 Model Lab")
    st.markdown("Analyze model performance and feature importance.")
    
    try:
        # Load model metrics (assuming you save them during training)
        model_path = Path(__file__).parent / 'data' / 'model_metrics.pkl'
        
        if model_path.exists():
            with open(model_path, 'rb') as f:
                metrics = pickle.load(f)
            
            # Display metrics
            st.subheader("📊 Model Performance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "R² Score",
                    f"{metrics.get('r2_score', 0):.3f}",
                    help="Coefficient of determination (higher is better)"
                )
            
            with col2:
                st.metric(
                    "RMSE",
                    f"{metrics.get('rmse', 0):.2f}",
                    help="Root Mean Squared Error (lower is better)"
                )
            
            with col3:
                st.metric(
                    "MAE",
                    f"{metrics.get('mae', 0):.2f}",
                    help="Mean Absolute Error (lower is better)"
                )
            
            # Feature importance
            if 'feature_importance' in metrics:
                st.subheader("📈 Feature Importance")
                st.markdown("Which statistics influence player points the most?")
                
                importance_df = pd.DataFrame(metrics['feature_importance'])
                importance_df = importance_df.sort_values('importance', ascending=True).tail(15)
                
                st.bar_chart(importance_df.set_index('feature')['importance'])
            
            # Prediction accuracy by position
            if 'accuracy_by_position' in metrics:
                st.subheader("🎯 Accuracy by Position")
                
                pos_df = pd.DataFrame(metrics['accuracy_by_position'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(pos_df, use_container_width=True, hide_index=True)
                
                with col2:
                    st.bar_chart(pos_df.set_index('position')['r2_score'])
            
            # Training history
            st.subheader("📅 Training Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Last Updated:** {metrics.get('last_updated', 'N/A')}")
                st.markdown(f"**Training Samples:** {metrics.get('n_samples', 'N/A'):,}")
            
            with col2:
                st.markdown(f"**Model:** {metrics.get('model_type', 'XGBoost')}")
                st.markdown(f"**Features Used:** {metrics.get('n_features', 'N/A')}")
        
        else:
            st.warning("⚠️ Model metrics not found. Please run `updater.py` first to train the model.")
            st.code("python updater.py", language="bash")
        
    except Exception as e:
        st.error(f"❌ Error loading model metrics: {str(e)}")

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    """Main application logic"""
    
    # Check authentication status
    name, authentication_status, username = authenticator.login('Login', 'main')
    
    # Handle authentication states
    if authentication_status == False:
        st.error('❌ Username/password is incorrect')
        st.stop()
    
    if authentication_status == None:
        display_login_page()
        st.stop()
    
    # User is authenticated
    if authentication_status:
        # Setup sidebar
        setup_sidebar(name, username)
        
        # Main content
        st.title("⚽ FPL AI Bot")
        st.markdown("### Your AI-Powered Fantasy Premier League Assistant")
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Transfer Wizard",
            "⭐ GW Dream Team",
            "📈 Long Term Predictions",
            "🏊 Player Pool",
            "🔬 Model Lab"
        ])
        
        # Tab content
        with tab1:
            transfer_wizard_tab(username)
        
        with tab2:
            gw_dream_team_tab()
        
        with tab3:
            long_term_predictions_tab()
        
        with tab4:
            player_pool_tab()
        
        with tab5:
            model_lab_tab(username)
        
        # Footer
        st.divider()
        st.markdown("""
        <div style='text-align: center; color: #666;'>
            Made with ❤️ by FPL AI Bot | Powered by XGBoost & Understat xG
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()
