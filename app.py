import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(page_title="Smokeback Analytics", layout="wide")

# --- Data Loading and Caching ---
@st.cache_data
def load_data(file_path):
    """Loads a CSV file into a pandas DataFrame."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return None

# --- UI for Dynasty Page ---
def display_dynasty_page():
    st.title("Owners Analytics")

    # --- Power Rankings ---
    st.header("Power Rankings")
    power_rankings_df = load_data("Sleeper_Dynasty_power_rankings.csv")

    if power_rankings_df is not None:
        # Select and display relevant columns
        pr_display_df = power_rankings_df[['Rank', 'Team', 'Power Score']].copy()
        pr_display_df['Power Score'] = pr_display_df['Power Score'].map('{:.2f}'.format)
        
        st.dataframe(
            pr_display_df.set_index('Rank'),
            width='stretch'
        )
    else:
        st.warning("Power rankings data not found. Please run `fantasy_analytics.py` first.")

    # --- Win Projections ---
    st.header("Monte Carlo Win Projections")
    win_proj_df = load_data("Sleeper_Dynasty_win_projections.csv")

    if win_proj_df is not None:
        # Set Team as index
        win_proj_df = win_proj_df.set_index('Team')

        #Round the projected wins to 1 decimal places
        win_proj_df['Projected Wins'] = win_proj_df['Projected Wins'].round(1)
        
        # Get only the win columns for styling
        win_cols = [col for col in win_proj_df.columns if 'Wins' in col and col != 'Projected Wins']
        
        # Create a formatter for the win probability columns
        formatter = {col: '{:.2%}' for col in win_cols}

        # Apply heatmap styling and specific formatting
        st.dataframe(
            win_proj_df.style.background_gradient(cmap='coolwarm', subset=win_cols).format(formatter),
            width='stretch'
        )
        st.caption("Each cell represents the simulated probability of that team achieving that many wins.")
    else:
        st.warning("Win projection data not found. Please run `fantasy_analytics.py` first.")

# --- UI for OG Page ---
def display_og_page():
    st.title("E-Boy League of Fantasy Football Analytics")

    # --- Power Rankings ---
    st.header("Power Rankings")
    power_rankings_df = load_data("ESPN_OG_power_rankings.csv")

    if power_rankings_df is not None:
        # Select and display relevant columns
        pr_display_df = power_rankings_df[['Rank', 'Team', 'Power Score']].copy()
        pr_display_df['Power Score'] = pr_display_df['Power Score'].map('{:.2f}'.format)
        
        st.dataframe(
            pr_display_df.set_index('Rank'),
            width='stretch'
        )
    else:
        st.warning("Power rankings data not found. Please run `fantasy_analytics.py` first.")

    # --- Win Projections ---
    st.header("Monte Carlo Win Projections")
    win_proj_df = load_data("ESPN_OG_win_projections.csv")

    if win_proj_df is not None:
        # Set Team as index
        win_proj_df = win_proj_df.set_index('Team')

        #Round the projected wins to 1 decimal places
        win_proj_df['Projected Wins'] = win_proj_df['Projected Wins'].round(1)
        
        # Get only the win columns for styling
        win_cols = [col for col in win_proj_df.columns if 'Wins' in col and col != 'Projected Wins']
        
        # Create a formatter for the win probability columns
        formatter = {col: '{:.2%}' for col in win_cols}

        # Apply heatmap styling and specific formatting
        st.dataframe(
            win_proj_df.style.background_gradient(cmap='coolwarm', subset=win_cols).format(formatter),
            width='stretch'
        )
        st.caption("Each cell represents the simulated probability of that team achieving that many wins.")
    else:
        st.warning("Win projection data not found. Please run `fantasy_analytics.py` first.")

# --- Main App Navigation ---
st.sidebar.title("Navigation")
page_selection = st.sidebar.radio(
    "Go to",
    ["Owners", "ELFF"]
)

if page_selection == "Owners":
    display_dynasty_page()
elif page_selection == "ELFF":
    display_og_page()