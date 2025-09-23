import os
import pandas as pd
import numpy as np
from sleeper_wrapper import League as SleeperLeague
from espn_api.football import League as EspnLeague

# --- CONFIGURATION ---
# Replace these with your actual league details and credentials.
# It's best practice to use environment variables for sensitive data.

# Sleeper Config
SLEEPER_LEAGUE_ID = os.environ.get("SLEEPER_LEAGUE_ID", "1180175868088176640")

# ESPN Config
ESPN_LEAGUE_ID = int(os.environ.get("ESPN_LEAGUE_ID", 12345))
ESPN_YEAR = 2024 # The current season year
ESPN_S2_COOKIE = os.environ.get("ESPN_S2_COOKIE", "YOUR_ESPN_S2_COOKIE")
SWID_COOKIE = os.environ.get("SWID_COOKIE", "{YOUR-SWID-COOKIE}")

# Simulation Config
NUM_SIMULATIONS = 10000
REGULAR_SEASON_WEEKS = 14 # Adjust if your league has a different regular season length

# --- HELPER FUNCTIONS ---

def get_team_mapping(league_api_obj, league_type):
    """Creates a mapping from team ID to team name."""
    team_map = {}
    if league_type == 'sleeper':
        users = league_api_obj.get_users()
        rosters = league_api_obj.get_rosters()
        user_map = {u['user_id']: u['display_name'] for u in users}
        for r in rosters:
            team_map[r['roster_id']] = user_map.get(r['owner_id'], f"Team {r['roster_id']}")
    elif league_type == 'espn':
        for team in league_api_obj.teams:
            team_map[team.team_id] = team.team_name
    return team_map

def calculate_power_rankings(df):
    """
    Calculates power rankings based on a team's record and points scored.
    Assumes df has columns: 'Team', 'W', 'L', 'T', 'PF'.
    """
    if df.empty or (df['W'] + df['L'] + df['T']).sum() == 0:
        return pd.DataFrame(columns=['Rank', 'Team', 'Power Score', 'W', 'L', 'T', 'PF'])

    df['Win %'] = (df['W'] + 0.5 * df['T']) / (df['W'] + df['L'] + df['T'])
    
    # Use rank(pct=True) to normalize scores, which is robust to outliers
    df['Win % Rank'] = df['Win %'].rank(pct=True)
    df['PF Rank'] = df['PF'].rank(pct=True)
    
    # Calculate Power Score with a 60/40 weighting towards points as it's a better indicator of team strength
    df['Power Score'] = (0.4 * df['Win % Rank'] + 0.6 * df['PF Rank']) * 100
    
    df_ranked = df.sort_values(by='Power Score', ascending=False).reset_index(drop=True)
    df_ranked['Rank'] = df_ranked.index + 1
    
    return df_ranked[['Rank', 'Team', 'Power Score', 'W', 'L', 'T', 'PF']]

def run_monte_carlo_simulation(teams_data, remaining_schedule, num_simulations):
    """
    Runs a Monte Carlo simulation for the rest of the season.
    
    Args:
        teams_data (dict): Team stats (mean, std, current wins).
        remaining_schedule (list): List of matchups for future weeks.
        num_simulations (int): Number of simulations to run.

    Returns:
        dict: A dictionary where keys are team names and values are numpy arrays
              of final win totals for each simulation.
    """
    simulation_wins_distribution = {team: np.zeros(num_simulations, dtype=int) for team in teams_data}

    for i in range(num_simulations):
        # Start with current wins for this simulation run
        temp_wins = {team: data['wins'] for team, data in teams_data.items()}

        for week_matchups in remaining_schedule:
            for team_a_name, team_b_name in week_matchups:
                if team_a_name is None or team_b_name is None: continue

                # Generate random scores based on team's historical performance
                score_a = np.random.normal(teams_data[team_a_name]['mean'], teams_data[team_a_name]['std'])
                score_b = np.random.normal(teams_data[team_b_name]['mean'], teams_data[team_b_name]['std'])
                
                if score_a > score_b:
                    temp_wins[team_a_name] += 1
                elif score_b > score_a:
                    temp_wins[team_b_name] += 1
                # Ties are rare and can be ignored for simplicity, or add 0.5 to each
        
        # Store the final win count for this simulation
        for team, wins in temp_wins.items():
            simulation_wins_distribution[team][i] = wins
            
    return simulation_wins_distribution

def create_win_probability_df(win_distribution, total_games):
    """
    Converts the simulation win distribution into a probability table.
    """
    win_projections = {}
    for team, wins_array in win_distribution.items():
        # Count occurrences of each win total
        win_counts = np.bincount(wins_array, minlength=total_games + 1)
        # Convert counts to probabilities
        probabilities = win_counts / len(wins_array)
        win_projections[team] = probabilities

    df = pd.DataFrame(win_projections).T
    df.columns = [f"{i} Wins" for i in range(total_games + 1)]
    df = df.reset_index().rename(columns={'index': 'Team'})
    return df

# --- LEAGUE PROCESSING ---

def process_league(league_name, league_api, league_type):
    """
    Main processing function for a single league. Fetches data, runs analytics,
    and saves CSV files.
    """
    print(f"--- Processing {league_name} ---")
    
    # 1. Fetch Data
    try:
        if league_type == 'sleeper':
            league_data = league_api.get_league()
            current_week = league_data['settings']['leg']
        else: # espn
            current_week = league_api.current_week
        if current_week == 0: # Pre-season
            print("Season has not started. Skipping analysis.")
            return
            
        team_map = get_team_mapping(league_api, league_type)
        
        # Get scores and records
        team_records = []
        all_scores = {team_id: [] for team_id in team_map.keys()}
        
        # Pre-fetch rosters for sleeper to avoid multiple calls
        rosters = league_api.get_rosters() if league_type == 'sleeper' else None

        for week in range(1, current_week):
            if league_type == 'sleeper':
                matchups = league_api.get_matchups(week)
                # Group matchups by matchup_id to process pairs
                matchup_groups = {}
                for m in matchups:
                    matchup_id = m.get('matchup_id')
                    if matchup_id:
                        if matchup_id not in matchup_groups:
                            matchup_groups[matchup_id] = []
                        matchup_groups[matchup_id].append(m)

                for matchup_id, teams in matchup_groups.items():
                    if len(teams) == 2:
                        team1 = teams[0]
                        team2 = teams[1]
                        # Ensure we have scores for both before appending
                        if 'points' in team1 and 'points' in team2:
                            all_scores[team1['roster_id']].append(team1['points'])
                            all_scores[team2['roster_id']].append(team2['points'])

            elif league_type == 'espn':
                matchups = league_api.box_scores(week)
                for m in matchups:
                    home_team, away_team = m.home_team, m.away_team
                    if not home_team or not away_team: continue # Bye week
                    home_id, away_id = home_team.team_id, away_team.team_id
                    home_score, away_score = m.home_score, m.away_score
                    all_scores[home_id].append(home_score)
                    all_scores[away_id].append(away_score)

        # Build records dataframe
        if league_type == 'sleeper':
            if rosters:
                for roster in rosters:
                    team_id = roster['roster_id']
                    name = team_map.get(team_id)
                    if name:
                        record = roster['settings']
                        team_records.append({'Team': name, 'W': record['wins'], 'L': record['losses'], 'T': record['ties'], 'PF': record['fpts']})
        elif league_type == 'espn':
            for team_id, name in team_map.items():
                team = next((t for t in league_api.teams if t.team_id == team_id), None)
                if team:
                    team_records.append({'Team': name, 'W': team.wins, 'L': team.losses, 'T': team.ties, 'PF': team.points_for})
        
        records_df = pd.DataFrame(team_records)

    except Exception as e:
        print(f"Error fetching data for {league_name}: {e}")
        return

    # 2. Calculate Power Rankings
    power_rankings_df = calculate_power_rankings(records_df.copy())
    power_rankings_df.to_csv(f"{league_name}_power_rankings.csv", index=False)
    print(f"✓ Saved {league_name}_power_rankings.csv")

    # 3. Run Monte Carlo Simulation
    if not all(all_scores.values()) or current_week > REGULAR_SEASON_WEEKS:
        print("Not enough data or season is over. Skipping win projection simulation.")
        return

    # Prep data for simulation
    teams_data_for_sim = {}
    for team_id, scores in all_scores.items():
        if scores:
            team_name = team_map.get(team_id)
            if not team_name: continue
            record_row = records_df[records_df['Team'] == team_name]
            if not record_row.empty:
                record = record_row.iloc[0]
                teams_data_for_sim[team_name] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores) if np.std(scores) > 0 else 1.0, # Avoid std of 0
                    'wins': record['W']
                }

    # Get remaining schedule
    remaining_schedule = []
    for week in range(current_week, REGULAR_SEASON_WEEKS + 1):
        weekly_matchups = []
        schedule = league_api.scoreboard(week) if league_type == 'espn' else league_api.get_matchups(week)
        
        if league_type == 'sleeper':
            matchup_pairs = {}
            for m in schedule:
                if m.get('matchup_id') and m.get('roster_id'):
                    matchup_id = m['matchup_id']
                    if matchup_id not in matchup_pairs:
                        matchup_pairs[matchup_id] = []
                    matchup_pairs[matchup_id].append(team_map.get(m['roster_id']))
            for pair in matchup_pairs.values():
                if len(pair) == 2:
                    weekly_matchups.append(tuple(pair))
        
        elif league_type == 'espn':
            for match in schedule:
                if match.home_team and match.away_team: # Not a bye week
                    weekly_matchups.append((match.home_team.team_name, match.away_team.team_name))
        
        remaining_schedule.append(weekly_matchups)

    # Run simulation and create output table
    if teams_data_for_sim and remaining_schedule:
        win_distribution = run_monte_carlo_simulation(teams_data_for_sim, remaining_schedule, NUM_SIMULATIONS)
        win_prob_df = create_win_probability_df(win_distribution, REGULAR_SEASON_WEEKS)
        win_prob_df.to_csv(f"{league_name}_win_projections.csv", index=False, float_format='%.4f')
        print(f"✓ Saved {league_name}_win_projections.csv")
    else:
        print("Could not run simulation due to missing schedule or team data.")


# --- MAIN EXECUTION ---

def main():
    """Main function to run the analytics for all configured leagues."""
    
    # Process Sleeper League
    if SLEEPER_LEAGUE_ID != "YOUR_SLEEPER_LEAGUE_ID":
        try:
            sleeper_league = SleeperLeague(SLEEPER_LEAGUE_ID)
            print(dir(sleeper_league))
            process_league("Sleeper_Dynasty", sleeper_league, 'sleeper')
        except Exception as e:
            print(f"Could not connect to Sleeper league. Check your LEAGUE_ID. Error: {e}")
    else:
        print("Sleeper league ID not set. Skipping.")

    print("\n" + "="*30 + "\n")

    # Process ESPN League
    if ESPN_LEAGUE_ID != 12345 and ESPN_S2_COOKIE != "YOUR_ESPN_S2_COOKIE":
        try:
            espn_league = EspnLeague(league_id=ESPN_LEAGUE_ID, year=ESPN_YEAR, espn_s2=ESPN_S2_COOKIE, swid=SWID_COOKIE)
            process_league("ESPN_OG", espn_league, 'espn')
        except Exception as e:
            print(f"Could not connect to ESPN league. Check your credentials and league settings. Error: {e}")
    else:
        print("ESPN league credentials not set. Skipping.")


if __name__ == "__main__":
    main()
