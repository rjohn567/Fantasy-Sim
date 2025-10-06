import os
import pandas as pd
import numpy as np
from sleeper_wrapper import League as SleeperLeague
from espn_api.football import League as EspnLeague
import csv

# --- CONFIGURATION ---
# Replace these with your actual league details and credentials.
# It's best practice to use environment variables for sensitive data.

# Sleeper Config
SLEEPER_LEAGUE_ID = os.environ.get("SLEEPER_LEAGUE_ID", "1180175868088176640")
# ESPN Config
ESPN_LEAGUE_ID = int(os.environ.get("ESPN_LEAGUE_ID", 1247675))
ESPN_YEAR = 2025 # The current season year


# Function to load ESPN cookies from a CSV file
def load_espn_cookies(file_path):
    swid = None
    espn_s2 = None
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            # Assuming SWID is in column 0, row 1 and S2 is in column 1, row 1
            # Skip header row if present, or adjust if no header
            rows = list(reader)
            if len(rows) > 1: # Ensure there's at least a second row for 0-based index 1
                if len(rows[1]) > 0:
                    swid = rows[1][0]
                if len(rows[1]) > 1:
                    espn_s2 = rows[1][1]
    except FileNotFoundError:
        print(f"Warning: ESPN cookie file not found at {file_path}. Using environment variables or defaults.")
    except IndexError:
        print(f"Warning: ESPN cookie file at {file_path} does not have expected format. Using environment variables or defaults.")
    return swid, espn_s2

# Load cookies from CSV
ESPN_COOKIES_FILE = "/Users/ryan/Documents/Passwords/ESPN.csv"
csv_swid, csv_espn_s2 = load_espn_cookies(ESPN_COOKIES_FILE)

# Prioritize CSV loaded cookies, then environment variables, then hardcoded defaults
SWID_COOKIE = csv_swid 
ESPN_S2_COOKIE = csv_espn_s2
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

def calculate_weekly_win_pct_vs_everyone(all_scores, team_map):
    """
    Calculates the 'win % vs everyone' metric and total W-L record for each team.
    Returns both the win percentage and total W-L record against all teams.
    """
    weekly_wins_vs_everyone = {team_id: [] for team_id in team_map.keys()}
    total_wins_vs_everyone = {team_id: 0 for team_id in team_map.keys()}
    total_games_vs_everyone = {team_id: 0 for team_id in team_map.keys()}
    
    num_teams = len(team_map)
    if num_teams <= 1:
        return {name: 0 for name in team_map.values()}, {name: "0-0" for name in team_map.values()}

    # Get a list of all weekly scores for all teams
    # Structure: {week_index: {team_id: score}}
    scores_by_week = {}
    for team_id, scores in all_scores.items():
        for week_idx, score in enumerate(scores):
            if week_idx not in scores_by_week:
                scores_by_week[week_idx] = {}
            scores_by_week[week_idx][team_id] = score

    # Calculate win % vs everyone for each week
    for week_idx, weekly_scores in scores_by_week.items():
        if len(weekly_scores) < 2: continue # Not enough scores to compare

        for team_id, score in weekly_scores.items():
            wins = 0
            opponents = 0
            for opponent_id, opponent_score in weekly_scores.items():
                if team_id != opponent_id:
                    opponents += 1
                    if score > opponent_score:
                        wins += 1
            if opponents > 0:
                weekly_wins_vs_everyone[team_id].append(wins / opponents)
                total_wins_vs_everyone[team_id] += wins
                total_games_vs_everyone[team_id] += opponents

    # Calculate the average across all weeks for each team
    win_pct_vs_everyone = {}
    total_record_vs_everyone = {}
    
    for team_id, weekly_pcts in weekly_wins_vs_everyone.items():
        team_name = team_map.get(team_id)
        if team_name:
            if weekly_pcts:
                win_pct_vs_everyone[team_name] = np.mean(weekly_pcts)
                total_wins = total_wins_vs_everyone[team_id]
                total_games = total_games_vs_everyone[team_id]
                total_losses = total_games - total_wins
                total_record_vs_everyone[team_name] = f"{total_wins}-{total_losses}"
            else:
                win_pct_vs_everyone[team_name] = 0
                total_record_vs_everyone[team_name] = "0-0"
    
    return win_pct_vs_everyone, total_record_vs_everyone

def calculate_power_rankings(df, win_vs_everyone, total_record_vs_everyone):
    """
    Calculates power rankings based on the user's specified formula.
    Formula: (PF / highest PF) + win % + win % vs everyone
    Also includes actual record and total record vs everyone
    """
    if df.empty or (df['W'] + df['L'] + df['T']).sum() == 0:
        return pd.DataFrame(columns=['Rank', 'Team', 'Power Score', 'W-L-T', 'Total W-L', 'PF'])

    # Normalize PF
    highest_pf = df['PF'].max()
    if highest_pf == 0:
        df['Normalized PF'] = 0
    else:
        df['Normalized PF'] = df['PF'] / highest_pf

    # Calculate Win %
    df['Win %'] = (df['W'] + 0.5 * df['T']) / (df['W'] + df['L'] + df['T'])
    
    # Add Win % vs Everyone
    df['Win % vs Everyone'] = df['Team'].map(win_vs_everyone).fillna(0)

    # Calculate final Power Score by averaging the components
    df['Power Score'] = (df['Normalized PF'] + df['Win %'] + df['Win % vs Everyone']) / 3
    
    # Create W-L-T record string
    df['W-L-T'] = df.apply(lambda x: f"{int(x['W'])}-{int(x['L'])}-{int(x['T'])}", axis=1)
    
    # Add total record vs everyone
    df['Total W-L'] = df['Team'].map(total_record_vs_everyone)
    
    df_ranked = df.sort_values(by='Power Score', ascending=False).reset_index(drop=True)
    df_ranked['Rank'] = df_ranked.index + 1
    
    return df_ranked[['Rank', 'Team', 'Power Score', 'W-L-T', 'Total W-L', 'PF']]

def run_monte_carlo_simulation(teams_data, remaining_schedule, num_simulations):
    """
    Runs a Monte Carlo simulation for the rest of the season.
    
    Args:
        teams_data (dict): Team stats (mean, std, current wins, current losses).
        remaining_schedule (list): List of matchups for future weeks.
        num_simulations (int): Number of simulations to run.

    Returns:
        dict: A dictionary where keys are team names and values are numpy arrays
              of final win totals for each simulation.
    """
    simulation_wins_distribution = {team: np.zeros(num_simulations, dtype=int) for team in teams_data}
    remaining_games = {team: REGULAR_SEASON_WEEKS - (data.get('wins', 0) + data.get('losses', 0)) 
                      for team, data in teams_data.items()}

    for i in range(num_simulations):
        # Start with current wins for this simulation run
        temp_wins = {team: data['wins'] for team, data in teams_data.items()}
        temp_remaining = remaining_games.copy()

        for week_matchups in remaining_schedule:
            for team_a_name, team_b_name in week_matchups:
                if team_a_name is None or team_b_name is None: continue
                
                # Skip if either team has no remaining games
                if temp_remaining[team_a_name] <= 0 or temp_remaining[team_b_name] <= 0:
                    continue

                # Generate random scores based on team's historical performance
                score_a = np.random.normal(teams_data[team_a_name]['mean'], teams_data[team_a_name]['std'])
                score_b = np.random.normal(teams_data[team_b_name]['mean'], teams_data[team_b_name]['std'])
                
                if score_a > score_b:
                    temp_wins[team_a_name] += 1
                elif score_b > score_a:
                    temp_wins[team_b_name] += 1
                # Update remaining games
                temp_remaining[team_a_name] -= 1
                temp_remaining[team_b_name] -= 1
        
        # Store the final win count for this simulation
        for team, wins in temp_wins.items():
            simulation_wins_distribution[team][i] = wins
            
    return simulation_wins_distribution

def create_win_probability_df(win_distribution, total_games, teams_data):
    """
    Converts the simulation win distribution into a probability table.
    """
    win_projections = {}
    projected_wins = {}
    for team, wins_array in win_distribution.items():
        # Get the number of losses for the current team
        team_losses = teams_data.get(team, {}).get('losses', 0)
        # Calculate the maximum possible wins for the team
        max_possible_wins = total_games - team_losses

        # Clip wins to the max possible for that specific team
        wins_array = np.clip(wins_array, 0, max_possible_wins)

        # Count occurrences of each win total
        win_counts = np.bincount(wins_array, minlength=total_games + 1)
        # Convert counts to probabilities
        probabilities = win_counts / len(wins_array)
        win_projections[team] = probabilities
        projected_wins[team] = np.mean(wins_array)

    df = pd.DataFrame(win_projections).T
    df.columns = [f"{i} Wins" for i in range(total_games + 1)]
    df = df.reset_index().rename(columns={'index': 'Team'})
    
    # Add and format projected wins
    df['Projected Wins'] = df['Team'].map(projected_wins)
    
    # Sort by projected wins (as float) before formatting
    df = df.sort_values(by='Projected Wins', ascending=False).reset_index(drop=True)
    df['Projected Wins'] = df['Projected Wins'].map('{:.2f}'.format)

    # Reorder columns to be descending from total_games
    win_cols = [f"{i} Wins" for i in range(total_games, -1, -1)]
    df = df[['Team', 'Projected Wins'] + win_cols]
    
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
        wins_from_scores = {team_id: 0 for team_id in team_map.keys()}
        
        # Pre-fetch rosters for sleeper to avoid multiple calls
        rosters = league_api.get_rosters() if league_type == 'sleeper' else None

        for week in range(1, current_week):
            if league_type == 'sleeper':
                matchups = league_api.get_matchups(week)
                matchup_groups = {}
                for m in matchups:
                    matchup_id = m.get('matchup_id')
                    if matchup_id:
                        if matchup_id not in matchup_groups:
                            matchup_groups[matchup_id] = []
                        matchup_groups[matchup_id].append(m)

                for matchup_id, teams in matchup_groups.items():
                    if len(teams) == 2:
                        team1, team2 = teams[0], teams[1]
                        if 'points' in team1 and 'points' in team2:
                            all_scores[team1['roster_id']].append(team1['points'])
                            all_scores[team2['roster_id']].append(team2['points'])
                            if team1['points'] > team2['points']:
                                wins_from_scores[team1['roster_id']] += 1
                            elif team2['points'] > team1['points']:
                                wins_from_scores[team2['roster_id']] += 1

            elif league_type == 'espn':
                matchups = league_api.box_scores(week)
                for m in matchups:
                    if not m.home_team or not m.away_team: continue
                    home_id, away_id = m.home_team.team_id, m.away_team.team_id
                    home_score, away_score = m.home_score, m.away_score
                    all_scores[home_id].append(home_score)
                    all_scores[away_id].append(away_score)
                    if home_score > away_score:
                        wins_from_scores[home_id] += 1
                    elif away_score > home_score:
                        wins_from_scores[away_id] += 1

        # Build records dataframe for display (using API data)
        if league_type == 'sleeper':
            if rosters:
                for roster in rosters:
                    team_id = roster['roster_id']
                    name = team_map.get(team_id)
                    if name:
                        record = roster['settings']
                        team_records.append({'Team': name, 'W': record['wins'], 'L': record['losses'], 'T': record['ties'], 'PF': record['fpts']})
        elif league_type == 'espn':
            for team in league_api.teams:
                team_records.append({'Team': team.team_name, 'W': team.wins, 'L': team.losses, 'T': team.ties, 'PF': team.points_for})
        
        records_df = pd.DataFrame(team_records)

    except Exception as e:
        print(f"Error fetching data for {league_name}: {e}")
        return

    # 2. Calculate Power Rankings
    win_vs_everyone, total_record_vs_everyone = calculate_weekly_win_pct_vs_everyone(all_scores, team_map)
    power_rankings_df = calculate_power_rankings(records_df.copy(), win_vs_everyone, total_record_vs_everyone)
    power_rankings_df.to_csv(f"{league_name}_power_rankings.csv", index=False)
    print(f"✓ Saved {league_name}_power_rankings.csv")

    # 3. Run Monte Carlo Simulation
    if not any(all_scores.values()) or current_week > REGULAR_SEASON_WEEKS:
        print("Not enough data or season is over. Skipping win projection simulation.")
        return

    # Prep data for simulation using wins from the records_df
    teams_data_for_sim = {}
    for team_id, scores in all_scores.items():
        if scores:
            team_name = team_map.get(team_id)
            if not team_name: continue

            team_record = records_df.loc[records_df['Team'] == team_name]
            if team_record.empty: continue

            teams_data_for_sim[team_name] = {
                'mean': np.mean(scores),
                'std': np.std(scores) if np.std(scores) > 0 else 1.0,
                'wins': team_record['W'].iloc[0],
                'losses': team_record['L'].iloc[0]
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
                if match.home_team and match.away_team:
                    weekly_matchups.append((match.home_team.team_name, match.away_team.team_name))
        
        remaining_schedule.append(weekly_matchups)

    # Run simulation and create output table
    if teams_data_for_sim and remaining_schedule:
        win_distribution = run_monte_carlo_simulation(teams_data_for_sim, remaining_schedule, NUM_SIMULATIONS)
        win_prob_df = create_win_probability_df(win_distribution, REGULAR_SEASON_WEEKS, teams_data_for_sim)
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
