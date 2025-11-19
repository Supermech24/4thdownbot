import nflreadpy as nfl
import pandas as pd

# Load play-by-play data for all available years
print("Loading play-by-play data for all years...")
pbp = nfl.load_pbp(seasons=True)  # Load all available seasons

# Convert to pandas for easier analysis
pbp_pandas = pbp.to_pandas()

print(f"Loaded {len(pbp_pandas)} total plays")

# Filter for Baltimore Ravens 4th down plays from 2001 season onwards
ravens_4th_down = pbp_pandas[
    (pbp_pandas['down'] == 4) & 
    (pbp_pandas['posteam'] == 'BAL') &
    (pbp_pandas['season'] >= 2001)
].copy()

print(f"Found {len(ravens_4th_down)} Baltimore Ravens 4th down plays")

if len(ravens_4th_down) > 0:
    # Extract relevant variables for 4th down decision making
    fourth_down_data = {
        # Basic play information
        'game_id': ravens_4th_down['game_id'].tolist(),
        'play_id': ravens_4th_down['play_id'].tolist(),
        'week': ravens_4th_down['week'].tolist(),
        'season': ravens_4th_down['season'].tolist(),
        'game_date': ravens_4th_down['game_date'].tolist(),
        
        # Field position and situation
        'yardline_100': ravens_4th_down['yardline_100'].tolist(),  # Yards from opponent's goal line
        'ydstogo': ravens_4th_down['ydstogo'].tolist(),  # Yards to go for first down
        'yrdln': ravens_4th_down['yrdln'].tolist(),  # Field position (e.g., "BAL 25")
        
        # Time and game situation
        'qtr': ravens_4th_down['qtr'].tolist(),  # Quarter
        'time': ravens_4th_down['time'].tolist(),  # Time remaining in quarter
        'quarter_seconds_remaining': ravens_4th_down['quarter_seconds_remaining'].tolist(),
        'game_seconds_remaining': ravens_4th_down['game_seconds_remaining'].tolist(),
        
        # Score situation
        'posteam_score': ravens_4th_down['posteam_score'].tolist(),  # Ravens score
        'defteam_score': ravens_4th_down['defteam_score'].tolist(),  # Opponent score
        'score_differential': ravens_4th_down['score_differential'].tolist(),  # Score difference (positive = Ravens ahead)
        'total_home_score': ravens_4th_down['total_home_score'].tolist(),
        'total_away_score': ravens_4th_down['total_away_score'].tolist(),
        
        # Timeouts
        'posteam_timeouts_remaining': ravens_4th_down['posteam_timeouts_remaining'].tolist(),  # Ravens timeouts
        'defteam_timeouts_remaining': ravens_4th_down['defteam_timeouts_remaining'].tolist(),  # Opponent timeouts
        'home_timeouts_remaining': ravens_4th_down['home_timeouts_remaining'].tolist(),
        'away_timeouts_remaining': ravens_4th_down['away_timeouts_remaining'].tolist(),
        
        # Win probability and expected points
        'wp': ravens_4th_down['wp'].tolist(),  # Win probability
        'def_wp': ravens_4th_down['def_wp'].tolist(),  # Defensive win probability
        'wpa': ravens_4th_down['wpa'].tolist(),  # Win probability added
        'ep': ravens_4th_down['ep'].tolist(),  # Expected points
        'epa': ravens_4th_down['epa'].tolist(),  # Expected points added
        
        # Play type and outcome
        'play_type': ravens_4th_down['play_type'].tolist(),
        'desc': ravens_4th_down['desc'].tolist(),  # Play description
        'yards_gained': ravens_4th_down['yards_gained'].tolist(),
        'first_down': ravens_4th_down['first_down'].tolist(),
        'touchdown': ravens_4th_down['touchdown'].tolist(),
        
        # Field goal specific
        'field_goal_attempt': ravens_4th_down['field_goal_attempt'].tolist(),
        'field_goal_result': ravens_4th_down['field_goal_result'].tolist(),
        'kick_distance': ravens_4th_down['kick_distance'].tolist(),
        
        # Punt specific
        'punt_attempt': ravens_4th_down['punt_attempt'].tolist(),
        
        # Weather and venue
        'weather': ravens_4th_down['weather'].tolist(),
        'temp': ravens_4th_down['temp'].tolist(),
        'wind': ravens_4th_down['wind'].tolist(),
        'stadium': ravens_4th_down['stadium'].tolist(),
        'surface': ravens_4th_down['surface'].tolist(),
        'roof': ravens_4th_down['roof'].tolist(),
        
        # Game context
        'home_team': ravens_4th_down['home_team'].tolist(),
        'away_team': ravens_4th_down['away_team'].tolist(),
        'posteam_type': ravens_4th_down['posteam_type'].tolist(),  # Home or away
        'defteam': ravens_4th_down['defteam'].tolist(),  # Opponent team
        
        # Drive information
        'drive': ravens_4th_down['drive'].tolist(),
        'series': ravens_4th_down['series'].tolist(),
        'series_success': ravens_4th_down['series_success'].tolist(),
        
        # Advanced metrics
        'success': ravens_4th_down['success'].tolist(),  # Play success (EPA > 0)
        'cp': ravens_4th_down['cp'].tolist(),  # Completion probability
        'cpoe': ravens_4th_down['cpoe'].tolist(),  # Completion probability over expected
        'xpass': ravens_4th_down['xpass'].tolist(),  # Expected pass probability
        'pass_oe': ravens_4th_down['pass_oe'].tolist(),  # Pass over expected
    }
    
    # Create a comprehensive DataFrame
    ravens_4th_df = pd.DataFrame(fourth_down_data)
    
    print(f"\nRavens 4th Down Data Summary:")
    print(f"Total 4th down plays: {len(ravens_4th_df)}")
    print(f"Seasons covered: {sorted(ravens_4th_df['season'].unique())}")
    print(f"Weeks covered: {sorted(ravens_4th_df['week'].unique())}")
    
    print(f"\nPlay type breakdown:")
    print(ravens_4th_df['play_type'].value_counts())
    
    print(f"\nField goal attempts: {ravens_4th_df['field_goal_attempt'].sum()}")
    print(f"Punt attempts: {ravens_4th_df['punt_attempt'].sum()}")
    print(f"Rush attempts: {ravens_4th_df['play_type'].str.contains('run', na=False).sum()}")
    print(f"Pass attempts: {ravens_4th_df['play_type'].str.contains('pass', na=False).sum()}")
    
    print(f"\nYard line distribution (yards from goal):")
    print(ravens_4th_df['yardline_100'].describe())
    
    print(f"\nYards to go distribution:")
    print(ravens_4th_df['ydstogo'].describe())
    
    print(f"\nSample of key data:")
    key_cols = ['game_id', 'week', 'yardline_100', 'ydstogo', 'qtr', 'time', 
                'score_differential', 'play_type', 'field_goal_attempt', 'punt_attempt', 'desc']
    print(ravens_4th_df[key_cols].head(10))
    
    # Store the data in variables for further analysis
    ravens_4th_down_plays = ravens_4th_df
    total_plays = len(ravens_4th_df)
    field_goal_attempts = ravens_4th_df[ravens_4th_df['field_goal_attempt'] == 1]
    punt_attempts = ravens_4th_df[ravens_4th_df['punt_attempt'] == 1]
    go_for_it_attempts = ravens_4th_df[
        (ravens_4th_df['field_goal_attempt'] == 0) & 
        (ravens_4th_df['punt_attempt'] == 0) &
        (ravens_4th_df['play_type'].isin(['run', 'pass']))
    ]
    
    print(f"\nData stored in variables:")
    print(f"- ravens_4th_down_plays: All Ravens 4th down plays ({len(ravens_4th_down_plays)} plays)")
    print(f"- field_goal_attempts: Field goal attempts ({len(field_goal_attempts)} plays)")
    print(f"- punt_attempts: Punt attempts ({len(punt_attempts)} plays)")
    print(f"- go_for_it_attempts: Go for it attempts ({len(go_for_it_attempts)} plays)")
    
    # Save data to CSV files
    print(f"\nSaving data to files...")
    
    # Save all Ravens 4th down plays (2001+)
    ravens_4th_down_plays.to_csv('ravens_4th_down_2001_onwards.csv', index=False)
    print(f"[SUCCESS] Saved all Ravens 4th down plays to 'ravens_4th_down_2001_onwards.csv'")
    
    # Save field goal attempts
    field_goal_attempts.to_csv('ravens_4th_down_field_goals_2001_onwards.csv', index=False)
    print(f"[SUCCESS] Saved field goal attempts to 'ravens_4th_down_field_goals_2001_onwards.csv'")
    
    # Save punt attempts
    punt_attempts.to_csv('ravens_4th_down_punts_2001_onwards.csv', index=False)
    print(f"[SUCCESS] Saved punt attempts to 'ravens_4th_down_punts_2001_onwards.csv'")
    
    # Save go-for-it attempts
    go_for_it_attempts.to_csv('ravens_4th_down_go_for_it_2001_onwards.csv', index=False)
    print(f"[SUCCESS] Saved go-for-it attempts to 'ravens_4th_down_go_for_it_2001_onwards.csv'")
    
    # Create a summary file
    summary_data = {
        'Metric': [
            'Total 4th Down Plays',
            'Field Goal Attempts',
            'Punt Attempts', 
            'Go For It Attempts',
            'Years Covered',
            'Weeks Covered',
            'Average Yard Line',
            'Average Yards to Go',
            'Field Goal Success Rate',
            'Go For It Success Rate'
        ],
        'Value': [
            len(ravens_4th_down_plays),
            len(field_goal_attempts),
            len(punt_attempts),
            len(go_for_it_attempts),
            f"{min(ravens_4th_down_plays['season'])}-{max(ravens_4th_down_plays['season'])}",
            f"{min(ravens_4th_down_plays['week'])}-{max(ravens_4th_down_plays['week'])}",
            f"{ravens_4th_down_plays['yardline_100'].mean():.1f}",
            f"{ravens_4th_down_plays['ydstogo'].mean():.1f}",
            f"{(field_goal_attempts['field_goal_result'] == 'made').mean()*100:.1f}%" if len(field_goal_attempts) > 0 else "N/A",
            f"{(go_for_it_attempts['success'] == 1).mean()*100:.1f}%" if len(go_for_it_attempts) > 0 else "N/A"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('ravens_4th_down_summary_2001_onwards.csv', index=False)
    print(f"[SUCCESS] Saved summary statistics to 'ravens_4th_down_summary_2001_onwards.csv'")
    
    print(f"\n[COMPLETE] All data successfully exported to CSV files!")

else:
    print("No Baltimore Ravens 4th down plays found in the dataset.")
    ravens_4th_down_plays = pd.DataFrame()
    field_goal_attempts = pd.DataFrame()
    punt_attempts = pd.DataFrame()
    go_for_it_attempts = pd.DataFrame()