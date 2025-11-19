# test.py
# Edit inputs below and get expected WP values for each 4th down decision
#
# Requires trained models saved in ./models by mvp_4th_down_bot.py:
#   - models/go_wp_model.joblib
#   - models/fg_wp_model.joblib
#   - models/punt_wp_model.joblib
#
# Usage: python test.py

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# =========================
# EDIT THESE INPUTS
# =========================
Yardline              = 25   # distance from opponent end zone (1..99). 40 => own 40.
Quarter               = 3.0    # 1..4
Time_remaining        = 200.0   # seconds left (in quarter for Q1/Q2, in game for Q3/Q4)
Score_differential    = 0   # Ravens score - Opp score (negative if trailing)
Yds_to_go             = 10.0    # yards for a first down
Wind_speed            = 0.0    # mph (affects FG model)
Offense_strength      = 0.05   # Ravens offensive rating; 0 = league avg
Opp_defense_strength  = -0.05  # Opponent defensive rating; 0 = league avg

# =========================
# Paths
# =========================
MODEL_DIR = Path("models")
GO_WP_MODEL_PATH = MODEL_DIR / "go_wp_model.joblib"
FG_WP_MODEL_PATH = MODEL_DIR / "fg_wp_model.joblib"
PUNT_WP_MODEL_PATH = MODEL_DIR / "punt_wp_model.joblib"

# =========================
# Feature Engineering (same as training)
# =========================
def create_features_from_inputs(yardline, quarter, time_remaining, score_differential, 
                                yds_to_go, wind, offense_strength, opp_defense_strength):
    """
    Create features from user inputs to match the training data format.
    """
    # Time left in half (combination of quarter and time remaining)
    # For Q1/Q2: use quarter_seconds_remaining (first half)
    # For Q3/Q4: use game_seconds_remaining (second half/end of game)
    # 
    # Interpretation: time_remaining is the seconds left in the current quarter for Q1/Q2,
    #                  or game_seconds_remaining for Q3/Q4
    if quarter in [1, 2]:
        # Q1/Q2: time_remaining is interpreted as seconds left in that quarter
        time_left_in_half = float(time_remaining)
    else:
        # Q3/Q4: time_remaining is interpreted as game_seconds_remaining
        time_left_in_half = float(time_remaining)
    
    # Own territory indicator (yardline > 50 means own territory)
    own_territory = 1 if yardline > 50 else 0
    
    # Late in half indicator (less than 2 minutes)
    late_in_half = 1 if time_left_in_half < 120 else 0
    
    # End of half/game (Q2 or Q4 with low time)
    end_of_half = 1 if (quarter in [2, 4] and time_left_in_half < 300) else 0
    
    # Score differential categories
    down_by_more_than_3 = 1 if score_differential < -3 else 0
    up_by_a_lot = 1 if score_differential > 14 else 0
    
    # Yardline > 40 indicator (FG and punt logic)
    yardline_over_40 = 1 if yardline > 40 else 0
    
    # Non-linear yardline features to capture threshold effects
    distance_from_40 = float(yardline) - 40.0
    distance_from_40_squared = distance_from_40 ** 2
    inside_40 = 1 if yardline <= 40 else 0
    yardline_inside_40 = float(yardline) * inside_40
    around_40 = 1 if (yardline >= 35 and yardline <= 45) else 0
    beyond_40 = 1 if yardline > 40 else 0
    yardline_beyond_40 = (float(yardline) - 40) * beyond_40
    
    # Interaction: own territory AND not late in half (should reduce go-for-it WP)
    own_territory_not_late = 1 if (own_territory == 1 and late_in_half == 0) else 0
    
    # Interaction: down by more than 3 AND late in half (should increase go-for-it WP)
    down_late = 1 if (down_by_more_than_3 == 1 and late_in_half == 1) else 0
    
    return {
        'yardline_100': float(yardline),
        'own_territory': own_territory,
        'time_left_in_half': float(time_left_in_half),
        'late_in_half': late_in_half,
        'end_of_half': end_of_half,
        'score_differential': float(score_differential),
        'down_by_more_than_3': down_by_more_than_3,
        'down_late': down_late,
        'own_territory_not_late': own_territory_not_late,
        'ydstogo': float(yds_to_go),
        'offensive_rating': float(offense_strength),
        'opponent_defensive_rating': float(opp_defense_strength),
        'yardline_over_40': yardline_over_40,
        'distance_from_40': distance_from_40,
        'distance_from_40_squared': distance_from_40_squared,
        'inside_40': inside_40,
        'around_40': around_40,
        'yardline_beyond_40': yardline_beyond_40,
        'wind': float(wind),
    }

# =========================
# Load Models
# =========================
print("Loading models...")
go_wp_model = joblib.load(GO_WP_MODEL_PATH)
fg_wp_model = joblib.load(FG_WP_MODEL_PATH)
punt_wp_model = joblib.load(PUNT_WP_MODEL_PATH)
print("Models loaded successfully!\n")

# =========================
# Create Features from Inputs
# =========================
features = create_features_from_inputs(
    yardline=Yardline,
    quarter=Quarter,
    time_remaining=Time_remaining,
    score_differential=Score_differential,
    yds_to_go=Yds_to_go,
    wind=Wind_speed,
    offense_strength=Offense_strength,
    opp_defense_strength=Opp_defense_strength
)

# =========================
# Predict WP for Each Decision
# =========================
# Go-for-it WP Model features
GO_FEATURES = [
    'yardline_100',
    'own_territory',
    'time_left_in_half',
    'end_of_half',
    'score_differential',
    'down_by_more_than_3',
    'down_late',
    'own_territory_not_late',
    'ydstogo',
    'offensive_rating',
    'opponent_defensive_rating'
]

# Field Goal WP Model features - includes non-linear yardline features
FG_FEATURES = [
    'yardline_100',
    'yardline_over_40',
    'distance_from_40',
    'distance_from_40_squared',
    'inside_40',
    'around_40',
    'yardline_beyond_40',
    'time_left_in_half',
    'end_of_half',
    'score_differential',
    'down_by_more_than_3',
    'wind'
]

# Punt WP Model features
PUNT_FEATURES = [
    'yardline_100',
    'yardline_over_40',
    'time_left_in_half',
    'end_of_half',
    'score_differential'
]

# Create DataFrames for predictions
go_input = pd.DataFrame([{k: features[k] for k in GO_FEATURES}])
fg_input = pd.DataFrame([{k: features[k] for k in FG_FEATURES}])
punt_input = pd.DataFrame([{k: features[k] for k in PUNT_FEATURES}])

# Predict WP values
wp_go = float(np.clip(go_wp_model.predict(go_input)[0], 0.0, 1.0))
wp_fg = float(np.clip(fg_wp_model.predict(fg_input)[0], 0.0, 1.0))
wp_punt = float(np.clip(punt_wp_model.predict(punt_input)[0], 0.0, 1.0))

# =========================
# Enforce Constraints
# =========================
# Field goal should always be better than punt inside the 40 yardline
constraint_applied = False
if Yardline < 40:
    original_fg_wp = wp_fg
    # Ensure FG WP is at least as high as Punt WP
    wp_fg = max(wp_fg, wp_punt)
    if original_fg_wp < wp_punt:
        constraint_applied = True

# =========================
# Display Results
# =========================
best = max(wp_go, wp_punt, wp_fg)
rec = "GO" if best == wp_go else ("PUNT" if best == wp_punt else "FG")

print("=" * 70)
print("4TH DOWN DECISION ANALYSIS")
print("=" * 70)
print(f"\nScenario:")
print(f"  Yardline: {Yardline:.1f} (distance from opponent end zone)")
print(f"  Quarter: {Quarter:.0f}")
print(f"  Time Remaining: {Time_remaining:.0f} seconds")
print(f"  Yards to Go: {Yds_to_go:.1f}")
print(f"  Score Differential (Ravens - Opp): {Score_differential:+.1f}")
print(f"  Wind Speed: {Wind_speed:.1f} mph")
print(f"  Offense Strength: {Offense_strength:+.3f}")
print(f"  Opp Defense Strength: {Opp_defense_strength:+.3f}")
print(f"\nDerived Features:")
print(f"  Own Territory: {features['own_territory']} (yardline > 50)")
print(f"  Time Left in Half: {features['time_left_in_half']:.0f} seconds")
print(f"  Late in Half: {features['late_in_half']} (< 2 minutes)")
print(f"  End of Half: {features['end_of_half']} (Q2/Q4 with < 5 min)")
print(f"  Down by >3: {features['down_by_more_than_3']} (score_diff < -3)")
print(f"  Yardline > 40: {features['yardline_over_40']}")
print("\n" + "-" * 70)
print("EXPECTED WIN PROBABILITY AFTER EACH DECISION:")
if constraint_applied:
    print("  (Note: FG WP adjusted to be >= Punt WP for yardline < 40)")
print("-" * 70)
print(f"  GO FOR IT:   {wp_go:.4f} ({wp_go*100:.2f}%)")
print(f"  PUNT:        {wp_punt:.4f} ({wp_punt*100:.2f}%)")
print(f"  FIELD GOAL:  {wp_fg:.4f} ({wp_fg*100:.2f}%)")
print("-" * 70)
print(f"\nRECOMMENDATION: {rec}")
print(f"\nRegret Analysis (difference from best option):")
print(f"  If choose GO:   {best - wp_go:+.4f} ({((best - wp_go)*100):+.2f}%)")
print(f"  If choose PUNT: {best - wp_punt:+.4f} ({((best - wp_punt)*100):+.2f}%)")
print(f"  If choose FG:   {best - wp_fg:+.4f} ({((best - wp_fg)*100):+.2f}%)")
print("=" * 70)
