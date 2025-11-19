# mvp_4th_down_bot.py
# Train models for 4th-down decision making:
#  - Go-for-it WP model: predicts win probability after going for it
#  - Field Goal WP model: predicts win probability after attempting FG
#  - Punt WP model: predicts win probability after punting
#
# Requirements:
#   pip install pandas numpy scikit-learn joblib

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# ----------------------------
# Config
# ----------------------------
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

GO_WP_MODEL_PATH = MODEL_DIR / "go_wp_model.joblib"
FG_WP_MODEL_PATH = MODEL_DIR / "fg_wp_model.joblib"
PUNT_WP_MODEL_PATH = MODEL_DIR / "punt_wp_model.joblib"

RANDOM_STATE = 42

# ----------------------------
# Load Ravens 4th down data
# ----------------------------
print("Loading Ravens 4th down data...")
ravens_full = pd.read_csv("ravens_4th_down_2016_onwards.csv")
ravens_inputs = pd.read_csv("ravens_4th_down_inputs.csv")

print(f"Loaded {len(ravens_full)} plays from full dataset")
print(f"Loaded {len(ravens_inputs)} input scenarios")

# ----------------------------
# Feature Engineering
# ----------------------------
def create_features(df):
    """Create engineered features for the models."""
    df = df.copy()
    
    # Time left in half (combination of quarter and time remaining)
    # For Q1/Q2: use quarter_seconds_remaining (first half)
    # For Q3/Q4: use game_seconds_remaining (second half/end of game)
    df['time_left_in_half'] = df.apply(
        lambda row: row['quarter_seconds_remaining'] if row['qtr'] in [1, 2] 
        else row['game_seconds_remaining'], axis=1
    )
    
    # Own territory indicator (yardline > 50 means own territory)
    df['own_territory'] = (df['yardline_100'] > 50).astype(int)
    
    # Late in half indicator (less than 2 minutes)
    df['late_in_half'] = (df['time_left_in_half'] < 120).astype(int)
    
    # End of half/game (Q2 or Q4 with low time)
    df['end_of_half'] = ((df['qtr'].isin([2, 4])) & (df['time_left_in_half'] < 300)).astype(int)
    
    # Score differential categories
    df['down_by_more_than_3'] = (df['score_differential'] < -3).astype(int)
    df['up_by_a_lot'] = (df['score_differential'] > 14).astype(int)
    
    # Yardline > 40 indicator (FG and punt logic)
    df['yardline_over_40'] = (df['yardline_100'] > 40).astype(int)
    
    # Non-linear yardline features to capture threshold effects
    # Distance from 40 yardline (captures the critical threshold)
    df['distance_from_40'] = df['yardline_100'] - 40.0
    
    # Squared distance from 40 (emphasizes the threshold region)
    df['distance_from_40_squared'] = df['distance_from_40'] ** 2
    
    # Piecewise features for different yardline ranges
    # Inside 40 (FG range): high value, changes slowly
    df['inside_40'] = (df['yardline_100'] <= 40).astype(int)
    df['yardline_inside_40'] = df['yardline_100'] * df['inside_40']  # Only non-zero inside 40
    
    # Around 40 (critical transition zone): 35-45 yardline
    df['around_40'] = ((df['yardline_100'] >= 35) & (df['yardline_100'] <= 45)).astype(int)
    
    # Beyond 40 (long FG range): changes rapidly
    df['beyond_40'] = (df['yardline_100'] > 40).astype(int)
    df['yardline_beyond_40'] = (df['yardline_100'] - 40) * df['beyond_40']  # Distance beyond 40
    
    # Interaction: own territory AND not late in half (should reduce go-for-it WP)
    df['own_territory_not_late'] = (df['own_territory'] == 1) & (df['late_in_half'] == 0)
    df['own_territory_not_late'] = df['own_territory_not_late'].astype(int)
    
    # Interaction: down by more than 3 AND late in half (should increase go-for-it WP)
    df['down_late'] = (df['down_by_more_than_3'] == 1) & (df['late_in_half'] == 1)
    df['down_late'] = df['down_late'].astype(int)
    
    # Fill missing values for ratings
    if 'offensive_rating' not in df.columns:
        df['offensive_rating'] = 0.0
    if 'opponent_defensive_rating' not in df.columns:
        df['opponent_defensive_rating'] = 0.0
    
    # Fill missing wind
    if 'wind' not in df.columns:
        df['wind'] = 0.0
    df['wind'] = df['wind'].fillna(0.0)
    
    return df

ravens_full = create_features(ravens_full)

# ----------------------------
# 1) Go-for-it WP Model
# Features: yardline, own_territory, time_left_in_half, end_of_half, 
#           score_differential, down_by_more_than_3, yds_to_go, 
#           offensive_rating, opponent_defensive_rating
# ----------------------------
print("\n" + "="*60)
print("Training Go-for-it WP Model")
print("="*60)

# Get actual 4th down attempts (run or pass)
# Note: All plays in this dataset are 4th downs
go_attempts = ravens_full[
    ravens_full['play_type'].isin(['run', 'pass'])
].copy()

print(f"Found {len(go_attempts)} 4th down go-for-it attempts")

if len(go_attempts) > 0:
    # Calculate WP after the attempt
    # For successful conversions, WP should increase
    # For failed conversions, WP should decrease
    # We'll use the next play's WP or estimate from WPA
    go_attempts['wp_after'] = go_attempts['wp'] + go_attempts['wpa'].fillna(0)
    go_attempts['wp_after'] = go_attempts['wp_after'].clip(0, 1)
    
    # Features for go-for-it model
    GO_FEATURES = [
        'yardline_100',
        'own_territory',
        'time_left_in_half',
        'end_of_half',
        'score_differential',
        'down_by_more_than_3',
        'down_late',  # Interaction: down by >3 AND late in half
        'own_territory_not_late',  # Interaction: own territory AND not late
        'ydstogo',
        'offensive_rating',
        'opponent_defensive_rating'
    ]
    
    # Prepare data
    go_X = go_attempts[GO_FEATURES].copy()
    go_y = go_attempts['wp_after'].copy()
    
    # Remove rows with missing critical features
    mask = go_X[['yardline_100', 'ydstogo', 'time_left_in_half']].notna().all(axis=1)
    go_X = go_X[mask]
    go_y = go_y[mask]
    
    if len(go_X) > 10:
        go_pipeline = Pipeline(steps=[
            ("prep", ColumnTransformer(
                transformers=[("num", SimpleImputer(strategy="median"), GO_FEATURES)],
                remainder="drop"
            )),
            ("model", GradientBoostingRegressor(
                random_state=RANDOM_STATE,
                n_estimators=400,  # More trees for better non-linear capture
                learning_rate=0.04,  # Slightly lower learning rate with more trees
                max_depth=5,  # Deeper trees to capture complex interactions
                subsample=0.9,
                min_samples_split=15  # Allow more splits for threshold effects
            ))
        ])
        
        Xtr, Xte, ytr, yte = train_test_split(go_X, go_y, test_size=0.2, random_state=RANDOM_STATE)
        go_pipeline.fit(Xtr, ytr)
        pred_te = np.clip(go_pipeline.predict(Xte), 0, 1)
        mae = mean_absolute_error(yte, pred_te)
        print(f"Go-for-it WP Model - Test MAE: {mae:.4f}")
        
        joblib.dump(go_pipeline, GO_WP_MODEL_PATH)
        print(f"Saved Go-for-it WP model → {GO_WP_MODEL_PATH}")
    else:
        print("Warning: Not enough go-for-it attempts to train model")
        go_pipeline = None
else:
    print("Warning: No go-for-it attempts found in data")
    go_pipeline = None

# ----------------------------
# 2) Field Goal WP Model
# Features: yardline, time_left_in_half, end_of_half, score_differential,
#           down_by_more_than_3, wind
# NOT: yds_to_go, timeouts
# ----------------------------
print("\n" + "="*60)
print("Training Field Goal WP Model")
print("="*60)

# Get actual field goal attempts
fg_attempts = ravens_full[
    (ravens_full['field_goal_attempt'] == 1)
].copy()

print(f"Found {len(fg_attempts)} field goal attempts")

if len(fg_attempts) > 0:
    # Calculate WP after FG attempt
    # If made: WP increases (score +3, better field position)
    # If missed: WP decreases (turnover on downs, worse field position)
    fg_attempts['wp_after'] = fg_attempts['wp'] + fg_attempts['wpa'].fillna(0)
    fg_attempts['wp_after'] = fg_attempts['wp_after'].clip(0, 1)
    
    # Features for FG model - includes non-linear yardline features
    # to capture the threshold effect around the 40 yardline
    FG_FEATURES = [
        'yardline_100',
        'yardline_over_40',  # Indicator for yardline > 40 (usually punt better)
        'distance_from_40',  # Linear distance from 40 (captures threshold)
        'distance_from_40_squared',  # Squared distance (emphasizes threshold region)
        'inside_40',  # Indicator for inside FG range
        'around_40',  # Indicator for critical transition zone (35-45)
        'yardline_beyond_40',  # Distance beyond 40 (rapidly changing region)
        'time_left_in_half',
        'end_of_half',
        'score_differential',
        'down_by_more_than_3',
        'wind'
    ]
    
    # Prepare data
    fg_X = fg_attempts[FG_FEATURES].copy()
    fg_y = fg_attempts['wp_after'].copy()
    
    # Remove rows with missing critical features
    mask = fg_X[['yardline_100', 'time_left_in_half']].notna().all(axis=1)
    fg_X = fg_X[mask]
    fg_y = fg_y[mask]
    
    if len(fg_X) > 10:
        fg_pipeline = Pipeline(steps=[
            ("prep", ColumnTransformer(
                transformers=[("num", SimpleImputer(strategy="median"), FG_FEATURES)],
                remainder="drop"
            )),
            ("model", GradientBoostingRegressor(
                random_state=RANDOM_STATE,
                n_estimators=400,  # More trees for better non-linear capture
                learning_rate=0.04,  # Slightly lower learning rate with more trees
                max_depth=5,  # Deeper trees to capture complex interactions
                subsample=0.9,
                min_samples_split=15  # Allow more splits for threshold effects
            ))
        ])
        
        Xtr, Xte, ytr, yte = train_test_split(fg_X, fg_y, test_size=0.2, random_state=RANDOM_STATE)
        fg_pipeline.fit(Xtr, ytr)
        pred_te = np.clip(fg_pipeline.predict(Xte), 0, 1)
        mae = mean_absolute_error(yte, pred_te)
        print(f"Field Goal WP Model - Test MAE: {mae:.4f}")
        
        joblib.dump(fg_pipeline, FG_WP_MODEL_PATH)
        print(f"Saved Field Goal WP model → {FG_WP_MODEL_PATH}")
    else:
        print("Warning: Not enough FG attempts to train model")
        fg_pipeline = None
else:
    print("Warning: No field goal attempts found in data")
    fg_pipeline = None

# ----------------------------
# 3) Punt WP Model
# Features: yardline, time_left_in_half, end_of_half, score_differential
# NOT: yds_to_go, timeouts, wind
# ----------------------------
print("\n" + "="*60)
print("Training Punt WP Model")
print("="*60)

# Get actual punt attempts
punt_attempts = ravens_full[
    (ravens_full['punt_attempt'] == 1)
].copy()

print(f"Found {len(punt_attempts)} punt attempts")

if len(punt_attempts) > 0:
    # Calculate WP after punt
    punt_attempts['wp_after'] = punt_attempts['wp'] + punt_attempts['wpa'].fillna(0)
    punt_attempts['wp_after'] = punt_attempts['wp_after'].clip(0, 1)
    
    # Features for punt model
    PUNT_FEATURES = [
        'yardline_100',
        'yardline_over_40',  # Indicator for yardline > 40 (usually punt)
        'time_left_in_half',
        'end_of_half',
        'score_differential'
    ]
    
    # Prepare data
    punt_X = punt_attempts[PUNT_FEATURES].copy()
    punt_y = punt_attempts['wp_after'].copy()
    
    # Remove rows with missing critical features
    mask = punt_X[['yardline_100', 'time_left_in_half']].notna().all(axis=1)
    punt_X = punt_X[mask]
    punt_y = punt_y[mask]
    
    if len(punt_X) > 10:
        punt_pipeline = Pipeline(steps=[
            ("prep", ColumnTransformer(
                transformers=[("num", SimpleImputer(strategy="median"), PUNT_FEATURES)],
                remainder="drop"
            )),
            ("model", GradientBoostingRegressor(
                random_state=RANDOM_STATE,
                n_estimators=400,  # More trees for better non-linear capture
                learning_rate=0.04,  # Slightly lower learning rate with more trees
                max_depth=5,  # Deeper trees to capture complex interactions
                subsample=0.9,
                min_samples_split=15  # Allow more splits for threshold effects
            ))
        ])
        
        Xtr, Xte, ytr, yte = train_test_split(punt_X, punt_y, test_size=0.2, random_state=RANDOM_STATE)
        punt_pipeline.fit(Xtr, ytr)
        pred_te = np.clip(punt_pipeline.predict(Xte), 0, 1)
        mae = mean_absolute_error(yte, pred_te)
        print(f"Punt WP Model - Test MAE: {mae:.4f}")
        
        joblib.dump(punt_pipeline, PUNT_WP_MODEL_PATH)
        print(f"Saved Punt WP model → {PUNT_WP_MODEL_PATH}")
    else:
        print("Warning: Not enough punt attempts to train model")
        punt_pipeline = None
else:
    print("Warning: No punt attempts found in data")
    punt_pipeline = None

print("\n" + "="*60)
print("Training Complete!")
print("="*60)
print(f"\nModels saved to: {MODEL_DIR}/")
print("  - go_wp_model.joblib")
print("  - fg_wp_model.joblib")
print("  - punt_wp_model.joblib")
