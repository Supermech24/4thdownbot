# mvp_4th_down_bot.py
# Train models for a 4th-down bot:
#  - WP model: maps game state -> win probability for current offense
#  - FG model: P(make | distance, wind, temp, roof)
#  - GO model: P(convert | yardline_100, ydstogo, qtr, time, ratings) using ONLY 4th-down run/pass attempts
#  - Punt sampler: historical gross punt distribution (MVP)
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# ----------------------------
# Config
# ----------------------------
SEASONS = list(range(2010, 2025))
PBP_CACHE = Path("pbp_2010_onwards.parquet")  # cached all-team play-by-play

MODEL_DIR = Path("models"); MODEL_DIR.mkdir(exist_ok=True)
WP_MODEL_PATH  = MODEL_DIR / "wp_model.joblib"
FG_MODEL_PATH  = MODEL_DIR / "fg_model.joblib"
GO_MODEL_PATH  = MODEL_DIR / "go_conv_model.joblib"
PUNT_DIST_PATH = MODEL_DIR / "punt_gross_dist.npy"

# Features expected by models
WP_FEATURES = [
    "yardline_100",
    "qtr",
    "game_seconds_remaining",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "score_differential",
    "offensive_rating",
    "opponent_defensive_rating",
]

FG_FEATURES = ["kick_distance", "temp", "wind", "roof_is_indoor"]

GO_FEATURES = [
    "yardline_100",
    "ydstogo",
    "qtr",
    "game_seconds_remaining",
    "score_differential",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "offensive_rating",
    "opponent_defensive_rating",
]

RANDOM_STATE = 42

# ----------------------------
# Load plays
# ----------------------------
if PBP_CACHE.exists():
    print(f"Loading cached play-by-play from {PBP_CACHE} ...")
    plays = pd.read_parquet(PBP_CACHE)
else:
    print(f"Downloading play-by-play via nflreadpy for seasons {SEASONS} ...")
    import nflreadpy as nfl

    pbp = nfl.load_pbp(seasons=SEASONS)
    plays = pbp.to_pandas()

    keep_cols = [
        "season",
        "game_id",
        "play_id",
        "week",
        "down",
        "play_type",
        "first_down",
        "touchdown",
        "yards_gained",
        "yardline_100",
        "ydstogo",
        "qtr",
        "time",
        "quarter_seconds_remaining",
        "game_seconds_remaining",
        "posteam",
        "defteam",
        "posteam_type",
        "posteam_score",
        "defteam_score",
        "score_differential",
        "posteam_timeouts_remaining",
        "defteam_timeouts_remaining",
        "home_team",
        "away_team",
        "field_goal_attempt",
        "field_goal_result",
        "kick_distance",
        "punt_attempt",
        "weather",
        "temp",
        "wind",
        "stadium",
        "surface",
        "roof",
        "desc",
        "wp",
        "def_wp",
        "wpa",
        "ep",
        "epa",
    ]
    existing_cols = [c for c in keep_cols if c in plays.columns]
    plays = plays[existing_cols].copy()

    print(f"Caching filtered play-by-play to {PBP_CACHE} ...")
    plays.to_parquet(PBP_CACHE, index=False)

# Add optional columns if missing
for col in ["offensive_rating", "opponent_defensive_rating"]:
    if col not in plays.columns:
        plays[col] = 0.0

# Roof indicator (indoor/dome/closed => 1)
if "roof" in plays.columns:
    plays["roof_is_indoor"] = plays["roof"].astype(str).str.lower().isin(["indoors", "dome", "closed"]).astype(float)
else:
    plays["roof_is_indoor"] = 0.0

# Ensure wind/temp exist (may be NaN)
for col in ["temp", "wind"]:
    if col not in plays.columns:
        plays[col] = np.nan

# ----------------------------
# 1) Win Probability (WP) model
#     Target = pre-play wp (nflfastR), calibrated and quick to learn from
# ----------------------------
wp_df = plays[pd.notnull(plays.get("wp"))].copy()
wp_df["wp"] = wp_df["wp"].clip(0, 1)

for f in WP_FEATURES:
    if f not in wp_df.columns:
        wp_df[f] = np.nan

X_wp = wp_df[WP_FEATURES]
y_wp = wp_df["wp"].astype(float)

wp_pipeline = Pipeline(steps=[
    ("prep", ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median"), WP_FEATURES)],
        remainder="drop"
    )),
    ("model", GradientBoostingRegressor(
        random_state=RANDOM_STATE, n_estimators=500, learning_rate=0.03, max_depth=3, subsample=0.9
    ))
])

Xtr, Xte, ytr, yte = train_test_split(X_wp, y_wp, test_size=0.2, random_state=RANDOM_STATE)
wp_pipeline.fit(Xtr, ytr)
pred_te = np.clip(wp_pipeline.predict(Xte), 0, 1)
print(f"[WP] Test MAE={mean_absolute_error(yte, pred_te):.4f}")
joblib.dump(wp_pipeline, WP_MODEL_PATH)
print(f"Saved WP model → {WP_MODEL_PATH}")

def predict_wp_from_state(state_dict):
    row = pd.DataFrame([state_dict])
    for k in WP_FEATURES:
        if k not in row.columns:
            row[k] = np.nan
    wp_hat = wp_pipeline.predict(row[WP_FEATURES])[0]
    return float(np.clip(wp_hat, 0.0, 1.0))

# ----------------------------
# 2) Field Goal make model (logistic)
# ----------------------------
fg = plays[(plays.get("field_goal_attempt", 0) == 1)].copy()
fg["made"] = (fg.get("field_goal_result", "").astype(str).str.lower() == "made").astype(int)
fg = fg[pd.notnull(fg["kick_distance"])].copy()

if len(fg) < 50:
    print("Warning: Very few FG attempts in data; FG model may be weak.")

fg_X = fg.assign(
    wind=fg["wind"],
    temp=fg["temp"],
    roof_is_indoor=fg["roof_is_indoor"],
)[FG_FEATURES]
fg_y = fg["made"]

fg_pipeline = Pipeline(steps=[
    ("prep", ColumnTransformer(
        transformers=[("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                        ("sc", StandardScaler())]), FG_FEATURES)],
        remainder="drop"
    )),
    ("clf", LogisticRegression(max_iter=300, solver="lbfgs"))
])
fg_pipeline.fit(fg_X, fg_y)

for d in [30, 45, 55, 65, 75]:
    test_row = pd.DataFrame([{"kick_distance": d, "temp": 60.0, "wind": 5.0, "roof_is_indoor": 0.0}])
    p = fg_pipeline.predict_proba(test_row)[0,1]
    print(f"[FG] P(make) at {d} yds ≈ {p:.3f}")

joblib.dump(fg_pipeline, FG_MODEL_PATH)
print(f"Saved FG model → {FG_MODEL_PATH}")

def p_fg_make_from_yardline(yardline_100, wind=0.0, temp=60.0, roof_is_indoor=0.0):
    dist = float(yardline_100) + 17.0  # LOS yardline_100 + 17 yards
    row = pd.DataFrame([{"kick_distance": dist, "temp": temp, "wind": wind, "roof_is_indoor": roof_is_indoor}])
    p = fg_pipeline.predict_proba(row)[0,1]
    return float(p), dist

# ----------------------------
# 3) Go-for-it conversion model (STRICT 4th-down run/pass only)
# ----------------------------
if "down" not in plays.columns:
    raise ValueError("Your plays CSV must include a 'down' column to train the 4th-down conversion model correctly.")

go = plays.copy()
go = go[go["down"] == 4]                          # 4th downs only
go = go[go["play_type"].isin(["run", "pass"])].copy()  # actual attempts only

# Exclude 'No Play' penalties/pre-snap
if "desc" in go.columns:
    go = go[~go["desc"].str.contains("No Play", case=False, na=False)]

# Feature hygiene
need = ["yardline_100", "ydstogo", "qtr", "game_seconds_remaining"]
for c in need:
    go = go[pd.notnull(go[c])]

# Success label: got a new first down or a TD
go["success"] = ((go.get("first_down", 0) == 1) | (go.get("touchdown", 0) == 1)).astype(int)

# Ratings defaults
for c in ["offensive_rating", "opponent_defensive_rating"]:
    if c not in go.columns: go[c] = 0.0

go_X = go[GO_FEATURES].copy()
go_y = go["success"]

go_pipeline = Pipeline(steps=[
    ("prep", ColumnTransformer(
        transformers=[("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), GO_FEATURES)],
        remainder="drop"
    )),
    ("clf", LogisticRegression(
        max_iter=1500,
        solver="lbfgs",
        C=0.7,
    )),
])
go_pipeline.fit(go_X, go_y)
joblib.dump(go_pipeline, GO_MODEL_PATH)
print(f"Saved Go conversion model → {GO_MODEL_PATH}")

clf = go_pipeline.named_steps["clf"]
if hasattr(clf, "coef_"):
    print("Go-model coefficients:")
    for name, coef in zip(GO_FEATURES, clf.coef_[0]):
        print(f"  {name:>26s} : {coef:+.4f}")

def p_go_convert(state):
    r = pd.DataFrame([state])
    for k in GO_FEATURES:
        if k not in r.columns:
            r[k] = np.nan
    return float(go_pipeline.predict_proba(r[GO_FEATURES])[0,1])

# ----------------------------
# 4) Punt gross distance sampler (MVP)
# ----------------------------
punts = plays[(plays.get("punt_attempt", 0) == 1) & pd.notnull(plays.get("kick_distance"))].copy()
gross = punts["kick_distance"].astype(float).values
if len(gross) < 10:
    # fallback typical NFL gross punt distribution
    gross = np.array([38, 40, 42, 44, 45, 46, 48, 50, 52], dtype=float)

np.save(PUNT_DIST_PATH, gross)
print(f"Saved punt gross distribution (n={len(gross)}) → {PUNT_DIST_PATH}")

def sample_punt_gross(n=1, rng=None):
    rng = np.random.default_rng(rng)
    arr = np.load(PUNT_DIST_PATH)
    return rng.choice(arr, size=n, replace=True)
