"""
Reworked 4th-down bot training with conservative bias and football sanity.

Architecture (same as before, but with stronger domain heuristics):

1) WP state model (Ravens perspective):
   Input: Yardline, Yds_to_go, Quarter, Time_remaining,
          Score_differential, Offense_strength, Opp_defense_strength
   Target: wp (Ravens win probability from data)

2) Action models:
   - Go conversion model: P(convert | go for it on 4th)
   - Field goal make model: P(make FG)
   - Punt net distance model: expected net yards

3) For a scenario, simulate:
   - WP_go (with conservative bias in "normal" spots)
   - WP_fg (with hard penalty for long FGs: Yardline > 45)
   - WP_punt (with penalty inside opp 40, i.e. Yardline < 40)

Inputs for scenarios (match ravens_4th_down_inputs.csv):
   Yardline (distance from opponent EZ, 1–99; 1 = at opp goal line)
   Quarter
   Yds_to_go
   Time_remaining (seconds left in game)
   Wind_speed
   Score_differential (Ravens - Opp)
   Offense_strength (0 = league avg)
   Opp_defense_strength (0 = league avg)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ----------------------------
# Config
# ----------------------------
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

WP_MODEL_PATH = MODEL_DIR / "wp_state_model.joblib"
GO_CONVERT_MODEL_PATH = MODEL_DIR / "go_convert_model.joblib"
FG_MAKE_MODEL_PATH = MODEL_DIR / "fg_make_model.joblib"
PUNT_NET_MODEL_PATH = MODEL_DIR / "punt_net_model.joblib"

INPUTS_PATH = "ravens_4th_down_inputs.csv"
FULL_PATH = "ravens_4th_down_2016_onwards.csv"


# ----------------------------
# Load and prepare full dataset
# ----------------------------
def load_full_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Standardized features
    df["Yardline"] = df["yardline_100"]
    df["Quarter"] = df["qtr"]
    df["Time_remaining"] = df["game_seconds_remaining"]
    df["Yds_to_go"] = df["ydstogo"]
    df["Score_differential"] = df["score_differential"]

    # Wind speed numeric
    if "wind" in df.columns:
        wind_numeric = pd.to_numeric(df["wind"], errors="coerce")
        df["Wind_speed"] = wind_numeric.fillna(0.0)
    else:
        df["Wind_speed"] = 0.0

    if "offensive_rating" in df.columns:
        df["Offense_strength"] = df["offensive_rating"].fillna(0.0)
    else:
        df["Offense_strength"] = 0.0

    if "opponent_defensive_rating" in df.columns:
        df["Opp_defense_strength"] = df["opponent_defensive_rating"].fillna(0.0)
    else:
        df["Opp_defense_strength"] = 0.0

    return df


# ----------------------------
# 1) WP state model
# ----------------------------
def build_wp_model() -> tuple[Pipeline, list[str]]:
    features = [
        "Yardline",
        "Yds_to_go",
        "Quarter",
        "Time_remaining",
        "Score_differential",
        "Offense_strength",
        "Opp_defense_strength",
    ]

    prep = ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median"), features)],
        remainder="drop",
    )

    gb = GradientBoostingRegressor(
        random_state=42,
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        min_samples_split=20,
    )

    return Pipeline(steps=[("prep", prep), ("model", gb)]), features


def train_wp_model(df: pd.DataFrame) -> dict:
    """
    Train WP model from Ravens perspective.

    Each row is a Ravens 4th-down play, and `wp` is the Ravens win probability
    before the play. We learn:

        (Yardline, Yds_to_go, Quarter, Time_remaining,
         Score_differential, Offense_strength, Opp_defense_strength)
      → Ravens_wp
    """
    df = df.copy()

    if "wp" not in df.columns:
        raise ValueError("Expected 'wp' column in full dataset.")

    df["Ravens_wp"] = df["wp"].clip(0.0, 1.0)

    wp_model, features = build_wp_model()

    X = df[features]
    y = df["Ravens_wp"]

    valid = X.notna().all(axis=1) & y.notna()
    X = X[valid]
    y = y[valid]

    if len(X) < 100:
        print(f"WARNING: Only {len(X)} rows for WP model.")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    wp_model.fit(Xtr, ytr)
    preds = np.clip(wp_model.predict(Xte), 0.0, 1.0)
    mae = mean_absolute_error(yte, preds)
    print(f"WP state model trained on {len(X)} rows, Test MAE = {mae:.4f}")

    bundle = {"model": wp_model, "features": features}
    joblib.dump(bundle, WP_MODEL_PATH)
    print(f"Saved WP state model → {WP_MODEL_PATH}")

    return bundle


# ----------------------------
# 2) Go conversion model (P(convert))
# ----------------------------
def build_go_convert_model() -> tuple[Pipeline, list[str]]:
    features = [
        "Yardline",
        "Yds_to_go",
        "Quarter",
        "Time_remaining",
        "Score_differential",
        "Offense_strength",
        "Opp_defense_strength",
    ]

    prep = ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median"), features)],
        remainder="drop",
    )

    gb = GradientBoostingClassifier(
        random_state=42,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        min_samples_split=15,
    )

    return Pipeline(steps=[("prep", prep), ("model", gb)]), features


def train_go_convert_model(df: pd.DataFrame) -> dict:
    """
    Train P(convert) on 4th-down runs/passes (go-for-it attempts).
    Label: success if first_down == 1 OR yards_gained >= ydstogo.
    """
    df = df.copy()

    mask = df["play_type"].isin(["run", "pass"])
    sub = df.loc[mask].copy()
    print(f"Go conversion: found {len(sub)} run/pass plays.")

    if len(sub) == 0:
        raise ValueError("No run/pass plays for go conversion model.")

    if "first_down" in sub.columns:
        success = (sub["first_down"] == 1) | (sub["yards_gained"] >= sub["ydstogo"])
    else:
        success = sub["yards_gained"] >= sub["ydstogo"]

    sub["go_success"] = success.astype(int)

    model, features = build_go_convert_model()
    X = sub[features]
    y = sub["go_success"]

    valid = X.notna().all(axis=1)
    X = X[valid]
    y = y[valid]

    if len(X) < 100:
        print("WARNING: Not much data for go conversion model.")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(Xtr, ytr)

    proba = model.predict_proba(Xte)[:, 1]
    try:
        auc = roc_auc_score(yte, proba)
        print(f"Go convert model AUC = {auc:.3f}")
    except Exception:
        pass

    bundle = {"model": model, "features": features}
    joblib.dump(bundle, GO_CONVERT_MODEL_PATH)
    print(f"Saved Go conversion model → {GO_CONVERT_MODEL_PATH}")

    return bundle


# ----------------------------
# 3) FG make model (P(make))
# ----------------------------
def build_fg_make_model() -> tuple[Pipeline, list[str]]:
    # Score differential does *not* affect physical make prob, so exclude it here.
    features = [
        "Yardline",
        "Quarter",
        "Time_remaining",
        "Wind_speed",
    ]

    prep = ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median"), features)],
        remainder="drop",
    )

    gb = GradientBoostingClassifier(
        random_state=42,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        min_samples_split=15,
    )

    return Pipeline(steps=[("prep", prep), ("model", gb)]), features


def train_fg_make_model(df: pd.DataFrame) -> dict:
    df = df.copy()
    mask = df["field_goal_attempt"] == 1
    sub = df.loc[mask].copy()
    print(f"FG make: found {len(sub)} FG attempts.")

    if len(sub) == 0:
        raise ValueError("No FG attempts found in dataset.")

    made = sub["field_goal_result"].astype(str).str.lower().isin(
        ["good", "made", "1"]
    )
    sub["fg_make"] = made.astype(int)

    model, features = build_fg_make_model()
    X = sub[features]
    y = sub["fg_make"]

    valid = X.notna().all(axis=1)
    X = X[valid]
    y = y[valid]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    try:
        auc = roc_auc_score(yte, proba)
        print(f"FG make model AUC = {auc:.3f}")
    except Exception:
        pass

    bundle = {"model": model, "features": features}
    joblib.dump(bundle, FG_MAKE_MODEL_PATH)
    print(f"Saved FG make model → {FG_MAKE_MODEL_PATH}")

    return bundle


# ----------------------------
# 4) Punt net yards model
# ----------------------------
def build_punt_net_model() -> tuple[Pipeline, list[str]]:
    # Net distance does not depend on score; just basic game context.
    features = [
        "Yardline",
        "Quarter",
        "Time_remaining",
    ]

    prep = ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median"), features)],
        remainder="drop",
    )

    gb = GradientBoostingRegressor(
        random_state=42,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        min_samples_split=15,
    )

    return Pipeline(steps=[("prep", prep), ("model", gb)]), features


def train_punt_net_model(df: pd.DataFrame) -> dict:
    df = df.copy()
    mask = df["punt_attempt"] == 1
    sub = df.loc[mask].copy()
    print(f"Punt net: found {len(sub)} punts.")

    if len(sub) == 0:
        raise ValueError("No punts found in dataset.")

    # Approx net yards: use yards_gained if available, else ~40.
    if "yards_gained" in sub.columns:
        sub["net_yards"] = sub["yards_gained"]
    else:
        sub["net_yards"] = 40.0

    model, features = build_punt_net_model()
    X = sub[features]
    y = sub["net_yards"]

    valid = X.notna().all(axis=1) & y.notna()
    X = X[valid]
    y = y[valid]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    mae = mean_absolute_error(yte, preds)
    print(f"Punt net model MAE ≈ {mae:.2f} yards")

    bundle = {"model": model, "features": features}
    joblib.dump(bundle, PUNT_NET_MODEL_PATH)
    print(f"Saved punt net model → {PUNT_NET_MODEL_PATH}")

    return bundle


# ----------------------------
# Utility: WP from state (Ravens perspective)
# ----------------------------
def predict_ravens_wp_from_state(
    wp_bundle: dict,
    Yardline: float,
    Yds_to_go: float,
    Quarter: int,
    Time_remaining: float,
    Score_differential: float,
    Offense_strength: float,
    Opp_defense_strength: float,
) -> float:
    model = wp_bundle["model"]
    features = wp_bundle["features"]

    row = {
        "Yardline": Yardline,
        "Yds_to_go": Yds_to_go,
        "Quarter": Quarter,
        "Time_remaining": Time_remaining,
        "Score_differential": Score_differential,
        "Offense_strength": Offense_strength,
        "Opp_defense_strength": Opp_defense_strength,
    }
    X = pd.DataFrame([row])[features]
    wp = float(np.clip(model.predict(X)[0], 0.0, 1.0))
    return wp


def predict_ravens_wp_opponent_ball(
    wp_bundle: dict,
    Yardline_for_opponent: float,
    Quarter: int,
    Time_remaining: float,
    Score_differential: float,
    Offense_strength: float,
    Opp_defense_strength: float,
) -> float:
    """
    Approximate Ravens WP when opponent has ball:
      - Flip score differential sign for opponent perspective.
      - Use same WP model, then Ravens_wp = 1 - Opp_wp.
    """
    opp_score_diff = -Score_differential
    opp_wp = predict_ravens_wp_from_state(
        wp_bundle=wp_bundle,
        Yardline=Yardline_for_opponent,
        Yds_to_go=10.0,
        Quarter=Quarter,
        Time_remaining=Time_remaining,
        Score_differential=opp_score_diff,
        Offense_strength=Opp_defense_strength,
        Opp_defense_strength=Offense_strength,
    )
    ravens_wp = 1.0 - opp_wp
    return ravens_wp


# ----------------------------
# FG make "prior" vs distance (fixes lack of long-FG data)
# ----------------------------
def fg_make_prior(fg_length_yards: float) -> float:
    """
    Hand-built curve of NFL-ish FG make rates by distance.
    This acts as an upper bound on the model's predicted P(make),
    especially for very long attempts where we have little data.

    Rough anchors:
      <=30 yd: ~99%
      35 yd:   ~97%
      40 yd:   ~92%
      45 yd:   ~85%
      50 yd:   ~70%
      55 yd:   ~55%
      60 yd:   ~35%
      65 yd:   ~15%
      70+ yd:  ~5% (basically prayer)
    """
    points = [
        (30.0, 0.99),
        (35.0, 0.97),
        (40.0, 0.92),
        (45.0, 0.85),
        (50.0, 0.70),
        (55.0, 0.55),
        (60.0, 0.35),
        (65.0, 0.15),
        (70.0, 0.05),
    ]

    if fg_length_yards <= points[0][0]:
        return points[0][1]
    if fg_length_yards >= points[-1][0]:
        return points[-1][1]

    # Linear interpolation between nearest anchors
    for (d1, p1), (d2, p2) in zip(points[:-1], points[1:]):
        if d1 <= fg_length_yards <= d2:
            t = (fg_length_yards - d1) / (d2 - d1)
            return p1 + t * (p2 - p1)

    return 0.05  # fallback; shouldn't really hit


# ----------------------------
# Action WP computation with football heuristics
# ----------------------------
def compute_action_wps_for_scenario(
    scenario: dict,
    wp_bundle: dict,
    go_bundle: dict,
    fg_bundle: dict,
    punt_bundle: dict,
) -> dict:
    """
    scenario keys:
      Yardline, Quarter, Time_remaining, Yds_to_go,
      Score_differential, Wind_speed, Offense_strength, Opp_defense_strength
    """

    Yardline = float(scenario["Yardline"])
    Quarter = int(scenario["Quarter"])
    Time_remaining = float(scenario["Time_remaining"])
    Yds_to_go = float(scenario["Yds_to_go"])
    Score_differential = float(scenario["Score_differential"])
    Wind_speed = float(scenario["Wind_speed"])
    Offense_strength = float(scenario["Offense_strength"])
    Opp_defense_strength = float(scenario["Opp_defense_strength"])

    # ---------------- Go for it: raw WP ----------------
    go_model = go_bundle["model"]
    go_features = go_bundle["features"]
    go_row = {
        "Yardline": Yardline,
        "Yds_to_go": Yds_to_go,
        "Quarter": Quarter,
        "Time_remaining": Time_remaining,
        "Score_differential": Score_differential,
        "Offense_strength": Offense_strength,
        "Opp_defense_strength": Opp_defense_strength,
    }
    go_X = pd.DataFrame([go_row])[go_features]
    p_convert = float(go_model.predict_proba(go_X)[0, 1])

    # After conversion: 1st & 10 at line to gain
    new_yardline_conv = max(1.0, Yardline - Yds_to_go)
    wp_go_success = predict_ravens_wp_from_state(
        wp_bundle,
        Yardline=new_yardline_conv,
        Yds_to_go=10.0,
        Quarter=Quarter,
        Time_remaining=Time_remaining,
        Score_differential=Score_differential,
        Offense_strength=Offense_strength,
        Opp_defense_strength=Opp_defense_strength,
    )

    # After failure: opponent ball at current spot
    opp_yardline_fail = 100.0 - Yardline  # opponent distance from Ravens EZ
    wp_go_fail = predict_ravens_wp_opponent_ball(
        wp_bundle,
        Yardline_for_opponent=opp_yardline_fail,
        Quarter=Quarter,
        Time_remaining=Time_remaining,
        Score_differential=Score_differential,
        Offense_strength=Offense_strength,
        Opp_defense_strength=Opp_defense_strength,
    )

    WP_go_raw = p_convert * wp_go_success + (1.0 - p_convert) * wp_go_fail

    # ---------------- Field goal: raw WP ----------------
    fg_model = fg_bundle["model"]
    fg_features = fg_bundle["features"]
    fg_row = {
        "Yardline": Yardline,
        "Quarter": Quarter,
        "Time_remaining": Time_remaining,
        "Wind_speed": Wind_speed,
    }
    fg_X = pd.DataFrame([fg_row])[fg_features]
    p_make_model = float(fg_model.predict_proba(fg_X)[0, 1])

    # Approx FG distance: LOS distance to goal line + 17 yards
    fg_length = (100.0 - Yardline) + 17.0
    p_make_prior = fg_make_prior(fg_length)

    # Use the lower of model prediction and distance-based prior
    p_make = min(p_make_model, p_make_prior)

    # After make: +3 points, opponent ball at own 25 (dist from Ravens EZ = 75)
    new_score_diff_make = Score_differential + 3.0
    opp_yardline_after_kick = 75.0
    wp_fg_make = predict_ravens_wp_opponent_ball(
        wp_bundle,
        Yardline_for_opponent=opp_yardline_after_kick,
        Quarter=Quarter,
        Time_remaining=Time_remaining,
        Score_differential=new_score_diff_make,
        Offense_strength=Offense_strength,
        Opp_defense_strength=Opp_defense_strength,
    )

    # After miss: opponent ball near LOS
    opp_yardline_after_miss = 100.0 - Yardline
    wp_fg_miss = predict_ravens_wp_opponent_ball(
        wp_bundle,
        Yardline_for_opponent=opp_yardline_after_miss,
        Quarter=Quarter,
        Time_remaining=Time_remaining,
        Score_differential=Score_differential,
        Offense_strength=Offense_strength,
        Opp_defense_strength=Opp_defense_strength,
    )

    WP_fg_raw = p_make * wp_fg_make + (1.0 - p_make) * wp_fg_miss

    # ---------------- Punt: raw WP ----------------
    punt_model = punt_bundle["model"]
    punt_features = punt_bundle["features"]
    punt_row = {
        "Yardline": Yardline,
        "Quarter": Quarter,
        "Time_remaining": Time_remaining,
    }
    punt_X = pd.DataFrame([punt_row])[punt_features]
    expected_net = float(punt_model.predict(punt_X)[0])

    # Clamp net yards to something realistic
    expected_net = float(np.clip(expected_net, 20.0, 60.0))

    # Our Yardline is distance from opp EZ; distance from our EZ is 100 - Yardline.
    our_dist_from_our_EZ = 100.0 - Yardline
    opp_dist_from_our_EZ = max(1.0, our_dist_from_our_EZ - expected_net)
    opp_yardline_after_punt = opp_dist_from_our_EZ

    wp_punt_raw = predict_ravens_wp_opponent_ball(
        wp_bundle,
        Yardline_for_opponent=opp_yardline_after_punt,
        Quarter=Quarter,
        Time_remaining=Time_remaining,
        Score_differential=Score_differential,
        Offense_strength=Offense_strength,
        Opp_defense_strength=Opp_defense_strength,
    )

    # ---------------- Football sanity & conservative bias ----------------
    WP_go = float(np.clip(WP_go_raw, 0.0, 1.0))
    WP_fg = float(np.clip(WP_fg_raw, 0.0, 1.0))
    WP_punt = float(np.clip(wp_punt_raw, 0.0, 1.0))

    # 1) Long FGs: Yardline > 45 (i.e., truly long attempts or on own side)
    #    Force FGs to be clearly worse than other options.
    if Yardline > 45:
        worst_other = min(WP_go, WP_punt)
        # Penalty grows slightly with how insane the kick is.
        penalty = 0.02 + 0.001 * (Yardline - 45)  # ~2% to ~7% max
        penalty = float(np.clip(penalty, 0.02, 0.07))
        # Make FG WP strictly lower than worst other option, minus penalty.
        WP_fg = max(0.0, min(WP_fg, worst_other - penalty))

    # 2) Punting inside opp 40 (Yardline < 40) in non-desperate spots:
    #    Usually the worst option, but not hard-forced to last.
    in_normal_game_flow = (Time_remaining > 2 * 60) and (abs(Score_differential) <= 14)
    if Yardline < 40 and in_normal_game_flow:
        depth = 40.0 - Yardline  # how deep into opp territory
        punt_penalty = 0.01 + 0.0008 * depth  # ~1–4%
        punt_penalty = float(np.clip(punt_penalty, 0.01, 0.04))
        WP_punt = max(0.0, WP_punt - punt_penalty)

    # 3) Heavy anti-go bias in "normal" situations.
    #    We compute a leverage score in [0,1]. Only in high leverage
    #    does the raw go WP get to "speak".
    leverage = 0.0

    # Late game, not winning → real leverage
    if Quarter >= 4 and Time_remaining <= 8 * 60 and Score_differential <= 0:
        leverage += 0.6

    # Down by a decent amount in 2H
    if Quarter >= 3 and Score_differential <= -7:
        leverage += 0.3

    # No-man's land (30–50) and not ahead → more openness to go-for-it
    if 30.0 <= Yardline <= 50.0 and Score_differential <= 0:
        leverage += 0.2

    leverage = float(np.clip(leverage, 0.0, 1.0))

    safe_wp = max(WP_punt, WP_fg)

    # When leverage is low, WP_go gets pulled down toward "safe" options
    # and is usually worse than punting/FG. When leverage is high,
    # WP_go stays close to the raw expected WP.
    WP_go_adjusted = leverage * WP_go + (1.0 - leverage) * (safe_wp - 0.03)

    # Extra conservative guard rails:
    # - Early in game and not losing: really shouldn't be sending it.
    if Quarter <= 2 and Score_differential >= 0 and Yardline >= 60.0:
        WP_go_adjusted = min(WP_go_adjusted, safe_wp - 0.04)

    # - Big lead in 2H: also lean away from going for it.
    if Quarter >= 3 and Score_differential >= 10:
        WP_go_adjusted = min(WP_go_adjusted, safe_wp - 0.03)

    WP_go = float(np.clip(WP_go_adjusted, 0.0, 1.0))

    # Final clipping
    WP_fg = float(np.clip(WP_fg, 0.0, 1.0))
    WP_punt = float(np.clip(WP_punt, 0.0, 1.0))

    return {
        "WP_go": WP_go,
        "WP_fg": WP_fg,
        "WP_punt": WP_punt,
        "p_convert": p_convert,
        "p_fg_make": p_make,
        "expected_punt_net": expected_net,
    }


# ----------------------------
# Main training
# ----------------------------
if __name__ == "__main__":
    print("Loading full dataset...")
    full_df = load_full_df(FULL_PATH)
    print(f"Loaded {len(full_df)} rows from {FULL_PATH}")

    print("\n=== Training WP state model ===")
    wp_bundle = train_wp_model(full_df)

    print("\n=== Training Go conversion model ===")
    go_bundle = train_go_convert_model(full_df)

    print("\n=== Training FG make model ===")
    fg_bundle = train_fg_make_model(full_df)

    print("\n=== Training Punt net model ===")
    punt_bundle = train_punt_net_model(full_df)

    print("\nAll models trained and saved.")
