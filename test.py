# simulate_one_4th_down.py
# Edit ONE scenario below and print WP_go, WP_punt, WP_fg.
#
# Requires trained models saved in ./models by mvp_4th_down_bot.py:
#   - models/wp_model.joblib
#   - models/fg_model.joblib
#   - models/go_conv_model.joblib
#   - models/punt_gross_dist.npy
#
# Usage: python simulate_one_4th_down.py

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# =========================
# EDIT THESE INPUTS
# =========================
Yardline              = 10.0   # distance from opponent end zone (1..99). 40 => own 40.
Quarter               = 4.0    # 1..4 (ignore OT for MVP)
Time_remaining        = 60.0  # seconds left in game
Timeouts_offense      = 0.0    # Ravens timeouts remaining
Timeouts_defense      = 0.0    # Opponent timeouts remaining
Score_differential    = 10.0    # Ravens score - Opp score (negative if trailing)
Yds_to_go             = 2.0   # yards for a first down
Wind_speed            = 6.0    # mph (FG model)
Offense_strength      = 0.05   # your rating; 0 = league avg
Opp_defense_strength  = -0.05  # opponent defensive rating; 0 = league avg

# =========================
# Paths
# =========================
MODEL_DIR = Path("models")
WP_MODEL_PATH  = MODEL_DIR / "wp_model.joblib"
FG_MODEL_PATH  = MODEL_DIR / "fg_model.joblib"
GO_MODEL_PATH  = MODEL_DIR / "go_conv_model.joblib"
PUNT_DIST_PATH = MODEL_DIR / "punt_gross_dist.npy"

# Feature lists
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

# Load models
wp_model = joblib.load(WP_MODEL_PATH)
fg_model = joblib.load(FG_MODEL_PATH)
go_model = joblib.load(GO_MODEL_PATH)
punt_gross = np.load(PUNT_DIST_PATH).astype(float)

# ---------------- Helpers ----------------
def clamp_yl(y):
    return float(np.clip(float(y), 1.0, 99.0))

def wp_offense(state_dict):
    """WP for whoever has the ball in `state_dict` coordinates."""
    row = pd.DataFrame([state_dict])
    for k in WP_FEATURES:
        if k not in row.columns:
            row[k] = np.nan
    return float(np.clip(wp_model.predict(row[WP_FEATURES])[0], 0.0, 1.0))

def flip_possession(state_ravens_persp):
    """
    Convert Ravens-perspective state (Ravens yardline_100, Ravens timeouts,
    score_differential = Ravens - Opp) to an opponent-offense state.
    """
    out = state_ravens_persp.copy()
    out["yardline_100"] = 100.0 - float(out["yardline_100"])
    out["score_differential"] = -float(out["score_differential"])
    # swap timeouts
    to_off = float(out["posteam_timeouts_remaining"])
    to_def = float(out["defteam_timeouts_remaining"])
    out["posteam_timeouts_remaining"] = to_def
    out["defteam_timeouts_remaining"] = to_off
    # swap ratings (simple symmetric MVP)
    off_r = float(out.get("offensive_rating", 0.0))
    def_r = float(out.get("opponent_defensive_rating", 0.0))
    out["offensive_rating"] = -def_r
    out["opponent_defensive_rating"] = -off_r
    return out

def ravens_wp_on_offense(state_ravens_offense):
    """Ravens have the ball -> ask offense WP directly."""
    return wp_offense(state_ravens_offense)

def ravens_wp_when_opponent_has_ball(state_ravens_persp):
    """
    Opponent has the ball: flip to opponent-offense, get opponent WP,
    then convert to Ravens WP as 1 - opponent WP.
    """
    opp_offense_state = flip_possession(state_ravens_persp)
    opp_wp = wp_offense(opp_offense_state)
    return 1.0 - opp_wp

def p_fg_make_from_yardline(yardline_100, wind=0.0, temp=60.0, roof_is_indoor=0.0):
    dist = float(yardline_100) + 17.0
    row = pd.DataFrame([{
        "kick_distance": dist,
        "temp": temp,
        "wind": wind,
        "roof_is_indoor": roof_is_indoor
    }])
    return float(fg_model.predict_proba(row)[0, 1]), dist

def p_go_convert(go_state):
    r = pd.DataFrame([go_state])
    for k in GO_FEATURES:
        if k not in r.columns:
            r[k] = np.nan
    return float(go_model.predict_proba(r[GO_FEATURES])[0, 1])

def sample_punt_gross(n=600, rng=None):
    rng = np.random.default_rng(rng)
    return rng.choice(punt_gross, size=n, replace=True)

# Base Ravens state (offense)
base_state = {
    "yardline_100": float(Yardline),
    "qtr": float(Quarter),
    "game_seconds_remaining": float(Time_remaining),
    "posteam_timeouts_remaining": float(Timeouts_offense),
    "defteam_timeouts_remaining": float(Timeouts_defense),
    "score_differential": float(Score_differential),
    "offensive_rating": float(Offense_strength),
    "opponent_defensive_rating": float(Opp_defense_strength),
}

# ---------------- Action evaluators ----------------
def WP_go():
    ydstogo = float(Yds_to_go)

    # Properly trained on 4th-down attempts only
    p_conv = p_go_convert({
        "yardline_100": base_state["yardline_100"],
        "ydstogo": ydstogo,
        "qtr": base_state["qtr"],
        "game_seconds_remaining": base_state["game_seconds_remaining"],
        "score_differential": base_state["score_differential"],
        "posteam_timeouts_remaining": base_state["posteam_timeouts_remaining"],
        "defteam_timeouts_remaining": base_state["defteam_timeouts_remaining"],
        "offensive_rating": base_state["offensive_rating"],
        "opponent_defensive_rating": base_state["opponent_defensive_rating"],
    })

    # Success: Ravens keep ball, 1st & 10 at (yardline - ydstogo)
    success = base_state.copy()
    success["yardline_100"] = clamp_yl(base_state["yardline_100"] - ydstogo)
    wp_success = ravens_wp_on_offense(success)

    # Fail: opponent ball at LOS
    fail = base_state.copy()
    wp_fail = ravens_wp_when_opponent_has_ball(fail)

    return p_conv * wp_success + (1.0 - p_conv) * wp_fail

def WP_fg():
    wind = float(Wind_speed)
    p_make, dist = p_fg_make_from_yardline(base_state["yardline_100"], wind=wind, temp=60.0, roof_is_indoor=0.0)

    # MAKE -> +3 points, opponent gets ball (approx start at their 25 -> Ravens yardline_100 = 75)
    make_state = base_state.copy()
    make_state["score_differential"] = make_state["score_differential"] + 3.0
    make_state["yardline_100"] = 75.0
    wp_make = ravens_wp_when_opponent_has_ball(make_state)

    # MISS -> opponent ball at LOS
    miss_state = base_state.copy()
    wp_miss = ravens_wp_when_opponent_has_ball(miss_state)

    return p_make * wp_make + (1.0 - p_make) * wp_miss

def WP_punt(nsamples=600, rng=None):
    dists = sample_punt_gross(n=nsamples, rng=rng)
    wps = []
    for d in dists:
        los_from_ravens_end = 100.0 - base_state["yardline_100"]
        new_field_pos = los_from_ravens_end + float(d)
        if new_field_pos >= 100.0:
            new_field_pos = 80.0  # touchback places ball at opponent 20
        new_field_pos = float(np.clip(new_field_pos, 0.0, 99.0))
        new_yl = clamp_yl(100.0 - new_field_pos)
        opp_start = base_state.copy()
        opp_start["yardline_100"] = new_yl
        wps.append(ravens_wp_when_opponent_has_ball(opp_start))
    return float(np.mean(wps))

# ---------------- Run once ----------------
if __name__ == "__main__":
    wp_go = WP_go()
    wp_fg = WP_fg()
    wp_punt = WP_punt()

    best = max(wp_go, wp_punt, wp_fg)
    rec = "GO" if best == wp_go else ("PUNT" if best == wp_punt else "FG")

    print("=== Hypothetical 4th-Down Scenario (edit variables at top) ===")
    print(f"Yardline_100: {Yardline:.1f} | Quarter: {Quarter:.0f} | Time_remaining: {Time_remaining:.0f}s")
    print(f"Yds_to_go: {Yds_to_go:.1f} | Timeouts O/D: {Timeouts_offense:.0f}/{Timeouts_defense:.0f}")
    print(f"Score_diff (Ravens - Opp): {Score_differential:.1f} | Wind: {Wind_speed:.1f} mph")
    print(f"Offense_strength: {Offense_strength:+.3f} | Opp_defense_strength: {Opp_defense_strength:+.3f}")
    print("---------------------------------------------------------------")
    print(f"WP_go   : {wp_go:.4f}")
    print(f"WP_punt : {wp_punt:.4f}")
    print(f"WP_fg   : {wp_fg:.4f}")
    print("---------------------------------------------------------------")
    print(f"Recommendation: {rec}  |  Regret if choose GO:   {best - wp_go:+.4f}")
    print(f"                               Regret if choose PUNT: {best - wp_punt:+.4f}")
    print(f"                               Regret if choose FG:   {best - wp_fg:+.4f}")
