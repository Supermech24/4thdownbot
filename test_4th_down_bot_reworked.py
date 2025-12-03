"""
test_4th_down_bot_reworked.py

Simple tester for the 4th-down bot.
- Loads trained models from the `models` folder
- Uses `compute_action_wps_for_scenario` from train_4th_down_bot_reworked.py
- Lets you define scenarios directly in this file and prints out:
    * WP_go, WP_fg, WP_punt (as percentages)
    * Recommended action
    * p_convert (go), p_fg_make, expected_punt_net

Edit the SCENARIOS list at the bottom to try different situations.
"""

import joblib
from pathlib import Path

from train_4th_down_bot_reworked import (
    compute_action_wps_for_scenario,
    WP_MODEL_PATH,
    GO_CONVERT_MODEL_PATH,
    FG_MAKE_MODEL_PATH,
    PUNT_NET_MODEL_PATH,
)

# ----------------------------
# Model loading
# ----------------------------

def load_bundles():
    """Load the four bundles saved by the training script."""
    if not Path(WP_MODEL_PATH).exists():
        raise FileNotFoundError(
            f"WP model not found at {WP_MODEL_PATH}. "
            "Make sure you ran train_4th_down_bot_reworked.py first."
        )

    wp_bundle = joblib.load(WP_MODEL_PATH)
    go_bundle = joblib.load(GO_CONVERT_MODEL_PATH)
    fg_bundle = joblib.load(FG_MAKE_MODEL_PATH)
    punt_bundle = joblib.load(PUNT_NET_MODEL_PATH)

    return wp_bundle, go_bundle, fg_bundle, punt_bundle


# ----------------------------
# Scenario evaluation
# ----------------------------

def print_scenario_result(name: str, scenario: dict,
                          wp_bundle, go_bundle, fg_bundle, punt_bundle):
    """
    scenario must contain:
      Yardline, Quarter, Time_remaining, Yds_to_go,
      Score_differential, Wind_speed, Offense_strength, Opp_defense_strength
    """
    result = compute_action_wps_for_scenario(
        scenario,
        wp_bundle=wp_bundle,
        go_bundle=go_bundle,
        fg_bundle=fg_bundle,
        punt_bundle=punt_bundle,
    )

    WP_go = result["WP_go"]
    WP_fg = result["WP_fg"]
    WP_punt = result["WP_punt"]

    p_convert = result["p_convert"]
    p_fg_make = result["p_fg_make"]
    expected_punt_net = result["expected_punt_net"]

    # Decide recommended action
    actions = {
        "GO": WP_go,
        "FG": WP_fg,
        "PUNT": WP_punt,
    }
    best_action = max(actions, key=actions.get)

    print("=" * 70)
    print(f"Scenario: {name}")
    print("-" * 70)
    print(
        f"Inputs -> "
        f"Yardline (dist from opp EZ): {scenario['Yardline']}, "
        f"Quarter: {scenario['Quarter']}, "
        f"Time_remaining (s): {scenario['Time_remaining']}, "
        f"Yds_to_go: {scenario['Yds_to_go']}, "
        f"Score_diff (Ravens - Opp): {scenario['Score_differential']}, "
        f"Wind_speed: {scenario['Wind_speed']}, "
        f"Off_str: {scenario['Offense_strength']}, "
        f"Def_str: {scenario['Opp_defense_strength']}"
    )
    print()
    print("Win probabilities (Ravens):")
    print(f"  Go for it : {WP_go * 100:6.2f} %")
    print(f"  Field goal: {WP_fg * 100:6.2f} %")
    print(f"  Punt      : {WP_punt * 100:6.2f} %")
    print()
    print(f"  p_convert (go) : {p_convert * 100:6.2f} %")
    print(f"  p_fg_make      : {p_fg_make * 100:6.2f} %")
    print(f"  expected_punt_net: {expected_punt_net:6.2f} yards")
    print()
    print(f"Recommended action: *** {best_action} ***")
    print("=" * 70)
    print()


# ----------------------------
# Main: edit SCENARIOS below
# ----------------------------

if __name__ == "__main__":
    print("Loading models...")
    wp_bundle, go_bundle, fg_bundle, punt_bundle = load_bundles()
    print("Models loaded.\n")

    # NOTE ON YARDLINE:
    #   Yardline here is distance from the OPPONENT end zone (like yardline_100).
    #   Examples:
    #     - Own 25 yard line   -> Yardline = 75
    #     - midfield (50)      -> Yardline = 50
    #     - Opp 40 yard line   -> Yardline = 40
    #     - Opp 10 yard line   -> Yardline = 10

    # You can freely edit/add scenarios in this list:
    SCENARIOS = [
        {
            "name": "Q1 10:00, up 7, 4th & 5 at own 25 (should be punt-heavy)",
            "Yardline": 5,             # own 25
            "Quarter": 4,
            "Time_remaining": 1 * 60,  # 50:00 left in game
            "Yds_to_go": 1,
            "Score_differential": -4,    # Ravens up 7
            "Wind_speed": 5,
            "Offense_strength": 0.5,
            "Opp_defense_strength": 0.0,
        },
        {
            "name": "Q2 02:00, tie game, 4th & 3 at opp 38 (borderline FG / go)",
            "Yardline": 38,
            "Quarter": 2,
            "Time_remaining": 30 * 60 - 2 * 60,  # 2:00 left in 2nd qtr -> 28:00 game secs
            "Yds_to_go": 3,
            "Score_differential": 0,
            "Wind_speed": 3,
            "Offense_strength": 0.0,
            "Opp_defense_strength": 0.0,
        },
        {
            "name": "Q4 02:00, down 4, 4th & 2 at opp 45 (should lean go-for-it)",
            "Yardline": 45,
            "Quarter": 4,
            "Time_remaining": 2 * 60,
            "Yds_to_go": 2,
            "Score_differential": -4,   # Ravens down 4
            "Wind_speed": 5,
            "Offense_strength": 0.0,
            "Opp_defense_strength": 0.0,
        },
        # Add your own scenarios below, or modify the ones above.
        # Just keep the keys the same.
    ]

    for sc in SCENARIOS:
        name = sc.get("name", "Unnamed scenario")
        scenario = {k: v for k, v in sc.items() if k != "name"}
        print_scenario_result(
            name,
            scenario,
            wp_bundle=wp_bundle,
            go_bundle=go_bundle,
            fg_bundle=fg_bundle,
            punt_bundle=punt_bundle,
        )
