"""
app_gradio.py

Gradio web UI for the Ravens 4th-down bot.

- Loads the four trained bundles from the `models` folder.
- Uses compute_action_wps_for_scenario from train_4th_down_bot_reworked.py
- Exposes a simple web form so anyone can plug in a situation and get:
    * WP_go, WP_fg, WP_punt (percent)
    * Recommended action
    * p(convert), p(FG make), expected punt net yards
"""

import gradio as gr
import joblib

from train_4th_down_bot_reworked import (
    compute_action_wps_for_scenario,
    WP_MODEL_PATH,
    GO_CONVERT_MODEL_PATH,
    FG_MAKE_MODEL_PATH,
    PUNT_NET_MODEL_PATH,
)


# ----------------------------
# Load model bundles once
# ----------------------------

def load_bundles():
    wp_bundle = joblib.load(WP_MODEL_PATH)
    go_bundle = joblib.load(GO_CONVERT_MODEL_PATH)
    fg_bundle = joblib.load(FG_MAKE_MODEL_PATH)
    punt_bundle = joblib.load(PUNT_NET_MODEL_PATH)
    return wp_bundle, go_bundle, fg_bundle, punt_bundle


wp_bundle, go_bundle, fg_bundle, punt_bundle = load_bundles()


# ----------------------------
# Prediction function for Gradio
# ----------------------------

def predict_wp(
    Yardline,
    Quarter,
    Time_remaining,
    Yds_to_go,
    Score_differential,
    Wind_speed,
    Offense_strength,
    Opp_defense_strength,
):
    """
    Yardline: distance from opponent end zone (1–99)
      - Own 25 -> 75
      - Midfield (50) -> 50
      - Opp 40 -> 40
      - Opp 10 -> 10
    """

    scenario = {
        "Yardline": float(Yardline),
        "Quarter": int(Quarter),
        "Time_remaining": float(Time_remaining),
        "Yds_to_go": float(Yds_to_go),
        "Score_differential": float(Score_differential),
        "Wind_speed": float(Wind_speed),
        "Offense_strength": float(Offense_strength),
        "Opp_defense_strength": float(Opp_defense_strength),
    }

    result = compute_action_wps_for_scenario(
        scenario,
        wp_bundle=wp_bundle,
        go_bundle=go_bundle,
        fg_bundle=fg_bundle,
        punt_bundle=punt_bundle,
    )

    WP_go = result["WP_go"] * 100
    WP_fg = result["WP_fg"] * 100
    WP_punt = result["WP_punt"] * 100

    p_convert = result["p_convert"] * 100
    p_fg_make = result["p_fg_make"] * 100
    expected_punt_net = result["expected_punt_net"]

    actions = {"Go for it": WP_go, "Field goal": WP_fg, "Punt": WP_punt}
    best_action = max(actions, key=actions.get)

    # Nicely formatted markdown output
    summary = f"""### Results

**Win probabilities (Ravens):**
- Go for it: **{WP_go:.2f}%**
- Field goal: **{WP_fg:.2f}%**
- Punt: **{WP_punt:.2f}%**


**Recommended action: 🟢 {best_action}**
"""

    return summary


# ----------------------------
# Gradio Interface
# ----------------------------

with gr.Blocks() as demo:
    gr.Markdown("# Ravens 4th-Down Decision Bot")
    gr.Markdown(
        "Enter a game situation and get win probabilities for **Go / Field Goal / Punt** "
        "based on historical Ravens data plus football-aware heuristics."
    )

    with gr.Row():
        with gr.Column():
            Yardline = gr.Slider(
                minimum=1,
                maximum=99,
                value=75,
                step=1,
                label="Yardline (distance from opponent end zone)",
                info="Own 25 → 75, Midfield → 50, Opp 40 → 40, Opp 10 → 10",
            )
            Quarter = gr.Slider(
                minimum=1,
                maximum=4,
                value=1,
                step=1,
                label="Quarter",
            )
            Time_remaining = gr.Number(
                value=50 * 60,
                label="Time remaining (seconds, in game)",
            )
            Yds_to_go = gr.Slider(
                minimum=1,
                maximum=20,
                value=5,
                step=1,
                label="Yards to go (for first down)",
            )

        with gr.Column():
            Score_differential = gr.Number(
                value=0,
                label="Score differential (Ravens - Opponent)",
                info="Positive = Ravens leading, Negative = Ravens trailing",
            )
            Wind_speed = gr.Number(
                value=5,
                label="Wind speed (mph, rough estimate)",
            )
            Offense_strength = gr.Number(
                value=0.0,
                label="Offense strength (0 = league average)",
                info="Positive = better offense, Negative = worse",
            )
            Opp_defense_strength = gr.Number(
                value=0.0,
                label="Opponent defense strength (0 = league average)",
                info="Positive = better defense, Negative = worse",
            )

    btn = gr.Button("Compute Win Probabilities")
    output = gr.Markdown(label="Results")

    btn.click(
        fn=predict_wp,
        inputs=[
            Yardline,
            Quarter,
            Time_remaining,
            Yds_to_go,
            Score_differential,
            Wind_speed,
            Offense_strength,
            Opp_defense_strength,
        ],
        outputs=output,
    )

if __name__ == "__main__":
    # launch() opens a local web UI; share=True can give you a public URL
    demo.launch(share=True)
