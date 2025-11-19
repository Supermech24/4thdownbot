# 4th Down Bot

A machine learning-based 4th down decision-making tool for NFL teams, specifically designed for the Baltimore Ravens.

## Overview

This project provides models to help make optimal 4th down decisions by calculating win probabilities for three options:
- **Go for it** (4th down conversion attempt)
- **Punt**
- **Field goal**

## Features

- **Win Probability (WP) Model**: Maps game state to win probability for the current offense
- **Field Goal (FG) Model**: Predicts probability of making a field goal based on distance, wind, temperature, and roof type
- **Go-for-it Conversion (GO) Model**: Predicts probability of converting a 4th down attempt based on yardline, yards to go, quarter, time, and team ratings
- **Punt Distribution**: Historical gross punt distance distribution

## Requirements

```bash
pip install pandas numpy scikit-learn joblib polars nflreadpy
```

## Usage

### Training Models

Train all models using the main script:

```bash
python mvp_4th_down_bot.py
```

This will:
1. Download play-by-play data (if not cached)
2. Train the WP, FG, and GO models
3. Generate the punt distribution
4. Save all models to the `models/` directory

### Simulating a 4th Down Scenario

Edit the inputs in `test.py` (or `simulate_one_4th_down.py`) and run:

```bash
python test.py
```

The script will output win probabilities for each option (Go, Punt, Field Goal).

## Model Files

Trained models are saved in the `models/` directory:
- `wp_model.joblib` - Win probability model
- `fg_model.joblib` - Field goal success model
- `go_conv_model.joblib` - 4th down conversion model
- `punt_gross_dist.npy` - Historical punt distance distribution

## Data

The project uses NFL play-by-play data from 2010 onwards, cached in `pbp_2010_onwards.parquet`.

## Files

- `mvp_4th_down_bot.py` - Main training script
- `test.py` - Simulation script for testing 4th down scenarios
- `data.py` - Data processing utilities

