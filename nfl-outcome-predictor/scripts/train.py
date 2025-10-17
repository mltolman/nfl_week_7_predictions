#!/usr/bin/env python
from __future__ import annotations
import json, os
import pandas as pd
from src.nfl_outcome.data import load_data, filter_offense_plays, team_game_aggregates, game_labels
from src.nfl_outcome.features import (
    build_home_away_matrix,
    compute_team_rolling_features,
    add_rolling_to_matrix,
    build_feature_target
)
from src.nfl_outcome.models import time_based_split, train_logistic, evaluate_model, save_artifacts

# --- NEW import for odds ---
from src.nfl_outcome.sportsdata_odds import fetch_odds_for_period

CFG_PATH = "config/settings.json"

def main():
    with open(CFG_PATH, "r") as f:
        cfg = json.load(f)
    start, end = cfg["train"]["start_season"], cfg["train"]["end_season"]
    test_start = cfg["train"]["test_start_season"]
    windows = cfg["train"]["rolling_windows"]
    model_dir = cfg["paths"]["models"]

    seasons = list(range(start, end + 1))
    print(f"Loading seasons {start}-{end}...")
    pbp, games = load_data(seasons)

    print("Filtering offense plays & aggregating team-game features...")
    plays = filter_offense_plays(pbp)
    team_game = team_game_aggregates(plays)
    labels = game_labels(games)

    print("Computing rolling features (leakage-safe)...")
    rolled = compute_team_rolling_features(team_game, labels, windows=windows)

    print("Building feature matrix...")
    matrix = build_home_away_matrix(labels, team_game)
    matrix = add_rolling_to_matrix(matrix, rolled)

    # --- NEW: Merge in odds data ---
    #print("Fetching SportsData.io closing odds...")
    # Unique (season, week) pairs where you need to fetch odds
    #epochs = matrix[["season","week"]].drop_duplicates().values.tolist()
    #odds_df = fetch_odds_for_period([(int(s), int(w)) for (s,w) in epochs])
    #print(f"Fetched {len(odds_df)} odds rows")

    #matrix = matrix.merge(
    #    odds_df,
    #    on=["season", "week", "home_team", "away_team"],
    #    how="left"
    #)

    # Now matrix has new columns: spread_close, total_close (if available)

    # Out-of-time split
    matrix = matrix.query("season_type in ['REG','POST']")
    train_df, test_df = time_based_split(matrix, test_start)
    X_train, y_train = build_feature_target(train_df)
    X_test, y_test = build_feature_target(test_df)

    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    scaler, model = train_logistic(X_train, y_train)

    print("Evaluating...")
    metrics_train = evaluate_model(model, scaler, X_train, y_train)
    metrics_test  = evaluate_model(model, scaler, X_test, y_test)

    print("Train:", metrics_train)
    print("Test:", metrics_test)

    print("Saving artifacts...")
    save_artifacts(model_dir, scaler, model)
    print("✅ Done. Artifacts saved in 'models/'.")

if __name__ == "__main__":
    main()
