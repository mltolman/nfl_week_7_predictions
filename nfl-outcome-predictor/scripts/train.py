#!/usr/bin/env python
from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
from typing import Iterable, List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

from src.nfl_outcome.data import load_data, filter_offense_plays, team_game_aggregates, game_labels
from src.nfl_outcome.features import (
    compute_team_rolling_features,
    build_home_away_matrix,
    add_rolling_to_matrix,
)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def build_training_matrix(seasons: Iterable[int]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build full historical feature matrix + targets (home_points, away_points).
    Leakage-safe via shift(1) inside compute_team_rolling_features.

    IMPORTANT: We drop games with missing scores (future/unplayed).
    """
    print(f"Loading seasons {min(seasons)}-{max(seasons)}...")
    pbp, games = load_data(seasons)

    # Labels (includes scores + meta). Filter out games with missing scores.
    labels = game_labels(games)
    before = len(labels)
    labels = labels[labels["home_score"].notna() & labels["away_score"].notna()].copy()
    after = len(labels)
    if after < before:
        print(f"Filtered out {before - after} games with missing scores (future/unplayed).")

    print("Filtering offense plays & aggregating team-game features...")
    plays = filter_offense_plays(pbp)
    team_game = team_game_aggregates(plays)

    print("Computing rolling features (leakage-safe)...")
    rolled = compute_team_rolling_features(team_game, labels, windows=[3, 5])

    print("Building feature matrix...")
    matrix = build_home_away_matrix(labels, team_game)
    matrix = add_rolling_to_matrix(matrix, rolled)

    # Targets
    y_home = matrix["home_score"].astype(float)
    y_away = matrix["away_score"].astype(float)

    # Final guard: in case any rows slipped through with NaN targets, drop them.
    target_mask = y_home.notna() & y_away.notna()
    if not target_mask.all():
        dropped = (~target_mask).sum()
        print(f"Dropping {dropped} rows with NaN targets after merge.")
    matrix = matrix.loc[target_mask].copy()
    y_home = y_home.loc[target_mask]
    y_away = y_away.loc[target_mask]

    # Features for points regression: home/away rolling + differentials
    feature_cols = [c for c in matrix.columns if (
        c.startswith("home_r") or c.startswith("away_r") or c.startswith("delta_")
    )]
    X = matrix[feature_cols].fillna(0)

    print(f"Training rows: {len(X)}, Features: {len(feature_cols)}")
    return X, y_home, y_away


def evaluate_and_train(X: pd.DataFrame, y_home: pd.Series, y_away: pd.Series):
    """
    Train two regressors (home_points, away_points) and a shared scaler.
    Report MAE and R^2 on a holdout split.
    """
    X_train, X_test, yh_train, yh_test, ya_train, ya_test = train_test_split(
        X, y_home, y_away, test_size=0.1, random_state=42
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)

    home_reg = RandomForestRegressor(
        n_estimators=600, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    away_reg = RandomForestRegressor(
        n_estimators=600, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1
    )

    print("Training regressors…")
    home_reg.fit(Xtr, yh_train)
    away_reg.fit(Xtr, ya_train)

    yh_pred = home_reg.predict(Xte)
    ya_pred = away_reg.predict(Xte)

    print("Home points — MAE: {:.2f}  R2: {:.3f}".format(
        mean_absolute_error(yh_test, yh_pred), r2_score(yh_test, yh_pred)))
    print("Away points — MAE: {:.2f}  R2: {:.3f}".format(
        mean_absolute_error(ya_test, ya_pred), r2_score(ya_test, ya_pred)))

    return scaler, home_reg, away_reg


def main():
    # Train on 2000–2025 (now includes 2025 Wk7 actuals if published by data source)
    seasons = list(range(2000, 2026))

    X, y_home, y_away = build_training_matrix(seasons)
    scaler, home_reg, away_reg = evaluate_and_train(X, y_home, y_away)

    # Persist artifacts
    joblib.dump(scaler, f"{MODELS_DIR}/points_scaler.joblib")
    joblib.dump(home_reg, f"{MODELS_DIR}/home_points_model.joblib")
    joblib.dump(away_reg, f"{MODELS_DIR}/away_points_model.joblib")

    # Ensure we preserve the training column order for robust inference
    if not hasattr(scaler, "feature_names_in_"):
        scaler.feature_names_in_ = np.array(list(X.columns))
        joblib.dump(scaler, f"{MODELS_DIR}/points_scaler.joblib")  # overwrite with attribute

    print("✅ Done. Artifacts saved in 'models/'.")


if __name__ == "__main__":
    main()
