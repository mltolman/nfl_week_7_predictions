#!/usr/bin/env python
from __future__ import annotations
import os
import joblib
import pandas as pd
import nflreadpy as nfl

from src.nfl_outcome.inference import week8_2025_features, predict_points_week

OUT_PATH = "reports/week8_2025_predictions.csv"

def main():
    print("Loading PBP + games through 2025…")
    seasons = list(range(2000, 2026))
    pbp = nfl.load_pbp(seasons).to_pandas()
    games = nfl.load_schedules(seasons).to_pandas()

    print("Building Week 8, 2025 feature matrix…")
    feats = week8_2025_features(pbp, games)

    print("Loading regression artifacts…")
    scaler = joblib.load("models/points_scaler.joblib")
    model_home = joblib.load("models/home_points_model.joblib")
    model_away = joblib.load("models/away_points_model.joblib")

    print("Scoring…")
    preds = predict_points_week(model_home, model_away, scaler, feats)

    os.makedirs("reports", exist_ok=True)
    preds.to_csv(OUT_PATH, index=False)
    print(f"✅ Predictions saved to {OUT_PATH}")

    # quick peek
    print("\nPreview:")
    print(preds.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
