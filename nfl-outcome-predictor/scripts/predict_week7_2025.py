#!/usr/bin/env python
from __future__ import annotations
import os, joblib, json
import pandas as pd
import nflreadpy as nfl
from src.nfl_outcome.inference import week7_2025_features, predict_week7

CFG_PATH = "config/settings.json"

def main():
    with open(CFG_PATH, "r") as f:
        cfg = json.load(f)

    # We need PBP + games through week 6 of 2025 to build rolling features
    seasons = list(range(cfg["train"]["start_season"], 2026))
    print(f"Loading PBP + games through 2025...")
    pbp = nfl.load_pbp(seasons).to_pandas()
    games = nfl.load_schedules(seasons).to_pandas()

    print("Building Week 7, 2025 feature matrix...")
    feats = week7_2025_features(pbp, games)

    print("Loading model artifacts...")
    scaler = joblib.load("models/scaler.joblib")
    model  = joblib.load("models/model.joblib")

    preds = predict_week7(model, scaler, feats)
    out_path = "reports/week7_2025_predictions.csv"
    os.makedirs("reports", exist_ok=True)
    preds.to_csv(out_path, index=False)
    print(f"✅ Predictions saved to {out_path}")

if __name__ == "__main__":
    main()
