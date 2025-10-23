#!/usr/bin/env python
from __future__ import annotations
import os
import json
import joblib
import numpy as np
import pandas as pd
import nflreadpy as nfl

from src.nfl_outcome.inference import week7_2025_features, predict_week7

CFG_PATH = "config/settings.json"

# ==== Tunables (heuristic mapping & total model) ====
# Spread mapping: how many points of spread per 1.0 of (p - 0.5)*2.
SPREAD_SCALE = 14.0  # try 10–18 to taste

# Game-specific total (points) anchored to league baseline.
TOTAL_BASE = 43.5  # your prior on an average NFL total

# Weights for the total estimator (heuristic, interpretable units)
W_EPA5   = 18.0   # r5 EPA mean (avg of home & away)
W_EPA3   =  6.0   # r3 EPA mean (avg of home & away)
W_PASS5  =  8.0   # pass rate (centered), more passing tends to inflate totals
W_SUCC5  = 12.0   # success rate (centered)
W_VOL5   =  4.0   # EPA std (centered): volatility can boost totals

# Centers for rate-like stats (rough league priors)
CENTER_PASS = 0.55
CENTER_SUCC = 0.45
CENTER_VOL  = 0.95  # typical r5 EPA std ~1-ish (rough prior)

# Clamp range for totals to keep things sane
TOTAL_MIN = 34.0
TOTAL_MAX = 58.0

def _estimate_game_totals(feats: pd.DataFrame) -> pd.Series:
    """
    Build a per-game total using rolling signal from both teams.
    Uses home/away r5 & r3 to form an 'offensive climate' score.
    Falls back to 0 where features are missing.
    """
    # Pull features (fill NaN→0 so missing doesn’t explode)
    h5 = feats.get("home_r5_epa_mean", pd.Series(0, index=feats.index)).fillna(0)
    a5 = feats.get("away_r5_epa_mean", pd.Series(0, index=feats.index)).fillna(0)
    h3 = feats.get("home_r3_epa_mean", pd.Series(0, index=feats.index)).fillna(0)
    a3 = feats.get("away_r3_epa_mean", pd.Series(0, index=feats.index)).fillna(0)

    # Pass rate / Success rate / Volatility (EPA std)
    h5_pass = feats.get("home_r5_pass_rate", pd.Series(0, index=feats.index)).fillna(0)
    a5_pass = feats.get("away_r5_pass_rate", pd.Series(0, index=feats.index)).fillna(0)
    h5_succ = feats.get("home_r5_success_rate", pd.Series(0, index=feats.index)).fillna(0)
    a5_succ = feats.get("away_r5_success_rate", pd.Series(0, index=feats.index)).fillna(0)
    h5_vol  = feats.get("home_r5_epa_std", pd.Series(0, index=feats.index)).fillna(0)
    a5_vol  = feats.get("away_r5_epa_std", pd.Series(0, index=feats.index)).fillna(0)

    # Team-averaged signals (we want “level”, not home-away difference)
    epa5_avg  = (h5 + a5) / 2.0
    epa3_avg  = (h3 + a3) / 2.0
    pass5_avg = (h5_pass + a5_pass) / 2.0
    succ5_avg = (h5_succ + a5_succ) / 2.0
    vol5_avg  = (h5_vol + a5_vol) / 2.0

    # Center rate-like stats so baseline sits at TOTAL_BASE
    pass5_c = pass5_avg - CENTER_PASS
    succ5_c = succ5_avg - CENTER_SUCC
    vol5_c  = vol5_avg  - CENTER_VOL

    raw_total = (
        TOTAL_BASE
        + W_EPA5 * epa5_avg
        + W_EPA3 * epa3_avg
        + W_PASS5 * pass5_c
        + W_SUCC5 * succ5_c
        + W_VOL5  * vol5_c
    )

    # Clamp to reasonable range; then round to 0.1 for tidy outputs
    clamped = np.clip(raw_total, TOTAL_MIN, TOTAL_MAX)
    return np.round(clamped, 1)

def _points_from_prob_and_total(p_home: pd.Series, total: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Convert home win probability → spread, then split total into home/away points.
    """
    # Spread in points: symmetric around 0; positive favors home
    spread = SPREAD_SCALE * (p_home - 0.5) * 2.0

    home_pts = (total + spread) / 2.0
    away_pts = (total - spread) / 2.0

    # No negative scores; round to tenths (or swap to integers if you prefer)
    home_pts = np.round(np.clip(home_pts, 0, None), 1)
    away_pts = np.round(np.clip(away_pts, 0, None), 1)
    return home_pts, away_pts

def main():
    with open(CFG_PATH, "r") as f:
        cfg = json.load(f)

    # Load data through 2025 (we only use through Wk 6 for rolling).
    seasons = list(range(cfg["train"]["start_season"], 2026))
    print("Loading PBP + games through 2025…")
    pbp = nfl.load_pbp(seasons).to_pandas()
    games = nfl.load_schedules(seasons).to_pandas()

    print("Building Week 7, 2025 feature matrix…")
    feats = week7_2025_features(pbp, games)

    # Quick sanity peek (optional)
    # print("Feature columns:", list(feats.columns))

    print("Loading model artifacts…")
    scaler = joblib.load("models/scaler.joblib")
    model  = joblib.load("models/model.joblib")

    # Probabilities (home team)
    prob_df = predict_week7(model, scaler, feats)  # cols: home_team, away_team, home_win_prob

    # Game-specific totals from rolling signals
    totals = _estimate_game_totals(feats)

    # Convert prob+total -> points
    home_pts, away_pts = _points_from_prob_and_total(prob_df["home_win_prob"], totals)

    # Build final table
    out = pd.DataFrame({
        "home_team": prob_df["home_team"],
        "home_points": home_pts,
        "away_team": prob_df["away_team"],
        "away_points": away_pts,
    })
    out["home_point_diff"] = np.round(out["home_points"] - out["away_points"], 1)
    out["points_total"]    = np.round(out["home_points"] + out["away_points"], 1)
    out["winning_team"]    = np.where(out["home_points"] >= out["away_points"], out["home_team"], out["away_team"])
    out["home_win_prob"]   = np.round(prob_df["home_win_prob"], 4)

    os.makedirs("reports", exist_ok=True)
    out_path = "reports/week7_2025_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"✅ Predictions saved to {out_path}\n")

    # Small preview
    with pd.option_context("display.max_columns", None,
                           "display.width", 120,
                           "display.float_format", lambda x: f"{x:0.4f}"):
        print("Preview:")
        print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
