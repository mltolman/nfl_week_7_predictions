from __future__ import annotations
import numpy as np
import pandas as pd
import nflreadpy as nfl
from typing import List, Optional

from .features import (
    build_home_away_matrix,
    add_rolling_to_matrix,
    compute_team_rolling_features,
)
from .data import team_game_aggregates, filter_offense_plays


def _normalize_sched(df: pd.DataFrame) -> pd.DataFrame:
    """Make schedules / games tables consistent: season_type + gameday."""
    out = df.copy()
    if "season_type" not in out.columns and "game_type" in out.columns:
        out["season_type"] = out["game_type"]

    if "gameday" not in out.columns:
        if "game_date" in out.columns:
            out["gameday"] = pd.to_datetime(out["game_date"])
        elif "gamedate" in out.columns:
            out["gameday"] = pd.to_datetime(out["gamedate"])
        elif "start_time" in out.columns:
            out["gameday"] = pd.to_datetime(out["start_time"])
        else:
            out["gameday"] = pd.NaT
    return out


def _labels_with_week_stub(
    games_meta: pd.DataFrame,
    stub_rows: pd.DataFrame
) -> pd.DataFrame:
    """
    Concatenate historical games with a zero-score stub for the predict-week rows.
    This allows rolling features for the predict-week rows without leakage (uses shift(1)).
    """
    cols = ["game_id", "season", "week", "season_type",
            "home_team", "away_team", "home_score", "away_score", "gameday"]
    hist = games_meta[cols]
    stub = stub_rows[cols]
    return pd.concat([hist, stub], ignore_index=True)


def week_features_for(
    pbp_all: pd.DataFrame,
    games_all: pd.DataFrame,
    predict_season: int,
    predict_week: int,
    windows: List[int] = [3, 5],
) -> pd.DataFrame:
    """
    Build feature matrix for `predict_season`/`predict_week` using games up through (week-1).
    """
    # Full season schedules for the predict season (get the matchups we’ll predict)
    sched = nfl.load_schedules([predict_season]).to_pandas()
    sched = _normalize_sched(sched)

    wk = sched.query("season_type == 'REG' and week == @predict_week")[
        ["game_id", "season", "week", "home_team", "away_team", "gameday"]
    ].copy()

    # Normalize the caller’s historical schedules/games
    games_meta = _normalize_sched(games_all.copy())
    # keep only the columns we need
    games_meta = games_meta[[
        "game_id", "season", "week", "season_type", "gameday",
        "home_team", "away_team", "home_score", "away_score"
    ]].copy()

    # Historical cutoff: strictly before the predict week in predict season
    mask_hist = (games_meta["season"] < predict_season) | (
        (games_meta["season"] == predict_season)
        & (games_meta["season_type"] == "REG")
        & (games_meta["week"] <= (predict_week - 1))
    )
    hist_games = games_meta[mask_hist].copy()

    # Stub labels for the predict week (scores unknown)
    wk_stub = wk.assign(
        home_score=0, away_score=0, home_win=0, season_type="REG"
    )

    # Compute rolling features on hist + stub rows (leakage-safe via shift(1))
    labels_for_rolling = _labels_with_week_stub(hist_games, wk_stub)

    # Team-game aggregates only for *historical* games that actually have plays
    pbp_hist = pbp_all[pbp_all["game_id"].isin(hist_games["game_id"])].copy()
    plays = filter_offense_plays(pbp_hist)
    team_game = team_game_aggregates(plays)

    rolled = compute_team_rolling_features(
        team_game=team_game,
        labels=labels_for_rolling,
        windows=windows,
    )

    # Base matrix for the predict week only
    base = build_home_away_matrix(wk_stub, team_game)

    # Attach rolling features (these *exist* for the predict-week rows)
    feats = add_rolling_to_matrix(base, rolled)
    return feats


# Convenience wrappers for Week 8, 2025 (keeps your scripts simple)
def week8_2025_features(pbp_2000_2025: pd.DataFrame, games_2000_2025: pd.DataFrame) -> pd.DataFrame:
    return week_features_for(
        pbp_all=pbp_2000_2025,
        games_all=games_2000_2025,
        predict_season=2025,
        predict_week=8,
        windows=[3, 5],
    )


# ====================== Prediction (Regression) ======================

from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler


def _reindex_to_train_columns(df: pd.DataFrame, feature_names_in_: Optional[np.ndarray]) -> pd.DataFrame:
    """Match inference columns to the scaler’s trained feature set."""
    if feature_names_in_ is None:
        return df
    cols = list(feature_names_in_)
    return df.reindex(columns=cols, fill_value=0)


def _points_feature_view(feats_full: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix for points regression.
    We include all rolling stats for home/away plus differentials (delta_).
    """
    cand = [c for c in feats_full.columns if (
        c.startswith("home_r") or c.startswith("away_r") or c.startswith("delta_")
    )]
    X = feats_full[cand].copy().fillna(0)
    return X


def predict_points_week(
    model_home: BaseEstimator,
    model_away: BaseEstimator,
    scaler: StandardScaler,
    feats_full: pd.DataFrame
) -> pd.DataFrame:
    """
    Predict home/away points with two regressors. Also returns totals/diffs/winner.
    """
    X = _points_feature_view(feats_full)
    X = _reindex_to_train_columns(X, getattr(scaler, "feature_names_in_", None))
    Xs = scaler.transform(X)

    home_points = model_home.predict(Xs)
    away_points = model_away.predict(Xs)

    # Post-process for sanity
    home_points = np.clip(home_points, 0, None)
    away_points = np.clip(away_points, 0, None)

    out = pd.DataFrame({
        "home_team": feats_full["home_team"].values,
        "home_points": np.round(home_points, 1),
        "away_team": feats_full["away_team"].values,
        "away_points": np.round(away_points, 1),
    })
    out["home_point_diff"] = np.round(out["home_points"] - out["away_points"], 1)
    out["points_total"]    = np.round(out["home_points"] + out["away_points"], 1)
    out["winning_team"]    = np.where(out["home_point_diff"] >= 0, out["home_team"], out["away_team"])
    return out

