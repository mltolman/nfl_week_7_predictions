from __future__ import annotations
import pandas as pd
import nflreadpy as nfl
from .features import (
    build_home_away_matrix,
    add_rolling_to_matrix,
    compute_team_rolling_features,
)
from .data import team_game_aggregates, filter_offense_plays

# -- local helper to normalize schedules when we pull just 2025 --
def _normalize_sched(df: pd.DataFrame) -> pd.DataFrame:
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

def week7_2025_features(pbp_2000_2025: pd.DataFrame, games_2000_2025: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix for Week 7, 2025 using games up through Week 6."""
    # Pull schedule and isolate week 7
    sched = nfl.load_schedules([2025]).to_pandas()
    sched = _normalize_sched(sched)
    wk7 = sched.query("season_type == 'REG' and week == 7")[
        ["game_id","season","week","home_team","away_team","gameday"]
    ].copy()

    # Ensure historical 'games' frame has normalized columns too (if caller passed raw schedules)
    games_meta = games_2000_2025.copy()
    if "season_type" not in games_meta.columns and "game_type" in games_meta.columns:
        games_meta["season_type"] = games_meta["game_type"]
    if "gameday" not in games_meta.columns:
        if "game_date" in games_meta.columns:
            games_meta["gameday"] = pd.to_datetime(games_meta["game_date"])
        elif "gamedate" in games_meta.columns:
            games_meta["gameday"] = pd.to_datetime(games_meta["gamedate"])
        elif "start_time" in games_meta.columns:
            games_meta["gameday"] = pd.to_datetime(games_meta["start_time"])
        else:
            games_meta["gameday"] = pd.NaT

    games_meta = games_meta[["game_id","season","week","season_type","gameday","home_team","away_team"]]

    # Limit to games strictly before week 7 of 2025
    mask_hist = (games_meta["season"] < 2025) | (
        (games_meta["season"] == 2025) & (games_meta["season_type"] == "REG") & (games_meta["week"] <= 6)
    )
    hist_games = games_meta[mask_hist]

    # Team-game aggregates for all historical games available up to week 6 (2025)
    pbp_hist = pbp_2000_2025[pbp_2000_2025["game_id"].isin(hist_games["game_id"])].copy()
    plays = filter_offense_plays(pbp_hist)
    team_game = team_game_aggregates(plays)

    # Rolling features on the historical set
    rolled = compute_team_rolling_features(team_game=team_game, labels=hist_games, windows=[3,5])

    # Build a temporary labels DF that contains wk7 games (labels unknown yet)
    tmp_labels = wk7.assign(home_score=0, away_score=0, home_win=0, season_type="REG")

    base = build_home_away_matrix(tmp_labels, team_game)
    feats = add_rolling_to_matrix(base, rolled)
    return feats

from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

def predict_week7(model: BaseEstimator, scaler: StandardScaler, feats_full: pd.DataFrame) -> pd.DataFrame:
    """
    Take the full Week 7 feature matrix (including team IDs) and return predictions.
    We select delta_* features internally to match the trained scaler/model.
    """
    feature_cols = [c for c in feats_full.columns if c.startswith("delta_")]
    X = feats_full[feature_cols].fillna(0)
    Xs = scaler.transform(X)
    p = model.predict_proba(Xs)[:,1]
    return pd.DataFrame({
        "home_team": feats_full["home_team"],
        "away_team": feats_full["away_team"],
        "home_win_prob": p
    })
