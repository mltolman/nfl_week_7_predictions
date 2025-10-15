from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Tuple

def build_home_away_matrix(labels: pd.DataFrame,
                           team_game: pd.DataFrame) -> pd.DataFrame:
    """Join team-game aggregates into a single row per game with home/away suffixes."""
    home = (team_game.rename(columns={
                "team":"home_team",
                "epa_mean":"home_epa_mean", "epa_std":"home_epa_std",
                "success_rate":"home_success_rate", "yards_mean":"home_yards_mean",
                "pass_rate":"home_pass_rate"}))
    away = (team_game.rename(columns={
                "team":"away_team",
                "epa_mean":"away_epa_mean", "epa_std":"away_epa_std",
                "success_rate":"away_success_rate", "yards_mean":"away_yards_mean",
                "pass_rate":"away_pass_rate"}))
    out = (labels
           .merge(home, on=["game_id","home_team"], how="left")
           .merge(away, on=["game_id","away_team"], how="left"))
    # Differentials
    for c in ["epa_mean","epa_std","success_rate","yards_mean","pass_rate"]:
        out[f"delta_{c}"] = out[f"home_{c}"] - out[f"away_{c}"]
    return out

def compute_team_rolling_features(team_game: pd.DataFrame,
                                  labels: pd.DataFrame,
                                  windows: List[int] = [3,5]) -> pd.DataFrame:
    """Compute leakage-safe rolling features per team using game date order.
    Returns a per-(game_id, team) table with columns like r3_epa_mean, etc.
    """
    # Bring in date and opponent side for ordering
    meta = labels[["game_id","gameday","home_team","away_team","season","week","season_type"]].copy()
    long = (pd.concat([
                meta[["game_id","gameday","home_team","season","week","season_type"]]
                    .rename(columns={"home_team":"team"})
                    .assign(is_home=1),
                meta[["game_id","gameday","away_team","season","week","season_type"]]
                    .rename(columns={"away_team":"team"})
                    .assign(is_home=0)
            ], ignore_index=True)
            .sort_values(["team","gameday"]))

    # Merge with aggregates
    tg = long.merge(team_game, on=["game_id","team"], how="left"
                    ).sort_values(["team","gameday"]).reset_index(drop=True)

    # For each team, rolling over prior rows (closed='left' behavior via shift)
    def _roll(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("gameday").reset_index(drop=True)
        base_cols = ["epa_mean","success_rate","yards_mean","pass_rate"]
        for w in windows:
            for c in base_cols:
                df[f"r{w}_{c}"] = df[c].shift(1).rolling(window=w, min_periods=1).mean()
        return df

    rolled = (tg.groupby("team", group_keys=False).apply(_roll))
    rolled = rolled[["game_id","team"] + [c for c in rolled.columns if c.startswith("r")]].copy()
    return rolled

def add_rolling_to_matrix(matrix: pd.DataFrame, rolled: pd.DataFrame) -> pd.DataFrame:
    home_r = (rolled.rename(columns={"team":"home_team"})
                   .add_prefix("home_")
                   .rename(columns={"home_game_id":"game_id","home_home_team":"home_team"}))
    away_r = (rolled.rename(columns={"team":"away_team"})
                   .add_prefix("away_")
                   .rename(columns={"away_game_id":"game_id","away_away_team":"away_team"}))

    out = (matrix
           .merge(home_r, on=["game_id","home_team"], how="left")
           .merge(away_r, on=["game_id","away_team"], how="left"))
    # Rolling differentials
    roll_cols = sorted([c for c in out.columns if c.startswith("home_r")])
    for hc in roll_cols:
        base = hc.replace("home_", "")
        ac = "away_" + base
        if ac in out.columns:
            out[f"delta_{base}"] = out[hc] - out[ac]
    return out

def build_feature_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    feature_cols = [c for c in df.columns if c.startswith("delta_")]
    X = df[feature_cols].fillna(0)
    y = df["home_win"].astype(int)
    return X, y