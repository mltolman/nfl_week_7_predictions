from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

# Toggle extra prints during feature building
DEBUG_FEATURES = False


def build_home_away_matrix(labels: pd.DataFrame,
                           team_game: pd.DataFrame) -> pd.DataFrame:
    """
    Join team-game aggregates into a single row per game with home/away suffixes.
    Expects team_game columns: ['game_id','team','epa_mean','epa_std','success_rate','yards_mean','pass_rate', ...]
    """
    home = (
        team_game.rename(columns={
            "team": "home_team",
            "epa_mean": "home_epa_mean",
            "epa_std": "home_epa_std",
            "success_rate": "home_success_rate",
            "yards_mean": "home_yards_mean",
            "pass_rate": "home_pass_rate",
        })
    )
    away = (
        team_game.rename(columns={
            "team": "away_team",
            "epa_mean": "away_epa_mean",
            "epa_std": "away_epa_std",
            "success_rate": "away_success_rate",
            "yards_mean": "away_yards_mean",
            "pass_rate": "away_pass_rate",
        })
    )

    out = (
        labels
        .merge(home, on=["game_id", "home_team"], how="left", validate="one_to_many")
        .merge(away, on=["game_id", "away_team"], how="left", validate="one_to_many")
    )

    # Differentials
    for c in ["epa_mean", "epa_std", "success_rate", "yards_mean", "pass_rate"]:
        out[f"delta_{c}"] = out[f"home_{c}"] - out[f"away_{c}"]
    return out


def compute_team_rolling_features(
    team_game: pd.DataFrame,
    labels: pd.DataFrame,
    windows: List[int] = [3, 5],
    fallback_season: Optional[int] = None,  # accepted (for compatibility); not used unless you enable logic below
) -> pd.DataFrame:
    """
    Compute leakage-safe rolling features per team using game-date order.
    Returns a per-(game_id, team) table with columns like r3_epa_mean, r5_success_rate, etc.

    Required columns:
      labels: ['game_id','gameday','home_team','away_team','season','week','season_type']
      team_game: ['game_id','team','epa_mean','success_rate','yards_mean','pass_rate', ...]
    """
    # Ensure labels carry required columns
    needed = ["game_id", "gameday", "home_team", "away_team", "season", "week", "season_type"]
    miss = [c for c in needed if c not in labels.columns]
    if miss:
        raise KeyError(f"labels missing required columns: {miss}")

    # Long index of (game_id, team) in chronological order
    meta = labels[needed].copy()
    long_home = (
        meta[["game_id", "gameday", "home_team", "season", "week", "season_type"]]
        .rename(columns={"home_team": "team"})
        .assign(is_home=1)
    )
    long_away = (
        meta[["game_id", "gameday", "away_team", "season", "week", "season_type"]]
        .rename(columns={"away_team": "team"})
        .assign(is_home=0)
    )
    long = pd.concat([long_home, long_away], ignore_index=True).sort_values(["team", "gameday"])

    # Merge base per-team per-game stats (keep one 'game_id' column — no _x/_y)
    base_cols = ["epa_mean", "success_rate", "yards_mean", "pass_rate", "epa_std"]
    cols_present = [c for c in base_cols if c in team_game.columns]
    tg = long.merge(
        team_game[["game_id", "team"] + cols_present],
        on=["game_id", "team"],
        how="left",
        validate="many_to_one",
    ).reset_index(drop=True)

    if DEBUG_FEATURES:
        print("DEBUG: After merge in compute_team_rolling_features — sample:")
        print(tg.head())
        nulls = tg[cols_present].isna().sum()
        print("DEBUG: Null counts for core columns:")
        print(nulls)

    # Leakage-safe rolling stats: shift(1) so current game doesn't leak into its own features
    def _roll(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("gameday").reset_index(drop=True)
        for w in windows:
            for c in cols_present:
                df[f"r{w}_{c}"] = df[c].shift(1).rolling(window=w, min_periods=1).mean()
        return df

    rolled = tg.groupby("team", group_keys=False).apply(_roll)

    # Keep only ID + rolling columns
    roll_cols = [c for c in rolled.columns if c.startswith("r")]
    # Fill NaNs produced by early weeks with 0 (common baseline)
    rolled[roll_cols] = rolled[roll_cols].fillna(0)

    # Select tidy output
    out = rolled[["game_id", "team"] + roll_cols].copy()

    # Optional: if you later want a true fallback (e.g., fill missing base stats from a prior season),
    # you can implement it here, making use of `fallback_season`. For now, we only accept the arg
    # for compatibility to avoid TypeError from older callers.
    return out


def add_rolling_to_matrix(matrix: pd.DataFrame, rolled: pd.DataFrame) -> pd.DataFrame:
    """
    Attach rolling features to home/away teams and compute differentials like delta_r3_epa_mean.
    """
    # Home side
    home_r = (
        rolled.rename(columns={"team": "home_team"})
        .add_prefix("home_")
        .rename(columns={"home_game_id": "game_id", "home_home_team": "home_team"})
    )
    # Away side
    away_r = (
        rolled.rename(columns={"team": "away_team"})
        .add_prefix("away_")
        .rename(columns={"away_game_id": "game_id", "away_away_team": "away_team"})
    )

    out = (
        matrix
        .merge(home_r, on=["game_id", "home_team"], how="left", validate="one_to_one")
        .merge(away_r, on=["game_id", "away_team"], how="left", validate="one_to_one")
    )

    # Rolling differentials (home minus away) for each rolling stat that exists on both sides
    roll_cols = sorted([c for c in out.columns if c.startswith("home_r")])
    for hc in roll_cols:
        base = hc.replace("home_", "")
        ac = "away_" + base
        if ac in out.columns:
            out[f"delta_{base}"] = out[hc] - out[ac]

    return out


def build_feature_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Select delta_* features and the target label for training.
    """
    feature_cols = [c for c in df.columns if c.startswith("delta_")]
    X = df[feature_cols].fillna(0)
    y = df["home_win"].astype(int)
    return X, y
