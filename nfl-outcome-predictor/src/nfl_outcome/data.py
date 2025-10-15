from __future__ import annotations
import pandas as pd
from typing import Iterable, Tuple
import nflreadpy as nfl

# ---------- helpers: normalize schedule/games schema ----------
def _normalize_games_df(games: pd.DataFrame) -> pd.DataFrame:
    df = games.copy()

    # nflreadpy schedules use 'game_type' (PRE/REG/POST). Create a stable 'season_type'.
    if "season_type" not in df.columns and "game_type" in df.columns:
        df["season_type"] = df["game_type"]

    # Ensure we have a 'gameday' datetime-like column
    if "gameday" not in df.columns:
        # common alternates seen across nflverse stacks
        if "game_date" in df.columns:
            df["gameday"] = pd.to_datetime(df["game_date"])
        elif "gamedate" in df.columns:
            df["gameday"] = pd.to_datetime(df["gamedate"])
        elif "start_time" in df.columns:
            df["gameday"] = pd.to_datetime(df["start_time"])
        else:
            # fallback to NaT; downstream code can still run (rolling features may be sparse)
            df["gameday"] = pd.NaT

    return df

# ---------------------- public API ----------------------
def load_data(seasons: Iterable[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download play-by-play and games tables for given seasons."""
    pbp = nfl.load_pbp(list(seasons)).to_pandas()
    games = nfl.load_schedules(list(seasons)).to_pandas()
    games = _normalize_games_df(games)
    return pbp, games

def filter_offense_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    plays = pbp[pbp.play_type.isin(["pass", "run"])].copy()
    plays["success"] = (plays["epa"] > 0).astype(int)
    return plays

def team_game_aggregates(plays: pd.DataFrame) -> pd.DataFrame:
    agg = (plays
           .groupby(["game_id", "posteam"], as_index=False)
           .agg(epa_mean=("epa", "mean"),
                epa_std=("epa", "std"),
                success_rate=("success", "mean"),
                yards_mean=("yards_gained", "mean"),
                pass_rate=("play_type", lambda x: (x == "pass").mean()))
           .rename(columns={"posteam":"team"}))
    return agg

def game_labels(games: pd.DataFrame) -> pd.DataFrame:
    # Expect these to be present in nflreadpy schedules; we normalized two of them above.
    needed = ["game_id", "season", "week", "season_type",
              "home_team", "away_team", "home_score", "away_score", "gameday"]
    missing = [c for c in needed if c not in games.columns]
    if missing:
        raise KeyError(f"Required columns missing from games: {missing}")

    lbl = games[needed].copy()
    lbl["home_win"] = (lbl["home_score"] > lbl["away_score"]).astype(int)
    return lbl
