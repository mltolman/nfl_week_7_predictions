import os
import requests
import pandas as pd
from typing import List, Dict, Any

# Load .env (if exists) into environment variables
from dotenv import load_dotenv
load_dotenv()

SPORTSDATA_API_KEY = os.getenv("SPORTSDATA_API_KEY")
if not SPORTSDATA_API_KEY:
    raise RuntimeError("SPORTSDATA_API_KEY environment variable not set.")

BASE_URL = "https://api.sportsdata.io/v3/nfl/odds/json"

def fetch_closing_odds(season: int, week: int) -> pd.DataFrame:
    """
    Fetch closing odds for a given season and week via SportsData.io API.
    Returns a DataFrame with columns: season, week, home_team, away_team, spread_close, total_close.
    """
    url = f"{BASE_URL}/GameOddsByWeek/{season}/{week}"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch odds for {season} week {week}: {resp.status_code} {resp.text}")

    data = resp.json()
    rows: List[Dict[str, Any]] = []

    for game in data:
        season_val = game.get("Season")
        week_val = game.get("Week")
        home = game.get("HomeTeamName")
        away = game.get("AwayTeamName")

        pre_odds = game.get("PregameOdds", [])
        if not pre_odds:
            rows.append({
                "season": season_val,
                "week": week_val,
                "home_team": home,
                "away_team": away,
                "spread_close": None,
                "total_close": None
            })
        else:
            # use the last odds in the list as the closing line
            last = pre_odds[-1]
            home_spread = last.get("HomePointSpread")
            ou = last.get("OverUnder")
            rows.append({
                "season": season_val,
                "week": week_val,
                "home_team": home,
                "away_team": away,
                "spread_close": home_spread,
                "total_close": ou
            })

    return pd.DataFrame(rows)

def fetch_odds_for_period(season_week_pairs: List[tuple[int, int]]) -> pd.DataFrame:
    """
    Fetch closing odds for a list of (season, week) pairs.
    Returns a concatenated DataFrame.
    """
    dfs = []
    for season, week in season_week_pairs:
        try:
            df = fetch_closing_odds(season, week)
            dfs.append(df)
        except Exception as e:
            # skip if failure for a particular week
            print(f"Warning: could not fetch odds for season {season}, week {week}: {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()
