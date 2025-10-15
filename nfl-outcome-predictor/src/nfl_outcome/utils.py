from __future__ import annotations
import pandas as pd

TEAM_ABBR_MAP = {
    # Historical/alias mappings to nflverse conventions when necessary
    "JAX": "JAC",
    "WSH": "WAS",
    "STL": "LA",     # for older rams seasons; nflverse uses 'LA' / 'LAR' depending on year
    "SD": "LAC",
}

def standardize_team(abbr: str) -> str:
    if pd.isna(abbr):
        return abbr
    return TEAM_ABBR_MAP.get(abbr, abbr)