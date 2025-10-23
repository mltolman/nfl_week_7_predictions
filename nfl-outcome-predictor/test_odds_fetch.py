# test_odds_fetch.py
from src.nfl_outcome.sportsdata_odds import fetch_closing_odds

def main():
    # pick a recent season and week you know should have odds
    season = 2024
    week = 1
    print(f"Fetching odds for Season {season}, Week {week}")
    df = fetch_closing_odds(season, week)
    print("Result rows:", len(df))
    print(df.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
