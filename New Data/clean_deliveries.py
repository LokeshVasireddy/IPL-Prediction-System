from pathlib import Path

import pandas as pd

RAW_PATH = "../New Data/deliveries_updated_ipl_upto_2025.csv"
MATCHES_PATH = "../ml-service/data/processed/clean_matches.parquet"
OUTPUT_PATH = "../ml-service/data/processed/clean_deliveries.parquet"


def clean_deliveries():

    print("Loading deliveries dataset...")
    balls = pd.read_csv(RAW_PATH)

    print("Converting date...")
    balls["date"] = pd.to_datetime(balls["date"])

    print("Filling missing values...")
    balls["isWide"] = balls["isWide"].fillna(0)
    balls["isNoBall"] = balls["isNoBall"].fillna(0)
    balls["player_dismissed"] = balls["player_dismissed"].fillna("0")
    balls["Byes"] = balls["Byes"].fillna(0)
    balls["LegByes"] = balls["LegByes"].fillna(0)
    balls["Penalty"] = balls["Penalty"].fillna(0)

    print("Creating Batsman runs...")
    balls["batsman_runs"] = balls["batsman_runs"] + balls["Byes"] + balls["LegByes"]

    print("Dropping unnecessary columns...")
    balls.drop(
        ["Byes", "LegByes", "over_ball", "dismissal_kind", "extras"],
        axis=1,
        inplace=True,
        errors="ignore",
    )

    print("Keeping only innings 1 and 2...")
    balls = balls[balls["inning"].isin([1, 2])].copy()

    print("Properly Calculating..")
    balls["isWide"] = balls["isWide"] + balls["Penalty"]
    balls.drop(columns=["Penalty"], inplace=True)

    print("Mapping innings...")
    balls["inning"] = balls["inning"].map({1: 0, 2: 1})

    print("Converting types...")
    balls["batsman_runs"] = balls["batsman_runs"].astype(int)
    balls["isWide"] = balls["isWide"].astype(int)
    balls["isNoBall"] = balls["isNoBall"].astype(int)

    print("Removing washed-out matches from deliveries...")
    matches = pd.read_parquet(MATCHES_PATH)
    valid_match_ids = set(matches["matchId"].unique())

    balls = balls[balls["matchId"].isin(valid_match_ids)]

    print("Sorting by date...")
    balls = balls.sort_values(["date", "matchId", "inning", "over", "ball"])

    print("Removing duplicate rows...")
    balls = balls.drop_duplicates()

    print("Saving parquet...")
    balls.to_parquet(OUTPUT_PATH, index=False)

    print("Clean deliveries saved successfully.")


if __name__ == "__main__":
    clean_deliveries()
