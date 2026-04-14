from pathlib import Path

import pandas as pd

RAW_PATH = "../New Data/matches_updated_ipl_upto_2025.csv"
OUTPUT_PATH = "../ml-service/data/processed/clean_matches.parquet"


def clean_matches():

    print("Loading matches dataset...")
    matches = pd.read_csv(RAW_PATH)

    print("Fixing season format...")
    matches["season"] = matches["season"].apply(
        lambda x: (
            int(x.split("/")[0])
            if x == "2020/21"
            else int("20" + x.split("/")[1]) if "/" in x else int(x)
        )
    )

    print("Handling missing values...")
    matches["winner_runs"] = matches["winner_runs"].fillna(0)
    matches["winner_wickets"] = matches["winner_wickets"].fillna(0)

    matches["winner"] = matches["winner"].fillna(matches["eliminator"])

    print("Dropping unnecessary columns...")
    matches.drop(
        [
            "event",
            "umpire2",
            "toss_winner",
            "neutralvenue",
            "umpire1",
            "reserve_umpire",
            "match_referee",
            "tv_umpire",
            "eliminator",
            "date1",
            "date2",
            "toss_decision",
            "method",
            "gender",
            "balls_per_over",
            "outcome",
            "city",
            "match_number",
        ],
        axis=1,
        inplace=True,
        errors="ignore",
    )

    print("Removing matches without winner...")
    matches = matches[matches["winner"].notna()]

    print("Converting types...")
    matches["date"] = pd.to_datetime(matches["date"])
    matches["winner_runs"] = matches["winner_runs"].astype(int)
    matches["winner_wickets"] = matches["winner_wickets"].astype(int)

    print("Sorting by date...")
    matches = matches.sort_values("date")

    print("Saving parquet...")
    matches.to_parquet(OUTPUT_PATH, index=False)

    print("Clean matches saved successfully.")


if __name__ == "__main__":
    clean_matches()
