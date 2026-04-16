from pathlib import Path

import pandas as pd
from core.config import CLEAN_MATCHES_PATH, RAW_MATCHES
from core.metadata import save_metadata


def clean_matches():

    print("Loading matches dataset...")
    matches = pd.read_csv(RAW_MATCHES)

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

    print("Removing D/L method matches...")
    matches = matches[matches["method"] != "D/L"]

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

    print("Checking if Folder is present...")
    CLEAN_MATCHES_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Saving parquet...")
    matches.to_parquet(CLEAN_MATCHES_PATH, index=False)

    print("Clean matches saved successfully.")

    save_metadata(
        dataset_name="clean_matches",
        dataset_path=CLEAN_MATCHES_PATH,
        raw_sources=[str(RAW_MATCHES)],
        preprocessing=[
            "season_fix",
            "missing_value_handling",
            "column_dropping",
            "date_conversion",
            "sorting",
        ],
    )


if __name__ == "__main__":
    clean_matches()
