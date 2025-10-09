"""
Main pipeline execution script.
Run this from the project root directory.
"""
import sys
from pathlib import Path

# Add src directory to Python path
# project_root = Path(__file__).parent
# sys.path.insert(0, str(project_root / "src"))

from src.track_features import (
    car_edge_distances,
    calculate_track_width,
    compute_distace_to_apex,
    compute_angle_to_apex,
    id_outoftrack,
    add_lap_id,
    car_from_ref_line
)
from src.data_cleaning import (
    remove_other_tracks,
    remove_na,
    # renaming_cols,
    filter_by_distance,
    remove_lowinfo_laps
)
from src.data_loader import load_race_data, load_entire_track


import pandas as pd
import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))


def build_dataset(df: pd.DataFrame = None):
    """Execute the complete data pipeline."""

    print()
    print("=" * 60)
    print("Starting Data Pipeline")
    print("=" * 60)

    print("\n[1/6] Loading data...")
    df = load_race_data("UNSW F12024.csv", df)

    track_left, track_right, ref_line, turns = load_entire_track()
    print(f"   Loaded {len(df)} race records")
    print(f"   Loaded track data")

    print("\n[2/6] Cleaning data...")
    df = remove_other_tracks(df)
    df = remove_na(df, subset=['WORLDPOSITIONX', "WORLDPOSITIONY"])

    print(f"   {len(df)} records after cleaning")

    print("\n[3/6] Filtering by distance and points...")
    df = filter_by_distance(df)
    print(f"   {len(df)} records after filtering")
    df = add_lap_id(df)
    df = remove_lowinfo_laps(df)

    print("\n[4/6] Computing track features...")
    left_with_width, right_with_width = calculate_track_width(
        track_left, track_right)
    df = car_edge_distances(df, left_with_width, right_with_width)
    df = car_from_ref_line(df, ref_line)

    print("\n[5/6] Computing apex features...")
    df = compute_distace_to_apex(df, turns)
    df = compute_angle_to_apex(df, turns)

    # Step 6: Identify off-track incidents
    print("\n[6/6] Identifying off-track incidents...")
    df = id_outoftrack(df)
    n_valid = len(df[df["invalid_lap"] == 0]["lap_id"].drop_duplicates())
    n_invalid = len(df[df["invalid_lap"] == 1]["lap_id"].drop_duplicates())
    print(
        f"Number of valid laps {n_valid}, number of invalid laps {n_invalid}")

    # Save processed data
    print("\n[7/7] Saving processed data...")
    # OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    # output_file = OUTPUT_FOLDER / "processed_race_data.csv"
    # race_data.to_csv(output_file, index=False)
    # print(f"   Saved to: {output_file}")

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    return df


if __name__ == "__main__":
    try:
        result = build_dataset()
        print("\nPipeline executed successfully!")
    except Exception as e:
        print(f"\n Pipeline failed with error:")
        print(f"  {type(e).__name__}: {e}")
        sys.exit(1)
