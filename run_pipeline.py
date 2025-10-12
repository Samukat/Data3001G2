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
    filter_by_distance,
    remove_lowinfo_laps
)

from src.data_loader import (
    load_race_data, 
    load_entire_track
)

from src.track_moments import (
    moment_generator,
    get_throttle_points,
    get_braking_points,
    get_apex_points,
    get_steering_points
)

from src.config import (
    FEATURES,
    SET_DISTANCES
)

import pandas as pd
import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))
OUTPUT_FOLDER = "data"
SET_DISTANCES = [360, 430, 530]


def build_dataset(df: pd.DataFrame = None, start_stage = 0):
    """Execute the complete data pipeline."""

    print()
    print("=" * 60)
    print("Starting Data Pipeline")
    print("=" * 60)

    if start_stage == 0:
        print("\n[1/6] Loading data...")
        df = load_race_data("UNSW F12024.csv", df) # loads only if df not passed

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
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        output_file = OUTPUT_FOLDER / "processed_race_data.csv"
        df.to_csv(output_file, index=False)
        print(f"   Saved to: {output_file}")

    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\n" + "=" * 60)
    print("Stage 1 (processed data) Complete!")
    print("=" * 60)


    print("=" * 60)
    print("Creating Final Product")
    print("=" * 60)

    throttle_points = get_throttle_points(df)
    braking_points = get_braking_points(df)
    apex_points = get_apex_points(df)
    steering_points = get_steering_points(df)

    features = []

    # --- Braking moments ---
    brake_point_start = moment_generator(df, "BPS", features,
                                     braking_points["BP_LD"],
                                     braking_points["max_brake_LD"])

    brake_point_end = moment_generator(df, "BPE", features,
                                    braking_points["brake_decrease_LD"],
                                    braking_points["brake_end_LD"])
    
    # --- Throttle moments ---
    throttle_point_start = moment_generator(df, "THS", features,
                                        throttle_points["first_lift_LD"],
                                        throttle_points["min_throttle_LD"])

    throttle_point_end = moment_generator(df, "THE", features,
                                        throttle_points["back_on_LD"],
                                        throttle_points["back_on_max_LD"])
    
    # --- Steering moments ---
    steering_point_start = moment_generator(df, "STS", features,
                                            steering_points["first_steer_LD"],
                                            steering_points["max_pos_angle_steer_LD"])

    steering_point_mid = moment_generator(df, "STM", features,
                                        steering_points["middle_TP_LD"])

    steering_point_end = moment_generator(df, "STE", features,
                                        steering_points["end_steer_LD"],
                                        steering_points["max_neg_LD"])
    
    # --- Apex moments ---
    apex_point1 = moment_generator(df, "APX1", features,
                              apex_points["apex1_LD"])
    
    apex_point1 = moment_generator(df, "APX2", features,
                              apex_points["apex2_LD"])


    dist_moments = []
    laps_df = df[["lap_id"]].drop_duplicates().set_index("lap_id").sort_index().copy()
    for d in SET_DISTANCES:
        dist_df = laps_df.copy()
        dist_df.loc[:, "dist"] = d
        dist_moment = moment_generator(df, f"dist_{d}", features, dist_df["dist"])
        dist_moments.append(dist_moment)

    dist_df = laps_df.copy()
    dist_df.loc[:, "dist"] = 900
    target_df = moment_generator(df, "Target", ["CURRENTLAPTIMEINMS"], dist_df["dist"])

    invalid_lap_flag = df[["lap_id", "invalid_lap"]].drop_duplicates().set_index("lap_id").sort_index().copy()
    all_moments = [invalid_lap_flag, brake_point_start, brake_point_end, throttle_point_start, throttle_point_end, steering_point_start, steering_point_mid, steering_point_end, apex_point1, apex_point1, target_df] + dist_moments

    output = pd.concat(all_moments, axis = 1)

    print("\nSaving processed data...")
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_FOLDER / "final_data_product.csv"
    output.to_csv(output_file, index=False)

    print(f"   Saved to: {output_file}")
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"\nFinal dataset shape: {output.shape}")
    print(f"Columns: {list(output.columns)}")

    return output


if __name__ == "__main__":
    try:
        result = build_dataset()
        print("\nPipeline executed successfully!")
    except Exception as e:
        print(f"\n Pipeline failed with error:")
        print(f"  {type(e).__name__}: {e}")
        sys.exit(1)
