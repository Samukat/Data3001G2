"""
Main pipeline execution script.
Run this from the project root directory.
"""
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add src directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

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

from src.config import FEATURES

# Configuration
OUTPUT_FOLDER = Path("data")
SET_DISTANCES = [360, 430, 530]


def print_header(message):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {message}")
    print("=" * 70)


def print_step(step_num, total_steps, message):
    """Print a formatted step message."""
    print(f"\n[{step_num}/{total_steps}] {message}")


def print_info(message, indent=3):
    """Print an indented info message."""
    print(" " * indent + f" {message}")


def build_dataset(df: pd.DataFrame = None, start_stage=0):
    """Execute the complete data pipeline."""
    
    print_header("STARTING DATA PIPELINE")

    # =====================================================================
    # STAGE 1: DATA PROCESSING
    # =====================================================================
    
    if start_stage == 0:
        total_steps = 7
        
        # Step 1: Load data
        print_step(1, total_steps, "Loading data...")
        df = load_race_data("UNSW F12024.csv", df)
        track_left, track_right, ref_line, turns = load_entire_track()
        print_info(f"Loaded {len(df):,} race records")
        print_info("Loaded track data (boundaries, reference line, turns)")

        # Step 2: Clean data
        print_step(2, total_steps, "Cleaning data...")
        initial_count = len(df)
        df = remove_other_tracks(df)
        df = remove_na(df, subset=['WORLDPOSITIONX', "WORLDPOSITIONY"])
        removed_count = initial_count - len(df)
        print_info(f"Removed {removed_count:,} records with missing/invalid data")
        print_info(f"Remaining: {len(df):,} records")

        # Step 3: Filter by distance
        print_step(3, total_steps, "Filtering by distance and creating laps...")
        initial_count = len(df)
        df = filter_by_distance(df)
        removed_count = initial_count - len(df)
        print_info(f"Filtered {removed_count:,} records outside distance range")
        df = add_lap_id(df)
        print_info(f"Created lap IDs for {len(df['lap_id'].unique()):,} unique laps")
        df = remove_lowinfo_laps(df)
        print_info(f"Remaining: {len(df):,} records after low-info lap removal")

        # Step 4: Compute track features
        print_step(4, total_steps, "Computing track features...")
        left_with_width, right_with_width = calculate_track_width(
            track_left, track_right
        )
        print_info("Calculated track width at all points")
        df = car_edge_distances(df, left_with_width, right_with_width)
        print_info("Computed distances to track edges")
        df = car_from_ref_line(df, ref_line)
        print_info("Computed distances from reference line")

        # Step 5: Compute apex features
        print_step(5, total_steps, "Computing apex features...")
        df = compute_distace_to_apex(df, turns)
        print_info("Calculated distances to apex points")
        df = compute_angle_to_apex(df, turns)
        print_info("Calculated angles to apex points")

        # Step 6: Identify off-track incidents
        print_step(6, total_steps, "Identifying off-track incidents...")
        df = id_outoftrack(df)
        n_valid = len(df[df["invalid_lap"] == 0]["lap_id"].drop_duplicates())
        n_invalid = len(df[df["invalid_lap"] == 1]["lap_id"].drop_duplicates())
        print_info(f"Valid laps: {n_valid:,}")
        print_info(f"Invalid laps (off-track): {n_invalid:,}")

        # Step 7: Save processed data
        print_step(7, total_steps, "Saving processed data...")
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        output_file = OUTPUT_FOLDER / "processed_race_data.csv"
        df.to_csv(output_file, index=False)
        print_info(f"Saved to: {output_file}")
        print_info(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    print_header("STAGE 1 COMPLETE: PROCESSED DATA READY")

    # =====================================================================
    # STAGE 2: FEATURE ENGINEERING
    # =====================================================================
    
    print_header("STAGE 2: CREATING FINAL FEATURE DATASET")

    # Extract moment points
    print("\nExtracting track moments...")
    with tqdm(total=4, desc="Computing moment points", ncols=70) as pbar:
        throttle_points = get_throttle_points(df)
        pbar.update(1)
        pbar.set_postfix_str("Throttle ")
        
        braking_points = get_braking_points(df)
        pbar.update(1)
        pbar.set_postfix_str("Braking ")
        
        apex_points = get_apex_points(df)
        pbar.update(1)
        pbar.set_postfix_str("Apex ")
        
        steering_points = get_steering_points(df)
        pbar.update(1)
        pbar.set_postfix_str("Steering ")

    features = FEATURES
    all_moments = []

    # Generate braking moments
    print("\n Generating braking moments...")
    with tqdm(total=2, desc="Braking features", ncols=70) as pbar:
        brake_point_start = moment_generator(
            df, "BPS", features,
            braking_points["BP_LD"],
            braking_points["max_brake_LD"]
        )
        pbar.update(1)
        
        brake_point_end = moment_generator(
            df, "BPE", features,
            braking_points["brake_decrease_LD"],
            braking_points["brake_end_LD"]
        )
        pbar.update(1)
    
    # Generate throttle moments
    print("Generating throttle moments...")
    with tqdm(total=2, desc="Throttle features", ncols=70) as pbar:
        throttle_point_start = moment_generator(
            df, "THS", features,
            throttle_points["first_lift_LD"],
            throttle_points["min_throttle_LD"]
        )
        pbar.update(1)
        
        throttle_point_end = moment_generator(
            df, "THE", features,
            throttle_points["back_on_LD"],
            throttle_points["back_on_max_LD"]
        )
        pbar.update(1)
    
    # Generate steering moments
    print("Generating steering moments...")
    with tqdm(total=3, desc="Steering features", ncols=70) as pbar:
        steering_point_start = moment_generator(
            df, "STS", features,
            steering_points["first_steer_LD"],
            steering_points["max_pos_angle_steer_LD"]
        )
        pbar.update(1)
        
        steering_point_mid = moment_generator(
            df, "STM", features,
            steering_points["middle_TP_LD"]
        )
        pbar.update(1)
        
        steering_point_end = moment_generator(
            df, "STE", features,
            steering_points["end_steer_LD"],
            steering_points["max_neg_LD"]
        )
        pbar.update(1)
    
    # Generate apex moments
    print("Generating apex moments...")
    with tqdm(total=2, desc="Apex features", ncols=70) as pbar:
        apex_point1 = moment_generator(
            df, "APX1", features,
            apex_points["apex1_LD"]
        )
        pbar.update(1)
        
        apex_point2 = moment_generator(
            df, "APX2", features,
            apex_points["apex2_LD"]
        )
        pbar.update(1)

    # Generate distance-based moments
    print("Generating distance-based moments...")
    laps_df = df[["lap_id"]].drop_duplicates().set_index("lap_id").sort_index().copy()
    dist_moments = []
    
    for d in tqdm(SET_DISTANCES, desc="Distance features", ncols=70):
        dist_df = laps_df.copy()
        dist_df.loc[:, "dist"] = d
        dist_moment = moment_generator(df, f"dist_{d}", features, dist_df["dist"])
        dist_moments.append(dist_moment)

    # Generate target variable
    print("Generating target variable...")
    dist_df = laps_df.copy()
    dist_df.loc[:, "dist"] = 900
    target_df = moment_generator(df, "Target", ["CURRENTLAPTIMEINMS"], dist_df["dist"])

    # Combine all features
    print("\nCombining all features...")
    invalid_lap_flag = df[["lap_id", "invalid_lap"]].drop_duplicates().set_index("lap_id").sort_index().copy()
    all_moments = [
        invalid_lap_flag,
        brake_point_start,
        brake_point_end,
        throttle_point_start,
        throttle_point_end,
        steering_point_start,
        steering_point_mid,
        steering_point_end,
        apex_point1,
        apex_point2,
        target_df
    ] + dist_moments

    output = pd.concat(all_moments, axis=1)

    # Save final output
    print("\nSaving final dataset...")
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_FOLDER / "final_data_product.csv"
    output.to_csv(output_file, index=False)
    print_info(f"Saved to: {output_file}")
    print_info(f"Final shape: {output.shape[0]:,} rows × {output.shape[1]} columns")
    print_info(f"Features: {output.shape[1]} total columns")

    print_header("PIPELINE COMPLETE! 🎉")
    
    print("\n Summary:")
    print(f"   • Total laps: {len(output):,}")
    print(f"   • Total features: {output.shape[1]}")
    print(f"   • Output file: {output_file}")

    return output


if __name__ == "__main__":
    try:
        result = build_dataset()
        print("\n Pipeline executed successfully!\n")
    except Exception as e:
        print(f"\n Pipeline failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        print("\n" + "=" * 70)
        print("Full traceback:")
        print("=" * 70)
        traceback.print_exc()
        sys.exit(1)