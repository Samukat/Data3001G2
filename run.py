import pandas as pd
import os

from cleaning_functions import (
    remove_other_tracks,
    renaming_cols
)

from functions import (
    load_race_data,
    load_entire_track,
    filter_by_distance,
    add_lap_id
)

from feature_functions import (
    compute_distance_to_apex,
    compute_angle_to_apex,
    car_edge_distances,
    calculate_track_width,
    id_outoftrack
)


def build_dataset(df: pd.DataFrame = None) -> pd.DataFrame:

    # Load data
    print("1. Loading Data")
    if df is None:
        data = load_race_data("UNSW F12024.csv")

    track_left, track_right, _, turns = load_entire_track()

    # Race data filtering
    print("2. Filtering")
    data = filter_by_distance(df)
    data = remove_other_tracks(data)
    data = remove_na(data, subset=['M_WORLDPOSITIONX_1', "M_WORLDPOSITIONY_1"])

    data = renaming_cols

    # Track Width
    print("3. Track width")
    left_with_width, right_with_width = calculate_track_width(track_left.sort_values(
        by="FRAME").reset_index(), track_right.sort_values(by="FRAME").reset_index())

    result_df = car_edge_distances(data, left_with_width, right_with_width)

    return result_df

    # Rename columns
    # data = renaming_cols(data)


if __name__ == "__main__":
    dataset = build_dataset()
    # Todo: Save dataset
