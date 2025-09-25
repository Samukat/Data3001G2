import pandas as pd
import os

from cleaning_functions import remove_other_tracks, remove_na, car_edge_distances, calculate_track_width
from functions import load_race_data, load_entire_track, filter_by_distance


def build_dataset():
    # Load data
    print("1. Loading Data")
    data = load_race_data("UNSW F12024.csv")
    track_left, track_right, _, turns = load_entire_track()

    # Race data filtering
    print("2. Filtering")
    data = remove_other_tracks(data)
    data = filter_by_distance(data)
    data = remove_na(data, subset=['M_WORLDPOSITIONX_1', "M_WORLDPOSITIONY_1"])

    # Track Width
    print("3. Track width")
    left_with_width, right_with_width = calculate_track_width(track_left.sort_values(
        by="FRAME").reset_index(), track_right.sort_values(by="FRAME").reset_index())

    # result_df = car_edge_distances(run_data_filtered, left_with_width, right_with_width)


if __name__ == "__main__":
    dataset = build_dataset()
    # Todo: Save dataset
