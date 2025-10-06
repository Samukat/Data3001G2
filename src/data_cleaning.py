import pandas as pd
from .config import TRACK_ID, MAX_DISTANCE, MIN_POINTS_LAP


def remove_other_tracks(data: pd.DataFrame) -> pd.DataFrame:
    """
    Removes all rows from the dataframe where the TRACKID is not 1.
    """
    return data[data['TRACKID'] == TRACK_ID].reset_index(drop=True)


def remove_na(data: pd.DataFrame, subset) -> pd.DataFrame:
    return data.dropna(subset=subset).reset_index(drop=True)


def filter_by_distance(data: pd.DataFrame) -> pd.DataFrame:
    return data[data['LAPDISTANCE'] <= MAX_DISTANCE].reset_index(drop=True)


def remove_lowinfo_laps(data: pd.DataFrame, min_points=MIN_POINTS_LAP) -> pd.DataFrame:
    gp_data = data.groupby("lap_id").size()
    lap_ids = gp_data[gp_data < min_points].sort_values()
    strs = '\n    '.join(
        [f"{id} - {count} data-point(s)" for id, count in lap_ids.items()])
    print(f"Removing laps: ")
    print("    " + strs)
    len_data = len(data)
    data = data[~data["lap_id"].isin(lap_ids.index)]
    print(f"Removed {len_data - len(data)} rows")
    return data
