import pandas as pd
from .config import TRACK_ID, MAX_DISTANCE


def remove_other_tracks(data: pd.DataFrame) -> pd.DataFrame:
    """
    Removes all rows from the dataframe where the TRACKID is not 1.
    """
    return data[data['TRACKID'] == TRACK_ID].reset_index(drop=True)


def remove_na(data: pd.DataFrame, subset) -> pd.DataFrame:
    return data.dropna(subset=subset).reset_index(drop=True)


def filter_by_distance(data: pd.DataFrame) -> pd.DataFrame:
    return data[data['LAPDISTANCE'] <= MAX_DISTANCE].reset_index(drop=True)

