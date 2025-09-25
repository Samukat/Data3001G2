import pandas as pd


def remove_other_tracks(data: pd.DataFrame) -> pd.DataFrame:
    """
    Removes all rows from the dataframe where the TRACKID is not 1.
    """
    return data[data['TRACKID'] == 0].reset_index(drop=True)


def filter_by_distance(data: pd.DataFrame, max_distance: float) -> pd.DataFrame:
    return data[data['M_LAPDISTANCE_1'] <= 1200].reset_index(drop=True)


def remove_na(data: pd.DataFrame, subset) -> pd.DataFrame:
    return data.dropna(subset=subset).reset_index(drop=True)
