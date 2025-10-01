import pandas as pd
import os

from .config import DATAFOLDER, RELEVANT_COLS, RENAME_COLS


def load_entire_track():
    ''' Load all reference track data from CSV files in the given directory.
    Returns track_left, track_right, track_line, turns DataFrames.
    '''

    track_left = pd.read_csv(f"{DATAFOLDER}/f1sim-ref-left.csv")
    track_right = pd.read_csv(f"{DATAFOLDER}/f1sim-ref-right.csv")
    track_line = pd.read_csv(f"{DATAFOLDER}/f1sim-ref-line.csv")
    turns = pd.read_csv(f"{DATAFOLDER}/f1sim-ref-turns.csv")

    return track_left, track_right, track_line, turns


def load_race_data(file_name, data=None):
    if data is None:
        data = pd.read_csv(os.path.join(DATAFOLDER, file_name),
                           usecols=RELEVANT_COLS)

    data = data[RELEVANT_COLS]

    data.columns = [col.replace("M_", "").replace("_1", "")
                    for col in data.columns]
    return data.rename(columns=RENAME_COLS)
