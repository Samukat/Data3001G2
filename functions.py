import pandas as pd
import numpy as np
import os

DATA_FOLDER = "data"
CAR_WIDTH_BUFFER = 0.54
MAX_DISTANCE = 1200

RELEVANT_COLS = [
    "M_WORLDPOSITIONX_1", "M_WORLDPOSITIONY_1", "M_WORLDPOSITIONZ_1", "M_LAPDISTANCE_1"
]

RENAME_COLS = {
    "M_WORLDPOSITIONX_1": "WORLDPOSX",
    "M_WORLDPOSITIONY_1": "WORLDPOSY",
    "M_WORLDPOSITIONZ_1": "WORLDPOSZ",
    "M_LAPDISTANCE_1": "LAPDISTANCE"
}


def load_entire_track():
    ''' Load all reference track data from CSV files in the given directory.
    Returns track_left, track_right, track_line, turns DataFrames.
    '''

    track_left = pd.read_csv(f"{DATA_FOLDER}/f1sim-ref-left.csv")
    track_right = pd.read_csv(f"{DATA_FOLDER}/f1sim-ref-right.csv")
    track_line = pd.read_csv(f"{DATA_FOLDER}/f1sim-ref-line.csv")
    turns = pd.read_csv(f"{DATA_FOLDER}/f1sim-ref-turns.csv")

    return track_left, track_right, track_line, turns


def load_race_data(file_name):
    data = pd.read_csv(os.path.join(DATA_FOLDER, file_name))
    data = data[RELEVANT_COLS].rename(columns=RENAME_COLS)
    return data


def filter_by_distance(data: pd.DataFrame) -> pd.DataFrame:
    return data[data['M_LAPDISTANCE_1'] <= MAX_DISTANCE].reset_index(drop=True)
