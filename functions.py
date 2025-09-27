import pandas as pd
import numpy as np
import os

DATA_FOLDER = "data"
CAR_WIDTH_BUFFER = 0.54
MAX_DISTANCE = 1200

RELEVANT_COLS = [
    "SESSION_GUID", "M_CURRENTLAPNUM", "M_SESSIONUID", "M_SPEED_1",
    "M_THROTTLE_1", "M_STEER_1", "M_BRAKE_1", "M_GEAR_1", "M_ENGINERPM_1",
    "M_DRS_1", "M_BRAKESTEMPERATURE_RL_1", "M_BRAKESTEMPERATURE_RR_1",
    "M_BRAKESTEMPERATURE_FL_1", "M_BRAKESTEMPERATURE_FR_1",
    "M_TYRESPRESSURE_RL_1", "M_TYRESPRESSURE_RR_1", "M_TYRESPRESSURE_FL_1",
    "M_TYRESPRESSURE_FR_1", "M_CURRENTLAPTIMEINMS_1", "M_LAPDISTANCE_1",
    "M_TOTALDISTANCE_1", "M_CURRENTLAPNUM_1", "M_CURRENTLAPINVALID_1",
    "M_WORLDPOSITIONX_1", "M_WORLDPOSITIONY_1", "M_WORLDPOSITIONZ_1",
    "M_WORLDFORWARDDIRX_1", "M_WORLDFORWARDDIRY_1", "M_WORLDFORWARDDIRZ_1",
    "M_WORLDRIGHTDIRX_1", "M_WORLDRIGHTDIRY_1", "M_WORLDRIGHTDIRZ_1",
    "M_YAW_1", "M_PITCH_1", "M_ROLL_1", "M_FRONTWHEELSANGLE", "M_TRACKID",
    "R_SESSION", "R_NAME", "R_STATUS", "LAPTIME", "CURRENTLAPTIME"

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
