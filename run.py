import pandas as pd
from cleaning_functions import remove_other_tracks, filter_by_distance

track_left = pd.read_csv("data/f1sim-ref-left.csv")
track_right = pd.read_csv("data/f1sim-ref-right.csv")
track_line = pd.read_csv("data/f1sim-ref-line.csv")
turns = pd.read_csv("data/f1sim-ref-turns.csv")


data = pd.read_csv("data/f1sim-2022-2023.csv")

data = remove_other_tracks(data)
data = filter_by_distance(data, 1200)
data = remove_na(data, subset=['M_WORLDPOSITIONX_1', "M_WORLDPOSITIONY_1"])
