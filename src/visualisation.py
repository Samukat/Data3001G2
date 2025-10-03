import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def plot_k_turns(data, turns, track_left, track_right, x_col="WORLDPOSITIONX", y_col="WORLDPOSITIONY", k=3, color_col="M_SPEED_1", title="Track Sections Near Turns", cmap="viridis"):
    fig, axs = plt.subplots(1, k, figsize=(18, 6))

    for i in range(k):
        turn_n = turns.iloc[i]

        x_min = min(turn_n["CORNER_X1"], turn_n["CORNER_X2"])
        x_max = max(turn_n["CORNER_X1"], turn_n["CORNER_X2"])
        y_min = min(turn_n["CORNER_Y1"], turn_n["CORNER_Y2"])
        y_max = max(turn_n["CORNER_Y1"], turn_n["CORNER_Y2"])

        near_turn_n = data[(data[x_col] >= x_min) & (data[x_col] <= x_max) &
                           (data[y_col] >= y_min) & (data[y_col] <= y_max)]

        tl_t2 = track_left[(track_left["WORLDPOSX"] >= x_min) & (track_left["WORLDPOSX"] <= x_max) &
                           (track_left["WORLDPOSY"] >= y_min) & (track_left["WORLDPOSY"] <= y_max)]
        tr_t2 = track_right[(track_right["WORLDPOSX"] >= x_min) & (track_right["WORLDPOSX"] <= x_max) &
                            (track_right["WORLDPOSY"] >= y_min) & (track_right["WORLDPOSY"] <= y_max)]

        scatter = axs[i].scatter(near_turn_n[x_col], near_turn_n[y_col],
                                 c=near_turn_n[color_col[i] if type(color_col) == list else color_col], cmap=cmap, s=3)

        axs[i].scatter(turn_n["APEX_X1"], turn_n["APEX_Y1"],
                       color="black", s=20, label="Apex")

        axs[i].scatter(tl_t2["WORLDPOSX"],
                       tl_t2["WORLDPOSY"], s=8, color="red")
        axs[i].scatter(tr_t2["WORLDPOSX"],
                       tr_t2["WORLDPOSY"], s=8, color="red")
        axs[i].set_title(f"Near Turn {i + 1}")

    fig.colorbar(scatter, ax=axs, orientation='vertical', label=color_col[0])
    plt.suptitle(title)
    plt.show()


def plot_laps(data, y_col="BRAKE", distance_range=(0, 300),
              exclude_laps=None, only_valid=True,
              figsize=(12, 8), point_size=1):
    """
    Plot lap data with LAPDISTANCE on the x-axis and a chosen column on the y-axis.

    Parameters
    ----------
    data : pd.DataFrame
        The cleaned lap data.
    y_col : str, default="BRAKE"
        Column to plot against LAPDISTANCE.
    distance_range : tuple, default=(0, 300)
        (min_distance, max_distance) range for LAPDISTANCE.
    exclude_laps : list or set, optional
        Laps to exclude from plotting.
    only_valid : bool, default=True
        If True, filter out invalid laps.
    figsize : tuple, default=(12, 8)
        Figure size.
    point_size : int, default=1
        Scatter point size.
    """

    d = data.copy()

    # Filter invalid laps
    if only_valid:
        d = d[d["invalid_lap"] == 0]

    # Exclude laps
    if exclude_laps is not None:
        d = d[~d["lap_id"].isin(exclude_laps)]

    # Distance filter
    d = d[(d["LAPDISTANCE"] >= distance_range[0]) &
          (d["LAPDISTANCE"] <= distance_range[1])]

    # Plot
    plt.figure(figsize=figsize)
    plt.scatter(d["LAPDISTANCE"], d[y_col], s=point_size, c=d["lap_id"])
    plt.xlabel("LAPDISTANCE")
    plt.ylabel(y_col)
    plt.title(f"{y_col} vs LAPDISTANCE")
    # plt.colorbar(label="lap_id")
    plt.show()
