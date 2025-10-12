import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import KDTree


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
              figsize=(12, 8), point_size=1, ax=None):
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
        Figure size (only used if ax is None).
    point_size : int, default=1
        Scatter point size.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. If None, a new figure/axis is created.
    """

    plot_true = True if ax is None else False
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

    # Make axis if not supplied
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot
    sc = ax.scatter(d["LAPDISTANCE"], d[y_col],
                    s=point_size, c=d["lap_id"])
    ax.set_xlabel("LAPDISTANCE")
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs LAPDISTANCE")
    if plot_true:
        plt.show()

    return ax


def plot_lap(data, track_left, track_right, turns,
             x_col="WORLDPOSITIONX", y_col="WORLDPOSITIONY",
             color_col="M_SPEED_1", title="Full Lap with Local Track Boundaries",
             cmap="viridis", radius=30, ax=None):
    """
    Plot a full lap trajectory, showing only track boundaries and apex points
    within a specified radius (in meters) of the lap line.

    Parameters
    ----------
    data : pd.DataFrame
        Lap telemetry data.
    track_left, track_right : pd.DataFrame
        Track boundary coordinates.
    turns : pd.DataFrame
        Apex point data (expects columns 'APEX_X1', 'APEX_Y1').
    radius : float
        Radius (m) around lap path within which to show boundaries/apexes.
    """

    plot_true = True if ax is None else False

    # Build KDTree for the lap path
    lap_points = np.vstack([data[x_col], data[y_col]]).T
    lap_tree = KDTree(lap_points)

    # Helper function to get nearby points
    def filter_nearby(df, x="WORLDPOSX", y="WORLDPOSY"):
        pts = np.vstack([df[x], df[y]]).T
        idx = lap_tree.query_ball_point(pts, r=radius)
        mask = np.array([len(i) > 0 for i in idx])
        return df[mask]

    # Filter left/right track points within radius
    tl_near = filter_nearby(track_left)
    tr_near = filter_nearby(track_right)

    # Filter apex points near lap
    apex_points = turns[["APEX_X1", "APEX_Y1"]].dropna()
    apex_near = filter_nearby(apex_points, x="APEX_X1", y="APEX_Y1")

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(data[x_col], data[y_col], c=data[color_col],
                         s=3, cmap=cmap, label="Lap Path")
    ax.scatter(tl_near["WORLDPOSX"], tl_near["WORLDPOSY"],
               s=1, color="red", label="Track Left", )
    ax.scatter(tr_near["WORLDPOSX"], tr_near["WORLDPOSY"],
               s=1, color="red", alpha=0.7, label="Track Right")
    ax.scatter(apex_near["APEX_X1"], apex_near["APEX_Y1"],
               s=25, color="black", alpha=0.7, label="Apex")

    if plot_true:
        plt.colorbar(scatter, ax=ax, label=color_col)

    ax.set_title(title)
    # ax.set_xlabel("World X")
    # ax.set_ylabel("World Y")
    ax.legend()

    if plot_true:
        plt.axis("equal")
        plt.show()
