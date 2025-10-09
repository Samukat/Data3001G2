import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm
from .geometry_utils import euclidean_distance, projection_values, angle_between


x_col = "WORLDPOSITIONX"
y_col = "WORLDPOSITIONY"


def car_edge_distances(data, track_left, track_right, x_col=x_col, y_col=y_col):
    ''' For each point in data, find the closest points on the left and right track edges.
    by:
    1. finding the two closest points on each
    2. projecting the point onto the line segment defined by those two points
    Returns a DataFrame with distances and track widths at the closest points.
    Assumes track_left and track_right have 'width' columns from calculate_track_width().
    '''

    # Fix: Use correct column names for track edges
    left_points = np.vstack(
        [track_left["WORLDPOSX"], track_left["WORLDPOSY"]]).T
    right_points = np.vstack(
        [track_right["WORLDPOSX"], track_right["WORLDPOSY"]]).T

    left_tree = KDTree(left_points)
    right_tree = KDTree(right_points)

    df = data.copy()

    # Pre-allocate arrays for faster assignment
    left_dists = np.zeros(len(df))
    right_dists = np.zeros(len(df))
    l_widths = np.zeros(len(df))
    r_widths = np.zeros(len(df))

    # Extract car points as numpy array for faster access
    car_points = df[[x_col, y_col]].values

    for i in tqdm(range(len(df)), desc="Processing car positions"):
        car_point = car_points[i]

        # Find two closest points on track edges
        left_dist_vals, left_idxs = left_tree.query(car_point, k=2)
        right_dist_vals, right_idxs = right_tree.query(car_point, k=2)

        # Get left edge points
        left_p1 = left_points[left_idxs[0]]
        left_p2 = left_points[left_idxs[1]]
        left_proj = projection_values(car_point, left_p1, left_p2)

        # Get right edge points
        right_p1 = right_points[right_idxs[0]]
        right_p2 = right_points[right_idxs[1]]
        right_proj = projection_values(car_point, right_p1, right_p2)

        # Use projection distance if on segment, otherwise closest point distance
        if 0 <= left_proj['c'] <= 1:
            left_dists[i] = left_proj['d']
            # Get width from closest point (first one)
            closest_left_idx = left_idxs[0]
        else:
            left_dists[i] = left_dist_vals[0]
            closest_left_idx = left_idxs[0]

        if 0 <= right_proj['c'] <= 1:
            right_dists[i] = right_proj['d']
            closest_right_idx = right_idxs[0]
        else:
            right_dists[i] = right_dist_vals[0]
            closest_right_idx = right_idxs[0]

        # Get track widths from the closest edge points
        l_widths[i] = track_left.iloc[closest_left_idx]['width']
        r_widths[i] = track_right.iloc[closest_right_idx]['width']

    # Assign all at once (faster than individual loc assignments)
    df["left_dist"] = left_dists
    df["right_dist"] = right_dists
    df["l_width"] = l_widths
    df["r_width"] = r_widths

    return df


def calculate_track_width(track_left, track_right):
    ''' For each point on track_left, calculate the width by projecting to the closest 2 points on track_right.
    Returns track_left and track_right DataFrames with added 'width' column.
    '''

    # Prepare points for KD tree
    left_points = np.vstack(
        [track_left["WORLDPOSX"], track_left["WORLDPOSY"]]).T
    right_points = np.vstack(
        [track_right["WORLDPOSX"], track_right["WORLDPOSY"]]).T

    left_tree = KDTree(left_points)
    right_tree = KDTree(right_points)

    # Calculate width for left edge points
    left_df = track_left.copy()
    left_widths = np.zeros(len(left_df))

    for i in tqdm(range(len(left_df)), desc="Calculating left edge widths"):
        left_point = left_points[i]

        # Find two closest points on right edge
        right_dist_vals, right_idxs = right_tree.query(left_point, k=2)

        # Get right edge points
        right_p1 = right_points[right_idxs[0]]
        right_p2 = right_points[right_idxs[1]]
        right_proj = projection_values(left_point, right_p1, right_p2)

        # Use projection distance if on segment, otherwise closest point distance
        if 0 <= right_proj['c'] <= 1:
            left_widths[i] = right_proj['d']
        else:
            left_widths[i] = right_dist_vals[0]

    left_df["width"] = left_widths

    # Calculate width for right edge points
    right_df = track_right.copy()
    right_widths = np.zeros(len(right_df))

    for i in tqdm(range(len(right_df)), desc="Calculating right edge widths"):
        right_point = right_points[i]

        # Find two closest points on left edge
        left_dist_vals, left_idxs = left_tree.query(right_point, k=2)

        # Get left edge points
        left_p1 = left_points[left_idxs[0]]
        left_p2 = left_points[left_idxs[1]]
        left_proj = projection_values(right_point, left_p1, left_p2)

        # Use projection distance if on segment, otherwise closest point distance
        if 0 <= left_proj['c'] <= 1:
            right_widths[i] = left_proj['d']
        else:
            right_widths[i] = left_dist_vals[0]

    right_df["width"] = right_widths

    return left_df, right_df


def compute_distace_to_apex(data: pd.DataFrame, turns: pd.DataFrame):
    turn_1, turn_2 = turns[turns['TURN'] == 1], turns[turns['TURN'] == 2]
    apex_turn_1 = (turn_1['APEX_X1'].values[0], turn_1['APEX_Y1'].values[0])
    apex_turn_2 = (turn_2['APEX_X1'].values[0], turn_2['APEX_Y1'].values[0])

    data['dist_apex_1'] = euclidean_distance(
        (data[x_col], data[y_col]),  # x1, y1
        (apex_turn_1[0], apex_turn_1[1])  # x2, y2
    )
    data['dist_apex_2'] = euclidean_distance(
        (data[x_col], data[y_col]),  # x1, y1
        (apex_turn_2[0], apex_turn_2[1])  # x2, y2
    )
    return data


def compute_angle_to_apex(data: pd.DataFrame, turns: pd.DataFrame):
    turn_1, turn_2 = turns[turns['TURN'] == 1], turns[turns['TURN'] == 2]
    apex_turn_1 = (turn_1['APEX_X1'].values[0], turn_1['APEX_Y1'].values[0])
    apex_turn_2 = (turn_2['APEX_X1'].values[0], turn_2['APEX_Y1'].values[0])

    df_new = data.copy()

    forward = df_new[['WORLDFORWARDDIRX',
                      'WORLDFORWARDDIRY']].to_numpy()
    pos = df_new[[x_col, y_col]].to_numpy()

    apex1_vec = np.array(apex_turn_1) - pos
    apex2_vec = np.array(apex_turn_2) - pos

    forward_rot = np.column_stack([-forward[:, 1], forward[:, 0]])
    df_new['angle_to_apex1'] = angle_between(forward_rot, apex1_vec)
    df_new['angle_to_apex2'] = angle_between(forward_rot, apex2_vec)

    return df_new


def id_outoftrack(df, buffer=0.53, start=350, end=600, inplace=True):
    if not inplace:
        df = df.copy()

    df_small = df[(df['LAPDISTANCE'] >= start) &
                  (df['LAPDISTANCE'] <= end)].copy()
    df_small["within_bounds"] = (df_small["left_dist"] + df_small["right_dist"]) < \
        (df_small["l_width"] + df_small["r_width"]) / 2 + buffer

    invalid_laps = df_small[df_small["within_bounds"] == False][[
        "SESSIONUID", "CURRENTLAPNUM"]].drop_duplicates()

    invalid_laps["invalid_lap"] = 1
    df = df.merge(
        invalid_laps,
        on=["SESSIONUID", "CURRENTLAPNUM"],
        how="left"
    )
    df["invalid_lap"] = df["invalid_lap"].fillna(0).astype(int)

    return df


def add_lap_id(df):
    df['lap_id'] = df.groupby(['SESSIONUID', 'CURRENTLAPNUM']).ngroup()
    return df


def car_from_ref_line(data, ref_line, x_col=x_col, y_col=y_col):
    ref_points = np.vstack([ref_line["WORLDPOSX"], ref_line["WORLDPOSY"]]).T
    tree = KDTree(ref_points)

    df = data.copy()
    proj_vals = np.zeros(len(df))
    car_points = df[[x_col, y_col]].values
    car_forward = df[["WORLDFORWARDDIRX", "WORLDFORWARDDIRY"]].values

    for i in tqdm(range(len(df)), desc="Referencing relative ref line"):
        car_point = car_points[i]
        dist_vals, idxs = tree.query(car_point, k=2)
        p0, p1 = ref_points[idxs[0]], ref_points[idxs[1]]

        # projection along the segment
        proj = projection_values(car_point, p0, p1)['d']

        # determine side: left (-) or right (+) relative to the segment
        vec_seg = p1 - p0
        vec_car = car_point - p0
        cross = vec_seg[0]*vec_car[1] - vec_seg[1] * \
            vec_car[0]  # 2D cross product

        proj_vals[i] = proj  # if cross >= 0 else -proj

    df["proj_from_ref"] = proj_vals
    return df
