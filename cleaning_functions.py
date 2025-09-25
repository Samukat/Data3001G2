import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm


def remove_other_tracks(data: pd.DataFrame) -> pd.DataFrame:
    """
    Removes all rows from the dataframe where the TRACKID is not 1.
    """
    return data[data['TRACKID'] == 0].reset_index(drop=True)


def remove_na(data: pd.DataFrame, subset) -> pd.DataFrame:
    return data.dropna(subset=subset).reset_index(drop=True)


def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def two_closest_points(point, query_points: pd.DataFrame, x_col="WORLDPOSX", y_col="WORLDPOSY"):
    points = np.vstack([query_points[x_col], query_points[y_col]]).T
    tree = KDTree(points)
    dists, idxs = tree.query(point, k=2)
    return dists, query_points.iloc[idxs]


def projection_values(pointA: list, pointB1: list, pointB2: list):
    AB1 = [pointA[0] - pointB1[0], pointA[1] - pointB1[1]]
    B1B2 = [pointB2[0] - pointB1[0], pointB2[1] - pointB1[1]]

    # Check for division by zero
    distance_sq = euclidean_distance(pointB1, pointB2)**2
    if distance_sq < 1e-12:  # Points are essentially the same
        return {'d': euclidean_distance(pointA, pointB1), 'projA': pointB1, 'c': 0}

    # Projection scalar
    c = (AB1[0]*B1B2[0] + AB1[1]*B1B2[1]) / distance_sq
    projA = [pointB1[0] + c*B1B2[0], pointB1[1] + c*B1B2[1]]

    # Distance from A to its projection
    d = euclidean_distance(pointA, projA)

    return {'d': d, 'projA': projA, 'c': c}


def car_edge_distances(data, track_left, track_right, x_col="M_WORLDPOSITIONX_1", y_col="M_WORLDPOSITIONY_1"):
    ''' For each point in data, find the closest points on the left and right track edges.
    by:
    1. finding the two closest points on each
    2. projecting the point onto the line segment defined by those two points
    Returns a DataFrame with the same number of rows as input, including closest left/right edge points and distances.
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
        else:
            left_dists[i] = left_dist_vals[0]

        if 0 <= right_proj['c'] <= 1:
            right_dists[i] = right_proj['d']
        else:
            right_dists[i] = right_dist_vals[0]

    # Assign all at once (faster than individual loc assignments)
    df["left_dist"] = left_dists
    df["right_dist"] = right_dists

    return df


def car_edge_distances(data, track_left, track_right, x_col="M_WORLDPOSITIONX_1", y_col="M_WORLDPOSITIONY_1"):
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
