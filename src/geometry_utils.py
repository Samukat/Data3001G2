
import pandas as pd
import numpy as np
from scipy.spatial import KDTree


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


def angle_between(v1, v2):
    v1_u = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
    v2_u = v2 / np.linalg.norm(v2, axis=1, keepdims=True)

    dot = np.clip(np.sum(v1_u * v2_u, axis=1), -1.0, 1.0)
    angle = np.arccos(dot)

    cross = v1_u[:, 0]*v2_u[:, 1] - v1_u[:, 1]*v2_u[:, 0]
    angle[cross < 0] = -angle[cross < 0]

    return np.degrees(angle)


def interpolate_time_atime(df, target):
    results = []
    grouped = df.groupby(["SESSIONUID", "CURRENTLAPNUM"])

    for (session, lap), L in grouped:
        L = L.sort_values("LAPDISTANCE").copy()

        # find the two points surrounding the "finish line"
        before = L[L["LAPDISTANCE"] <= target].tail(1)
        after = L[L["LAPDISTANCE"] >= target].head(1)

        if before["LAPDISTANCE"].values[0] == after["LAPDISTANCE"].values[0]:
            time = before["CURRENTLAPTIMEINMS"].values[0]
        else:
            p0 = np.array([before["WORLDPOSITIONX"].values[0],
                           before["WORLDPOSITIONY"].values[0],
                           before["WORLDPOSITIONZ"].values[0]])
            p1 = np.array([after["WORLDPOSITIONX"].values[0],
                           after["WORLDPOSITIONY"].values[0],
                           after["WORLDPOSITIONZ"].values[0]])

            # ratio of how far along the segment the target lies (by LAPDISTANCE)
            d0, d1 = before["LAPDISTANCE"].values[0], after["LAPDISTANCE"].values[0]
            ratio = (target - d0) / (d1 - d0)

            # interpolate time using the ratio
            t0, t1 = before["CURRENTLAPTIMEINMS"].values[0], after["CURRENTLAPTIMEINMS"].values[0]
            time = t0 + ratio * (t1 - t0)

        results.append({
            "SESSIONUID": session,
            "CURRENTLAPNUM": lap,
            "target_distance": target,
            "interpolated_time_ms": time
        })

    return pd.DataFrame(results)
