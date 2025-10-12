import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

from .geometry_utils import interpolate_at_distance
from .config import FEATURES


def moment_generator(data, moment_name, feature_names, moment_LD, extrema_LD = None, epsilon=0.1):
    ### from the point row data -> transform into lap row data
    ### 1. for each element of the moment_LD (moment lap distance where the index is the lap_id) we want to contruct a row that looks like
    ### Lap_id, moment_name_feature1, moment_name_feature2 .... for all the features in feature names
    ### the data is formated is a pd df with each row conaining a point and the columns including the features, lapdistance, lap_id etc..
    ### the row that you are adding to our new dataframe should be at at a lap_distance from the moment_LD dataframe (a dataframe where the index is the lap_id, the column is the lap distance)

    results = []
    # print(md)

    for lap_id, target_LD in moment_LD.items():
        # print(len(moment_LD))
        # print(lap_id)
        # Extract this lap's data
        lap_data = data[data["lap_id"] == lap_id]
        if lap_data.empty or pd.isna(target_LD):
            continue

        # Find row closest to target LAPDISTANCE less then epsilon
        distances = (lap_data["LAPDISTANCE"] - target_LD).abs()
        min_distance = distances.min()

        entry = {"lap_id": lap_id}
        
        if min_distance <= epsilon:
            idx = distances.idxmin()
            row = lap_data.loc[idx]

            for feat in feature_names:
                entry[f"{moment_name}_{feat}"] = row.get(feat, np.nan)
        
        else:
            ### THESE ARE THE ROWS THAT NEED INTERPOLATING AT THAT DISTANCE
            interpolated = interpolate_at_distance(lap_data, target_LD, features=feature_names)
            
            if not interpolated.empty:
                for feat in feature_names:
                    entry[f"{moment_name}_{feat}"] = interpolated[f"interpolated_{feat}"].values[0]
            else:
                for feat in feature_names:
                    entry[f"{moment_name}_{feat}"] = np.nan
        
        

        # For extrema
        if extrema_LD is not None and lap_id in extrema_LD.index:
            extrema_target = extrema_LD.loc[lap_id]
            if not pd.isna(extrema_target):
                idx_extrema = (lap_data["LAPDISTANCE"] - extrema_target).abs().idxmin()
                extrema_row = lap_data.loc[idx_extrema]
                entry[f"{moment_name}_ext_LAPDISTANCE"] = extrema_row.get("LAPDISTANCE", np.nan)
                entry[f"{moment_name}_ext_TIMETOINMS"] = extrema_row.get("CURRENTLAPTIMEINMS") - row.get("CURRENTLAPTIMEINMS")

        results.append(entry)

    # Return as DataFrame
    # print(pd.DataFrame(results))
    return pd.DataFrame(results).set_index("lap_id")


def get_throttle_points(df, distance_range=(10, 710), lift_threshold=0.02):
    """
    Find when the driver first lifts off the throttle, the minimum throttle afterwards,
    when they start getting back on the throttle, and when throttle reaches its max again.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'lap_id', 'LAPDISTANCE', and 'THROTTLE' columns.
    distance_range : tuple
        (min_distance, max_distance) range to consider for each lap.
    lift_threshold : float
        Minimum change in throttle considered as a lift or reapply event.

    Returns
    -------
    pd.DataFrame
        lap_id, first_lift_LD, min_throttle_LD, back_on_LD, back_on_max_LD for each lap.
    """
    results = []

    for lap_id, group in df.groupby("lap_id"):
        group = group.sort_values("LAPDISTANCE").reset_index(drop=True)
        group = group[(group["LAPDISTANCE"] >= distance_range[0]) &
                      (group["LAPDISTANCE"] <= distance_range[1])]

        if group["THROTTLE"].isna().all():
            results.append({
                'lap_id': lap_id,
                'first_lift_LD': None,
                'min_throttle_LD': None,
                'back_on_LD': None,
                'back_on_max_LD': None
            })
            continue

        # Smooth throttle to reduce noise
        throttle = group["THROTTLE"] #.rolling(3, center=True, min_periods=1).mean()
        d_throttle = throttle.diff()

        # Find first lift
        lift_points = d_throttle[d_throttle < -lift_threshold]
        if lift_points.empty:
            results.append({
                'lap_id': lap_id,
                'first_lift_LD': None,
                'min_throttle_LD': None,
                'back_on_LD': None,
                'back_on_max_LD': None
            })
            continue

        lift_idx = lift_points.index[0]
        first_lift_LD = group.loc[lift_idx, "LAPDISTANCE"]

        # Minimum throttle after lift 
        after_lift = group.loc[lift_idx:]
        min_idx = after_lift["THROTTLE"].idxmin()
        min_throttle_LD = group.loc[min_idx, "LAPDISTANCE"]

        # First "back on" point
        after_min = group.loc[min_idx:]
        d_throttle_after_min = after_min["THROTTLE"].diff() #.rolling(3, center=True, min_periods=1).mean()
        back_on_points = d_throttle_after_min[d_throttle_after_min > lift_threshold]

        if not back_on_points.empty:
            back_on_idx = back_on_points.index[0]
            back_on_LD = group.loc[back_on_idx, "LAPDISTANCE"]
        else:
            back_on_LD = None

        # Max throttle after back on
        if back_on_LD is not None:
            after_back_on = group.loc[group["LAPDISTANCE"] >= back_on_LD]
            max_idx = after_back_on["THROTTLE"].idxmax()
            back_on_max_LD = group.loc[max_idx, "LAPDISTANCE"]
        else:
            back_on_max_LD = None

        results.append({
            'lap_id': lap_id,
            'first_lift_LD': first_lift_LD,
            'min_throttle_LD': min_throttle_LD,
            'back_on_LD': back_on_LD,
            'back_on_max_LD': back_on_max_LD
        })

    return pd.DataFrame(results)


def get_braking_points(df, distance_range = (10,800)):
    results =[]

    for lap_id, group in df.groupby("lap_id"):
        group = group.sort_values("LAPDISTANCE").reset_index(drop=True)
        group = group[(group["LAPDISTANCE"] >= distance_range[0]) & (group["LAPDISTANCE"] <= distance_range[1])]

        if group["BRAKE"].isna().all():
            results.append({
            'lap_id': lap_id,
            'max_brake_LD': None,
            'BP_LD': None,
            'brake_decrease_LD': None,
            'brake_end_LD': None
            })
            continue

        max_idx = group["BRAKE"].idxmax()

        if pd.isna(max_idx) or max_idx not in group.index:
            results.append({
            'lap_id': lap_id,
            'max_brake_LD': None,
            'BP_LD': None,
            'brake_decrease_LD': None,
            'brake_end_LD': None
            })
            continue
        
        max_brake_LD = group.loc[max_idx, "LAPDISTANCE"]
            
        before_max = group.loc[:group.index.get_loc(max_idx)]
        zero_brake = before_max[before_max["BRAKE"] == 0]

        if not zero_brake.empty:
            bp_ld = zero_brake["LAPDISTANCE"].iloc[-1]
        elif not before_max.empty:
            bp_ld = before_max["LAPDISTANCE"].iloc[-1]
        else:
            bp_ld = None

        after_max = group.iloc[group.index > group.index.get_loc(max_idx)]

        brake_decrease_LD = None
        brake_end_LD = None

        if not after_max.empty:
            brake_values  =after_max["BRAKE"].values
            diffs = pd.Series(brake_values).diff().fillna(0)
            dec_indices = after_max.index[diffs < 0]

            if len(dec_indices) > 0:
                first_dec_idx = dec_indices[0]
                brake_decrease_LD = group.loc[first_dec_idx, "LAPDISTANCE"]

                after_decrease = group.loc[first_dec_idx:]
                zeros = after_decrease[after_decrease["BRAKE"] == 0]

                if not zeros.empty:
                    brake_end_LD = zeros["LAPDISTANCE"].iloc[0]
                else:
                    min_idx = after_decrease["BRAKE"].idxmin()
                    brake_end_LD = group.loc[min_idx, "LAPDISTANCE"]

        results.append({
            'lap_id': lap_id,
            'BP_LD': bp_ld,
            'max_brake_LD': max_brake_LD,
            'brake_decrease_LD': brake_decrease_LD,
            'brake_end_LD': brake_end_LD
        })

    return pd.DataFrame(results)


def get_apex_points(data, apex_columns = ["dist_apex_1", "dist_apex_2"]):
    """
    Find the LAPDISTANCE where each apex distance column is minimized for each lap.
    """
    results = []
    
    for lap_id, group in data.groupby("lap_id"):
        group = group.set_index("LAPDISTANCE")
        
        entry = {"lap_id": lap_id}
        
        for apex_col in apex_columns:
            min_lapdistance = group[apex_col].idxmin()
            # Extract apex number or use full column name
            apex_name = apex_col.replace("dist_", "").replace("apex_", "apex")
            entry[f"{apex_name}_LD"] = min_lapdistance

        
        results.append(entry)
    
    return pd.DataFrame(results).set_index("lap_id")


def get_steering_points(df, distance_range = (10,800)):
    results=[]

    for lap_id, group in df.groupby("lap_id"):
        group = group.sort_values("LAPDISTANCE").reset_index(drop=True)
        group = group[(group["LAPDISTANCE"] >= distance_range[0]) & (group["LAPDISTANCE"] <= distance_range[1])]


        if group["STEER"].isna().all():
            results.append({
            'lap_id': lap_id,
            'first_steer_LD': None,
            'max_pos_angle_steer_LD': None,
            'middle_TP_LD': None,
            'max_neg_LD': None,
            'end_steer_LD': None
            })
            continue

        max_pos_idx = group["STEER"].idxmax()
        max_neg_idx = group["STEER"].idxmin()

        max_pos_angle_LD = group.loc[max_pos_idx, "LAPDISTANCE"]
        max_neg_angle_LD = group.loc[max_neg_idx, "LAPDISTANCE"]

        before_max_pos = group.loc[:group.index.get_loc(max_pos_idx)]
        zero_steer = before_max_pos[np.abs(before_max_pos["STEER"]) <= 0.01 ]

        if not zero_steer.empty:
            first_steer_LD = zero_steer["LAPDISTANCE"].iloc[-1]
        else:
            first_steer_LD = None

        after_max_pos = group.iloc[group.index > group.index.get_loc(max_pos_idx)]
        after_max_neg = group.iloc[group.index > group.index.get_loc(max_neg_idx)]

        middle_steering = None
        steer_end_LD = None

        if not after_max_pos.empty:    
            steer_series = after_max_pos["STEER"]
            sign_change = steer_series.shift(1) * steer_series < 0

            if sign_change.any():
                change_idx = sign_change.idxmax()
                idx_before = steer_series.index.get_loc(change_idx) - 1

                if idx_before >= 0:
                    idx_A = steer_series.index[idx_before]
                    idx_B = change_idx
                    steer_A = group.loc[idx_A, "STEER"]
                    steer_B = group.loc[idx_B, "STEER"]
                    LD_A = group.loc[idx_A, "LAPDISTANCE"]
                    LD_B = group.loc[idx_B, "LAPDISTANCE"]

                    middle_steering = LD_A + (LD_B - LD_A) * (steer_A) / (steer_A - steer_B)

        if not after_max_neg.empty:
            steer_series = after_max_neg["STEER"]
            sign_change = steer_series.shift(1) * steer_series < 0

            if sign_change.any():
                change_idx = sign_change.idxmax()
                idx_before = steer_series.index.get_loc(change_idx) - 1

                if idx_before >= 0:
                    idx_A = steer_series.index[idx_before]
                    idx_B = change_idx
                    steer_A = group.loc[idx_A, "STEER"]
                    steer_B = group.loc[idx_B, "STEER"]
                    LD_A = group.loc[idx_A, "LAPDISTANCE"]
                    LD_B = group.loc[idx_B, "LAPDISTANCE"]

                    steer_end_LD = LD_A + (LD_B - LD_A) * (steer_A) / (steer_A - steer_B)

        results.append({
        'lap_id': lap_id,
        'first_steer_LD': first_steer_LD,
        'max_pos_angle_steer_LD': max_pos_angle_LD,
        'middle_TP_LD': middle_steering,
        'max_neg_LD': max_neg_angle_LD,
        'end_steer_LD': steer_end_LD
        })

    return pd.DataFrame(results)



