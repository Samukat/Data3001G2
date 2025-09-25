import pandas as pd
import numpy as np


def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


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

    forward = df_new[['M_WORLDFORWARDDIRX_1',
                      'M_WORLDFORWARDDIRY_1']].to_numpy()
    pos = df_new[['M_WORLDPOSITIONX_1', 'M_WORLDPOSITIONY_1']].to_numpy()

    apex1_vec = np.array(apex_turn_1) - pos
    apex2_vec = np.array(apex_turn_2) - pos

    def angle_between(v1, v2):
        v1_u = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
        v2_u = v2 / np.linalg.norm(v2, axis=1, keepdims=True)

        dot = np.clip(np.sum(v1_u * v2_u, axis=1), -1.0, 1.0)
        angle = np.arccos(dot)

        cross = v1_u[:, 0]*v2_u[:, 1] - v1_u[:, 1]*v2_u[:, 0]
        angle[cross < 0] = -angle[cross < 0]

        return np.degrees(angle)

    forward_rot = np.column_stack([-forward[:, 1], forward[:, 0]])
    df_new['angle_to_apex1'] = angle_between(forward_rot, apex1_vec)
    df_new['angle_to_apex2'] = angle_between(forward_rot, apex2_vec)

    return df_new
