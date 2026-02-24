import cv2 # OpenCV
import mediapipe as mp
import numpy as np # NumPy
import pandas as pd # Pandas: used to build and query REBA scoring tables (DataFrame)
import time # FPS
import csv  # CSV
import os
from numpy.linalg import norm # NumPy
import sys 
import traceback 
import math 
# ==== /CSV ====
RUN_TS = time.strftime("%Y%m%d-%H%M%S")
OUTPUT_DIR = r"C:\Users\azcon\OneDrive\Escritorio\SEXTO SEMESTRE\CAPSTONE\CAPSTONE-PROJECT"
BASE_NAME = f"{RUN_TS}_1-1"

VIDEO_FPS = 15.0
VIDEO_CODEC = 'mp4v'
VIDEO_OUT_PATH = os.path.join(OUTPUT_DIR, f"{BASE_NAME}.mp4")
CSV_OUT_PATH   = os.path.join(OUTPUT_DIR, f"{BASE_NAME}.csv")

video_writer = None
csv_fh = None
csv_writer = None
recording = False
path_mp4 = r"C:\Users\azcon\OneDrive\Escritorio\SEXTO SEMESTRE\CAPSTONE\CAPSTONE-PROJECT/1002_6-5.mp4"
# index
LEFT_SIDE_CAMERA_INDEX = 1  # Camera for LEFT side view
RIGHT_SIDE_CAMERA_INDEX = 2 # Camera for RIGHT side view
FRONT_CAMERA_INDEX = 0      # Camera for Front view

VIDEO_FPS = 15.0                     # FPS 30.0
VIDEO_CODEC = 'mp4v'                 # 'mp4v' -> .mp4, 'XVID' -> .avi
# VIDEO_OUT_PATH = time.strftime(path_mp4)
video_writer = None                  # combined1

# Mediapipe Hands
mp_hands = mp.solutions.hands
hands_left_side = mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1, min_detection_confidence=0.5)
hands_right_side = mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1, min_detection_confidence=0.5)
hands_front = mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=1, min_detection_confidence=0.5)

# Mediapipe Pose
mp_pose = mp.solutions.pose
# Initialize separate Pose instances for potentially different views/tracking
pose_left_side = mp_pose.Pose(min_detection_confidence=0.4, min_tracking_confidence=0.4)
pose_right_side = mp_pose.Pose(min_detection_confidence=0.4, min_tracking_confidence=0.4)
pose_front = mp_pose.Pose(min_detection_confidence=0.4, min_tracking_confidence=0.4)

# !! IMPORTANT: Adjust Abduction Threshold if using Shoulder Angle !!
# ARM_ABDUCTION_THRESHOLD = 25 #
# NECK_TWIST_Z_THRESHOLD = 0.05 # Z landmark z
# TRUNK_TWIST_Z_THRESHOLD = 0.08 # Z landmark z

""" Degree based posture thresholds: upper arm abduction angle,
neck flexion/extension angle and trunk flexion/extension angle;
values above each threshold are treated as non-neutral and may increase the ergonomic
risk score"""

# ADDED CODE: GROUP A (Trunk, Neck, Legs)
TRUNK_FLEXION_BINS = (5,20,60)
TRUNK_EXTENSION_THRESHOLD = 5
TRUNK_SIDE_BEND_THRESHOLD = 10
TRUNK_TWIST_Z_THRESHOLD = 0.08

NECK_FLEXION_THRESHOLD = 20
NECK_EXTENSION_THRESHOLD = 5
NECK_SIDE_BEND_THRESHOLD = 10
NECK_TWIST_Z_THRESHOLD = 0.05

KNEE_FLEXION_THRESHOLD_1 = 30
KNEE_FLEXION_THRESHOLD_2 = 60
LEGS_ASYMMETRY_THRESHOLD = 0.10

# ADDED CODE: GROUP B (Upper Arm, Lower Arm, Wrist)
UPPER_ARM_FLEXION_BINS = (20,45,90)
ARM_ABDUCTION_THRESHOLD = 25
SHOULDER_RAISE_THRESHOLD = 15

LOWER_ARM_NEUTRAL_MIN = 60
LOWER_ARM_NEUTRAL_MAX = 100

WRIST_FLEXION_THRESHOLD = 15
WRIST_DEVIATION_THRESHOLD = 10
WRIST_TWIST_THRESHOLD= 10

# ADDED CODE: LOAD AND FORCE
LOAD_WEIGHT_THRESHOLDS_KG = (5,10)
FORCE_SUDDEN_BONUS = 1

# ADDED CODE: COUPLING
COUPLING_SCORE = {
    "good": 0,
    "fair": 1,
    "poor": 2,
    "unacceptable": 3
}
"""add one if the posture is held for more than one minute, 
    repeated or unstable"""
STATIC_HOLD_SECONDS = 60 
REPETITION_PER_MIN_THRESHOLD = 4
UNSTABLE_POSTURE_BONUS= 1


# ADDED CODE: REBA ACTION LEVEL
ACTION_LEVEL_DESCRIPTIONS = {
    (1, 1): "Negligible risk, no action required.",
    (2, 3): "Low risk; change may be needed.",
    (4, 7): "Medium risk; further investigation, change soon.",
    (8, 10): "High risk; investigate and implement change.",
    (11, float('inf')): "Very high risk; implement change now." 
}


def calculate_angle_with_sign(a, b, c):  # b
    """Calculates signed angle in 2D space."""
    a = np.array(a[:2]); b = np.array(b[:2]); c = np.array(c[:2]) # x, y
    ba = a - b; bc = c - b
    if norm(ba) < 1e-6 or norm(bc) < 1e-6: return 0.0
    angle_rad = np.arctan2(bc[1], bc[0]) - np.arctan2(ba[1], ba[0])
    angle_deg = np.degrees(angle_rad) 
    angle_deg = (angle_deg + 180) % 360 - 180
    return angle_deg
def calculate_angle_acos(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b; bc = c - b
    if norm(ba) < 1e-6 or norm(bc) < 1e-6: return 0.0 
    # cos() = (ba bc) / (ba*bc)
    # np.clip -1.0 1.0 arccos
    cos_theta = np.clip(np.dot(ba, bc) / (norm(ba) * norm(bc)), -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    return angle_deg
# 3. elbow-wrist-middle finger MCP
def compute_wrist_angle_2dxy(elbow, wrist, middle, width, height):
    # normalized pixel
    a = np.array([elbow.x * width, elbow.y * height])
    b = np.array([wrist.x * width, wrist.y * height])
    c = np.array([middle.x * width, middle.y * height])
    # v1 = elbow wristv2 = middle wrist
    v1 = a - b
    v2 = c - b
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 180.0
    cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))
    return angle
def draw_text_with_background(img, text, org, font, scale, text_color, bg_color, thickness=1, padding=2):

    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    cv2.rectangle(img,
                  (x - padding, y - text_h - padding),
                  (x + text_w + padding, y + baseline + padding),
                  bg_color, -1)
    cv2.putText(img, text, (x, y), font, scale, text_color, thickness)

'''Returns the REBA action level description based on the final score.'''

# ------ ESTA FUNCION LA DEBO BORRAR -------
def get_rula_action_level(final_score):
    if final_score is None or not isinstance(final_score, (int, float)):
        return "Score Error or N/A"
    for level_range, description in ACTION_LEVEL_DESCRIPTIONS.items():
        if level_range[0] <= final_score <= level_range[1]:
            return f"Level {level_range[0]}-{level_range[1]}: {description}"
    # Handle case where score might be valid but > max range key (e.g. score 8+)
    if final_score == 7:
         return f"Level 7: Investigate and implement changes ."
    elif final_score >= 7:
        return "Score out of typical range"
    
def get_reba_action_level(final_score):
    """Return REBA action level description from final REBA score (1..15)."""
    if final_score is None:
        return "Score Error or N/A"

    try:
        s = int(round(float(final_score)))
    except (TypeError, ValueError):
        return "Score Error or N/A"

    if s < 1:
        return "Score out of range (<1)"
    if s == 1:
        return "Level 1: Negligible risk, no action required."
    if 2 <= s <= 3:
        return "Level 2: Low risk, change may be needed."
    if 4 <= s <= 7:
        return "Level 3: Medium risk, investigate and implement change soon."
    if 8 <= s <= 10:
        return "Level 4: High risk, investigate and implement change."
    if 11 <= s <= 15:
        return "Level 5: Very high risk, implement change now."

    return "Score out of range (>15)"

# 3REBA
# REBAA/B/C
# REBA A// (Neck / Trunk / Legs)

table_a_data = {
    "Trunk": [1, 2, 3, 4, 5],
    "N1_L1": [1, 2, 2, 3, 4],
    "N1_L2": [2, 3, 4, 5, 6],
    "N1_L3": [3, 4, 5, 6, 7],
    "N1_L4": [4, 5, 6, 7, 8],
    "N2_L1": [1, 3, 4, 5, 6],
    "N2_L2": [2, 4, 5, 6, 7],
    "N2_L3": [3, 5, 6, 7, 8],
    "N2_L4": [4, 6, 7, 8, 9],
    "N3_L1": [3, 4, 5, 6, 7],
    "N3_L2": [3, 5, 6, 7, 8],
    "N3_L3": [5, 6, 7, 8, 9],
    "N3_L4": [6, 7, 8, 9, 9],
}
table_a = pd.DataFrame(table_a_data)

def validate_table_a_schema(df: pd.DataFrame) -> None:
    """Raise Value Error if REBA Table A schema is invalid."""
    required_cols= {"Trunk"}
    for n in range(1,4):
        for l in range(1,5):
            required_cols.add(f"N{n}_L{l}")
    
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Reba Table A missing columns: {sorted(missing)}")
    
    if df.empty:
        raise ValueError("Reba Table A is empty")
    
def _to_valid_int_score(name: str, value, lo:int, hi: int) -> int:
    """ Convert input to bounded int score with strict validation """
    if value is None:
        raise ValueError(f"{name} is None")

    try:
        x = float(value)
    except (TypeError, ValueError) as e:
        raise TypeError (f"{name} must be numeric, got {type(value).__name__}") from e
    
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got{x}")
    
    return int (max(lo,min(hi,round(x))))

def get_table_a_score(neck_score, trunk_score,legs_score, table_df = None):
    """
    Robust REBA Table A lookup
    Inputs: 
            neck_score: expected to be 1...3
            trunk_score: expected to be 1...5
            legs_score: expected to be 1...4
            
    Ouput:
            int score or None on Failure
    """

    df = table_a if table_df is None else table_df
    
    try:
        validate_table_a_schema(df)

        neck = _to_valid_int_score("neck_score", neck_score,1,3)
        trunk= _to_valid_int_score("trunk_score", trunk_score, 1, 5)
        legs = _to_valid_int_score("legs_score", legs_score,1,4)
    
        col = f"N{neck}_L{legs}"
        row = df.loc[df["Trunk"]== trunk]

        if row.empty:
            raise ValueError(f"No row found for TRUNK = {trunk}")
        
        value = row.iloc[0][col]
        
        if pd.isna(value):
            raise ValueError(f"Cell value is NaN for Trunk={trunk}, Col={col}")

        value_int = int(value)
        return value_int

    except (TypeError, ValueError, KeyError, IndexError) as e:
        print(f"Err REBA Table A: {e}")
        return None
    except Exception as e:
        print(f"Err REBA Table A (unexpected): {e}")
        return None

# REBA B// (Upper Arm / Lower Arm / Wrist)
table_b_data = {
    'UpperArm': [1, 2, 3, 4, 5, 6],
    'L1_W1': [1, 1, 3, 4, 6, 7], 
    'L1_W2': [2, 2, 4, 5, 7, 8],
    'L1_W3': [2, 3, 5, 5, 8, 8], 

    'L2_W1': [1, 2, 4, 5, 7, 8],
    'L2_W2': [2, 3, 5, 6, 8, 9], 
    'L2_W3': [3, 4, 5, 7, 8, 9],
}
table_b = pd.DataFrame(table_b_data)

def validate_table_b_schema(df: pd.DataFrame)-> None:
    required_cols= {"UpperArm"}
    for la in (1,2):
        for ws in (1,2,3):
            required_cols.add(f"L{la}_W{ws}")
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"REBA Table B missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("REBA Table B is empty")

def get_table_b_score(upper_arm_score, lower_arm_score, wrist_score, table_df = None):
    """REBA Table B lookup
    Inputs: 
        upper_arm_score: expected 1..6
        lower_arm_score: expected 1..2
        wrist_score: expected 1..3
    Output:
        int score or None on failure 
    """
    df = table_b if table_df is None else table_df

    try:
        validate_table_b_schema(df)

        ua = _to_valid_int_score("upper_arm_score",upper_arm_score,1,6)
        la = _to_valid_int_score("lower_arm_score",lower_arm_score,1,2)
        ws= _to_valid_int_score("wrist_score",wrist_score, 1,3)

        row = df.loc[df["UpperArm"]== ua]
        if row.empty:
            raise ValueError(f"No row found for UpperArm = {ua}")
        
        col = f"L{la}_W{ws}"
        value = row.iloc[0][col]

        if pd.isna(value):
            raise ValueError(f"Cell value is NaN for UpperArm={ua}, Col={col}") 
    
        return int(value)
    except  (TypeError,ValueError,KeyError,IndexError) as e:
        print (f"Err REBA table B: {e}")
        return None
    except Exception as e:
        print(f"Err REBA Table B ( unexpected ): {e}")
        return None



# REBA C (Final Combination Table)
table_c_data = {
    "ScoreA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "B1":  [1, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12],
    "B2":  [1, 2, 3, 4, 4, 6, 7, 8, 9, 10, 11, 12],
    "B3":  [1, 2, 3, 4, 4, 6, 7, 8, 9, 10, 11, 12],
    "B4":  [2, 3, 3, 4, 5, 7, 8, 9, 10, 11, 11, 12],
    "B5":  [3, 4, 4, 5, 6, 8, 9, 10, 10, 11, 12, 12],
    "B6":  [3, 4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 12],
    "B7":  [4, 5, 6, 7, 8, 9, 9, 10, 11, 11, 12, 12],
    "B8":  [5, 6, 7, 8, 8, 9, 10, 10, 11, 12, 12, 12],
    "B9":  [6, 6, 7, 8, 9, 10, 10, 10, 11, 12, 12, 12],
    "B10": [7, 7, 8, 9, 9, 10, 11, 11, 12, 12, 12, 12],
    "B11": [7, 7, 8, 9, 9, 10, 11, 11, 12, 12, 12, 12],
    "B12": [7, 8, 8, 9, 9, 10, 11, 11, 12, 12, 12, 12],
}
table_c = pd.DataFrame(table_c_data)

def validate_table_c_schema(df: pd.DataFrame) -> None:
    """
    Raise ValueError if REBA Table C schema is invalid.
    Required columns: ScoreA, B1..B12
    """
    required_cols = {"ScoreA"} | {f"B{i}" for i in range(1, 13)}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"REBA Table C missing columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("REBA Table C is empty")


def get_table_c_score(score_a, score_b, table_df=None):
    """
    Robust REBA Table C lookup.
    Inputs:
        score_a: expected 1..12
        score_b: expected 1..12
    Output:
        int score or None on failure
    """
    df = table_c if table_df is None else table_df

    try:
        validate_table_c_schema(df)

        sa = _to_valid_int_score("score_a", score_a, 1, 12)
        sb = _to_valid_int_score("score_b", score_b, 1, 12)

        row = df.loc[df["ScoreA"] == sa]
        if row.empty:
            raise ValueError(f"No row found for ScoreA={sa}")

        col = f"B{sb}"
        value = row.iloc[0][col]

        if pd.isna(value):
            raise ValueError(f"Cell value is NaN for ScoreA={sa}, Col={col}")

        return int(value)

    except (TypeError, ValueError, KeyError, IndexError) as e:
        print(f"Err REBA Table C: {e}")
        return None
    except Exception as e:
        print(f"Err REBA Table C (unexpected): {e}")
        return None


# 4  RULA <-------------> PUNTO A CONTINUAR
def get_rula_component_scores(
    side, # 'left' or 'right'
    upper_arm_angle, lower_arm_angle, wrist_angle, neck_angle, trunk_angle_side_vertical,
    adj_flags = {} 
):
    
    upper_arm_score = 1; lower_arm_score = 1;wrist_score = 1; wrist_twist_score = 1; neck_score = 1; trunk_score = 1; leg_score = 1
    if side == 'left':
        # 1. Left Upper Arm Score
        ua_ref = -upper_arm_angle
        if -20 <= ua_ref <= 20: upper_arm_score = 1
        elif ua_ref < -20: upper_arm_score = 2 # Extension
        elif 20 < ua_ref <= 45: upper_arm_score = 2
        elif 45 < ua_ref <= 90: upper_arm_score = 3
        elif 90 < ua_ref : upper_arm_score = 4
    # else: upper_arm_score = 4 # > 90
        if adj_flags.get('is_left_arm_abducted', False): upper_arm_score += 1
        if adj_flags.get('is_left_shoulder_raised', False): upper_arm_score += 1

        # 2. Left Lower Arm Score
        la_ref = -lower_arm_angle # Apply negation as per user logic
        if 80 <= la_ref <= 120: lower_arm_score = 1
        else: lower_arm_score = 2
        if adj_flags.get('is_left_lower_arm_across_midline', False) or adj_flags.get('is_left_lower_arm_abducted', False): lower_arm_score += 1

        # 3. Left Wrist Score
        wr_ref = abs(180 - (wrist_angle))
        if wr_ref <= 10: wrist_score = 1
        elif 10 < wr_ref <= 20: wrist_score = 2
        elif 20 < wr_ref : wrist_score = 3
        # Apply LEFT Wrist Adjustment
        if adj_flags.get('is_left_wrist_bent_from_midline', False): wrist_score += 1
        # 4. Left Wrist Twist Score
        wrist_twist_score = 1 # Default =1
    # if wrist_angle < 140 : wrist_twist_score = 2
    # if adj_flags.get('is_left_wrist_twist_so_over', False): wrist_twist_score = 2

        # 5. Neck Score
        nk_ref = -neck_angle
# if 160 - nk_ref < 0: neck_score = 4 # Extension threshold (verify)
# if -160 - nk_ref < -15: neck_score = 4
        if  -15 <= 160 - nk_ref <= 5: neck_score = 1
        elif 5 < 160 - nk_ref <= 15: neck_score = 2
        elif 160 - nk_ref > 15 and 160 - nk_ref < 200: neck_score = 3
        elif 160 - nk_ref > 15 and 160 - nk_ref > 200: neck_score = 4
# else: neck_score = 1 if nk_ref >= -4 else 4
        if adj_flags.get('is_neck_side_bent', False): neck_score += 1
        if adj_flags.get('is_neck_twisted', False): neck_score += 1

        # 6. Trunk Score
        tr_ref = -trunk_angle_side_vertical
        if -5 <= 180-abs(tr_ref) <= 5 : trunk_score = 1 # Upright
        elif 5 < 180-abs(tr_ref) <= 20 : trunk_score = 2 # Flexion 0-20 or Extension
        elif 20 < 180-abs(tr_ref) <= 60: trunk_score = 3 # Flexion 20-60
        elif 180-abs(tr_ref) > 60: trunk_score = 4 # Flexion > 60
        if adj_flags.get('is_trunk_side_bent', False): trunk_score += 1
        if adj_flags.get('is_trunk_twisted', False): trunk_score += 1

        # 7. Leg Score
        leg_score = 1 # Default

    elif side == 'right':
        # 1. Right Upper Arm Score
        ua_ref = upper_arm_angle 
        if -20 <= ua_ref <= 20: upper_arm_score = 1
        elif ua_ref < -20: upper_arm_score = 2 # Extension
        elif 20 < ua_ref <= 45: upper_arm_score = 2
        elif 45 < ua_ref <= 90: upper_arm_score = 3
        elif 90 < ua_ref : upper_arm_score = 4
        # else: upper_arm_score = 4 # > 90
        if adj_flags.get('is_right_arm_abducted', False): upper_arm_score += 1
        if adj_flags.get('is_right_shoulder_raised', False): upper_arm_score += 1

        # 2. Right Lower Arm Score
        la_ref = lower_arm_angle  
        if 80 <= la_ref <= 120: lower_arm_score = 1 
        else: lower_arm_score = 2
        # Apply RIGHT Lower Arm Adjustment
        if adj_flags.get('is_right_lower_arm_across_midline', False)or adj_flags.get('is_right_lower_arm_abducted', False): lower_arm_score += 1

        # 3. Right Wrist Score
        wr_ref = abs(180 - (wrist_angle))
        if wr_ref <= 10: wrist_score = 1
        elif 10 < wr_ref <= 20: wrist_score = 2
        elif 20 < wr_ref : wrist_score = 3
        # Apply RIGHT Wrist Adjustment
        if adj_flags.get('is_right_wrist_bent_from_midline', False): wrist_score += 1
        # 4. Right Wrist Twist Score
        wrist_twist_score = 1 # Default
        if wrist_angle < 140 : wrist_twist_score = 2
# if adj_flags.get('is_right_wrist_twist_so_over', False): wrist_twist_score = 2

        # 5. Neck Score (Common logic, no negation assumed for right baseline)
        nk_ref = neck_angle
        # Original script's logic:
# if 160 - nk_ref < 0: neck_score = 4 # Extension threshold (verify)
# if -160 - nk_ref < -15: neck_score = 4
        if -15 <= 160 - nk_ref <= 5: neck_score = 1
        elif 5 < 160 - nk_ref <= 15: neck_score = 2
        elif 160 - nk_ref > 15 and 160 - nk_ref < 200: neck_score = 3
        elif 160 - nk_ref > 15 and 160 - nk_ref > 200: neck_score = 4
# else: neck_score = 1 if nk_ref >= -4 else 4
        # Apply Common Neck Adjustments
        if adj_flags.get('is_neck_side_bent', False): neck_score += 1
        if adj_flags.get('is_neck_twisted', False): neck_score += 1

        # 6. Trunk Score (Common logic, no negation assumed for right baseline)
        tr_ref = trunk_angle_side_vertical # Angle relative to vertical
        # Original script's logic:
        if -5 <= 180-abs(tr_ref) <= 5 : trunk_score = 1 # Upright
        elif 5 < 180-abs(tr_ref) <= 20 : trunk_score = 2 # Flexion 0-20 or Extension
        elif 20 < 180-abs(tr_ref) <= 60: trunk_score = 3 # Flexion 20-60
        elif 180-abs(tr_ref) > 60: trunk_score = 4 # Flexion > 60
        # Apply Common Trunk Adjustments
        if adj_flags.get('is_trunk_side_bent', False): trunk_score += 1
        if adj_flags.get('is_trunk_twisted', False): trunk_score += 1

        # 7. Leg Score
        leg_score = 1 # Default

    else:
        print(f"Error: Invalid side '{side}' passed to get_rula_component_scores")
        # Return default scores or raise an error
        return (1, 1, 1, 1, 1, 1, 1)

    # RULA
    upper_arm_score = max(1, min(upper_arm_score, 6))
    lower_arm_score = max(1, min(lower_arm_score, 3))
    wrist_score = max(1, min(wrist_score, 4))
    wrist_twist_score = max(1, min(wrist_twist_score, 2))
    neck_score = max(1, min(neck_score, 6))
    trunk_score = max(1, min(trunk_score, 6))
    leg_score = max(1, min(leg_score, 2))
    # print(f"[{side.upper()} Comp Scores] UA:{upper_arm_score}, LA:{lower_arm_score}, WR:{wrist_score}, WRT:{wrist_twist_score}, NK:{neck_score}, TR:{trunk_score}, LG:{leg_score}")
    return (upper_arm_score, lower_arm_score, wrist_score, wrist_twist_score, neck_score, trunk_score, leg_score)
print("Initializing cameras...")
cap_left_side = cv2.VideoCapture(LEFT_SIDE_CAMERA_INDEX, cv2.CAP_DSHOW)
cap_right_side = cv2.VideoCapture(RIGHT_SIDE_CAMERA_INDEX, cv2.CAP_DSHOW)
cap_front = cv2.VideoCapture(FRONT_CAMERA_INDEX, cv2.CAP_DSHOW)
caps = {"Left Side": cap_left_side, "Right Side": cap_right_side, "Front": cap_front}
for name, cap in caps.items():
    idx = -1
    if name == "Left Side": idx = LEFT_SIDE_CAMERA_INDEX
    elif name == "Right Side": idx = RIGHT_SIDE_CAMERA_INDEX
    elif name == "Front": idx = FRONT_CAMERA_INDEX
    if not cap.isOpened():
        print(f"Error: Cannot open camera {name} (Index: {idx})")
        for c in caps.values():
            if c.isOpened(): c.release()
        sys.exit()
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
print("Starting RULA analysis for Left and Right sides...")

'''
# Dictionaries to store results for each side
left_results = {"angles": {}, "scores": {}, "table_scores": {"A": "N/A", "B": "N/A", "C": "N/A"}, "valid": False}
right_results = {"angles": {}, "scores": {}, "table_scores": {"A": "N/A", "B": "N/A", "C": "N/A"}, "valid": False}
front_adjustments = {} # Stores adjustments calculated from front view
'''

frame_count = 0
while True:
    
    
    # Read frames
    ret_left, frame_left = cap_left_side.read()
    ret_right, frame_right = cap_right_side.read()
    ret_front, frame_front = cap_front.read()
    # FPS FPS
    prev_time_left, prev_time_right, prev_time_front = time.time(), time.time(), time.time()
    fps_left, fps_right, fps_front = 0, 0, 0
    
    l_low_abd_diff = 0.0
    r_low_abd_diff = 0.0
    dist_l_ear_sh = 0.0
    dist_r_ear_sh  = 0.0

    if not ret_left or not ret_right or not ret_front:
        # print("Error reading frame from one or more cameras. Skipping.") # Less verbose
        time.sleep(0.05)
        continue

    try:
        h_left, w_left, _ = frame_left.shape
        h_right, w_right, _ = frame_right.shape
        h_front, w_front, _ = frame_front.shape

        left_results = {"angles": {}, "scores": {}, "table_scores": {"A": "N/A", "B": "N/A", "C": "N/A"}, "valid": False}
        right_results = {"angles": {}, "scores": {}, "table_scores": {"A": "N/A", "B": "N/A", "C": "N/A"}, "valid": False}
        front_adjustments = { # Reset all flags
            'is_left_lower_arm_across_midline': False, 'is_right_lower_arm_across_midline': False,
            'is_left_lower_arm_abducted': False, 'is_right_lower_arm_abducted': False,
            'is_trunk_side_bending': False, 'is_neck_side_bending': False,
            'is_left_arm_abducted': False, 'is_right_arm_abducted': False,
            'is_neck_twisted': False, 'is_trunk_twisted': False, 
            'is_left_shoulder_raised': False, 'is_right_shoulder_raised': False,
            'is_left_wrist_bent_from_midline': False, 'is_right_wrist_bent_from_midline': False,
            'is_left_wrist_twist_so_over': False, 'is_right_wrist_twist_so_over': False,
        }
        angle_left_shoulder_deg = 0.0
        angle_right_shoulder_deg = 0.0
        angle_diff_trunk_deg = 0.0
        angle_neck_vert_deg = 0.0
        neck_z_diff = 0.0
        trunk_z_diff = 0.0

        frame_left_rgb = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB) # MediaPipe RGB
        frame_left_rgb.flags.writeable = False # MediaPipe
        results_ls = pose_left_side.process(frame_left_rgb)
        results_lh = hands_left_side.process(frame_left_rgb)
        hand_lm_l = None
        if results_lh.multi_hand_landmarks and results_lh.multi_handedness:
            for lm, handed in zip(results_lh.multi_hand_landmarks, results_lh.multi_handedness):
                label = handed.classification[0].label
                if label == 'Left':
                    hand_lm_l = lm
                    break

        frame_left_rgb.flags.writeable = True
        frame_left_out = frame_left

        if results_ls.pose_landmarks and results_lh.multi_hand_landmarks:
            landmarks_ls = results_ls.pose_landmarks.landmark
            hand_lm_l = results_lh.multi_hand_landmarks[0]
            try:
                l_sh_lm = landmarks_ls[mp_pose.PoseLandmark.LEFT_SHOULDER]
                l_el_lm = landmarks_ls[mp_pose.PoseLandmark.LEFT_ELBOW]
                l_wr_lm = landmarks_ls[mp_pose.PoseLandmark.LEFT_WRIST]
                l_hip_lm = landmarks_ls[mp_pose.PoseLandmark.LEFT_HIP]
                l_ear_lm = landmarks_ls[mp_pose.PoseLandmark.LEFT_EAR]
                l_ind_lm = landmarks_ls[mp_pose.PoseLandmark.LEFT_INDEX]

                required_lms = [l_sh_lm, l_el_lm, l_wr_lm, l_hip_lm, l_ear_lm, l_ind_lm]
                if all(lm.visibility > 0.6 for lm in required_lms):  # MediaPipe
                    left_results["valid"] = True
                    l_sh_pt = (int(l_sh_lm.x * w_left), int(l_sh_lm.y * h_left))
                    l_el_pt = (int(l_el_lm.x * w_left), int(l_el_lm.y * h_left))
                    l_wr_pt = (int(l_wr_lm.x * w_left), int(l_wr_lm.y * h_left))
                    l_hip_pt = (int(l_hip_lm.x * w_left), int(l_hip_lm.y * h_left))
                    l_ear_pt = (int(l_ear_lm.x * w_left), int(l_ear_lm.y * h_left))
                    l_ind_pt = (int(l_ind_lm.x * w_left), int(l_ind_lm.y * h_left))

                    # (calculate_angle_with_sign=arctan2,)
                    left_results["angles"]['upper_arm'] = calculate_angle_with_sign(l_hip_pt, l_sh_pt, l_el_pt)
                    left_results["angles"]['lower_arm'] = calculate_angle_with_sign(l_sh_pt, l_el_pt, l_wr_pt)
                    left_results["angles"]['neck'] = calculate_angle_with_sign(l_ear_pt, l_sh_pt, l_hip_pt) # Verify definition
                    hip_vert_down = (l_hip_pt[0], h_left)
                    left_results["angles"]['trunk_new'] = calculate_angle_with_sign(l_sh_pt, l_hip_pt, hip_vert_down)
                    if hand_lm_l is not None:
                        angle_l = compute_wrist_angle_2dxy(
                            elbow=l_el_lm,
                            wrist=l_wr_lm,
                            middle=hand_lm_l.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
                            width=w_left,
                            height=h_left
                        )
                        left_results["angles"]['wrist'] = angle_l

                        mid_lm = hand_lm_l.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
                        mid_pt = (int(mid_lm.x * w_left), int(mid_lm.y * h_left))

                        cv2.circle(frame_left_out, mid_pt, 4, (0, 255, 0), -1)
                        cv2.line(frame_left_out, mid_pt, l_wr_pt, (255, 0, 0), 1)
                    
                    cv2.circle(frame_left_out, l_sh_pt, 3, (0, 255, 0), -1)
                    cv2.circle(frame_left_out, l_el_pt, 3, (0, 255, 0), -1)
                    cv2.circle(frame_left_out, l_wr_pt, 3, (0, 255, 0), -1)
                    cv2.circle(frame_left_out, l_ear_pt, 3, (0, 255, 0), -1)
                    cv2.circle(frame_left_out, l_hip_pt, 3, (0, 255, 0), -1)
                    # cv2.line(frame_left_out, l_wr_pt, mid_pt, (255, 100, 0), 1)
                    cv2.line(frame_left_out, l_sh_pt, l_el_pt, (255, 0, 0), 1)
                    cv2.line(frame_left_out, l_el_pt, l_wr_pt, (255, 0, 0), 1)
                    cv2.line(frame_left_out, l_sh_pt, l_hip_pt, (0, 0, 255), 1)
                    cv2.line(frame_left_out, l_hip_pt, hip_vert_down, (0, 0, 255), 1)
                    cv2.line(frame_left_out, l_sh_pt, l_ear_pt, (0, 0, 255), 1)

            except IndexError: # Handle case where a landmark index might not exist
                left_results["valid"] = False; print(f"Warn: Left Side Landmark index out of bounds.")
            except Exception as e: left_results["valid"] = False; print(f"Err Left Side LMs: {e}")

        frame_right_rgb = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
        frame_right_rgb.flags.writeable = False
        results_rs = pose_right_side.process(frame_right_rgb)
        results_rh = hands_right_side.process(frame_right_rgb)
        hand_lm_r = None
        if results_rh.multi_hand_landmarks and results_rh.multi_handedness:
            for lm, handed in zip(results_rh.multi_hand_landmarks, results_rh.multi_handedness):
                label = handed.classification[0].label
                if label == 'Right':
                    hand_lm_r = lm
                    break
        frame_right_rgb.flags.writeable = True
        frame_right_out = frame_right

        if results_rs.pose_landmarks and results_rh.multi_hand_landmarks:
            landmarks_rs = results_rs.pose_landmarks.landmark
            hand_lm_r = results_rh.multi_hand_landmarks[0]
            try:
                r_sh_lm = landmarks_rs[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                r_el_lm = landmarks_rs[mp_pose.PoseLandmark.RIGHT_ELBOW]
                r_wr_lm = landmarks_rs[mp_pose.PoseLandmark.RIGHT_WRIST]
                r_hip_lm = landmarks_rs[mp_pose.PoseLandmark.RIGHT_HIP]
                r_ear_lm = landmarks_rs[mp_pose.PoseLandmark.RIGHT_EAR]
                r_ind_lm = landmarks_rs[mp_pose.PoseLandmark.RIGHT_INDEX]
                # r_mid_lm = hand_lm_r[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]

                required_rms = [r_sh_lm, r_el_lm, r_wr_lm, r_hip_lm, r_ear_lm, r_ind_lm]
                if all(lm.visibility > 0.6 for lm in required_rms):
                    right_results["valid"] = True
                    r_sh_pt = (int(r_sh_lm.x * w_right), int(r_sh_lm.y * h_right))
                    r_el_pt = (int(r_el_lm.x * w_right), int(r_el_lm.y * h_right))
                    r_wr_pt = (int(r_wr_lm.x * w_right), int(r_wr_lm.y * h_right))
                    r_hip_pt = (int(r_hip_lm.x * w_right), int(r_hip_lm.y * h_right))
                    r_ear_pt = (int(r_ear_lm.x * w_right), int(r_ear_lm.y * h_right))
                    r_ind_pt = (int(r_ind_lm.x * w_right), int(r_ind_lm.y * h_right))
                    # r_mid_pt=(int(r_mid_lm.x * w_right), int(r_mid_lm.y * h_right))
                    
                    # (calculate_angle_with_sign=arctan2,)
                    right_results["angles"]['upper_arm'] = calculate_angle_with_sign(r_hip_pt, r_sh_pt, r_el_pt)
                    right_results["angles"]['lower_arm'] = calculate_angle_with_sign(r_sh_pt, r_el_pt, r_wr_pt)
                    right_results["angles"]['neck'] = calculate_angle_with_sign(r_ear_pt, r_sh_pt, r_hip_pt) # Verify definition
                    hip_vert_down_r = (r_hip_pt[0], h_right)
                    right_results["angles"]['trunk_new'] = calculate_angle_with_sign(r_sh_pt, r_hip_pt, hip_vert_down_r)
                    if hand_lm_r is not None:
                        angle_r = compute_wrist_angle_2dxy(
                            elbow=r_el_lm,
                            wrist=r_wr_lm,
                            middle=hand_lm_r.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
                            width=w_right,
                            height=h_right
                        )
                        right_results["angles"]["wrist"] = angle_r
                        mid_lm = hand_lm_r.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
                        mid_pt = (int(mid_lm.x * w_left), int(mid_lm.y * h_left))

                        cv2.circle(frame_right_out, mid_pt, 4, (0, 255, 0), -1)
                        cv2.line(frame_right_out, mid_pt, r_wr_pt, (255, 0, 0), 1)


                    cv2.circle(frame_right_out, r_sh_pt, 3, (0, 255, 0), -1)
                    cv2.circle(frame_right_out, r_el_pt, 3, (0, 255, 0), -1)
                    cv2.circle(frame_right_out, r_wr_pt, 3, (0, 255, 0), -1)
                    cv2.circle(frame_right_out, r_hip_pt, 3, (0, 255, 0), -1)
                    cv2.circle(frame_right_out, r_ear_pt, 3, (0, 255, 0), -1)
                    # cv2.circle(frame_right_out, r_mid_pt, 3, (0, 255, 0), -1)
                    cv2.line(frame_right_out, r_sh_pt, r_el_pt, (255, 0, 0), 1)
                    cv2.line(frame_right_out, r_el_pt, r_wr_pt, (255, 0, 0), 1)
                    cv2.line(frame_right_out, r_sh_pt, r_hip_pt, (0, 0, 255), 1)
                    cv2.line(frame_right_out, r_hip_pt, hip_vert_down_r, (0, 0, 255), 1)
                    cv2.line(frame_right_out, r_sh_pt, r_ear_pt, (0, 0, 255), 1)

            except IndexError: # Handle case where a landmark index might not exist
                right_results["valid"] = False; print(f"Warn: Right Side Landmark index out of bounds.")
            except Exception as e: right_results["valid"] = False; print(f"Err Right Side LMs: {e}")
        frame_front_rgb = cv2.cvtColor(frame_front, cv2.COLOR_BGR2RGB)
        frame_front_rgb.flags.writeable = False

        results_front = pose_front.process(frame_front_rgb)
        hands_results_front = hands_front.process(frame_front_rgb)

        frame_front_rgb.flags.writeable = True
        frame_front_out = frame_front

        if results_front.pose_landmarks:
            landmarks_f = results_front.pose_landmarks.landmark
            try:
                # Extract all needed landmarks for adjustments
                l_sh_lm_f = landmarks_f[mp_pose.PoseLandmark.LEFT_SHOULDER]
                r_sh_lm_f = landmarks_f[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                l_hip_lm_f = landmarks_f[mp_pose.PoseLandmark.LEFT_HIP]
                r_hip_lm_f = landmarks_f[mp_pose.PoseLandmark.RIGHT_HIP]
                l_wr_lm_f = landmarks_f[mp_pose.PoseLandmark.LEFT_WRIST]
                r_wr_lm_f = landmarks_f[mp_pose.PoseLandmark.RIGHT_WRIST]
                l_ear_lm_f = landmarks_f[mp_pose.PoseLandmark.LEFT_EAR]
                r_ear_lm_f = landmarks_f[mp_pose.PoseLandmark.RIGHT_EAR]
                l_el_lm_f = landmarks_f[mp_pose.PoseLandmark.LEFT_ELBOW]
                r_el_lm_f = landmarks_f[mp_pose.PoseLandmark.RIGHT_ELBOW]
                required_f_lms = [l_sh_lm_f, r_sh_lm_f, l_hip_lm_f, r_hip_lm_f, l_wr_lm_f, r_wr_lm_f, l_ear_lm_f, r_ear_lm_f, l_el_lm_f, r_el_lm_f]

                if all(lm.visibility > 0.6 for lm in required_f_lms):
                    # Calculate front points (X, Y)
                    l_sh_f = np.array([l_sh_lm_f.x*w_front, l_sh_lm_f.y*h_front])
                    r_sh_f = np.array([r_sh_lm_f.x*w_front, r_sh_lm_f.y*h_front])
                    l_hip_f = np.array([l_hip_lm_f.x*w_front, l_hip_lm_f.y*h_front])
                    r_hip_f = np.array([r_hip_lm_f.x*w_front, r_hip_lm_f.y*h_front])
                    l_wr_f = np.array([l_wr_lm_f.x*w_front, l_wr_lm_f.y*h_front])
                    r_wr_f = np.array([r_wr_lm_f.x*w_front, r_wr_lm_f.y*h_front])
                    l_el_f = np.array([l_el_lm_f.x*w_front, l_el_lm_f.y*h_front])
                    r_el_f = np.array([r_el_lm_f.x*w_front, r_el_lm_f.y*h_front])
                    l_ear_f_xy = np.array([l_ear_lm_f.x * w_front, l_ear_lm_f.y * h_front]) # Renamed for clarity
                    r_ear_f_xy = np.array([r_ear_lm_f.x * w_front, r_ear_lm_f.y * h_front]) # Renamed for clarity
                    sh_mid_f = (l_sh_f + r_sh_f) / 2
                    hip_mid_f = (l_hip_f + r_hip_f) / 2
                    ear_mid_f = (l_ear_f_xy + r_ear_f_xy) / 2
                    l_low_abd_diff = abs(l_wr_f[0] - l_sh_f[0])  # pixel
                    r_low_abd_diff = abs(r_wr_f[0] - r_sh_f[0])

                    # --- Calculate Adjustment Flags ---
                    # wrist
                    WRIST_OUTSIDE_THRESHOLD = 45
                    # wrist
                    if l_wr_f[0] > l_sh_f[0] and l_low_abd_diff > WRIST_OUTSIDE_THRESHOLD: # x
                        front_adjustments['is_left_lower_arm_abducted'] = True 
                    # wrist
                    if r_wr_f[0] < r_sh_f[0] and r_low_abd_diff > WRIST_OUTSIDE_THRESHOLD:
                        front_adjustments['is_right_lower_arm_abducted'] = True

                # Midline Cross (Left)
                    # (sh_mid_f) (hip_mid_f)
                    # Ax+By+C=0 ABC
                    # (x,y) Ax+By+C 0
                    if abs(sh_mid_f[0] - hip_mid_f[0]) < 1e-6: # Vertical
                        if (l_sh_f[0] < sh_mid_f[0] and l_wr_f[0] > sh_mid_f[0]) or \
                           (l_sh_f[0] > sh_mid_f[0] and l_wr_f[0] < sh_mid_f[0]):
                            front_adjustments['is_left_lower_arm_across_midline'] = True
                    else: # Non-vertical
                        A=hip_mid_f[1]-sh_mid_f[1]; B=sh_mid_f[0]-hip_mid_f[0]; C=-A*sh_mid_f[0]-B*sh_mid_f[1]
                        if (A*l_sh_f[0]+B*l_sh_f[1]+C) * (A*l_wr_f[0]+B*l_wr_f[1]+C) < 0:
                            front_adjustments['is_left_lower_arm_across_midline'] = True
                    # Midline Cross (Right)
                    if abs(sh_mid_f[0] - hip_mid_f[0]) < 1e-6: # Vertical
                         if (r_sh_f[0] > sh_mid_f[0] and r_wr_f[0] < sh_mid_f[0]) or \
                            (r_sh_f[0] < sh_mid_f[0] and r_wr_f[0] > sh_mid_f[0]):
                             front_adjustments['is_right_lower_arm_across_midline'] = True
                    else: # Non-vertical
                        A=hip_mid_f[1]-sh_mid_f[1]; B=sh_mid_f[0]-hip_mid_f[0]; C=-A*sh_mid_f[0]-B*sh_mid_f[1]
                        if (A*r_sh_f[0]+B*r_sh_f[1]+C) * (A*r_wr_f[0]+B*r_wr_f[1]+C) < 0:
                            front_adjustments['is_right_lower_arm_across_midline'] = True

                    # (Left Shoulder Angle: Elbow-Shoulder-Hip):
                    try:
                        v_se_l=l_el_f-l_sh_f; v_sh_l=l_hip_f-l_sh_f
                        n_se_l=norm(v_se_l); n_sh_l=norm(v_sh_l)
                        if n_se_l > 1e-6 and n_sh_l > 1e-6:
                            cos_l = np.clip(np.dot(v_se_l,v_sh_l)/(n_se_l*n_sh_l),-1.0,1.0)
                            angle_left_shoulder_deg = np.degrees(np.arccos(cos_l))
                            if angle_left_shoulder_deg > ARM_ABDUCTION_THRESHOLD: front_adjustments['is_left_arm_abducted'] = True
                    except Exception as e: print(f"Warn: L Sh Angle Calc Err: {e}")

                    # (Right Shoulder Angle: Elbow-Shoulder-Hip)
                    try:
                        v_se_r=r_el_f-r_sh_f; v_sh_r=r_hip_f-r_sh_f
                        n_se_r=norm(v_se_r); n_sh_r=norm(v_sh_r)
                        if n_se_r > 1e-6 and n_sh_r > 1e-6:
                            cos_r = np.clip(np.dot(v_se_r,v_sh_r)/(n_se_r*n_sh_r),-1.0,1.0)
                            angle_right_shoulder_deg = np.degrees(np.arccos(cos_r))
                            if angle_right_shoulder_deg > ARM_ABDUCTION_THRESHOLD: front_adjustments['is_right_arm_abducted'] = True
                    except Exception as e: print(f"Warn: R Sh Angle Calc Err: {e}")

                # Trunk Bend
                    # (v_sh)(v_hip)
                    # arctan2

                    v_sh=r_sh_f-l_sh_f; v_hip=r_hip_f-l_hip_f
                    n_sh=norm(v_sh); n_hip=norm(v_hip)
                    if n_sh > 1e-6 and n_hip > 1e-6:
                        a_sh=np.arctan2(v_sh[1],v_sh[0]); a_hip=np.arctan2(v_hip[1],v_hip[0])
                        a_diff = (a_sh - a_hip + np.pi) % (2*np.pi) - np.pi
                        angle_diff_trunk_deg = np.degrees(a_diff)
                        if abs(angle_diff_trunk_deg) > 15: front_adjustments['is_trunk_side_bending'] = True

                # Neck Bend
                    # (ear_mid_f) v_nk
                    # v_vert_up OpenCV Y (0, -1)
                    # (Dot Product)

                    v_nk = ear_mid_f - sh_mid_f; n_nk=norm(v_nk)
                    if n_nk > 1e-6: 
                        v_vert_up=np.array([0,-1]) # Vertical Up vector
                        cos_nk = np.clip(np.dot(v_nk,v_vert_up)/n_nk, -1.0,1.0)
                        angle_neck_vert_deg = np.degrees(np.arccos(cos_nk)) # Angle wrt vertical up = 0
                        if angle_neck_vert_deg > 15: front_adjustments['is_neck_side_bending'] = True # Deviation > 15 deg from vertical
                    # ( Y )
                    if not front_adjustments.get('is_neck_side_bending', False):
                        SHRUG_THRESHOLD = 40  # Y
                        dist_l_ear_sh = abs(l_ear_f_xy[1] - l_sh_f[1])
                        dist_r_ear_sh = abs(r_ear_f_xy[1] - r_sh_f[1])
                        if dist_l_ear_sh < SHRUG_THRESHOLD:
                            front_adjustments['is_left_shoulder_raised'] = True
                        if dist_r_ear_sh < SHRUG_THRESHOLD:
                            front_adjustments['is_right_shoulder_raised'] = True

                    # ( Z )
                    neck_z_diff = abs(l_ear_lm_f.z - r_ear_lm_f.z)
                    if neck_z_diff > NECK_TWIST_Z_THRESHOLD:
                        front_adjustments['is_neck_twisted'] = True

                    # ( Z )
                    trunk_z_diff = abs(l_sh_lm_f.z - r_sh_lm_f.z)
                    if trunk_z_diff > TRUNK_TWIST_Z_THRESHOLD:
                        front_adjustments['is_trunk_twisted'] = True

                    # Draw on front frame
                    cv2.line(frame_front_out, tuple(sh_mid_f.astype(int)), tuple(hip_mid_f.astype(int)), (255, 255, 0), 1)
                    cv2.line(frame_front_out, tuple(l_sh_f.astype(int)), tuple(r_sh_f.astype(int)), (0, 255, 255), 1)
                    cv2.line(frame_front_out, tuple(l_hip_f.astype(int)), tuple(r_hip_f.astype(int)), (0, 255, 255), 1)
                    cv2.line(frame_front_out, tuple(sh_mid_f.astype(int)), tuple(ear_mid_f.astype(int)), (255, 0, 255), 1)
                    cv2.line(frame_front_out, tuple(l_sh_f.astype(int)), tuple(l_el_f.astype(int)), (255,100,0), 1) # Left arm
                    cv2.line(frame_front_out, tuple(r_sh_f.astype(int)), tuple(r_el_f.astype(int)), (0,100,255), 1) # Right arm (Orange)

            except IndexError: print(f"Warn: Front Landmark index out of bounds.")
            except Exception as e: print(f"Err Front LMs/Adjust: {e}")

# 5.RULA
        # Left Side
        if left_results["valid"]:
            # print(f"DEBUG: Attempting Left RULA Calc. Angles: {left_results['angles']}") # Uncomment for deep debug
            try:
                # --- CORRECTED FUNCTION CALL ---
                angles_l = left_results["angles"]
                comps = get_rula_component_scores(
                    'left',
                    angles_l.get('upper_arm', 0),
                    angles_l.get('lower_arm', 0),
                    angles_l.get('wrist', 0),
                    angles_l.get('neck', 0),
                    angles_l.get('trunk_new', 0),
                    adj_flags=front_adjustments
                )
                # --- End Corrected Call ---
                left_results["scores"] = dict(zip(['UA', 'LA', 'WR', 'WR-T', 'NK', 'TR', 'LG'], comps))
                sa = get_table_a_score(comps[4], comps[5], comps[6])
                sb = get_table_b_score(comps[0], comps[1], comps[2])
                sc = get_table_c_score(sa,sb)
                left_results["table_scores"]["A"] = sa if sa is not None else "Err"
                left_results["table_scores"]["B"] = sb if sb is not None else "Err"
                if sa is not None and sb is not None:
                    sc = get_table_c_score(sa, sb)
                    left_results["table_scores"]["C"] = sc if sc is not None else "Err"
                else: left_results["table_scores"]["C"] = "Err"
            except Exception as e: print(f"Error Calc Left RULA: {e}"); traceback.print_exc() # Print traceback for detail
        # Right Side
        if right_results["valid"]:
            # print(f"DEBUG: Attempting Right RULA Calc. Angles: {right_results['angles']}") # Uncomment for deep debug
            try:
                 # --- CORRECTED FUNCTION CALL ---
                angles_r = right_results["angles"]
                comps = get_rula_component_scores(
                    'right',
                    angles_r.get('upper_arm', 0),
                    angles_r.get('lower_arm', 0),
                    angles_r.get('wrist', 0),
                    angles_r.get('neck', 0),
                    angles_r.get('trunk_new', 0),
                    adj_flags=front_adjustments
                )
                 # --- End Corrected Call ---
                right_results["scores"] = dict(zip(['UA', 'LA', 'WR', 'WR-T', 'NK', 'TR', 'LG'], comps))
                sa = get_table_a_score(comps[4], comps[5], comps[6])
                sb = get_table_b_score(comps[0], comps[1], comps[2])
                sc = get_table_c_score(sa,sb)
                right_results["table_scores"]["A"] = sa if sa is not None else "Err"
                right_results["table_scores"]["B"] = sb if sb is not None else "Err"
                if sa is not None and sb is not None:
                    sc = get_table_c_score(sa, sb)
                    right_results["table_scores"]["C"] = sc if sc is not None else "Err"
                else: right_results["table_scores"]["C"] = "Err"
            except Exception as e: print(f"Error Calc Right RULA: {e}"); traceback.print_exc() # Print traceback for detail
        # RULA
        final_rula_score = "N/A"
        dominant_side = "N/A"
        left_c_score = left_results["table_scores"]["C"]
        right_c_score = right_results["table_scores"]["C"]

        # --- CORRECTED CHECK for numerical types (including NumPy types) ---
        score_l_is_num = np.issubdtype(type(left_c_score), np.number)
        score_r_is_num = np.issubdtype(type(right_c_score), np.number)

        # --- End Corrected Check ---
        # Convert scores to numbers for comparison, treat non-numeric as -1
        score_l_num = left_c_score if score_l_is_num else -1
        score_r_num = right_c_score if score_r_is_num else -1

       # ( left_c_score right_c_score )
        if not score_l_is_num and not score_r_is_num: # Check if BOTH failed the number check
            final_rula_score = "N/A" # If both sides non-numeric, final is N/A
            dominant_side = "Both Invalid"
        elif score_l_num >= score_r_num: # Compare using the numerical values (-1 for non-numbers)
            final_rula_score = left_c_score # Use original value (could be "Err", "N/A", or number)
            dominant_side = "Left" if score_l_is_num else "Right (L Invalid)" # Indicate based on validityQ
        else: # score_r_num > score_l_num
            final_rula_score = right_c_score # Use original value
            dominant_side = "Right" if score_r_is_num else "Left (R Invalid)" # Indicate based on validity
# final_action_level = get_rula_action_level(final_rula_score)
        csv_path = "C:\Users\azcon\OneDrive\Escritorio\SEXTO SEMESTRE\CAPSTONE\CAPSTONE-PROJECT/csv1002_1-3.csv"
       # 7. CSV
        row_data = {
            "Frame": frame_count,
            # Left
            "L_UA_Score": left_results["scores"].get("UA", "N/A"),
            "L_LA_Score": left_results["scores"].get("LA", "N/A"),
            "L_WR_Score": left_results["scores"].get("WR", "N/A"),
            "L_WRT_Score": left_results["scores"].get("WR-T", "N/A"),
            "L_NK_Score": left_results["scores"].get("NK", "N/A"),
            "L_TR_Score": left_results["scores"].get("TR", "N/A"),
            "L_A": left_results["table_scores"].get("A", "N/A"),
            "L_B": left_results["table_scores"].get("B", "N/A"),
            "L_C": left_results["table_scores"].get("C", "N/A"),
            # Right R_A/R_B/R_C left_results
            "R_UA_Score": right_results["scores"].get("UA", "N/A"),
            "R_LA_Score": right_results["scores"].get("LA", "N/A"),
            "R_WR_Score": right_results["scores"].get("WR", "N/A"),
            "R_WRT_Score": right_results["scores"].get("WR-T", "N/A"),
            "R_NK_Score": right_results["scores"].get("NK", "N/A"),
            "R_TR_Score": right_results["scores"].get("TR", "N/A"),
            "R_A": right_results["table_scores"].get("A", "N/A"),
            "R_B": right_results["table_scores"].get("B", "N/A"),
            "R_C": right_results["table_scores"].get("C", "N/A"),
            # Final
            "Final_RULA": final_rula_score,
            "Dominant_Side": dominant_side,
        }

        # if "N/A" not in row_data.values() and csv_writer is not None:
        # csv_writer.writerow(row_data)
        # csv_fh.flush()


        # with open(csv_path, mode="a", newline="") as f:
        # writer = csv.DictWriter(f, fieldnames=row_data.keys())
        # writer.writerow(row_data)

        ls_sc = left_results["table_scores"]
        cv2.putText(frame_left_out, f'L RULA: {ls_sc["C"]} (A:{ls_sc["A"]}, B:{ls_sc["B"]})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- START: Added Display for Left Angles and Component Scores ---
        start_x_detail = int(w_left * 0.65) # Adjust X position as needed
        start_y_detail = 60          # Start Y below existing scores
        line_height = 20             # Vertical space between lines
        font_scale_detail = 0.45
        color_detail = (0, 255, 255)   # Blue color for details

        draw_text_with_background(frame_left_out, "--- Left Angles ---", (start_x_detail, start_y_detail), cv2.FONT_HERSHEY_SIMPLEX, font_scale_detail, color_detail, 1); start_y_detail += line_height
        angle_keys_l = ['upper_arm', 'lower_arm', 'wrist', 'neck', 'trunk_new']
        angle_labels_l = ["L UA Ang", "L LA Ang", "L WR Ang", "L NK Ang", "L TR Ang"]
        for i, key in enumerate(angle_keys_l):
            value = left_results["angles"].get(key, 'N/A')
            text = f"{angle_labels_l[i]}: {value:.1f}" if isinstance(value, (int, float)) else f"{angle_labels_l[i]}: {value}"
            draw_text_with_background(frame_left_out, text, (start_x_detail, start_y_detail + i * line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale_detail, color_detail, 1)
        start_y_detail += len(angle_keys_l) * line_height + 5 # Add space

        draw_text_with_background(frame_left_out, "--- Left Scores ---", (start_x_detail, start_y_detail), cv2.FONT_HERSHEY_SIMPLEX, font_scale_detail, color_detail, 1); start_y_detail += line_height
        score_keys_l = ['UA', 'LA', 'WR', 'WR-T', 'NK', 'TR', 'LG']
        score_labels_l = ["L UA Sc", "L LA Sc", "L WR Sc", "L WRT Sc", "L NK Sc", "L TR Sc", "L LG Sc"]
        for i, key_abbr in enumerate(score_keys_l):
              # Assuming left_results["scores"] uses keys like 'UA', 'LA' etc. directly
            value = left_results["scores"].get(key_abbr, 'N/A')
            text = f"{score_labels_l[i]}: {value}"
            draw_text_with_background(frame_left_out, text, (start_x_detail, start_y_detail + i * line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale_detail, color_detail, 1)
        # --- END: Added Display ---
        rs_sc = right_results["table_scores"]
        cv2.putText(frame_right_out, f'R RULA: {rs_sc["C"]} (A:{rs_sc["A"]}, B:{rs_sc["B"]})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # --- START: Added Display for Right Angles and Component Scores ---
        start_x_detail_r = int(w_right * 0.05) # Adjust X position as needed
        start_y_detail_r = 60          # Start Y below existing scores
        # Use the same line_height, font_scale_detail, color_detail from above

        draw_text_with_background(frame_right_out, "--- Right Angles ---", (start_x_detail_r, start_y_detail_r), cv2.FONT_HERSHEY_SIMPLEX, font_scale_detail, color_detail, 1); start_y_detail_r += line_height
        angle_keys_r = ['upper_arm', 'lower_arm', 'wrist', 'neck', 'trunk_new']
        angle_labels_r = ["R UA Ang", "R LA Ang", "R WR Ang", "R NK Ang", "R TR Ang"]
        for i, key in enumerate(angle_keys_r):
            value = right_results["angles"].get(key, 'N/A')
            text = f"{angle_labels_r[i]}: {value:.1f}" if isinstance(value, (int, float)) else f"{angle_labels_r[i]}: {value}"
            draw_text_with_background(frame_right_out, text, (start_x_detail_r, start_y_detail_r + i * line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale_detail, color_detail, 1)
        start_y_detail_r += len(angle_keys_r) * line_height + 5 # Add space

        draw_text_with_background(frame_right_out, "--- Right Scores ---", (start_x_detail_r, start_y_detail_r), cv2.FONT_HERSHEY_SIMPLEX, font_scale_detail, color_detail, 1); start_y_detail_r += line_height
        score_keys_r = ['UA', 'LA', 'WR', 'WR-T', 'NK', 'TR', 'LG']
        score_labels_r = ["R UA Sc", "R LA Sc", "R WR Sc", "R WRT Sc", "R NK Sc", "R TR Sc", "R LG Sc"]
        for i, key_abbr in enumerate(score_keys_r):
            # Assuming right_results["scores"] uses keys like 'UA', 'LA' etc. directly
            value = right_results["scores"].get(key_abbr, 'N/A')
            text = f"{score_labels_r[i]}: {value}"
            draw_text_with_background(frame_right_out, text, (start_x_detail_r, start_y_detail_r + i * line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale_detail, color_detail, 1)
        # --- END: Added Display ---

        adj_y = 30; adj_x = 10; adj_line_h = 18; adj_font_scale=0.4 # Adjusted line height and font scale for more text
        draw_text_with_background(frame_front_out, f"left lower arm across midline: {front_adjustments['is_left_lower_arm_across_midline']}", (adj_x, adj_y), cv2.FONT_HERSHEY_SIMPLEX, adj_font_scale, (0,255,255), 1); adj_y+=adj_line_h
        draw_text_with_background(frame_front_out, f"right lower arm across midline: {front_adjustments['is_right_lower_arm_across_midline']}",(adj_x, adj_y), cv2.FONT_HERSHEY_SIMPLEX, adj_font_scale, (0,255,255), 1); adj_y+=adj_line_h
        draw_text_with_background(frame_front_out, f"left lower arm out to side of body: {front_adjustments['is_left_lower_arm_abducted']} ({l_low_abd_diff:.1f})",(adj_x, adj_y), cv2.FONT_HERSHEY_SIMPLEX, adj_font_scale, (0,255,255), 1); adj_y+=adj_line_h
        draw_text_with_background(frame_front_out, f"right lower arm out to side of body: {front_adjustments['is_right_lower_arm_abducted']} ({r_low_abd_diff:.1f})",(adj_x, adj_y), cv2.FONT_HERSHEY_SIMPLEX, adj_font_scale, (0,255,255), 1); adj_y+=adj_line_h
        draw_text_with_background(frame_front_out, f"---------------",(adj_x, adj_y), cv2.FONT_HERSHEY_SIMPLEX, adj_font_scale, (0,255,255), 1); adj_y+=adj_line_h
        # draw_text_with_background(frame_front_out, f"L Sh Ang: {angle_left_shoulder_deg:.1f}",(adj_x, adj_y), cv2.FONT_HERSHEY_SIMPLEX, adj_font_scale, (0,255,255), 1); adj_y+=adj_line_h
        # draw_text_with_background(frame_front_out, f"R Sh Ang: {angle_right_shoulder_deg:.1f}",(adj_x, adj_y), cv2.FONT_HERSHEY_SIMPLEX, adj_font_scale, (0,255,255), 1); adj_y+=adj_line_h
        draw_text_with_background(frame_front_out, f"left_arm_abducted: {front_adjustments['is_left_arm_abducted']}({angle_left_shoulder_deg:.1f})",(adj_x, adj_y), cv2.FONT_HERSHEY_SIMPLEX, adj_font_scale, (0,255,255), 1); adj_y+=adj_line_h
        draw_text_with_background(frame_front_out, f"right_arm_abducted: {front_adjustments['is_right_arm_abducted']}({angle_right_shoulder_deg:.1f})",(adj_x, adj_y), cv2.FONT_HERSHEY_SIMPLEX, adj_font_scale, (0,255,255), 1); adj_y+=adj_line_h
        draw_text_with_background(frame_front_out, f"left_shoulder_raised: {front_adjustments['is_left_shoulder_raised']} ({dist_l_ear_sh:.1f})",(adj_x, adj_y), cv2.FONT_HERSHEY_SIMPLEX, adj_font_scale, (0,255,255), 1); adj_y+=adj_line_h
        draw_text_with_background(frame_front_out, f"right_shoulder_raised: {front_adjustments['is_right_shoulder_raised']} ({dist_r_ear_sh:.1f})",(adj_x, adj_y), cv2.FONT_HERSHEY_SIMPLEX, adj_font_scale, (0,255,255), 1); adj_y+=adj_line_h
        draw_text_with_background(frame_front_out, f"---------------",(adj_x, adj_y), cv2.FONT_HERSHEY_SIMPLEX, adj_font_scale, (0,255,255), 1); adj_y+=adj_line_h
        # MODIFIED: Display Neck and Trunk Bend and Twist info
        draw_text_with_background(frame_front_out, f"Neck Twisted: {front_adjustments['is_neck_twisted']} ({neck_z_diff:.3f})",(adj_x, adj_y), cv2.FONT_HERSHEY_SIMPLEX, adj_font_scale, (0,255,255), 1); adj_y+=adj_line_h
        draw_text_with_background(frame_front_out, f"Trunk Twisted: {front_adjustments['is_trunk_twisted']} ({trunk_z_diff:.3f})",(adj_x, adj_y), cv2.FONT_HERSHEY_SIMPLEX, adj_font_scale, (0,255,255), 1); adj_y+=adj_line_h
        draw_text_with_background(frame_front_out, f"Trunk Bending: {front_adjustments['is_trunk_side_bending']} ({angle_diff_trunk_deg:.1f})",(adj_x, adj_y), cv2.FONT_HERSHEY_SIMPLEX, adj_font_scale, (0,255,255), 1); adj_y+=adj_line_h
        draw_text_with_background(frame_front_out, f"Neck Bending: {front_adjustments['is_neck_side_bending']} ({angle_neck_vert_deg:.1f})",(adj_x, adj_y), cv2.FONT_HERSHEY_SIMPLEX, adj_font_scale, (0,255,255), 1); adj_y+=adj_line_h
        draw_text_with_background(frame_front_out, f"Frame: {frame_count}", (10, adj_y), cv2.FONT_HERSHEY_SIMPLEX, adj_font_scale, (0, 255, 0), 1)

        # Final Score Display (on Front View)
        box_w = 300; box_h = 60
        box_x = w_front // 2 - box_w // 2
        box_y = h_front - box_h - 10
        cv2.rectangle(frame_front_out, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1) # Black background
        cv2.putText(frame_front_out, f"FINAL RULA: {final_rula_score}", (box_x + 10, box_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # Yellow Text
        # cv2.putText(frame_front_out, f"(Dominant: {dominant_side})", (box_x + 10, box_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        # Display Action Level Text below the final score box or elsewhere
       # cv2.putText(frame_front_out, final_action_level, (10, h_front - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # FPS
        curr_time = time.time()
        fps_left = 1.0 / (curr_time - prev_time_left)
        fps_right = 1.0 / (curr_time - prev_time_right)
        fps_front = 1.0 / (curr_time - prev_time_front)
        prev_time_left, prev_time_right, prev_time_front = curr_time, curr_time, curr_time

        # FPS
        cv2.putText(frame_left_out, f"FPS: {fps_left:.1f}", (10, h_left - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame_right_out, f"FPS: {fps_right:.1f}", (10, h_right - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame_front_out, f"FPS: {fps_front:.1f}", (10, h_front - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # frame_left_out = cv2.resize(frame_left_out, (320, 160))
        # frame_right_out = cv2.resize(frame_right_out, (320, 160))
        # frame_front_out = cv2.resize(frame_front_out, (320, 160))

        # top_row = cv2.hconcat([frame_left_out, frame_right_out])
        # combined1 = cv2.vconcat([top_row, frame_front_out])
        combined1 = cv2.hconcat([frame_left_out, frame_front_out, frame_right_out])
        
        if recording:
            # === VideoWriter ===
            if video_writer is None:
                h, w = combined1.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
                video_writer = cv2.VideoWriter(VIDEO_OUT_PATH, fourcc, VIDEO_FPS, (w, h))
                if not video_writer.isOpened():
                    print("Error: VideoWriter open failed, recording disabled.")
            # === CSV===
            if csv_writer is None:
                is_new = not os.path.exists(CSV_OUT_PATH)
                csv_fh = open(CSV_OUT_PATH, "a", newline="", encoding="utf-8")
                # row_data writer
                # row_data
                csv_fieldnames = [
                    "Frame",
                    "L_UA_Score","L_LA_Score","L_WR_Score","L_WRT_Score","L_NK_Score","L_TR_Score","L_A","L_B","L_C",
                    "R_UA_Score","R_LA_Score","R_WR_Score","R_WRT_Score","R_NK_Score","R_TR_Score","R_A","R_B","R_C",
                    "Final_RULA","Dominant_Side",
                ]
                csv_writer = csv.DictWriter(csv_fh, fieldnames=csv_fieldnames)
                if is_new:
                    csv_writer.writeheader()
                    csv_fh.flush()
            if video_writer is not None and video_writer.isOpened():
                video_writer.write(combined1)

            if "N/A" not in row_data.values() and csv_writer is not None:
                csv_writer.writerow(row_data)
                csv_fh.flush()


        cv2.imshow("Multi-Camera View", combined1)

        # === X ===
        # OpenCV
        if cv2.getWindowProperty("Multi-Camera View", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed by user.")
            break
        



        # cv2.imshow("RULA Left Side View", frame_left_out)
        # cv2.imshow("RULA Right Side View", frame_right_out)
        # cv2.imshow("RULA Front View & Final Score", frame_front_out)

    except Exception as main_loop_error:
        print(f"Error in main loop: {main_loop_error}")
        traceback.print_exc() # Print detailed traceback
        time.sleep(0.1) # Prevent rapid error loops
    
    
    # + / (p / q or esc)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('p'):
        recording = True
        print("Start recording and CSV logging...")
    if key == ord('q') or key == 27:
        print("Exit key pressed.")
        break
    frame_count += 1
 
print("Releasing resources...")

if video_writer is not None:
    try:
        video_writer.release()
        print(f"Saved recording to: {VIDEO_OUT_PATH}")
    except Exception as e:
        print(f"VideoWriter release error: {e}")
# === CSV ===
try:
    if csv_fh is not None and not csv_fh.closed:
        csv_fh.flush()
        try:
            os.fsync(csv_fh.fileno())
        except Exception:
            pass
        csv_fh.close()
        print(f"Saved CSV to: {CSV_OUT_PATH}")
except Exception as e:
    print(f"CSV close error: {e}")

cap_left_side.release()
cap_right_side.release()
cap_front.release()
cv2.destroyAllWindows()
pose_left_side.close()
pose_right_side.close()
pose_front.close()
hands_left_side.close()
hands_right_side.close()
print("Done.")
