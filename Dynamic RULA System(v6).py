import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import datetime
import csv
import os
from numpy.linalg import norm
import sys
import traceback
import math
import urllib.request

# ==== Output Paths (relative to script location) ====
RUN_TS     = time.strftime("%Y%m%d-%H%M%S")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_NAME  = f"{RUN_TS}_REBA"

VIDEO_FPS      = 15.0
VIDEO_CODEC    = 'mp4v'
VIDEO_OUT_PATH = os.path.join(OUTPUT_DIR, f"{BASE_NAME}.mp4")
CSV_OUT_PATH   = os.path.join(OUTPUT_DIR, f"{BASE_NAME}.csv")

video_writer = None
csv_fh       = None
csv_writer   = None
recording    = False

# ==== Camera Indices ====
LEFT_SIDE_CAMERA_INDEX  = 1   # Camera for LEFT side view
RIGHT_SIDE_CAMERA_INDEX = 2   # Camera for RIGHT side view
FRONT_CAMERA_INDEX      = 0   # Camera for Front view

# ==== REBA Thresholds ====
# --- Group A: Trunk, Neck, Legs ---
TRUNK_FLEXION_BINS        = (5, 20, 60)   # degrees: boundaries for trunk scores 1/2/3/4
TRUNK_SIDE_BEND_THRESHOLD = 15            # degrees (front view)
TRUNK_TWIST_Z_THRESHOLD   = 0.08         # MediaPipe Z difference

NECK_FLEXION_THRESHOLD    = 20            # degrees: 0-20 = score 1, >20 = score 2
NECK_SIDE_BEND_THRESHOLD  = 15           # degrees (front view)
NECK_TWIST_Z_THRESHOLD    = 0.05

KNEE_FLEXION_MODERATE     = 30            # +1 to leg score
KNEE_FLEXION_HIGH         = 60            # +2 to leg score

# --- Group B: Upper Arm, Lower Arm, Wrist ---
ARM_ABDUCTION_THRESHOLD   = 25            # degrees (elbow-shoulder-hip from front)
SHOULDER_RAISE_THRESHOLD  = 0.06          # fraction of frame height (ear-to-shoulder Y distance)
LOWER_ARM_NEUTRAL_MIN     = 60            # degrees elbow flexion
LOWER_ARM_NEUTRAL_MAX     = 100
WRIST_FLEXION_THRESHOLD   = 15            # degrees deviation from straight
WRIST_DEVIATION_RATIO     = 0.20          # fraction of forearm length for radial/ulnar deviation
WRIST_OUTSIDE_THRESHOLD   = 0.07          # fraction of frame width (wrist-shoulder X gap for lower arm abduction)

# ==== REBA Manual Factors (adjustable via keyboard: L=load, C=coupling) ====
REBA_LOAD_SCORE     = 0   # 0 (<5 kg), 1 (5–10 kg), 2 (>10 kg), +1 if shock/rapid force
REBA_COUPLING_SCORE = 0   # 0 good grip, 1 fair, 2 poor, 3 unacceptable
REBA_ACTIVITY_SCORE = 0   # 0–3: +1 static hold, +1 repeated >4/min, +1 rapid/unstable

# ==== MediaPipe Tasks API ====
BaseOptions           = mp.tasks.BaseOptions
PoseLandmarker        = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

# Pose landmark indices (replaces mp_pose.PoseLandmark)
POSE_LEFT_EAR       = 7
POSE_RIGHT_EAR      = 8
POSE_LEFT_SHOULDER  = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_ELBOW     = 13
POSE_RIGHT_ELBOW    = 14
POSE_LEFT_WRIST     = 15
POSE_RIGHT_WRIST    = 16
POSE_LEFT_HIP       = 23
POSE_RIGHT_HIP      = 24
POSE_LEFT_KNEE      = 25
POSE_RIGHT_KNEE     = 26
POSE_LEFT_ANKLE     = 27
POSE_RIGHT_ANKLE    = 28

# Hand landmark index (replaces mp_hands.HandLandmark)
HAND_MIDDLE_FINGER_PIP = 10

# ==== Model files (auto-download if missing) ====
POSE_MODEL_PATH = os.path.join(OUTPUT_DIR, 'pose_landmarker_full.task')
HAND_MODEL_PATH = os.path.join(OUTPUT_DIR, 'hand_landmarker.task')
_POSE_URL = ('https://storage.googleapis.com/mediapipe-models/'
             'pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task')
_HAND_URL = ('https://storage.googleapis.com/mediapipe-models/'
             'hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task')
for _path, _url, _name in [(POSE_MODEL_PATH, _POSE_URL, 'Pose'),
                            (HAND_MODEL_PATH, _HAND_URL, 'Hand')]:
    if not os.path.exists(_path):
        print(f"Downloading {_name} model → {_path} ...")
        urllib.request.urlretrieve(_url, _path)
        print(f"{_name} model ready.")

# ==== Landmarker instances ====
_pose_opts = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    min_pose_detection_confidence=0.4,
    min_pose_presence_confidence=0.4,
    min_tracking_confidence=0.4)
_hand_opts_1 = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5)
_hand_opts_2 = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5)

pose_left_side   = PoseLandmarker.create_from_options(_pose_opts)
pose_right_side  = PoseLandmarker.create_from_options(_pose_opts)
pose_front       = PoseLandmarker.create_from_options(_pose_opts)
hands_left_side  = HandLandmarker.create_from_options(_hand_opts_1)
hands_right_side = HandLandmarker.create_from_options(_hand_opts_1)
hands_front      = HandLandmarker.create_from_options(_hand_opts_2)

# Per-instance monotonic timestamp counters (Tasks API requires strictly increasing values)
_ts_pose_l = _ts_pose_r = _ts_pose_f = 0
_ts_hand_l = _ts_hand_r = _ts_hand_f = 0


# ============================================================
# Geometry helpers
# ============================================================
def calculate_angle_with_sign(a, b, c):
    """Signed angle at vertex b (degrees, range -180..180)."""
    a = np.array(a[:2]); b = np.array(b[:2]); c = np.array(c[:2])
    ba = a - b;           bc = c - b
    if norm(ba) < 1e-6 or norm(bc) < 1e-6:
        return 0.0
    angle_deg = np.degrees(np.arctan2(bc[1], bc[0]) - np.arctan2(ba[1], ba[0]))
    return (angle_deg + 180) % 360 - 180


def calculate_angle_acos(a, b, c):
    """Unsigned angle at vertex b via acos (degrees, range 0..180)."""
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b; bc = c - b
    if norm(ba) < 1e-6 or norm(bc) < 1e-6:
        return 0.0
    cos_theta = np.clip(np.dot(ba, bc) / (norm(ba) * norm(bc)), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def compute_wrist_angle_2dxy(elbow, wrist, middle, width, height):
    """Angle elbow–wrist–middle_finger in 2-D pixel space."""
    a = np.array([elbow.x  * width, elbow.y  * height])
    b = np.array([wrist.x  * width, wrist.y  * height])
    c = np.array([middle.x * width, middle.y * height])
    v1 = a - b; v2 = c - b
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def draw_text_with_background(img, text, org, font, scale, text_color,
                               bg_color=(0, 0, 0), thickness=1, padding=2):
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - padding, y - th - padding),
                  (x + tw + padding, y + baseline + padding), bg_color, -1)
    cv2.putText(img, text, (x, y), font, scale, text_color, thickness)


# ============================================================
# REBA Action Level
# ============================================================
def get_reba_action_level(final_score):
    """Return REBA action level string for score 1..15+."""
    if final_score is None:
        return "N/A"
    try:
        s = int(round(float(final_score)))
    except (TypeError, ValueError):
        return "Score Error"
    if s == 1:   return "Level 1: Negligible risk"
    if s <= 3:   return "Level 2: Low risk, change may be needed"
    if s <= 7:   return "Level 3: Medium risk, investigate soon"
    if s <= 10:  return "Level 4: High risk, implement change"
    if s <= 15:  return "Level 5: Very high risk, change NOW"
    return "Out of range (>15)"


# ============================================================
# REBA Lookup Tables  (A, B, C)
# ============================================================
# Table A: rows = Trunk 1-5, cols = Neck(1-3) x Legs(1-4)
table_a_data = {
    "Trunk": [1, 2, 3, 4, 5],
    "N1_L1": [1, 2, 2, 3, 4], "N1_L2": [2, 3, 4, 5, 6],
    "N1_L3": [3, 4, 5, 6, 7], "N1_L4": [4, 5, 6, 7, 8],
    "N2_L1": [1, 3, 4, 5, 6], "N2_L2": [2, 4, 5, 6, 7],
    "N2_L3": [3, 5, 6, 7, 8], "N2_L4": [4, 6, 7, 8, 9],
    "N3_L1": [3, 4, 5, 6, 7], "N3_L2": [3, 5, 6, 7, 8],
    "N3_L3": [5, 6, 7, 8, 9], "N3_L4": [6, 7, 8, 9, 9],
}
table_a = pd.DataFrame(table_a_data)

# Table B: rows = UpperArm 1-6, cols = LowerArm(1-2) x Wrist(1-3)
table_b_data = {
    'UpperArm': [1, 2, 3, 4, 5, 6],
    'L1_W1': [1, 1, 3, 4, 6, 7], 'L1_W2': [2, 2, 4, 5, 7, 8], 'L1_W3': [2, 3, 5, 5, 8, 8],
    'L2_W1': [1, 2, 4, 5, 7, 8], 'L2_W2': [2, 3, 5, 6, 8, 9], 'L2_W3': [3, 4, 5, 7, 8, 9],
}
table_b = pd.DataFrame(table_b_data)

# Table C: Score A (1-12) x Score B (1-12) → REBA C score
table_c_data = {
    "ScoreA": [1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
    "B1":  [1,  1,  2,  3,  4,  6,  7,  8,  9, 10, 11, 12],
    "B2":  [1,  2,  3,  4,  4,  6,  7,  8,  9, 10, 11, 12],
    "B3":  [1,  2,  3,  4,  4,  6,  7,  8,  9, 10, 11, 12],
    "B4":  [2,  3,  3,  4,  5,  7,  8,  9, 10, 11, 11, 12],
    "B5":  [3,  4,  4,  5,  6,  8,  9, 10, 10, 11, 12, 12],
    "B6":  [3,  4,  5,  6,  7,  8,  9, 10, 10, 11, 12, 12],
    "B7":  [4,  5,  6,  7,  8,  9,  9, 10, 11, 11, 12, 12],
    "B8":  [5,  6,  7,  8,  8,  9, 10, 10, 11, 12, 12, 12],
    "B9":  [6,  6,  7,  8,  9, 10, 10, 10, 11, 12, 12, 12],
    "B10": [7,  7,  8,  9,  9, 10, 11, 11, 12, 12, 12, 12],
    "B11": [7,  7,  8,  9,  9, 10, 11, 11, 12, 12, 12, 12],
    "B12": [7,  8,  8,  9,  9, 10, 11, 11, 12, 12, 12, 12],
}
table_c = pd.DataFrame(table_c_data)


def _to_valid_int_score(name, value, lo, hi):
    """Clamp-and-round numeric score to [lo, hi]."""
    if value is None:
        raise ValueError(f"{name} is None")
    try:
        x = float(value)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}") from e
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {x}")
    return int(max(lo, min(hi, round(x))))


def get_table_a_score(neck_score, trunk_score, legs_score, table_df=None):
    """REBA Table A lookup. Inputs: neck 1-3, trunk 1-5, legs 1-4."""
    df = table_a if table_df is None else table_df
    try:
        neck  = _to_valid_int_score("neck_score",  neck_score,  1, 3)
        trunk = _to_valid_int_score("trunk_score", trunk_score, 1, 5)
        legs  = _to_valid_int_score("legs_score",  legs_score,  1, 4)
        col = f"N{neck}_L{legs}"
        row = df.loc[df["Trunk"] == trunk]
        if row.empty:
            raise ValueError(f"No row for Trunk={trunk}")
        value = row.iloc[0][col]
        if pd.isna(value):
            raise ValueError(f"NaN at Trunk={trunk}, {col}")
        return int(value)
    except Exception as e:
        print(f"Err REBA Table A: {e}"); return None


def get_table_b_score(upper_arm_score, lower_arm_score, wrist_score, table_df=None):
    """REBA Table B lookup. Inputs: upper_arm 1-6, lower_arm 1-2, wrist 1-3."""
    df = table_b if table_df is None else table_df
    try:
        ua = _to_valid_int_score("upper_arm_score", upper_arm_score, 1, 6)
        la = _to_valid_int_score("lower_arm_score", lower_arm_score, 1, 2)
        ws = _to_valid_int_score("wrist_score",     wrist_score,     1, 3)
        col = f"L{la}_W{ws}"
        row = df.loc[df["UpperArm"] == ua]
        if row.empty:
            raise ValueError(f"No row for UpperArm={ua}")
        value = row.iloc[0][col]
        if pd.isna(value):
            raise ValueError(f"NaN at UpperArm={ua}, {col}")
        return int(value)
    except Exception as e:
        print(f"Err REBA Table B: {e}"); return None


def get_table_c_score(score_a, score_b, table_df=None):
    """REBA Table C lookup. Inputs: score_a 1-12, score_b 1-12."""
    df = table_c if table_df is None else table_df
    try:
        sa = _to_valid_int_score("score_a", score_a, 1, 12)
        sb = _to_valid_int_score("score_b", score_b, 1, 12)
        col = f"B{sb}"
        row = df.loc[df["ScoreA"] == sa]
        if row.empty:
            raise ValueError(f"No row for ScoreA={sa}")
        value = row.iloc[0][col]
        if pd.isna(value):
            raise ValueError(f"NaN at ScoreA={sa}, {col}")
        return int(value)
    except Exception as e:
        print(f"Err REBA Table C: {e}"); return None


# ============================================================
# REBA Component Score Calculator
# ============================================================
def get_reba_component_scores(side, upper_arm_angle, lower_arm_angle, wrist_angle,
                               neck_angle, trunk_angle, adj_flags=None):
    """
    Compute REBA component scores from landmark angles.

    Parameters
    ----------
    side              : 'left' or 'right'
    upper_arm_angle   : signed angle from calculate_angle_with_sign (hip-shoulder-elbow)
    lower_arm_angle   : signed angle at elbow (shoulder-elbow-wrist)
    wrist_angle       : angle elbow-wrist-middle_finger (180 = straight), or None
    neck_angle        : signed angle at shoulder (ear-shoulder-hip)
    trunk_angle       : signed angle at hip (shoulder-hip-vertical_down)
    adj_flags         : dict of boolean adjustment flags from front-view analysis

    Returns
    -------
    (upper_arm, lower_arm, wrist, neck, trunk, legs)
    REBA ranges: UA 1-6, LA 1-2, WR 1-3, NK 1-3, TR 1-5, LG 1-4
    """
    if adj_flags is None:
        adj_flags = {}

    if side not in ('left', 'right'):
        print(f"Error: invalid side '{side}'")
        return (1, 1, 1, 1, 1, 1)

    negate   = (side == 'left')
    side_key = side   # 'left' or 'right'

    # ---- Upper Arm (REBA: base 1-4, +adjustments → max 6) ----
    ua_ref = -upper_arm_angle if negate else upper_arm_angle
    if   -20 <= ua_ref <= 20: upper_arm_score = 1
    elif  ua_ref < -20:       upper_arm_score = 2   # extension
    elif  ua_ref <= 45:       upper_arm_score = 2   # 20-45° flexion
    elif  ua_ref <= 90:       upper_arm_score = 3   # 45-90°
    else:                     upper_arm_score = 4   # >90°
    if adj_flags.get(f'is_{side_key}_arm_abducted',    False): upper_arm_score += 1
    if adj_flags.get(f'is_{side_key}_shoulder_raised', False): upper_arm_score += 1

    # ---- Lower Arm (REBA: 1-2, no adjustments) ----
    la_ref = -lower_arm_angle if negate else lower_arm_angle
    lower_arm_score = 1 if LOWER_ARM_NEUTRAL_MIN <= la_ref <= LOWER_ARM_NEUTRAL_MAX else 2

    # ---- Wrist (REBA: base 1-2, +1 for deviation → max 3) ----
    if wrist_angle is not None:
        wr_flex = abs(180.0 - wrist_angle)
        wrist_score = 1 if wr_flex <= WRIST_FLEXION_THRESHOLD else 2
        if adj_flags.get(f'is_{side_key}_wrist_bent_from_midline', False):
            wrist_score += 1
    else:
        wrist_score = 1   # default neutral when hand not detected

    # ---- Neck (REBA: base 1-2, +1 twist, +1 side bend → max 4, capped at 3) ----
    nk_ref     = -neck_angle if negate else neck_angle
    neck_flex  = 160 - nk_ref   # empirical offset; ~0 when upright, increases with flexion
    neck_score = 1 if neck_flex <= NECK_FLEXION_THRESHOLD else 2
    if adj_flags.get('is_neck_side_bent', False): neck_score += 1
    if adj_flags.get('is_neck_twisted',   False): neck_score += 1

    # ---- Trunk (REBA: base 1-4, +1 twist, +1 side bend → max 6, capped at 5) ----
    tr_ref     = -trunk_angle if negate else trunk_angle
    trunk_flex = 180 - abs(tr_ref)   # ~0 when upright, increases with flexion
    if   trunk_flex <= TRUNK_FLEXION_BINS[0]: trunk_score = 1
    elif trunk_flex <= TRUNK_FLEXION_BINS[1]: trunk_score = 2
    elif trunk_flex <= TRUNK_FLEXION_BINS[2]: trunk_score = 3
    else:                                      trunk_score = 4
    if adj_flags.get('is_trunk_side_bent', False): trunk_score += 1
    if adj_flags.get('is_trunk_twisted',   False): trunk_score += 1

    # ---- Legs (REBA: base 1-2, +knee flexion modifier → max 4) ----
    leg_score = 2 if adj_flags.get('is_unilateral_stance', False) else 1
    if   adj_flags.get('is_knee_flexed_high',     False): leg_score += 2
    elif adj_flags.get('is_knee_flexed_moderate', False): leg_score += 1

    # Clamp to REBA table input ranges
    upper_arm_score = max(1, min(upper_arm_score, 6))
    lower_arm_score = max(1, min(lower_arm_score, 2))
    wrist_score     = max(1, min(wrist_score,     3))
    neck_score      = max(1, min(neck_score,      3))
    trunk_score     = max(1, min(trunk_score,     5))
    leg_score       = max(1, min(leg_score,       4))

    return (upper_arm_score, lower_arm_score, wrist_score, neck_score, trunk_score, leg_score)


# ============================================================
# Camera Initialization
# ============================================================
print("Initializing cameras...")
cap_left_side  = cv2.VideoCapture(LEFT_SIDE_CAMERA_INDEX,  cv2.CAP_DSHOW)
cap_right_side = cv2.VideoCapture(RIGHT_SIDE_CAMERA_INDEX, cv2.CAP_DSHOW)
cap_front      = cv2.VideoCapture(FRONT_CAMERA_INDEX,       cv2.CAP_DSHOW)

cam_map = {
    "Left Side":  (cap_left_side,  LEFT_SIDE_CAMERA_INDEX),
    "Right Side": (cap_right_side, RIGHT_SIDE_CAMERA_INDEX),
    "Front":      (cap_front,       FRONT_CAMERA_INDEX),
}
for name, (cap, idx) in cam_map.items():
    if not cap.isOpened():
        print(f"Error: Cannot open camera '{name}' (Index {idx})")
        for c, _ in cam_map.values():
            if c.isOpened(): c.release()
        sys.exit(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("Starting REBA analysis  |  Keys: P=record  Q/ESC=quit  L=load  C=coupling  A=activity")
frame_count     = 0
rec_frame_count = 0          # resets to 0 when recording starts
prev_time   = time.time()   # initialized here; updated at end of each iteration

def _resize_h(f, h):
    """Resize frame to height h keeping aspect ratio (for safe hconcat)."""
    fh, fw = f.shape[:2]
    return f if fh == h else cv2.resize(f, (int(fw * h / fh), h))

# ============================================================
# Main Loop
# ============================================================
while True:
    ret_left,  frame_left  = cap_left_side.read()
    ret_right, frame_right = cap_right_side.read()
    ret_front, frame_front = cap_front.read()

    # Initialise per-frame debug vars to safe defaults
    l_low_abd_diff = r_low_abd_diff = 0.0
    dist_l_ear_sh  = dist_r_ear_sh  = 0.0
    angle_left_shoulder_deg = angle_right_shoulder_deg = 0.0
    angle_diff_trunk_deg    = angle_neck_vert_deg       = 0.0
    neck_z_diff             = trunk_z_diff              = 0.0
    l_knee_flex             = r_knee_flex               = 0.0

    if not ret_left or not ret_right or not ret_front:
        time.sleep(0.05)
        continue

    try:
        h_left,  w_left,  _ = frame_left.shape
        h_right, w_right, _ = frame_right.shape
        h_front, w_front, _ = frame_front.shape

        left_results  = {"angles": {}, "scores": {}, "table_scores": {"A": "N/A", "B": "N/A", "C": "N/A"}, "valid": False}
        right_results = {"angles": {}, "scores": {}, "table_scores": {"A": "N/A", "B": "N/A", "C": "N/A"}, "valid": False}

        front_adjustments = {
            'is_left_lower_arm_across_midline':  False,
            'is_right_lower_arm_across_midline': False,
            'is_left_lower_arm_abducted':        False,
            'is_right_lower_arm_abducted':       False,
            'is_trunk_side_bent':                False,   # NOTE: was 'bending' (bug fixed)
            'is_neck_side_bent':                 False,   # NOTE: was 'bending' (bug fixed)
            'is_left_arm_abducted':              False,
            'is_right_arm_abducted':             False,
            'is_neck_twisted':                   False,
            'is_trunk_twisted':                  False,
            'is_left_shoulder_raised':           False,
            'is_right_shoulder_raised':          False,
            'is_left_wrist_bent_from_midline':   False,
            'is_right_wrist_bent_from_midline':  False,
            'is_unilateral_stance':              False,
            'is_knee_flexed_moderate':           False,
            'is_knee_flexed_high':               False,
        }

        # ================================================================
        # LEFT SIDE camera — Group A angles (trunk/neck) + Group B (arm)
        # ================================================================
        frame_left_rgb = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
        mp_img_l = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_left_rgb)
        _ts_pose_l += 1; results_ls = pose_left_side.detect_for_video(mp_img_l, _ts_pose_l)
        _ts_hand_l += 1; results_lh = hands_left_side.detect_for_video(mp_img_l, _ts_hand_l)
        frame_left_out = frame_left.copy()

        hand_lm_l = None
        if results_lh.hand_landmarks and results_lh.handedness:
            for lm_list, handed_list in zip(results_lh.hand_landmarks, results_lh.handedness):
                if handed_list[0].category_name == 'Left':
                    hand_lm_l = lm_list; break
        if hand_lm_l is None and results_lh.hand_landmarks:
            hand_lm_l = results_lh.hand_landmarks[0]

        if results_ls.pose_landmarks:
            landmarks_ls = results_ls.pose_landmarks[0]
            try:
                l_sh_lm  = landmarks_ls[POSE_LEFT_SHOULDER]
                l_el_lm  = landmarks_ls[POSE_LEFT_ELBOW]
                l_wr_lm  = landmarks_ls[POSE_LEFT_WRIST]
                l_hip_lm = landmarks_ls[POSE_LEFT_HIP]
                l_ear_lm = landmarks_ls[POSE_LEFT_EAR]

                required_lms = [l_sh_lm, l_el_lm, l_wr_lm, l_hip_lm, l_ear_lm]
                if all(lm.visibility > 0.6 for lm in required_lms):
                    left_results["valid"] = True
                    l_sh_pt  = (int(l_sh_lm.x * w_left),  int(l_sh_lm.y * h_left))
                    l_el_pt  = (int(l_el_lm.x * w_left),  int(l_el_lm.y * h_left))
                    l_wr_pt  = (int(l_wr_lm.x * w_left),  int(l_wr_lm.y * h_left))
                    l_hip_pt = (int(l_hip_lm.x * w_left), int(l_hip_lm.y * h_left))
                    l_ear_pt = (int(l_ear_lm.x * w_left), int(l_ear_lm.y * h_left))
                    hip_vert_down = (l_hip_pt[0], h_left)

                    left_results["angles"]["upper_arm"] = calculate_angle_with_sign(l_hip_pt, l_sh_pt, l_el_pt)
                    left_results["angles"]["lower_arm"] = calculate_angle_with_sign(l_sh_pt, l_el_pt, l_wr_pt)
                    left_results["angles"]["neck"]      = calculate_angle_with_sign(l_ear_pt, l_sh_pt, l_hip_pt)
                    left_results["angles"]["trunk"]     = calculate_angle_with_sign(l_sh_pt, l_hip_pt, hip_vert_down)

                    if hand_lm_l is not None:
                        angle_l = compute_wrist_angle_2dxy(
                            l_el_lm, l_wr_lm,
                            hand_lm_l[HAND_MIDDLE_FINGER_PIP],
                            w_left, h_left)
                        left_results["angles"]["wrist"] = angle_l
                        mid_lm = hand_lm_l[HAND_MIDDLE_FINGER_PIP]
                        mid_pt = (int(mid_lm.x * w_left), int(mid_lm.y * h_left))
                        cv2.circle(frame_left_out, mid_pt, 4, (0, 255, 0), -1)
                        cv2.line(frame_left_out, mid_pt, l_wr_pt, (255, 0, 0), 1)

                    # Draw skeleton
                    for pt in [l_sh_pt, l_el_pt, l_wr_pt, l_ear_pt, l_hip_pt]:
                        cv2.circle(frame_left_out, pt, 3, (0, 255, 0), -1)
                    cv2.line(frame_left_out, l_sh_pt, l_el_pt, (255, 0, 0), 1)
                    cv2.line(frame_left_out, l_el_pt, l_wr_pt, (255, 0, 0), 1)
                    cv2.line(frame_left_out, l_sh_pt, l_hip_pt, (0, 0, 255), 1)
                    cv2.line(frame_left_out, l_hip_pt, hip_vert_down, (0, 0, 255), 1)
                    cv2.line(frame_left_out, l_sh_pt, l_ear_pt, (0, 0, 255), 1)

            except (IndexError, Exception) as e:
                left_results["valid"] = False
                print(f"Err Left Side LMs: {e}")

        # ================================================================
        # RIGHT SIDE camera
        # ================================================================
        frame_right_rgb = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
        mp_img_r = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_right_rgb)
        _ts_pose_r += 1; results_rs = pose_right_side.detect_for_video(mp_img_r, _ts_pose_r)
        _ts_hand_r += 1; results_rh = hands_right_side.detect_for_video(mp_img_r, _ts_hand_r)
        frame_right_out = frame_right.copy()

        hand_lm_r = None
        if results_rh.hand_landmarks and results_rh.handedness:
            for lm_list, handed_list in zip(results_rh.hand_landmarks, results_rh.handedness):
                if handed_list[0].category_name == 'Right':
                    hand_lm_r = lm_list; break
        if hand_lm_r is None and results_rh.hand_landmarks:
            hand_lm_r = results_rh.hand_landmarks[0]

        if results_rs.pose_landmarks:
            landmarks_rs = results_rs.pose_landmarks[0]
            try:
                r_sh_lm  = landmarks_rs[POSE_RIGHT_SHOULDER]
                r_el_lm  = landmarks_rs[POSE_RIGHT_ELBOW]
                r_wr_lm  = landmarks_rs[POSE_RIGHT_WRIST]
                r_hip_lm = landmarks_rs[POSE_RIGHT_HIP]
                r_ear_lm = landmarks_rs[POSE_RIGHT_EAR]

                required_rms = [r_sh_lm, r_el_lm, r_wr_lm, r_hip_lm, r_ear_lm]
                if all(lm.visibility > 0.6 for lm in required_rms):
                    right_results["valid"] = True
                    r_sh_pt  = (int(r_sh_lm.x * w_right),  int(r_sh_lm.y * h_right))
                    r_el_pt  = (int(r_el_lm.x * w_right),  int(r_el_lm.y * h_right))
                    r_wr_pt  = (int(r_wr_lm.x * w_right),  int(r_wr_lm.y * h_right))
                    r_hip_pt = (int(r_hip_lm.x * w_right), int(r_hip_lm.y * h_right))
                    r_ear_pt = (int(r_ear_lm.x * w_right), int(r_ear_lm.y * h_right))
                    hip_vert_down_r = (r_hip_pt[0], h_right)

                    right_results["angles"]["upper_arm"] = calculate_angle_with_sign(r_hip_pt, r_sh_pt, r_el_pt)
                    right_results["angles"]["lower_arm"] = calculate_angle_with_sign(r_sh_pt, r_el_pt, r_wr_pt)
                    right_results["angles"]["neck"]      = calculate_angle_with_sign(r_ear_pt, r_sh_pt, r_hip_pt)
                    right_results["angles"]["trunk"]     = calculate_angle_with_sign(r_sh_pt, r_hip_pt, hip_vert_down_r)

                    if hand_lm_r is not None:
                        angle_r = compute_wrist_angle_2dxy(
                            r_el_lm, r_wr_lm,
                            hand_lm_r[HAND_MIDDLE_FINGER_PIP],
                            w_right, h_right)
                        right_results["angles"]["wrist"] = angle_r
                        mid_lm_r = hand_lm_r[HAND_MIDDLE_FINGER_PIP]
                        mid_pt_r = (int(mid_lm_r.x * w_right), int(mid_lm_r.y * h_right))
                        cv2.circle(frame_right_out, mid_pt_r, 4, (0, 255, 0), -1)
                        cv2.line(frame_right_out, mid_pt_r, r_wr_pt, (255, 0, 0), 1)

                    # Draw skeleton
                    for pt in [r_sh_pt, r_el_pt, r_wr_pt, r_ear_pt, r_hip_pt]:
                        cv2.circle(frame_right_out, pt, 3, (0, 255, 0), -1)
                    cv2.line(frame_right_out, r_sh_pt, r_el_pt, (255, 0, 0), 1)
                    cv2.line(frame_right_out, r_el_pt, r_wr_pt, (255, 0, 0), 1)
                    cv2.line(frame_right_out, r_sh_pt, r_hip_pt, (0, 0, 255), 1)
                    cv2.line(frame_right_out, r_hip_pt, hip_vert_down_r, (0, 0, 255), 1)
                    cv2.line(frame_right_out, r_sh_pt, r_ear_pt, (0, 0, 255), 1)

            except (IndexError, Exception) as e:
                right_results["valid"] = False
                print(f"Err Right Side LMs: {e}")

        # ================================================================
        # FRONT camera — adjustment flags + knee flexion
        # ================================================================
        frame_front_rgb = cv2.cvtColor(frame_front, cv2.COLOR_BGR2RGB)
        mp_img_f = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_front_rgb)
        _ts_pose_f += 1; results_front = pose_front.detect_for_video(mp_img_f, _ts_pose_f)
        frame_front_out = frame_front.copy()

        if results_front.pose_landmarks:
            landmarks_f = results_front.pose_landmarks[0]
            try:
                l_sh_lm_f  = landmarks_f[POSE_LEFT_SHOULDER]
                r_sh_lm_f  = landmarks_f[POSE_RIGHT_SHOULDER]
                l_hip_lm_f = landmarks_f[POSE_LEFT_HIP]
                r_hip_lm_f = landmarks_f[POSE_RIGHT_HIP]
                l_wr_lm_f  = landmarks_f[POSE_LEFT_WRIST]
                r_wr_lm_f  = landmarks_f[POSE_RIGHT_WRIST]
                l_ear_lm_f = landmarks_f[POSE_LEFT_EAR]
                r_ear_lm_f = landmarks_f[POSE_RIGHT_EAR]
                l_el_lm_f  = landmarks_f[POSE_LEFT_ELBOW]
                r_el_lm_f  = landmarks_f[POSE_RIGHT_ELBOW]
                l_kn_lm_f  = landmarks_f[POSE_LEFT_KNEE]
                r_kn_lm_f  = landmarks_f[POSE_RIGHT_KNEE]
                l_an_lm_f  = landmarks_f[POSE_LEFT_ANKLE]
                r_an_lm_f  = landmarks_f[POSE_RIGHT_ANKLE]

                required_f = [l_sh_lm_f, r_sh_lm_f, l_hip_lm_f, r_hip_lm_f,
                               l_wr_lm_f, r_wr_lm_f, l_ear_lm_f, r_ear_lm_f,
                               l_el_lm_f, r_el_lm_f]

                if all(lm.visibility > 0.6 for lm in required_f):
                    l_sh_f      = np.array([l_sh_lm_f.x * w_front,  l_sh_lm_f.y * h_front])
                    r_sh_f      = np.array([r_sh_lm_f.x * w_front,  r_sh_lm_f.y * h_front])
                    l_hip_f     = np.array([l_hip_lm_f.x * w_front, l_hip_lm_f.y * h_front])
                    r_hip_f     = np.array([r_hip_lm_f.x * w_front, r_hip_lm_f.y * h_front])
                    l_wr_f      = np.array([l_wr_lm_f.x * w_front,  l_wr_lm_f.y * h_front])
                    r_wr_f      = np.array([r_wr_lm_f.x * w_front,  r_wr_lm_f.y * h_front])
                    l_el_f      = np.array([l_el_lm_f.x * w_front,  l_el_lm_f.y * h_front])
                    r_el_f      = np.array([r_el_lm_f.x * w_front,  r_el_lm_f.y * h_front])
                    l_ear_f_xy  = np.array([l_ear_lm_f.x * w_front, l_ear_lm_f.y * h_front])
                    r_ear_f_xy  = np.array([r_ear_lm_f.x * w_front, r_ear_lm_f.y * h_front])

                    sh_mid_f  = (l_sh_f + r_sh_f) / 2
                    hip_mid_f = (l_hip_f + r_hip_f) / 2
                    ear_mid_f = (l_ear_f_xy + r_ear_f_xy) / 2

                    l_low_abd_diff = abs(l_wr_f[0] - l_sh_f[0])
                    r_low_abd_diff = abs(r_wr_f[0] - r_sh_f[0])

                    # Wrist outside shoulder → lower arm abducted (threshold normalized to frame width)
                    wrist_outside_px = WRIST_OUTSIDE_THRESHOLD * w_front
                    if l_wr_f[0] > l_sh_f[0] and l_low_abd_diff > wrist_outside_px:
                        front_adjustments['is_left_lower_arm_abducted'] = True
                    if r_wr_f[0] < r_sh_f[0] and r_low_abd_diff > wrist_outside_px:
                        front_adjustments['is_right_lower_arm_abducted'] = True

                    # Wrist crossing body midline
                    if abs(sh_mid_f[0] - hip_mid_f[0]) < 1e-6:
                        if (l_sh_f[0] < sh_mid_f[0]) != (l_wr_f[0] < sh_mid_f[0]):
                            front_adjustments['is_left_lower_arm_across_midline'] = True
                        if (r_sh_f[0] > sh_mid_f[0]) != (r_wr_f[0] > sh_mid_f[0]):
                            front_adjustments['is_right_lower_arm_across_midline'] = True
                    else:
                        A = hip_mid_f[1] - sh_mid_f[1]
                        B = sh_mid_f[0]  - hip_mid_f[0]
                        C = -A * sh_mid_f[0] - B * sh_mid_f[1]
                        sign_l_sh = A*l_sh_f[0] + B*l_sh_f[1] + C
                        sign_l_wr = A*l_wr_f[0] + B*l_wr_f[1] + C
                        if sign_l_sh * sign_l_wr < 0:
                            front_adjustments['is_left_lower_arm_across_midline'] = True
                        sign_r_sh = A*r_sh_f[0] + B*r_sh_f[1] + C
                        sign_r_wr = A*r_wr_f[0] + B*r_wr_f[1] + C
                        if sign_r_sh * sign_r_wr < 0:
                            front_adjustments['is_right_lower_arm_across_midline'] = True

                    # Arm abduction: from front view, check how far the elbow is lateral
                    # of the shoulder–hip vertical line. This is a better proxy than the
                    # elbow-shoulder-hip angle (which conflates forward flexion with abduction).
                    try:
                        # Left arm: left elbow should be to the LEFT of left shoulder for abduction
                        sh_width_px = abs(r_sh_f[0] - l_sh_f[0])  # shoulder width as scale ref
                        if sh_width_px > 1e-6:
                            l_el_lateral = l_sh_f[0] - l_el_f[0]  # positive = elbow left of shoulder
                            angle_left_shoulder_deg = np.degrees(np.arctan2(
                                abs(l_el_f[0] - l_sh_f[0]), max(abs(l_el_f[1] - l_sh_f[1]), 1e-6)))
                            if l_el_lateral > 0 and angle_left_shoulder_deg > ARM_ABDUCTION_THRESHOLD:
                                front_adjustments['is_left_arm_abducted'] = True
                    except Exception as e:
                        print(f"Warn L Sh Angle: {e}")

                    try:
                        # Right arm: right elbow should be to the RIGHT of right shoulder for abduction
                        sh_width_px = abs(r_sh_f[0] - l_sh_f[0])
                        if sh_width_px > 1e-6:
                            r_el_lateral = r_el_f[0] - r_sh_f[0]  # positive = elbow right of shoulder
                            angle_right_shoulder_deg = np.degrees(np.arctan2(
                                abs(r_el_f[0] - r_sh_f[0]), max(abs(r_el_f[1] - r_sh_f[1]), 1e-6)))
                            if r_el_lateral > 0 and angle_right_shoulder_deg > ARM_ABDUCTION_THRESHOLD:
                                front_adjustments['is_right_arm_abducted'] = True
                    except Exception as e:
                        print(f"Warn R Sh Angle: {e}")

                    # Shoulder raise (ear-to-shoulder Y distance, normalized to frame height)
                    dist_l_ear_sh = abs(l_ear_f_xy[1] - l_sh_f[1])
                    dist_r_ear_sh = abs(r_ear_f_xy[1] - r_sh_f[1])
                    shoulder_raise_px = SHOULDER_RAISE_THRESHOLD * h_front
                    if dist_l_ear_sh < shoulder_raise_px:
                        front_adjustments['is_left_shoulder_raised'] = True
                    if dist_r_ear_sh < shoulder_raise_px:
                        front_adjustments['is_right_shoulder_raised'] = True

                    # Wrist lateral (radial/ulnar) deviation from front view.
                    # If the wrist X deviates from the elbow-to-shoulder line by more than
                    # WRIST_DEVIATION_RATIO of the forearm length, flag as bent from midline.
                    try:
                        l_forearm = norm(l_wr_f - l_el_f)
                        if l_forearm > 1e-6:
                            l_wr_dev = abs((l_wr_f[0] - l_el_f[0]) - (l_el_f[0] - l_sh_f[0]) *
                                          (l_forearm / max(norm(l_el_f - l_sh_f), 1e-6)))
                            if l_wr_dev / l_forearm > WRIST_DEVIATION_RATIO:
                                front_adjustments['is_left_wrist_bent_from_midline'] = True
                    except Exception as e:
                        print(f"Warn L Wrist Dev: {e}")
                    try:
                        r_forearm = norm(r_wr_f - r_el_f)
                        if r_forearm > 1e-6:
                            r_wr_dev = abs((r_wr_f[0] - r_el_f[0]) - (r_el_f[0] - r_sh_f[0]) *
                                          (r_forearm / max(norm(r_el_f - r_sh_f), 1e-6)))
                            if r_wr_dev / r_forearm > WRIST_DEVIATION_RATIO:
                                front_adjustments['is_right_wrist_bent_from_midline'] = True
                    except Exception as e:
                        print(f"Warn R Wrist Dev: {e}")

                    # Trunk side bend
                    v_sh  = r_sh_f  - l_sh_f;  v_hip = r_hip_f - l_hip_f
                    n_sh  = norm(v_sh); n_hip = norm(v_hip)
                    if n_sh > 1e-6 and n_hip > 1e-6:
                        a_sh  = np.arctan2(v_sh[1],  v_sh[0])
                        a_hip = np.arctan2(v_hip[1], v_hip[0])
                        a_diff = (a_sh - a_hip + np.pi) % (2 * np.pi) - np.pi
                        angle_diff_trunk_deg = np.degrees(a_diff)
                        if abs(angle_diff_trunk_deg) > TRUNK_SIDE_BEND_THRESHOLD:
                            front_adjustments['is_trunk_side_bent'] = True   # FIXED name

                    # Neck side bend
                    v_nk = ear_mid_f - sh_mid_f; n_nk = norm(v_nk)
                    if n_nk > 1e-6:
                        cos_nk = np.clip(np.dot(v_nk, np.array([0, -1])) / n_nk, -1.0, 1.0)
                        angle_neck_vert_deg = np.degrees(np.arccos(cos_nk))
                        if angle_neck_vert_deg > NECK_SIDE_BEND_THRESHOLD:
                            front_adjustments['is_neck_side_bent'] = True    # FIXED name

                    # Neck twist (Z depth difference of ears)
                    neck_z_diff = abs(l_ear_lm_f.z - r_ear_lm_f.z)
                    if neck_z_diff > NECK_TWIST_Z_THRESHOLD:
                        front_adjustments['is_neck_twisted'] = True

                    # Trunk twist (Z depth difference of shoulders)
                    trunk_z_diff = abs(l_sh_lm_f.z - r_sh_lm_f.z)
                    if trunk_z_diff > TRUNK_TWIST_Z_THRESHOLD:
                        front_adjustments['is_trunk_twisted'] = True

                    # Knee flexion (for REBA Leg score) ← NEW
                    try:
                        kn_visible = (l_kn_lm_f.visibility > 0.5 and r_kn_lm_f.visibility > 0.5
                                      and l_an_lm_f.visibility > 0.5 and r_an_lm_f.visibility > 0.5)
                        if kn_visible:
                            l_kn_f = np.array([l_kn_lm_f.x * w_front, l_kn_lm_f.y * h_front])
                            r_kn_f = np.array([r_kn_lm_f.x * w_front, r_kn_lm_f.y * h_front])
                            l_an_f = np.array([l_an_lm_f.x * w_front, l_an_lm_f.y * h_front])
                            r_an_f = np.array([r_an_lm_f.x * w_front, r_an_lm_f.y * h_front])

                            # Knee angle at knee joint (hip–knee–ankle); 180 = straight
                            l_knee_ang = calculate_angle_acos(
                                [l_hip_f[0], l_hip_f[1]], [l_kn_f[0], l_kn_f[1]], [l_an_f[0], l_an_f[1]])
                            r_knee_ang = calculate_angle_acos(
                                [r_hip_f[0], r_hip_f[1]], [r_kn_f[0], r_kn_f[1]], [r_an_f[0], r_an_f[1]])

                            l_knee_flex = 180.0 - l_knee_ang
                            r_knee_flex = 180.0 - r_knee_ang
                            max_knee_flex = max(l_knee_flex, r_knee_flex)

                            if max_knee_flex > KNEE_FLEXION_HIGH:
                                front_adjustments['is_knee_flexed_high']     = True
                            elif max_knee_flex > KNEE_FLEXION_MODERATE:
                                front_adjustments['is_knee_flexed_moderate'] = True
                    except Exception as e:
                        print(f"Warn Knee Flex: {e}")

                    # Unilateral stance detection (ankle X spread relative to hip width)
                    try:
                        an_visible = (l_an_lm_f.visibility > 0.5 and r_an_lm_f.visibility > 0.5)
                        if an_visible:
                            l_an_f_x = l_an_lm_f.x * w_front
                            r_an_f_x = r_an_lm_f.x * w_front
                            hip_width = abs(r_hip_f[0] - l_hip_f[0])
                            ankle_spread = abs(r_an_f_x - l_an_f_x)
                            # If ankle spread < 30% of hip width, person is on one leg or feet together
                            if hip_width > 1e-6 and (ankle_spread / hip_width) < 0.30:
                                front_adjustments['is_unilateral_stance'] = True
                    except Exception as e:
                        print(f"Warn Stance: {e}")

                    # Draw front skeleton overlays
                    cv2.line(frame_front_out, tuple(sh_mid_f.astype(int)), tuple(hip_mid_f.astype(int)), (255, 255, 0), 1)
                    cv2.line(frame_front_out, tuple(l_sh_f.astype(int)),   tuple(r_sh_f.astype(int)),   (0, 255, 255), 1)
                    cv2.line(frame_front_out, tuple(l_hip_f.astype(int)),  tuple(r_hip_f.astype(int)),  (0, 255, 255), 1)
                    cv2.line(frame_front_out, tuple(sh_mid_f.astype(int)), tuple(ear_mid_f.astype(int)),(255, 0, 255), 1)
                    cv2.line(frame_front_out, tuple(l_sh_f.astype(int)),   tuple(l_el_f.astype(int)),   (255, 100, 0), 1)
                    cv2.line(frame_front_out, tuple(r_sh_f.astype(int)),   tuple(r_el_f.astype(int)),   (0, 100, 255), 1)

            except (IndexError, Exception) as e:
                print(f"Err Front LMs: {e}")

        # ================================================================
        # REBA Score Calculation — Left Side
        # ================================================================
        if left_results["valid"]:
            try:
                angles_l = left_results["angles"]
                comps = get_reba_component_scores(
                    'left',
                    angles_l.get('upper_arm', 0),
                    angles_l.get('lower_arm', 0),
                    angles_l.get('wrist', None),
                    angles_l.get('neck', 0),
                    angles_l.get('trunk', 0),
                    adj_flags=front_adjustments
                )
                # comps = (UA, LA, WR, NK, TR, LG)
                left_results["scores"] = dict(zip(['UA', 'LA', 'WR', 'NK', 'TR', 'LG'], comps))
                sa = get_table_a_score(comps[3], comps[4], comps[5])  # neck, trunk, legs
                sb = get_table_b_score(comps[0], comps[1], comps[2])  # upper_arm, lower_arm, wrist
                left_results["table_scores"]["A"] = sa if sa is not None else "Err"
                left_results["table_scores"]["B"] = sb if sb is not None else "Err"
                if sa is not None and sb is not None:
                    sc = get_table_c_score(sa, sb)
                    left_results["table_scores"]["C"] = sc if sc is not None else "Err"
                else:
                    left_results["table_scores"]["C"] = "Err"
            except Exception as e:
                print(f"Error Left REBA: {e}"); traceback.print_exc()

        # ================================================================
        # REBA Score Calculation — Right Side
        # ================================================================
        if right_results["valid"]:
            try:
                angles_r = right_results["angles"]
                comps = get_reba_component_scores(
                    'right',
                    angles_r.get('upper_arm', 0),
                    angles_r.get('lower_arm', 0),
                    angles_r.get('wrist', None),
                    angles_r.get('neck', 0),
                    angles_r.get('trunk', 0),
                    adj_flags=front_adjustments
                )
                right_results["scores"] = dict(zip(['UA', 'LA', 'WR', 'NK', 'TR', 'LG'], comps))
                sa = get_table_a_score(comps[3], comps[4], comps[5])
                sb = get_table_b_score(comps[0], comps[1], comps[2])
                right_results["table_scores"]["A"] = sa if sa is not None else "Err"
                right_results["table_scores"]["B"] = sb if sb is not None else "Err"
                if sa is not None and sb is not None:
                    sc = get_table_c_score(sa, sb)
                    right_results["table_scores"]["C"] = sc if sc is not None else "Err"
                else:
                    right_results["table_scores"]["C"] = "Err"
            except Exception as e:
                print(f"Error Right REBA: {e}"); traceback.print_exc()

        # ================================================================
        # Final REBA Score = worst side Table C + Load + Coupling + Activity
        # ================================================================
        final_reba_score = "N/A"
        dominant_side    = "N/A"

        left_c  = left_results["table_scores"]["C"]
        right_c = right_results["table_scores"]["C"]
        l_is_num = isinstance(left_c,  (int, float))
        r_is_num = isinstance(right_c, (int, float))

        if not l_is_num and not r_is_num:
            final_reba_score = "N/A"
            dominant_side    = "Both Invalid"
        else:
            l_val = left_c  if l_is_num else -1
            r_val = right_c if r_is_num else -1
            if l_val >= r_val:
                worst_c       = left_c
                dominant_side = "Left"
            else:
                worst_c       = right_c
                dominant_side = "Right"

            # Apply REBA additional factors
            try:
                final_reba_score = int(worst_c) + REBA_LOAD_SCORE + REBA_COUPLING_SCORE + REBA_ACTIVITY_SCORE
                final_reba_score = max(1, min(final_reba_score, 15))
            except (TypeError, ValueError):
                final_reba_score = "N/A"

        final_action_level = get_reba_action_level(final_reba_score
                                                   if isinstance(final_reba_score, (int, float))
                                                   else None)

        # ================================================================
        # CSV row data
        # ================================================================
        _now = datetime.datetime.now()
        row_data = {
            "Date":          _now.strftime("%Y-%m-%d"),
            "Time":          _now.strftime("%H:%M:%S"),
            "Frame":         rec_frame_count,
            # Left
            "L_UA_Score":    left_results["scores"].get("UA",  "N/A"),
            "L_LA_Score":    left_results["scores"].get("LA",  "N/A"),
            "L_WR_Score":    left_results["scores"].get("WR",  "N/A"),
            "L_NK_Score":    left_results["scores"].get("NK",  "N/A"),
            "L_TR_Score":    left_results["scores"].get("TR",  "N/A"),
            "L_LG_Score":    left_results["scores"].get("LG",  "N/A"),
            "L_A":           left_results["table_scores"].get("A", "N/A"),
            "L_B":           left_results["table_scores"].get("B", "N/A"),
            "L_C":           left_results["table_scores"].get("C", "N/A"),
            # Right
            "R_UA_Score":    right_results["scores"].get("UA", "N/A"),
            "R_LA_Score":    right_results["scores"].get("LA", "N/A"),
            "R_WR_Score":    right_results["scores"].get("WR", "N/A"),
            "R_NK_Score":    right_results["scores"].get("NK", "N/A"),
            "R_TR_Score":    right_results["scores"].get("TR", "N/A"),
            "R_LG_Score":    right_results["scores"].get("LG", "N/A"),
            "R_A":           right_results["table_scores"].get("A", "N/A"),
            "R_B":           right_results["table_scores"].get("B", "N/A"),
            "R_C":           right_results["table_scores"].get("C", "N/A"),
            # Final
            "Final_REBA":    final_reba_score,
            "Dominant_Side": dominant_side,
            "Load_Score":    REBA_LOAD_SCORE,
            "Coupling_Score":REBA_COUPLING_SCORE,
            "Activity_Score":REBA_ACTIVITY_SCORE,
            # Left angles
            "L_Ang_UA":      round(left_results["angles"].get("upper_arm", "N/A"), 2) if isinstance(left_results["angles"].get("upper_arm"), float) else "N/A",
            "L_Ang_LA":      round(left_results["angles"].get("lower_arm", "N/A"), 2) if isinstance(left_results["angles"].get("lower_arm"), float) else "N/A",
            "L_Ang_WR":      round(left_results["angles"].get("wrist",     "N/A"), 2) if isinstance(left_results["angles"].get("wrist"),     float) else "N/A",
            "L_Ang_NK":      round(left_results["angles"].get("neck",      "N/A"), 2) if isinstance(left_results["angles"].get("neck"),      float) else "N/A",
            "L_Ang_TR":      round(left_results["angles"].get("trunk",     "N/A"), 2) if isinstance(left_results["angles"].get("trunk"),     float) else "N/A",
            # Right angles
            "R_Ang_UA":      round(right_results["angles"].get("upper_arm", "N/A"), 2) if isinstance(right_results["angles"].get("upper_arm"), float) else "N/A",
            "R_Ang_LA":      round(right_results["angles"].get("lower_arm", "N/A"), 2) if isinstance(right_results["angles"].get("lower_arm"), float) else "N/A",
            "R_Ang_WR":      round(right_results["angles"].get("wrist",     "N/A"), 2) if isinstance(right_results["angles"].get("wrist"),     float) else "N/A",
            "R_Ang_NK":      round(right_results["angles"].get("neck",      "N/A"), 2) if isinstance(right_results["angles"].get("neck"),      float) else "N/A",
            "R_Ang_TR":      round(right_results["angles"].get("trunk",     "N/A"), 2) if isinstance(right_results["angles"].get("trunk"),     float) else "N/A",
            # Knee angles (front camera)
            "L_Ang_Knee":    round(l_knee_flex, 2),
            "R_Ang_Knee":    round(r_knee_flex, 2),
        }

        # ================================================================
        # Display — Left frame
        # ================================================================
        ls_sc = left_results["table_scores"]
        cv2.putText(frame_left_out,
                    f'L REBA: {ls_sc["C"]}  (A:{ls_sc["A"]} B:{ls_sc["B"]})',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        sx = int(w_left * 0.65); sy = 55; lh = 20; fs = 0.42; clr = (0, 255, 255)
        draw_text_with_background(frame_left_out, "-- Left Angles --", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, fs, clr); sy += lh
        for lbl, key in [("UA", "upper_arm"), ("LA", "lower_arm"), ("WR", "wrist"),
                          ("NK", "neck"),      ("TR", "trunk")]:
            v = left_results["angles"].get(key)
            txt = f"L {lbl}: {v:.1f}" if isinstance(v, float) else f"L {lbl}: N/A"
            draw_text_with_background(frame_left_out, txt, (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, fs, clr); sy += lh
        sy += 4
        draw_text_with_background(frame_left_out, "-- Left Scores --", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, fs, clr); sy += lh
        for abbr in ['UA', 'LA', 'WR', 'NK', 'TR', 'LG']:
            v = left_results["scores"].get(abbr, 'N/A')
            draw_text_with_background(frame_left_out, f"L {abbr}: {v}", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, fs, clr); sy += lh

        cv2.putText(frame_left_out, f"FPS: {1.0/(time.time()-prev_time+1e-9):.1f}",
                    (10, h_left - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # ================================================================
        # Display — Right frame
        # ================================================================
        rs_sc = right_results["table_scores"]
        cv2.putText(frame_right_out,
                    f'R REBA: {rs_sc["C"]}  (A:{rs_sc["A"]} B:{rs_sc["B"]})',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        sx2 = 10; sy2 = 55
        draw_text_with_background(frame_right_out, "-- Right Angles --", (sx2, sy2), cv2.FONT_HERSHEY_SIMPLEX, fs, clr); sy2 += lh
        for lbl, key in [("UA", "upper_arm"), ("LA", "lower_arm"), ("WR", "wrist"),
                          ("NK", "neck"),      ("TR", "trunk")]:
            v = right_results["angles"].get(key)
            txt = f"R {lbl}: {v:.1f}" if isinstance(v, float) else f"R {lbl}: N/A"
            draw_text_with_background(frame_right_out, txt, (sx2, sy2), cv2.FONT_HERSHEY_SIMPLEX, fs, clr); sy2 += lh
        sy2 += 4
        draw_text_with_background(frame_right_out, "-- Right Scores --", (sx2, sy2), cv2.FONT_HERSHEY_SIMPLEX, fs, clr); sy2 += lh
        for abbr in ['UA', 'LA', 'WR', 'NK', 'TR', 'LG']:
            v = right_results["scores"].get(abbr, 'N/A')
            draw_text_with_background(frame_right_out, f"R {abbr}: {v}", (sx2, sy2), cv2.FONT_HERSHEY_SIMPLEX, fs, clr); sy2 += lh

        cv2.putText(frame_right_out, f"FPS: {1.0/(time.time()-prev_time+1e-9):.1f}",
                    (10, h_right - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # ================================================================
        # Display — Front frame (adjustments + final score)
        # ================================================================
        ax = 10; ay = 28; alh = 18; afs = 0.4
        lines_front = [
            f"L lower arm across midline: {front_adjustments['is_left_lower_arm_across_midline']}",
            f"R lower arm across midline: {front_adjustments['is_right_lower_arm_across_midline']}",
            f"L lower arm abducted: {front_adjustments['is_left_lower_arm_abducted']} ({l_low_abd_diff:.0f}px)",
            f"R lower arm abducted: {front_adjustments['is_right_lower_arm_abducted']} ({r_low_abd_diff:.0f}px)",
            f"---",
            f"L arm abducted: {front_adjustments['is_left_arm_abducted']} ({angle_left_shoulder_deg:.1f}deg)",
            f"R arm abducted: {front_adjustments['is_right_arm_abducted']} ({angle_right_shoulder_deg:.1f}deg)",
            f"L shoulder raised: {front_adjustments['is_left_shoulder_raised']} ({dist_l_ear_sh:.0f}px)",
            f"R shoulder raised: {front_adjustments['is_right_shoulder_raised']} ({dist_r_ear_sh:.0f}px)",
            f"---",
            f"Neck twisted: {front_adjustments['is_neck_twisted']} (z={neck_z_diff:.3f})",
            f"Neck side bent: {front_adjustments['is_neck_side_bent']} ({angle_neck_vert_deg:.1f}deg)",
            f"Trunk twisted: {front_adjustments['is_trunk_twisted']} (z={trunk_z_diff:.3f})",
            f"Trunk side bent: {front_adjustments['is_trunk_side_bent']} ({angle_diff_trunk_deg:.1f}deg)",
            f"---",
            f"Knee flex moderate (>{KNEE_FLEXION_MODERATE}): {front_adjustments['is_knee_flexed_moderate']} L={l_knee_flex:.0f} R={r_knee_flex:.0f}",
            f"Knee flex high (>{KNEE_FLEXION_HIGH}):     {front_adjustments['is_knee_flexed_high']}",
            f"---",
            f"Load:{REBA_LOAD_SCORE}  Coupling:{REBA_COUPLING_SCORE}  Activity:{REBA_ACTIVITY_SCORE}  [L/C/A keys]",
            f"Frame: {rec_frame_count if recording else frame_count}  {'[REC]' if recording else ''}",
        ]
        for line in lines_front:
            draw_text_with_background(frame_front_out, line, (ax, ay), cv2.FONT_HERSHEY_SIMPLEX, afs, (0, 255, 255)); ay += alh

        # Final REBA score box (bottom-centre of front view)
        box_w = 350; box_h = 65
        bx = w_front // 2 - box_w // 2
        by = h_front - box_h - 10
        cv2.rectangle(frame_front_out, (bx, by), (bx + box_w, by + box_h), (0, 0, 0), -1)
        cv2.putText(frame_front_out,
                    f"FINAL REBA: {final_reba_score}  ({dominant_side})",
                    (bx + 8, by + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame_front_out, final_action_level,
                    (bx + 8, by + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.putText(frame_front_out, f"FPS: {1.0/(time.time()-prev_time+1e-9):.1f}",
                    (10, h_front - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Combine all three views horizontally (resize to common height to prevent hconcat crash)
        target_h = min(h_left, h_front, h_right)
        combined1 = cv2.hconcat([_resize_h(frame_left_out, target_h),
                                  _resize_h(frame_front_out, target_h),
                                  _resize_h(frame_right_out, target_h)])

        # ================================================================
        # Recording (P to start)
        # ================================================================
        if recording:
            if video_writer is None:
                h_c, w_c = combined1.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
                video_writer = cv2.VideoWriter(VIDEO_OUT_PATH, fourcc, VIDEO_FPS, (w_c, h_c))
                if not video_writer.isOpened():
                    print("Error: VideoWriter open failed, recording disabled.")
                    video_writer = None

            if csv_writer is None:
                is_new = not os.path.exists(CSV_OUT_PATH)
                csv_fh = open(CSV_OUT_PATH, "a", newline="", encoding="utf-8")
                csv_fieldnames = list(row_data.keys())
                csv_writer = csv.DictWriter(csv_fh, fieldnames=csv_fieldnames)
                if is_new:
                    csv_writer.writeheader(); csv_fh.flush()

            if video_writer is not None and video_writer.isOpened():
                video_writer.write(combined1)

            # Write CSV row if at least one side produced valid scores
            l_valid_score = isinstance(left_results["table_scores"]["C"], (int, float))
            r_valid_score = isinstance(right_results["table_scores"]["C"], (int, float))
            if (l_valid_score or r_valid_score) and csv_writer is not None:
                csv_writer.writerow(row_data)
                csv_fh.flush()

            rec_frame_count += 1

        cv2.imshow("Dynamic REBA System", combined1)

        # Close if window X button clicked
        if cv2.getWindowProperty("Dynamic REBA System", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed.")
            break

    except Exception as main_loop_error:
        print(f"Error in main loop: {main_loop_error}")
        traceback.print_exc()
        time.sleep(0.1)

    # ================================================================
    # Keyboard controls
    # ================================================================
    key = cv2.waitKey(5) & 0xFF

    if key == ord('p'):
        recording = True
        rec_frame_count = 0
        print(f"Recording started → {VIDEO_OUT_PATH}")
        print(f"CSV logging started → {CSV_OUT_PATH}")

    elif key == ord('l'):
        REBA_LOAD_SCORE = (REBA_LOAD_SCORE + 1) % 4
        print(f"Load score set to {REBA_LOAD_SCORE}  (0=<5kg 1=5-10kg 2=>10kg 3=shock/rapid)")

    elif key == ord('c'):
        REBA_COUPLING_SCORE = (REBA_COUPLING_SCORE + 1) % 4
        print(f"Coupling score set to {REBA_COUPLING_SCORE}  (0=good 1=fair 2=poor 3=unacceptable)")

    elif key == ord('a'):
        REBA_ACTIVITY_SCORE = (REBA_ACTIVITY_SCORE + 1) % 4
        print(f"Activity score set to {REBA_ACTIVITY_SCORE}  (0=none 1=static 2=+repeated 3=+rapid)")

    elif key == ord('q') or key == 27:
        print("Exit key pressed.")
        break

    frame_count += 1
    prev_time = time.time()   # reset timer at end of iteration for accurate FPS

# ================================================================
# Release resources
# ================================================================
print("Releasing resources...")

if video_writer is not None:
    try:
        video_writer.release()
        print(f"Video saved: {VIDEO_OUT_PATH}")
    except Exception as e:
        print(f"VideoWriter release error: {e}")

try:
    if csv_fh is not None and not csv_fh.closed:
        csv_fh.flush()
        try: os.fsync(csv_fh.fileno())
        except Exception: pass
        csv_fh.close()
        print(f"CSV saved: {CSV_OUT_PATH}")
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
hands_front.close()
print("Done.")
