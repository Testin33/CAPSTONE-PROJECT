# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dynamic REBA (Rapid Entire Body Assessment) system that uses MediaPipe pose estimation across three simultaneous camera feeds to compute real-time ergonomic risk scores. The project is a capstone study comparing ergonomic and non-ergonomic workstations in a human-robot collaborative assembly line.

## Running the System

```bash
python "Dynamic RULA System(v6).py"
```

Requires three USB cameras connected. Camera indices are configured at the top of the file:
- `FRONT_CAMERA_INDEX = 0` — front view
- `LEFT_SIDE_CAMERA_INDEX = 1` — left side view
- `RIGHT_SIDE_CAMERA_INDEX = 2` — right side view

**Dependencies:** `opencv-python`, `mediapipe`, `numpy`, `pandas`

## Keyboard Controls (while running)

| Key | Action |
|-----|--------|
| `P` | Start recording (MP4 + CSV output) |
| `L` | Cycle load score (0–3) |
| `C` | Cycle coupling score (0–3) |
| `A` | Cycle activity score (0–3) |
| `Q` / `ESC` | Quit |

## Outputs

Both files are saved in the same directory as the script, named `YYYYMMDD-HHMMSS_REBA.mp4` and `.csv`.

## Architecture

The entire system is a single-file script (`Dynamic RULA System(v6).py`) structured in these layers:

### 1. Configuration Constants (top of file)
All tunable thresholds for REBA scoring are global constants: `TRUNK_FLEXION_BINS`, `NECK_FLEXION_THRESHOLD`, `ARM_ABDUCTION_THRESHOLD`, etc. Manual REBA factors (`REBA_LOAD_SCORE`, `REBA_COUPLING_SCORE`, `REBA_ACTIVITY_SCORE`) are toggled at runtime via keyboard.

### 2. REBA Lookup Tables
Three pandas DataFrames (`table_a`, `table_b`, `table_c`) encode the standard REBA scoring matrices. Access functions `get_table_a_score()`, `get_table_b_score()`, `get_table_c_score()` take integer component scores and return table values with clamping via `_to_valid_int_score()`.

### 3. Component Score Calculator (`get_reba_component_scores()`)
Takes raw joint angles from a side camera plus boolean `adj_flags` from the front camera, and returns the 6 REBA component scores: `(upper_arm, lower_arm, wrist, neck, trunk, legs)`. The `negate` flag handles sign convention differences between left and right views.

### 4. Main Loop (three-camera pipeline per frame)
Each frame processes three camera feeds independently:
- **Left side camera** → left-body `upper_arm`, `lower_arm`, `neck`, `trunk` angles + optional wrist angle from MediaPipe Hands
- **Right side camera** → same for right body
- **Front camera** → populates `front_adjustments` dict (15+ boolean flags for abduction, twist, side bend, shoulder raise, knee flexion, stance)

The `front_adjustments` dict is passed to `get_reba_component_scores()` for both sides. Final REBA score = worst of left/right Table C score + load + coupling + activity, clamped to 1–15.

### 5. REBA Score → Action Level
`get_reba_action_level()` maps the final integer score to the 5-level REBA action level string displayed on the front view overlay.

### Sign Convention for Angles
`calculate_angle_with_sign()` returns signed angles (-180 to 180). For the left side, angles are negated inside `get_reba_component_scores()` (via `negate = (side == 'left')`) so that flexion is always positive regardless of which camera view is used. Trunk flexion uses `180 - abs(tr_ref)` so that 0° = upright.
