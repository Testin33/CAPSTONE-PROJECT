# Dynamic REBA System

**Dynamic Rapid Entire Body Assessment (REBA) System Using Machine Learning**
A capstone study comparing ergonomic and non-ergonomic workstations in a human-robot collaborative assembly line.

---

## Overview

This system performs **real-time ergonomic risk scoring** using three simultaneous USB camera feeds and MediaPipe pose estimation. It computes REBA scores frame-by-frame to evaluate worker posture during assembly tasks, enabling a data-driven comparison between ergonomic and non-ergonomic workstation setups.

---

## Requirements

```bash
pip install opencv-python mediapipe numpy pandas
```

Three USB cameras must be connected before running.

---

## Running the System

```bash
python "Dynamic REBA System(v6).py"
```

Camera indices are configured at the top of the file:

| Constant | Default | View |
|---|---|---|
| `FRONT_CAMERA_INDEX` | `0` | Front |
| `LEFT_SIDE_CAMERA_INDEX` | `1` | Left side |
| `RIGHT_SIDE_CAMERA_INDEX` | `2` | Right side |

MediaPipe model files are **auto-downloaded** on first run.

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `P` | Start recording (MP4 + CSV output) |
| `L` | Cycle load score (0–3) |
| `C` | Cycle coupling score (0–3) |
| `A` | Cycle activity score (0–3) |
| `Q` / `ESC` | Quit |

---

## Outputs

Pressing `P` starts recording. Two files are saved in the same directory as the script:

- `YYYYMMDD-HHMMSS_REBA.mp4` — combined video of all three camera views
- `YYYYMMDD-HHMMSS_REBA.csv` — frame-by-frame REBA scores and joint angles

The CSV includes: `Date`, `Time`, `Frame`, left/right component scores (upper arm, lower arm, wrist, neck, trunk, legs), Table A/B/C scores, final REBA score, dominant side, and manual factors (load, coupling, activity).

---

## How It Works

### Three-Camera Pipeline

Each frame is processed across three feeds simultaneously:

- **Left side camera** → left-body joint angles (upper arm, lower arm, neck, trunk, wrist)
- **Right side camera** → same for the right body
- **Front camera** → detects postural adjustments: abduction, lateral twist, side bend, shoulder raise, knee flexion, stance

The front camera flags are passed into the REBA component scorer for both sides. The **final REBA score** is the worst of left/right Table C scores plus load, coupling, and activity — clamped to 1–15.

### REBA Scoring

Follows the standard REBA methodology:

1. **Table A** — trunk, neck, and legs → Score A
2. **Table B** — upper arm, lower arm, and wrist → Score B
3. **Table C** — combines Score A + Score B → Score C
4. **Final score** = Score C + load + coupling + activity

### Action Levels

| Score | Action Level |
|-------|-------------|
| 1 | Negligible risk |
| 2–3 | Low risk |
| 4–7 | Medium risk |
| 8–10 | High risk |
| 11–15 | Very high risk |

---

## File Organizer

`organizer.py` classifies recorded files by participant and task after a session.

**Step 1 — after recording one participant (3 tasks):**
```bash
python organizer.py --tasks 3
```
Creates a sample folder: `1_20260327_145100/`

**Step 2 — distribute files into task folders:**
```bash
python organizer.py --distribute --task-folders "1_task" "2_task" "3_task"
```
Copies the most recent recording from each sample into `1_task/`, the second most recent into `2_task/`, and so on.
