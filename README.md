# Dynamic REBA System

**Dynamic Rapid Entire Body Assessment (REBA) System Using Machine Learning**  
A capstone study comparing ergonomic and non-ergonomic workstations in a Human-Robot Collaborative Assembly Line.

---

## Overview

This system performs real-time ergonomic risk assessment using the REBA methodology. It processes three simultaneous USB camera feeds (front, left side, right side) with MediaPipe pose estimation to compute body joint angles and generate a live REBA score per frame.

---

## Requirements

```bash
pip install opencv-python mediapipe numpy pandas
```

Three USB cameras are required.

---

## Running

```bash
python "Dynamic REBA System(v6).py"
```

---

## Camera Setup

| Camera | Index | Purpose |
|--------|-------|---------|
| Front  | 2     | Abduction, twist, side bend, knee detection |
| Left side | 1  | Left body angles (trunk, neck, upper arm, lower arm, wrist) |
| Right side | 0 | Right body angles |

Indices can be changed at the top of the script:
```python
LEFT_SIDE_CAMERA_INDEX  = 1
RIGHT_SIDE_CAMERA_INDEX = 0
FRONT_CAMERA_INDEX      = 2
```

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `P` | Start recording (MP4 + CSV) |
| `L` | Cycle Load score (0–3) |
| `C` | Cycle Coupling score (0–3) |
| `A` | Cycle Activity score (0–3) |
| `Q` / `ESC` | Quit |

**Load (L):** 0 = <5 kg · 1 = 5–10 kg · 2 = >10 kg · 3 = shock/rapid force  
**Coupling (C):** 0 = good grip · 1 = fair · 2 = poor · 3 = unacceptable  
**Activity (A):** 0 = none · 1 = static >1 min · 2 = repeated >4/min · 3 = rapid/unstable

---

## Output Files

Both files are saved in the script directory, named by recording timestamp:

```
YYYYMMDD-HHMMSS_REBA.mp4
YYYYMMDD-HHMMSS_REBA.csv
```

### CSV Columns

| Column group | Columns |
|---|---|
| Timestamp | `Date`, `Time`, `Frame` |
| Left scores | `L_UA_Score`, `L_LA_Score`, `L_WR_Score`, `L_NK_Score`, `L_TR_Score`, `L_LG_Score` |
| Left table scores | `L_A`, `L_B`, `L_C` |
| Right scores | `R_UA_Score` … `R_C` (same structure) |
| Final | `Final_REBA`, `Dominant_Side`, `Load_Score`, `Coupling_Score`, `Activity_Score` |
| Left angles (°) | `L_Ang_UA`, `L_Ang_LA`, `L_Ang_WR`, `L_Ang_NK`, `L_Ang_TR` |
| Right angles (°) | `R_Ang_UA`, `R_Ang_LA`, `R_Ang_WR`, `R_Ang_NK`, `R_Ang_TR` |
| Knee angles (°) | `L_Ang_Knee`, `R_Ang_Knee` |

---

## REBA Score Interpretation

| Score | Action Level | Risk |
|-------|-------------|------|
| 1 | 0 | Negligible |
| 2–3 | 1 | Low — change may be needed |
| 4–7 | 2 | Medium — further investigation |
| 8–10 | 3 | High — investigate and change soon |
| 11–15 | 4 | Very High — implement change immediately |

---

## Architecture

The system is a single-file script structured in layers:

1. **Configuration** — all REBA thresholds and camera indices as constants at the top
2. **REBA lookup tables** — `table_a`, `table_b`, `table_c` (standard REBA matrices)
3. **Score calculator** — `get_reba_component_scores()` takes joint angles + front-camera boolean flags, returns 6 component scores
4. **Main loop** — reads 3 camera feeds per frame, computes left/right REBA independently, takes the worst side as the final score
5. **Front camera** — detects 15+ adjustment flags (abduction, twist, side bend, shoulder raise, knee flexion, stance) passed to both side score calculations
