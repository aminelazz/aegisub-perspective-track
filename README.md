# Aegisub 3D Perspective Planar Tracker

An Aegisub Automation 4 Lua macro (`perspective_tracker.lua`) paired with a CPU-optimized Python planar tracker and exact libass 3D perspective solver (`perspective_solver.py`).

It tracks a 4-corner planar sign across video frames using OpenCV (Pyramidal Lucas-Kanade Optical Flow, ORB + RANSAC Homography, or Hybrid) and converts the 2D quadrilateral warp into exact frame-by-frame ASS 3D transform tags (`\an5\pos(...)\org(...)\fscx...\fscy...\frx...\fry...\frz...`).

---

## Files

- `perspective_tracker.lua` — Aegisub Automation 4 macro with GUI, `\clip` validation, JSON config persistence, and frame-by-frame subtitle insertion.
- `perspective_solver.py` — CPU-optimized OpenCV planar tracker and exact homography-to-libass 3D perspective solver (also usable standalone via CLI).
- `requirements.txt` — Python dependencies (`opencv-python`, `numpy`).

---

## Installation

### 1. Install Python Dependencies
Ensure Python 3.8+ is installed and `python` (Windows) or `python3` (Linux/macOS) is available on your system `PATH`.

```bash
pip install -r requirements.txt
```

### 2. Install the Aegisub Macro
1. Copy `perspective_tracker.lua` into your Aegisub Automation autoload folder:
   - **Windows**: `%APPDATA%\Aegisub\automation\autoload\`
   - **Linux**: `~/.aegisub/automation/autoload/` (or `$XDG_CONFIG_HOME/Aegisub/automation/autoload/`)
   - **macOS**: `~/Library/Application Support/Aegisub/automation/autoload/`
2. Place `perspective_solver.py` either:
   - In the same `automation/autoload/` directory next to `perspective_tracker.lua` (auto-detected automatically), **or**
   - In any directory of your choice and select it directly from the GUI using the **`Choose .py Script...`** button.
3. In Aegisub, click **Automation → Reload Autoload Scripts** (or restart Aegisub). The macro appears under **Automation → 3D Perspective Planar Tracker**.

### 3. Configuration File Location
Settings (including the selected `perspective_solver.py` path and last-used tracking parameters) are automatically loaded on launch and saved whenever you click **Track & Solve** or **Choose .py Script...**:
- **Windows**: `%APPDATA%\aegisub\config\perspective_tracker.json`
- **Linux**: `$XDG_CONFIG_HOME/aegisub/config/perspective_tracker.json` (defaults to `~/.config/aegisub/config/perspective_tracker.json`)

> **Tip:** If your Python binary is not on `PATH` or has a custom name (e.g. `py -3` or a virtualenv path), you can edit `"python_executable"` inside `perspective_tracker.json`.

---

## Guide Line Workflow in Aegisub

1. **Open your video** in Aegisub (`Video → Open Video...`).
2. **Create a guide line** spanning the start frame to end frame of the sign appearance.
3. **Style your sign text** on the guide line as desired (`\fn`, `\fs`, `\1c`, `\3c`, `\bord`, `\shad`, `\blur`, `\alpha`, etc.).
4. **Seek to the reference frame**:
   - **Forward tracking**: seek to the **first frame** of the guide line.
   - **Backward tracking**: seek to the **last frame** of the guide line.
5. **Draw the 4-corner `\clip`**:
   - Select Aegisub's **Vector Clip** tool (`\clip(m x1 y1 l x2 y2 x3 y3 x4 y4)`).
   - Click the **4 corners of the planar sign in clockwise order**:
     1. **Top-Left (TL)**
     2. **Top-Right (TR)**
     3. **Bottom-Right (BR)**
     4. **Bottom-Left (BL)**
6. Select the guide line and run **Automation → 3D Perspective Planar Tracker**.
7. Configure the dialog options and click **Track & Solve**.
8. The macro inserts exact single-frame dialogue lines directly above your guide line and comments out the original guide line.

---

## GUI Parameters

| Parameter | Options / Range | Description |
|---|---|---|
| **Tracking Method** | `Hybrid Optical Flow + Feature Homography (Recommended)`<br>`Dense Corner Lucas-Kanade Optical Flow`<br>`ORB Feature Matching + RANSAC` | **Hybrid** combines forward-backward filtered dense grid + Shi-Tomasi pyramidal Lucas-Kanade flow with CLAHE-enhanced ORB descriptor matches into a unified RANSAC + least-squares refined homography.<br>**Dense LK** uses dense perspective grid + corner neighborhoods + Shi-Tomasi points with forward-backward consistency checking.<br>**ORB + RANSAC** uses mutual kNN Hamming matches inside the quad ROI (with automatic LK augmentation on low-texture regions). |
| **Tracking Direction** | `Forward (Start Frame -> End Frame)`<br>`Backward (End Frame -> Start Frame)` | Choose whether your initial `\clip` was drawn on the line's first frame (track forward) or last frame (track backward). Backward mode decodes sequential chunks to avoid inaccurate keyframe seeking. |
| **Aspect Ratio Mode** | `Auto-Estimate Plane Aspect Ratio`<br>`Keep Original Text Aspect Ratio` | **Auto-Estimate** recovers the true un-foreshortened 3D plane aspect ratio using your Virtual Focal Length (`fov_factor`), adjusting `\fscx`/`\fscy` so the text sits naturally on tilted surfaces.<br>**Keep Original** keeps `100% : 100%` aspect ratio on the reference frame while tracking scale and perspective deformation across the shot. |
| **Virtual Focal Length (FOV factor)** | Float, default `1.0` | Focal length relative to `PlayResX` (`1.0` ≈ standard 50mm lens). Used to recover true 3D foreshortening proportions on tilted planes when converting to libass perspective. |
| **Smoothing Window (Frames)** | Integer `0`–`10`, default `0` | Applies a symmetric Gaussian-weighted moving average across the tracked corner trajectories (while pinning the initial reference frame) to damp sub-pixel micro-jitter. |
| **Python Solver (.py) Path** | File path + `Choose .py Script...` | Path to `perspective_solver.py`. Clicking **Choose .py Script...** opens a file picker and saves the path to `perspective_tracker.json`. |

---

## How the 3D Decomposition Works

1. **Planar Homography Tracking (Step A)**:
   On each frame step, dense interior/edge grid samples, corner neighborhoods, Shi-Tomasi keypoints, and/or ORB features inside the quadrilateral are matched to the next frame and filtered via forward-backward consistency + RANSAC homography estimation (`cv2.findHomography`), projecting the 4 corners cleanly even under partial occlusion or motion blur.
2. **Exact libass Projection Math (Step B)**:
   - libass projects text around `\org(ox, oy)` using a fixed focal plane distance `z = 312.5` and rotation order `\frz → \frx → \fry`.
   - For each frame's 4 corners `(TL, TR, BR, BL)`, the solver solves a singularity-free rationalized quadratic equation for the optimal `\org(ox, oy)` closest to the quad center where the reconstructed 3D basis vectors are orthogonal (eliminating 3D skew without requiring `\fax`).
   - It extracts exact `\frx`, `\fry`, `\frz`, `\pos(px, py)`, and `\fscx`/`\fscy` values that reproject back onto the 4 tracked 2D corners with sub-pixel precision, while unwrapping Euler angles continuously across the `±180°` boundary.
   - Old positioning/transform/clip tags (`\pos`, `\org`, `\fr*`, `\fsc*`, `\fa*`, `\an*`, `\clip`, `\iclip`) are stripped cleanly while preserving all user styling tags (`\fn`, `\fs`, `\1c`, `\3c`, `\bord`, `\shad`, `\blur`, `\alpha`, etc.).

---

## Troubleshooting & Edge Cases

- **Rectangular `\clip(x1,y1,x2,y2)`**: Not supported because it only defines 2 corners. Switch to Aegisub's **Vector Clip** tool and mark all 4 corners (`m x1 y1 l x2 y2 x3 y3 x4 y4`).
- **Non-4-point `\clip` or self-intersecting quad**: The script verifies that exactly 4 distinct convex corners exist. Mark corners in perimeter order (`TL → TR → BR → BL`). Counter-clockwise clicks are automatically normalized to clockwise.
- **Dummy video loaded**: Tracking requires real video frames. Open an actual video via `Video → Open Video...`.
- **Missing Python or OpenCV**: Ensure `python` (or `python3`) works in your terminal and `pip install -r requirements.txt` succeeded.
