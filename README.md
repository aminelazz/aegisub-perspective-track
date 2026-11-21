# Aegisub 3D Perspective & Non-Rigid Mesh Tracker

An Aegisub Automation 4 Lua macro (`perspective_tracker.lua`) paired with a CPU-optimized Python planar & non-rigid mesh solver (`perspective_solver.py`) and an optional Google Colab AI notebook (`colab_ai_mesh_tracker.ipynb`).

It supports:
1. **Planar 3D Perspective Tracking (CPU)** — Tracks a 4-corner planar sign via OpenCV (Lucas-Kanade Optical Flow, ORB + RANSAC Homography, or Hybrid) into exact frame-by-frame ASS 3D tags (`\an5\pos\org\fscx\fscy\frx\fry\frz`).
2. **Non-Rigid Bending / Distortion / Folding Tracking (CPU Mesh or Google Colab CoTracker3 AI)** — Tracks a dense 2D grid across curved, waving, bending, or folding surfaces and reproduces the motion via **one selectable ASS distortion mode** (Option A, B, or C), with direct `.ttf`/`.otf`/`.ttc` font selection (compatible with NexusFont) and **Arabic Text** shaping/RTL support.

---

## Files

- `perspective_tracker.lua` — Aegisub Automation 4 macro with GUI, `\clip` validation, JSON config persistence, and frame-by-frame insertion.
- `perspective_solver.py` — CPU planar & non-rigid mesh tracker, AI mesh JSON importer, HarfBuzz/FreeType glyph outline warper + Arabic Presentation Forms-B reshaper, and 3D ASS solver (also usable standalone via CLI).
- `colab_ai_mesh_tracker.ipynb` — Google Colab notebook running Meta AI's **CoTracker3** (`cotracker3_offline`) on a free T4 GPU for complex deforming/occluded signs, exporting `mesh_track.json`.
- `requirements.txt` — Python dependencies (`opencv-python`, `numpy`, `freetype-py`, `fonttools`, `uharfbuzz`).

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
   - In any folder and pick it via the **`Choose .py Script...`** button inside the GUI.
3. In Aegisub, click **Automation → Reload Autoload Scripts** (or restart Aegisub). The macro appears under **Automation → 3D Perspective Planar Tracker**.

### 3. Configuration File Location
Settings (solver path, Colab mesh JSON path, selected font file path, Arabic Text toggle, and tracking options) are loaded on launch and saved automatically when you click **Track & Solve**, **Choose Font...**, **Choose Mesh JSON...**, or **Choose .py Script...**:
- **Windows**: `%APPDATA%\aegisub\config\perspective_tracker.json`
- **Linux**: `$XDG_CONFIG_HOME/aegisub/config/perspective_tracker.json` (defaults to `~/.config/aegisub/config/perspective_tracker.json`)

---

## Workflows

### A. CPU Planar or CPU Non-Rigid Mesh Workflow
1. Open your video in Aegisub (`Video → Open Video...`).
2. Create a guide line spanning the sign's start frame to end frame and style your text (`\fn`, `\fs`, `\1c`, `\3c`, `\bord`, `\shad`, `\blur`, `\alpha`, etc.).
3. Seek to the reference frame (**first frame** for Forward tracking, **last frame** for Backward tracking).
4. Draw a 4-corner vector `\clip(m x1 y1 l x2 y2 x3 y3 x4 y4)` in clockwise order (`TL → TR → BR → BL`).
5. Select the guide line, open **Automation → 3D Perspective Planar Tracker**, choose a tracking method:
   - **Planar 3D methods**: `Hybrid Optical Flow + Feature Homography (Recommended)`, `Dense Corner Lucas-Kanade Optical Flow`, or `ORB Feature Matching + RANSAC`.
   - **CPU Non-Rigid Mesh**: `CPU Non-Rigid Mesh Optical Flow (Bending / Distort)` — adjust `CPU Mesh Grid (Cols x Rows)` (default `10 x 6`), select your single **Distortion Mode** (Option A, B, or C), click **`Choose Font...`** (required for Option A & B; works with uninstalled NexusFont fonts), and check **`Arabic Text`** if translating into Arabic.
6. Click **Track & Solve**.

### B. Complex Distortion / Occlusion AI Workflow (Google Colab CoTracker3)
For blank/low-texture paper, heavy curling/crumpling, waving cloth, or temporary occlusions (e.g., a hand turning a page):
1. Upload `colab_ai_mesh_tracker.ipynb` to [Google Colab](https://colab.research.google.com/) and switch runtime to **T4 GPU** (**Runtime → Change runtime type → T4 GPU**).
2. Upload your trimmed video clip (or mount Google Drive), paste your `\clip(...)` coordinates from Aegisub, set `PLAY_RES_X` / `PLAY_RES_Y` / frame range, and run all cells.
3. Download the generated `mesh_track.json`.
4. Back in Aegisub, select your styled guide line (covering the matching frame count or exact frame numbers), run **3D Perspective Planar Tracker**, click **`Choose Mesh JSON...`** to load `mesh_track.json` (this also selects `AI Mesh Import (Google Colab CoTracker3 JSON)` automatically), select your **Distortion Mode** + font file (and **`Arabic Text`** if needed), and click **Track & Solve**.

---

## Distortion Output Modes (CPU Mesh & AI Mesh)

Choose **one** output mode from the **Distortion Mode (Mesh / AI)** dropdown:

| Mode | Output `Effect` Tag | How It Works | Best For |
|---|---|---|---|
| **Option A: Vector Drawing Mesh Warp (`\p4` Outlines)** *(Default)* | `mesh_vector_p4` | Loads your chosen `.ttf`/`.otf`/`.ttc` directly (no system installation required), shapes text via HarfBuzz (`uharfbuzz`) + FreeType, subdivides curves, and warps every vertex through the tracked mesh into ASS `\p4` (`0.125 px` sub-pixel) drawings preserving colors, borders, shadows, blur, and alpha. | True arbitrary bending, cursive Arabic signs, uninstalled/NexusFont fonts, wavy flags, and mid-letter folds. |
| **Option B: Per-Character 3D Tags (Native Font)** | `mesh_per_char` | Splits text into individual characters aligned on the typographic baseline and solves local per-character `\pos\org\fscx\fscy\frx\fry\frz` tags from surrounding mesh cells. When **Arabic Text** is checked, automatically converts logical Arabic into Presentation Forms-B in visual RTL order (e.g., `"صباح الخير"` → `"ﺮﻴﺨﻟﺍ ﺡﺎﺒﺻ"`). | Gently curving text or waving banners with lightweight native font rendering. |
| **Option C: Multi-Plane Fold Split + Dynamic `\clip`** | `mesh_fold_p1`, `mesh_fold_p2` | Automatically detects the optimal vertical or horizontal fold crease across the mesh, fits two independent 3D panel homographies, and generates two native-font layers dynamically clipped along the tracked crease line. Keeps logical Arabic intact for libass native rendering. | Crisp 2-panel paper folds, greeting cards, and book page turns. |

---

## GUI Parameters

| Parameter | Options / Range | Description |
|---|---|---|
| **Tracking Method** | Planar (`Hybrid` / `LK` / `ORB`), `CPU Non-Rigid Mesh Optical Flow`, `AI Mesh Import` | Choose single-plane homography, CPU non-rigid mesh flow, or imported Colab CoTracker3 mesh JSON. |
| **Tracking Direction** | `Forward` / `Backward` | Tracks forward from the start frame or backward from the end frame. |
| **Aspect Ratio Mode** | `Auto-Estimate Plane Aspect Ratio` / `Keep Original Text Aspect Ratio` | Adjusts `\fscx`/`\fscy` proportions for 3D foreshortening using `fov_factor`, or keeps 1:1 text scale on the reference frame. |
| **Virtual Focal Length (FOV factor)** | Float, default `1.0` | Focal length relative to `PlayResX` (`1.0` ≈ standard 50mm lens). |
| **Smoothing Window (Frames)** | Integer `0`–`10`, default `0` | Symmetric shrinking-radius Gaussian smoothing to reduce micro-jitter without endpoint lag. |
| **CPU Mesh Grid (Cols x Rows)** | Integers `3`–`24` (default `10 x 6`) | Grid density for CPU Non-Rigid Mesh tracking. |
| **Distortion Mode (Mesh / AI)** | Dropdown (Option A / Option B / Option C) | Single-choice distortion output mode for CPU Non-Rigid Mesh & AI Mesh Import. |
| **Arabic Text** | Checkbox (default unchecked) | Automatically joins Arabic characters (HarfBuzz for Option A; Presentation Forms-B + visual RTL order for Option B) with a double-inversion guard if already converted. |
| **Font File (.ttf/.otf/.ttc)** | File path + `Choose Font...` | Direct font file path used by Option A and Option B (supports uninstalled / NexusFont fonts). |
| **AI Mesh JSON (Colab)** | File path + `Choose Mesh JSON...` | Path to `mesh_track.json` exported from `colab_ai_mesh_tracker.ipynb`. |
| **Python Solver (.py) Path** | File path + `Choose .py Script...` | Path to `perspective_solver.py`. |

---

## Troubleshooting & Edge Cases

- **Font mismatch with NexusFont / uninstalled fonts**: Always use **`Choose Font...`** to pick the `.ttf` / `.otf` / `.ttc` file directly. Option A (`\p4` Vector Warp) renders the exact outlines from that file even if the font isn't permanently installed in Windows/macOS/Linux.
- **Arabic letters separated or reversed**: Check **`Arabic Text`**. Note that if your text was already converted with an online Presentation Forms-B tool, the solver automatically detects it and avoids double-inverting it.
- **Rectangular `\clip(x1,y1,x2,y2)`**: Not supported for tracking because it only specifies 2 corners. Use Aegisub's **Vector Clip** tool (`m x1 y1 l x2 y2 x3 y3 x4 y4`).
- **Frame count mismatch with AI Mesh JSON**: Either match the guide line's exact frame numbers or ensure the guide line spans the exact same number of frames as the trimmed Colab export (`len(frames)`).
- **Missing `freetype-py` or `uharfbuzz`**: Install all dependencies via `pip install -r requirements.txt`.
