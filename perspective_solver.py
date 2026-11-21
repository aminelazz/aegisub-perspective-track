#!/usr/bin/env python3
"""CPU-optimized 4-corner planar tracker and exact libass 3D perspective solver.

Tracks a 4-point quadrilateral across a frame range in a video file using
OpenCV (Lucas-Kanade Optical Flow, ORB+RANSAC, or Hybrid) and decomposes each
frame's tracked quad into exact libass 3D transform tags:
    \\an5\\pos(x,y)\\org(ox,oy)\\fscx...\\fscy...\\frx...\\fry...\\frz...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any

# Silence OpenCV/FFmpeg console noise before importing cv2 so stdout stays pure JSON.
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2
import numpy as np

# Fixed virtual screen / focal distance used internally by libass & VSFilter.
LIBASS_SCREEN_Z = 312.5

# Normalized method keys matching GUI dropdown labels.
METHOD_HYBRID = "hybrid"
METHOD_LK = "lk"
METHOD_ORB = "orb"
METHOD_CPU_MESH = "cpu_mesh"
METHOD_AI_MESH = "ai_mesh"

DIRECTION_FORWARD = "forward"
DIRECTION_BACKWARD = "backward"

ASPECT_AUTO = "auto"
ASPECT_KEEP = "keep"


class TrackerError(Exception):
    """Domain error carrying a user-friendly message for Aegisub UI display."""


def normalize_method(method: str) -> str:
    """Map GUI dropdown string or CLI alias to a canonical tracking method key."""
    m = (method or "").strip().lower()
    if "ai" in m or "colab" in m or "json" in m or "cotracker" in m:
        return METHOD_AI_MESH
    if "mesh" in m or "non-rigid" in m or "nonrigid" in m or "distort" in m or "bend" in m:
        return METHOD_CPU_MESH
    if "hybrid" in m:
        return METHOD_HYBRID
    if "orb" in m or ("feature" in m and "flow" not in m):
        return METHOD_ORB
    if "lucas" in m or "kanade" in m or "lk" in m or "optical" in m:
        return METHOD_LK
    return METHOD_HYBRID


def normalize_direction(direction: str) -> str:
    """Map GUI dropdown string or CLI alias to 'forward' or 'backward'."""
    d = (direction or "").strip().lower()
    if "back" in d or "end frame -> start frame" in d:
        return DIRECTION_BACKWARD
    return DIRECTION_FORWARD


def normalize_aspect_mode(mode: str) -> str:
    """Map GUI dropdown string or CLI alias to 'auto' or 'keep'."""
    m = (mode or "").strip().lower()
    if "keep" in m or "original" in m:
        return ASPECT_KEEP
    return ASPECT_AUTO


def _polygon_signed_area(pts: np.ndarray) -> float:
    """Compute signed shoelace area*2 in screen coordinates (+X right, +Y down).

    Positive value corresponds to clockwise winding (TL -> TR -> BR -> BL).
    """
    x = pts[:, 0]
    y = pts[:, 1]
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def validate_and_order_quad(points: list[list[float]] | np.ndarray) -> np.ndarray:
    """Validate 4 distinct convex quad corners and enforce clockwise winding."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape != (4, 2):
        raise TrackerError(
            f"Expected exactly 4 corner points (x, y), got {pts.shape[0] if pts.ndim >= 1 else 0}."
        )
    if not np.all(np.isfinite(pts)):
        raise TrackerError("Corner coordinates contain invalid non-finite values.")

    # Reject duplicate or nearly coincident vertices.
    for i in range(4):
        for j in range(i + 1, 4):
            if np.linalg.norm(pts[i] - pts[j]) < 1.0:
                raise TrackerError(
                    "Two or more \\clip corners are virtually identical (< 1 px apart). "
                    "Please draw 4 distinct corners around the sign."
                )

    # Check convexity via consecutive edge cross products.
    crosses = []
    for i in range(4):
        e1 = pts[(i + 1) % 4] - pts[i]
        e2 = pts[(i + 2) % 4] - pts[(i + 1) % 4]
        cross = float(e1[0] * e2[1] - e1[1] * e2[0])
        crosses.append(cross)

    min_abs_cross = min(abs(c) for c in crosses)
    if min_abs_cross < 1e-2:
        raise TrackerError(
            "Three or more \\clip corners are collinear (zero area). "
            "Please draw a non-degenerate 4-corner quadrilateral."
        )

    pos_count = sum(1 for c in crosses if c > 0)
    neg_count = sum(1 for c in crosses if c < 0)
    if pos_count != 4 and neg_count != 4:
        raise TrackerError(
            "The 4 \\clip corners form a self-intersecting or concave polygon. "
            "Please mark the 4 corners in order around the sign perimeter "
            "(Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left)."
        )

    # In screen coords (+X right, +Y down), positive signed area is clockwise.
    # If counter-clockwise, reverse vertices while keeping vertex 0 anchored.
    if _polygon_signed_area(pts) < 0:
        pts = np.array([pts[0], pts[3], pts[2], pts[1]], dtype=np.float64)

    return pts


def _extract_clip_payload(text_or_clip: str) -> str:
    """Extract the inner content of \\clip(...) or return raw drawing string."""
    s = (text_or_clip or "").strip()
    if not s:
        raise TrackerError("No \\clip tag or corner coordinates provided.")

    # Look for \clip(...) or \iclip(...) anywhere in an ASS line.
    match = re.search(r"\\i?clip\(([^)]*)\)", s)
    if match:
        return match.group(1).strip()

    # Check if raw ASS line without any clip tag was supplied.
    if "{" in s and "}" in s:
        raise TrackerError(
            "No \\clip(...) tag found on the guide line. "
            "Please draw a 4-corner vector clip (m x1 y1 l x2 y2 x3 y3 x4 y4) on the guide line."
        )
    return s


def parse_clip_corners(
    clip_input: str | list[list[float]] | np.ndarray,
) -> np.ndarray:
    """Parse a 4-point ASS vector \\clip into clockwise float64 corners (4, 2).

    Accepts:
        - A pre-parsed list/array of 4 [x, y] pairs
        - A raw ASS dialogue text containing \\clip(...)
        - A raw vector clip payload string such as 'm x1 y1 l x2 y2 x3 y3 x4 y4'
    """
    if isinstance(clip_input, (list, tuple, np.ndarray)):
        return validate_and_order_quad(clip_input)

    payload = _extract_clip_payload(str(clip_input))

    # Handle optional vector clip scale prefix: \clip(scale, m x1 y1 ...)
    scale_denom = 1.0
    if "," in payload:
        parts = [p.strip() for p in payload.split(",")]
        if len(parts) == 4 and not re.search(r"[a-zA-Z]", payload):
            raise TrackerError(
                "Rectangular \\clip(x1,y1,x2,y2) detected. "
                "Please use Aegisub's Vector Clip tool to mark the 4 perspective corners "
                "(m x1 y1 l x2 y2 x3 y3 x4 y4)."
            )
        if len(parts) == 2 and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", parts[0]):
            scale = float(parts[0])
            scale_denom = 2.0 ** (scale - 1.0)
            payload = parts[1]
        else:
            raise TrackerError(
                f"Unsupported \\clip format: '{payload}'. "
                "Expected a 4-point vector clip: \\clip(m x1 y1 l x2 y2 x3 y3 x4 y4)."
            )

    if re.search(r"\b[bspc]\b", payload, re.IGNORECASE):
        raise TrackerError(
            "Bezier/spline commands are not supported in \\clip. "
            "Please use straight line segments (m ... l ...) with 4 corners."
        )

    tokens = re.findall(r"[a-zA-Z]+|[-+]?(?:\d+(?:\.\d*)?|\.\d+)", payload)
    if not tokens:
        raise TrackerError("Empty \\clip drawing command.")

    coords: list[float] = []
    cmd: str | None = None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.isalpha():
            cmd = tok.lower()
            if cmd not in ("m", "n", "l"):
                raise TrackerError(
                    f"Unsupported ASS drawing command '{tok}' in \\clip. "
                    "Only 'm' (move) and 'l' (line) are supported for 4-corner quads."
                )
            i += 1
            continue
        if cmd is None:
            raise TrackerError(
                "Invalid vector \\clip syntax (coordinates appeared before 'm' command)."
            )
        coords.append(float(tok) / scale_denom)
        i += 1

    if len(coords) % 2 != 0:
        raise TrackerError(
            "Odd number of coordinate values in \\clip; each corner requires an (x, y) pair."
        )

    pts = [[coords[j], coords[j + 1]] for j in range(0, len(coords), 2)]

    # Drop optional closing 5th point if it closes back onto the 1st point.
    if len(pts) == 5:
        if math.hypot(pts[4][0] - pts[0][0], pts[4][1] - pts[0][1]) <= 1.0:
            pts.pop()

    if len(pts) != 4:
        raise TrackerError(
            f"The \\clip must contain exactly 4 corners (found {len(pts)}). "
            "Please mark only the 4 corners of the target plane in clockwise order "
            "(Top-Left, Top-Right, Bottom-Right, Bottom-Left)."
        )

    return validate_and_order_quad(pts)


# ---------------------------------------------------------------------------
# Step A: CPU-Optimized Planar Tracking
# ---------------------------------------------------------------------------


def _eroded_quad_mask(shape_hw: tuple[int, int], quad: np.ndarray) -> np.ndarray:
    """Create a binary polygon mask eroded slightly inward to avoid background bleed."""
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
    xs, ys = quad[:, 0], quad[:, 1]
    min_span = min(float(np.max(xs) - np.min(xs)), float(np.max(ys) - np.min(ys)))
    if min_span >= 24.0:
        erode_px = max(2, min(6, int(0.04 * min_span)))
        kernel = np.ones((2 * erode_px + 1, 2 * erode_px + 1), dtype=np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=1)
        if np.count_nonzero(eroded) >= 64:
            return eroded
    return mask


def _sample_quad_points(gray: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Sample interior dense grid, inset corner patches, and eroded Shi-Tomasi points."""
    h, w = gray.shape[:2]
    unit_sq = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    h_mat = cv2.getPerspectiveTransform(unit_sq, quad.astype(np.float32))

    # 18x18 dense grid strictly inside [0.06, 0.94]^2 so edge pixels don't bleed background
    grid_vals = np.linspace(0.06, 0.94, 18, dtype=np.float32)
    gx, gy = np.meshgrid(grid_vals, grid_vals)
    unit_batches = [np.column_stack([gx.ravel(), gy.ravel()])]

    # Inset 4x4 subgrids near each corner inside the quad
    low = np.linspace(0.06, 0.16, 4, dtype=np.float32)
    high = np.linspace(0.84, 0.94, 4, dtype=np.float32)
    for ux_vals, uy_vals in ((low, low), (high, low), (high, high), (low, high)):
        cgx, cgy = np.meshgrid(ux_vals, uy_vals)
        unit_batches.append(np.column_stack([cgx.ravel(), cgy.ravel()]))

    all_unit = np.vstack(unit_batches).reshape(-1, 1, 2)
    sampled = cv2.perspectiveTransform(all_unit, h_mat).reshape(-1, 2)

    # Shi-Tomasi corners strictly inside the eroded quadrilateral mask
    mask = _eroded_quad_mask((h, w), quad)
    gftt = cv2.goodFeaturesToTrack(
        gray, maxCorners=200, qualityLevel=0.01, minDistance=5, mask=mask, blockSize=7
    )

    batches = [sampled]
    if gftt is not None and len(gftt) > 0:
        batches.append(gftt.reshape(-1, 2).astype(np.float32))

    pts = np.vstack(batches)
    valid = (
        np.isfinite(pts[:, 0])
        & np.isfinite(pts[:, 1])
        & (pts[:, 0] >= 1.0)
        & (pts[:, 0] <= w - 2.0)
        & (pts[:, 1] >= 1.0)
        & (pts[:, 1] <= h - 2.0)
    )
    pts = pts[valid]
    if len(pts) == 0:
        return np.empty((0, 1, 2), dtype=np.float32)
    return pts.reshape(-1, 1, 2).astype(np.float32)


def _track_lk_correspondences(
    prev_gray: np.ndarray, curr_gray: np.ndarray, quad: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Track dense quad points via pyramidal Lucas-Kanade with forward-backward filtering."""
    p0 = _sample_quad_points(prev_gray, quad)
    if len(p0) < 4:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)
    if p1 is None or st1 is None:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    p0_back, st2, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, p1, None, **lk_params)
    if p0_back is None or st2 is None:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    valid = (st1.ravel() == 1) & (st2.ravel() == 1)
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    src_all = p0.reshape(-1, 2)[valid]
    dst_all = p1.reshape(-1, 2)[valid]
    back_all = p0_back.reshape(-1, 2)[valid]
    fb_err = np.linalg.norm(src_all - back_all, axis=1)

    keep = fb_err < 1.5
    if np.sum(keep) < 16:
        keep = fb_err < 3.0
    if np.sum(keep) < 8 and len(fb_err) >= 8:
        order = np.argsort(fb_err)
        keep_idx = order[: max(8, len(order) // 2)]
        return src_all[keep_idx].astype(np.float32), dst_all[keep_idx].astype(np.float32)

    return src_all[keep].astype(np.float32), dst_all[keep].astype(np.float32)


def _track_orb_correspondences(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    quad: np.ndarray,
    orb: cv2.ORB,
    clahe: cv2.CLAHE,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect and match ORB keypoints inside the quad ROI using ratio test + mutual match."""
    h, w = prev_gray.shape[:2]
    xs, ys = quad[:, 0], quad[:, 1]
    min_x, max_x = float(np.min(xs)), float(np.max(xs))
    min_y, max_y = float(np.min(ys)), float(np.max(ys))
    span = max(max_x - min_x, max_y - min_y, 32.0)

    pad_src = max(32, int(0.25 * span))
    pad_dst = max(64, int(0.60 * span))

    sx0 = max(0, int(math.floor(min_x)) - pad_src)
    sy0 = max(0, int(math.floor(min_y)) - pad_src)
    sx1 = min(w, int(math.ceil(max_x)) + pad_src)
    sy1 = min(h, int(math.ceil(max_y)) + pad_src)

    dx0 = max(0, int(math.floor(min_x)) - pad_dst)
    dy0 = max(0, int(math.floor(min_y)) - pad_dst)
    dx1 = min(w, int(math.ceil(max_x)) + pad_dst)
    dy1 = min(h, int(math.ceil(max_y)) + pad_dst)

    if sx1 - sx0 < 16 or sy1 - sy0 < 16 or dx1 - dx0 < 16 or dy1 - dy0 < 16:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    src_roi = clahe.apply(prev_gray[sy0:sy1, sx0:sx1])
    dst_roi = clahe.apply(curr_gray[dy0:dy1, dx0:dx1])

    local_quad = quad - np.array([sx0, sy0], dtype=np.float64)
    src_mask = _eroded_quad_mask(src_roi.shape[:2], local_quad)

    kp1, des1 = orb.detectAndCompute(src_roi, src_mask)
    kp2, des2 = orb.detectAndCompute(dst_roi, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches_12 = bf.knnMatch(des1, des2, k=2)
    matches_21 = bf.knnMatch(des2, des1, k=2)

    best_rev: dict[int, int] = {}
    for pair in matches_21:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.85 * n.distance:
                best_rev[m.queryIdx] = m.trainIdx
        elif len(pair) == 1:
            m = pair[0]
            best_rev[m.queryIdx] = m.trainIdx

    src_pts = []
    dst_pts = []
    for pair in matches_12:
        if len(pair) == 2:
            m, n = pair
            if m.distance >= 0.78 * n.distance:
                continue
        elif len(pair) == 1:
            m = pair[0]
        else:
            continue

        if best_rev.get(m.trainIdx) == m.queryIdx:
            p_src = kp1[m.queryIdx].pt
            p_dst = kp2[m.trainIdx].pt
            src_pts.append([p_src[0] + sx0, p_src[1] + sy0])
            dst_pts.append([p_dst[0] + dx0, p_dst[1] + dy0])

    if not src_pts:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    return np.asarray(src_pts, dtype=np.float32), np.asarray(dst_pts, dtype=np.float32)


def _is_valid_projected_quad(
    next_quad: np.ndarray, prev_quad: np.ndarray, homography: np.ndarray, img_w: int, img_h: int
) -> bool:
    """Verify projected quad is finite, convex, non-collapsed, and bounded."""
    if not np.all(np.isfinite(next_quad)):
        return False

    prev_area = abs(_polygon_signed_area(prev_quad)) * 0.5
    next_area = abs(_polygon_signed_area(next_quad)) * 0.5
    if next_area < max(9.0, 0.08 * prev_area) or next_area > max(16.0, 12.0 * prev_area):
        return False

    # Preserve clockwise orientation and convexity.
    if _polygon_signed_area(next_quad) <= 0:
        return False
    for i in range(4):
        e1 = next_quad[(i + 1) % 4] - next_quad[i]
        e2 = next_quad[(i + 2) % 4] - next_quad[(i + 1) % 4]
        if (e1[0] * e2[1] - e1[1] * e2[0]) <= 0:
            return False

    # Ensure corners haven't exploded far outside reasonable screen range.
    margin = max(img_w, img_h) * 1.5
    if (
        np.min(next_quad[:, 0]) < -margin
        or np.max(next_quad[:, 0]) > img_w + margin
        or np.min(next_quad[:, 1]) < -margin
        or np.max(next_quad[:, 1]) > img_h + margin
    ):
        return False

    det = float(np.linalg.det(homography[:2, :2]))
    if not (0.05 < abs(det) < 20.0):
        return False
    return True


def _collect_correspondences(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    quad: np.ndarray,
    method_key: str,
    orb: cv2.ORB,
    clahe: cv2.CLAHE,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Collect feature/flow correspondences and RANSAC threshold for a tracking method."""
    if method_key == METHOD_LK:
        src_pts, dst_pts = _track_lk_correspondences(prev_gray, curr_gray, quad)
        return src_pts, dst_pts, 2.0

    if method_key == METHOD_ORB:
        src_pts, dst_pts = _track_orb_correspondences(prev_gray, curr_gray, quad, orb, clahe)
        # Augment with dense LK if ORB texture is too sparse on this frame.
        if len(src_pts) < 12:
            lk_src, lk_dst = _track_lk_correspondences(prev_gray, curr_gray, quad)
            if len(lk_src) >= 4:
                src_pts = np.vstack([src_pts, lk_src]) if len(src_pts) else lk_src
                dst_pts = np.vstack([dst_pts, lk_dst]) if len(dst_pts) else lk_dst
        return src_pts, dst_pts, 2.5

    lk_src, lk_dst = _track_lk_correspondences(prev_gray, curr_gray, quad)
    orb_src, orb_dst = _track_orb_correspondences(prev_gray, curr_gray, quad, orb, clahe)
    if len(orb_src) >= 6 and len(lk_src) >= 4:
        src_pts = np.vstack([lk_src, orb_src])
        dst_pts = np.vstack([lk_dst, orb_dst])
    elif len(lk_src) >= 4:
        src_pts, dst_pts = lk_src, lk_dst
    else:
        src_pts, dst_pts = orb_src, orb_dst
    return src_pts, dst_pts, 1.8


def _estimate_next_quad(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    quad: np.ndarray,
    method_key: str,
    orb: cv2.ORB,
    clahe: cv2.CLAHE,
) -> np.ndarray:
    """Compute optimal homography between prev_gray and curr_gray and project quad."""
    img_h, img_w = prev_gray.shape[:2]
    src_pts, dst_pts, ransac_thresh = _collect_correspondences(
        prev_gray, curr_gray, quad, method_key, orb, clahe
    )

    if len(src_pts) < 4:
        return quad.copy()

    homography, inliers = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=ransac_thresh, maxIters=2000, confidence=0.995
    )

    if homography is not None and inliers is not None:
        inlier_mask = inliers.ravel().astype(bool)
        if np.sum(inlier_mask) >= 6:
            h_refined, _ = cv2.findHomography(src_pts[inlier_mask], dst_pts[inlier_mask], 0)
            if h_refined is not None and np.all(np.isfinite(h_refined)):
                homography = h_refined

    if homography is not None:
        projected = cv2.perspectiveTransform(
            quad.reshape(-1, 1, 2).astype(np.float32), homography
        ).reshape(4, 2)
        projected = projected.astype(np.float64)
        if _is_valid_projected_quad(projected, quad, homography, img_w, img_h):
            return projected

    # Robust translation fallback if homography estimation failed on heavy occlusion.
    if len(src_pts) >= 3:
        delta = np.median(dst_pts - src_pts, axis=0)
        if np.all(np.isfinite(delta)) and np.linalg.norm(delta) < max(img_w, img_h) * 0.25:
            return quad + delta.astype(np.float64)

    return quad.copy()


def _prepare_reference_roi(ref_gray: np.ndarray, ref_quad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop padded ROI around initial reference quad for drift-free canonical anchoring."""
    h, w = ref_gray.shape[:2]
    pad = 24
    x0 = max(0, int(math.floor(float(np.min(ref_quad[:, 0])))) - pad)
    y0 = max(0, int(math.floor(float(np.min(ref_quad[:, 1])))) - pad)
    x1 = min(w, int(math.ceil(float(np.max(ref_quad[:, 0])))) + pad)
    y1 = min(h, int(math.ceil(float(np.max(ref_quad[:, 1])))) + pad)
    ref_roi = ref_gray[y0:y1, x0:x1].copy()
    ref_quad_local = ref_quad - np.array([x0, y0], dtype=np.float64)
    return ref_roi, ref_quad_local


def _refine_against_reference(
    ref_roi: np.ndarray,
    ref_quad_local: np.ndarray,
    curr_gray: np.ndarray,
    pred_quad: np.ndarray,
    method_key: str,
    orb: cv2.ORB,
    clahe: cv2.CLAHE,
) -> np.ndarray:
    """Warp curr_gray back to reference ROI and refine pred_quad to eliminate sequential drift."""
    roi_h, roi_w = ref_roi.shape[:2]
    if roi_h < 16 or roi_w < 16:
        return pred_quad

    h_ref_to_curr = cv2.getPerspectiveTransform(
        ref_quad_local.astype(np.float32), pred_quad.astype(np.float32)
    )
    warped_curr_roi = cv2.warpPerspective(
        curr_gray,
        h_ref_to_curr,
        (roi_w, roi_h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )

    src_pts, dst_pts, _ = _collect_correspondences(
        ref_roi, warped_curr_roi, ref_quad_local, method_key, orb, clahe
    )
    if len(src_pts) < 12:
        return pred_quad

    h_corr, inliers = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=1.5, maxIters=1500, confidence=0.995
    )
    if h_corr is None or inliers is None:
        return pred_quad

    inlier_mask = inliers.ravel().astype(bool)
    inlier_count = int(np.sum(inlier_mask))
    if inlier_count < 12 or (inlier_count / len(src_pts)) < 0.45:
        return pred_quad

    h_corr_ls, _ = cv2.findHomography(src_pts[inlier_mask], dst_pts[inlier_mask], 0)
    if h_corr_ls is not None and np.all(np.isfinite(h_corr_ls)):
        h_corr = h_corr_ls

    h_total = h_ref_to_curr.astype(np.float64) @ h_corr.astype(np.float64)
    refined = cv2.perspectiveTransform(
        ref_quad_local.reshape(-1, 1, 2).astype(np.float32), h_total.astype(np.float32)
    ).reshape(4, 2).astype(np.float64)

    img_h, img_w = curr_gray.shape[:2]
    if not _is_valid_projected_quad(refined, pred_quad, h_corr, img_w, img_h):
        return pred_quad

    # Guard against wild jumps on heavy occlusion or major appearance change.
    if float(np.max(np.linalg.norm(refined - pred_quad, axis=1))) > 6.0:
        return pred_quad

    return refined


def _read_gray_chunk(
    cap: cv2.VideoCapture, start_idx: int, count: int
) -> list[np.ndarray]:
    """Seek once to start_idx and sequentially read `count` grayscale frames."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_idx))
    frames: list[np.ndarray] = []
    for _ in range(count):
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    return frames


def track_planar_quad(
    video_path: str,
    initial_quad_ass: np.ndarray,
    start_frame: int,
    end_frame: int,
    play_res_x: float,
    play_res_y: float,
    method: str,
    direction: str,
) -> list[np.ndarray]:
    """Track a 4-point quad across [start_frame..end_frame] and return ASS-space quads."""
    if not video_path or not os.path.isfile(video_path):
        raise TrackerError(f"Video file not found: '{video_path}'.")
    if start_frame < 0 or end_frame < start_frame:
        raise TrackerError(
            f"Invalid frame range ({start_frame} to {end_frame}). "
            "Ensure the guide line has a positive duration."
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise TrackerError(f"OpenCV failed to open video file: '{video_path}'.")

    try:
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if vid_w <= 0 or vid_h <= 0:
            raise TrackerError("Could not determine video dimensions from VideoCapture.")
        if total_frames > 0 and start_frame >= total_frames:
            raise TrackerError(
                f"Start frame {start_frame} exceeds video frame count ({total_frames})."
            )
        if total_frames > 0 and end_frame >= total_frames:
            end_frame = total_frames - 1

        res_x = float(play_res_x) if play_res_x > 0 else float(vid_w)
        res_y = float(play_res_y) if play_res_y > 0 else float(vid_h)
        scale_to_vid = np.array([vid_w / res_x, vid_h / res_y], dtype=np.float64)
        scale_to_ass = np.array([res_x / vid_w, res_y / vid_h], dtype=np.float64)

        init_quad_vid = initial_quad_ass * scale_to_vid
        num_frames = end_frame - start_frame + 1

        if num_frames == 1:
            return [initial_quad_ass.copy()]

        method_key = normalize_method(method)
        dir_key = normalize_direction(direction)
        orb = cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, nlevels=8, fastThreshold=7)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        tracked_vid: list[np.ndarray] = [None] * num_frames  # type: ignore[list-item]

        if dir_key == DIRECTION_FORWARD:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
            ok, bgr0 = cap.read()
            if not ok or bgr0 is None:
                raise TrackerError(f"Failed to decode start frame {start_frame} from video.")
            prev_gray = cv2.cvtColor(bgr0, cv2.COLOR_BGR2GRAY)
            ref_roi, ref_quad_local = _prepare_reference_roi(prev_gray, init_quad_vid)
            curr_quad = init_quad_vid.copy()
            tracked_vid[0] = curr_quad

            for offset in range(1, num_frames):
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    # Repeat last valid quad if container reports slightly more frames than decodable.
                    tracked_vid[offset] = curr_quad.copy()
                    continue
                curr_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                pred_quad = _estimate_next_quad(
                    prev_gray, curr_gray, curr_quad, method_key, orb, clahe
                )
                curr_quad = _refine_against_reference(
                    ref_roi, ref_quad_local, curr_gray, pred_quad, method_key, orb, clahe
                )
                tracked_vid[offset] = curr_quad
                prev_gray = curr_gray
        else:
            # Backward tracking: read sequential chunks (<= 512 MB grayscale budget)
            # to avoid inaccurate keyframe-by-keyframe reverse seeks.
            max_chunk_frames = max(16, (512 * 1024 * 1024) // max(1, vid_w * vid_h))
            curr_quad = init_quad_vid.copy()
            tracked_vid[num_frames - 1] = curr_quad
            bridge_gray: np.ndarray | None = None
            ref_roi: np.ndarray | None = None
            ref_quad_local: np.ndarray | None = None

            chunk_end = num_frames - 1
            while chunk_end >= 0:
                chunk_start = max(0, chunk_end - max_chunk_frames + 1)
                chunk_len = chunk_end - chunk_start + 1
                grays = _read_gray_chunk(cap, start_frame + chunk_start, chunk_len)
                if not grays:
                    raise TrackerError(
                        f"Failed to decode frame range {start_frame + chunk_start}..{start_frame + chunk_end}."
                    )

                if ref_roi is None:
                    ref_roi, ref_quad_local = _prepare_reference_roi(grays[-1], init_quad_vid)

                # If continuing from a previous chunk, bridge across chunk boundary first.
                if bridge_gray is not None and len(grays) > 0:
                    pred_quad = _estimate_next_quad(
                        bridge_gray, grays[-1], curr_quad, method_key, orb, clahe
                    )
                    curr_quad = _refine_against_reference(
                        ref_roi, ref_quad_local, grays[-1], pred_quad, method_key, orb, clahe
                    )
                    tracked_vid[chunk_end] = curr_quad

                for local_idx in range(len(grays) - 2, -1, -1):
                    global_idx = chunk_start + local_idx
                    pred_quad = _estimate_next_quad(
                        grays[local_idx + 1], grays[local_idx], curr_quad, method_key, orb, clahe
                    )
                    curr_quad = _refine_against_reference(
                        ref_roi, ref_quad_local, grays[local_idx], pred_quad, method_key, orb, clahe
                    )
                    tracked_vid[global_idx] = curr_quad

                # Handle any tail frames if decoder stopped early.
                for local_idx in range(len(grays), chunk_len):
                    global_idx = chunk_start + local_idx
                    if tracked_vid[global_idx] is None:
                        tracked_vid[global_idx] = curr_quad.copy()

                bridge_gray = grays[0]
                chunk_end = chunk_start - 1

        return [q * scale_to_ass for q in tracked_vid]
    finally:
        cap.release()


def smooth_corner_trajectories(
    quads: list[np.ndarray], smoothing_window: int, ref_idx: int
) -> list[np.ndarray]:
    """Apply symmetric Gaussian temporal smoothing across corner trajectories.

    The initial reference frame `ref_idx` is pinned so the user's drawn \\clip stays exact.
    """
    radius = max(0, min(10, int(smoothing_window)))
    n = len(quads)
    if radius == 0 or n <= 2:
        return quads

    arr = np.stack(quads, axis=0)  # (N, 4, 2)
    smoothed = np.empty_like(arr)
    for t in range(n):
        r_t = min(radius, t, n - 1 - t)
        if t == ref_idx or r_t == 0:
            smoothed[t] = arr[t]
            continue
        sigma = max(0.5, r_t / 2.0)
        offsets = np.arange(-r_t, r_t + 1)
        weights = np.exp(-0.5 * (offsets / sigma) ** 2)
        weights /= np.sum(weights)
        acc = np.zeros((4, 2), dtype=np.float64)
        for k, w in zip(offsets, weights):
            acc += w * arr[t + int(k)]
        smoothed[t] = acc

    return [smoothed[t] for t in range(n)]


# ---------------------------------------------------------------------------
# Step B: Exact Homography-to-ASS 3D Tag Decomposition
# ---------------------------------------------------------------------------


def _rot_x(v: np.ndarray, a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([v[0], v[1] * c + v[2] * s, -v[1] * s + v[2] * c], dtype=np.float64)


def _rot_y(v: np.ndarray, a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([v[0] * c - v[2] * s, v[1], v[0] * s + v[2] * c], dtype=np.float64)


def _rot_z(v: np.ndarray, a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([v[0] * c + v[1] * s, -v[0] * s + v[1] * c, v[2]], dtype=np.float64)


def _intersect_diagonals(quad: np.ndarray) -> np.ndarray:
    """Return 2D intersection of quad diagonals (P0-P2 and P1-P3)."""
    p0, p1, p2, p3 = quad
    mat = np.array(
        [[p2[0] - p0[0], p1[0] - p3[0]], [p2[1] - p0[1], p1[1] - p3[1]]],
        dtype=np.float64,
    )
    rhs = np.array([p1[0] - p0[0], p1[1] - p0[1]], dtype=np.float64)
    if abs(float(np.linalg.det(mat))) < 1e-9:
        return 0.25 * (p0 + p1 + p2 + p3)
    t = np.linalg.solve(mat, rhs)
    return p0 + t[0] * (p2 - p0)


def _reconstruct_3d_corners_from_pinhole(
    quad: np.ndarray, center_2d: np.ndarray, optical_center: np.ndarray, focal_z: float
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct 3D quad vertices [A0, A1, A2, A3] and center ray relative to optical_center."""
    c_rel = center_2d - optical_center
    center_3d = np.array([c_rel[0], c_rel[1], focal_z], dtype=np.float64)
    rays = [
        np.array([p[0] - optical_center[0], p[1] - optical_center[1], focal_z], dtype=np.float64)
        for p in quad
    ]

    pts3d = [np.zeros(3, dtype=np.float64) for _ in range(4)]
    for i in range(2):
        vp1 = float(np.linalg.norm(np.cross(rays[i], center_3d)))
        vp2 = float(np.linalg.norm(np.cross(rays[2 + i], center_3d)))
        a = rays[i]
        c = rays[2 + i] * (vp1 / max(vp2, 1e-12))
        m = 0.5 * (a + c)
        r = center_3d[2] / (m[2] if abs(m[2]) > 1e-12 else 1e-12)
        pts3d[i] = a * r
        pts3d[2 + i] = c * r

    return np.stack(pts3d, axis=0), center_3d


def solve_optimal_org(
    quad: np.ndarray,
    ref_pt: np.ndarray,
    screen_z: float = LIBASS_SCREEN_Z,
    max_offset: float = 2500.0,
) -> np.ndarray:
    """Find the libass \\org(ox, oy) closest to ref_pt that yields orthogonal 3D plane axes.

    Solves (U0 - du*o) . (V0 - dv*o) + du*dv*screen_z^2 = 0 in a rationalized
    quadratic form that stays continuous and singularity-free even when opposite
    edges become parallel (du -> 0 or dv -> 0).
    """
    p0, p1, p2, p3 = quad
    c2d = _intersect_diagonals(quad)
    d0 = float(np.linalg.norm(p0 - c2d))
    d2 = float(np.linalg.norm(p2 - c2d))
    d1 = float(np.linalg.norm(p1 - c2d))
    d3 = float(np.linalg.norm(p3 - c2d))

    a0 = 2.0 * d2 / max(d0 + d2, 1e-9)
    a1 = 2.0 * d3 / max(d1 + d3, 1e-9)
    a3 = 2.0 * d1 / max(d1 + d3, 1e-9)

    du = a1 - a0
    dv = a3 - a0
    u0 = a1 * p1 - a0 * p0
    v0 = a3 * p3 - a0 * p0

    u_ref = u0 - du * ref_pt
    v_ref = v0 - dv * ref_pt

    a_coef = du * dv
    b_vec = dv * u_ref + du * v_ref
    c_coef = float(np.dot(u_ref, v_ref) + a_coef * (screen_z ** 2))
    b_norm = float(np.linalg.norm(b_vec))

    if b_norm < 1e-9 and abs(a_coef) < 1e-12:
        return ref_pt.copy()

    if b_norm < 1e-9:
        # Degenerate symmetric case: circle centered at ref_pt
        if -c_coef / a_coef > 0.0:
            s = min(max_offset, math.sqrt(-c_coef / a_coef))
            u_norm = float(np.linalg.norm(u_ref))
            direction = (u_ref / u_norm) if u_norm > 1e-9 else np.array([1.0, 0.0])
            return ref_pt + s * direction
        return ref_pt.copy()

    disc = b_norm * b_norm - 4.0 * a_coef * c_coef
    if disc >= 0.0:
        denom = b_norm + math.sqrt(disc)
        s = (2.0 * c_coef) / denom if denom > 1e-12 else 0.0
    else:
        # Closest point minimizing residual shear when orthogonality circle doesn't exist
        s = b_norm / (2.0 * a_coef) if abs(a_coef) > 1e-12 else 0.0

    if abs(s) > max_offset:
        s = math.copysign(max_offset, s)

    return ref_pt + s * (b_vec / b_norm)


def _unwrap_angles(angles: list[float]) -> list[float]:
    """Unwrap degree sequence across +-180 deg boundary for smooth frame interpolation."""
    if not angles:
        return []
    out = [angles[0]]
    for ang in angles[1:]:
        prev = out[-1]
        diff = (ang - prev + 180.0) % 360.0 - 180.0
        out.append(prev + diff)
    return out


def decompose_quad_to_ass_params(
    quads: list[np.ndarray],
    play_res_x: float,
    play_res_y: float,
    fov_factor: float,
    aspect_mode: str,
    ref_idx: int = 0,
) -> list[dict[str, float]]:
    """Decompose tracked 2D quads into exact libass 3D transform tag values per frame."""
    screen_z = LIBASS_SCREEN_Z
    aspect_key = normalize_aspect_mode(aspect_mode)

    # First pass: compute exact 3D pose & pivot-plane dimensions in libass space per frame.
    raw_frames: list[dict[str, float]] = []
    for quad in quads:
        c2d = _intersect_diagonals(quad)
        org = solve_optimal_org(quad, c2d, screen_z=screen_z)

        pts3d, center_3d = _reconstruct_3d_corners_from_pinhole(quad, c2d, org, screen_z)
        a0, a1, _, a3 = pts3d
        ab = a1 - a0
        ad = a3 - a0

        n = np.cross(ab, ad)
        n_norm = float(np.linalg.norm(n))
        if n_norm < 1e-9:
            n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            n /= n_norm
        if n[2] < 0:
            n = -n

        fry_rad = math.atan2(n[0], n[2])
        rot_n = _rot_y(n, fry_rad)
        frx_rad = -math.atan2(rot_n[1], rot_n[2])
        ab_unrot = _rot_x(_rot_y(ab, fry_rad), frx_rad)
        frz_rad = math.atan2(ab_unrot[1], ab_unrot[0])

        o_piv = np.array([0.0, 0.0, screen_z], dtype=np.float64)
        denom = float(np.dot(center_3d, n))
        lam = float(np.dot(o_piv, n)) / (denom if abs(denom) > 1e-9 else 1e-9)

        d_c = lam * center_3d - o_piv
        d_c_unrot = _rot_z(_rot_x(_rot_y(d_c, fry_rad), frx_rad), frz_rad)
        px = float(org[0] + d_c_unrot[0])
        py = float(org[1] + d_c_unrot[1])

        w3d = float(lam * np.linalg.norm(ab))
        # Project ad onto the orthogonal Y axis after unrotating so tiny residual shear stays clean.
        ad_unrot = _rot_z(_rot_x(_rot_y(ad * lam, fry_rad), frx_rad), frz_rad)
        h3d = float(abs(ad_unrot[1])) if abs(ad_unrot[1]) > 1e-6 else float(lam * np.linalg.norm(ad))

        raw_frames.append(
            {
                "px": px,
                "py": py,
                "ox": float(org[0]),
                "oy": float(org[1]),
                "w3d": w3d,
                "h3d": h3d,
                "frx": -math.degrees(frx_rad),
                "fry": -math.degrees(fry_rad),
                "frz": -math.degrees(frz_rad),
            }
        )

    # Determine reference baseline width & height from the reference frame.
    ref_quad = quads[ref_idx]
    ref_raw = raw_frames[ref_idx]

    if aspect_key == ASPECT_AUTO:
        f_cam = max(0.1, float(fov_factor)) * max(1.0, float(play_res_x))
        cam_center = np.array([play_res_x * 0.5, play_res_y * 0.5], dtype=np.float64)
        c2d_ref = _intersect_diagonals(ref_quad)
        pts_cam, _ = _reconstruct_3d_corners_from_pinhole(ref_quad, c2d_ref, cam_center, f_cam)
        base_w = float(np.linalg.norm(pts_cam[1] - pts_cam[0]))
        base_h = float(np.linalg.norm(pts_cam[3] - pts_cam[0]))
    else:
        base_w = ref_raw["w3d"]
        base_h = ref_raw["h3d"]

    base_w = max(base_w, 1e-6)
    base_h = max(base_h, 1e-6)

    # Smoothly unwrap Euler rotations chronologically across all frames.
    frx_unwrapped = _unwrap_angles([f["frx"] for f in raw_frames])
    fry_unwrapped = _unwrap_angles([f["fry"] for f in raw_frames])
    frz_unwrapped = _unwrap_angles([f["frz"] for f in raw_frames])

    params: list[dict[str, float]] = []
    for idx, f in enumerate(raw_frames):
        params.append(
            {
                "px": f["px"],
                "py": f["py"],
                "ox": f["ox"],
                "oy": f["oy"],
                "fscx": 100.0 * f["w3d"] / base_w,
                "fscy": 100.0 * f["h3d"] / base_h,
                "frx": frx_unwrapped[idx],
                "fry": fry_unwrapped[idx],
                "frz": frz_unwrapped[idx],
            }
        )
    return params


# ---------------------------------------------------------------------------
# ASS Override Tag Stripping & Line Formatting
# ---------------------------------------------------------------------------

_STRIP_TAG_PATTERNS = [
    re.compile(r"^\\pos\([^)]*\)\s*$"),
    re.compile(r"^\\org\([^)]*\)\s*$"),
    re.compile(r"^\\i?clip\([^)]*\)\s*$"),
    re.compile(r"^\\an[1-9]\s*$"),
    re.compile(r"^\\a\d+\s*$"),
    re.compile(r"^\\fr(?:x|y|z)?\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)?\s*$"),
    re.compile(r"^\\fsc[xy]\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)?\s*$"),
    re.compile(r"^\\fa[xy]\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)?\s*$"),
]


def _should_strip_tag(tag: str) -> bool:
    return any(pat.match(tag) for pat in _STRIP_TAG_PATTERNS)


def _strip_override_block(block_inner: str) -> str:
    """Remove positioning/transform/clip tags from inside {...} while keeping styling."""
    if "\\" not in block_inner:
        return block_inner

    first_bs = block_inner.find("\\")
    prefix = block_inner[:first_bs]
    rest = block_inner[first_bs:]

    # Split at top-level backslashes (paren_depth == 0) so \t(...\frz...) stays intact.
    tags: list[str] = []
    start = 0
    depth = 0
    for idx, ch in enumerate(rest):
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
        elif ch == "\\" and depth == 0 and idx > start:
            tags.append(rest[start:idx])
            start = idx
    tags.append(rest[start:])

    kept = [t for t in tags if not _should_strip_tag(t)]
    return prefix + "".join(kept)


def strip_transform_and_clip_tags(ass_text: str) -> str:
    """Strip \\pos, \\org, \\fr*, \\fsc*, \\fa*, \\an*, and \\clip/\\iclip tags from ASS text."""
    out: list[str] = []
    i = 0
    n = len(ass_text)
    while i < n:
        if ass_text[i] == "{":
            close = ass_text.find("}", i + 1)
            if close == -1:
                out.append(ass_text[i:])
                break
            inner = _strip_override_block(ass_text[i + 1 : close])
            if inner:
                out.append("{" + inner + "}")
            i = close + 1
        else:
            nxt = ass_text.find("{", i)
            if nxt == -1:
                out.append(ass_text[i:])
                break
            out.append(ass_text[i:nxt])
            i = nxt
    return "".join(out)


def format_ass_line(guide_text: str, p: dict[str, float], extra_tags: str = "") -> str:
    """Inject computed 3D transform tags at the start of the cleaned ASS guide line."""
    tag_prefix = (
        f"\\an5"
        f"\\pos({p['px']:.2f},{p['py']:.2f})"
        f"\\org({p['ox']:.2f},{p['oy']:.2f})"
        f"\\fscx{p['fscx']:.2f}\\fscy{p['fscy']:.2f}"
        f"\\frx{p['frx']:.2f}\\fry{p['fry']:.2f}\\frz{p['frz']:.2f}"
        f"{extra_tags}"
    )

    cleaned = strip_transform_and_clip_tags(guide_text)
    if cleaned.startswith("{"):
        close = cleaned.find("}")
        if close != -1:
            existing_tags = cleaned[1:close]
            return "{" + tag_prefix + existing_tags + "}" + cleaned[close + 1 :]
    return "{" + tag_prefix + "}" + cleaned


# ---------------------------------------------------------------------------
# Non-Rigid Mesh Tracking (CPU Optical Flow) & Google Colab AI Mesh Import
# ---------------------------------------------------------------------------


def _build_mesh_regularizer(
    rows: int, cols: int, lam_lap: float = 0.03, lam_bend: float = 0.08
) -> np.ndarray:
    """Precompute (R*C, R*C) symmetric Laplacian + thin-plate bending matrix."""
    n = rows * cols
    mat = np.zeros((n, n), dtype=np.float64)

    def idx(r: int, c: int) -> int:
        return r * cols + c

    # First-order 4-neighbor Laplacian
    for r in range(rows):
        for c in range(cols):
            i = idx(r, c)
            for nr, nc in ((r, c + 1), (r + 1, c)):
                if nr < rows and nc < cols:
                    j = idx(nr, nc)
                    mat[i, i] += lam_lap
                    mat[j, j] += lam_lap
                    mat[i, j] -= lam_lap
                    mat[j, i] -= lam_lap

    # Second-order bending smoothness along rows & columns
    stencil = np.array([1.0, -2.0, 1.0], dtype=np.float64)
    outer = np.outer(stencil, stencil) * lam_bend
    for r in range(rows):
        for c in range(cols - 2):
            tri = [idx(r, c), idx(r, c + 1), idx(r, c + 2)]
            mat[np.ix_(tri, tri)] += outer
    for r in range(rows - 2):
        for c in range(cols):
            tri = [idx(r, c), idx(r + 1, c), idx(r + 2, c)]
            mat[np.ix_(tri, tri)] += outer

    return mat


def _mesh_outer_quad(mesh: np.ndarray) -> np.ndarray:
    """Return the 4 outer corner coordinates [TL, TR, BR, BL] of an (R, C, 2) mesh."""
    return np.array(
        [mesh[0, 0], mesh[0, -1], mesh[-1, -1], mesh[-1, 0]], dtype=np.float64
    )


def _sample_mesh_interior_uv(mesh: np.ndarray) -> np.ndarray:
    """Generate dense interior UV samples kept safely inside the foreground mesh boundary."""
    rows, cols = mesh.shape[:2]
    top_w = float(np.sum(np.linalg.norm(np.diff(mesh[0], axis=0), axis=1)))
    bot_w = float(np.sum(np.linalg.norm(np.diff(mesh[-1], axis=0), axis=1)))
    left_h = float(np.sum(np.linalg.norm(np.diff(mesh[:, 0], axis=0), axis=1)))
    right_h = float(np.sum(np.linalg.norm(np.diff(mesh[:, -1], axis=0), axis=1)))
    avg_w = max(20.0, 0.5 * (top_w + bot_w))
    avg_h = max(20.0, 0.5 * (left_h + right_h))

    # Keep ~11px inset from outer boundary so 21x21 LK windows never touch static background
    u_margin = min(0.10, 11.0 / avg_w)
    v_margin = min(0.10, 11.0 / avg_h)

    n_u = max(cols * 2 + 2, 16)
    n_v = max(rows * 2 + 2, 12)
    us = np.linspace(u_margin, 1.0 - u_margin, n_u, dtype=np.float64)
    vs = np.linspace(v_margin, 1.0 - v_margin, n_v, dtype=np.float64)
    gx, gy = np.meshgrid(us, vs)
    return np.column_stack([gx.ravel(), gy.ravel()])


def _sample_mesh_interior_points(gray: np.ndarray, mesh: np.ndarray) -> np.ndarray:
    """Sample dense interior cell points across a potentially curved (R, C, 2) mesh."""
    h, w = gray.shape[:2]
    uv_pts = _sample_mesh_interior_uv(mesh)
    warper = MeshWarper(mesh)
    pts = warper.warp_points(uv_pts).astype(np.float32)

    hull = cv2.convexHull(np.round(mesh.reshape(-1, 2)).astype(np.int32))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    xs, ys = mesh[:, :, 0], mesh[:, :, 1]
    min_span = min(float(np.max(xs) - np.min(xs)), float(np.max(ys) - np.min(ys)))
    if min_span >= 24.0:
        erode_px = max(3, min(8, int(0.05 * min_span)))
        mask = cv2.erode(mask, np.ones((2 * erode_px + 1, 2 * erode_px + 1), dtype=np.uint8))

    gftt = cv2.goodFeaturesToTrack(
        gray, maxCorners=220, qualityLevel=0.01, minDistance=5, mask=mask, blockSize=7
    )
    arr = pts
    if gftt is not None and len(gftt) > 0:
        arr = np.vstack([arr, gftt.reshape(-1, 2).astype(np.float32)])

    valid = (
        np.isfinite(arr[:, 0])
        & np.isfinite(arr[:, 1])
        & (arr[:, 0] >= 1.0)
        & (arr[:, 0] <= w - 2.0)
        & (arr[:, 1] >= 1.0)
        & (arr[:, 1] <= h - 2.0)
    )
    arr = arr[valid]
    if len(arr) == 0:
        return np.empty((0, 1, 2), dtype=np.float32)
    return arr.reshape(-1, 1, 2).astype(np.float32)


def _track_mesh_step_cpu(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_mesh: np.ndarray,
    reg_mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Track non-rigid (R, C, 2) mesh from prev_gray to curr_gray via bilinear LK + regularization."""
    rows, cols = prev_mesh.shape[:2]
    h, w = prev_gray.shape[:2]
    n_pts = rows * cols

    # 1. Global homography warm-start from interior LK correspondences
    pred_mesh = prev_mesh.copy()
    p_dense = _sample_mesh_interior_points(prev_gray, prev_mesh)
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if len(p_dense) >= 8:
        p1_d, st1_d, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p_dense, None, **lk_params)
        if p1_d is not None and st1_d is not None:
            p0_b, st2_d, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, p1_d, None, **lk_params)
            if p0_b is not None and st2_d is not None:
                v_d = (st1_d.ravel() == 1) & (st2_d.ravel() == 1)
                if np.sum(v_d) >= 8:
                    s_d = p_dense.reshape(-1, 2)[v_d]
                    d_d = p1_d.reshape(-1, 2)[v_d]
                    fb_d = np.linalg.norm(s_d - p0_b.reshape(-1, 2)[v_d], axis=1)
                    keep_d = fb_d < 2.0
                    if np.sum(keep_d) >= 8:
                        h_glob, inl_d = cv2.findHomography(
                            s_d[keep_d], d_d[keep_d], cv2.RANSAC, 3.0, maxIters=1500, confidence=0.995
                        )
                        if h_glob is not None and inl_d is not None and np.sum(inl_d) >= 6:
                            det = float(np.linalg.det(h_glob[:2, :2]))
                            if 0.1 < abs(det) < 10.0:
                                cand = cv2.perspectiveTransform(
                                    prev_mesh.reshape(-1, 1, 2).astype(np.float32), h_glob
                                ).reshape(rows, cols, 2).astype(np.float64)
                                if np.all(np.isfinite(cand)):
                                    pred_mesh = cand

    # 2. Track dense interior UV samples safely inset from border occlusion edges
    uv_pts = _sample_mesh_interior_uv(prev_mesh)
    prev_warper = MeshWarper(prev_mesh)
    pred_warper = MeshWarper(pred_mesh)
    p0_all = prev_warper.warp_points(uv_pts)
    p_init_all = pred_warper.warp_points(uv_pts)

    in_bounds = (
        np.isfinite(p0_all[:, 0])
        & np.isfinite(p0_all[:, 1])
        & (p0_all[:, 0] >= 2.0)
        & (p0_all[:, 0] <= w - 3.0)
        & (p0_all[:, 1] >= 2.0)
        & (p0_all[:, 1] <= h - 3.0)
        & np.isfinite(p_init_all[:, 0])
        & np.isfinite(p_init_all[:, 1])
    )
    if np.sum(in_bounds) < 8:
        return pred_mesh, np.ones((rows, cols), dtype=np.float64)

    uv_sub = uv_pts[in_bounds]
    p0 = p0_all[in_bounds].reshape(-1, 1, 2).astype(np.float32)
    p1_init = p_init_all[in_bounds].reshape(-1, 1, 2).astype(np.float32)
    p1_init[:, 0, 0] = np.clip(p1_init[:, 0, 0], 1.0, w - 2.0)
    p1_init[:, 0, 1] = np.clip(p1_init[:, 0, 1], 1.0, h - 2.0)

    lk_init_params = dict(
        winSize=(21, 21),
        maxLevel=4,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, p1_init, **lk_init_params)
    if p1 is None or st1 is None:
        return pred_mesh, np.ones((rows, cols), dtype=np.float64)

    p0_back, st2, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, p1, p0.copy(), **lk_init_params)
    if p0_back is None or st2 is None:
        return pred_mesh, np.ones((rows, cols), dtype=np.float64)

    p0_2d = p0.reshape(-1, 2).astype(np.float64)
    p1_2d = p1.reshape(-1, 2).astype(np.float64)
    back_2d = p0_back.reshape(-1, 2).astype(np.float64)
    pred_2d = p_init_all[in_bounds]

    fb_err = np.linalg.norm(p0_2d - back_2d, axis=1)
    valid = (st1.ravel() == 1) & (st2.ravel() == 1) & np.isfinite(p1_2d[:, 0]) & np.isfinite(p1_2d[:, 1]) & (fb_err < 1.8)
    if np.sum(valid) < 8:
        return pred_mesh, np.ones((rows, cols), dtype=np.float64)

    # Shi-Tomasi 2D cornerness so 1D border lines with aperture ambiguity receive zero weight
    min_eig = cv2.cornerMinEigenVal(prev_gray, blockSize=7, ksize=3)
    eig_smooth = cv2.boxFilter(min_eig, -1, (7, 7))

    delta_obs = p1_2d - pred_2d
    # Robust spatial outlier check against local UV neighborhood median
    valid_idx = np.where(valid)[0]
    uv_v = uv_sub[valid_idx]
    d_v = delta_obs[valid_idx]
    keep_mask = np.ones(len(valid_idx), dtype=bool)
    for i_v in range(len(valid_idx)):
        dist_uv = np.linalg.norm(uv_v - uv_v[i_v], axis=1)
        neigh = dist_uv < 0.22
        if np.sum(neigh) >= 4:
            med = np.median(d_v[neigh], axis=0)
            if float(np.linalg.norm(d_v[i_v] - med)) > 8.0:
                keep_mask[i_v] = False

    final_idx = valid_idx[keep_mask]
    if len(final_idx) < 4:
        return pred_mesh, np.ones((rows, cols), dtype=np.float64)

    # 3. Accumulate bilinear interpolation normal equations AtWA @ delta = AtWb
    uv_f = uv_sub[final_idx]
    d_f = delta_obs[final_idx]
    p0_f = p0_2d[final_idx]
    fb_f = fb_err[final_idx]

    fx = uv_f[:, 0] * (cols - 1)
    fy = uv_f[:, 1] * (rows - 1)
    c0 = np.clip(np.floor(fx).astype(np.int32), 0, cols - 2)
    r0 = np.clip(np.floor(fy).astype(np.int32), 0, rows - 2)
    tx = fx - c0
    ty = fy - r0

    A_mat = np.zeros((len(final_idx), n_pts), dtype=np.float64)
    row_range = np.arange(len(final_idx))
    A_mat[row_range, r0 * cols + c0] = (1.0 - tx) * (1.0 - ty)
    A_mat[row_range, r0 * cols + (c0 + 1)] = tx * (1.0 - ty)
    A_mat[row_range, (r0 + 1) * cols + (c0 + 1)] = tx * ty
    A_mat[row_range, (r0 + 1) * cols + c0] = (1.0 - tx) * ty

    ix = np.clip(np.round(p0_f[:, 0]).astype(np.int32), 0, w - 1)
    iy = np.clip(np.round(p0_f[:, 1]).astype(np.int32), 0, h - 1)
    tex_w = np.clip(eig_smooth[iy, ix].astype(np.float64) / 0.004, 0.0, 1.0)
    if np.sum(tex_w >= 0.05) >= 12:
        tex_w = np.where(tex_w >= 0.05, tex_w, 0.0)
    else:
        tex_w = np.maximum(tex_w, 0.05)
    fb_w = np.exp(-0.5 * (fb_f / 0.9) ** 2)
    sample_w = tex_w * fb_w

    Aw = A_mat * sample_w[:, None]
    sys_mat = (A_mat.T @ Aw) + reg_mat
    sys_mat[np.arange(n_pts), np.arange(n_pts)] += 1e-5
    rhs = Aw.T @ d_f
    delta_solved = np.linalg.solve(sys_mat, rhs)

    next_mesh = pred_mesh + delta_solved.reshape(rows, cols, 2)
    if not np.all(np.isfinite(next_mesh)):
        return pred_mesh, np.ones((rows, cols), dtype=np.float64)

    # Derive per-vertex visibility from supporting bilinear sample density
    vert_support = np.sum(Aw, axis=0).reshape(rows, cols)
    vis_out = np.where(vert_support > 0.05, 1.0, 0.85)
    return next_mesh, vis_out


def smooth_mesh_trajectories(
    meshes: np.ndarray, smoothing_window: int, ref_idx: int
) -> np.ndarray:
    """Apply symmetric shrinking-radius Gaussian smoothing to (N, R, C, 2) mesh vertices."""
    radius = max(0, min(10, int(smoothing_window)))
    n = meshes.shape[0]
    if radius == 0 or n <= 2:
        return meshes

    smoothed = np.empty_like(meshes)
    for t in range(n):
        r_t = min(radius, t, n - 1 - t)
        if t == ref_idx or r_t == 0:
            smoothed[t] = meshes[t]
            continue
        sigma = max(0.5, r_t / 2.0)
        offsets = np.arange(-r_t, r_t + 1)
        weights = np.exp(-0.5 * (offsets / sigma) ** 2)
        weights /= np.sum(weights)
        acc = np.zeros_like(meshes[t])
        for k, w in zip(offsets, weights):
            acc += w * meshes[t + int(k)]
        smoothed[t] = acc
    return smoothed


def track_nonrigid_mesh_cpu(
    video_path: str,
    initial_quad_ass: np.ndarray,
    start_frame: int,
    end_frame: int,
    play_res_x: float,
    play_res_y: float,
    direction: str,
    grid_cols: int = 10,
    grid_rows: int = 6,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Track a (R, C, 2) non-rigid mesh on CPU across [start_frame..end_frame]."""
    if not video_path or not os.path.isfile(video_path):
        raise TrackerError(f"Video file not found: '{video_path}'.")
    if start_frame < 0 or end_frame < start_frame:
        raise TrackerError(
            f"Invalid frame range ({start_frame} to {end_frame}). "
            "Ensure the guide line has a positive duration."
        )

    cols = max(3, min(24, int(grid_cols)))
    rows = max(3, min(24, int(grid_rows)))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise TrackerError(f"OpenCV failed to open video file: '{video_path}'.")

    try:
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if vid_w <= 0 or vid_h <= 0:
            raise TrackerError("Could not determine video dimensions from VideoCapture.")
        if total_frames > 0 and start_frame >= total_frames:
            raise TrackerError(
                f"Start frame {start_frame} exceeds video frame count ({total_frames})."
            )
        if total_frames > 0 and end_frame >= total_frames:
            end_frame = total_frames - 1

        res_x = float(play_res_x) if play_res_x > 0 else float(vid_w)
        res_y = float(play_res_y) if play_res_y > 0 else float(vid_h)
        scale_to_vid = np.array([vid_w / res_x, vid_h / res_y], dtype=np.float64)
        scale_to_ass = np.array([res_x / vid_w, res_y / vid_h], dtype=np.float64)

        init_quad_vid = initial_quad_ass * scale_to_vid
        unit_sq = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        h_init = cv2.getPerspectiveTransform(unit_sq, init_quad_vid.astype(np.float32))
        us = np.linspace(0.0, 1.0, cols, dtype=np.float32)
        vs = np.linspace(0.0, 1.0, rows, dtype=np.float32)
        gx, gy = np.meshgrid(us, vs)
        unit_grid = np.column_stack([gx.ravel(), gy.ravel()]).reshape(-1, 1, 2)
        init_mesh_vid = cv2.perspectiveTransform(unit_grid, h_init).reshape(rows, cols, 2).astype(np.float64)

        num_frames = end_frame - start_frame + 1
        dir_key = normalize_direction(direction)
        ref_idx = 0 if dir_key == DIRECTION_FORWARD else (num_frames - 1)

        meshes_vid = np.empty((num_frames, rows, cols, 2), dtype=np.float64)
        vis_all = np.ones((num_frames, rows, cols), dtype=np.float64)

        if num_frames == 1:
            meshes_vid[0] = init_mesh_vid
            return meshes_vid * scale_to_ass, vis_all, 0

        reg_mat = _build_mesh_regularizer(rows, cols)

        if dir_key == DIRECTION_FORWARD:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
            ok, bgr0 = cap.read()
            if not ok or bgr0 is None:
                raise TrackerError(f"Failed to decode start frame {start_frame} from video.")
            prev_gray = cv2.cvtColor(bgr0, cv2.COLOR_BGR2GRAY)
            curr_mesh = init_mesh_vid.copy()
            meshes_vid[0] = curr_mesh

            for offset in range(1, num_frames):
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    meshes_vid[offset] = curr_mesh.copy()
                    continue
                curr_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                curr_mesh, vis = _track_mesh_step_cpu(prev_gray, curr_gray, curr_mesh, reg_mat)
                meshes_vid[offset] = curr_mesh
                vis_all[offset] = vis
                prev_gray = curr_gray
        else:
            max_chunk_frames = max(16, (512 * 1024 * 1024) // max(1, vid_w * vid_h))
            curr_mesh = init_mesh_vid.copy()
            meshes_vid[num_frames - 1] = curr_mesh
            bridge_gray: np.ndarray | None = None

            chunk_end = num_frames - 1
            while chunk_end >= 0:
                chunk_start = max(0, chunk_end - max_chunk_frames + 1)
                chunk_len = chunk_end - chunk_start + 1
                grays = _read_gray_chunk(cap, start_frame + chunk_start, chunk_len)
                if not grays:
                    raise TrackerError(
                        f"Failed to decode frame range {start_frame + chunk_start}..{start_frame + chunk_end}."
                    )

                if bridge_gray is not None and len(grays) > 0:
                    curr_mesh, vis = _track_mesh_step_cpu(
                        bridge_gray, grays[-1], curr_mesh, reg_mat
                    )
                    meshes_vid[chunk_end] = curr_mesh
                    vis_all[chunk_end] = vis

                for local_idx in range(len(grays) - 2, -1, -1):
                    global_idx = chunk_start + local_idx
                    curr_mesh, vis = _track_mesh_step_cpu(
                        grays[local_idx + 1], grays[local_idx], curr_mesh, reg_mat
                    )
                    meshes_vid[global_idx] = curr_mesh
                    vis_all[global_idx] = vis

                for local_idx in range(len(grays), chunk_len):
                    global_idx = chunk_start + local_idx
                    meshes_vid[global_idx] = curr_mesh.copy()

                bridge_gray = grays[0]
                chunk_end = chunk_start - 1

        return meshes_vid * scale_to_ass, vis_all, ref_idx
    finally:
        cap.release()


def load_ai_mesh_json(
    mesh_json_path: str,
    start_frame: int,
    end_frame: int,
    play_res_x: float,
    play_res_y: float,
) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    """Load an exported Colab CoTracker3 mesh JSON and align to [start_frame..end_frame]."""
    if not mesh_json_path or not os.path.isfile(mesh_json_path):
        raise TrackerError(
            f"AI Mesh JSON file not found: '{mesh_json_path}'.\n"
            "Click 'Choose Mesh JSON...' to select the mesh_track.json exported from Colab."
        )

    try:
        data = json.loads(Path(mesh_json_path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise TrackerError(f"Failed to parse AI Mesh JSON '{mesh_json_path}': {exc}") from exc

    if not isinstance(data, dict) or not isinstance(data.get("frames"), list) or not data["frames"]:
        raise TrackerError(
            "Invalid AI Mesh JSON format: missing non-empty 'frames' array."
        )

    frames_raw: list[dict[str, Any]] = data["frames"]
    n_req = end_frame - start_frame + 1

    by_frame: dict[int, dict[str, Any]] = {}
    for item in frames_raw:
        if isinstance(item, dict) and "frame" in item:
            by_frame[int(item["frame"])] = item

    if all(f in by_frame for f in range(start_frame, end_frame + 1)):
        selected = [by_frame[f] for f in range(start_frame, end_frame + 1)]
        json_ref = int(data.get("ref_frame", start_frame))
        ref_idx = max(0, min(n_req - 1, json_ref - start_frame))
    elif len(frames_raw) == n_req:
        # Trimmed clip exported from Colab (0..N-1) mapped directly onto guide line [start..end]
        selected = frames_raw
        json_start = int(data.get("start_frame", frames_raw[0].get("frame", 0)))
        json_ref = int(data.get("ref_frame", json_start))
        ref_idx = max(0, min(n_req - 1, json_ref - json_start))
    else:
        f_min = frames_raw[0].get("frame", 0)
        f_max = frames_raw[-1].get("frame", len(frames_raw) - 1)
        raise TrackerError(
            f"Frame range mismatch between guide line ({start_frame}..{end_frame}, {n_req} frames) "
            f"and AI Mesh JSON ({f_min}..{f_max}, {len(frames_raw)} frames). "
            "Ensure the guide line duration matches the exported Colab clip length or frame numbers."
        )

    pts_list = []
    vis_list = []
    expected_shape: tuple[int, int] | None = None
    for idx, entry in enumerate(selected):
        pts = np.asarray(entry.get("points"), dtype=np.float64)
        if pts.ndim != 3 or pts.shape[2] != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
            raise TrackerError(
                f"Invalid mesh point grid shape at frame offset {idx}: expected (rows>=2, cols>=2, 2)."
            )
        if expected_shape is None:
            expected_shape = (pts.shape[0], pts.shape[1])
        elif (pts.shape[0], pts.shape[1]) != expected_shape:
            raise TrackerError("Inconsistent mesh grid dimensions across frames in AI Mesh JSON.")
        pts_list.append(pts)

        if "visibility" in entry and entry["visibility"] is not None:
            v_arr = np.asarray(entry["visibility"], dtype=np.float64)
            if v_arr.shape != expected_shape:
                v_arr = np.ones(expected_shape, dtype=np.float64)
            vis_list.append(np.clip(v_arr, 0.0, 1.0))
        else:
            vis_list.append(np.ones(expected_shape, dtype=np.float64))

    meshes = np.stack(pts_list, axis=0)
    visibilities = np.stack(vis_list, axis=0)

    src_w = float(data.get("play_res_x") or 0.0)
    src_h = float(data.get("play_res_y") or 0.0)
    dst_w = float(play_res_x) if play_res_x > 0 else (src_w if src_w > 0 else 1920.0)
    dst_h = float(play_res_y) if play_res_y > 0 else (src_h if src_h > 0 else 1080.0)

    if src_w > 0 and src_h > 0 and (abs(src_w - dst_w) > 1e-3 or abs(src_h - dst_h) > 1e-3):
        meshes[:, :, :, 0] *= dst_w / src_w
        meshes[:, :, :, 1] *= dst_h / src_h

    return meshes, visibilities, ref_idx, dst_w, dst_h


# ---------------------------------------------------------------------------
# Bilinear Mesh Warper, Font Discovery & Glyph Vector Outline Extraction
# ---------------------------------------------------------------------------


class MeshWarper:
    """Fast vectorized bilinear warper over a tracked (R, C, 2) mesh."""

    def __init__(self, grid_pts: np.ndarray, vis: np.ndarray | None = None) -> None:
        self.grid = np.asarray(grid_pts, dtype=np.float64)
        self.rows, self.cols = self.grid.shape[:2]
        if vis is not None:
            self.vis = np.asarray(vis, dtype=np.float64)
        else:
            self.vis = np.ones((self.rows, self.cols), dtype=np.float64)

    def _cell_coords(self, uv_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        fx = uv_pts[:, 0] * (self.cols - 1)
        fy = uv_pts[:, 1] * (self.rows - 1)
        c0 = np.clip(np.floor(fx).astype(np.int32), 0, self.cols - 2)
        r0 = np.clip(np.floor(fy).astype(np.int32), 0, self.rows - 2)
        return r0, c0, fx - c0, fy - r0

    def warp_points(self, uv_pts: np.ndarray) -> np.ndarray:
        """Map (K, 2) UV coordinates in [0, 1]^2 to screen (K, 2) coordinates."""
        if len(uv_pts) == 0:
            return np.empty((0, 2), dtype=np.float64)
        r0, c0, tx, ty = self._cell_coords(uv_pts)
        tx = tx[:, None]
        ty = ty[:, None]
        p00 = self.grid[r0, c0]
        p10 = self.grid[r0, c0 + 1]
        p11 = self.grid[r0 + 1, c0 + 1]
        p01 = self.grid[r0 + 1, c0]
        return (
            (1.0 - tx) * (1.0 - ty) * p00
            + tx * (1.0 - ty) * p10
            + tx * ty * p11
            + (1.0 - tx) * ty * p01
        )

    def cell_visibility_and_orientation(self, uv_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return interpolated visibility and Jacobian determinant (>0 = front-facing CW) per UV point."""
        if len(uv_pts) == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        r0, c0, tx, ty = self._cell_coords(uv_pts)
        v00 = self.vis[r0, c0]
        v10 = self.vis[r0, c0 + 1]
        v11 = self.vis[r0 + 1, c0 + 1]
        v01 = self.vis[r0 + 1, c0]
        vis_out = (
            (1.0 - tx) * (1.0 - ty) * v00
            + tx * (1.0 - ty) * v10
            + tx * ty * v11
            + (1.0 - tx) * ty * v01
        )

        p00 = self.grid[r0, c0]
        p10 = self.grid[r0, c0 + 1]
        p11 = self.grid[r0 + 1, c0 + 1]
        p01 = self.grid[r0 + 1, c0]
        dp_du = (1.0 - ty)[:, None] * (p10 - p00) + ty[:, None] * (p11 - p01)
        dp_dv = (1.0 - tx)[:, None] * (p01 - p00) + tx[:, None] * (p11 - p10)
        det_j = dp_du[:, 0] * dp_dv[:, 1] - dp_du[:, 1] * dp_dv[:, 0]
        return vis_out, det_j


_FONT_CACHE: dict[tuple[str, bool, bool], tuple[str, int]] = {}


def _import_freetype() -> Any:
    try:
        import freetype  # type: ignore

        return freetype
    except ImportError as exc:
        raise TrackerError(
            "Vector / Per-Character mesh output requires 'freetype-py'.\n"
            "Install it with: pip install -r requirements.txt"
        ) from exc


def _inspect_font_file(freetype_mod: Any, path: str, target_lower: str, bold: bool, italic: bool) -> tuple[int, int] | None:
    """Return (best_score, face_index) if font file matches target family, else None."""
    best: tuple[int, int] | None = None
    try:
        face0 = freetype_mod.Face(path, 0)
        num_faces = max(1, int(getattr(face0, "num_faces", 1)))
    except Exception:
        return None

    for face_idx in range(num_faces):
        try:
            face = face0 if face_idx == 0 else freetype_mod.Face(path, face_idx)
            fam = (face.family_name or b"").decode("utf-8", errors="ignore").strip().lower()
            if not fam:
                continue
            if fam == target_lower:
                score = 100
            elif target_lower in fam or fam in target_lower:
                score = 40
            else:
                continue
            is_bold = bool(face.style_flags & freetype_mod.FT_STYLE_FLAG_BOLD)
            is_italic = bool(face.style_flags & freetype_mod.FT_STYLE_FLAG_ITALIC)
            if is_bold == bold:
                score += 20
            if is_italic == italic:
                score += 15
            if best is None or score > best[0]:
                best = (score, face_idx)
        except Exception:
            continue
    return best


def resolve_font_path(family: str, bold: bool = False, italic: bool = False) -> tuple[str, int]:
    """Locate a system TrueType/OpenType font file and face index for (family, bold, italic)."""
    clean_family = (family or "Arial").strip().lstrip("@")
    key = (clean_family.lower(), bool(bold), bool(italic))
    if key in _FONT_CACHE and os.path.isfile(_FONT_CACHE[key][0]):
        return _FONT_CACHE[key]

    ft = _import_freetype()
    if os.path.isfile(clean_family):
        _FONT_CACHE[key] = (clean_family, 0)
        return clean_family, 0

    candidates: list[str] = []

    if sys.platform == "win32":
        import winreg

        font_dirs = [
            Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts",
        ]
        reg_paths = [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"),
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"),
        ]
        target_toks = [t for t in re.split(r"[\s_-]+", key[0]) if t]
        for hive, subkey in reg_paths:
            try:
                with winreg.OpenKey(hive, subkey) as reg_key:
                    n_vals = winreg.QueryInfoKey(reg_key)[1]
                    for i in range(n_vals):
                        name, val, _ = winreg.EnumValue(reg_key, i)
                        name_l = name.lower()
                        if any(tok in name_l for tok in target_toks):
                            p = Path(str(val))
                            if not p.is_absolute():
                                for fdir in font_dirs:
                                    cand = fdir / p
                                    if cand.is_file():
                                        candidates.append(str(cand))
                                        break
                            elif p.is_file():
                                candidates.append(str(p))
            except OSError:
                continue
    else:
        style_parts = [clean_family]
        if bold:
            style_parts.append("bold")
        if italic:
            style_parts.append("italic")
        try:
            proc = subprocess.run(
                ["fc-match", "-f", "%{file}|%{index}", ":".join(style_parts)],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if proc.returncode == 0 and "|" in proc.stdout:
                fpath, idx_str = proc.stdout.strip().split("|", 1)
                if os.path.isfile(fpath):
                    res = (fpath, int(idx_str or 0))
                    _FONT_CACHE[key] = res
                    return res
        except Exception:
            pass

    best_match: tuple[int, str, int] | None = None
    for cand_path in dict.fromkeys(candidates):
        inspected = _inspect_font_file(ft, cand_path, key[0], bold, italic)
        if inspected is not None:
            score, fidx = inspected
            if best_match is None or score > best_match[0]:
                best_match = (score, cand_path, fidx)

    if best_match is not None:
        res = (best_match[1], best_match[2])
        _FONT_CACHE[key] = res
        return res

    # Universal fallback fonts across Windows / Linux / macOS
    fallbacks = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\msgothic.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for fb in fallbacks:
        if os.path.isfile(fb):
            _FONT_CACHE[key] = (fb, 0)
            return fb, 0

    raise TrackerError(
        f"Could not find font '{clean_family}' or a fallback system font on this machine."
    )


def _split_ass_runs(guide_text: str, base_style: dict[str, Any]) -> list[dict[str, Any]]:
    """Split ASS dialogue text into styled visible runs, expanding \\N/\\n/\\h."""
    state = {
        "fontname": str(base_style.get("fontname") or "Arial"),
        "fontsize": float(base_style.get("fontsize") or 48.0),
        "bold": bool(base_style.get("bold", False)),
        "italic": bool(base_style.get("italic", False)),
        "spacing": float(base_style.get("spacing") or 0.0),
        "raw_override_prefix": "",
    }
    runs: list[dict[str, Any]] = []
    i = 0
    n = len(guide_text)
    while i < n:
        if guide_text[i] == "{":
            close = guide_text.find("}", i + 1)
            if close == -1:
                break
            inner = guide_text[i + 1 : close]
            if "\\" in inner:
                first_bs = inner.find("\\")
                rest = inner[first_bs:]
                tags: list[str] = []
                start = 0
                depth = 0
                for idx, ch in enumerate(rest):
                    if ch == "(":
                        depth += 1
                    elif ch == ")" and depth > 0:
                        depth -= 1
                    elif ch == "\\" and depth == 0 and idx > start:
                        tags.append(rest[start:idx])
                        start = idx
                tags.append(rest[start:])
                kept_style = []
                for t in tags:
                    if t.startswith("\\fn") and len(t) > 3:
                        state["fontname"] = t[3:].strip()
                    elif re.match(r"^\\fs\s*\d", t):
                        m = re.match(r"^\\fs\s*([0-9.]+)", t)
                        if m:
                            state["fontsize"] = max(4.0, float(m.group(1)))
                    elif re.match(r"^\\fsp\s*[-+0-9.]", t):
                        m = re.match(r"^\\fsp\s*([-+]?[0-9.]+)", t)
                        if m:
                            state["spacing"] = float(m.group(1))
                    elif re.match(r"^\\b\s*\d+$", t):
                        val = int(re.sub(r"\D", "", t) or "0")
                        state["bold"] = val != 0
                    elif re.match(r"^\\i\s*\d+$", t):
                        val = int(re.sub(r"\D", "", t) or "0")
                        state["italic"] = val != 0
                    if not _should_strip_tag(t):
                        kept_style.append(t)
                if kept_style:
                    state["raw_override_prefix"] += "".join(kept_style)
            i = close + 1
        else:
            nxt = guide_text.find("{", i)
            seg = guide_text[i:] if nxt == -1 else guide_text[i:nxt]
            seg = seg.replace("\\h", " ")
            # Split hard/soft line breaks \N and \n
            parts = re.split(r"(\\[Nn])", seg)
            for part in parts:
                if part in ("\\N", "\\n"):
                    runs.append({"newline": True})
                elif part:
                    runs.append({"newline": False, "text": part, **state})
            if nxt == -1:
                break
            i = nxt
    return runs


def _try_import_uharfbuzz() -> Any | None:
    """Import uharfbuzz if installed for full OpenType Arabic/Latin shaping."""
    try:
        import uharfbuzz as hb  # type: ignore

        return hb
    except ImportError:
        return None


_MANDATORY_LAM_ALEF = {"\u0644\u0622", "\u0644\u0623", "\u0644\u0625", "\u0644\u0627"}
_ARABIC_BRACKET_MIRRORS = {
    "(": ")",
    ")": "(",
    "[": "]",
    "]": "[",
    "{": "}",
    "}": "{",
    "<": ">",
    ">": "<",
    "«": "»",
    "»": "«",
}


def _build_arabic_forms() -> dict[str, dict[str, str]]:
    """Build contextual Arabic Presentation Forms map using stdlib unicodedata."""
    forms: dict[str, dict[str, str]] = {}
    for cp in range(0xFB50, 0xFEFD):
        ch = chr(cp)
        dec = unicodedata.decomposition(ch)
        if not dec:
            continue
        parts = dec.split()
        tag = parts[0].strip("<>")
        if tag in ("isolated", "final", "initial", "medial") and len(parts) == 2:
            base = chr(int(parts[1], 16))
            # Prefer standard Presentation Forms-B (0xFE70..0xFEFF) over Forms-A when both exist.
            slot = forms.setdefault(base, {})
            if tag not in slot or cp >= 0xFE70:
                slot[tag] = ch
        elif tag in ("isolated", "final") and len(parts) == 3:
            pair = chr(int(parts[1], 16)) + chr(int(parts[2], 16))
            if pair in _MANDATORY_LAM_ALEF:
                forms.setdefault(pair, {})[tag] = ch
    return forms


_ARABIC_FORMS = _build_arabic_forms()


def _is_arabic_tashkeel(ch: str) -> bool:
    cp = ord(ch)
    return (0x064B <= cp <= 0x065F) or cp == 0x0670 or (0x06D6 <= cp <= 0x06ED)


def _is_arabic_letter_or_form(ch: str) -> bool:
    cp = ord(ch)
    if (0x0660 <= cp <= 0x0669) or (0x06F0 <= cp <= 0x06F9):
        return False  # Digits are LTR
    return (
        (0x0600 <= cp <= 0x06FF)
        or (0x0750 <= cp <= 0x077F)
        or (0x08A0 <= cp <= 0x08FF)
        or (0xFB50 <= cp <= 0xFDFF)
        or (0xFE70 <= cp <= 0xFEFF)
    )


def _is_ltr_char(ch: str) -> bool:
    cp = ord(ch)
    if (0x0660 <= cp <= 0x0669) or (0x06F0 <= cp <= 0x06F9):
        return True
    if "0" <= ch <= "9" or "A" <= ch <= "Z" or "a" <= ch <= "z":
        return True
    return 0x00C0 <= cp <= 0x024F


def _is_already_visual_arabic(text: str) -> bool:
    """Return True if text already uses Presentation Forms-B without logical base Arabic."""
    has_pf = any((0xFB50 <= ord(c) <= 0xFDFF) or (0xFE70 <= ord(c) <= 0xFEFF) for c in text)
    has_base = any((0x0621 <= ord(c) <= 0x064A) or (0x0671 <= ord(c) <= 0x06D5) for c in text)
    return has_pf and not has_base


def _split_bidi_runs(text: str) -> list[tuple[str, bool]]:
    """Split text into visual runs (segment, is_rtl) assuming RTL paragraph base direction."""
    n = len(text)
    if n == 0:
        return []
    types: list[str] = []
    for ch in text:
        if _is_arabic_letter_or_form(ch):
            types.append("R")
        elif _is_ltr_char(ch):
            types.append("L")
        else:
            types.append("N")

    # Resolve neutral chars flanked by LTR tokens on both sides to LTR.
    resolved = list(types)
    for i, tp in enumerate(types):
        if tp != "N":
            continue
        prev_strong = None
        for j in range(i - 1, -1, -1):
            if types[j] in ("L", "R"):
                prev_strong = types[j]
                break
        next_strong = None
        for j in range(i + 1, n):
            if types[j] in ("L", "R"):
                next_strong = types[j]
                break
        if prev_strong == "L" and next_strong == "L":
            resolved[i] = "L"
        else:
            resolved[i] = "R"

    runs: list[tuple[str, bool]] = []
    start = 0
    for i in range(1, n + 1):
        if i == n or resolved[i] != resolved[start]:
            runs.append((text[start:i], resolved[start] == "R"))
            start = i
    return runs


def _reshape_arabic_rtl_segment(seg: str) -> str:
    """Apply Arabic contextual joining + Lam-Alef ligatures on a logical RTL segment."""
    # Group base units + attached combining marks (Tashkeel) and collapse Lam + Alef ligatures.
    units: list[tuple[str, str]] = []  # (base_key, trailing_marks)
    i = 0
    n = len(seg)
    while i < n:
        ch = seg[i]
        if _is_arabic_tashkeel(ch):
            if units:
                b, m = units[-1]
                units[-1] = (b, m + ch)
            else:
                units.append((ch, ""))
            i += 1
            continue

        marks = ""
        j = i + 1
        while j < n and _is_arabic_tashkeel(seg[j]):
            marks += seg[j]
            j += 1

        # Mandatory Lam + Alef ligature check
        if ch == "\u0644" and j < n and (ch + seg[j]) in _MANDATORY_LAM_ALEF:
            lig = ch + seg[j]
            j2 = j + 1
            while j2 < n and _is_arabic_tashkeel(seg[j2]):
                marks += seg[j2]
                j2 += 1
            units.append((lig, marks))
            i = j2
        else:
            units.append((ch, marks))
            i = j

    def can_join_right(base: str) -> bool:
        if base == "\u0640":
            return True
        f = _ARABIC_FORMS.get(base)
        return bool(f and ("final" in f or "medial" in f))

    def can_join_left(base: str) -> bool:
        if base == "\u0640":
            return True
        f = _ARABIC_FORMS.get(base)
        return bool(f and ("initial" in f or "medial" in f))

    out: list[str] = []
    m_len = len(units)
    for idx, (base, marks) in enumerate(units):
        f = _ARABIC_FORMS.get(base)
        if not f:
            out.append(_ARABIC_BRACKET_MIRRORS.get(base, base) + marks)
            continue

        prev_ok = idx > 0 and can_join_left(units[idx - 1][0])
        next_ok = idx + 1 < m_len and can_join_right(units[idx + 1][0])

        if prev_ok and next_ok and "medial" in f:
            glyph = f["medial"]
        elif prev_ok and "final" in f:
            glyph = f["final"]
        elif next_ok and "initial" in f:
            glyph = f["initial"]
        else:
            glyph = f.get("isolated", base)
        out.append(glyph + marks)

    return "".join(out)


def reshape_arabic_to_visual(text: str) -> str:
    """Convert logical Arabic text to Presentation Forms-B in visual LTR order (e.g. 'صباح الخير' -> 'ﺮﻴﺨﻟﺍ ﺡﺎﺒﺻ')."""
    if not text or _is_already_visual_arabic(text):
        return text

    runs = _split_bidi_runs(text)
    visual_parts: list[str] = []
    for seg, is_rtl in reversed(runs):
        if is_rtl:
            shaped = _reshape_arabic_rtl_segment(seg)
            # Reverse characters within RTL run while keeping any combining mark attached to its base.
            clusters: list[str] = []
            for ch in shaped:
                if _is_arabic_tashkeel(ch) and clusters:
                    clusters[-1] += ch
                else:
                    clusters.append(ch)
            visual_parts.append("".join(reversed(clusters)))
        else:
            visual_parts.append(seg)
    return "".join(visual_parts)


def _subdivide_outline_to_polylines(
    outline: Any, scale: float, pen_x: float, baseline_y: float, max_seg_len: float
) -> list[np.ndarray]:
    """Convert a FreeType FT_LOAD_NO_SCALE outline (raw EM units) into subdivided ASS-space polylines."""
    pts_raw = outline.points
    tags = outline.tags
    contours = outline.contours
    if not pts_raw or not contours:
        return []

    pts_ass = [
        np.array([pen_x + float(p[0]) * scale, baseline_y - float(p[1]) * scale], dtype=np.float64)
        for p in pts_raw
    ]

    polylines: list[np.ndarray] = []
    start_idx = 0
    for end_idx in contours:
        c_pts = pts_ass[start_idx : end_idx + 1]
        c_tgs = tags[start_idx : end_idx + 1]
        start_idx = end_idx + 1
        n_c = len(c_pts)
        if n_c < 2:
            continue

        # Normalize rotation so contour walks cleanly from an on-curve start point
        if c_tgs[0] & 1:
            curr = c_pts[0]
            ordered = [(c_pts[k], c_tgs[k]) for k in range(1, n_c)] + [(c_pts[0], c_tgs[0])]
        elif c_tgs[-1] & 1:
            curr = c_pts[-1]
            ordered = [(c_pts[k], c_tgs[k]) for k in range(n_c)]
        else:
            curr = 0.5 * (c_pts[-1] + c_pts[0])
            ordered = [(c_pts[k], c_tgs[k]) for k in range(n_c)] + [(curr, 1)]

        poly: list[np.ndarray] = [curr]
        idx = 0
        m = len(ordered)
        while idx < m:
            p_cur, t_cur = ordered[idx]
            if t_cur & 1:
                dist = float(np.linalg.norm(p_cur - curr))
                n_seg = max(1, int(math.ceil(dist / max_seg_len)))
                for s in range(1, n_seg + 1):
                    t = s / n_seg
                    poly.append((1.0 - t) * curr + t * p_cur)
                curr = p_cur
                idx += 1
            elif (t_cur & 2) == 0:
                # Conic quadratic Bezier
                if idx + 1 < m:
                    p_next, t_next = ordered[idx + 1]
                    if t_next & 1:
                        p_end = p_next
                        idx += 2
                    else:
                        p_end = 0.5 * (p_cur + p_next)
                        idx += 1
                else:
                    p_end = ordered[0][0]
                    idx += 1
                approx_len = 0.5 * (
                    float(np.linalg.norm(p_cur - curr))
                    + float(np.linalg.norm(p_end - p_cur))
                    + float(np.linalg.norm(p_end - curr))
                )
                n_seg = max(4, int(math.ceil(approx_len / max_seg_len)))
                for s in range(1, n_seg + 1):
                    t = s / n_seg
                    mt = 1.0 - t
                    poly.append((mt * mt) * curr + (2.0 * mt * t) * p_cur + (t * t) * p_end)
                curr = p_end
            else:
                # Cubic Bezier (OpenType CFF)
                if idx + 2 < m:
                    p_c2, _ = ordered[idx + 1]
                    p_end, _ = ordered[idx + 2]
                    idx += 3
                elif idx + 1 < m:
                    p_c2 = p_cur
                    p_end, _ = ordered[idx + 1]
                    idx += 2
                else:
                    p_c2 = p_cur
                    p_end = ordered[0][0]
                    idx += 1
                approx_len = (
                    float(np.linalg.norm(p_cur - curr))
                    + float(np.linalg.norm(p_c2 - p_cur))
                    + float(np.linalg.norm(p_end - p_c2))
                )
                n_seg = max(5, int(math.ceil(approx_len / max_seg_len)))
                for s in range(1, n_seg + 1):
                    t = s / n_seg
                    mt = 1.0 - t
                    poly.append(
                        (mt ** 3) * curr
                        + 3.0 * (mt ** 2) * t * p_cur
                        + 3.0 * mt * (t ** 2) * p_c2
                        + (t ** 3) * p_end
                    )
                curr = p_end

        if len(poly) >= 3:
            polylines.append(np.asarray(poly[:-1], dtype=np.float64))
    return polylines


def _resolve_layout_font(
    run: dict[str, Any], font_file_path: str
) -> tuple[str, int]:
    """Use user-selected font_file_path if given, else resolve from system fonts."""
    if font_file_path:
        if not os.path.isfile(font_file_path):
            raise TrackerError(
                f"Selected font file not found: '{font_file_path}'.\n"
                "Please click 'Choose Font...' to select a valid .ttf / .otf / .ttc file."
            )
        return font_file_path, 0
    return resolve_font_path(run["fontname"], run["bold"], run["italic"])


def extract_text_vector_layout(
    guide_text: str,
    style_info: dict[str, Any] | None = None,
    font_file_path: str = "",
    arabic_text: bool = False,
    use_harfbuzz: bool = True,
) -> dict[str, Any]:
    """Extract subdivided glyph vector contours and typographic cell boxes for ASS guide text.

    Uses uharfbuzz (when use_harfbuzz=True and available) + FreeType FT_LOAD_NO_SCALE for
    exact OpenType cursive Arabic/ligature shaping, and falls back to the built-in Arabic
    Presentation Forms-B reshaper (which is also used directly for Option B per-char lines).
    """
    ft = _import_freetype()
    hb = _try_import_uharfbuzz() if use_harfbuzz else None
    base_style = style_info or {}
    runs = _split_ass_runs(guide_text, base_style)

    # Group runs into lines
    lines: list[list[dict[str, Any]]] = [[]]
    for run in runs:
        if run.get("newline"):
            lines.append([])
        else:
            lines[-1].append(run)

    chars_out: list[dict[str, Any]] = []
    y_cursor = 0.0
    hb_cache: dict[tuple[str, int], Any] = {}
    ft_cache: dict[tuple[str, int], Any] = {}

    for line_runs in lines:
        if arabic_text and len(line_runs) > 1 and not _is_already_visual_arabic(
            "".join(r["text"] for r in line_runs)
        ):
            line_runs = list(reversed(line_runs))

        pen_x = 0.0
        max_lh = 0.0
        max_asc = 0.0
        line_chars: list[dict[str, Any]] = []
        prev_glyph_idx: int | None = None
        prev_face_key: tuple[str, int, float] | None = None

        for run in line_runs:
            fsize = float(run["fontsize"])
            fpath, fidx = _resolve_layout_font(run, font_file_path)
            if (fpath, fidx) not in ft_cache:
                ft_cache[(fpath, fidx)] = ft.Face(fpath, fidx)
            face = ft_cache[(fpath, fidx)]

            units_h = max(1.0, float(face.ascender - face.descender))
            scale = fsize / units_h
            asc_px = float(face.ascender) * scale
            lh_px = fsize
            max_lh = max(max_lh, lh_px)
            max_asc = max(max_asc, asc_px)
            face_key = (fpath, fidx, fsize)

            raw_text = run["text"]
            use_hb_for_run = hb is not None and not (arabic_text and _is_already_visual_arabic(raw_text))

            if use_hb_for_run:
                if (fpath, fidx) not in hb_cache:
                    blob = hb.Blob.from_file_path(fpath)
                    hb_face = hb.Face(blob, fidx)
                    hb_cache[(fpath, fidx)] = hb.Font(hb_face)
                hb_font = hb_cache[(fpath, fidx)]

                if arabic_text:
                    bidi_segs = list(reversed(_split_bidi_runs(raw_text)))
                else:
                    bidi_segs = [(raw_text, False)]

                for seg_text, is_rtl in bidi_segs:
                    if not seg_text:
                        continue
                    buf = hb.Buffer()
                    buf.add_str(seg_text)
                    if is_rtl:
                        buf.direction = "rtl"
                        buf.script = "Arab"
                    else:
                        buf.guess_segment_properties()
                        buf.direction = "ltr"
                    hb.shape(hb_font, buf)
                    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
                        gid = int(info.codepoint)
                        face.load_glyph(gid, ft.FT_LOAD_NO_SCALE | ft.FT_LOAD_NO_BITMAP)
                        adv_px = float(pos.x_advance) * scale + float(run["spacing"])
                        x_off = float(pos.x_offset) * scale
                        y_off = float(pos.y_offset) * scale
                        outline = face.glyph.outline
                        cl_idx = int(info.cluster)
                        ch_repr = seg_text[cl_idx] if 0 <= cl_idx < len(seg_text) else " "
                        line_chars.append(
                            {
                                "char": ch_repr,
                                "pen_x": pen_x + x_off,
                                "y_off": y_off,
                                "adv_px": max(1.0, adv_px),
                                "scale": scale,
                                "outline_pts": list(outline.points),
                                "outline_tags": list(outline.tags),
                                "outline_contours": list(outline.contours),
                                "max_seg_len": max(2.0, fsize * 0.075),
                                "style_prefix": run["raw_override_prefix"],
                                "fontname": run["fontname"],
                                "fontsize": fsize,
                                "bold": run["bold"],
                                "italic": run["italic"],
                            }
                        )
                        pen_x += adv_px
                prev_glyph_idx = None
                prev_face_key = face_key
            else:
                # Per-character path (Option B native chars or fallback when uharfbuzz absent)
                render_text = reshape_arabic_to_visual(raw_text) if arabic_text else raw_text
                for ch in render_text:
                    g_idx = face.get_char_index(ch)
                    if prev_face_key == face_key and prev_glyph_idx is not None and face.has_kerning:
                        kern = face.get_kerning(prev_glyph_idx, g_idx, ft.FT_KERNING_UNSCALED)
                        pen_x += float(kern.x) * scale
                    face.load_char(ch, ft.FT_LOAD_NO_SCALE | ft.FT_LOAD_NO_BITMAP)
                    adv_px = float(face.glyph.advance.x) * scale + float(run["spacing"])
                    outline = face.glyph.outline
                    line_chars.append(
                        {
                            "char": ch,
                            "pen_x": pen_x,
                            "y_off": 0.0,
                            "adv_px": max(1.0, adv_px),
                            "scale": scale,
                            "outline_pts": list(outline.points),
                            "outline_tags": list(outline.tags),
                            "outline_contours": list(outline.contours),
                            "max_seg_len": max(2.0, fsize * 0.075),
                            "style_prefix": run["raw_override_prefix"],
                            "fontname": run["fontname"],
                            "fontsize": fsize,
                            "bold": run["bold"],
                            "italic": run["italic"],
                        }
                    )
                    pen_x += adv_px
                    prev_glyph_idx = g_idx
                    prev_face_key = face_key

        if not line_chars:
            default_lh = float(base_style.get("fontsize") or 48.0)
            y_cursor += default_lh
            continue

        line_w = pen_x
        line_top = y_cursor
        baseline_y = line_top + max_asc
        y_cursor += max_lh

        # Center each line horizontally around x = 0
        x_offset = -0.5 * line_w
        for cinfo in line_chars:
            px0 = x_offset + cinfo["pen_x"]

            class _OutProxy:
                points = cinfo["outline_pts"]
                tags = cinfo["outline_tags"]
                contours = cinfo["outline_contours"]

            polys = _subdivide_outline_to_polylines(
                _OutProxy,
                cinfo["scale"],
                px0,
                baseline_y - cinfo["y_off"],
                cinfo["max_seg_len"],
            )
            chars_out.append(
                {
                    "char": cinfo["char"],
                    "cell_box": (px0, line_top, px0 + cinfo["adv_px"], line_top + max_lh),
                    "contours": polys,
                    "style_prefix": cinfo["style_prefix"],
                    "fontname": cinfo["fontname"],
                    "fontsize": cinfo["fontsize"],
                    "bold": cinfo["bold"],
                    "italic": cinfo["italic"],
                }
            )

    if not chars_out or not any(c["contours"] for c in chars_out):
        raise TrackerError(
            "Guide line contains no renderable text glyphs to convert to vector paths."
        )

    all_x = [c["cell_box"][0] for c in chars_out] + [c["cell_box"][2] for c in chars_out]
    all_y = [c["cell_box"][1] for c in chars_out] + [c["cell_box"][3] for c in chars_out]
    bbox = (min(all_x), min(all_y), max(all_x), max(all_y))
    return {"chars": chars_out, "bbox": bbox}


def map_layout_to_mesh_uv(
    layout: dict[str, Any],
    ref_mesh: np.ndarray,
    play_res_x: float,
    play_res_y: float,
    fov_factor: float,
    aspect_mode: str,
) -> tuple[dict[str, Any], float]:
    """Map layout coordinates into normalized [0, 1]^2 mesh UV space, returning (uv_layout, fit_scale)."""
    ref_quad = _mesh_outer_quad(ref_mesh)
    aspect_key = normalize_aspect_mode(aspect_mode)

    if aspect_key == ASPECT_AUTO:
        f_cam = max(0.1, float(fov_factor)) * max(1.0, float(play_res_x))
        cam_center = np.array([play_res_x * 0.5, play_res_y * 0.5], dtype=np.float64)
        c2d = _intersect_diagonals(ref_quad)
        pts_cam, _ = _reconstruct_3d_corners_from_pinhole(ref_quad, c2d, cam_center, f_cam)
        plane_w = float(np.linalg.norm(pts_cam[1] - pts_cam[0]))
        plane_h = float(np.linalg.norm(pts_cam[3] - pts_cam[0]))
    else:
        plane_w = 0.5 * (
            float(np.linalg.norm(ref_quad[1] - ref_quad[0]))
            + float(np.linalg.norm(ref_quad[2] - ref_quad[3]))
        )
        plane_h = 0.5 * (
            float(np.linalg.norm(ref_quad[3] - ref_quad[0]))
            + float(np.linalg.norm(ref_quad[2] - ref_quad[1]))
        )

    plane_w = max(plane_w, 8.0)
    plane_h = max(plane_h, 8.0)

    min_x, min_y, max_x, max_y = layout["bbox"]
    box_w = max(1.0, max_x - min_x)
    box_h = max(1.0, max_y - min_y)
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)

    fit_scale = 1.0
    if box_w > 0.92 * plane_w or box_h > 0.92 * plane_h:
        fit_scale = min((0.92 * plane_w) / box_w, (0.92 * plane_h) / box_h)

    def to_uv(xy: np.ndarray) -> np.ndarray:
        u = 0.5 + ((xy[:, 0] - cx) * fit_scale) / plane_w
        v = 0.5 + ((xy[:, 1] - cy) * fit_scale) / plane_h
        return np.column_stack([u, v])

    uv_chars = []
    for c in layout["chars"]:
        x0, y0, x1, y1 = c["cell_box"]
        box_uv = (
            0.5 + ((x0 - cx) * fit_scale) / plane_w,
            0.5 + ((y0 - cy) * fit_scale) / plane_h,
            0.5 + ((x1 - cx) * fit_scale) / plane_w,
            0.5 + ((y1 - cy) * fit_scale) / plane_h,
        )
        uv_contours = [to_uv(poly) for poly in c["contours"]]
        uv_chars.append(
            {
                **c,
                "cell_uv": box_uv,
                "cell_size_px": (max(1.0, (x1 - x0) * fit_scale), max(1.0, (y1 - y0) * fit_scale)),
                "uv_contours": uv_contours,
            }
        )
    return {"chars": uv_chars, "plane_size": (plane_w, plane_h)}, fit_scale


# ---------------------------------------------------------------------------
# Three Distortion Output Modes (Options A, B, C)
# ---------------------------------------------------------------------------

_VECTOR_DRAWING_STRIP_PATTERNS = [
    *_STRIP_TAG_PATTERNS,
    re.compile(r"^\\fn.*$"),
    re.compile(r"^\\fs\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)?\s*$"),
    re.compile(r"^\\fsp\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)?\s*$"),
    re.compile(r"^\\fe\s*\d+\s*$"),
    re.compile(r"^\\b\s*\d*\s*$"),
    re.compile(r"^\\i\s*\d*\s*$"),
    re.compile(r"^\\u\s*\d*\s*$"),
    re.compile(r"^\\s\s*\d*\s*$"),
    re.compile(r"^\\p\s*\d+\s*$"),
]


def _extract_vector_drawing_style_tags(guide_text: str) -> str:
    """Keep visual color/border/shadow/blur/alpha tags from leading override block for \\p4 drawing."""
    if not guide_text.startswith("{"):
        return ""
    close = guide_text.find("}")
    if close == -1:
        return ""
    inner = guide_text[1:close]
    if "\\" not in inner:
        return ""
    rest = inner[inner.find("\\") :]
    tags: list[str] = []
    start = 0
    depth = 0
    for idx, ch in enumerate(rest):
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
        elif ch == "\\" and depth == 0 and idx > start:
            tags.append(rest[start:idx])
            start = idx
    tags.append(rest[start:])
    kept = [t for t in tags if not any(pat.match(t) for pat in _VECTOR_DRAWING_STRIP_PATTERNS)]
    return "".join(kept)


def apply_option_a_vector_warp(
    meshes: np.ndarray,
    visibilities: np.ndarray,
    uv_layout: dict[str, Any],
    guide_text: str,
    start_frame: int,
) -> list[dict[str, Any]]:
    """Option A: Warp subdivided glyph outlines through the tracked mesh into ASS \\p4 drawings."""
    style_tags = _extract_vector_drawing_style_tags(guide_text)
    header = f"{{\\an7\\pos(0,0){style_tags}\\p4}}"
    sub_scale = 8.0  # ASS \p4 coordinate multiplier = 2^(4-1) = 8

    out: list[dict[str, Any]] = []
    for t in range(meshes.shape[0]):
        warper = MeshWarper(meshes[t], visibilities[t])
        cmd_parts: list[str] = []
        for ch in uv_layout["chars"]:
            for uv_poly in ch["uv_contours"]:
                if len(uv_poly) < 3:
                    continue
                vis_vals, det_vals = warper.cell_visibility_and_orientation(uv_poly)
                if float(np.mean(vis_vals)) < 0.2 or float(np.mean(det_vals <= 0.0)) > 0.75:
                    continue
                pts_scr = warper.warp_points(uv_poly)
                pts_int = np.round(pts_scr * sub_scale).astype(np.int64)
                dedup: list[tuple[int, int]] = []
                for x_i, y_i in pts_int:
                    pt_t = (int(x_i), int(y_i))
                    if not dedup or pt_t != dedup[-1]:
                        dedup.append(pt_t)
                if len(dedup) >= 2 and dedup[-1] == dedup[0]:
                    dedup.pop()
                if len(dedup) < 3:
                    continue
                coords_str = " ".join(f"{px} {py}" for px, py in dedup[1:])
                cmd_parts.append(f"m {dedup[0][0]} {dedup[0][1]} l {coords_str}")

        if not cmd_parts:
            continue
        out.append(
            {
                "frame": start_frame + t,
                "text": header + " ".join(cmd_parts),
                "effect": "mesh_vector_p4",
            }
        )
    return out


def _is_valid_cw_quad(quad: np.ndarray) -> bool:
    if not np.all(np.isfinite(quad)):
        return False
    if _polygon_signed_area(quad) <= 2.0:
        return False
    for i in range(4):
        e1 = quad[(i + 1) % 4] - quad[i]
        e2 = quad[(i + 2) % 4] - quad[(i + 1) % 4]
        if (e1[0] * e2[1] - e1[1] * e2[0]) <= 0.0:
            return False
    return True


def apply_option_b_per_char(
    meshes: np.ndarray,
    visibilities: np.ndarray,
    uv_layout: dict[str, Any],
    fit_scale: float,
    guide_text: str,
    play_res_x: float,
    play_res_y: float,
    fov_factor: float,
    aspect_mode: str,
    ref_idx: int,
    start_frame: int,
) -> list[dict[str, Any]]:
    """Option B: Emit native-font per-character lines with local mesh-solved 3D ASS tags."""
    n_frames = meshes.shape[0]
    warpers = [MeshWarper(meshes[t], visibilities[t]) for t in range(n_frames)]
    base_prefix = _extract_vector_drawing_style_tags(guide_text)
    out: list[dict[str, Any]] = []

    for ch in uv_layout["chars"]:
        if not ch["contours"]:
            continue
        u0, v0, u1, v1 = ch["cell_uv"]
        if u1 - u0 < 1e-4:
            u1 = u0 + 1e-4
        if v1 - v0 < 1e-4:
            v1 = v0 + 1e-4
        cell_uv_corners = np.array(
            [[u0, v0], [u1, v0], [u1, v1], [u0, v1]], dtype=np.float64
        )

        # Collect contiguous valid visible frame segments for this character
        segments: list[list[tuple[int, np.ndarray]]] = []
        curr_seg: list[tuple[int, np.ndarray]] = []
        for t in range(n_frames):
            vis_c, _ = warpers[t].cell_visibility_and_orientation(cell_uv_corners)
            q_t = warpers[t].warp_points(cell_uv_corners)
            if float(np.mean(vis_c)) >= 0.2 and _is_valid_cw_quad(q_t):
                curr_seg.append((t, q_t))
            else:
                if curr_seg:
                    segments.append(curr_seg)
                    curr_seg = []
        if curr_seg:
            segments.append(curr_seg)

        char_tags = ch["style_prefix"] or base_prefix
        char_guide = f"{{{char_tags}}}{ch['char']}" if char_tags else ch["char"]
        # Ensure fontname/fontsize from style are explicitly preserved if guide had none
        if "\\fn" not in char_guide:
            char_guide = f"{{\\fn{ch['fontname']}\\fs{ch['fontsize']:.2f}" + (
                char_guide[1:] if char_guide.startswith("{") else f"}}{char_guide}"
            )

        for seg in segments:
            seg_frames = [item[0] for item in seg]
            seg_quads = [item[1] for item in seg]
            seg_ref = seg_frames.index(ref_idx) if ref_idx in seg_frames else 0
            params = decompose_quad_to_ass_params(
                quads=seg_quads,
                play_res_x=play_res_x,
                play_res_y=play_res_y,
                fov_factor=fov_factor,
                aspect_mode=aspect_mode,
                ref_idx=seg_ref,
            )
            for t, p in zip(seg_frames, params):
                p_scaled = dict(p)
                p_scaled["fscx"] *= fit_scale
                p_scaled["fscy"] *= fit_scale
                out.append(
                    {
                        "frame": start_frame + t,
                        "text": format_ass_line(char_guide, p_scaled),
                        "effect": "mesh_per_char",
                    }
                )

    out.sort(key=lambda item: item["frame"])
    return out


def _fit_panel_homography(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray | None:
    """Fit 3x3 homography from src_pts (K, 2) to dst_pts (K, 2) with affine fallback."""
    if len(src_pts) >= 4:
        h_mat, inliers = cv2.findHomography(
            src_pts.astype(np.float32),
            dst_pts.astype(np.float32),
            cv2.RANSAC,
            ransacReprojThreshold=2.5,
            maxIters=1000,
            confidence=0.99,
        )
        if h_mat is not None and inliers is not None and np.sum(inliers) >= 4:
            mask = inliers.ravel().astype(bool)
            h_ls, _ = cv2.findHomography(
                src_pts[mask].astype(np.float32), dst_pts[mask].astype(np.float32), 0
            )
            if h_ls is not None and np.all(np.isfinite(h_ls)):
                return h_ls.astype(np.float64)
            if np.all(np.isfinite(h_mat)):
                return h_mat.astype(np.float64)
    if len(src_pts) >= 3:
        aff, _ = cv2.estimateAffine2D(src_pts.astype(np.float32), dst_pts.astype(np.float32))
        if aff is not None and np.all(np.isfinite(aff)):
            return np.vstack([aff.astype(np.float64), [0.0, 0.0, 1.0]])
    return None


def detect_fold_crease(meshes: np.ndarray, ref_idx: int) -> tuple[str, int]:
    """Find the optimal vertical column or horizontal row crease splitting the mesh into 2 flat panels."""
    n, rows, cols = meshes.shape[:3]
    if rows < 3 and cols < 3:
        return ("vertical", 1) if cols >= 3 else ("horizontal", 1)

    ref_mesh = meshes[ref_idx]
    ref_all = ref_mesh.reshape(-1, 2)

    # Pick up to 8 frames with strongest non-planar deformation relative to a single homography
    frame_scores: list[tuple[float, int]] = []
    step = max(1, n // 16)
    for t in range(0, n, step):
        if t == ref_idx:
            continue
        dst_all = meshes[t].reshape(-1, 2)
        h_glob = _fit_panel_homography(ref_all, dst_all)
        if h_glob is None:
            continue
        proj = cv2.perspectiveTransform(ref_all.reshape(-1, 1, 2).astype(np.float32), h_glob).reshape(-1, 2)
        err = float(np.mean(np.sum((proj - dst_all) ** 2, axis=1)))
        frame_scores.append((err, t))

    frame_scores.sort(reverse=True)
    sample_frames = [t for _, t in frame_scores[:8]]
    if not sample_frames:
        return ("vertical", cols // 2) if cols >= 3 else ("horizontal", rows // 2)

    best_err = float("inf")
    best_split = ("vertical", cols // 2) if cols >= 3 else ("horizontal", rows // 2)

    # Candidate vertical creases c in [1..cols-2]
    for c in range(1, cols - 1):
        s_left = ref_mesh[:, : c + 1].reshape(-1, 2)
        s_right = ref_mesh[:, c:].reshape(-1, 2)
        total_err = 0.0
        for t in sample_frames:
            d_left = meshes[t, :, : c + 1].reshape(-1, 2)
            d_right = meshes[t, :, c:].reshape(-1, 2)
            h_l = _fit_panel_homography(s_left, d_left)
            h_r = _fit_panel_homography(s_right, d_right)
            if h_l is None or h_r is None:
                total_err += 1e12
                continue
            p_l = cv2.perspectiveTransform(s_left.reshape(-1, 1, 2).astype(np.float32), h_l).reshape(-1, 2)
            p_r = cv2.perspectiveTransform(s_right.reshape(-1, 1, 2).astype(np.float32), h_r).reshape(-1, 2)
            total_err += float(np.sum((p_l - d_left) ** 2) + np.sum((p_r - d_right) ** 2))
        if total_err < best_err:
            best_err = total_err
            best_split = ("vertical", c)

    # Candidate horizontal creases r in [1..rows-2]
    for r in range(1, rows - 1):
        s_top = ref_mesh[: r + 1, :].reshape(-1, 2)
        s_bot = ref_mesh[r:, :].reshape(-1, 2)
        total_err = 0.0
        for t in sample_frames:
            d_top = meshes[t, : r + 1, :].reshape(-1, 2)
            d_bot = meshes[t, r:, :].reshape(-1, 2)
            h_t = _fit_panel_homography(s_top, d_top)
            h_b = _fit_panel_homography(s_bot, d_bot)
            if h_t is None or h_b is None:
                total_err += 1e12
                continue
            p_t = cv2.perspectiveTransform(s_top.reshape(-1, 1, 2).astype(np.float32), h_t).reshape(-1, 2)
            p_b = cv2.perspectiveTransform(s_bot.reshape(-1, 1, 2).astype(np.float32), h_b).reshape(-1, 2)
            total_err += float(np.sum((p_t - d_top) ** 2) + np.sum((p_b - d_bot) ** 2))
        if total_err < best_err:
            best_err = total_err
            best_split = ("horizontal", r)

    return best_split


def _build_panel_clip_tag(
    mesh_t: np.ndarray, axis: str, split_idx: int, panel_idx: int
) -> str:
    """Build dynamic vector \\clip(m ... l ...) along the tracked fold crease with outward outer margin."""
    rows, cols = mesh_t.shape[:2]
    if axis == "vertical":
        crease = mesh_t[:, split_idx]  # (rows, 2) top -> bottom
        outer = mesh_t[:, 0] if panel_idx == 0 else mesh_t[:, -1]
        # Expand outer edge outward away from crease & extend top/bottom slightly for borders
        outward_dir = outer - crease
        exp_outer = outer + 0.45 * outward_dir
        vert_dir = crease[-1] - crease[0]
        exp_outer[0] = exp_outer[0] - 0.35 * vert_dir
        exp_outer[-1] = exp_outer[-1] + 0.35 * vert_dir
        ext_crease_top = crease[0] - 0.35 * vert_dir
        ext_crease_bot = crease[-1] + 0.35 * vert_dir
        poly = np.vstack(
            [
                ext_crease_top[None, :],
                crease,
                ext_crease_bot[None, :],
                exp_outer[::-1],
            ]
        )
    else:
        crease = mesh_t[split_idx, :]  # (cols, 2) left -> right
        outer = mesh_t[0, :] if panel_idx == 0 else mesh_t[-1, :]
        outward_dir = outer - crease
        exp_outer = outer + 0.45 * outward_dir
        horiz_dir = crease[-1] - crease[0]
        exp_outer[0] = exp_outer[0] - 0.35 * horiz_dir
        exp_outer[-1] = exp_outer[-1] + 0.35 * horiz_dir
        ext_crease_l = crease[0] - 0.35 * horiz_dir
        ext_crease_r = crease[-1] + 0.35 * horiz_dir
        poly = np.vstack(
            [
                ext_crease_l[None, :],
                crease,
                ext_crease_r[None, :],
                exp_outer[::-1],
            ]
        )

    coords = " ".join(f"{pt[0]:.1f} {pt[1]:.1f}" for pt in poly[1:])
    return f"\\clip(m {poly[0, 0]:.1f} {poly[0, 1]:.1f} l {coords})"


def apply_option_c_fold_split(
    meshes: np.ndarray,
    guide_text: str,
    play_res_x: float,
    play_res_y: float,
    fov_factor: float,
    aspect_mode: str,
    ref_idx: int,
    start_frame: int,
) -> list[dict[str, Any]]:
    """Option C: Split sign across detected hinge crease into 2 rigid 3D panels with dynamic \\clip."""
    n_frames, rows, cols = meshes.shape[:3]
    if rows < 3 and cols < 3:
        raise TrackerError(
            "Option C (Fold Split) requires at least 3 columns or 3 rows in the mesh grid."
        )

    axis, split_idx = detect_fold_crease(meshes, ref_idx)
    ref_mesh = meshes[ref_idx]
    ref_quad = _mesh_outer_quad(ref_mesh)
    ref_quad_f = ref_quad.reshape(-1, 1, 2).astype(np.float32)

    if axis == "vertical":
        src_p1 = ref_mesh[:, : split_idx + 1].reshape(-1, 2)
        src_p2 = ref_mesh[:, split_idx:].reshape(-1, 2)
    else:
        src_p1 = ref_mesh[: split_idx + 1, :].reshape(-1, 2)
        src_p2 = ref_mesh[split_idx:, :].reshape(-1, 2)

    out: list[dict[str, Any]] = []
    for panel_idx, src_pts in enumerate((src_p1, src_p2)):
        panel_frames: list[int] = []
        panel_quads: list[np.ndarray] = []
        panel_clips: list[str] = []

        for t in range(n_frames):
            if axis == "vertical":
                dst_pts = (
                    meshes[t, :, : split_idx + 1].reshape(-1, 2)
                    if panel_idx == 0
                    else meshes[t, :, split_idx:].reshape(-1, 2)
                )
            else:
                dst_pts = (
                    meshes[t, : split_idx + 1, :].reshape(-1, 2)
                    if panel_idx == 0
                    else meshes[t, split_idx:, :].reshape(-1, 2)
                )
            h_panel = _fit_panel_homography(src_pts, dst_pts)
            if h_panel is None:
                continue
            q_proj = cv2.perspectiveTransform(ref_quad_f, h_panel).reshape(4, 2).astype(np.float64)
            if not _is_valid_cw_quad(q_proj):
                continue
            panel_frames.append(t)
            panel_quads.append(q_proj)
            panel_clips.append(_build_panel_clip_tag(meshes[t], axis, split_idx, panel_idx))

        if not panel_quads:
            continue

        seg_ref = panel_frames.index(ref_idx) if ref_idx in panel_frames else 0
        params = decompose_quad_to_ass_params(
            quads=panel_quads,
            play_res_x=play_res_x,
            play_res_y=play_res_y,
            fov_factor=fov_factor,
            aspect_mode=aspect_mode,
            ref_idx=seg_ref,
        )
        eff_name = f"mesh_fold_p{panel_idx + 1}"
        for t, p, clip_tag in zip(panel_frames, params, panel_clips):
            out.append(
                {
                    "frame": start_frame + t,
                    "text": format_ass_line(guide_text, p, extra_tags=clip_tag),
                    "effect": eff_name,
                }
            )

    out.sort(key=lambda item: (item["frame"], item["effect"]))
    return out


# ---------------------------------------------------------------------------
# End-to-End Pipeline & CLI Entry Point
# ---------------------------------------------------------------------------


def _resolve_play_res(video_path: str, play_res_x: float, play_res_y: float) -> tuple[float, float]:
    """Resolve fallback ASS PlayRes using video dimensions when omitted."""
    if play_res_x > 0 and play_res_y > 0:
        return play_res_x, play_res_y
    if video_path and os.path.isfile(video_path):
        cap = cv2.VideoCapture(video_path)
        try:
            if play_res_x <= 0:
                play_res_x = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920.0)
            if play_res_y <= 0:
                play_res_y = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080.0)
        finally:
            cap.release()
    return (play_res_x if play_res_x > 0 else 1920.0), (play_res_y if play_res_y > 0 else 1080.0)


def _resolve_distortion_flags(request: dict[str, Any]) -> tuple[bool, bool, bool]:
    """Resolve Option A / B / C flags from distortion_mode string or legacy booleans."""
    d_mode = str(request.get("distortion_mode", "")).strip().lower()
    if d_mode:
        if "option b" in d_mode or "per_char" in d_mode or "per-char" in d_mode or "character" in d_mode:
            return False, True, False
        if "option c" in d_mode or "fold" in d_mode or "split" in d_mode:
            return False, False, True
        if "option a" in d_mode or "vector" in d_mode or "p4" in d_mode:
            return True, False, False
    return (
        bool(request.get("opt_vector_warp", True)),
        bool(request.get("opt_per_char", False)),
        bool(request.get("opt_fold_split", False)),
    )


def solve_perspective_tracking(request: dict[str, Any]) -> dict[str, Any]:
    """Run planar or non-rigid mesh tracking and convert to frame-by-frame ASS lines."""
    video_path = str(request.get("video_path", "")).strip()
    start_frame = int(request["start_frame"])
    end_frame = int(request["end_frame"])
    play_res_x = float(request.get("play_res_x", 0.0))
    play_res_y = float(request.get("play_res_y", 0.0))
    guide_text = str(request.get("text", ""))
    method = str(request.get("tracking_method", METHOD_HYBRID))
    direction = str(request.get("tracking_direction", DIRECTION_FORWARD))
    aspect_mode = str(request.get("aspect_ratio_mode", ASPECT_AUTO))
    fov_factor = float(request.get("fov_factor", 1.0))
    smoothing_window = max(0, min(10, int(request.get("smoothing_window", 0))))
    font_file_path = str(request.get("font_file_path", "")).strip()
    arabic_text = bool(request.get("arabic_text", False))

    if fov_factor <= 0.05:
        raise TrackerError("Virtual Focal Length (FOV factor) must be greater than 0.05.")

    method_key = normalize_method(method)

    # Non-rigid mesh pipeline (CPU Mesh Optical Flow or Google Colab AI Mesh JSON Import)
    if method_key in (METHOD_CPU_MESH, METHOD_AI_MESH):
        opt_vector, opt_per_char, opt_fold = _resolve_distortion_flags(request)
        if not (opt_vector or opt_per_char or opt_fold):
            raise TrackerError(
                "Please select a distortion output mode (Option A, B, or C)."
            )

        if method_key == METHOD_AI_MESH:
            mesh_json_path = str(request.get("mesh_json_path", "")).strip()
            meshes, visibilities, ref_idx, play_res_x, play_res_y = load_ai_mesh_json(
                mesh_json_path=mesh_json_path,
                start_frame=start_frame,
                end_frame=end_frame,
                play_res_x=play_res_x,
                play_res_y=play_res_y,
            )
        else:
            clip_source = request.get("clip_corners") or request.get("clip") or guide_text
            initial_quad = parse_clip_corners(clip_source)
            grid_cols = int(request.get("mesh_cols", 10))
            grid_rows = int(request.get("mesh_rows", 6))
            meshes, visibilities, ref_idx = track_nonrigid_mesh_cpu(
                video_path=video_path,
                initial_quad_ass=initial_quad,
                start_frame=start_frame,
                end_frame=end_frame,
                play_res_x=play_res_x,
                play_res_y=play_res_y,
                direction=direction,
                grid_cols=grid_cols,
                grid_rows=grid_rows,
            )
            play_res_x, play_res_y = _resolve_play_res(video_path, play_res_x, play_res_y)

        if smoothing_window > 0:
            meshes = smooth_mesh_trajectories(meshes, smoothing_window, ref_idx)

        frames_out: list[dict[str, Any]] = []
        style_info = request.get("style") or {}

        if opt_vector:
            layout_vec = extract_text_vector_layout(
                guide_text,
                style_info,
                font_file_path=font_file_path,
                arabic_text=arabic_text,
                use_harfbuzz=True,
            )
            uv_vec, _ = map_layout_to_mesh_uv(
                layout=layout_vec,
                ref_mesh=meshes[ref_idx],
                play_res_x=play_res_x,
                play_res_y=play_res_y,
                fov_factor=fov_factor,
                aspect_mode=aspect_mode,
            )
            frames_out.extend(
                apply_option_a_vector_warp(
                    meshes=meshes,
                    visibilities=visibilities,
                    uv_layout=uv_vec,
                    guide_text=guide_text,
                    start_frame=start_frame,
                )
            )

        if opt_per_char:
            # Per-character native ASS lines require Unicode Presentation Forms-B glyphs (no HarfBuzz GIDs).
            layout_char = extract_text_vector_layout(
                guide_text,
                style_info,
                font_file_path=font_file_path,
                arabic_text=arabic_text,
                use_harfbuzz=False,
            )
            uv_char, fit_scale = map_layout_to_mesh_uv(
                layout=layout_char,
                ref_mesh=meshes[ref_idx],
                play_res_x=play_res_x,
                play_res_y=play_res_y,
                fov_factor=fov_factor,
                aspect_mode=aspect_mode,
            )
            frames_out.extend(
                apply_option_b_per_char(
                    meshes=meshes,
                    visibilities=visibilities,
                    uv_layout=uv_char,
                    fit_scale=fit_scale,
                    guide_text=guide_text,
                    play_res_x=play_res_x,
                    play_res_y=play_res_y,
                    fov_factor=fov_factor,
                    aspect_mode=aspect_mode,
                    ref_idx=ref_idx,
                    start_frame=start_frame,
                )
            )

        if opt_fold:
            frames_out.extend(
                apply_option_c_fold_split(
                    meshes=meshes,
                    guide_text=guide_text,
                    play_res_x=play_res_x,
                    play_res_y=play_res_y,
                    fov_factor=fov_factor,
                    aspect_mode=aspect_mode,
                    ref_idx=ref_idx,
                    start_frame=start_frame,
                )
            )

        if not frames_out:
            raise TrackerError(
                "No valid visible output frames could be generated from the tracked mesh."
            )

        return {"status": "ok", "frames": frames_out}

    # Standard 4-corner planar homography pipeline
    clip_source = request.get("clip_corners") or request.get("clip") or guide_text
    initial_quad = parse_clip_corners(clip_source)

    tracked_quads = track_planar_quad(
        video_path=video_path,
        initial_quad_ass=initial_quad,
        start_frame=start_frame,
        end_frame=end_frame,
        play_res_x=play_res_x,
        play_res_y=play_res_y,
        method=method,
        direction=direction,
    )

    dir_key = normalize_direction(direction)
    ref_idx = 0 if dir_key == DIRECTION_FORWARD else (len(tracked_quads) - 1)

    if smoothing_window > 0:
        tracked_quads = smooth_corner_trajectories(tracked_quads, smoothing_window, ref_idx)

    play_res_x, play_res_y = _resolve_play_res(video_path, play_res_x, play_res_y)

    tag_params = decompose_quad_to_ass_params(
        quads=tracked_quads,
        play_res_x=play_res_x,
        play_res_y=play_res_y,
        fov_factor=fov_factor,
        aspect_mode=aspect_mode,
        ref_idx=ref_idx,
    )

    frames_out = []
    for offset, (quad, p) in enumerate(zip(tracked_quads, tag_params)):
        frame_num = start_frame + offset
        frames_out.append(
            {
                "frame": frame_num,
                "text": format_ass_line(guide_text, p),
                "corners": [[round(float(pt[0]), 3), round(float(pt[1]), 3)] for pt in quad],
                "tags": {k: round(float(v), 4) for k, v in p.items()},
            }
        )

    return {"status": "ok", "frames": frames_out}


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track a 4-corner planar sign or non-rigid mesh and solve ASS perspective/distortion tags."
    )
    parser.add_argument("--input", "-i", help="Path to JSON request file (or '-' for stdin).")
    parser.add_argument("--video", default="", help="Path to video file.")
    parser.add_argument("--start-frame", type=int, help="Start frame index (inclusive).")
    parser.add_argument("--end-frame", type=int, help="End frame index (inclusive).")
    parser.add_argument("--play-res-x", type=float, default=0.0, help="ASS PlayResX.")
    parser.add_argument("--play-res-y", type=float, default=0.0, help="ASS PlayResY.")
    parser.add_argument("--clip", help="4-point \\clip string or ASS line.")
    parser.add_argument("--text", default="", help="Original guide line ASS text.")
    parser.add_argument("--method", default=METHOD_HYBRID, help="Tracking method.")
    parser.add_argument("--direction", default=DIRECTION_FORWARD, help="Tracking direction.")
    parser.add_argument("--aspect-mode", default=ASPECT_AUTO, help="Aspect ratio mode.")
    parser.add_argument("--fov-factor", type=float, default=1.0, help="Virtual focal length factor.")
    parser.add_argument("--smoothing", type=int, default=0, help="Smoothing window (0..10).")
    parser.add_argument("--mesh-json", default="", help="Path to exported Colab AI Mesh JSON.")
    parser.add_argument("--mesh-cols", type=int, default=10, help="CPU mesh grid columns (3..24).")
    parser.add_argument("--mesh-rows", type=int, default=6, help="CPU mesh grid rows (3..24).")
    parser.add_argument("--font-file", default="", help="Path to .ttf/.otf/.ttc font file.")
    parser.add_argument("--arabic", action="store_true", help="Enable Arabic shaping & RTL visual reorder.")
    parser.add_argument("--distortion-mode", default="", help="Distortion mode (vector, per_char, fold).")
    parser.add_argument("--opt-vector", action="store_true", help="Enable Option A (\\p4 vector warp).")
    parser.add_argument("--opt-per-char", action="store_true", help="Enable Option B (per-character 3D tags).")
    parser.add_argument("--opt-fold", action="store_true", help="Enable Option C (fold split + dynamic \\clip).")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    try:
        if args.input:
            if args.input == "-":
                request = json.load(sys.stdin)
            else:
                req_text = Path(args.input).read_text(encoding="utf-8")
                request = json.loads(req_text)
        else:
            method_key = normalize_method(args.method)
            if args.start_frame is None or args.end_frame is None or (
                method_key != METHOD_AI_MESH and not args.video
            ):
                raise TrackerError(
                    "Missing required arguments. Provide --input <request.json> or "
                    "--video, --start-frame, and --end-frame."
                )
            opt_any = args.opt_vector or args.opt_per_char or args.opt_fold
            request = {
                "video_path": args.video,
                "start_frame": args.start_frame,
                "end_frame": args.end_frame,
                "play_res_x": args.play_res_x,
                "play_res_y": args.play_res_y,
                "clip": args.clip or args.text,
                "text": args.text,
                "tracking_method": args.method,
                "tracking_direction": args.direction,
                "aspect_ratio_mode": args.aspect_mode,
                "fov_factor": args.fov_factor,
                "smoothing_window": args.smoothing,
                "mesh_json_path": args.mesh_json,
                "mesh_cols": args.mesh_cols,
                "mesh_rows": args.mesh_rows,
                "font_file_path": args.font_file,
                "arabic_text": args.arabic,
                "distortion_mode": args.distortion_mode,
                "opt_vector_warp": args.opt_vector if opt_any else True,
                "opt_per_char": args.opt_per_char,
                "opt_fold_split": args.opt_fold,
            }

        result = solve_perspective_tracking(request)
        sys.stdout.write(json.dumps(result, ensure_ascii=True))
        sys.stdout.flush()
        return 0
    except TrackerError as exc:
        sys.stdout.write(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=True))
        sys.stdout.flush()
        return 1
    except Exception as exc:  # pragma: no cover - defensive boundary for Lua IPC
        sys.stdout.write(
            json.dumps(
                {"status": "error", "message": f"Unexpected solver error: {exc}"},
                ensure_ascii=True,
            )
        )
        sys.stdout.flush()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
