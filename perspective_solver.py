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
import sys
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

DIRECTION_FORWARD = "forward"
DIRECTION_BACKWARD = "backward"

ASPECT_AUTO = "auto"
ASPECT_KEEP = "keep"


class TrackerError(Exception):
    """Domain error carrying a user-friendly message for Aegisub UI display."""


def normalize_method(method: str) -> str:
    """Map GUI dropdown string or CLI alias to a canonical tracking method key."""
    m = (method or "").strip().lower()
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


def format_ass_line(guide_text: str, p: dict[str, float]) -> str:
    """Inject computed 3D transform tags at the start of the cleaned ASS guide line."""
    tag_prefix = (
        f"\\an5"
        f"\\pos({p['px']:.2f},{p['py']:.2f})"
        f"\\org({p['ox']:.2f},{p['oy']:.2f})"
        f"\\fscx{p['fscx']:.2f}\\fscy{p['fscy']:.2f}"
        f"\\frx{p['frx']:.2f}\\fry{p['fry']:.2f}\\frz{p['frz']:.2f}"
    )

    cleaned = strip_transform_and_clip_tags(guide_text)
    if cleaned.startswith("{"):
        close = cleaned.find("}")
        if close != -1:
            existing_tags = cleaned[1:close]
            return "{" + tag_prefix + existing_tags + "}" + cleaned[close + 1 :]
    return "{" + tag_prefix + "}" + cleaned


# ---------------------------------------------------------------------------
# End-to-End Pipeline & CLI Entry Point
# ---------------------------------------------------------------------------


def solve_perspective_tracking(request: dict[str, Any]) -> dict[str, Any]:
    """Run planar tracking and 3D ASS decomposition from a structured request dict."""
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
    smoothing_window = int(request.get("smoothing_window", 0))

    if fov_factor <= 0.05:
        raise TrackerError("Virtual Focal Length (FOV factor) must be greater than 0.05.")
    if smoothing_window < 0 or smoothing_window > 10:
        smoothing_window = max(0, min(10, smoothing_window))

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

    # Resolve fallback play_res using video dimensions if omitted in request.
    if play_res_x <= 0 or play_res_y <= 0:
        cap = cv2.VideoCapture(video_path)
        try:
            if play_res_x <= 0:
                play_res_x = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920.0)
            if play_res_y <= 0:
                play_res_y = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080.0)
        finally:
            cap.release()

    tag_params = decompose_quad_to_ass_params(
        quads=tracked_quads,
        play_res_x=play_res_x,
        play_res_y=play_res_y,
        fov_factor=fov_factor,
        aspect_mode=aspect_mode,
        ref_idx=ref_idx,
    )

    frames_out: list[dict[str, Any]] = []
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
        description="Track a 4-corner planar quad in video and solve libass 3D perspective tags."
    )
    parser.add_argument("--input", "-i", help="Path to JSON request file (or '-' for stdin).")
    parser.add_argument("--video", help="Path to video file.")
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
            if args.video is None or args.start_frame is None or args.end_frame is None:
                raise TrackerError(
                    "Missing required arguments. Provide --input <request.json> or "
                    "--video, --start-frame, and --end-frame."
                )
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
