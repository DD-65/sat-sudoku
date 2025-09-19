# vision/detect_board.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class DetectorOptions:
    """
    Tuning knobs for board detection.
    """

    # Binarization: "adaptive" (robust to uneven lighting) or "otsu" (fast, global)
    bin_method: str = "adaptive"
    adaptive_block: int = 31  # odd
    adaptive_C: int = 5

    # Morphology to connect grid lines
    morph_kernel: int = 3  # odd
    morph_iters: int = 1

    # Edge detection + consolidation
    canny1: int = 50
    canny2: int = 150
    dilate_iters: int = 1

    # Contour plausibility
    min_board_area_ratio: float = 0.2  # ≥20% of image area
    max_aspect_deviation: float = 0.35  # allow up to ±35% from square

    # Polygon approximation attempts (fraction of contour perimeter)
    approx_eps_fracs: Tuple[float, ...] = (0.02, 0.015, 0.03, 0.01)

    # Warp target
    out_size: int = 1024  # square output (pixels)

    # Internal downscale for speed (homography is rescaled back automatically)
    max_process_side: int = 1600

    # Collect stage images for upstream debugging
    collect_debug: bool = False


def detect_board(
    img_bgr: np.ndarray,
    opts: DetectorOptions = DetectorOptions(),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Detect the Sudoku board in a BGR image and produce a square, top-down warp.

    Args:
        img_bgr: np.ndarray (H,W,3) BGR image.
        opts: tuning parameters (see DetectorOptions).

    Returns:
        warped      : (S,S,3) BGR top-down view (S == opts.out_size)
        corners_xy  : (4,2) float32, TL, TR, BR, BL in ORIGINAL image coords
        H           : (3,3) float32 homography (original -> warped)
        H_inv       : (3,3) float32 inverse homography (warped -> original)
        debug_imgs  : dict[str, np.ndarray] of intermediate stages (if requested)

    Raises:
        RuntimeError if the board cannot be detected.
    """
    if img_bgr is None or img_bgr.size == 0:
        raise RuntimeError("detect_board: input image is empty.")

    dbg: Dict[str, np.ndarray] = {}

    # Optional downscale for speed; keep scale to lift homography back.
    img_proc, scale = _maybe_downscale(img_bgr, max_side=opts.max_process_side)
    if opts.collect_debug:
        dbg["00_input"] = img_proc.copy()

    gray = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarize: make grid lines white (foreground)
    if opts.bin_method == "adaptive":
        th = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            opts.adaptive_block | 1,  # ensure odd
            opts.adaptive_C,
        )
    elif opts.bin_method == "otsu":
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        raise ValueError("DetectorOptions.bin_method must be 'adaptive' or 'otsu'.")

    # Heuristic invert: ensure dark grid becomes white foreground
    if np.mean(th) > 127:
        th = cv2.bitwise_not(th)
    if opts.collect_debug:
        dbg["10_thresh"] = th.copy()

    # Close small gaps in grid lines
    k = cv2.getStructuringElement(
        cv2.MORPH_RECT, (opts.morph_kernel, opts.morph_kernel)
    )
    th_m = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=opts.morph_iters)
    if opts.collect_debug:
        dbg["20_morph_close"] = th_m.copy()

    # Edges + dilation to consolidate
    edges = cv2.Canny(th_m, opts.canny1, opts.canny2)
    if opts.dilate_iters > 0:
        edges = cv2.dilate(edges, k, iterations=opts.dilate_iters)
    if opts.collect_debug:
        dbg["30_edges"] = edges.copy()

    # Find largest plausible contour and try to quad-approximate it
    contour = _find_likely_board_contour(edges, img_proc.shape, opts)
    quad = None
    if contour is not None:
        peri = cv2.arcLength(contour, True)
        for eps_frac in opts.approx_eps_fracs:
            approx = cv2.approxPolyDP(contour, eps_frac * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad = approx.reshape(4, 2).astype(np.float32)
                break
        # Fallback: minAreaRect if approximate poly isn't a clean quad
        if quad is None:
            rect = cv2.minAreaRect(contour)
            quad = cv2.boxPoints(rect).astype(np.float32)

    if quad is None:
        raise RuntimeError("detect_board: failed to localize a board quadrilateral.")

    # Corner ordering: TL, TR, BR, BL
    corners_proc = _order_corners(quad)
    if opts.collect_debug:
        dbg_overlay = img_proc.copy()
        cv2.drawContours(dbg_overlay, [corners_proc.astype(int)], -1, (0, 255, 0), 3)
        for i, (x, y) in enumerate(corners_proc):
            cv2.circle(dbg_overlay, (int(x), int(y)), 6, (0, 0, 255), -1)
            cv2.putText(
                dbg_overlay,
                str(i),
                (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )
        dbg["40_detected_quad"] = dbg_overlay

    # Homography to square canvas
    S = opts.out_size
    dst = np.array([[0, 0], [S - 1, 0], [S - 1, S - 1], [0, S - 1]], dtype=np.float32)
    H_proc = cv2.getPerspectiveTransform(corners_proc, dst)
    warped = cv2.warpPerspective(img_proc, H_proc, (S, S))
    if opts.collect_debug:
        dbg["50_warped"] = warped.copy()

    # Lift homography to ORIGINAL image coordinates (undo downscale)
    if scale != 1.0:
        # img_proc = resize(img_bgr, scale), so original->proc is S_scale
        S_scale = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float32)
        H_full = H_proc @ np.linalg.inv(S_scale)
        corners_xy = (corners_proc / scale).astype(np.float32)
    else:
        H_full = H_proc
        corners_xy = corners_proc.astype(np.float32)

    H_inv = np.linalg.inv(H_full)
    return warped, corners_xy, H_full, H_inv, (dbg if opts.collect_debug else {})


# ----------------- helpers -----------------


def _maybe_downscale(img: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return img, 1.0
    scale = max_side / side
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _find_likely_board_contour(
    edges: np.ndarray, shape_hw: Tuple[int, int, int], opts: DetectorOptions
) -> Optional[np.ndarray]:
    h, w = shape_hw[:2]
    img_area = float(h * w)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours[:10]:
        area = cv2.contourArea(cnt)
        if area < opts.min_board_area_ratio * img_area:
            continue

        # Prefer near-square boxes but don't discard aggressively
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / (bh + 1e-6)
        if not (
            1.0 - opts.max_aspect_deviation <= aspect <= 1.0 + opts.max_aspect_deviation
        ):
            # Keep considering; Sudoku at angle may look skewed.
            pass

        # Convex is a good heuristic for the outer board
        if cv2.isContourConvex(cnt):
            return cnt

        # Otherwise accept the largest plausible contour we see
        if cnt is contours[0]:
            return cnt

    return contours[0] if contours else None


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Return points ordered as TL, TR, BR, BL."""
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)  # TL=min sum, BR=max sum
    d = np.diff(pts, axis=1)  # TR=min diff, BL=max diff
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered
