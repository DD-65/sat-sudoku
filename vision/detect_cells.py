# vision/detect_cells.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------- Public dataclasses ----------


@dataclass(frozen=True)
class CellBBox:
    row: int  # 0-based
    col: int  # 0-based
    x: int  # left
    y: int  # top
    w: int  # width
    h: int  # height


@dataclass
class CellStats:
    ink_ratio: float  # fraction of foreground pixels inside inner margin
    mean_intensity: float  # mean of grayscale (0-255)
    var_intensity: float  # variance of grayscale


@dataclass
class Cell:
    bbox: CellBBox
    image: np.ndarray  # BGR crop (exact bbox)
    stats: CellStats  # quick features to aid later classification


@dataclass
class GridResult:
    rows: int
    cols: int
    cells: List[Cell]  # len == rows * cols, row-major
    x_lines: np.ndarray  # vertical line x-positions (len = cols+1), float32
    y_lines: np.ndarray  # horizontal line y-positions (len = rows+1), float32
    debug: Dict[str, np.ndarray]  # optional stage images


@dataclass
class GridOptions:
    """
    Tuning for grid line detection on a warped, square, top-down board.
    """

    # Pre-binarization (operates on the warped BGR image)
    blur_ksize: int = 3  # 0 or odd (3/5). 0 disables blur.
    bin_method: str = "adaptive"  # "adaptive" | "otsu"
    adaptive_block: int = 31
    adaptive_C: int = 5

    # Morphological line extraction
    line_kernel_frac: float = (
        0.065  # fraction of board size for long-line kernels (e.g., 6.5%)
    )
    min_line_strength: float = (
        0.35  # min normalized projection peak height to accept as a grid line
    )

    # Peak detection/merging
    peak_merge_px: int = 8  # merge close peaks (in pixels)
    expected_rows: Optional[int] = None  # if set, locks rows
    expected_cols: Optional[int] = None  # if set, locks cols

    # Cropping/cell features
    inner_margin_frac: float = (
        0.10  # crop this margin off each side before computing ink stats
    )

    # Debug
    collect_debug: bool = False


# ---------- Public API ----------


def detect_cells(
    warped_board_bgr: np.ndarray, opts: GridOptions = GridOptions()
) -> GridResult:
    """
    Given a top-down, square Sudoku board (e.g., 1024x1024 BGR) from detect_board,
    find the r×c grid and return per-cell crops with simple stats.

    Returns:
        GridResult with rows, cols, per-cell Cell (bbox+image+stats), grid lines, and debug images.
    Raises:
        RuntimeError if a consistent grid cannot be found.
    """
    if warped_board_bgr is None or warped_board_bgr.size == 0:
        raise RuntimeError("detect_cells: input warped image is empty.")

    dbg: Dict[str, np.ndarray] = {}

    # 1) Pre-binarize to emphasize the grid
    gray = cv2.cvtColor(warped_board_bgr, cv2.COLOR_BGR2GRAY)
    if opts.blur_ksize and opts.blur_ksize > 0:
        k = opts.blur_ksize | 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    if opts.bin_method == "adaptive":
        bw = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            opts.adaptive_block | 1,
            opts.adaptive_C,
        )
    elif opts.bin_method == "otsu":
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        raise ValueError("GridOptions.bin_method must be 'adaptive' or 'otsu'.")

    # Ensure dark grid becomes white foreground
    if np.mean(bw) > 127:
        bw = cv2.bitwise_not(bw)

    if opts.collect_debug:
        dbg["10_bw"] = _vis_gray(bw)

    H, W = bw.shape
    assert H == W, "warped board is expected to be square"

    # 2) Extract horizontal/vertical long lines
    k_len = max(3, int(round(opts.line_kernel_frac * W)))
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, 1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_len))

    # Open to keep only long lines (remove digits & small blobs), then close to solidify
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k_v, iterations=1)
    vert = cv2.morphologyEx(vert, cv2.MORPH_CLOSE, k_v, iterations=1)

    hori = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k_h, iterations=1)
    hori = cv2.morphologyEx(hori, cv2.MORPH_CLOSE, k_h, iterations=1)

    # Optional one-pixel thinning helps peak localization
    vert = cv2.erode(vert, np.ones((3, 1), np.uint8), iterations=1)
    hori = cv2.erode(hori, np.ones((1, 3), np.uint8), iterations=1)

    if opts.collect_debug:
        dbg["20_vert"] = _vis_gray(vert)
        dbg["21_hori"] = _vis_gray(hori)

    # 3) Project to 1D and find grid line positions (peaks)
    x_proj = vert.sum(axis=0).astype(np.float32)  # sum over rows -> per-x strength
    y_proj = hori.sum(axis=1).astype(np.float32)  # sum over cols -> per-y strength

    x_lines = _find_peaks_1d(
        x_proj, peak_merge_px=opts.peak_merge_px, min_rel_height=opts.min_line_strength
    )
    y_lines = _find_peaks_1d(
        y_proj, peak_merge_px=opts.peak_merge_px, min_rel_height=opts.min_line_strength
    )

    if opts.collect_debug:
        dbg["30_x_proj"] = _plot_vector_as_image(
            x_proj, height=160, horizontal=True, width=W
        )
        dbg["31_y_proj"] = _plot_vector_as_image(
            y_proj, height=160, horizontal=False, width=W
        )
        dbg["32_x_lines"] = _overlay_lines(warped_board_bgr, x_lines, axis="x")
        dbg["33_y_lines"] = _overlay_lines(warped_board_bgr, y_lines, axis="y")

    # 4) Infer rows/cols (or validate overrides)
    # Ideal: 10 lines each direction for a 9x9 grid.
    def _infer_count(lines: np.ndarray, override: Optional[int]) -> int:
        if override is not None:
            return override
        # Try to convert (n_lines) -> (cells = n_lines - 1)
        if len(lines) >= 3:
            # Clamp to [2..25] for sanity; typical Sudoku is 9.
            cells = int(np.clip(len(lines) - 1, 2, 25))
            return cells
        raise RuntimeError("detect_cells: insufficient grid lines to infer size.")

    rows = _infer_count(y_lines, opts.expected_rows)
    cols = _infer_count(x_lines, opts.expected_cols)

    # If we have too many line candidates, refine by picking the best (rows+1)/(cols+1) with near-uniform spacing.
    x_lines = _regularize_lines(x_lines, target=cols + 1, length=W)
    y_lines = _regularize_lines(y_lines, target=rows + 1, length=H)

    if opts.collect_debug:
        dbg["34_x_lines_reg"] = _overlay_lines(warped_board_bgr, x_lines, axis="x")
        dbg["35_y_lines_reg"] = _overlay_lines(warped_board_bgr, y_lines, axis="y")

    if len(x_lines) != cols + 1 or len(y_lines) != rows + 1:
        raise RuntimeError(
            f"detect_cells: inconsistent grid lines (got {len(y_lines)-1}x{len(x_lines)-1}, "
            f"expected {rows}x{cols})."
        )

    # 5) Build cell bboxes and crops (row-major)
    cells: List[Cell] = []
    for r in range(rows):
        y0 = int(round(y_lines[r]))
        y1 = int(round(y_lines[r + 1]))
        for c in range(cols):
            x0 = int(round(x_lines[c]))
            x1 = int(round(x_lines[c + 1]))
            x0c, y0c = max(0, x0), max(0, y0)
            x1c, y1c = min(W, x1), min(H, y1)
            w, h = max(0, x1c - x0c), max(0, y1c - y0c)
            if w <= 1 or h <= 1:
                raise RuntimeError("detect_cells: degenerate cell bbox encountered.")

            crop = warped_board_bgr[y0c:y1c, x0c:x1c]
            stats = _compute_cell_stats(crop, inner_margin_frac=opts.inner_margin_frac)
            cells.append(
                Cell(bbox=CellBBox(r, c, x0c, y0c, w, h), image=crop, stats=stats)
            )

    return GridResult(
        rows=rows,
        cols=cols,
        cells=cells,
        x_lines=x_lines.astype(np.float32),
        y_lines=y_lines.astype(np.float32),
        debug=dbg if opts.collect_debug else {},
    )


# ---------- Helpers ----------


def _vis_gray(g: np.ndarray) -> np.ndarray:
    if len(g.shape) == 2:
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return g


def _find_peaks_1d(
    v: np.ndarray, peak_merge_px: int, min_rel_height: float
) -> np.ndarray:
    """
    Find broad peaks in a 1D projection vector v. Returns peak centers (float).
    - Smooths, thresholds by a fraction of the max, merges close plateaus to a single center.
    """
    v = v.copy()
    # Smooth lightly (box filter)
    k = max(3, int(round(len(v) * 0.003)))
    if k % 2 == 0:
        k += 1
    v_s = cv2.blur(v.reshape(1, -1), (k, 1)).reshape(-1)
    vmax = float(v_s.max() + 1e-6)
    thr = min_rel_height * vmax

    # Binary regions above threshold
    mask = (v_s >= thr).astype(np.uint8)
    # Merge short gaps
    kernel = np.ones((1, max(1, peak_merge_px * 2 + 1)), np.uint8)
    mask = cv2.dilate(mask.reshape(1, -1), kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1).reshape(-1)

    # Extract connected segments -> centers by weighted average
    peaks: List[float] = []
    i = 0
    while i < len(mask):
        if mask[i] == 0:
            i += 1
            continue
        j = i
        while j < len(mask) and mask[j] == 1:
            j += 1
        segment = slice(i, j)
        seg_vals = v_s[segment]
        # center = argmax within segment (more stable than plain midpoint)
        center = float(np.argmax(seg_vals) + i)
        peaks.append(center)
        i = j
    return np.array(peaks, dtype=np.float32)


def _regularize_lines(lines: np.ndarray, target: int, length: int) -> np.ndarray:
    """
    If we have many candidate lines, choose 'target' lines that are most
    uniformly spaced across [0, length). If we have too few, try to extrapolate
    endpoints towards 0 and length.
    """
    lines = np.sort(lines.astype(np.float32))
    if len(lines) == target:
        return lines

    if len(lines) > target:
        # Greedy pick: keep endpoints, then iteratively pick peaks that best improve uniformity.
        picks = [lines[0], lines[-1]]
        remain = list(lines[1:-1])
        while len(picks) < target and remain:
            # Ideal positions for uniform spacing
            picks_sorted = np.sort(np.array(picks))
            ideal = np.linspace(0, length - 1, num=target, dtype=np.float32)
            # Pick the candidate closest to any yet-unfilled ideal slot
            unused = [
                p for p in ideal if not _is_close_to_any(p, picks_sorted, tol=5.0)
            ]
            if not unused:
                break
            best = None
            best_err = 1e9
            for cand in list(remain):
                err = min(abs(float(cand) - float(u)) for u in unused)
                if err < best_err:
                    best_err = err
                    best = cand
            if best is None:
                break
            picks.append(best)
            remain.remove(best)
        picks = np.array(sorted(picks), dtype=np.float32)
        # If still short, fill with ideal slots
        if len(picks) < target:
            ideal = np.linspace(0, length - 1, num=target, dtype=np.float32)
            missing = [p for p in ideal if not _is_close_to_any(p, picks, tol=2.0)]
            picks = np.sort(
                np.concatenate(
                    [picks, np.array(missing[: target - len(picks)], dtype=np.float32)]
                )
            )
        return picks[:target]

    # len(lines) < target: pad towards ends if necessary
    if len(lines) >= 2:
        # Fit an arithmetic progression across current span
        span0, span1 = float(lines[0]), float(lines[-1])
        reg = np.linspace(span0, span1, num=target, dtype=np.float32)
        return reg
    # If we have 0 or 1 line, fall back to equally spaced lines
    return np.linspace(0, length - 1, num=target, dtype=np.float32)


def _is_close_to_any(x: float, arr: np.ndarray, tol: float) -> bool:
    return bool(np.any(np.abs(arr - x) <= tol))


def _overlay_lines(img: np.ndarray, lines: np.ndarray, axis: str) -> np.ndarray:
    vis = img.copy()
    if axis == "x":
        for x in lines:
            x = int(round(float(x)))
            cv2.line(vis, (x, 0), (x, img.shape[0] - 1), (0, 255, 0), 1)
    else:
        for y in lines:
            y = int(round(float(y)))
            cv2.line(vis, (0, y), (img.shape[1] - 1, y), (0, 255, 0), 1)
    return vis


def _plot_vector_as_image(
    v: np.ndarray, height: int, horizontal: bool, width: int
) -> np.ndarray:
    v = v.astype(np.float32)
    vmax = float(v.max() + 1e-6)
    n = len(v)
    if vmax <= 0:
        img = np.zeros((height, width, 3), np.uint8)
        return img
    scaled = v / vmax
    if horizontal:
        img = np.zeros((height, width, 3), np.uint8)
        # resize to full width
        bar = (scaled * (height - 1)).astype(np.int32)
        xs = np.linspace(0, width - 1, num=n).astype(np.int32)
        for x, hval in zip(xs, bar):
            cv2.line(img, (x, height - 1), (x, height - 1 - int(hval)), (0, 255, 0), 1)
    else:
        img = np.zeros((width, height, 3), np.uint8)
        bar = (scaled * (width - 1)).astype(np.int32)
        ys = np.linspace(0, height - 1, num=n).astype(np.int32)
        for y, wval in zip(ys, bar):
            cv2.line(img, (0, y), (int(wval), y), (0, 255, 0), 1)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def _compute_cell_stats(cell_bgr: np.ndarray, inner_margin_frac: float) -> CellStats:
    h, w = cell_bgr.shape[:2]
    m = int(round(min(h, w) * inner_margin_frac))
    x0, y0 = m, m
    x1, y1 = max(m + 1, w - m), max(m + 1, h - m)
    roi = cell_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        roi = cell_bgr

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Local binarization for ink ratio
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5
    )
    if np.mean(th) > 127:
        th = cv2.bitwise_not(th)
    ink_ratio = float(np.count_nonzero(th)) / float(th.size + 1e-6)
    mean = float(gray.mean())
    var = float(gray.var())
    return CellStats(ink_ratio=ink_ratio, mean_intensity=mean, var_intensity=var)
