from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract

from vision.detect_cells import GridResult, Cell, CellStats


# ---------- Public dataclasses ----------


@dataclass(frozen=True)
class OCRCellReading:
    row: int
    col: int
    status: str  # "given", "empty", or "blocked"
    digit: Optional[int]  # 1..9 (or up to N), None if empty/blocked or low confidence
    confid: float  # best OCR confidence in [0..100], -1 if none
    raw_text: str  # raw best text from Tesseract (before filtering)


@dataclass
class OCRGridResult:
    rows: int
    cols: int
    cells: List[OCRCellReading]  # len == rows * cols, row-major
    debug: Dict[str, np.ndarray]  # optional images for inspection


@dataclass
class OcrOptions:
    """
    Tuning & thresholds for OCR-based digit extraction.
    """

    # Classification thresholds using CellStats (from detect_cells)
    blocked_ink_ratio: float = 0.4  # >= this -> consider "blocked" (filled/shaded cell)
    empty_ink_ratio: float = 0.03  # <= this -> consider "empty" (no content)
    min_digit_conf: float = 50.0  # accept digit if best conf >= this

    # Preprocessing for OCR
    inner_margin_frac: float = 0.12  # crop border to avoid grid lines
    resize_px: int = 64  # normalize to square for OCR attempts
    pad_px: int = 6  # add padding after resize

    # OCR engine config
    lang: str = "eng"
    oem: int = 1  # LSTM
    psm: int = 10  # single character
    whitelist: str = "0123456789"

    # Try a few variants and pick the best confidence
    try_variants: Tuple[str, ...] = (
        "bin_mean",
        "bin_gauss",
        "otsu",
        "bin_mean_inv",
        "otsu_inv",
        "thin",
        "thicken",
    )

    # Collect debug crops and per-variant images
    collect_debug: bool = False


def _prep_for_ocr(cell_img, out_size: int = 64) -> np.ndarray:
    """
    Preprocess a cell image for Tesseract:
    - grayscale, blur
    - adaptive threshold (invert: digits white on black background)
    - crop tight to ink
    - pad to square
    - resize to out_size
    Returns a binarized uint8 image.
    """
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold for uneven lighting
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7
    )

    coords = cv2.findNonZero(th)
    if coords is None:
        return cv2.resize(th, (out_size, out_size), interpolation=cv2.INTER_AREA)
    x, y, w, h = cv2.boundingRect(coords)
    roi = th[y : y + h, x : x + w]

    # pad to square
    s = max(w, h)
    square = np.zeros((s, s), np.uint8)
    square[:, :] = 0
    square[(s - h) // 2 : (s - h) // 2 + h, (s - w) // 2 : (s - w) // 2 + w] = roi

    # resize
    return cv2.resize(square, (out_size, out_size), interpolation=cv2.INTER_AREA)


# ---------- Public API ----------


def ocr_digits_from_grid(
    grid: GridResult, opts: OcrOptions = OcrOptions()
) -> OCRGridResult:
    """
    Uses Tesseract to read digits from the given GridResult (from detect_cells).
    Classifies each cell as "given", "empty", or "blocked".
    """
    dbg: Dict[str, np.ndarray] = {}

    ocr_cells: List[OCRCellReading] = []
    for cell in grid.cells:
        status, digit, conf, raw, var_debug = _read_single_cell(cell, opts)
        ocr_cells.append(
            OCRCellReading(
                row=cell.bbox.row,
                col=cell.bbox.col,
                status=status,
                digit=digit,
                confid=conf,
                raw_text=raw,
            )
        )
        if opts.collect_debug and var_debug:
            # key like "r0c0_variantName"
            for k, im in var_debug.items():
                dbg[f"r{cell.bbox.row}c{cell.bbox.col}_{k}"] = im

    return OCRGridResult(
        rows=grid.rows,
        cols=grid.cols,
        cells=ocr_cells,
        debug=dbg if opts.collect_debug else {},
    )


# ---------- Core logic ----------


def _read_single_cell(cell, opts):
    debug_imgs = {}
    s = cell.stats

    # 1. Heuristics: blocked vs empty by ink_ratio
    if s.ink_ratio >= opts.blocked_ink_ratio:
        return "blocked", None, -1.0, "", debug_imgs
    if s.ink_ratio <= opts.empty_ink_ratio:
        return "empty", None, -1.0, "", debug_imgs

    # 2. Preprocess
    g = _prep_for_ocr(cell.image, out_size=64)
    if opts.collect_debug:
        debug_imgs["prep"] = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    # 3. Try multiple configs
    configs = [
        "--oem 1 --psm 10 -c tessedit_char_whitelist=123456789",
        "--oem 1 --psm 6 -c tessedit_char_whitelist=123456789",
        "--oem 1 --psm 13 -c tessedit_char_whitelist=123456789",
    ]

    best_digit, best_conf, best_text = None, -1.0, ""
    for cfg in configs:
        data = pytesseract.image_to_data(
            g, config=cfg, output_type=pytesseract.Output.DICT
        )
        if len(data["text"]) == 0:
            continue
        txt = data["text"][0].strip()
        conf = float(data["conf"][0]) if data["conf"][0] != "-1" else -1.0
        if txt and txt in "123456789" and conf > best_conf:
            best_digit, best_conf, best_text = int(txt), conf, txt

    # 4. Decide outcome
    if best_digit is not None and best_conf >= opts.min_digit_conf:
        return "given", best_digit, best_conf, best_text, debug_imgs

    # fallback
    return "occupied_uncertain", None, best_conf, best_text, debug_imgs


# ---------- OCR helpers ----------


def _prep_cell_for_ocr(
    img_bgr: np.ndarray, inner_margin_frac: float, resize_px: int, pad_px: int
) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    m = int(round(min(h, w) * inner_margin_frac))
    x0, y0 = m, m
    x1, y1 = max(m + 1, w - m), max(m + 1, h - m)
    roi = img_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        roi = img_bgr

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Contrast normalization helps a lot for camera photos
    gray = cv2.equalizeHist(gray)

    # Resize to a canonical square, keep aspect with border
    side = max(gray.shape)
    canvas = np.full((side, side), 255, np.uint8)
    y_off = (side - gray.shape[0]) // 2
    x_off = (side - gray.shape[1]) // 2
    canvas[y_off : y_off + gray.shape[0], x_off : x_off + gray.shape[1]] = gray

    if side != resize_px:
        canvas = cv2.resize(
            canvas, (resize_px, resize_px), interpolation=cv2.INTER_AREA
        )

    # Gentle blur to reduce speckles
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

    # Pad to prevent character clipping in PSM 10
    if pad_px > 0:
        canvas = cv2.copyMakeBorder(
            canvas, pad_px, pad_px, pad_px, pad_px, cv2.BORDER_CONSTANT, value=255
        )

    return canvas


def _generate_variants(g: np.ndarray, names: Tuple[str, ...]):
    """
    Yield (name, image) pairs with different binarization / morphology styles.
    """
    # Base binaries
    mean_bin = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5
    )
    gaus_bin = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 3
    )
    _, otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    inv = lambda im: cv2.bitwise_not(im)

    # Thin / thicken by morphology (on otsu)
    kernel3 = np.ones((3, 3), np.uint8)
    thin = cv2.erode(otsu, kernel3, iterations=1)
    thick = cv2.dilate(otsu, kernel3, iterations=1)

    variants = {
        "bin_mean": mean_bin,
        "bin_gauss": gaus_bin,
        "otsu": otsu,
        "bin_mean_inv": inv(mean_bin),
        "otsu_inv": inv(otsu),
        "thin": thin,
        "thicken": thick,
    }
    for n in names:
        if n in variants:
            yield n, variants[n]


def _tesseract_single_char(gray_u8: np.ndarray, opts: OcrOptions) -> Tuple[str, float]:
    """
    Run Tesseract expecting a single digit. Returns (text, conf in [0..100]).
    Uses image_to_data to get confidence; falls back to image_to_string if needed.
    """
    config = f"-l {opts.lang} --oem {opts.oem} --psm {opts.psm} -c tessedit_char_whitelist={opts.whitelist}"
    # Prefer TSV so we get confidences per symbol
    try:
        tsv = pytesseract.image_to_data(
            gray_u8, config=config, output_type=pytesseract.Output.DATAFRAME
        )
        if tsv is not None and len(tsv) > 0:
            # Keep rows with a non-negative conf
            tsv = tsv[tsv.conf.astype(float) >= 0.0]
            if len(tsv) > 0:
                # For PSM 10, there should be at most one char; take the best
                row = tsv.iloc[tsv.conf.astype(float).idxmax()]
                txt = str(row["text"]).strip()
                conf = float(row["conf"])
                return txt, conf
    except Exception:
        pass

    # Fallback: image_to_string (no reliable conf; approximate)
    s = pytesseract.image_to_string(gray_u8, config=config) or ""
    s = s.strip()
    conf = 60.0 if s else -1.0
    return s, conf


def _postprocess_digit(text: str) -> Optional[int]:
    if not text:
        return None
    # Keep only digits; take the first one if multiple somehow appear
    for ch in text:
        if ch.isdigit():
            d = int(ch)
            if d != 0:  # Sudoku digits 1..9 (or up to N), exclude 0 by default
                return d
    return None


def _to_bgr(g: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
