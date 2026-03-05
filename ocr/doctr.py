from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from doctr.models import recognition_predictor

from vision.detect_cells import GridResult, Cell

_model = None
_model_device: Optional[str] = None


# ---------- Public dataclasses ----------


@dataclass(frozen=True)
class OCRCellReading:
    row: int
    col: int
    status: str  # "given", "empty", or "blocked"
    digit: Optional[int]  # 1..9 (or up to N), None if empty/blocked or low confidence
    confid: float  # best OCR confidence in [0..1], -1 if none
    raw_text: str  # raw best text from doctr


@dataclass
class OCRGridResult:
    rows: int
    cols: int
    cells: List[OCRCellReading]  # len == rows * cols, row-major
    debug: Dict[str, np.ndarray]  # optional images for inspection


@dataclass
class OcrOptions:
    """
    Tuning & thresholds for OCR-based digit extraction with doctr.
    """

    # Classification thresholds using CellStats (from detect_cells)
    blocked_ink_ratio: float = 0.5  # >= this -> consider "blocked" (filled/shaded cell)
    empty_ink_ratio: float = 0.05  # <= this -> consider "empty" (no content)
    min_digit_conf: float = 0.7  # accept digit if best conf >= this

    # Collect debug crops
    collect_debug: bool = False


# ---------- Preprocessing ----------


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest foreground component in a binary mask."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = np.zeros_like(mask)
    component[labels == largest_idx] = 255
    return component


def _preprocess_for_doctr(
    img: np.ndarray,
    size: int = 32,
    inner_margin_frac: float = 0.12,
    pad_frac: float = 0.2,
) -> np.ndarray:
    """
    Normalize a cell crop for OCR by isolating the digit, centering it on a square
    canvas, then converting to DocTR's expected RGB float image format.
    """
    h, w = img.shape[:2]
    margin_y = min(max(0, int(round(h * inner_margin_frac))), max(0, h // 2 - 1))
    margin_x = min(max(0, int(round(w * inner_margin_frac))), max(0, w // 2 - 1))
    cropped = img[margin_y : h - margin_y, margin_x : w - margin_x]
    if cropped.size == 0:
        cropped = img

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Digits are darker than the background, so we invert after Otsu thresholding.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove tiny specks before selecting the dominant glyph-like component.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )

    if cv2.countNonZero(mask) > 0:
        mask = _largest_connected_component(mask)
        ys, xs = np.where(mask > 0)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        glyph = mask[y0:y1, x0:x1]
    else:
        glyph = mask

    canvas = np.full((size, size), 255, dtype=np.uint8)
    if glyph.size > 0 and cv2.countNonZero(glyph) > 0:
        gh, gw = glyph.shape[:2]
        content_size = max(1, int(round(size * (1.0 - 2.0 * pad_frac))))
        scale = content_size / max(gh, gw)
        new_w = max(1, int(round(gw * scale)))
        new_h = max(1, int(round(gh * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(glyph, (new_w, new_h), interpolation=interpolation)

        y_off = (size - new_h) // 2
        x_off = (size - new_w) // 2
        canvas[y_off : y_off + new_h, x_off : x_off + new_w] = 255 - resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    rgb = rgb.astype("float32") / 255.0
    return rgb


# ---------- Public API ----------


def get_model():
    """
    Gets the recognition model, loading it if needed.
    Caches the model in a global variable.
    """
    global _model, _model_device
    if _model is None:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        _model = recognition_predictor(arch="vitstr_small", pretrained=True).to(device)
        #_model = recognition_predictor(arch="crnn_mobilenet_v3_small", pretrained=True).to(device)
        _model.eval()
        _model_device = device
        #_model.model = torch.compile(_model.model)
    return _model


def ocr_digits_from_grid(
    grid: GridResult, opts: OcrOptions = OcrOptions()
) -> OCRGridResult:
    """
    Uses doctr to read digits from the given GridResult (from detect_cells).
    Classifies each cell as "given", "empty", or "blocked".
    """
    dbg: Dict[str, np.ndarray] = {}
    model = get_model()

    needs_ocr_indices: List[int] = []
    needs_ocr_crops: List[np.ndarray] = []
    ocr_cells: List[Optional[OCRCellReading]] = [None] * len(grid.cells)

    for idx, cell in enumerate(grid.cells):
        s = cell.stats
        if s.ink_ratio >= opts.blocked_ink_ratio:
            ocr_cells[idx] = OCRCellReading(
                row=cell.bbox.row,
                col=cell.bbox.col,
                status="blocked",
                digit=None,
                confid=-1.0,
                raw_text="",
            )
            continue
        if s.ink_ratio <= opts.empty_ink_ratio:
            ocr_cells[idx] = OCRCellReading(
                row=cell.bbox.row,
                col=cell.bbox.col,
                status="empty",
                digit=None,
                confid=-1.0,
                raw_text="",
            )
            continue
        needs_ocr_indices.append(idx)
        prep = _preprocess_for_doctr(cell.image)
        needs_ocr_crops.append(prep)
        if opts.collect_debug:
            dbg[f"r{cell.bbox.row}c{cell.bbox.col}_prep"] = prep

    predictions: List[Tuple[str, float]] = []
    # chunking: run in batches for approx .2s faster ocr (on larger grids)
    if needs_ocr_crops:
        with torch.no_grad():
            bs = 32  # seems to be the best out of 8, 16, 32
            for i in range(0, len(needs_ocr_crops), bs):
                chunk = needs_ocr_crops[i:i+bs]
                predictions.extend(model(chunk))

    for idx, pred in zip(needs_ocr_indices, predictions):
        word, confidence = pred
        cell = grid.cells[idx]
        status, digit, conf, raw = _read_single_cell(cell, word, confidence, opts)
        ocr_cells[idx] = OCRCellReading(
            row=cell.bbox.row,
            col=cell.bbox.col,
            status=status,
            digit=digit,
            confid=conf,
            raw_text=raw,
        )

    # fill any remaining cells (should not happen but safe)
    for idx, cell in enumerate(grid.cells):
        if ocr_cells[idx] is None:
            ocr_cells[idx] = OCRCellReading(
                row=cell.bbox.row,
                col=cell.bbox.col,
                status="empty",
                digit=None,
                confid=-1.0,
                raw_text="",
            )

    return OCRGridResult(rows=grid.rows, cols=grid.cols, cells=ocr_cells, debug=dbg)


# ---------- Core logic ----------


def _read_single_cell(
    cell: Cell, word: str, confidence: float, opts: OcrOptions
) -> Tuple[str, Optional[int], float, str]:
    """
    Determines the status of a cell based on ink ratio and the direct
    output from a recognition model.
    """
    s = cell.stats

    # 1. Heuristics: blocked vs empty by ink_ratio
    if s.ink_ratio >= opts.blocked_ink_ratio:
        return "blocked", None, -1.0, ""
    if s.ink_ratio <= opts.empty_ink_ratio:
        return "empty", None, -1.0, ""

    # 2. Process recognition results
    best_digit, best_conf, best_text = None, -1.0, ""
    # The word and confidence are passed in directly
    if word.isdigit():
        d = int(word)
        if d != 0:  # Sudoku digits 1..9
            best_digit = d
            best_conf = confidence
            best_text = word

    # 3. Decide outcome
    if best_digit is not None and best_conf >= opts.min_digit_conf:
        return "given", best_digit, best_conf, best_text

    # fallback
    return "occupied_uncertain", None, best_conf, best_text
