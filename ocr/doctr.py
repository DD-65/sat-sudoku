from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from doctr.models import recognition_predictor

from vision.detect_cells import GridResult, Cell

_model = None


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


def _preprocess_for_doctr(img: np.ndarray, size: int = 64, pad: int = 5) -> np.ndarray:
    """
    Pads and resizes a cell image to a fixed square for consistent OCR.
    """
    # 1. Add a white border (padding)
    img = cv2.copyMakeBorder(
        img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )

    # 2. Resize to a fixed square
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    return img


# ---------- Public API ----------


def get_model():
    """
    Gets the recognition model, loading it if needed.
    Caches the model in a global variable.
    """
    global _model
    if _model is None:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        _model = recognition_predictor(
            arch="vitstr_small", pretrained=True#, device=device
        ).to(device)
    return _model


def ocr_digits_from_grid(
    grid: GridResult, opts: OcrOptions = OcrOptions()
) -> OCRGridResult:
    """
    Uses doctr to read digits from the given GridResult (from detect_cells).
    Classifies each cell as "given", "empty", or "blocked".
    """
    dbg: Dict[str, np.ndarray] = {}
    # Load a RECOGNITION-only predictor
    model = get_model()

    # Preprocess cell images and collect them into a batch
    cell_images = [_preprocess_for_doctr(cell.image) for cell in grid.cells]

    # Run recognition on all cell images in a single batch
    # The model returns a list of (word, confidence) tuples
    predictions = model(cell_images)

    ocr_cells: List[OCRCellReading] = []
    for i, cell in enumerate(grid.cells):
        # The prediction for this cell is at the same index
        word, confidence = predictions[i]
        status, digit, conf, raw = _read_single_cell(cell, word, confidence, opts)

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
        if opts.collect_debug:
            dbg[f"r{cell.bbox.row}c{cell.bbox.col}_prep"] = cell_images[i]

    return OCRGridResult(
        rows=grid.rows, cols=grid.cols, cells=ocr_cells, debug=dbg
    )


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
