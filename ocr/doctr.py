from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from doctr.models import ocr_predictor

from vision.detect_cells import GridResult


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
    blocked_ink_ratio: float = 0.4  # >= this -> consider "blocked" (filled/shaded cell)
    empty_ink_ratio: float = 0.04  # <= this -> consider "empty" (no content)
    min_digit_conf: float = 0.41  # accept digit if best conf >= this

    # Collect debug crops
    collect_debug: bool = False


# ---------- Public API ----------


def ocr_digits_from_grid(
    grid: GridResult, opts: OcrOptions = OcrOptions()
) -> OCRGridResult:
    """
    Uses doctr to read digits from the given GridResult (from detect_cells).
    Classifies each cell as "given", "empty", or "blocked".
    """
    dbg: Dict[str, np.ndarray] = {}
    model = ocr_predictor(pretrained=True)

    ocr_cells: List[OCRCellReading] = []
    # Extract cell images into a batch
    cell_images = [cell.image for cell in grid.cells]

    # Run OCR on all cell images in a single batch
    results = model(cell_images)

    for i, cell in enumerate(grid.cells):
        status, digit, conf, raw = _read_single_cell(cell, results.pages[i], opts)
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
            # Add debug images if any
            pass

    return OCRGridResult(
        rows=grid.rows,
        cols=grid.cols,
        cells=ocr_cells,
        debug=dbg,
    )


# ---------- Core logic ----------


def _read_single_cell(cell, page_result, opts):
    s = cell.stats

    # 1. Heuristics: blocked vs empty by ink_ratio
    if s.ink_ratio >= opts.blocked_ink_ratio:
        return "blocked", None, -1.0, ""
    if s.ink_ratio <= opts.empty_ink_ratio:
        return "empty", None, -1.0, ""

    # 2. Process OCR results
    best_digit, best_conf, best_text = None, -1.0, ""

    if page_result.blocks:
        for block in page_result.blocks:
            for line in block.lines:
                for word in line.words:
                    if word.value.isdigit() and word.confidence > best_conf:
                        d = int(word.value)
                        if d != 0:  # Sudoku digits 1..9
                            best_digit = d
                            best_conf = word.confidence
                            best_text = word.value

    # 3. Decide outcome
    if best_digit is not None and best_conf >= opts.min_digit_conf:
        return "given", best_digit, best_conf, best_text

    # fallback
    return "occupied_uncertain", None, best_conf, best_text
