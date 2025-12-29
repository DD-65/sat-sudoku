from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import Iterable, Iterator, List, Optional

from ocr.doctr import OCRCellReading, OCRGridResult


_EMPTY_CHARS = {".", "0"}
_DIGIT_CHARS = set("123456789")
_COMMENT_CHAR = "#"


def _iter_compact_values(lines: Iterable[str]) -> Iterator[int]:
    for line in lines:
        if _COMMENT_CHAR in line:
            line = line.split(_COMMENT_CHAR, 1)[0]
        for ch in line:
            if ch in _EMPTY_CHARS:
                yield 0
            elif ch in _DIGIT_CHARS:
                yield int(ch)


@dataclass
class CompactSudokuReader:
    path: str
    size: int = 9

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("size must be positive")
        self._total_cells = self.size * self.size
        self._fh = open(self.path, "r", encoding="utf-8", errors="ignore")
        self._values = _iter_compact_values(self._fh)

    def read_next(self) -> Optional[List[int]]:
        chunk = list(islice(self._values, self._total_cells))
        if len(chunk) != self._total_cells:
            return None
        return chunk

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "CompactSudokuReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def compact_digits_to_ocr(digits: List[int], size: int = 9) -> OCRGridResult:
    expected = size * size
    if len(digits) != expected:
        raise ValueError(f"Expected {expected} digits, got {len(digits)}")

    cells: List[OCRCellReading] = []
    for idx, val in enumerate(digits):
        row = idx // size
        col = idx % size
        if val == 0:
            cells.append(
                OCRCellReading(
                    row=row,
                    col=col,
                    status="empty",
                    digit=None,
                    confid=-1.0,
                    raw_text=".",
                )
            )
        else:
            cells.append(
                OCRCellReading(
                    row=row,
                    col=col,
                    status="given",
                    digit=val,
                    confid=1.0,
                    raw_text=str(val),
                )
            )
    return OCRGridResult(rows=size, cols=size, cells=cells, debug={})
