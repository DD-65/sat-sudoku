from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set

from ocr.doctr import OCRGridResult, OCRCellReading


# ---------- Public dataclasses ----------


@dataclass(frozen=True)
class SudokuSpec:
    """
    A general Sudoku spec:
      N = size (rows == cols == N).
      p x q = subgrid shape with p*q == N (e.g., 3x3 for 9x9).
    """

    N: int
    p: int
    q: int

    @staticmethod
    def from_size(
        N: int, p: Optional[int] = None, q: Optional[int] = None
    ) -> "SudokuSpec":
        """
        Create a SudokuSpec for size N. If p,q not given, choose balanced factors.
        """
        if p is not None and q is not None:
            if p * q != N:
                raise ValueError(f"Invalid box shape: {p}*{q} != {N}")
            return SudokuSpec(N=N, p=p, q=q)
        # Find a "nice" factorization with p<=q and p*q=N (defaults to square if possible)
        # Try square root first, then search small p.
        import math

        r = int(round(math.sqrt(N)))
        for dp in (0, -1, 1, -2, 2, -3, 3):
            cand = r + dp
            if cand >= 1 and N % cand == 0:
                return SudokuSpec(N=N, p=cand, q=N // cand)
        # If not factorizable into integer p,q (e.g., prime N), fall back to 1xN boxes (no real subgrid).
        return SudokuSpec(N=N, p=1, q=N)


@dataclass
class CNF:
    num_vars: int
    clauses: List[List[int]]  # each clause is a list of ints, 0-terminated when written
    var_names: Dict[int, str]  # optional pretty names for debugging / reverse mapping


# ---------- Variable manager ----------


class VarManager:
    """
    Maps (r,c,d) 1-based to a DIMACS integer id in [1..N^3].
    """

    def __init__(self, N: int):
        self.N = N

    def vid(self, r: int, c: int, d: int) -> int:
        # (r-1)*N*N + (c-1)*N + d  in 1..N^3
        return (r - 1) * self.N * self.N + (c - 1) * self.N + d

    def decode(self, v: int) -> Tuple[int, int, int]:
        v0 = v - 1
        d = (v0 % self.N) + 1
        c = ((v0 // self.N) % self.N) + 1
        r = (v0 // (self.N * self.N)) + 1
        return r, c, d


# ---------- Public API ----------


@dataclass
class BuildOptions:
    """
    Options for CNF generation.
    """

    # Encoding for "exactly one": we use pairwise by default (simple, fine for N<=16).
    pairwise_cell_atmost1: bool = True
    pairwise_rowcolbox_atmost1: bool = True

    # If True, add negations for blocked cells (not strictly necessary if we skip them elsewhere,
    # but it makes the model explicit).
    explicit_blocked_negations: bool = True

    # Sanity checks
    fail_on_out_of_range_digit: bool = True


def build_sudoku_cnf(
    ocr: OCRGridResult,
    box_spec: Optional[
        Tuple[int, int]
    ] = None,  # (p,q) if you want to force subgrid shape
    opts: BuildOptions = BuildOptions(),
) -> Tuple[CNF, SudokuSpec, Set[Tuple[int, int]], List[Tuple[int, int, int]]]:
    """
    Build a DIMACS CNF for the Sudoku described by the OCR output.

    Args:
        ocr: OCRGridResult from ocr/doctr.py
        box_spec: optional (p,q) subgrid shape. If None, factor N automatically.
        opts: encoding options.

    Returns:
        (cnf, spec, blocked_cells, givens)
        - cnf: CNF object with num_vars=N^3 and clauses.
        - spec: SudokuSpec(N, p, q).
        - blocked_cells: set of 1-based (r,c) blocked by the puzzle (not playable).
        - givens: list of 1-based (r,c,d) digits pre-filled in the puzzle.

    Raises:
        ValueError if OCR grid is not square or digits are invalid.
    """
    if ocr.rows != ocr.cols:
        raise ValueError(f"Sudoku must be square; got {ocr.rows}x{ocr.cols}")

    N = ocr.rows
    spec = SudokuSpec.from_size(N, *(box_spec or (None, None)))

    vm = VarManager(N)
    clauses: List[List[int]] = []
    var_names: Dict[int, str] = {}

    # Collect blocked & givens (convert to 1-based)
    blocked: Set[Tuple[int, int]] = set()
    givens: List[Tuple[int, int, int]] = []

    for c in ocr.cells:
        r1, c1 = c.row + 1, c.col + 1
        if c.status == "blocked":
            blocked.add((r1, c1))
            continue
        if c.status == "given" and c.digit is not None:
            if opts.fail_on_out_of_range_digit and not (1 <= c.digit <= N):
                raise ValueError(
                    f"Digit {c.digit} at ({r1},{c1}) out of range for N={N}"
                )
            givens.append((r1, c1, c.digit))

    # 1) Cell constraints (only for playable cells)
    #    a) at least one digit per cell
    #    b) at most one digit per cell (pairwise)
    for r in range(1, N + 1):
        for c in range(1, N + 1):
            if (r, c) in blocked:
                continue
            # (a) at least one
            clause = [vm.vid(r, c, d) for d in range(1, N + 1)]
            clauses.append(clause)
            # (b) at most one (pairwise)
            if opts.pairwise_cell_atmost1:
                for d in range(1, N + 1):
                    vd = vm.vid(r, c, d)
                    var_names[vd] = f"X({r},{c})={d}"
                    for d2 in range(d + 1, N + 1):
                        clauses.append([-vd, -vm.vid(r, c, d2)])

    # 2) Row constraints: each digit appears at most once per row
    if opts.pairwise_rowcolbox_atmost1:
        for r in range(1, N + 1):
            for d in range(1, N + 1):
                # consider only playable cells in the row
                cols_play = [c for c in range(1, N + 1) if (r, c) not in blocked]
                for i in range(len(cols_play)):
                    for j in range(i + 1, len(cols_play)):
                        c1, c2 = cols_play[i], cols_play[j]
                        clauses.append([-vm.vid(r, c1, d), -vm.vid(r, c2, d)])

    # 3) Column constraints: each digit appears at most once per column
    if opts.pairwise_rowcolbox_atmost1:
        for c in range(1, N + 1):
            for d in range(1, N + 1):
                rows_play = [r for r in range(1, N + 1) if (r, c) not in blocked]
                for i in range(len(rows_play)):
                    for j in range(i + 1, len(rows_play)):
                        r1, r2 = rows_play[i], rows_play[j]
                        clauses.append([-vm.vid(r1, c, d), -vm.vid(r2, c, d)])

    # 4) Box constraints: each digit appears at most once per p×q box
    p, q = spec.p, spec.q
    if opts.pairwise_rowcolbox_atmost1:
        for br in range(0, N, p):
            for bc in range(0, N, q):
                # gather playable cells within this box
                cells = [
                    (r, c)
                    for r in range(br + 1, br + p + 1)
                    for c in range(bc + 1, bc + q + 1)
                    if (r, c) not in blocked
                ]
                for d in range(1, N + 1):
                    for i in range(len(cells)):
                        r1, c1 = cells[i]
                        for j in range(i + 1, len(cells)):
                            r2, c2 = cells[j]
                            clauses.append([-vm.vid(r1, c1, d), -vm.vid(r2, c2, d)])

    # 5) Givens: unit clauses
    for r, c, d in givens:
        clauses.append([vm.vid(r, c, d)])
        var_names[vm.vid(r, c, d)] = f"X({r},{c})={d}"

    # 6) Explicitly forbid all digits in blocked cells (optional but explicit)
    if opts.explicit_blocked_negations:
        for r, c in blocked:
            for d in range(1, N + 1):
                clauses.append([-vm.vid(r, c, d)])

    cnf = CNF(num_vars=N * N * N, clauses=clauses, var_names=var_names)
    return cnf, spec, blocked, givens


# ---------- Utility: writing & simple pretty-print ----------


def write_dimacs(cnf: CNF, path: str) -> None:
    """
    Write CNF to a DIMACS file at 'path'.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"p cnf {cnf.num_vars} {len(cnf.clauses)}\n")
        for cl in cnf.clauses:
            f.write(" ".join(map(str, cl)) + " 0\n")


def summarize_problem(
    N: int, blocked: Iterable[Tuple[int, int]], givens: Sequence[Tuple[int, int, int]]
) -> str:
    """
    Human-readable summary (rows/cols are 1-based).
    """
    b = set(blocked)
    g = {(r, c): d for (r, c, d) in givens}
    lines: List[str] = []
    for r in range(1, N + 1):
        row = []
        for c in range(1, N + 1):
            if (r, c) in b:
                row.append("■")
            elif (r, c) in g:
                row.append(str(g[(r, c)]))
            else:
                row.append(".")
        lines.append(" ".join(row))
    return "\n".join(lines)
