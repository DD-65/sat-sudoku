from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from sat.builder import CNF, SudokuSpec, VarManager


@dataclass(frozen=True)
class SolverStatus:
    code: str  # "SAT", "UNSAT", or "UNKNOWN"

    def __str__(self) -> str:
        return self.code


SAT = SolverStatus("SAT")
UNSAT = SolverStatus("UNSAT")
UNKNOWN = SolverStatus("UNKNOWN")


@dataclass
class SolveResult:
    status: SolverStatus  # SAT / UNSAT / UNKNOWN
    grid: Optional[
        List[List[Optional[int]]]
    ]  # 2D list (N×N); None for blocked/unused cells if unsolved
    pos_literals: Set[
        int
    ]  # positive literals returned by solver (raw model), empty if none
    stdout: str  # raw solver stdout
    stderr: str  # raw solver stderr
    exit_code: int  # process return code
    runtime_sec: float  # wall time
    cnf_path: Optional[str] = None  # for debugging; None if temp removed
    model_path: Optional[str] = None  # for debugging; None if temp removed


def run_kissat_on_cnf(
    cnf: CNF,
    spec: SudokuSpec,
    blocked: Iterable[Tuple[int, int]],
    *,
    kissat_path: str = "/Users/danielcairoli/Downloads/kissat-4.0.3-apple-arm64",
    extra_args: Optional[Sequence[str]] = None,
    timeout_sec: Optional[float] = None,
    keep_files: bool = False,
    workdir: Optional[str] = None,
) -> SolveResult:
    """
    Run a local Kissat binary on the given CNF and reconstruct the Sudoku grid.

    Args:
        cnf: CNF from sat.builder.build_sudoku_cnf(...)
        spec: SudokuSpec (N, p, q)
        blocked: iterable of 1-based (r,c) cells that are blocked
        kissat_path: path to the kissat executable (must be on PATH or absolute)
        extra_args: optional extra CLI flags for kissat (e.g., ["--relaxed"])
        timeout_sec: wall-clock timeout; uses subprocess timeout (not Kissat's own flag)
        keep_files: if True, keep the temp CNF and model files on disk (paths returned)
        workdir: working directory for the subprocess; CNF/model files are created there if provided

    Returns:
        SolveResult with status, (optional) grid, raw stdout/stderr, and metadata.
    """
    if shutil.which(kissat_path) is None and not os.path.isfile(kissat_path):
        raise FileNotFoundError(f"Kissat binary not found: {kissat_path}")

    N = spec.N
    vm = VarManager(N)
    blocked_set: Set[Tuple[int, int]] = set(blocked)

    # Prepare temp files
    tmp_dir = workdir or tempfile.mkdtemp(prefix="sudoku_sat_")
    cnf_path = os.path.join(tmp_dir, "problem.cnf")
    model_path = os.path.join(tmp_dir, "model.out")

    # Write CNF
    _write_dimacs_fast(cnf, cnf_path)

    # Assemble command
    cmd = [kissat_path, cnf_path]
    # Kissat prints model to stdout; we’ll capture it and also save to model_path for debugging.
    if extra_args:
        cmd = [kissat_path, *extra_args, cnf_path]

    # Run
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=tmp_dir,
            timeout=timeout_sec,
            check=False,
            text=True,
        )
        runtime = time.time() - t0
    except subprocess.TimeoutExpired as e:
        # Best-effort capture of partial output
        out = e.stdout or ""
        err = e.stderr or ""
        if not keep_files:
            _cleanup_tmp(tmp_dir)
            cnf_path = None
            model_path = None
        return SolveResult(
            status=UNKNOWN,
            grid=None,
            pos_literals=set(),
            stdout=out,
            stderr=f"Timeout after {timeout_sec}s.\n{err}",
            exit_code=-1,
            runtime_sec=runtime,
            cnf_path=cnf_path,
            model_path=model_path,
        )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    exit_code = proc.returncode

    # Persist stdout to model_path for debug reproducibility
    try:
        with open(model_path, "w", encoding="utf-8") as f:
            f.write(stdout)
    except Exception:
        # Non-fatal
        pass

    # Interpret
    status = _parse_status(stdout, stderr)
    if status is UNSAT:
        if not keep_files:
            _cleanup_tmp(tmp_dir)
            cnf_path = None
            model_path = None
        return SolveResult(
            status=UNSAT,
            grid=None,
            pos_literals=set(),
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            runtime_sec=runtime,
            cnf_path=cnf_path,
            model_path=model_path,
        )

    if status is UNKNOWN or exit_code not in (
        0,
        10,
        20,
    ):  # Kissat uses SAT=10, UNSAT=20; but we already parsed status
        if not keep_files:
            _cleanup_tmp(tmp_dir)
            cnf_path = None
            model_path = None
        return SolveResult(
            status=UNKNOWN,
            grid=None,
            pos_literals=set(),
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            runtime_sec=runtime,
            cnf_path=cnf_path,
            model_path=model_path,
        )

    # SAT: parse model → grid
    pos = _parse_positive_literals(stdout)
    grid = _model_to_grid(pos, vm, N, blocked_set)

    if not keep_files:
        _cleanup_tmp(tmp_dir)
        cnf_path = None
        model_path = None

    return SolveResult(
        status=SAT,
        grid=grid,
        pos_literals=pos,
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        runtime_sec=runtime,
        cnf_path=cnf_path,
        model_path=model_path,
    )


# --------------------------- internals ---------------------------


def _write_dimacs_fast(cnf: CNF, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"p cnf {cnf.num_vars} {len(cnf.clauses)}\n")
        # Write quickly without per-line joins where possible
        for cl in cnf.clauses:
            f.write(" ".join(map(str, cl)))
            f.write(" 0\n")


def _parse_status(stdout: str, stderr: str) -> SolverStatus:
    # Kissat commonly prints a line starting with "s "
    text = (stdout + "\n" + stderr).lower()
    if "unsat" in text or "unsatisfiable" in text:
        return UNSAT
    if "sat" in text or "satisfiable" in text:
        # Beware "unsatisfiable" contains "satisfiable"; we handled UNSAT first.
        return SAT
    return UNKNOWN


def _parse_positive_literals(stdout: str) -> Set[int]:
    """
    Extract all positive integers from typical SAT solver model output.
    Supports both DIMACS 'v ... 0' lines and raw integer dumps.
    """
    pos: Set[int] = set()
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # Accept lines that look like models: start with 'v' or are just ints
        if line.startswith("c ") or line.startswith("s "):
            continue
        if line.startswith("v "):
            line = line[2:]
        # Split on whitespace and collect ints until a trailing 0
        for tok in line.split():
            if tok == "0":
                break
            try:
                lit = int(tok)
            except ValueError:
                continue
            if lit > 0:
                pos.add(lit)
    return pos


def _model_to_grid(
    pos_literals: Set[int],
    vm: VarManager,
    N: int,
    blocked: Set[Tuple[int, int]],
) -> List[List[Optional[int]]]:
    """
    Build a 2D grid from the set of positive literals. For blocked cells, returns None.
    If multiple digits are True for a cell (shouldn't happen), picks the first.
    """
    grid: List[List[Optional[int]]] = [[None for _ in range(N)] for _ in range(N)]
    # Build a quick reverse index: (r,c) -> chosen d
    # We iterate through all positives and keep the last one seen; typical models will have exactly one.
    for v in pos_literals:
        r, c, d = vm.decode(v)
        if (r, c) in blocked:
            continue
        # DIMACS variables are 1-based; grid is 0-based in Python
        grid[r - 1][c - 1] = d
    return grid


def _cleanup_tmp(tmp_dir: str) -> None:
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass
