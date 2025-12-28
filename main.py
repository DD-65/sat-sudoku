from __future__ import annotations

import os
import sys
import json
import argparse
import time
from typing import List, Tuple, Optional

import cv2
import numpy as np

from vision.detect_board import detect_board, DetectorOptions
from vision.detect_cells import detect_cells, GridOptions, GridResult
from ocr.doctr import ocr_digits_from_grid, OcrOptions, OCRGridResult
from sat.builder import build_sudoku_cnf, write_dimacs, summarize_problem, SudokuSpec
from sat.runner import run_kissat_on_cnf, SAT, SolveResult


# ---------------- Orchestrator ----------------


def solve_sudoku_image(
    image_path: str,
    *,
    kissat_path: str = "/Users/daniel/Downloads/kissat-4.0.3-apple-arm64",
    expected_rows: Optional[int] = 9,
    expected_cols: Optional[int] = 9,
    box_spec: Optional[Tuple[int, int]] = (3, 3),  # (p, q) for standard 9x9
    timeout_sec: Optional[float] = 10.0,
    out_dir: str = "out",
    save_debug: bool = True,
    timing_debug: bool = True,
    keep_solver_files: bool = False,
    save_artifacts: bool = True,
) -> dict:
    """
    pipeline: image -> warped board -> cells -> OCR -> CNF -> Kissat -> solution.
    
    Returns a dict with key artifacts and file paths (if saved).
    Raises RuntimeError on fatal errors.
    """
    if not os.path.exists(image_path):
        raise RuntimeError(f"Image not found: {image_path}")

    if save_debug or save_artifacts or keep_solver_files:
        os.makedirs(out_dir, exist_ok=True)

    starttime = time.time()

    # 1) Read image
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    
    time_after_image_read = time.time()
    image_read_duration = time.time() - starttime

    # 2) Detect board & warp
    det_opts = DetectorOptions(collect_debug=save_debug)
    warped_bgr, corners_xy, H, H_inv, dbg_board = detect_board(img_bgr, det_opts)
    warped_path = os.path.join(out_dir, "warped.png")
    if save_debug:
        cv2.imwrite(warped_path, warped_bgr)

    time_after_board_dec = time.time()
    board_detection_duration = time_after_board_dec - time_after_image_read

    # 3) Detect cells / grid lines on the warped board
    grid_opts = GridOptions(
        expected_rows=expected_rows,
        expected_cols=expected_cols,
        collect_debug=save_debug,
    )
    grid: GridResult = detect_cells(warped_bgr, grid_opts)

    time_after_cell_dec = time.time()
    cell_detection_duration = time_after_cell_dec - time_after_board_dec

    # 4) OCR givens/blocked
    ocr_opts = OcrOptions(collect_debug=save_debug)
    ocr_res: OCRGridResult = ocr_digits_from_grid(grid, ocr_opts)

    time_after_ocr = time.time()
    ocr_duration = time_after_ocr - time_after_cell_dec

    # 5) Build CNF (subgrid shape)
    N = ocr_res.rows
    # If user didn't force a box_spec and N != 9, spec will be auto-factored inside builder
    cnf, spec, blocked, givens = build_sudoku_cnf(
        ocr_res, box_spec=box_spec if box_spec else None
    )

    time_after_cnf = time.time()
    cnf_build_duration = time_after_cnf - time_after_ocr

    # 6) Optionally write CNF for inspection
    cnf_path = os.path.join(out_dir, "sudoku.cnf")
    if save_artifacts:
        write_dimacs(cnf, cnf_path)
    else:
        cnf_path = None

    time_after_cnf_write = time.time()
    cnf_write_duration = time_after_cnf_write - time_after_cnf

    # 7) Run Kissat
    solve_res: SolveResult = run_kissat_on_cnf(
        cnf,
        spec,
        blocked,
        kissat_path=kissat_path,
        timeout_sec=timeout_sec,
        keep_files=keep_solver_files if save_artifacts else False,
        workdir=(
            out_dir if keep_solver_files and save_artifacts else None
        ),  # keep alongside other artifacts if requested
    )

    time_after_solver = time.time()
    solver_duration = time_after_solver - time_after_cnf_write

    # 8) Persist textual artifacts / summaries
    if save_artifacts:
        summary_txt = summarize_problem(N, blocked, givens)
        with open(
            os.path.join(out_dir, "problem_summary.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(summary_txt + "\n")

        with open(os.path.join(out_dir, "solver_stdout.txt"), "w", encoding="utf-8") as f:
            f.write(solve_res.stdout)

        with open(os.path.join(out_dir, "solver_stderr.txt"), "w", encoding="utf-8") as f:
            f.write(solve_res.stderr)

    time_after_metadata = time.time()
    metadata_duration = time_after_metadata - time_after_solver

    # 9) If SAT, render solution (warped & original)
    solved_grid_img_path = None
    solved_on_original_path = None
    grid_json_path = None

    if save_artifacts and solve_res.status is SAT and solve_res.grid is not None:
        # Save solved grid as JSON
        grid_json_path = os.path.join(out_dir, "solution_grid.json")
        with open(grid_json_path, "w", encoding="utf-8") as f:
            json.dump(solve_res.grid, f, indent=2)

        # A) Render digits on warped board for clarity
        solved_warped = _render_solution_on_warped(
            warped_bgr,
            grid,
            ocr_res,
            solve_res.grid,
            color_given=(160, 160, 255),
            color_new=(60, 220, 60),
        )
        solved_grid_img_path = os.path.join(out_dir, "solved_warped.png")
        cv2.imwrite(solved_grid_img_path, solved_warped)

        # B) Warp overlay back to original image coordinates
        solved_on_orig = _project_overlay_back(
            base_bgr=img_bgr,
            overlay_warped=solved_warped,
            H_inv=H_inv,
        )
        solved_on_original_path = os.path.join(out_dir, "solved_on_original.png")
        cv2.imwrite(solved_on_original_path, solved_on_orig)

    time_after_rendering = time.time()
    rendering_duration = time_after_rendering - time_after_metadata

    # 10) Collect debug images if requested
    if save_debug:
        _save_debug_images(dbg_board, out_dir, prefix="board_")
        _save_debug_images(grid.debug, out_dir, prefix="grid_")
        _save_debug_images(ocr_res.debug, out_dir, prefix="ocr_")


    time_after_debug = time.time()
    debug_duration = time_after_debug - time_after_rendering

    if timing_debug:
        print("[TIMINGS] (seconds)")
        print(f"  image read:          {image_read_duration:.3f}")
        print(f"  board detection:     {board_detection_duration:.3f}")
        print(f"  cell detection:      {cell_detection_duration:.3f}")
        print(f"  OCR:                 {ocr_duration:.3f}")
        print(f"  CNF build:           {cnf_build_duration:.3f}")
        print(f"  CNF write:           {cnf_write_duration:.3f}")
        print(f"  Solver run:          {solver_duration:.3f}")
        print(f"  Metadata write:      {metadata_duration:.3f}")
        print(f"  Rendering solution:  {rendering_duration:.3f}")
        print(f"  Debug images save:   {debug_duration:.3f}")
        total_duration = time_after_debug - starttime
        print(f"  TOTAL:               {total_duration:.3f}")

    return {
        "status": str(solve_res.status),
        "N": N,
        "spec": {"N": spec.N, "p": spec.p, "q": spec.q},
        "blocked": sorted(list(blocked)),
        "givens": sorted(list(givens)),
        "paths": {
            "warped": warped_path if save_debug else None,
            "cnf": cnf_path if save_artifacts else None,
            "summary": (
                os.path.join(out_dir, "problem_summary.txt") if save_artifacts else None
            ),
            "solver_stdout": (
                os.path.join(out_dir, "solver_stdout.txt") if save_artifacts else None
            ),
            "solver_stderr": (
                os.path.join(out_dir, "solver_stderr.txt") if save_artifacts else None
            ),
            "solution_grid_json": grid_json_path if save_artifacts else None,
            "solved_warped": solved_grid_img_path if save_artifacts else None,
            "solved_on_original": solved_on_original_path if save_artifacts else None,
        },
    }


# ---------------- Visualization helpers ----------------


def _render_solution_on_warped(
    warped_bgr,
    grid,
    ocr_res,
    solved_grid,
    *,
    color_given=(180, 180, 255),
    color_new=(60, 220, 60),
):
    vis = warped_bgr.copy()
    avg_cell_w = float(np.mean(np.diff(grid.x_lines)))
    font_scale = max(0.5, min(2.5, avg_cell_w / 55.0))
    thickness = max(1, int(round(avg_cell_w / 35.0)))

    # quick lookups
    given_set = {(c.row, c.col) for c in ocr_res.cells if c.status == "given"}
    blocked_set = {(c.row, c.col) for c in ocr_res.cells if c.status == "blocked"}
    # ink map from detect_cells stats
    ink_map = {(c.bbox.row, c.bbox.col): c.stats.ink_ratio for c in grid.cells}
    # threshold for “there is ink” (match your OCR empty threshold)
    OCCUPIED_INK = 0.08

    for r in range(grid.rows):
        y0 = int(round(grid.y_lines[r]))
        y1 = int(round(grid.y_lines[r + 1]))
        cy = (y0 + y1) // 2
        for c in range(grid.cols):
            x0 = int(round(grid.x_lines[c]))
            x1 = int(round(grid.x_lines[c + 1]))
            cx = (x0 + x1) // 2

            if (r, c) in blocked_set:
                cv2.rectangle(vis, (x0 + 3, y0 + 3), (x1 - 3, y1 - 3), (50, 50, 50), -1)
                continue

            val = solved_grid[r][c]
            if val is None:
                continue

            # NEW RULE: if the original cell had noticeable ink but was NOT confirmed as a given,
            # do NOT draw a new digit on top (avoid visual overwrite of mis-OCR'd givens).
            orig_has_ink = ink_map.get((r, c), 0.0) > OCCUPIED_INK
            is_given = (r, c) in given_set
            if orig_has_ink and not is_given:
                continue  # skip drawing here

            txt = str(val)
            color = color_given if is_given else color_new
            (tw, th), baseline = cv2.getTextSize(
                txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            tx, ty = cx - tw // 2, cy + th // 2
            cv2.putText(
                vis,
                txt,
                (tx + 1, ty + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                txt,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
    return vis


def _project_overlay_back(
    base_bgr: np.ndarray, overlay_warped: np.ndarray, H_inv: np.ndarray
) -> np.ndarray:
    """
    Warps the 'overlay_warped' (same size as warped board) back onto the base image using H_inv.
    Blends it with the original image.
    """
    H0, W0 = base_bgr.shape[:2]
    # Warp overlay back to original
    overlay_back = cv2.warpPerspective(overlay_warped, H_inv, (W0, H0))
    # Create a mask where overlay has non-white text (rough heuristic)
    gray = cv2.cvtColor(overlay_back, cv2.COLOR_BGR2GRAY)
    # any pixel that differs from the base more than a small threshold -> treat as overlay content
    diff = cv2.absdiff(gray, cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY))
    mask = (diff > 8).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 5)
    mask3 = cv2.merge([mask, mask, mask])

    # Blend
    out = base_bgr.copy()
    out[mask > 0] = overlay_back[mask > 0]
    return out


def _save_debug_images(images: dict, out_dir: str, prefix: str = "") -> None:
    if not images:
        return

    def _to_u8(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            return img
        if img.dtype == np.bool_:
            return (img.astype(np.uint8)) * 255
        if img.dtype in (np.float32, np.float64):
            arr = np.nan_to_num(img, nan=0.0, posinf=255.0, neginf=0.0)
            max_val = arr.max()
            scale = 255.0 if max_val <= 1.0 else 1.0
            arr = np.clip(arr * scale, 0.0, 255.0)
            return arr.astype(np.uint8)
        return np.clip(img, 0, 255).astype(np.uint8)

    for k, im in images.items():
        path = os.path.join(out_dir, f"{prefix}{k}.png")
        try:
            cv2.imwrite(path, _to_u8(im))
        except Exception:
            pass


# ---------------- CLI ----------------


def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Solve a Sudoku from an image using OCR + Kissat SAT solver."
    )
    ap.add_argument("image", help="Path to Sudoku photo")
    ap.add_argument(
        "--kissat",
        default="/Users/daniel/Downloads/kissat-4.0.3-apple-arm64",
        help="Path to kissat binary (default: 'kissat' in PATH)",
    )
    ap.add_argument("--rows", type=int, default=9, help="Expected rows (default: 9)")
    ap.add_argument("--cols", type=int, default=9, help="Expected cols (default: 9)")
    ap.add_argument(
        "--box",
        type=str,
        default="3x3",
        help="Subgrid size p×q, e.g., '3x3' (use 'auto' to auto-factor)",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Solver timeout in seconds (wall clock)",
    )
    ap.add_argument("--out", default="out", help="Output directory for artifacts")
    ap.add_argument(
        "--no-debug",
        "-d",
        action="store_true",
        help="Disable debug output, timings, and artifact files",
    )
    ap.add_argument(
        "--keep-solver-files",
        action="store_true",
        help="Keep CNF/model and solver temp files",
    )
    return ap.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    box_spec = None if args.box.lower() == "auto" else _parse_box_spec(args.box)

    try:
        result = solve_sudoku_image(
            args.image,
            kissat_path=args.kissat,
            expected_rows=args.rows,
            expected_cols=args.cols,
            box_spec=box_spec,
            timeout_sec=args.timeout,
            out_dir=args.out,
            save_debug=(not args.no_debug),
            timing_debug=(not args.no_debug),
            keep_solver_files=args.keep_solver_files,
            save_artifacts=(not args.no_debug),
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    status = result["status"]
    if status == "SAT":
        status_note = " (Solve successful)"
    elif status == "UNSAT":
        status_note = " (Solve not possible)"
    elif status == "UNKNOWN":
        status_note = " (Solve failed)"
    else:
        status_note = ""
    print(f"Status: {status}{status_note}")

    if not args.no_debug:
        print(f"N={result['N']}  spec={result['spec']}")
        print("Artifacts:")
        for k, v in result["paths"].items():
            if v:
                print(f"  - {k}: {v}")
    return 0


def _parse_box_spec(s: str) -> Tuple[int, int]:
    try:
        if "x" in s.lower():
            p, q = s.lower().split("x")
        elif "×" in s:
            p, q = s.split("×")
        else:
            raise ValueError
        return int(p), int(q)
    except Exception:
        raise RuntimeError(f"Invalid --box value: {s!r}. Use '3x3' or 'auto'.")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
