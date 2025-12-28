# SAT Sudoku Solver

This project is a Sudoku solver that uses a combination of computer vision and a SAT solver to solve Sudoku puzzles from an image, in under 500ms.

## How it works

1.  Preprocessing: The input image is processed using OpenCV to detect the Sudoku board and warp it into a top-down perspective.
2.  Cell Detection: The individual cells of the Sudoku grid are detected and cut out to be used later on.
3.  OCR: The digits in the cells are recognized using doctr's vitstr-small.
4.  SAT-Solving: The Sudoku puzzle is converted into a satisfiability problem ([SAT](https://en.wikipedia.org/wiki/Boolean_satisfiability_problem)) and solved using the Kissat SAT solver.
5.  Solution Rendering: The sat-compiled solution is rendered back onto the original image.

## Installation

Prerequisites:
- Python 3 and pip
- [Kissat SAT solver binary](https://github.com/arminbiere/kissat)

1.  Create and activate a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2.  Install Python dependencies:

```bash
pip install -r requirements.txt
```

3.  [Install Kissat](https://github.com/arminbiere/kissat/releases/latest) and ensure it is on your PATH (or pass `--kissat /path/to/kissat`).

4.  Verify the setup on the sample image:

```bash
python main.py sudoku.png
```

Notes:
- By default, `torch` runs on CPU. For improved performance, ensure it is configured to use CUDA, MPS, or another supported GPU backend.

## Usage

### Solve a Sudoku from an image

```bash
python main.py path/to/sudoku.png
```

Common flags:
- `--kissat kissat` Path to the kissat binary (default: `kissat` in PATH).
- `--rows 9` Expected row count (default: 9).
- `--cols 9` Expected column count (default: 9).
- `--box 3x3` Subgrid size (use `auto` to infer for non-9x9).
- `--timeout 10` Solver timeout in seconds.
- `--out out` Output directory for artifacts.
- `--no-debug` / `-d` Disable debug images, timings, and artifact files.
- `--keep-solver-files` Keep CNF/model temp files.

Artifacts (when debug is enabled) are written under `--out`:
- `warped.png`, `sudoku.cnf`, `problem_summary.txt`
- `solver_stdout.txt`, `solver_stderr.txt`
- `solution_grid.json`, `solved_warped.png`, `solved_on_original.png`

### Generate a Sudoku puzzle

```bash
cd generate_sudoku
python generator.py --difficulty 0.5
```

- `--difficulty` is a float in `[0, 1]` where higher is harder (default: 0.5).
- The generator saves the PNG to `generate_sudoku/img/` when run from that folder.

## Examples

Here's an example of the solver in action:

**Input Image:**

![Input Image](sudoku.png)

**Output Image:**

![Output Image](solution.png)

---
Also has a Sudoku generation feature using the Algorithm outlined in [this Stack Overflow post](https://stackoverflow.com/a/56581709)
