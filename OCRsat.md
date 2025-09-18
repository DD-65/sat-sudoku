# Pipeline at a glance
1. Load image → deskew + perspective-correct the Sudoku board.
2. Detect the grid → compute an (r×c) cell lattice.
3. Segment each cell → classify it as: **given digit**, **empty**, or **blocked**.
4. OCR the given digits (or run a tiny CNN digit classifier).
5. Build the SAT model for a $begin:math:text$(p\\!\\times\\!q)\\times(p\\!\\times\\!q)$end:math:text$ Sudoku (e.g., 9×9 with $begin:math:text$p{=}q{=}3$end:math:text$) and write DIMACS CNF.
6. Run Kissat (or any CLI solver) on the CNF.
7. Parse the model → map back to a solved grid → render an image/output.

---

# 1) Preprocessing & board rectification (OpenCV)
- Convert to grayscale → adaptive threshold (or Otsu).
- Find the **largest external contour** (the board), approximate to 4 points, then `cv2.getPerspectiveTransform` → warp to a square canvas.
- Light cleanup: `cv2.medianBlur`, morphological opening/closing.

**Key OpenCV ops**
```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
th   = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY_INV, 31, 5)

# biggest contour -> 4-corner polygon -> perspective warp to N×N
# (N ~ 900..1200 px so each cell is ~100–130 px on 9x9)
```

---

# 2) Grid detection → r×c lattice
Two solid approaches:

**A. Morphological lines (robust)**
- Extract horizontal lines: `cv2.getStructuringElement((k,1))` + morphology.
- Extract vertical lines likewise; then find intersections = grid nodes.
- Sort nodes by x/y → you’ve got a rectilinear grid and can infer r,c.
- Works for 9×9 and is adaptable for other sizes.

**B. Equal slicing (fast)**
- If the warp is good and lines have uniform spacing, just divide the warped board into r×c equal tiles.  
- If you don’t know r,c, detect peaks in the projection profiles (sums of black pixels along x/y) to estimate line positions → count gaps.

> Tip: If you want “n×m” variants (e.g., 6×6 with 2×3 subgrids, 12×12 with 3×4), keep r,c detected; later derive $begin:math:text$p,q$end:math:text$ such that $begin:math:text$r=pq$end:math:text$ and $begin:math:text$c=pq$end:math:text$ for standard Sudoku. If the puzzle truly has blocked cells, your SAT will handle non-rectangular play areas anyway.

---

# 3) Cell segmentation & 3-type classification
For each cell image:
- Compute **ink ratio** = (# black pixels)/(cell area).  
  - Very low → likely **empty**.
  - Very high + no digit shape → likely **blocked** (shaded/filled).  
- If ink is moderate, try **digit detection** inside an inner margin (to ignore borders/notes). Use contour filtering (area/roundness/aspect) to isolate a single character candidate.
- Some sources use a small CNN to distinguish **digit vs not-digit vs blocked**; this beats hand-tuned thresholds if your sources vary.

Heuristics that help:
- Remove border with a margin (e.g., 10% padding) before analyzing.
- For “blocked” cells (solid fill), the variance of intensities is low and ink ratio is high. For “digit” cells, you’ll see thin strokes and higher contour eccentricity.

---

# 4) OCR: Tesseract vs a tiny CNN
**Tesseract (simple, printed digits)**
- `--oem 1 --psm 10` (single character), whitelist digits: `-c tessedit_char_whitelist=0123456789`.
- Preprocess: binarize + invert if needed + center the blob.

**Mini CNN classifier (robust)**
- 28×28 grayscale, trained on MNIST + synthetic Sudoku fonts.  
- Classifies 0–9; treat 0 as “empty” or disallow 0 and map no-digit to empty.
- Often outperforms Tesseract on low-res phone pics.

**Python libs**
- OpenCV (`cv2`), `pytesseract` (for Tesseract), or `torch/keras` for a small CNN.  
- Alternatives: PaddleOCR or EasyOCR if you want an all-Python OCR stack.

---

# 5) SAT encoding (general $begin:math:text$(p\\!\\times\\!q)$end:math:text$ Sudoku)
Let $begin:math:text$N = p \\times q$end:math:text$ (e.g., 9), rows $begin:math:text$r\\in[1..N]$end:math:text$, cols $begin:math:text$c\\in[1..N]$end:math:text$, digits $begin:math:text$d\\in[1..N]$end:math:text$.

Create Boolean vars $begin:math:text$X_{r,c,d}$end:math:text$ = “cell (r,c) holds digit d”. Map them to DIMACS integers, e.g.:
```
var(r,c,d) = (r-1)*N*N + (c-1)*N + d
```

**Clauses**
1) **Cell has at least one digit**  
   $begin:math:text$\\bigvee_{d=1}^{N} X_{r,c,d}$end:math:text$ for all cells that are **not blocked**.

2) **Cell has at most one digit** (pairwise)  
   $begin:math:text$\\neg X_{r,c,d} \\lor \\neg X_{r,c,d'}$end:math:text$ for $begin:math:text$d<d'$end:math:text$.

3) **Row uniqueness**  
   For each row r and digit d: each pair of distinct columns $begin:math:text$c \\neq c'$end:math:text$:  
   $begin:math:text$\\neg X_{r,c,d} \\lor \\neg X_{r,c',d}$end:math:text$.

4) **Column uniqueness**  
   Similarly for columns.

5) **Subgrid (box) uniqueness**  
   For each $begin:math:text$p\\times q$end:math:text$ box and digit d: pairwise at-most-one inside that box.

6) **Givens**  
   If OCR says cell (r,c) = k: add unit clause $begin:math:text$X_{r,c,k}$end:math:text$.

7) **Blocked cells**  
   For blocked cell (r,c): add $begin:math:text$\\neg X_{r,c,d}$end:math:text$ for all d (or simply skip 1) and add these negations to forbid usage).

> Optimization: Use cardinality encodings (e.g., pairwise is fine for N≤16). For speed on large variants, use ladder or commander encodings for “exactly one”.

**Minimal writer (CNF)**
```python
def vid(r,c,d,N): return (r-1)*N*N + (c-1)*N + d

clauses = []

# 1) at least one digit for playable cells
for r in range(1,N+1):
  for c in range(1,N+1):
    if not blocked[r][c]:
      clauses.append([vid(r,c,d,N) for d in range(1,N+1)])

# 2) at most one per cell
for r in range(1,N+1):
  for c in range(1,N+1):
    if not blocked[r][c]:
      for d in range(1,N+1):
        for dp in range(d+1,N+1):
          clauses.append([-vid(r,c,d,N), -vid(r,c,dp,N)])

# 3) row uniqueness
for r in range(1,N+1):
  for d in range(1,N+1):
    for c in range(1,N+1):
      for cp in range(c+1,N+1):
        clauses.append([-vid(r,c,d,N), -vid(r,cp,d,N)])

# 4) column uniqueness
for c in range(1,N+1):
  for d in range(1,N+1):
    for r in range(1,N+1):
      for rp in range(r+1,N+1):
        clauses.append([-vid(r,c,d,N), -vid(rp,c,d,N)])

# 5) box uniqueness
for br in range(0,N,p):
  for bc in range(0,N,q):
    for d in range(1,N+1):
      cells = [(r,c) for r in range(br+1, br+p+1)
                       for c in range(bc+1, bc+q+1)]
      for i in range(len(cells)):
        for j in range(i+1,len(cells)):
          r1,c1 = cells[i]; r2,c2 = cells[j]
          clauses.append([-vid(r1,c1,d,N), -vid(r2,c2,d,N)])

# 6) givens
for (r,c,k) in givens:
  clauses.append([vid(r,c,k,N)])

# 7) blocked
for r in range(1,N+1):
  for c in range(1,N+1):
    if blocked[r][c]:
      for d in range(1,N+1):
        clauses.append([-vid(r,c,d,N)])

num_vars = N*N*N
with open("sudoku.cnf","w") as f:
    f.write(f"p cnf {num_vars} {len(clauses)}\n")
    for cl in clauses:
        f.write(" ".join(map(str,cl))+" 0\n")
```

---

# 6) Run Kissat & parse model
```bash
kissat sudoku.cnf > model.out
```
The model is a list of signed integers. A **positive** literal like `1234` means that variable is True. Parse all positives, invert the `vid` mapping to fill the solved grid.

Parsing idea:
```python
assign = set(int(x) for x in tokens if int(x) > 0)
for r in range(1,N+1):
  for c in range(1,N+1):
    for d in range(1,N+1):
      if vid(r,c,d,N) in assign:
        solution[r][c] = d
```

---

# 7) Rendering & QA
- Overlay the solved digits onto the rectified board (OpenCV `putText`) and then reverse-warp back to the original perspective if you want a “solved photo”.
- Add **confidence checks**: If OCR confidence < threshold, flag cell for manual review or try the CNN fallback.

---

# Recommended tools & why
- **Python + OpenCV**: grid detection, warping, segmentation.
- **Tesseract (pytesseract)**: quick, good for printed digits. Use `--psm 10` and digit whitelist.
- **Tiny CNN (PyTorch/Keras)**: more robust than OCR for digits; train on MNIST + synthetic Sudoku fonts.
- **Kissat**: super fast SAT; DIMACS CNF is dead simple to emit.
- **PySAT (optional)**: if you want to orchestrate solving in-process, but you asked for CLI so it’s optional.
- **NumPy / scikit-image**: handy helpers for morphology and projection profiles.

---

# Project skeleton
```
sudoku_sat/
  ocr/
    tesseract.py        # single-char OCR wrapper
    cnn_digit.py        # optional CNN classifier
  vision/
    detect_board.py     # contour + warp + grid extraction
    segment_cells.py    # produce cell crops + classify type
  sat/
    encode.py           # DIMACS builder
    run_solver.py       # calls kissat; parses model
  main.py               # glue: image -> solution -> visualization
```

---

# Practical tips & pitfalls
- **Border noise**: Crop an inner margin of each cell before OCR.
- **Handwritten digits**: Prefer the CNN. Tesseract will struggle.
- **Thick/shaded blocks**: Use ink ratio + variance to mark as blocked.
- **Skew**: Always perspective-correct before you try equal slicing.
- **n×m variants**: Detect r,c from lines; derive $begin:math:text$N=r=c$end:math:text$ for classic Sudoku. For true irregular/blocked layouts, SAT already handles it—just omit constraints for blocked cells and keep row/col/box constraints for playable ones.
- **Speed**: Pairwise at-most-one is fine for 9×9. For 16×16, consider cardinality encodings (ladder/commander) if it ever becomes slow.

---
