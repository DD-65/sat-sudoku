from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import torch
from cnn.model import SmallSudokuCNN, class_to_label
from vision.detect_cells import GridResult, Cell


@dataclass(frozen=True)
class CNNCellReading:
    row: int
    col: int
    status: str  # "given" | "empty" | "blocked" | "occupied_uncertain"
    digit: Optional[int]  # 1..9 if given else None
    prob: float  # softmax prob of predicted class


@dataclass
class CNNGridResult:
    rows: int
    cols: int
    cells: List[CNNCellReading]
    debug: Dict[str, np.ndarray]


def _prep(cell_img_bgr: np.ndarray, side: int = 64) -> np.ndarray:
    # grayscale, center, resize (like tesseract prep but simpler)
    g = cv2.cvtColor(cell_img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.equalizeHist(g)
    g = cv2.resize(g, (side, side), interpolation=cv2.INTER_AREA)
    g = (g.astype(np.float32) / 255.0 - 0.5) / 0.5
    return g[None, :, :]  # 1xHxW


def recognize_grid_with_cnn(
    grid: GridResult,
    weights_path: str = "cnn_weights.pt",
    device: str = "cpu",
    empty_prob_hi: float = 0.7,  # if predicted empty with prob>= this -> empty
    blocked_prob_hi: float = 0.7,  # if predicted blocked with prob>= this -> blocked
    digit_prob_min: float = 0.55,  # else if predicted digit with prob>= this -> given
) -> CNNGridResult:
    ckpt = torch.load(weights_path, map_location=device)
    side = ckpt.get("side", 64)
    model = SmallSudokuCNN()
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    xs, meta = [], []
    for c in grid.cells:
        xs.append(_prep(c.image, side))
        meta.append((c.bbox.row, c.bbox.col))
    X = torch.from_numpy(np.stack(xs)).to(device)

    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(1)

    out: List[CNNCellReading] = []
    dbg: Dict[str, np.ndarray] = {}

    for i, (r, c) in enumerate(meta):
        p = probs[i]
        k = preds[i]
        label = class_to_label(int(k))
        prob = float(p[k])

        if label == "empty" and prob >= empty_prob_hi:
            status, digit = "empty", None
        elif label == "blocked" and prob >= blocked_prob_hi:
            status, digit = "blocked", None
        elif label not in ("empty", "blocked") and prob >= digit_prob_min:
            status, digit = "given", int(label)
        else:
            status, digit = "occupied_uncertain", None

        out.append(CNNCellReading(row=r, col=c, status=status, digit=digit, prob=prob))

    return CNNGridResult(rows=grid.rows, cols=grid.cols, cells=out, debug=dbg)
