from __future__ import annotations
import math, random, os, glob
from typing import Tuple, Optional, List
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
import torch
from torch.utils.data import Dataset

from cnn.model import NUM_CLASSES, label_to_class


def _load_fonts() -> List[ImageFont.FreeTypeFont]:
    # Try a few common fonts; fall back to default if missing
    candidates = []
    paths = []
    # macOS examples
    paths += glob.glob("/System/Library/Fonts/*.ttf")
    paths += glob.glob("/Library/Fonts/*.ttf")
    # Linux examples
    paths += glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
    want = [
        "Arial",
        "Helvetica",
        "Verdana",
        "DejaVuSans",
        "Courier",
        "Menlo",
        "LiberationSans",
        "Tahoma",
    ]
    selected = []
    for p in paths:
        name = os.path.basename(p).split(".")[0].lower()
        if any(w.lower() in name for w in want):
            try:
                selected.append(ImageFont.truetype(p, size=48))
            except Exception:
                pass
    if not selected:
        selected = [ImageFont.load_default()]
    return selected


_FONTS = _load_fonts()


def _rand_digit_image(side: int, label: str) -> Image.Image:
    """Generate a synthetic 1-channel (L) cell."""
    img = Image.new("L", (side, side), 255)
    dr = ImageDraw.Draw(img)

    # Light paper color jitter
    bg = random.randint(230, 255)
    img.paste(bg)

    # Optionally draw soft box tint (like beige blocks)
    if random.random() < 0.15:
        tint = random.randint(235, 250)
        dr.rectangle([0, 0, side - 1, side - 1], fill=tint)

    if label == "blocked":
        tone = random.randint(150, 210)
        dr.rectangle(
            [int(side * 0.1), int(side * 0.1), int(side * 0.9), int(side * 0.9)],
            fill=tone,
        )
    elif label == "empty":
        pass
    else:
        d = int(label)
        font = random.choice(_FONTS)
        # Resize font to cell
        fsize = random.randint(int(side * 0.45), int(side * 0.70))
        try:
            font = ImageFont.truetype(font.path, fsize)  # type: ignore[attr-defined]
        except Exception:
            # Some PIL fonts don't expose .path
            pass
        # Text color
        color = random.randint(0, 30)  # dark ink
        # Slight random offset/rotation
        txt = str(d)
        w, h = dr.textbbox((0, 0), txt, font=font)[2:]
        ox = (side - w) // 2 + random.randint(-3, 3)
        oy = (side - h) // 2 + random.randint(-3, 3)
        # render onto a temporary layer to rotate
        layer = Image.new("L", (side, side), 255)
        d2 = ImageDraw.Draw(layer)
        d2.text((ox, oy), txt, fill=color, font=font)
        angle = random.uniform(-6, 6)
        layer = layer.rotate(angle, resample=Image.BICUBIC, expand=0, fillcolor=255)
        img = Image.composite(
            layer, img, Image.fromarray((np.array(layer) < 250).astype(np.uint8) * 255)
        )

    # Add light Gaussian blur / noise occasionally
    if random.random() < 0.6:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.0, 0.7)))
    if random.random() < 0.6:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, random.uniform(2.0, 6.0), size=arr.shape).astype(
            np.float32
        )
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")

    return img


class SudokuCellsSynthetic(Dataset):
    """
    Synthetic Sudoku-cell dataset with labels in:
      {"empty", "blocked", "1"..."9"} → 11 classes.
    """

    CLASSES = ["empty", "blocked"] + [str(i) for i in range(1, 10)]

    def __init__(self, side: int = 64, length: int = 50000):
        self.side = side
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Balanced-ish sampling
        lab = random.choice(self.CLASSES)
        img = _rand_digit_image(self.side, lab)
        x = torch.from_numpy(np.array(img, dtype=np.uint8)).float() / 255.0
        x = (x - 0.5) / 0.5  # normalize to [-1,1]
        x = x.unsqueeze(0)  # CxHxW
        y = label_to_class(lab)
        return x, y
