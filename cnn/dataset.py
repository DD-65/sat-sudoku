from __future__ import annotations
import glob, os, random
from typing import List
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torch
from torch.utils.data import Dataset

from cnn.model import label_to_class


def _find_font_paths() -> List[str]:
    paths = []
    # macOS system/user fonts
    paths += glob.glob("/System/Library/Fonts/*.ttf")
    paths += glob.glob("/Library/Fonts/*.ttf")
    # Linux
    paths += glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
    # common names first
    prefer = [
        "Arial",
        "Helvetica",
        "Verdana",
        "DejaVuSans",
        "LiberationSans",
        "Tahoma",
        "Menlo",
        "Courier",
    ]
    scored = []
    for p in paths:
        name = os.path.basename(p).split(".")[0].lower()
        score = max((name.find(k.lower()) >= 0) for k in prefer)
        scored.append((score, p))
    # keep unique paths, prefer “preferred” fonts up front
    scored.sort(key=lambda t: (-int(t[0]), t[1]))
    uniq = []
    seen = set()
    for _, p in scored:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq[:50] or []


_FONT_PATHS = _find_font_paths()


def _rand_digit_image(side: int, label: str) -> Image.Image:
    """Generate a synthetic 1-channel (L) cell."""
    # background
    bg = random.randint(230, 255)
    img = Image.new("L", (side, side), bg)
    dr = ImageDraw.Draw(img)

    # Optional beige tint blocks (like newspaper puzzles)
    if random.random() < 0.15:
        tint = random.randint(235, 250)
        dr.rectangle([0, 0, side - 1, side - 1], fill=tint)

    if label == "blocked":
        tone = random.randint(150, 210)
        m = int(side * 0.1)
        dr.rectangle([m, m, side - 1 - m, side - 1 - m], fill=tone)
    elif label == "empty":
        pass
    else:
        d = int(label)
        # choose a font
        if _FONT_PATHS:
            font_path = random.choice(_FONT_PATHS)
            fsize = random.randint(int(side * 0.45), int(side * 0.70))
            try:
                font = ImageFont.truetype(font_path, fsize)
            except Exception:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()

        color = random.randint(0, 30)  # dark ink
        txt = str(d)

        # measure text
        bbox = dr.textbbox((0, 0), txt, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # random offset
        ox = (side - tw) // 2 + random.randint(-3, 3)
        oy = (side - th) // 2 + random.randint(-3, 3)

        # render on a layer to rotate
        layer = Image.new("L", (side, side), 255)
        d2 = ImageDraw.Draw(layer)
        d2.text((ox, oy), txt, fill=color, font=font)
        angle = random.uniform(-6, 6)
        layer = layer.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)
        # composite layer onto img using its non-white alpha
        alpha = np.array(layer) < 250
        base = np.array(img)
        base[alpha] = np.array(layer)[alpha]
        img = Image.fromarray(base, mode="L")

    # light blur/noise
    if random.random() < 0.6:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.0, 0.7)))
    if random.random() < 0.6:
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, random.uniform(2.0, 6.0), size=arr.shape).astype(
            np.float32
        )
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")

    return img


class SudokuCellsSynthetic(Dataset):
    """
    Synthetic Sudoku-cell dataset with labels in:
    {"empty","blocked","1"..."9"}.
    """

    CLASSES = ["empty", "blocked"] + [str(i) for i in range(1, 10)]

    def __init__(self, side: int = 64, length: int = 50000):
        self.side = side
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        lab = random.choice(self.CLASSES)
        img = _rand_digit_image(self.side, lab)
        x = np.array(img, dtype=np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = torch.from_numpy(x).unsqueeze(0)  # 1xHxW
        y = label_to_class(lab)
        return x, y
