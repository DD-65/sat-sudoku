# cnn/dataset_real.py
from __future__ import annotations
import os, csv, glob, random
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torch.utils.data import Dataset
from .model import label_to_class

def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    # img: L (1-channel)
    x = np.array(img, dtype=np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return torch.from_numpy(x).unsqueeze(0)

def _augment(img: Image.Image) -> Image.Image:
    # Gentle real-world augmentations (match your camera domain)
    if random.random() < 0.5:
        img = ImageOps.autocontrast(img)
    if random.random() < 0.3:
        img = ImageOps.invert(img) if random.random() < 0.5 else img
    if random.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.0, 0.7)))
    if random.random() < 0.4:
        # brightness/contrast jitter
        arr = np.array(img).astype(np.float32)
        a = random.uniform(0.85, 1.15)  # contrast
        b = random.uniform(-15, 15)     # brightness
        arr = np.clip(a * arr + b, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
    if random.random() < 0.3:
        # tiny rotation / perspective
        angle = random.uniform(-5, 5)
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=255)
    return img

class RealSudokuCellsCSV(Dataset):
    """
    CSV: filename,label under a root folder of images.
    """
    def __init__(self, root: str, csv_path: str, side: int = 64, augment: bool = True):
        self.root = root
        self.side = side
        self.augment = augment
        self.items: List[Tuple[str,str]] = []
        with open(csv_path, newline="") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                fn, lab = row["filename"], row["label"]
                if lab in {"empty","blocked"} or (lab.isdigit() and 1 <= int(lab) <= 9):
                    self.items.append((fn, lab))
        if not self.items:
            raise RuntimeError("No items found in CSV")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        fn, lab = self.items[idx]
        path = os.path.join(self.root, fn)
        img = Image.open(path).convert("L")
        img = ImageOps.pad(img, (self.side, self.side), color=255, centering=(0.5,0.5))
        if self.augment:
            img = _augment(img)
        x = _pil_to_tensor(img)
        y = label_to_class(lab)
        return x, y

class RealSudokuCellsFolders(Dataset):
    """
    data/real/{empty,blocked,1..9}/*.png
    """
    CLASSES = ["empty","blocked"] + [str(i) for i in range(1,10)]
    def __init__(self, root: str, side: int = 64, augment: bool = True):
        self.items: List[Tuple[str,str]] = []
        self.side = side
        self.augment = augment
        for lab in self.CLASSES:
            for path in glob.glob(os.path.join(root, lab, "*.*")):
                self.items.append((path, lab))
        if not self.items:
            raise RuntimeError("No images in folder tree")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, lab = self.items[idx]
        img = Image.open(path).convert("L")
        img = ImageOps.pad(img, (self.side, self.side), color=255, centering=(0.5,0.5))
        if self.augment: img = _augment(img)
        x = _pil_to_tensor(img)
        y = label_to_class(lab)
        return x, y