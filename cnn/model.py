from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 11  # [0]=empty, [1]=blocked, [2..10]=digits 1..9


class SmallSudokuCNN(nn.Module):
    """
    Compact CNN for 64x64 grayscale inputs.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32x32
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16x16
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8x8
        return self.head(x)


def class_to_label(idx: int) -> str:
    if idx == 0:
        return "empty"
    if idx == 1:
        return "blocked"
    return str(idx - 1)  # 2->"1", ..., 10->"9"


def label_to_class(label: str) -> int:
    if label == "empty":
        return 0
    if label == "blocked":
        return 1
    d = int(label)
    assert 1 <= d <= 9
    return d + 1
