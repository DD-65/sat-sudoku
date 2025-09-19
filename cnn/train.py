from __future__ import annotations
import os, time, math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from cnn.model import SmallSudokuCNN, NUM_CLASSES
from cnn.dataset import SudokuCellsSynthetic


def train(
    out_path: str = "cnn_weights.pt",
    side: int = 64,
    total_samples: int = 60000,
    batch_size: int = 128,
    epochs: int = 5,
    lr: float = 1e-3,
    device: str = "cpu",
):
    ds = SudokuCellsSynthetic(side=side, length=total_samples)
    # 90/10 split
    n_val = max(2000, total_samples // 10)
    n_train = total_samples - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    model = SmallSudokuCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.startswith("cuda") or device == "mps")
    )

    best_val = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        tot, correct, seen = 0.0, 0, 0
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(
                enabled=(device.startswith("cuda") or device == "mps")
            ):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            seen += y.numel()
            tot += loss.item() * y.numel()
        sched.step()
        train_acc = correct / seen
        train_loss = tot / seen

        # val
        model.eval()
        with torch.no_grad():
            v_tot, v_correct, v_seen = 0.0, 0, 0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                v_tot += loss.item() * y.numel()
                v_correct += (logits.argmax(1) == y).sum().item()
                v_seen += y.numel()
        val_acc = v_correct / v_seen
        val_loss = v_tot / v_seen
        dt = time.time() - t0
        print(
            f"[ep {ep}/{epochs}] loss {train_loss:.4f} acc {train_acc:.3f} | val {val_loss:.4f} acc {val_acc:.3f} | {dt:.1f}s"
        )

        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model": model.state_dict(), "side": side}, out_path)
            print(f"  saved best to {out_path} (val_acc={best_val:.3f})")

    print("Done.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cnn_weights.pt")
    ap.add_argument("--side", type=int, default=64)
    ap.add_argument("--samples", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    args = ap.parse_args()
    train(
        args.out, args.side, args.samples, args.batch, args.epochs, args.lr, args.device
    )
