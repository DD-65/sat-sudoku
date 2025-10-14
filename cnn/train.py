from __future__ import annotations
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset


from cnn.model import SmallSudokuCNN
from cnn.dataset import SudokuCellsSynthetic
from cnn.dataset_real import RealSudokuCellsCSV, RealSudokuCellsFolders
from __future__ import annotations


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
    n_val = max(2000, total_samples // 10)
    n_train = total_samples - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    use_cuda = device.startswith("cuda")
    pin = bool(use_cuda)
    workers = 0 if not use_cuda else 2  # mac/CPU: 0 workers is safest

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
    )

    model = SmallSudokuCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler("cuda") if use_cuda else None

    for ep in range(1, epochs + 1):
        model.train()
        tot, correct, seen = 0.0, 0, 0
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            if use_cuda:
                with torch.amp.autocast("cuda"):
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                opt.step()

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

        # save best
        if ep == 1 or val_acc >= getattr(train, "_best", 0.0):
            torch.save({"model": model.state_dict(), "side": side}, out_path)
            train._best = val_acc
            print(f"  saved best to {out_path} (val_acc={val_acc:.3f})")

    print("Done.")

def finetune(
    out_path: str = "cnn_weights.pt",
    side: int = 64,
    real_root: str = "data/real/cells",
    real_csv: str = "data/real/labels.csv",
    use_folders: bool = False,
    mix_synth: int = 0,         # e.g. 20000 to mix in synthetic
    batch_size: int = 128,
    epochs: int = 5,
    lr: float = 5e-4,           # lower LR for finetune
    device: str = "cpu",
    init_weights: str = "cnn_weights.pt",  # pretrain on synthetic first
):
    # Datasets
    if use_folders:
        real_ds = RealSudokuCellsFolders(real_root, side=side, augment=True)
    else:
        real_ds = RealSudokuCellsCSV(real_root, real_csv, side=side, augment=True)

    if mix_synth > 0:
        synth_ds = SudokuCellsSynthetic(side=side, length=mix_synth)
        ds = ConcatDataset([real_ds, synth_ds])
    else:
        ds = real_ds

    n_val = max(500, int(0.1 * len(real_ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    use_cuda = device.startswith("cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2 if use_cuda else 0, pin_memory=use_cuda)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2 if use_cuda else 0, pin_memory=use_cuda)

    model = SmallSudokuCNN().to(device)
    if init_weights and torch.cuda.is_available() or init_weights:
        ckpt = torch.load(init_weights, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler("cuda") if use_cuda else None

    best = 0.0
    for ep in range(1, epochs+1):
        model.train()
        tot, correct, seen = 0.0, 0, 0
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            if use_cuda:
                with torch.amp.autocast("cuda"):
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                logits = model(x); loss = criterion(logits, y); loss.backward(); opt.step()
            pred = logits.argmax(1)
            correct += (pred == y).sum().item(); seen += y.numel()
            tot += loss.item() * y.numel()
        sched.step()
        train_acc, train_loss = correct/seen, tot/seen

        model.eval()
        with torch.no_grad():
            v_tot, v_correct, v_seen = 0.0, 0, 0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                v_tot += criterion(logits, y).item() * y.numel()
                v_correct += (logits.argmax(1) == y).sum().item()
                v_seen += y.numel()
        val_acc, val_loss = v_correct/v_seen, v_tot/v_seen
        dt = time.time() - t0
        print(f"[ep {ep}/{epochs}] loss {train_loss:.4f} acc {train_acc:.3f} | val {val_loss:.4f} acc {val_acc:.3f} | {dt:.1f}s")

        if val_acc >= best:
            best = val_acc
            torch.save({"model": model.state_dict(), "side": side}, out_path)
            print(f"  saved best to {out_path} (val_acc={val_acc:.3f})")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cnn_weights.pt")
    ap.add_argument("--side", type=int, default=64)
    ap.add_argument("--samples", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    args = ap.parse_args()
    train(
        args.out, args.side, args.samples, args.batch, args.epochs, args.lr, args.device
    )
