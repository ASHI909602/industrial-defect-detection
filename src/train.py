"""Usage: python -m src.train"""
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from .config import (BATCH_SIZE, DATA_DIR, EPOCHS, LR, MODEL_PATH, REPORT_DIR,
                     SEED)
from .model import build_model
from .preprocess import eval_transform, train_transform


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = datasets.ImageFolder(DATA_DIR / "train", train_transform)
    val_ds = datasets.ImageFolder(DATA_DIR / "val", eval_transform)
    classes = train_ds.classes
    print("Classes:", classes, "| device:", device)

    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, BATCH_SIZE, num_workers=2)

    model = build_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    history, best = [], 0.0
    for epoch in range(1, EPOCHS + 1):
        tl, ta = run_epoch(model, train_dl, criterion, device, optimizer)
        vl, va = run_epoch(model, val_dl, criterion, device)
        scheduler.step()
        history.append(dict(epoch=epoch, train_loss=tl, train_acc=ta,
                            val_loss=vl, val_acc=va))
        print(f"Epoch {epoch:02d} | train {tl:.3f}/{ta:.3f} | val {vl:.3f}/{va:.3f}")
        if va > best:
            best = va
            torch.save({"state_dict": model.state_dict(), "classes": classes},
                       MODEL_PATH)
    (REPORT_DIR / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Best val acc: {best:.4f} -> saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
