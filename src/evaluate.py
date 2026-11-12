"""Usage: python -m src.evaluate  -> reports/ me metrics.json + confusion_matrix.png"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (classification_report, confusion_matrix,  # noqa: E402
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import datasets  # noqa: E402

from .config import BATCH_SIZE, DATA_DIR, NO_DEFECT_CLASS, REPORT_DIR  # noqa: E402
from .predict import load_model  # noqa: E402
from .preprocess import eval_transform  # noqa: E402


def main():
    model, classes = load_model()
    ds = datasets.ImageFolder(DATA_DIR / "test", eval_transform)
    assert ds.classes == classes, "Train/test classes match nahi karte"
    dl = DataLoader(ds, BATCH_SIZE, num_workers=2)

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in dl:
            y_pred += model(x).argmax(1).tolist()
            y_true += y.tolist()
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    report = classification_report(y_true, y_pred, target_names=classes,
                                   output_dict=True, zero_division=0)

    # Binary view: Defect vs No Defect
    ok_idx = classes.index(NO_DEFECT_CLASS)
    b_true, b_pred = (y_true != ok_idx).astype(int), (y_pred != ok_idx).astype(int)
    p, r, f, _ = precision_recall_fscore_support(b_true, b_pred, average="binary",
                                                 zero_division=0)
    metrics = {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "binary_defect": {"precision": p, "recall": r, "f1": f,
                          "accuracy": float((b_true == b_pred).mean())},
        "per_class": {c: report[c] for c in classes},
    }
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=120)

    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    print("Binary defect metrics:", metrics["binary_defect"])


if __name__ == "__main__":
    main()
