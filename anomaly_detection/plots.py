"""Figures for the README. Re-fits each model on one seed to draw curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from .benchmark import default_detectors  # noqa: E402
from .data import make_split  # noqa: E402


def curves(X: np.ndarray, y: np.ndarray, out_dir: Path, tag: str, seed: int = 0, epochs: int = 50) -> list[Path]:
    """Save ROC and precision-recall curves (test set) for every detector."""
    out_dir.mkdir(parents=True, exist_ok=True)
    split = make_split(X, y, seed=seed)
    fitted = {}
    for name, factory in default_detectors(epochs=epochs).items():
        det = factory(seed).fit(split.X_train)
        fitted[name] = det.score(split.X_test)

    paths = []

    fig, ax = plt.subplots(figsize=(5, 4))
    for name, s in fitted.items():
        fpr, tpr, _ = roc_curve(split.y_test, s)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC - {tag}")
    ax.legend()
    fig.tight_layout()
    p = out_dir / f"{tag}_roc.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(5, 4))
    for name, s in fitted.items():
        prec, rec, _ = precision_recall_curve(split.y_test, s)
        ax.plot(rec, prec, label=name)
    ax.axhline(split.y_test.mean(), color="k", ls="--", lw=0.8, label="chance")
    ax.set(xlabel="Recall", ylabel="Precision", title=f"Precision-recall - {tag}")
    ax.legend()
    fig.tight_layout()
    p = out_dir / f"{tag}_pr.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)
    return paths
