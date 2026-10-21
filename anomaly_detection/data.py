"""Data loading and leak-free splitting.

Protocol (semi-supervised anomaly detection):
  * The detector is trained on NORMAL samples only.
  * The decision threshold is chosen on a validation set that has both classes.
  * Final metrics are reported on a held-out test set the model never touched.
  * The scaler is fitted on the training samples only.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CREDITCARD_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
DEFAULT_CACHE = Path.home() / ".cache" / "anomaly-detection"


@dataclass
class Split:
    """Scaled train/validation/test arrays. ``y`` is 1 for anomaly, 0 for normal."""

    X_train: np.ndarray  # normal samples only
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    n_anomalies_dropped_from_train: int


def make_synthetic(
    n_normal: int = 1000,
    n_anomaly: int = 100,
    n_features: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian normals N(0, 1) and shifted, wider anomalies N(2, 2)."""
    rng = np.random.default_rng(seed)
    normal = rng.normal(loc=0.0, scale=1.0, size=(n_normal, n_features))
    anomalies = rng.normal(loc=2.0, scale=2.0, size=(n_anomaly, n_features))
    X = np.vstack((normal, anomalies))
    y = np.hstack((np.zeros(n_normal), np.ones(n_anomaly))).astype(int)
    return X, y


def download_creditcard(cache_dir: Path = DEFAULT_CACHE) -> Path:
    """Download the ULB credit-card-fraud CSV (150 MB) once and cache it."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "creditcard.csv"
    if not path.exists():
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(CREDITCARD_URL, tmp)  # noqa: S310 (fixed https URL)
        tmp.replace(path)
    return path


def load_creditcard(path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return features and labels for the credit-card-fraud dataset.

    Features are V1..V28 (PCA components supplied by the dataset authors) plus
    log1p(Amount). The raw ``Time`` column is dropped: it is seconds since the
    first transaction and carries no transferable signal.
    """
    csv = Path(path) if path else download_creditcard()
    df = pd.read_csv(csv)
    features = df[[f"V{i}" for i in range(1, 29)]].copy()
    features["log_amount"] = np.log1p(df["Amount"])
    return features.to_numpy(dtype=np.float64), df["Class"].to_numpy(dtype=int)


def make_split(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
    val_size: float = 0.2,
    test_size: float = 0.2,
) -> Split:
    """Stratified split. Training keeps normal samples only; the scaler sees only those."""
    if not 0 < val_size + test_size < 1:
        raise ValueError("val_size + test_size must be in (0, 1)")

    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    val_fraction = val_size / (1.0 - test_size)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_rest, y_rest, test_size=val_fraction, stratify=y_rest, random_state=seed
    )

    normal_mask = y_tr == 0
    X_train = X_tr[normal_mask]
    dropped = int((~normal_mask).sum())

    scaler = StandardScaler().fit(X_train)
    return Split(
        X_train=scaler.transform(X_train),
        X_val=scaler.transform(X_val),
        y_val=y_val,
        X_test=scaler.transform(X_test),
        y_test=y_test,
        scaler=scaler,
        n_anomalies_dropped_from_train=dropped,
    )
