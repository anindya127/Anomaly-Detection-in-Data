"""Detectors. Every detector exposes ``fit(X_normal)`` and ``score(X)``.

``score`` returns one number per row; HIGHER means MORE anomalous.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


class PCADetector:
    """Reconstruction error after projecting onto the top principal components."""

    name = "PCA"

    def __init__(self, n_components: int | float = 0.9, seed: int = 0) -> None:
        self.n_components = n_components
        self.seed = seed
        self._pca: PCA | None = None

    def fit(self, X: np.ndarray) -> "PCADetector":
        self._pca = PCA(n_components=self.n_components, random_state=self.seed).fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("call fit() first")
        recon = self._pca.inverse_transform(self._pca.transform(X))
        return np.mean((X - recon) ** 2, axis=1)


class IsolationForestDetector:
    """Isolation Forest (Liu et al., 2008)."""

    name = "IsolationForest"

    def __init__(self, n_estimators: int = 200, seed: int = 0) -> None:
        self.n_estimators = n_estimators
        self.seed = seed
        self._model: IsolationForest | None = None

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        self._model = IsolationForest(
            n_estimators=self.n_estimators, random_state=self.seed, n_jobs=-1
        ).fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("call fit() first")
        return -self._model.score_samples(X)


class AutoencoderDetector:
    """Symmetric dense autoencoder; the score is the per-row reconstruction MSE.

    Differences from the original single-layer script: trained on normal data only,
    a linear output layer (inputs are standardised, so they go negative), early
    stopping on a held-out slice of the normal data, and seeded training.
    """

    name = "Autoencoder"

    def __init__(
        self,
        hidden: tuple[int, ...] = (64, 32),
        latent_dim: int | None = None,
        epochs: int = 50,
        batch_size: int = 256,
        patience: int = 5,
        seed: int = 0,
    ) -> None:
        self.hidden = hidden
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self.history: dict[str, list[float]] = {}
        self._model = None

    def _build(self, input_dim: int):
        from tensorflow import keras  # imported lazily so the package works without TF

        latent = self.latent_dim or max(2, input_dim // 2)
        inputs = keras.layers.Input(shape=(input_dim,))
        x = inputs
        for units in self.hidden:
            x = keras.layers.Dense(units, activation="relu")(x)
        x = keras.layers.Dense(latent, activation="relu")(x)
        for units in reversed(self.hidden):
            x = keras.layers.Dense(units, activation="relu")(x)
        outputs = keras.layers.Dense(input_dim, activation="linear")(x)
        model = keras.models.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mse")
        return model

    def fit(self, X: np.ndarray) -> "AutoencoderDetector":
        from tensorflow import keras

        keras.utils.set_random_seed(self.seed)
        self._model = self._build(X.shape[1])
        stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.patience, restore_best_weights=True
        )
        hist = self._model.fit(
            X,
            X,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            shuffle=True,
            callbacks=[stop],
            verbose=0,
        )
        self.history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("call fit() first")
        recon = self._model.predict(X, batch_size=4096, verbose=0)
        return np.mean((X - recon) ** 2, axis=1)
