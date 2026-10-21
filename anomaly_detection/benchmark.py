"""Run every detector over several seeds and aggregate the test-set metrics."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .data import Split, make_split
from .metrics import best_f1_threshold, evaluate_scores
from .models import AutoencoderDetector, IsolationForestDetector, PCADetector

DetectorFactory = Callable[[int], object]

METRICS = ("roc_auc", "pr_auc", "precision", "recall", "f1")


def default_detectors(epochs: int = 50, include_autoencoder: bool = True) -> dict[str, DetectorFactory]:
    detectors: dict[str, DetectorFactory] = {
        "PCA": lambda seed: PCADetector(seed=seed),
        "IsolationForest": lambda seed: IsolationForestDetector(seed=seed),
    }
    if include_autoencoder:
        detectors["Autoencoder"] = lambda seed: AutoencoderDetector(epochs=epochs, seed=seed)
    return detectors


def run_one(detector, split: Split) -> dict[str, float]:
    """Fit on normals, pick the threshold on validation, report on test."""
    start = time.perf_counter()
    detector.fit(split.X_train)
    fit_seconds = time.perf_counter() - start
    threshold = best_f1_threshold(split.y_val, detector.score(split.X_val))
    result = evaluate_scores(split.y_test, detector.score(split.X_test), threshold)
    result["fit_seconds"] = fit_seconds
    return result


def run_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    detectors: dict[str, DetectorFactory] | None = None,
) -> dict:
    """Return per-seed results and mean/std summaries for each detector."""
    detectors = detectors or default_detectors()
    per_seed: dict[str, list[dict[str, float]]] = {name: [] for name in detectors}

    for seed in seeds:
        split = make_split(X, y, seed=seed)
        for name, factory in detectors.items():
            per_seed[name].append(run_one(factory(seed), split))

    summary = {}
    for name, runs in per_seed.items():
        summary[name] = {
            m: {"mean": float(np.mean([r[m] for r in runs])), "std": float(np.std([r[m] for r in runs]))}
            for m in (*METRICS, "fit_seconds")
        }
    return {
        "n_samples": int(len(y)),
        "n_anomalies": int(y.sum()),
        "anomaly_rate": float(y.mean()),
        "seeds": list(seeds),
        "summary": summary,
        "per_seed": per_seed,
    }


def format_table(report: dict) -> str:
    """Markdown results table, mean ± std over seeds."""
    header = "| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |\n|---|---|---|---|---|---|\n"
    rows = []
    for name, stats in report["summary"].items():
        cells = [f"{stats[m]['mean']:.3f} ± {stats[m]['std']:.3f}" for m in METRICS]
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return header + "\n".join(rows)


def save_report(report: dict, out_dir: Path, tag: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}.json"
    path.write_text(json.dumps(report, indent=2))
    (out_dir / f"{tag}.md").write_text(format_table(report) + "\n")
    return path
