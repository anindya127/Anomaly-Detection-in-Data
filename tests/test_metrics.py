import numpy as np
import pytest

from anomaly_detection.metrics import best_f1_threshold, evaluate_scores


def test_perfect_separation():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    thr = best_f1_threshold(y, scores)
    m = evaluate_scores(y, scores, thr)
    assert m["f1"] == 1.0 and m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 0, 0, 3)


def test_threshold_is_inclusive():
    y = np.array([0, 1])
    m = evaluate_scores(y, np.array([0.2, 0.5]), threshold=0.5)
    assert m["tp"] == 1 and m["fp"] == 0


def test_confusion_counts_known_case():
    y = np.array([0, 0, 1, 1, 1])
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
    m = evaluate_scores(y, scores, threshold=0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)


def test_no_predicted_positives_gives_zero_not_error():
    y = np.array([0, 1, 1])
    m = evaluate_scores(y, np.array([0.1, 0.2, 0.3]), threshold=10.0)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_random_scores_roc_auc_near_half():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    m = evaluate_scores(y, rng.random(5000), threshold=0.5)
    assert m["roc_auc"] == pytest.approx(0.5, abs=0.03)


def test_pr_auc_baseline_equals_prevalence_for_random_scores():
    rng = np.random.default_rng(1)
    y = (rng.random(20000) < 0.02).astype(int)
    m = evaluate_scores(y, rng.random(20000), threshold=0.5)
    assert m["pr_auc"] == pytest.approx(y.mean(), abs=0.01)
