import numpy as np
import pytest

from anomaly_detection.data import make_split, make_synthetic
from anomaly_detection.models import IsolationForestDetector, PCADetector


@pytest.fixture(scope="module")
def split():
    X, y = make_synthetic()
    return make_split(X, y, seed=0)


@pytest.mark.parametrize("cls", [PCADetector, IsolationForestDetector])
def test_anomalies_score_higher_than_normals(split, cls):
    det = cls().fit(split.X_train)
    scores = det.score(split.X_test)
    assert scores[split.y_test == 1].mean() > scores[split.y_test == 0].mean()


@pytest.mark.parametrize("cls", [PCADetector, IsolationForestDetector])
def test_score_before_fit_raises(cls):
    with pytest.raises(RuntimeError):
        cls().score(np.zeros((2, 3)))


@pytest.mark.parametrize("cls", [PCADetector, IsolationForestDetector])
def test_same_seed_same_scores(split, cls):
    a = cls(seed=7).fit(split.X_train).score(split.X_test)
    b = cls(seed=7).fit(split.X_train).score(split.X_test)
    np.testing.assert_allclose(a, b)


def test_autoencoder_beats_chance_on_synthetic(split):
    tf = pytest.importorskip("tensorflow")  # noqa: F841
    from anomaly_detection.metrics import evaluate_scores
    from anomaly_detection.models import AutoencoderDetector

    det = AutoencoderDetector(hidden=(16,), epochs=15, seed=0).fit(split.X_train)
    scores = det.score(split.X_test)
    assert scores.shape == (len(split.X_test),)
    assert evaluate_scores(split.y_test, scores, threshold=np.median(scores))["roc_auc"] > 0.7
