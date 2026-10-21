import numpy as np
import pytest

from anomaly_detection.data import make_split, make_synthetic


def test_synthetic_shapes_and_labels():
    X, y = make_synthetic(n_normal=200, n_anomaly=20, n_features=4, seed=1)
    assert X.shape == (220, 4)
    assert y.sum() == 20


def test_synthetic_is_reproducible():
    a, _ = make_synthetic(seed=3)
    b, _ = make_synthetic(seed=3)
    np.testing.assert_array_equal(a, b)


def test_train_contains_only_normals_and_counts_dropped():
    X, y = make_synthetic()
    s = make_split(X, y, seed=0)
    # train rows come only from the normal class, so the scaled train mean is ~0
    assert abs(s.X_train.mean()) < 1e-9
    assert s.n_anomalies_dropped_from_train > 0
    assert s.y_val.sum() > 0 and s.y_test.sum() > 0


def test_split_is_stratified():
    X, y = make_synthetic()
    s = make_split(X, y, seed=0)
    overall = y.mean()
    assert s.y_test.mean() == pytest.approx(overall, abs=0.01)
    assert s.y_val.mean() == pytest.approx(overall, abs=0.01)


def test_no_row_overlap_between_sets():
    X, y = make_synthetic()
    s = make_split(X, y, seed=0)
    total = len(s.X_train) + s.n_anomalies_dropped_from_train + len(s.X_val) + len(s.X_test)
    assert total == len(y)


def test_scaler_fitted_on_train_only():
    X, y = make_synthetic()
    s = make_split(X, y, seed=0)
    # validation/test are NOT exactly zero-mean: they were transformed, not fitted
    assert abs(s.X_test.mean()) > 1e-6


def test_bad_fractions_rejected():
    X, y = make_synthetic()
    with pytest.raises(ValueError):
        make_split(X, y, val_size=0.6, test_size=0.5)
