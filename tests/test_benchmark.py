from anomaly_detection.benchmark import default_detectors, format_table, run_benchmark
from anomaly_detection.data import make_synthetic


def test_benchmark_runs_without_tensorflow_and_reports_all_metrics():
    X, y = make_synthetic()
    report = run_benchmark(
        X, y, seeds=(0, 1), detectors=default_detectors(include_autoencoder=False)
    )
    assert set(report["summary"]) == {"PCA", "IsolationForest"}
    for stats in report["summary"].values():
        for metric in ("roc_auc", "pr_auc", "precision", "recall", "f1"):
            assert 0.0 <= stats[metric]["mean"] <= 1.0
    assert len(report["per_seed"]["PCA"]) == 2


def test_table_is_markdown_with_a_row_per_model():
    X, y = make_synthetic()
    report = run_benchmark(
        X, y, seeds=(0,), detectors=default_detectors(include_autoencoder=False)
    )
    lines = format_table(report).splitlines()
    assert lines[0].startswith("| Model")
    assert len(lines) == 2 + 2
