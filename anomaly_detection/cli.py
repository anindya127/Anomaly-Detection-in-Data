"""Command line: ``python -m anomaly_detection {synthetic,creditcard} [options]``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import default_detectors, format_table, run_benchmark, save_report
from .data import load_creditcard, make_synthetic


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="anomaly_detection", description=__doc__)
    p.add_argument("dataset", choices=["synthetic", "creditcard"])
    p.add_argument("--seeds", type=int, default=5, help="number of random splits (default 5)")
    p.add_argument("--epochs", type=int, default=50, help="autoencoder max epochs (default 50)")
    p.add_argument("--no-autoencoder", action="store_true", help="skip TensorFlow models")
    p.add_argument("--csv", type=Path, help="path to creditcard.csv (otherwise downloaded and cached)")
    p.add_argument("--out", type=Path, default=Path("results"), help="output directory")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dataset == "synthetic":
        X, y = make_synthetic()
    else:
        X, y = load_creditcard(args.csv)

    detectors = default_detectors(epochs=args.epochs, include_autoencoder=not args.no_autoencoder)
    report = run_benchmark(X, y, seeds=tuple(range(args.seeds)), detectors=detectors)
    save_report(report, args.out, args.dataset)

    print(
        f"{args.dataset}: {report['n_samples']} samples, {report['n_anomalies']} anomalies "
        f"({report['anomaly_rate']:.2%}), {args.seeds} seeds\n"
    )
    print(format_table(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
